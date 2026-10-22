from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from speed_trap.config import FusionConfig, RoadConfig, SpeedConfig
from speed_trap.fusion import FusedTrack, Fusion, gate_cost
from speed_trap.geometry import CameraModel, RadarModel, nominal_matrix
from speed_trap.types import RadarScan, RadarTarget, TrackObservation

W, H = 1296, 972


def make_camera() -> CameraModel:
    return CameraModel(nominal_matrix(W, H), np.zeros(5), W, H, 3.5, 12.0, 0.0, 0.0)


def make_radar() -> RadarModel:
    return RadarModel((0.0, 0.0), 3.5, 0.0, 1, 20.0, 1.0, 100.0)


def fusion(
    fusion_cfg: FusionConfig | None = None,
    speed: SpeedConfig | None = None,
    road: RoadConfig | None = None,
) -> Fusion:
    return Fusion(
        make_camera(),
        make_radar(),
        fusion_cfg or FusionConfig(),
        speed or SpeedConfig(),
        road or RoadConfig(),
    )


def ts(seconds: float) -> int:
    return round(seconds * 1e9)


def box_at_road(x: float, y: float) -> tuple[float, float, float, float]:
    uv, ok = make_camera().project(np.array([[x, y, 0.0]]))
    assert ok[0]
    u, v = float(uv[0][0]), float(uv[0][1])
    return (u - 20.0, v - 30.0, u + 20.0, v)


_ids = itertools.count(1)


def obs_at_road(x: float, y: float, track_id: int | None = None) -> TrackObservation:
    tid = track_id if track_id is not None else next(_ids)
    return TrackObservation(tid, box_at_road(x, y), 0.9, "car", True)


def target(
    angle: int = 0, dist: int = 20, approaching: bool = True, speed: int = 40, t_ns: int = 0
) -> RadarTarget:
    return RadarTarget(t_ns, angle, dist, approaching, speed, 10, 0)


def scan(*targets: RadarTarget, t_ns: int | None = None) -> RadarScan:
    t = t_ns if t_ns is not None else (targets[0].t_ns if targets else 0)
    return RadarScan(t, tuple(targets))


def track(f: Fusion, track_id: int = 1) -> FusedTrack:
    return f.tracks[track_id]


def test_obs_helper_backprojects_exactly() -> None:
    x0, _, x1, y1 = box_at_road(20, 1.5)
    xy = make_camera().backproject_ground((x0 + x1) / 2, y1)
    assert xy == pytest.approx((20, 1.5), abs=1e-6)


def test_single_track_matches() -> None:
    f = fusion()
    t = obs_at_road(20, 0)
    tracks = f.update([t], [scan(target(angle=0, dist=20, approaching=True, speed=50))], ts(0))
    m = tracks[0].last_match
    assert m is not None
    assert m.speed_raw_kmh == 50
    assert m.speed_kmh == 50
    assert m.road_xy == pytest.approx(make_radar().target_to_road(0, 20))
    assert tracks[0].road_xy == pytest.approx((20, 0), abs=1e-6)
    assert tracks[0].subject_id == t.track_id


def test_two_nearby_tracks_are_ambiguous() -> None:
    tracks = fusion().update(
        [obs_at_road(20, 0.15), obs_at_road(20, -0.15)], [scan(target(0, 20))], ts(0)
    )
    assert all(t.last_match is None for t in tracks)


def test_two_nearby_targets_are_ambiguous() -> None:
    tracks = fusion().update([obs_at_road(20, 0)], [scan(target(0, 20), target(0, 21))], ts(0))
    # Δr 0.31 vs 0.71 m: costs 0.012 vs 0.064 -> clearly distinct, kept
    assert tracks[0].last_match is not None
    # 20.2 m sits halfway between the 19.69 m and 20.71 m ground ranges: equal costs
    tracks = fusion().update([obs_at_road(20.2, 0)], [scan(target(0, 20), target(0, 21))], ts(0))
    assert tracks[0].last_match is None


def test_other_lane_track_does_not_steal_match() -> None:
    tracks = fusion().update(
        [obs_at_road(20, 0), obs_at_road(20, -3.5)], [scan(target(0, 20))], ts(0)
    )
    assert [t.last_match is not None for t in tracks] == [True, False]


def test_lane_boundaries_reject_cross_lane_pairs() -> None:
    # track just right of the 0.0 boundary, target just left of it: inside the gate
    lanes = FusionConfig(lanes=[-3.5, 0.0, 3.5])
    tracks = fusion(lanes).update([obs_at_road(20, -0.1)], [scan(target(1, 20))], ts(0))
    assert tracks[0].last_match is None
    tracks = fusion().update([obs_at_road(20, -0.1)], [scan(target(1, 20))], ts(0))
    assert tracks[0].last_match is not None


def test_beyond_max_range_never_matches() -> None:
    assert fusion().update([obs_at_road(40, 0)], [scan(target(0, 40))], ts(0))[0].last_match is None


def test_direction_disagreement_rejected() -> None:
    f = fusion()
    f.update([obs_at_road(30, 0, track_id=1)], [], ts(0))
    tr = f.update(
        [obs_at_road(25, 0, track_id=1)],
        [scan(target(0, 25, approaching=False, t_ns=ts(0.5)))],
        ts(0.5),
    )[0]
    assert tr.last_match is None


def test_direction_agreement_accepted() -> None:
    f = fusion()
    f.update([obs_at_road(30, 0, track_id=1)], [], ts(0))
    tr = f.update(
        [obs_at_road(25, 0, track_id=1)],
        [scan(target(0, 25, approaching=True, t_ns=ts(0.5)))],
        ts(0.5),
    )[0]
    assert tr.last_match is not None


def test_going_track_rejects_approaching_target() -> None:
    f = fusion()
    f.update([obs_at_road(20, 0, track_id=1)], [], ts(0))
    tr = f.update(
        [obs_at_road(25, 0, track_id=1)],
        [scan(target(0, 25, approaching=True, t_ns=ts(0.4)))],
        ts(0.4),
    )[0]
    assert tr.last_match is None


def test_small_motion_leaves_direction_unknown() -> None:
    f = fusion()
    f.update([obs_at_road(20.0, 0, track_id=1)], [], ts(0))
    tr = f.update(
        [obs_at_road(20.3, 0, track_id=1)],
        [scan(target(0, 20, approaching=False, t_ns=ts(0.4)))],
        ts(0.4),
    )[0]
    assert tr.last_match is not None


def test_reliability_needs_two_matches_and_expires() -> None:
    f = fusion()
    o = obs_at_road(20, 0, track_id=1)
    f.update([o], [scan(target(0, 20, t_ns=ts(0)))], ts(0))
    assert not track(f).reliable
    assert track(f).direction is None
    f.update([o], [scan(target(0, 20, t_ns=ts(0.1)))], ts(0.1))
    assert track(f).reliable
    assert track(f).reliable_ever
    assert track(f).direction == "coming"
    f.update([o], [], ts(1.7))
    assert not track(f).reliable and track(f).last_match is not None
    assert track(f).reliable_ever
    assert track(f).direction == "coming"  # unchanged; renderer draws gray


def test_reliable_direction_going() -> None:
    f = fusion()
    o = obs_at_road(20, 0, track_id=1)
    for i in range(2):
        f.update([o], [scan(target(0, 20, approaching=False, t_ns=ts(i * 0.1)))], ts(i * 0.1))
    assert track(f).reliable
    assert track(f).direction == "going"


def test_box_above_horizon_has_no_road_point_and_no_match() -> None:
    f = fusion()
    obs = TrackObservation(1, (600.0, 10.0, 700.0, 50.0), 0.9, "car", True)
    tr = f.update([obs], [scan(target(0, 20))], ts(0))[0]
    assert tr.road_xy is None
    assert tr.last_match is None
    assert len(tr.road_history) == 0


def test_cosine_correction_off_by_default_and_capped() -> None:
    o = obs_at_road(20, 0, track_id=1)
    road = RoadConfig(lane_heading_deg=70.0)
    tr = fusion(road=road).update([o], [scan(target(0, 20, speed=50))], ts(0))[0]
    assert tr.last_match is not None
    assert tr.last_match.speed_kmh == 50
    on = SpeedConfig(cosine_correction=True)
    tr = fusion(speed=on, road=road).update([o], [scan(target(0, 20, speed=50))], ts(0))[0]
    assert tr.last_match is not None
    assert tr.last_match.speed_raw_kmh == 50
    assert tr.last_match.speed_kmh == pytest.approx(50 / 0.5)


def test_cosine_correction_uncapped_small_angle() -> None:
    o = obs_at_road(20, 0, track_id=1)
    on = SpeedConfig(cosine_correction=True)
    tr = fusion(speed=on, road=RoadConfig(lane_heading_deg=30.0)).update(
        [o], [scan(target(0, 20, speed=50))], ts(0)
    )[0]
    assert tr.last_match is not None
    assert tr.last_match.speed_kmh == pytest.approx(50 / math.cos(math.radians(30)))
    # going lane: heading 180 deg is the same line of travel, correction is 1
    tr = fusion(speed=on, road=RoadConfig(lane_heading_deg=180.0)).update(
        [o], [scan(target(0, 20, speed=50))], ts(0)
    )[0]
    assert tr.last_match is not None
    assert tr.last_match.speed_kmh == pytest.approx(50)


def test_gate_widens_with_distance() -> None:
    radar, cfg = make_radar(), FusionConfig()
    assert gate_cost(radar, cfg, (28, 0), target(0, 30)) is not None  # Δr ≈ 1.8 m, gate ≈ 3.6 m
    assert gate_cost(radar, cfg, (8, 0), target(0, 11)) is None  # Δr ≈ 2.4 m, gate ≈ 2.1 m


def test_gate_cost_values() -> None:
    radar, cfg = make_radar(), FusionConfig()
    g = math.sqrt(30**2 - 3.5**2)
    gate_r = 1.5 + 0.002 * 28**2 + 0.5
    assert gate_cost(radar, cfg, (28, 0), target(0, 30)) == pytest.approx(((g - 28) / gate_r) ** 2)
    # angle gate is 1.5 deg
    y = 20 * math.tan(math.radians(1.6))
    assert gate_cost(radar, cfg, (20, y), target(0, 20)) is None
    y = 20 * math.tan(math.radians(1.4))
    assert gate_cost(radar, cfg, (20, y), target(0, 20)) is not None
    # target beyond max range even though track is inside
    assert gate_cost(radar, cfg, (29.5, 0), target(0, 31)) is None


def test_track_history_and_speed_stats() -> None:
    f = fusion()
    o = obs_at_road(20, 0, track_id=1)
    f.update([o], [scan(target(0, 20, speed=40, t_ns=ts(0)))], ts(0))
    f.update([o], [scan(target(0, 20, speed=50, t_ns=ts(0.1)))], ts(0.1))
    tr = track(f)
    assert tr.max_speed_kmh == 50
    assert tr.mean_speed_kmh == pytest.approx(45)
    assert [t for t, _ in tr.boxes] == [ts(0), ts(0.1)]
    assert tr.first_ts_ns == ts(0) and tr.last_ts_ns == ts(0.1)
    assert len(tr.road_history) == 2


def test_road_history_bounded_in_time() -> None:
    f = fusion()
    for i in range(50):
        f.update([obs_at_road(20, 0, track_id=1)], [], ts(i * 0.1))
    hist = track(f).road_history
    assert hist[-1][0] - hist[0][0] <= ts(4 * 0.5)
    assert len(hist) <= 21


def test_unmatched_track_has_no_speed() -> None:
    tr = fusion().update([obs_at_road(20, 0)], [], ts(0))[0]
    assert tr.last_match is None
    assert tr.max_speed_kmh is None and tr.mean_speed_kmh is None
    assert not tr.reliable and tr.direction is None


def test_finish_returns_and_removes() -> None:
    f = fusion()
    f.update([obs_at_road(20, 0, track_id=7), obs_at_road(15, 3, track_id=8)], [], ts(0))
    done = f.finish([7, 99])
    assert [t.track_id for t in done] == [7]
    assert 7 not in f.tracks and 8 in f.tracks


def test_long_observed_stationary_track_rejects_moving_target() -> None:
    f = fusion()
    for i in range(6):  # 0.5 s without moving
        f.update([obs_at_road(20, 0, track_id=1)], [], ts(i * 0.1))
    tr = f.update(
        [obs_at_road(20, 0, track_id=1)],
        [scan(target(0, 20, approaching=True, speed=50, t_ns=ts(0.6)))],
        ts(0.6),
    )[0]
    assert tr.last_match is None


def test_speed_consistent_moving_track_matches() -> None:
    f = fusion()
    for i in range(6):  # 50 km/h toward the radar
        f.update([obs_at_road(30 - 50 / 3.6 * i * 0.1, 0, track_id=1)], [], ts(i * 0.1))
    x = 30 - 50 / 3.6 * 0.6
    tr = f.update(
        [obs_at_road(x, 0, track_id=1)],
        [scan(target(0, round(make_radar().slant_range(x)), speed=48, t_ns=ts(0.6)))],
        ts(0.6),
    )[0]
    assert tr.last_match is not None


def test_speed_inconsistent_moving_track_rejected() -> None:
    f = fusion()
    for i in range(6):  # 20 km/h track vs a 50 km/h target
        f.update([obs_at_road(30 - 20 / 3.6 * i * 0.1, 0, track_id=1)], [], ts(i * 0.1))
    x = 30 - 20 / 3.6 * 0.6
    tr = f.update(
        [obs_at_road(x, 0, track_id=1)],
        [scan(target(0, round(make_radar().slant_range(x)), speed=50, t_ns=ts(0.6)))],
        ts(0.6),
    )[0]
    assert tr.last_match is None


def test_short_history_track_is_not_speed_gated() -> None:
    f = fusion()
    f.update([obs_at_road(20, 0, track_id=1)], [], ts(0))
    tr = f.update(
        [obs_at_road(20, 0, track_id=1)],
        [scan(target(0, 20, speed=50, t_ns=ts(0.3)))],
        ts(0.3),
    )[0]
    assert tr.last_match is not None


# --- radar scan time vs track position (frame gaps) --------------------------------------


def test_stale_scan_after_frame_gap_never_reaches_the_follower() -> None:
    """Lead 60 km/h, follower 45 km/h; scans from a 1 s frame gap arrive with the next frame.

    At the frame after the gap the follower stands where the lead was when the radar saw it.
    """
    f = fusion()
    radar = make_radar()
    y = -2.0
    v_lead, v_follow = 60 / 3.6, 45 / 3.6

    def x_lead(t: float) -> float:
        return 50.0 - v_lead * t

    t_after = 2.6

    def x_follow(t: float) -> float:
        return x_lead(1.7) + v_follow * (t_after - t)

    def tgt(x: float, t: float, speed: int) -> RadarTarget:
        r, az = radar.road_to_polar(x, y)
        return target(round(az), round(radar.slant_range(r)), True, speed, ts(t))

    for i in range(40):  # 1.6 s of history, no radar
        t = i * 0.04
        f.update([obs_at_road(x_lead(t), y, 1), obs_at_road(x_follow(t), y, 2)], [], ts(t))
    scans = [scan(tgt(x_lead(s), s, 60)) for s in (1.6, 1.7)]
    tracks = f.update(
        [obs_at_road(x_lead(t_after), y, 1), obs_at_road(x_follow(t_after), y, 2)],
        scans,
        ts(t_after),
    )
    follower = next(t for t in tracks if t.track_id == 2)
    assert all(m.speed_raw_kmh != 60 for m in follower.matches)
    assert not follower.reliable


def test_several_scans_per_frame_match_the_position_at_scan_time() -> None:
    """With a wider max_scan_age_s, two older scans match the interpolated position."""
    f = fusion(FusionConfig(max_scan_age_s=0.5))
    radar = make_radar()
    v = 60 / 3.6

    def x_at(t: float) -> float:
        return 28.0 - v * t

    def tgt(t: float) -> RadarTarget:
        r, _ = radar.road_to_polar(x_at(t), 0.0)
        return target(0, round(radar.slant_range(r)), True, 60, ts(t))

    for i in range(16):  # 0.6 s of history every 40 ms
        f.update([obs_at_road(x_at(i * 0.04), 0, 1)], [], ts(i * 0.04))
    t_now = 1.0  # 0.4 s frame gap: current position is ~5 m past the scans' positions
    f.update([obs_at_road(x_at(t_now), 0, 1)], [], ts(t_now))
    t_now = 1.04
    tr = f.update([obs_at_road(x_at(t_now), 0, 1)], [scan(tgt(0.7)), scan(tgt(0.8))], ts(t_now))[0]
    assert [m.t_ns for m in tr.matches] == [ts(0.7), ts(0.8)]


def test_scan_older_than_max_scan_age_is_skipped() -> None:
    f = fusion()
    o = obs_at_road(20, 0, track_id=1)
    tr = f.update([o], [scan(target(0, 20, t_ns=ts(0.0)))], ts(0.2))[0]
    assert tr.last_match is None
    tr = f.update([o], [scan(target(0, 20, t_ns=ts(0.3)))], ts(0.4))[0]
    assert tr.last_match is not None


def test_scan_without_nearby_track_sample_is_invalid() -> None:
    """Skipped-frame observations add no road sample; a scan far from any sample is dropped."""
    f = fusion()
    f.update([obs_at_road(20, 0, track_id=1)], [], ts(0.0))
    f.update([obs_at_road(20, 0, track_id=1)], [], ts(0.1))
    predicted = TrackObservation(1, box_at_road(20, 0), 0.9, "car", False)
    f.update([predicted], [], ts(0.2))
    tr = f.update([predicted], [scan(target(0, 20, t_ns=ts(0.28)))], ts(0.3))[0]
    assert len(tr.road_history) == 2
    assert tr.last_match is None


def test_young_track_match_is_unverified() -> None:
    f = fusion()
    tr = f.update([obs_at_road(20, 0, track_id=1)], [scan(target(0, 20))], ts(0))[0]
    assert tr.last_match is not None and not tr.last_match.verified


def test_match_after_direction_and_speed_checks_is_verified() -> None:
    f = fusion()
    for i in range(6):  # 50 km/h toward the radar
        f.update([obs_at_road(30 - 50 / 3.6 * i * 0.1, 0, track_id=1)], [], ts(i * 0.1))
    x = 30 - 50 / 3.6 * 0.6
    tr = f.update(
        [obs_at_road(x, 0, track_id=1)],
        [scan(target(0, round(make_radar().slant_range(x)), speed=48, t_ns=ts(0.6)))],
        ts(0.6),
    )[0]
    assert tr.last_match is not None and tr.last_match.verified


def test_stationary_track_match_with_unknown_direction_is_unverified() -> None:
    f = fusion()
    for i in range(6):
        f.update([obs_at_road(20, 0, track_id=1)], [], ts(i * 0.1))
    tr = f.update(
        [obs_at_road(20, 0, track_id=1)],
        [scan(target(0, 20, speed=5, t_ns=ts(0.6)))],
        ts(0.6),
    )[0]
    assert tr.last_match is not None and not tr.last_match.verified


def test_predicted_observation_is_drawn_but_not_measured() -> None:
    f = fusion()
    f.update([obs_at_road(20, 0, track_id=1)], [], ts(0.0))
    predicted = TrackObservation(1, box_at_road(19, 0), 0.9, "car", False)
    tr = f.update([predicted], [], ts(0.04))[0]
    assert tr.xyxy == predicted.xyxy and tr.last_ts_ns == ts(0.04)
    assert [t for t, _ in tr.boxes] == [ts(0.0)]
    assert [t for t, _ in tr.road_points] == [ts(0.0)]
    assert [h[0] for h in tr.road_history] == [ts(0.0)]
    assert tr.road_xy == pytest.approx((20, 0), abs=1e-6)
