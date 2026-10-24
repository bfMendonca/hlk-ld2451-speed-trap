# Security policy

If you find a security issue, please report it privately through
[GitHub security advisories](https://github.com/bfMendonca/hlk-ld2451-speed-trap/security/advisories/new)
rather than opening a public issue.

Things to keep in mind when running `speed-trap`:

- **Model weights are code.** `detector.weights` can point to a PyTorch checkpoint, and loading
  a checkpoint can execute code. Only use weights from sources you trust. Named models
  (`rfdetr-small`, `yolo11s.pt`, ...) are downloaded by the RF-DETR and Ultralytics libraries
  from their official locations on first use.
- **Config files** are read with `yaml.safe_load` and validated strictly, so they cannot run
  code, but they choose where outputs are written.
- **Recordings and captures are personal data.** Video, speeder captures and future
  licence-plate crops can identify people. Check the rules that apply where you record, and
  keep recordings out of public repositories and issues.
