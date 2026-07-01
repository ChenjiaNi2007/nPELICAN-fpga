# Equivariance-sweep checkpoints

Small (~60 KB) PELICAN-nano QAT checkpoints, vendored here so the equivariance sweep is
self-contained (the machine running the sweep may not have every checkpoint in
`../PELICAN-nano/model/`). `config.yaml` references these by `equivariance/checkpoints/...`.

These are exact copies of `PELICAN-nano/model/fpga_model_qat_*.pt`:

| file | sweep label (W:A:I) |
|---|---|
| `fpga_model_qat_w6a6i6_best.pt`    | 6:6:6 |
| `fpga_model_qat_w6a6i8_best.pt`    | 6:6:8 |
| `fpga_model_qat_w6a6i12_best.pt`   | 6:6:12 |
| `fpga_model_qat_w8a8i16_best.pt`   | 8:8:16 |
| `fpga_model_qat_w12a12i16_best.pt` | 12:12:16 |
| `fpga_model_qat_w16a16i16_best.pt` | 16:16:16 |
| `fpga_model_qat_best.pt`           | 24:24:24 (reference) + float build |

`model_loader.py --repo ../PELICAN-nano` still supplies the model *architecture* to load
these state-dicts; only the checkpoint files moved.
