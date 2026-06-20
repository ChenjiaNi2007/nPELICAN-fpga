# nPELICAN-fpga

C++ HLS code for implementing nanoPELICAN on FPGAs.

`clean.sh` removes files from previous builds. Vitis configuration values are set at the
top of `build_prj.tcl`.

## Regenerate-on-retrain workflow

The firmware datapath uses per-stage fixed-point types **derived from the trained model's
learned QAT scales**, so retraining (at any bit width) regenerates the firmware with no manual
header edits. The loop:

1. **Train** (in `../PELICAN-nano/`) with QAT:
   ```bash
   python train_pelican_nano.py --datadir ./data/sample_data --target is_signal \
       --quant --po2-scales --weight-bit-width 24 --act-bit-width 24 --input-bit-width 24 \
       --n-hidden 2 --nobj 20 --nobj-avg 49 --prefix fpga_model_qat
   ```
   Set the bit widths up front; QAT learns each quantizer's po2 scale `2^-k`, which *is* the
   split — the firmware type for a B-bit quantizer is `ap_fixed<B, B-k>` (fractional bits = k,
   integer bits = B-k). `scripts/check_scales.py` prints the learned split per quantizer.

2. **Export weights + types** (regenerates both headers):
   ```bash
   python model_loader.py --model ../PELICAN-nano/model/fpga_model_qat_best.pt --quant \
       --out firmware/weights/weights.h --repo ../PELICAN-nano
   ```
   Writes `firmware/weights/weights.h` and `firmware/weights/types_generated.h`
   (`firmware/nPELICAN.h` `#include`s the latter). The quantization-point types
   (`dot_t, t2_t, relu_t, t0_t, out_t, w1_gen_t, w2_gen_t`) come straight from the learned
   scales; the intermediate/accumulator/MAC types (`bn1out_t, tr_t, acc*_t, mac*_t, bias_t_gen,
   bn_t_gen, norm_t`) are formula-derived from those plus the term counts. `--bn-eps` (default
   1e-5) must match the training BatchNorm eps — the BN scale is `weight/sqrt(var+eps)`.

3. **Export golden vectors** (for the csim gate), in `../PELICAN-nano/`:
   ```bash
   python scripts/export_golden.py
   ```
   Writes `tb_data/golden_{pmu,nobj,logits,dots}.dat` and `golden_stage_dump.txt`.

4. **C-sim** (`vitis_hls -f build_prj.tcl`). By default the testbench runs the legacy 10k flow;
   define `RUN_GOLDEN_GATE` to run the bit-exactness gate instead (see below).

## Bit-exactness gate

The testbench compares firmware logits to the PyTorch quant-model logits and prints:

- `GOLDEN GATE` — momenta path (firmware computes `d_ij` from the four-momenta via `dot4`).
- `DOTS-LEVEL GATE` — network isolated: PyTorch's quantized `d_ij` are injected in place of the
  `dot4` front-end (`tb_data/golden_dots.dat` -> `npelican_dots_override`, csim-only).

Each gate is a **tolerance** gate (PASS = `max|delta|` under tolerance), with the zero-tolerance
exact count reported alongside. Zero-tolerance 200/200 is **not achievable** here by design:
BatchNorm is kept in float, so the datapath has unquantized float segments between the learned
quantizers that fixed-point cannot reproduce bit-for-bit at every quantizer boundary (PyTorch's
own float32-vs-float64 logits already differ by ~3.6e-6). See `docs/resource_log.md` for the
full interpretation and the dot4 front-end caveat. Current result: dots-level 142/200 exact,
`max|delta|`=1.1e-5; golden 133/200 exact, `max|delta|`=6.3e-4.

## Layout

- `firmware/nPELICAN.{h,cpp}` — the datapath. `nPELICAN.h` holds the IO/interface typedefs and
  `#include`s the generated `weights/types_generated.h`; `weights/weights.h` is generated.
- `model_loader.py` — reads a PELICAN-nano checkpoint, writes `weights.h` + `types_generated.h`.
- `nPELICAN_tb.cpp` — testbench (golden + dots-level modes, stage-dump harness).
- `docs/FIRMWARE_QAT_PLAN.md` — the restructure spec; `docs/resource_log.md` — phase results.
