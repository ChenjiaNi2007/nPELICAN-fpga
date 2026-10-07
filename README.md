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

   **Model dimensions** (`NPARTICLES`, `NHIDDEN`, `NOUT`) are also emitted at the top of
   `types_generated.h` from the checkpoint; `nPELICAN.h` keeps `#ifndef`-guarded hand defaults
   (20/2/1, `NPARTICLES2 = NPARTICLES + 2`) and `weights.h` `static_assert`s that they match.
   A K-class checkpoint (`--n-out K` in PELICAN-nano) exports `NOUT=K` and the firmware emits K
   raw logits via `model_out[NOUT]` (softmax/argmax off-chip); `golden_logits.dat` then carries
   NOUT space-separated values per line. `w2_2to0` element order is row-major `(NOUT, NHIDDEN*2)`
   = `np.ravel(agg_2to0.mixing.weight)`, i.e. `w2_2to0[o*(NHIDDEN*2) + h*2 + a]`.

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

## Numerical contracts added 2026-10-06 (5-class / NOUT generalization)

- `nobj` port = particles present in the NPARTICLES slots; clamp raw multiplicities to NPARTICLES
  (testbench `clamp_nobj`, `export_golden.py`). Port width `NOBJ_BITS` follows NPARTICLES2.
- `model_loader.py --bias-guard-bits 8` / `--agg-guard-bits 8` (defaults): bias literals and the
  stored BN1 output keep 8 fractional bits beyond the MAC / post-agg grids so exact half-LSB ties
  cannot form (float PyTorch never ties). New generated type `mac2b_t` (2->2 bias end-add; falls
  back to `mac2_t` for pre-change headers, byte-identical). Set both to 0 for the old rule.
- Gate record for the first NOUT=5 export: `reports/hls4ml5_smoke_gate/GATE.md` (200/200 exact).
- `--jet-quant-split` checkpoints: loader emits `NPELICAN_JET_QUANT_SPLIT` + `dotj_t`/`dotm_t` (jet row/col,
  m_jet^2 dot grids), `jet_t` (jet_input port, pmu_quant_jet) and `dotall_t` (dots container); all alias
  dot_t/input_t otherwise (byte-identical). Gate: `reports/hls4ml5_smoke_gate/GATE_JET_QUANT_SPLIT.md`.

## Synthesizing the 5-class jet+head model (2026-10-07)

`firmware/weights/` holds the export of `PELICAN-nano/model/qat6wj16_h4_jh_e40_s2_best.pt`
(N=16, h=4, w6a6i6 pmu12, jet spurion + 16-wide head, split jet quantizers at 10/16/20 bits;
test accuracy 69.9%, see PELICAN-nano `docs/HLS4ML_5CLASS.md`), with the default DSP-tuned BN1
literal. Matching golden vectors (incl. `tb_data/golden_jet.dat`) are in `tb_data/`.
On the remote: `git pull`, then the usual `vitis_hls -f build_prj.tcl ...`. For a meaningful
csim add `-DRUN_GOLDEN_GATE` to the testbench cflags in `build_prj.tcl` (the `add_files -tb
nPELICAN_tb.cpp -cflags` line); expected `GOLDEN SUMMARY: events=200 exact=199` (the one residual
is the documented BN1-literal tie; `--bn-guard-bits 8` on export gives 200/200 at DSP cost).
The legacy 10k csim flow is not meaningful for this model (toptag inputs, 20 particles, no jet).
To re-export: `python model_loader.py --model ../PELICAN-nano/model/qat6wj16_h4_jh_e40_s2_best.pt
--quant --repo ../PELICAN-nano --out firmware/weights/weights.h`.
