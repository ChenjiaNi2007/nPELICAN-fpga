# nanoPELICAN QAT → FPGA — full workflow (training → implementation)

End-to-end loop across the two sibling repos. Run everything from the workspace root
(`Rankin Research/`). See `CLAUDE.md` for the workspace map + firmware deep-dive,
`PELICAN-nano/CLAUDE.md` for the model architecture, and
`nPELICAN-fpga/docs/FIRMWARE_QAT_PLAN.md` for the restructure spec.

```
Rankin Research/
├── PELICAN-nano/      # QAT training (PyTorch/Brevitas)  → model/<prefix>_best.pt
│   └── .venv/         # Python 3.12, torch 2.12, brevitas 0.12
└── nPELICAN-fpga/     # Vitis HLS firmware (C++)          → firmware/weights/*.h
```

The loop: **train → inspect splits → export weights+types → export golden → C-sim/synth.**
Changing bit widths means re-running from step 1 — nothing downstream is hand-edited.

> Two independent git repos, NOT submodules. `cd` into the relevant repo before
> `git add/commit/push`. Nothing to commit at the workspace root.

---

## Step 1 — Train (QAT) — in `PELICAN-nano/`

```bash
cd PELICAN-nano
source .venv/bin/activate          # or prefix commands with .venv/bin/python

python train_pelican_nano.py \
    --datadir ./data/sample_data --target is_signal \
    --quant --po2-scales \
    --weight-bit-width 24 --act-bit-width 24 --input-bit-width 24 \
    --n-hidden 2 --nobj 20 --nobj-avg 49 \
    --num-epoch 8 --batch-size 256 \
    --drop-rate 0.05 --drop-rate-out 0.05 --weight-decay 0.005 \
    --prefix fpga_model_qat --cpu
```

Produces `model/fpga_model_qat_best.pt`. Key points:

- **Set the total bit widths here** (`--weight/act/input-bit-width`). QAT *learns* each
  quantizer's power-of-two scale `2^-k` — and `k` is the fractional-bit count, so the
  integer/fractional **split is discovered by training**, not chosen by hand. To sweep
  widths, change these flags and rerun the whole loop (see "Changing bit widths" below).
- `--po2-scales` is **required** — it constrains scales to powers of two, which is what makes
  an exact `ap_fixed<B, B-k>` match possible. (Without it, pass `--no-po2` to `model_loader.py`.)
- **Gotchas (see PELICAN-nano/CLAUDE.md):**
  - `--num-epoch` minimum is **8** for the default `cos` LR schedule (warmup+cooldown room);
    use `--lr-decay-type flat` to go as low as 5.
  - On **GPU**, add `--no-reproducible`. `--reproducible` defaults to True
    (`torch.use_deterministic_algorithms(True)`), but Brevitas scale calibration uses
    `torch.kthvalue`, which has no deterministic CUDA kernel and will crash. CPU is fine as-is.

## Step 2 — Inspect the learned splits (optional) — in `PELICAN-nano/`

```bash
python scripts/check_scales.py --checkpoint model/fpga_model_qat_best.pt \
    --n-hidden 2 --weight-bit-width 24 --act-bit-width 24 --input-bit-width 24
```

Prints each quantizer's `scale = 2^-k => k fractional bits` and signedness — i.e., the optimal
split QAT found, before touching any firmware.

## Step 3 — Export weights + generated types — in `nPELICAN-fpga/`

```bash
cd ../nPELICAN-fpga
../PELICAN-nano/.venv/bin/python model_loader.py \
    --model ../PELICAN-nano/model/fpga_model_qat_best.pt --quant \
    --repo ../PELICAN-nano \
    --out firmware/weights/weights.h
```

Writes **both** headers with zero manual edits:

- `firmware/weights/weights.h` — snapped on-grid weights, BN constants, biases (read via Brevitas
  `quant_weight()`, never re-snapped to a coarser grid).
- `firmware/weights/types_generated.h` — every per-stage `ap_fixed` typedef, derived from the
  learned scales. `firmware/nPELICAN.h` `#include`s it.

Flags:

- `--bn-eps` (default `1e-5`) must match the **training** BatchNorm eps — the firmware BN scale
  is `weight / sqrt(var + eps)`. Omitting eps silently breaks small-variance channels.
- `--n-hidden` and the bit widths are auto-detected from the checkpoint shapes/quantizers.
- Drop `--quant`/`--repo` for a plain-float checkpoint; that path is standalone (no Brevitas).

**Type taxonomy generated:** quantization-point types straight from the learned scales
(`dot_t, t2_t, relu_t, t0_t, out_t, w1_gen_t, w2_gen_t`, each `ap_fixed<B, B-k>`); plus
intermediate/accumulator/MAC types formula-derived from those and the term counts
(`bn1out_t, tr_t, acc2_t, accrow_t, acc0_t, acc0row_t, mac2_t, mac0_t, bias_t_gen, bn_t_gen,
norm_t`). The quantization points use `AP_RND_CONV`; everything between them is exact-widened
(normalize-late: raw sums in wide accumulators, one rescale after).

## Step 4 — Export golden vectors — in `PELICAN-nano/`

```bash
cd ../PELICAN-nano
.venv/bin/python scripts/export_golden.py     # defaults: --num 200, test.h5 → ../nPELICAN-fpga/tb_data
```

Writes into `nPELICAN-fpga/tb_data/`:
`golden_pmu.dat`, `golden_nobj.dat`, `golden_logits.dat` (PyTorch quant-model logits),
`golden_dots.dat` (quantized `d_ij` for the dots-level gate), and `golden_stage_dump.txt`
(per-stage values for the first 3 events, for debugging). Options: `--checkpoint`, `--testfile`,
`--outdir`, `--num`, `--dump-events`.

## Step 5 — C-sim / synthesis — in `nPELICAN-fpga/`

```bash
cd ../nPELICAN-fpga

# C-sim only (fast functional check):
vitis_hls -f build_prj.tcl reset=1 csim=1 synth=0 cosim=0 validation=0 export=0 vsynth=0

# C-sim + synthesis (resources/timing):
vitis_hls -f build_prj.tcl reset=1 csim=1 synth=1 cosim=0 validation=0 export=0 vsynth=0
```

Target part `xcvu13p-flga2577-1-e`, 5 ns clock (`project.tcl`). The testbench auto-runs in golden
mode when `tb_data/golden_pmu.dat` exists and prints two **tolerance gates**:

```
GOLDEN SUMMARY:     events=200 exact=133 mismatch=67 max_abs_delta=6.3e-04 ...
GOLDEN GATE:        PASS (max_abs_delta=6.3e-04 vs tol=1e-3; 133/200 zero-tolerance exact)
DOTS-LEVEL SUMMARY: events=200 exact=142 mismatch=58 max_abs_delta=1.1e-05 ...
DOTS-LEVEL GATE:    PASS (max_abs_delta=1.1e-05 vs tol=1e-4; 142/200 zero-tolerance exact)
```

- **GOLDEN GATE** — momenta path: the firmware computes `d_ij` from the four-momenta via `dot4`.
  Tolerance `1e-3`, which covers the dot4 front-end caveat.
- **DOTS-LEVEL GATE** — network isolated: PyTorch's quantized `d_ij` are injected in place of
  `dot4` (`golden_dots.dat` → `npelican_dots_override`, csim-only). Tolerance `1e-4`.

**Why tolerance, not zero-tolerance 200/200:** BatchNorm is kept in float (an architecture
invariant), so the datapath has unquantized float segments between the learned quantizers.
PyTorch evaluates these in float32, the firmware in fixed-point; they round to the same grid
point at a quantizer output on most—but not all—events, so float-vs-fixed rounding tips a
boundary on a minority, cascading to ≤~1e-5 on the logit. This is intrinsic (PyTorch's own
float32-vs-float64 logits already differ ~3.6e-6), not a width bug. The **dot4 front-end caveat**
(plan D4) is the extra golden-path residual: PyTorch computes `d_ij = E^2-|p|^2` in lossy float32
(catastrophic cancellation on self-dots), the firmware exactly from the momenta. The dots-level
gate removes that variable and verifies the network in isolation. See `docs/resource_log.md`.

> **Vitis runs remotely.** Vitis HLS lives on the JupyterHub Linux box
> (`/home/jovyan/PTQWorkflow/nPELICAN-fpga`), not on macOS. Transport is git: commit/push here,
> `git pull` there, run `vitis_hls`. `csynth` currently does not complete (pre-existing
> high-particle-count issue), so LUT/FF/DSP/II numbers are still TBD.

### Local smoke check (macOS, no Vitis)

```bash
./build_local.sh && ./tb_local      # g++ + open-source ap_types; prints the same GATE lines
```

For the retyped firmware this is **bit-identical to Vitis** (verified: `-O0`==`-O2`, and it
matched the remote csim to 16 digits), so it's a trustworthy local oracle for iterating.
Requires `third_party/ap_types` (cloned by `build_local.sh`; git-ignored).

---

## Changing bit widths (the QAT split sweep)

Re-run the loop with new widths — e.g. 16/16/16:

```bash
# 1. retrain
cd PELICAN-nano
.venv/bin/python train_pelican_nano.py --datadir ./data/sample_data --target is_signal \
    --quant --po2-scales --weight-bit-width 16 --act-bit-width 16 --input-bit-width 16 \
    --n-hidden 2 --nobj 20 --nobj-avg 49 --num-epoch 8 --batch-size 256 \
    --drop-rate 0.05 --drop-rate-out 0.05 --weight-decay 0.005 --prefix fpga_model_qat16 --cpu

# 2. (optional) inspect the learned splits
.venv/bin/python scripts/check_scales.py --checkpoint model/fpga_model_qat16_best.pt \
    --weight-bit-width 16 --act-bit-width 16 --input-bit-width 16

# 3. regenerate firmware weights + types (no header edits)
cd ../nPELICAN-fpga
../PELICAN-nano/.venv/bin/python model_loader.py \
    --model ../PELICAN-nano/model/fpga_model_qat16_best.pt --quant --repo ../PELICAN-nano \
    --out firmware/weights/weights.h

# 4. golden vectors for the new model
cd ../PELICAN-nano && .venv/bin/python scripts/export_golden.py \
    --checkpoint model/fpga_model_qat16_best.pt

# 5. C-sim (+ synth)
cd ../nPELICAN-fpga && vitis_hls -f build_prj.tcl reset=1 csim=1 synth=1 cosim=0 validation=0 export=0 vsynth=0
```

`types_generated.h` regenerates with `W=16` and `I=16-k` per quantizer automatically.

**Caveats for the sweep:**

- Widths are **per-category** (`weight`/`act`/`input` each share one flag) — not yet
  per-individual-quantizer. Per-quantizer control is a small training-side enhancement
  (more flags + `QuantConfig` fields).
- At much smaller widths, re-verify the generator's intermediate-type margins. The BN/bias/
  accumulator widths are formula-derived for bit-exactness and were validated at 24-bit;
  the coarser grids of small widths are generally easier, but the offline half-LSB checks
  should be re-confirmed.
- `input_t` (raw-momentum interface type, `ap_fixed<36,12>`) is separate from the learned
  widths: `--input-bit-width` sets the `input_quant` (dots) grid, while the momentum container
  is hand-sized for range/precision of `dot4`.

---

## Quick reference

| Step | Repo | Command |
|------|------|---------|
| Train | PELICAN-nano | `python train_pelican_nano.py --quant --po2-scales --weight/act/input-bit-width N ...` |
| Inspect splits | PELICAN-nano | `python scripts/check_scales.py --checkpoint model/<prefix>_best.pt ...` |
| Export weights+types | nPELICAN-fpga | `python model_loader.py --model ../PELICAN-nano/model/<prefix>_best.pt --quant --repo ../PELICAN-nano --out firmware/weights/weights.h` |
| Export golden | PELICAN-nano | `python scripts/export_golden.py --checkpoint model/<prefix>_best.pt` |
| C-sim / synth | nPELICAN-fpga | `vitis_hls -f build_prj.tcl reset=1 csim=1 synth=1 cosim=0 validation=0 export=0 vsynth=0` |
| Local smoke | nPELICAN-fpga | `./build_local.sh && ./tb_local` |
