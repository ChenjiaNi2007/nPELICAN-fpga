# Input-width retrain plan: spend freed headroom on finer fractional resolution

Status: Phase A ready to run (2026-07-09). Companion analysis:
`analysis/dot_scales.py` → `analysis/out/` (dot-product scale distribution,
bits-needed plots, `dot_scales_summary.txt`). Resource context:
`docs/RESOURCE_REDUCTION_LEVERS.md` (Lever 2 = post-hoc `--max-input-bits` cap).

## Motivation (measured on tb_data/10k_pmu_test.dat, 2026-07-09)

The masked off-diagonal dot products d_ij (top-20 constituents + massless beams,
float32) span 2^-25.5 → 2^13.5, median 6.7, p99.9 = 1279. Against the current
`input_quant`-derived dot grid `ap_fixed<24,12>` (clip 2^11, LSB 2^-12):

- only **0.023%** of dots exceed the clip point (517 dots across 443 of 10k events,
  median 1 per affected event);
- **92.5%** of those saturating pairs involve the leading (highest-energy)
  constituent, **99.2%** involve rank ≤ 1.

The high tail is a handful of leading-constituent pairs. If the task tolerates
clipping them, integer headroom can be re-spent on fractional resolution for the
bulk — or the total width cut.

## Reframing: why "divide by C and retrain" must be run as a width/scale sweep

The Brevitas input quantizer (`QuantIdentity` built on `Int8ActPerTensorFloat`
with `RestrictValueType.POWER_OF_TWO`) has a **learned** scale: initialized from
runtime percentile stats, then trained by gradient. Dividing all momenta by 2^k
moves the distribution and the learned scale together — a pure binary-point
relabel (already proven no-op for Lever 2). The two experiments that are NOT
no-ops:

1. **Shrink the quantizer width W** (`--input-bit-width`) and retrain: the learned
   po2 scale 2^-k re-places the clip point 2^(W-1-k), trading tail clipping
   against bulk resolution, and the weights/BN adapt to the clipped dots.
2. **Force the scale** (fixed k) at constant W: directly maps AUC vs clip
   threshold, decoupled from resolution.

The earlier ~14-bit cliff (RESOURCE_REDUCTION_LEVERS.md Lever 2 sweep) was
post-hoc: 24-bit-trained model, scale frozen at 2^-12, bits shaved with no
retraining. Retraining at low W is the untested regime.

## Phase A — width sweep with retraining (zero code changes)

Sweep `--input-bit-width` at fixed weight/act width 24, otherwise the exact
flags of the current firmware checkpoint `fpga_model_qat` (from
`log/fpga_model_qat.log`). Runner:

```bash
cd PELICAN-nano
bash scripts/sweep_input_width.sh                    # sample_data smoke, CPU
# full run (cluster / GPU):
DATADIR=<full-dataset-dir> EPOCHS=35 DEVICE="--cuda --no-reproducible" \
    bash scripts/sweep_input_width.sh
```

Checkpoints land at `model/fpga_model_qat_w24a24i<W>_best.pt`, logs at
`log/fpga_model_qat_w24a24i<W>.log`. Per width record:

| record | how |
|---|---|
| valid/test AUC | trainer log (script greps the tail) |
| learned k (input_quant scale = 2^-k) | `scripts/check_scales.py` (script runs it) |
| implied clip point 2^(W-1-k) | arithmetic |
| predicted clipped fraction | CCDF table below / `analysis/out/dot_scales_summary.txt` |

Reference CCDF of the dots (predicted fraction clipped for a given clip point):

| clip at | 2^11 | 2^10 | 2^9 | 2^8 | 2^5 |
|---|---|---|---|---|---|
| clipped | 0.023% | 0.18% | 0.81% | 2.6% | ~25% |

Sanity check: at W=14 a sensible learned optimum is k ≈ 3–5 (clip 2^9–2^10,
grid 2^-3–2^-5 vs median dot 6.7). A wildly different learned k suggests the
po2 log-domain scale learning plateaued, not physics — rerun with a different
seed / init before drawing conclusions.

Baselines needed for the AUC column: float model (no `--quant`) and the
existing 24-bit QAT run, same data/epochs/seed.

## Phase B — forced-scale scan (small code change; decouples clip from resolution)

Add to `PELICAN-nano/src/layers/quant.py`:

- `QuantConfig.input_scale_exp: Optional[int] = None`
- In `make_act_quant`, when set: `scaling_impl_type = ScalingImplType.CONST`,
  `scaling_init = 2.0 ** (bw - 1 - k)` (Brevitas scaling init is the clip
  threshold; LSB 2^-k follows from the width).
- CLI: `--input-scale-exp` in `src/trainer/args.py`, plumbed through
  `train_pelican_nano.py`.

Then at fixed W=24 force k ∈ {13, 14, 15, 16} (clip 2^10 → 2^7) and retrain.
AUC flat down to clip ≈ 2^8–2^9 proves 2–3 integer bits are expendable,
independent of width.

## Firmware validation loop (per surviving checkpoint)

1. `cd nPELICAN-fpga && python model_loader.py --model ../PELICAN-nano/model/<prefix>_best.pt --quant --repo ../PELICAN-nano`
   — emits `weights/weights.h` + `types_generated.h`; the coarser learned dot
   grid narrows `dot_t` and (via dot_F → INPUT_F) `input_t` automatically, no
   Lever-2 cap needed.
2. Regenerate golden vectors (`PELICAN-nano/scripts/export_golden.py`), run the
   bit-exact C-sim gate.
3. Remote csynth; compare vs baseline 1347 DSP / 230k LUT / 63k FF.

Expected payoff, stated honestly: below the 18-bit input_t operating point the
dot multipliers stay 1 DSP each (DSP48 packing threshold), so further gains are
**LUT (the binding resource, 53% SLR) and FF**, plus the option of binding
narrow mults to LUTs. The science deliverable is the AUC-vs-width curve WITH
retraining: does the cliff move below 14, and how did each width spend its
headroom (learned k)?

## Phase C — structural fallback (only if uniform clipping hurts)

Tail is 99% leading-constituent pairs → per-event po2 normalization by the
leading energy's exponent (barrel shift, exactly representable). Changes the
input distribution (needs its own retraining) and adds firmware shift logic.
Reach for it only if Phases A/B show the tail carries signal.

## Practicalities / gotchas

- `--no-reproducible` required for QAT on **GPU** (Brevitas `torch.kthvalue`
  has no deterministic CUDA kernel); CPU runs can keep `--reproducible`.
- LR scheduler: `--lr-decay-type cos` needs `--num-epoch ≥ 8`.
- Fix `--seed` across widths for comparability.
- QAT checkpoint reload: `model.train(); model(batch)` before
  `load_state_dict(strict=True)` (check_scales.py handles its own path).
- Keep `--nobj 20` (firmware NPARTICLES), NOT the 80 used by the old full
  PELICAN slurm job.

## Results

| W (input) | learned k | clip 2^(W-1-k) | % clipped | valid AUC | test AUC | notes |
|---|---|---|---|---|---|---|
| float |  |  |  |  |  | baseline |
| 24 | 12 | 2^11 | 0.023% |  |  | current fpga_model_qat |
| 20 |  |  |  |  |  |  |
| 18 |  |  |  |  |  |  |
| 16 |  |  |  |  |  |  |
| 14 |  |  |  |  |  |  |
| 12 |  |  |  |  |  |  |
| 10 |  |  |  |  |  |  |
