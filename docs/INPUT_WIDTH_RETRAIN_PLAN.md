# Input-width retrain plan: spend freed headroom on finer fractional resolution

Status: REVISED 2026-07-09 after review — the original Phase A (sweep
`--input-bit-width`) targets the DOT grid, which existing w6a6i6 runs already
prove tolerant down to 6 bits. The genuinely open experiment is the MOMENTUM
grid (`input_t`), which today exists only at export time. See "Two widths".
Companion analysis: `analysis/dot_scales.py` → `analysis/out/`. Resource
context: `docs/RESOURCE_REDUCTION_LEVERS.md` (Lever 2).

## Two widths — do not conflate them

| | training knob | firmware type | what it grids | resource role |
|---|---|---|---|---|
| dot grid | `--input-bit-width` (Brevitas `input_quant`, learned po2 scale) | `dot_t` | the dot products d_ij | minor (post-dot datapath) |
| momentum grid | **none today** — only export-time cap `--max-input-bits` | `input_t` | raw 4-momenta, the dot4 multiplier operands | **major** (dot4 mults dominate DSP/LUT) |

`input_t` is NOT a learned quantizer. The loader widens it analytically
(`model_loader.py` ~382–397): `INPUT_I = int_bits(|p|max)` (physics range),
`INPUT_F = ceil(log2|p|max) + dot_F + 3` — the F needed so momentum rounding
never moves a firmware dot off the PyTorch-float dot's grid point (bit-exact
contract). `--max-input-bits` shaves that F **post-hoc, no retraining by
construction**: the golden sweep cliffed at ~14 bits and collapsed ≤12
(momentum-grid drift + beams ±1 unrepresentable at negative F).

## Facts established so far (2026-07-09)

- Dot distribution (10k events, masked i<j, float32): span 2^-25.5→2^13.5,
  median 6.7, p99.9 = 1279. CCDF: ≥2^11 0.023%, ≥2^10 0.18%, ≥2^9 0.81%,
  ≥2^8 2.6%, ≥2^7 ≈5%, ≥2^5 ≈25%. Saturating pairs are 99.2% rank ≤1
  (leading constituents). `analysis/out/dot_scales_summary.txt`.
- **Dot grid is cheap (already proven by training runs):** w6a6i6 checkpoint's
  6-bit input_quant learned scale 4 = 2^+2 → grid step 4, clip 2^7 = 128,
  ~5% of dots clipped — AUC held. The professor's hypothesis (clip the tail,
  keep the bulk coarse) is CONFIRMED at the dot level.
- Phase A smoke (w24a24i14, sample_data, 8 ep): learned scale 2^-1, clip 2^12,
  test AUC 0.9007 vs 0.9036 baseline. Consistent: retrained dot-width is a
  solved axis. Sweep runner `PELICAN-nano/scripts/sweep_input_width.sh` kept
  for reference/reproduction, but this axis is NOT the priority.
- Momentum rescaling (divide by 2^s) is width-invariant with a learned/derived
  scale — pure binary-point relabel, proven no-op (Lever 2 notes).

## Phase A* — momentum quantizer in training (the real experiment)

Why: nothing in training currently sees the momentum grid, so `--max-input-bits`
failures at ≤14 bits say nothing about what a RETRAINED model tolerates. Put the
grid in the loop:

1. **Model change (PELICAN-nano):** add an optional Brevitas `QuantIdentity` on
   the 4-momenta immediately before `dot4` in `PELICANNano.forward`
   (`src/models/pelican_nano.py:100-102`), controlled by
   `QuantConfig.pmu_bit_width: Optional[int] = None` + CLI `--pmu-bit-width`
   (None = off, exact current behavior — float path and existing QAT
   checkpoints unaffected). Per-tensor, signed, po2 scale. Learned scale is
   fine (loader reads it); physics says it should land near
   I = int_bits(|p|max) ≈ 12.
   - Quantize BEFORE beams are dotted too — beams flow through the same
     quantizer (they're part of Pmu after collate), matching firmware where
     beams are input_t.
   - Masking invariant: 0 must stay exactly representable (po2 grid ⇒ yes).
2. **Golden export (PELICAN-nano `scripts/export_golden.py`):** goldens must
   come from the pmu-quantized model so the bit-exact gate stays meaningful.
3. **Loader (nPELICAN-fpga `model_loader.py`):** when the checkpoint has a pmu
   quantizer, derive `input_t` W/I directly from its learned scale + width
   (I = B−k), replacing the analytic `INPUT_F` bound and `--max-input-bits`.
   The bit-exact contract becomes exact again: PyTorch and firmware now compute
   dots from identically gridded momenta.
4. **Sweep** `--pmu-bit-width` ∈ {18, 16, 14, 12, 10} at the known-good
   production quant settings (w6a6i6), fixed seed, full dataset. 18 first as a
   sanity anchor (should match the current 18-bit operating point's AUC).

Per width record: AUC (float + current baselines alongside), learned pmu scale
k (extend `scripts/check_scales.py` — it walks QuantIdentity modules, so the
new one appears automatically), fraction of momentum components clipped, and
after export: golden gate pass + remote csynth LUT/FF/DSP.

## What success buys (honest expectation)

Current operating point: input_t 18 → dot mults 1 DSP each (DSP48 threshold
18). Retrained 14→12-bit momenta would shrink the 24×24→(capped 18×18) mults
to 12×12: DSP count likely flat (already 1/mult) but **LUT/FF shrink across
the whole dot front-end + BN1** (LUT is the binding resource, 53% SLR), and
12×12 mults become candidates for LUT implementation, freeing ~840-1012 DSPs
entirely if we choose to bind them there. If AUC holds at 10–12 bits where the
post-hoc cap collapsed, that's the headline result: retraining moved the cliff.

## Phase B — forced-scale scan (unchanged, optional)

Decouple clip tolerance from resolution at the DOT level if ever needed:
`QuantConfig.input_scale_exp` forcing `ScalingImplType.CONST`,
`scaling_init = 2^(W-1-k)`. Largely superseded by the w6a6i6 evidence.

## Phase C — structural fallback (unchanged)

Per-event po2 normalization by leading-energy exponent (barrel shift). Only if
uniform momentum clipping in Phase A* hurts; tail is 99% leading-constituent.

## Practicalities / gotchas

- `--no-reproducible` for QAT on GPU (`torch.kthvalue` has no deterministic
  CUDA kernel); CPU can keep `--reproducible`. `--lr-decay-type cos` needs
  `--num-epoch ≥ 8`. Fix `--seed` across sweep points.
- QAT reload: `model.train(); model(batch)` before `load_state_dict` when
  building a fresh model around a checkpoint.
- Keep `--nobj 20` (firmware NPARTICLES).
- Float path must stay bit-for-bit unaffected when quant flags are off.

## Results — Phase A* (momentum width, at w6/a6/i6)

| pmu W | learned k | momentum clip | AUC | golden gate | LUT / FF / DSP | notes |
|---|---|---|---|---|---|---|
| off (float pmu) | — | — | | n/a | 1347 DSP / 230k LUT baseline | current 18-cap operating point |
| 18 | | | | | | sanity anchor |
| 16 | | | | | | |
| 14 | | | | | | post-hoc cliff was here |
| 12 | | | | | | post-hoc collapse ≤ here |
| 10 | | | | | | |

## Appendix: original Phase A (dot-width sweep) — superseded

Sweep `--input-bit-width` at w24/a24 via `scripts/sweep_input_width.sh`
(env: WIDTHS/DATADIR/EPOCHS/DEVICE/SEED). Kept for reproduction; the w6a6i6
production runs already answered this axis (6 bits OK, learned clip 2^7).
