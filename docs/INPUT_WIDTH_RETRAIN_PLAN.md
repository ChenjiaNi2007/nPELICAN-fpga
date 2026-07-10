# Input-width retrain plan: spend freed headroom on finer fractional resolution

Status: Phase A* SWEEP DONE 2026-07-09 (full-dataset sweep {18,16,14,12,10}
complete — retraining moved the cliff from 14 to below 12; see Results).
Owed: clip-fraction measurement, golden gate + remote csynth per width.
Original Phase A (sweep `--input-bit-width`) targeted the DOT grid,
which existing w6a6i6 runs already prove tolerant down to 6 bits. See "Two
widths".
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

**Implementation status (2026-07-09): steps 1–3 DONE.**
`QuantConfig.pmu_bit_width` / `--pmu-bit-width` (train, check_scales,
export_golden — old checkpoints unaffected, field defaults None), loader reads
the trained grid (auto-detects from ckpt args, `--pmu-bit-width` override,
`--max-input-bits` ignored with a notice when present). 48/48 tests pass incl.
new `tests/test_pmu_quant.py`; loader output byte-identical on non-pmu
checkpoints (gated vs git HEAD). Sweep runner:

```bash
cd PELICAN-nano
bash scripts/sweep_pmu_width.sh          # sample_data smoke, CPU
DATADIR=<full-dataset-dir> EPOCHS=35 DEVICE="--cuda --no-reproducible" \
    bash scripts/sweep_pmu_width.sh      # full Phase A* sweep
```

Smoke result (pmu=14 @ w6a6i6, sample_data, 8 ep, CPU): learned pmu scale
2^-3 → `input_t = ap_fixed<14,11>` (clip ±1024 vs |p|max≈1947, LSB 0.125 GeV),
test AUC 0.9005 vs 0.9036 no-pmu baseline. Loader + export_golden verified on
that checkpoint (`model/fpga_model_qat_w6a6i6p14_best.pt`).

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

Full-dataset sweep run 2026-07-09 (`scripts/sweep_pmu_width.sh`, checkpoints
`model/fpga_model_qat_w6a6i6p<W>_best.pt`). AUC = best-checkpoint test AUC
(that's what gets exported); learned k = pmu_quant fractional bits →
`input_t = ap_fixed<W, W−k>`, clip ±2^(W−k−1). Momentum clip fraction not yet
measured; golden gate + csynth owed per width.

| pmu W | learned k | implied input_t (clip / LSB) | AUC | golden gate | LUT / FF / DSP | notes |
|---|---|---|---|---|---|---|
| off (float pmu) | — | analytic <18,12> (post-hoc cap) | — (not re-run this sweep) | n/a | 1347 DSP / 230k LUT baseline | current 18-cap operating point |
| 18 | 7 | <18,11> (±1024 / 2⁻⁷) | 0.9568 | owed | owed | sanity anchor — matches operating point ✓; dot scale 2⁴; best ep 5 |
| 16 | 5 | <16,11> (±1024 / 2⁻⁵) | **0.9592** | owed | owed | best of sweep; dot scale 2³; best ep 6 |
| 14 | 4 | <14,10> (±512 / 2⁻⁴) | 0.9563 | owed | owed | post-hoc cliff was here — retrained it's free; dot scale 2³; best ep 6 |
| 12 | 2 | <12,10> (±512 / 2⁻²) | 0.9519 | owed | 229.3k / 61.4k / 1173 (14 cyc, II=1) | post-hoc collapse ≤ here — retrained costs only ~0.005 AUC; dot scale 2⁴; best ep 3, final/best loss gap 0.295/0.274 (least stable run) |
| 10 | 1 | <10,9> (±256 / 2⁻¹) | 0.9305 | owed | owed | real degradation — the retrained cliff is between 12 and 10 |

### Analysis (2026-07-09)

**Headline: retraining moved the cliff.** The post-hoc `--max-input-bits` cap
cliffed at ~14 and collapsed ≤12; with the grid in the training loop, 14 bits
is accuracy-free (0.9563, within noise of the 18-bit anchor 0.9568 and the
16-bit best 0.9592), 12 bits costs only ~0.005–0.007 AUC (0.9519), and the
cliff has moved below 12 — only 10 bits shows real damage (0.9305, ≈−0.026).

- **Sanity anchor holds.** 18-bit lands at 0.9568; the 16/18/14 spread
  (0.9592/0.9568/0.9563) is ~0.003, i.e. seed-level noise. 16 nominally
  winning over 18 confirms these widths are not the binding constraint.
- **Learned scales confirm clip-the-tail at the momentum level too.** Physics
  says I ≈ 12 (|p|max ≈ 1947 GeV), but every width learned I = 9–11
  (clip ±256–1024): the model consistently spends bits on resolution and
  saturates leading constituents, mirroring the dot-level finding. Note the
  full-dataset 14-bit run learned k=4 (clip ±512), one notch tighter than the
  sample_data smoke's k=3 (±1024).
- **The ≤12-bit post-hoc failure mode (beams unrepresentable) is gone by
  construction:** all learned grids have k ≥ 1 (LSB ≤ 0.5), so the ±1 beam
  spurions sit exactly on-grid at every width including 10.
- **Dot grid co-adapts** (input_quant scale moved 2⁴/2³/2³/2⁴/2² across
  18/16/14/12/10) — another reason post-hoc capping understated tolerance.
- **Caveats:** best epochs were early (3–6), and the 12-bit run shows a
  final-vs-best loss gap (0.2954 vs 0.2740, best at ep 3) — low-width QAT is
  noisier; a second seed at 12 would firm up the ~0.005 cost estimate before
  committing. Clip fractions not yet measured.

**Recommendation:** take **12 bits** as the resource-work target (12×12 dot
mults → LUT-implementable, per "What success buys") if ~0.005 AUC is
acceptable; **14 bits** is the zero-cost fallback. Next: export both, run the
golden gate, remote csynth for the LUT/FF/DSP columns, and add a clip-fraction
measurement to `check_scales.py`.

### csynth @ pmu 12 (2026-07-09) vs baseline 1347 DSP / 63.3k FF / 229.8k LUT

DSP 1173 (−174), FF 61.4k (−1.9k), LUT 229.3k (−0.5k, ~flat), 14 cyc II=1
intact. Reading: the −174 DSP ≈ the +172 beam-port cost measured in the split
attribution — consistent with narrow beam-side mults dropping out of DSP.
LUT barely moved because the binding LUT is NOT in the dot front-end:
np_eq2to2 alone is 148.6k (51%) and BN1 38k, neither touched by input_t.
So the 12-bit retrain leaves LUT (~53% SLR) as the binding resource. Do NOT
bind the 12×12 dot mults to LUTs — that trades into the binding resource.
UPDATE 2026-07-10: Lever 6 (`mac_dsp=1`) measured byte-identical → DEAD
(constant-weight mults are strength-reduced, nothing for BIND_OP to bind;
see RESOURCE_REDUCTION_LEVERS.md). Remaining LUT levers: w1_2to2 sparsity
retrain, Vivado post-synth reality check, II relaxation, fewer particles.
Owed: `split=1` re-attribution at pmu 12, golden gate.

## Appendix: original Phase A (dot-width sweep) — superseded

Sweep `--input-bit-width` at w24/a24 via `scripts/sweep_input_width.sh`
(env: WIDTHS/DATADIR/EPOCHS/DEVICE/SEED). Kept for reproduction; the w6a6i6
production runs already answered this axis (6 bits OK, learned clip 2^7).
