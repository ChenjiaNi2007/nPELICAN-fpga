# Findings — Option A: boostable beams isolate the pure fixed-point equivariance effect

**What changed.** The firmware now takes the two beam spurions as a top-level input
(`beam_input[2*4]`), so the equivariance harness can Lorentz-boost the beams *with* the
particles. We sweep the **same firmware build** in two modes:

* **`boostedbeams`** — beams transform with the particles. PELICAN's `d_ij = p_i·p_j` are then
  Lorentz scalars over the **full 22-vector set**, so the float-exact output is *exactly
  invariant*. **100% of the residual violation is a fixed-point artifact.**
* **`fixedbeams`** — beams held at `(1,0,0,±1)` (the pre-Option-A measurement). The reference
  curve is a non-zero **fixed-beam floor** by design (network invariant only under the
  beam-axis subgroup).

**Run.** 7 QAT checkpoints (`W:A:I` = weight:act:input bits), `n_jets=2000` balanced,
`n_dir=8`, `|β| ∈ {0,0.1,…,0.95}`, seed 1234. Oracle: local g++ csim (bit-identical to Vitis).
162 000 events/build/mode. Numbers below are read off `results/aggregates_{boostedbeams,fixedbeams}.csv`
and `results/plots_overlay/`.

---

## Validation gates (all PASS)

| Gate | Statement | Result |
|---|---|---|
| **G1** | Widened firmware == old hardcoded-beam firmware at `\|β\|=0` (constant beams) | **byte-exact, 200/200**, all 7 checkpoints (verified by diffing new-vs-old golden output) |
| **G2** | Float64 **full 22×22** Gram invariant under boosted beams | **PASS**, max relative `\|Δd_ij\|` = **1.5e-14** (machine ε) over the full 2000-jet set |
| **G3** | `boostedbeams` 24-bit reference violation ≪ fixed-beam floor at small/moderate `\|β\|` | **PASS**, gap **6 700×–61 000×** in score-drift for `\|β\|≤0.3` (table below) |
| **G4** | Firmware-vs-PyTorch bit-faithfulness gate (β=0, constant beams) still holds | **PASS**, unchanged from the deployed firmware (24-bit max\|Δ\|=6.3e-4) |

> **G2 numerical note.** A *flat absolute* 1e-9 tolerance (as literally written in the spec) is
> below float64 noise here: Minkowski dots suffer catastrophic cancellation (`E_iE_j` terms
> reach ~1e7–1e8 at large boosts of GeV energies), so the rounding floor scales as `ε·E_iE_j`,
> not `ε·d_ij`. The absolute max we see is 1.8e-8 — which would *fail* a flat 1e-9 check. The
> physically correct gate normalizes by the term magnitude `E_iE_j`; that relative error is
> 1.5e-14, i.e. machine precision. This is the same convention the real-real check already used.

---

## 1. Headline — the reference collapses, and the gap IS the fixed-beam floor

`24:24:24` reference, boosted vs fixed (the `plots_overlay/reference_overlay_*.png` figures):

| `\|β\|` | score-drift fixed (floor) | score-drift boosted | ratio | AUC fixed | AUC boosted | flip-rate fixed | flip-rate boosted |
|---|---|---|---|---|---|---|---|
| 0.10 | 4.16e-05 | **6.23e-09** | 6 669× | 0.852 | 0.852 | 0.0091 | **0.0000** |
| 0.20 | 1.75e-04 | **5.87e-09** | 29 844× | 0.851 | 0.852 | 0.0148 | **0.0000** |
| 0.30 | 4.30e-04 | **7.07e-09** | 60 797× | 0.849 | 0.852 | 0.0216 | **0.0000** |
| 0.40 | 8.59e-04 | 3.58e-06 | 240× | 0.847 | 0.852 | 0.0298 | **0.0000** |
| 0.50 | 1.57e-03 | 1.23e-05 | 128× | 0.843 | 0.852 | 0.0383 | **0.0000** |
| 0.60 | 2.77e-03 | 1.81e-05 | 153× | 0.835 | 0.852 | 0.0552 | **0.0000** |
| 0.70 | 4.98e-03 | 1.21e-04 | 41× | 0.823 | 0.851 | 0.0751 | 0.0006 |
| 0.80 | 9.11e-03 | 4.19e-04 | 22× | 0.802 | 0.850 | 0.1008 | 0.0023 |
| 0.90 | 1.91e-02 | 1.93e-03 | 10× | 0.750 | 0.841 | 0.1601 | 0.0103 |
| 0.95 | 3.21e-02 | 6.17e-03 | 5× | 0.675 | 0.813 | 0.2293 | 0.0327 |

Reading this:

* **At small/moderate boost the reference is invariant to the numerical floor** (~6e-9 score
  drift, **zero** decision flips through `|β|≤0.6`, AUC pinned at 0.852). Under fixed beams the
  *same* events showed 4e-5–3e-3 drift and 1–6% flips. **That entire fixed-beam signal was
  intended frame-dependence, not precision loss** — exactly what Option A set out to prove.
* **The gap between the two reference curves IS the fixed-beam symmetry-breaking floor.** It is
  large (10³–10⁴×) where the boosted reference sits on the numerical floor, and shrinks at large
  `|β|` as the boosted curve climbs the saturation ramp (below).
* **Discrimination survives the boost under Option A**: AUC 0.852 → 0.813 across the whole grid,
  versus the fixed-beam collapse 0.852 → 0.675. The tagger is, for practical purposes, Lorentz
  invariant — the residual at large `|β|` is a precision effect, not a broken symmetry.

---

## 2. The large-`|β|` residual is input-range saturation, not rounding (honest caveat)

The boosted reference does **not** reach zero at large `|β|`, and that is expected and correct.
Boosting inflates every energy by up to `γ(0.95) ≈ 3.2`; once a momentum component exceeds the
`input_t` integer range the `AP_SAT` encoding **saturates**, which is a genuine *dynamic-range*
precision effect (not just rounding). The onset is visible as the knee in the boosted reference
`score_drift`: flat at ~6e-9 for `|β|≤0.3`, then 3.6e-6 (0.4) → 1.2e-5 (0.5) → 1.2e-4 (0.7) →
6.2e-3 (0.95). This residual is now **correctly isolated**: under fixed beams it was buried
inside the much larger symmetry-breaking floor.

This sharpens the precision story rather than weakening it: at reduced bit-width the violation
has two separable origins — a **rounding floor** at small `|β|` (set mostly by input/dot bits)
and a **range/saturation ramp** at large `|β|` (set by `input_t`'s integer bits). The optional
diagnostic (re-run the reference with 1–2 extra input integer bits) would push the ramp to
larger `|β|` while leaving the small-`|β|` floor unchanged; not required for the result.

---

## 3. Pure precision floor vs bit-width (boostedbeams, small-`|β|` where saturation is absent)

With symmetry breaking removed, the small-`|β|` score-drift floor is now a clean fixed-point
metric. `score_drift_mse` at `|β|=0.1` (no saturation yet):

| config | 6:6:6 | 6:6:8 | 6:6:12 | 8:8:16 | 12:12:16 | 16:16:16 | 24:24:24 |
|---|---|---|---|---|---|---|---|
| drift @ β=0.1 | 1.75e-05 | 8.43e-06 | 3.91e-08 | 7.30e-08 | 6.51e-09 | 2.30e-10 | 6.23e-09 |

* **Input bits set the small-`|β|` floor.** Along the input-bit sub-sweep (W=A=6 fixed) the floor
  drops ~3 orders of magnitude from i6 → i12 (1.75e-5 → 3.9e-8): more input/dot precision ⇒ a
  finer `d_ij` grid ⇒ a lower invariance floor. The two `i6` models (6:6:6, 6:6:8) sit
  ~10²–10³× above the `i≥12` models.
* **Not strictly monotonic across independently-trained checkpoints.** `16:16:16` (2.3e-10) sits
  *below* `24:24:24` (6.2e-9): these are separately-trained QAT checkpoints with different learned
  scales and different float-BN boundary-tipping, so the small-`|β|` round-off floor is not a pure
  function of the bit-budget. Read the floor as "input precision dominates, ± a per-checkpoint
  factor," not as a strict ladder.
* **At large `|β|` the curves converge onto the saturation ramp** (all configs reach 5e-4–9e-3 at
  `|β|=0.95`), because there the dominant error is `input_t` range, which the W:A:I budgets share
  more than they share the rounding floor. The notable exception is `16:16:16`, whose larger input
  integer range keeps it ~10× below the pack even at `|β|=0.95` (4.7e-4 vs ~6e-3).

(Full per-config × per-`|β|` numbers in `results/aggregates_boostedbeams.csv`; the same curves
under fixed beams in `results/aggregates_fixedbeams.csv`.)

---

## 4. Decision stability (boostedbeams) — the cleanest single statement

Decision-flip rate at `|β|=0.95` (fraction of jets whose hard tag crosses `w=0` under the
strongest boost) and AUC retention `β=0 → 0.95`:

| config | flip @ β=0.95 (boosted) | flip @ β=0.95 (fixed) | AUC β=0 | AUC β=0.95 (boosted) | AUC β=0.95 (fixed) |
|---|---|---|---|---|---|
| 6:6:6   | 0.0014 | 0.081 | 0.703 | 0.700 | 0.793 (tie-inflated) |
| 6:6:8   | 0.0082 | — | 0.779 | 0.773 | — |
| 6:6:12  | 0.0084 | — | 0.766 | 0.761 | — |
| 8:8:16  | 0.0057 | — | 0.815 | 0.799 | — |
| 12:12:16| 0.0024 | — | 0.768 | 0.761 | — |
| 16:16:16| 0.0009 | — | 0.731 | 0.718 | — |
| 24:24:24| 0.0327 | 0.229 | 0.852 | 0.813 | 0.675 |

Under Option A every config keeps essentially all of its discrimination across the full boost
grid (AUC within ~0.01–0.04 of the `β=0` value) and flips fewer than ~3% of decisions even at
`|β|=0.95`. Under fixed beams the 24-bit reference alone lost 0.18 AUC and flipped 23% of tags —
that difference is the frame-dependence Option A removes.

---

## 5. Reproduce

```bash
cd nPELICAN-fpga/equivariance
PY=../../PELICAN-nano/.venv/bin/python
$PY gen_boosted_inputs.py                 # emits both beams files + runs G2
$PY run_sweep.py --mode both              # 7 checkpoints × {boosted,fixed}; gate G1/G4 per build
$PY compute_metrics.py --mode both        # per-mode aggregates + plots_overlay/ + floor table
```

Artifacts: `results/aggregates_boostedbeams.csv`, `results/aggregates_fixedbeams.csv`,
`results/plots_boostedbeams/`, `results/plots_fixedbeams/`, `results/plots_overlay/`
(`reference_overlay_*.png` = headline; `all_overlay_*.png` = all bit-widths, solid=boosted /
dashed=fixed). The legacy pre-Option-A fixed-beam `results/logits_<label>.dat` and
`aggregates.csv` are left untouched.
