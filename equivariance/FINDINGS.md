# Findings — equivariance & performance vs bit-width (nPELICAN firmware csim)

**Run:** 7 QAT checkpoints, labelled `W:A:I` = weight:act:input bit-widths. Two controlled
sub-sweeps sharing one architecture (nhid=2, config s, BN b, beams, scale 1):
  * **input-bit** sweep (W=A=6 fixed): `6:6:6`, `6:6:8`, `6:6:12`
  * **weight/act** sweep (I=16 fixed): `8:8:16`, `12:12:16`, `16:16:16`
  * **reference** `24:24:24` (highest precision).

`n_jets=2000` (balanced), `n_dir=8`, `|β| ∈ {0,…,0.95}`, seed 1234. Oracle: local g++ csim
(bit-identical to Vitis). Every model passed the **equiv == firmware gate bit-exactly
(max|Δ|=0, 200/200)**, so the oracle plumbing is verified for all builds. Boost protocol:
real particles only, beams fixed (see README) → the curves carry the by-design fixed-beam
floor; read them *relative to each other*, not against zero.

---

## 0. A firmware bug this sweep found and fixed (read first)

The final-logit type `result_t` was **hardcoded** in `firmware/nPELICAN.h` to `ap_fixed<24,1>`
(range [−1,1)). But each checkpoint's `output_quant` grid (`out_t`) differs — e.g. `16:16:16`
has `out_t = ap_fixed<16,3>` (range [−4,4)). So the firmware silently **clamped every logit to
[−1,1)**, collapsing all out-of-range scores onto ±1 and destroying the score ranking. Effect
before the fix: `16:16:16` firmware AUC = **0.30** (sub-random) vs its PyTorch model's 0.76;
`6:6:6` = 0.36. The clamp masqueraded as "low-bit models are broken."

Fix: `model_loader.py` now generates `result_t == out_t` (guarded, like `input_t`); `nPELICAN.h`
keeps the `<24,1>` value only as the float-export fallback. After the fix, firmware-vs-PyTorch
for `16:16:16` dropped from **max|Δ|=2.02 → 0.0011**, and all AUCs recovered to 0.70–0.85.
All numbers below are post-fix. (The 24:24:24 reference was unaffected — its `out_t` already
was `<24,1>`.)

---

## 1. Bit-faithfulness: does the firmware compute the trained model? (cleanest bit-width law)

Firmware vs PyTorch on 200 golden events, max|Δlogit| — this is *not* an equivariance metric;
it measures whether the deployed fixed-point firmware reproduces the float-trained model:

| config | 6:6:6 | 6:6:8 | 6:6:12 | 8:8:16 | 12:12:16 | 16:16:16 | 24:24:24 |
|---|---|---|---|---|---|---|---|
| max\|Δ\| | 0.375 | 0.25 | 0.25 | 0.188 | 0.047 | 0.0011 | 0.0006 |

Monotonic ↓ with bits. The residual is **float-BN boundary tipping**: PyTorch keeps BatchNorm
in float while the firmware uses fixed BN, and on coarse grids a tiny BN difference tips a
quantizer boundary and cascades to the logit. It shrinks as grids refine. **Takeaway:** below
~12 bits the FPGA computes a meaningfully *different* function than the trained checkpoint
(a deployment gap independent of equivariance) — worth closing (e.g. revisit fixed-BN
precision) if low-bit deployment is the goal.

---

## 2. Equivariance violation vs bit-width — two regimes, NO cliff

The naive hypothesis "fewer bits → larger violation, with a cliff" is **only half right, and
there is no cliff.** Two regimes (clearest in logit-space drift `⟨|w₀−w_β|⟩`, which — unlike
the σ-based score drift — is independent of where each model sits on the sigmoid):

**Small/moderate boost (|β| ≲ 0.3): fewer bits → MORE violation.** A quantization-noise floor:
coarse grids tip on the tiny perturbations a small boost induces, so the low-bit models sit
*above* the high-bit reference. Near-monotonic in total bit budget:

| config (W+A+I bits) | logit drift @ β=0.1 | score drift @ β=0.1 |
|---|---|---|
| 6:6:6 (18)   | 0.047 | 2.2e-4 |
| 6:6:8 (20)   | 0.033 | 2.3e-4 |
| 6:6:12 (24)  | 0.027 | 2.6e-4 |
| 8:8:16 (32)  | 0.034 | 1.7e-4 |
| 12:12:16 (40)| 0.022 | 4.4e-5 |
| 16:16:16 (48)| 0.012 | 1.1e-5 |
| 24:24:24 (72)| 0.013 | 4.2e-5 |

**Large boost (|β| ≳ 0.6): the genuine fixed-beam symmetry breaking dominates** and scales with
each model's *responsiveness* (its learned weight/BN magnitudes), so the bit-width ordering
washes out. The high-precision reference, which tracks the perturbation faithfully, rises
fastest and ends with the **largest** drift (score drift @ β=0.95: 24:24:24 = 3.2e-2, the top
curve; logit drift there is led by 8:8:16 = 0.62, with 24:24:24 = 0.47 — non-monotonic, set by
per-checkpoint weight scale, not bit budget). The curves cross around |β| ≈ 0.3–0.5.

**Score-drift vs logit-drift caveat.** The SEAL-style σ-drift over-weights `24:24:24`: its
logits straddle 0 (steep sigmoid) so a given Δw makes a larger Δσ, while the reduced-precision
models park in the sigmoid tail. Report both; logit-drift is the fairer cross-bit comparison.

**No cliff** in either axis: degradation is smooth in |β| (a soft knee ~0.8 for every config)
and smooth in bit-width. Nothing falls off a precision edge in 6→24 bits.

---

## 3. Tagging performance under boost (AUC, 1/ε_B@0.3)

AUC at β=0 (firmware, 2000 balanced jets), and how it survives the largest boost:

| config | AUC @ β=0 | AUC @ β=0.95 | 1/ε_B@0.3 (β=0) | flip-rate @ β=0.95 |
|---|---|---|---|---|
| 6:6:6    | 0.703 | 0.697 | 18.9 | 0.056 |
| 6:6:8    | 0.779 | 0.760 | 28.6 | 0.139 |
| 6:6:12   | 0.766 | 0.739 | 28.6 | 0.185 |
| 8:8:16   | 0.815 | 0.766 | 25.0 | 0.111 |
| 12:12:16 | 0.768 | 0.749 | 28.6 | 0.086 |
| 16:16:16 | 0.732 | 0.731 | 21.7 | 0.045 |
| 24:24:24 | 0.852 | 0.675 |  9.9 | 0.229 |

- **Full precision tags best** (0.852) — clearly above all reduced configs (0.70–0.82).
- Among reduced configs the AUC scatter is **dominated by run-to-run training variance**, not a
  clean bit-width law (8:8:16 at 0.815 beats 12:12:16 and 16:16:16, which were trained
  independently). The one clean single-knob effect is the **input-bit sweep at W=A=6: I=6→8
  lifts AUC 0.70→0.78** — 6-bit *inputs* are the binding constraint at that weight precision.
- **AUC robustness under boost mirrors §2:** the high-precision tagger loses the *most* AUC
  under boost (0.852→0.675, and 1/ε_B@0.3 collapses 9.9→4.0, flip-rate 23%) precisely because
  it is the most responsive; the blunt low-bit models barely move (16:16:16: 0.732→0.731,
  flips 4.5%). Quantization buys boost-stability at the cost of peak discrimination.
- `1/ε_B@0.3` is **larger for the reduced models** — an artifact: their logits are discretized
  with heavy ties, which inflates the rank-threshold rejection. Trust AUC over `1/ε_B` here.

---

## 4. Bottom line

1. **No equivariance cliff** across 6→24 bits, and the effect is *not* monotonic "fewer bits →
   more violation." It is two regimes: a **quantization-noise floor that raises small-boost
   violation as bits drop**, crossing over to **large-boost fixed-beam breaking that scales with
   model responsiveness** (so the full-precision model is *least* invariant at large |β|).
2. The dominant precision risk for this firmware is **not invariance but the float-BN deployment
   gap** (§1): below ~12 bits the FPGA function drifts from the trained model.
3. Full precision (24:24:24) is the best tagger but the *least* boost-stable; the reduced models
   trade discrimination for boost-stability. If small detector-frame boosts are the concern,
   higher precision is better (lower noise floor); if robustness to large boosts is the concern,
   the picture inverts.
4. **Fixed a real firmware bug** (`result_t` clamp) that the multi-bit sweep exposed.

## Caveats
- Reduced-precision checkpoints are independently trained, so cross-config AUC/large-β-drift
  differences mix the bit-width effect with training scatter. The *within-config* β-trends and
  the *small-β* bit ordering are the robust signals.
- Beams are held fixed (firmware constraint) → the violation includes the by-design fixed-beam
  floor; this is a relative comparison across bit-widths, not an absolute symmetry measurement.
- `sample_data/test.h5` (~40k jets) is the bundled sample; final numbers could use the full set.
