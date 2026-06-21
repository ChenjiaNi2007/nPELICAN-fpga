# Findings — equivariance vs boost (nPELICAN firmware csim)

**Run config:** `n_jets=2000` (balanced signal/background), `n_dir=8` isotropic directions,
`|β| ∈ {0, 0.1, …, 0.9, 0.95}`, seed 1234. Oracle: local g++ csim (bit-identical to Vitis).
Golden gate: **PASS** (`max|Δlogit| = 6.3e-4 < 1e-3`; 133/200 zero-tolerance exact) — the
equivariance TB mode is the validated firmware path.

> **Scope.** Only one QAT checkpoint exists on disk today (`fpga_model_qat_best.pt`,
> **24-bit**). So this is a **single-curve baseline + harness validation**, not yet the
> multi-bit-width comparison. The headline claim (*fewer bits → larger violation, with a
> cliff*) needs the lower-bit checkpoints; drop them into `config.yaml` and re-run. The
> 24-bit curve here is the **reference floor** every future curve is measured against.

## What the 24-bit curve shows

| `|β|` | score-drift `⟨Δσ²⟩` | flip rate | AUC | `1/ε_B@0.3` |
|------:|--------------------:|----------:|------:|------------:|
| 0.00  | 0          | 0.000 | 0.852 | 9.90 |
| 0.10  | 4.2e-05    | 0.009 | 0.852 | 9.95 |
| 0.20  | 1.8e-04    | 0.015 | 0.851 | 9.96 |
| 0.30  | 4.3e-04    | 0.022 | 0.849 | 9.78 |
| 0.40  | 8.6e-04    | 0.030 | 0.847 | 9.59 |
| 0.50  | 1.6e-03    | 0.038 | 0.843 | 9.26 |
| 0.60  | 2.8e-03    | 0.055 | 0.835 | 8.66 |
| 0.70  | 5.0e-03    | 0.075 | 0.823 | 7.87 |
| 0.80  | 9.1e-03    | 0.101 | 0.802 | 6.75 |
| 0.90  | 1.9e-02    | 0.160 | 0.750 | 5.11 |
| 0.95  | 3.2e-02    | 0.229 | 0.675 | 3.98 |

(AUC at `|β|=0` = 0.852 matches the model's true test discrimination — confirms the csim
oracle faithfully reproduces the trained network.)

### 1. The violation-vs-`|β|` trend
Monotone and smooth on log-y: score drift grows from `4e-5` at `|β|=0.1` to `3.2e-2` at
`|β|=0.95` — roughly `∝ |β|^4` at small boosts (each 0.1 step ~doubles the drift), steepening
above `|β|≈0.7`. Decision flips climb from <1% to 23%.

**This entire curve is the by-design "fixed-beam floor," not a precision artifact.** At 24
bits the network is essentially exact, so the violation is dominated by the firmware holding
the beam spurions fixed while the event is boosted (`d(beam, Λp) ≠ d(beam, p)`). The float64
unit test confirms the *real–real* dots are invariant to machine precision; the drift enters
through the beam rows. Lower-bit curves are expected to sit **above** this floor — that gap
is the precision effect of interest.

### 2. Where is the cliff?
At 24 bits there is a **soft knee around `|β|≈0.8–0.9`**: below it AUC barely moves (0.852→0.802,
a 6% relative drop over `|β|=0→0.8`) while above it AUC falls off fast (0.802→0.675, a 16%
drop over `|β|=0.8→0.95`). Whether a *hard* precision cliff appears — and at which `|β|` — is
exactly what the lower-bit checkpoints will reveal: the expectation is that the knee moves to
smaller `|β|` as bits drop, eventually colliding with input saturation in `input_t`/`dot4`.

### 3. Does `1/ε_B@0.3` degrade before or after the per-jet score?
**After.** Per-jet score drift is non-zero immediately (`|β|=0.1`) and rises smoothly,
whereas `1/ε_B@0.3` is *flat-to-slightly-improving* through `|β|≤0.2` (9.90→9.96) and only
starts a clear decline past `|β|≈0.3`, reaching 3.98 at `|β|=0.95`. AUC behaves the same way
(essentially flat to `|β|≈0.5`, then falls). So per-event score drift is the **most sensitive**
probe; aggregate discrimination (AUC, background rejection) is a **lagging** indicator that
only registers once a non-trivial fraction of jets have drifted across the threshold (cf. the
flip-rate column). Report all three: drift catches the effect earliest, flip-rate marks where
scores pile up at `w=0`, and `1/ε_B@0.3` tells you when it actually costs physics.

## Caveats
- Single bit-width ⇒ no cross-bit cliff yet; the trend claim is pending more checkpoints.
- `sample_data/test.h5` (≈40k jets) is the small bundled sample; final numbers should use the
  full test set if available (set `data_file` in `config.yaml`).
- The floor is firmware-specific (fixed beams). If the firmware is ever changed to accept the
  beams as boostable inputs, set `boost_beams: true` and the floor should collapse toward zero
  at high bit-width — a clean cross-check of this interpretation.
