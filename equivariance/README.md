# Equivariance sweep — how bit-width reduction breaks Lorentz invariance

Measures how reducing nPELICAN's QAT bit-width degrades its Lorentz invariance, as a
curve of **equivariance violation vs boost magnitude `|β|`, one curve per bit-width**
(SEAL arXiv:2511.01982, GSEAL / Fig. 2 methodology).

**The oracle is the C++ HLS firmware run in csim.** Vitis is *not* required: the harness
builds the testbench with `g++` over the open-source `third_party/ap_types` (the
`build_local.sh` path), which `docs/WORKFLOW.md` certifies as **bit-identical to remote
Vitis csim**. Python owns only boost sampling, encoding, metrics and plots — the datapath
is never reimplemented in Python.

---

## Option A — beams are a firmware input, so they can be boosted (read first)

The firmware now takes the two beam spurions as a **top-level input** (`beam_input[2*4]`)
alongside the 20 real particles, so the harness can Lorentz-boost the beams *with* the
particles. This lets us run the measurement in **two modes against the same firmware build**:

**1. `boostedbeams` — the well-posed precision question.**
Boost the beams alongside the particles. PELICAN's `d_ij = p_i·p_j` are then Lorentz scalars
over the **full 22-vector set**, so the float-exact output is *exactly invariant*. Therefore
**100% of the residual violation at reduced bit-width is a fixed-point artifact** — nothing is
contaminated by intended frame-dependence. `gen_boosted_inputs.py` proves the float64
invariance with a **full 22×22 Gram gate (G2)** before any sweep runs.

**2. `fixedbeams` — the old "fixed-beam floor", kept as the baseline.**
Hold the beams at `(1,0,0,±1)`. The beam–particle dots `d(beam, Λp) ≠ d(beam, p)` even in
exact float64, so the reference curve is a **non-zero floor by design** (the network is only
invariant under the beam-axis subgroup). This is the curve the pre-Option-A sweep measured.

**The headline result is the overlay.** The **gap between the two `24:24:24` reference curves
IS the fixed-beam symmetry-breaking floor** — it quantifies how much of the original
"violation" was intended frame-dependence (fixed beams) versus genuine precision loss
(everything left under boosted beams). See `FINDINGS_optionA.md`.

**At `|β|=0` the two modes coincide** (boosted beams = fixed beams = `(1,0,0,±1)`), and the
widened firmware is byte-identical to the old hardcoded-beam firmware (**gate G1**, verified
200/200 bit-exact). For real deployment, drive `beam_input` with the two constant vectors.

> **Honest caveat (large `|β|`):** the boosted-beam reference need *not* reach zero at large
> `|β|`. Boosting inflates all energies; once they exceed the `input_t` integer range the
> encoding **saturates**. That residual is a real precision effect (dynamic range, not just
> rounding) — now correctly isolated, not conflated with symmetry breaking. See §7 / FINDINGS.

---

## Files

| File | Role |
|---|---|
| `config.yaml` | single source of truth (paths, boost grid, model list) |
| `equiv_common.py` | shared helpers (boost matrix, paths, AUC, `1/ε_B`) |
| `gen_boosted_inputs.py` | build the **canonical, build-independent** boosted dataset + float64 unit tests (incl. full-22×22 G2); emits **both** beams files |
| `run_sweep.py` | per model: regen weights/types → gate → run csim → stash logits, for `--mode {boostedbeams,fixedbeams,both}` |
| `compute_metrics.py` | pair `f_b(Λx)` vs `f_b(x)`, write per-mode `results_<mode>.csv`/`aggregates_<mode>.csv`/`plots_<mode>/`, and the **boosted-vs-fixed overlay** |
| `canonical/` | generated: `equiv_pmu.dat`, `equiv_nobj.dat`, `equiv_beams_fixedbeams.dat`, `equiv_beams_boostedbeams.dat`, `manifest.csv`, `meta.json` |
| `results/` | generated: `logits_<mode>_<label>.dat`, `results_<mode>.csv`, `aggregates_<mode>.csv`, `plots_<mode>/`, `plots_overlay/` |

The input `.dat` is **float text**; the cast onto each build's `input_t` happens in C++
(`copy_data`), so the canonical set is build-independent and reused verbatim across all
bit-widths (only `weights.h` / `types_generated.h` change per build).

---

## Run it

All commands use the PELICAN-nano virtualenv (brevitas/torch/yaml/matplotlib live there):

```bash
cd nPELICAN-fpga/equivariance
PY=../../PELICAN-nano/.venv/bin/python

# 1. Build the canonical boosted set once (reused by every build). Emits BOTH beams
#    files (fixed + boosted) and runs the float64 G2 full-22x22 invariance gate.
$PY gen_boosted_inputs.py                 # uses config.yaml (n_jets=2000, n_dir=8)

# 2. Sweep every model in config.yaml, BOTH beam conventions (regen weights → golden
#    gate → csim → stash mode-tagged logits). --mode {boostedbeams,fixedbeams,both}.
$PY run_sweep.py --mode both

# 3. Per-mode metrics + the boosted-vs-fixed overlay (the headline / fixed-beam floor).
$PY compute_metrics.py --mode both
```

> Legacy un-tagged `logits_<label>.dat` from a pre-Option-A run are left untouched;
> `compute_metrics.py --mode legacy` still reads them.

### Smoke test first (validate the loop in ~10 s)

```bash
$PY gen_boosted_inputs.py --n-jets 50 --n-dir 2
$PY run_sweep.py            # one 24-bit build
$PY compute_metrics.py
```

### Runtime expectations

Cost ≈ `n_jets × (1 + |β-grid|·n_dir)` csim evals per build. Defaults (2000 jets, 10
nonzero magnitudes, 8 dirs) = **162 000 evals/build ≈ 100 s** in compiled C++ on this
machine. `gen` ≈ 10 s. Scale `n_jets`/`n_dir` down for a smoke run, up for final numbers.

---

## Adding more bit-widths

This is the whole point of the sweep, and the harness is built for it. As you train QAT
checkpoints at other bit-widths, add one block per checkpoint to `config.yaml`:

```yaml
models:
  - {bit_width: 24, checkpoint: ../PELICAN-nano/model/fpga_model_qat_best.pt, qat: true, reference: true}
  - {bit_width: 16, checkpoint: ../PELICAN-nano/model/<prefix16>_best.pt,     qat: true, reference: false}
  - {bit_width:  8, checkpoint: ../PELICAN-nano/model/<prefix8>_best.pt,      qat: true, reference: false}
```

`bit_width` is just a label; the real fixed-point grids come from each checkpoint's learned
Brevitas scales via `model_loader.py`. Mark exactly one model `reference: true` (drawn as
the invariance floor on every plot — conventionally the highest bit width). Re-run
`run_sweep.py` then `compute_metrics.py`; no need to regenerate the canonical set.

> Each checkpoint must be a real `--quant --po2-scales` QAT model. Do **not** fake a
> bit-width by overriding `--weight/act-bit-width` on a checkpoint trained at a different
> width — that re-snaps weights onto a grid the model never trained for and corrupts the
> measurement.

---

## Where this fits in the project workflow

The sweep is an **evaluation** step that hangs off the existing train → export → C-sim loop
(see the workspace `CLAUDE.md`). It trains nothing; it consumes finished QAT checkpoints.

```
            PELICAN-nano                         nPELICAN-fpga
   ┌──────────────────────────┐        ┌──────────────────────────────────┐
   │ train_pelican_nano.py     │        │ model_loader.py  (per bit-width)  │
   │  --quant --po2-scales      │  .pt   │   → firmware/weights/*.h          │
   │  at bit-width B  ──────────┼───────▶│ export_golden.py → golden gate    │
   └──────────────────────────┘        │ g++ csim oracle (RUN_EQUIVARIANCE)│
                                         └───────────────┬──────────────────┘
                                                         │ logits
   equivariance/ harness:                                ▼
   gen_boosted_inputs.py ──▶ run_sweep.py ──▶ compute_metrics.py ──▶ plots + CSV
        (once)                (per checkpoint)        (once)
```

**Typical loop when you add a new bit-width:**

1. Train the QAT checkpoint in `PELICAN-nano/` (`--quant --po2-scales`, the bit-width you want).
2. Add one `models:` block to `config.yaml` pointing at the `.pt` (set its `bit_width` label;
   leave `reference: true` on the highest-bit model).
3. `run_sweep.py` — it regenerates that build's `weights.h`/`types_generated.h`, runs the
   **gate**, then runs csim over the canonical boosted set. The gate has two parts: (a) a hard,
   bit-exact check that `RUN_EQUIVARIANCE` reproduces the firmware's own golden-path output
   (validates the oracle plumbing — must pass for every build), and (b) an informational
   **bit-faithfulness** number, firmware-vs-PyTorch, which legitimately grows at low bit-width
   (float-BN boundary tipping) and is reported, not aborted. `model_loader.py` also now
   generates `result_t == out_t` so the firmware never clamps the logit (see `FINDINGS.md` §0).
4. `compute_metrics.py` — re-reads every available `logits_bit{bw}.dat` and redraws all curves
   with the new bit-width added. You do **not** regenerate the canonical set (same physical
   boosts across all bit-widths is the whole point), and you do **not** re-run csim for
   bit-widths already swept.

The canonical boosted set (`gen_boosted_inputs.py`) is generated **once** and is the fixed
experimental control. Only re-run it if you change `n_jets`, `n_dir`, `beta_grid`, `seed`, or
the data file — and if you do, re-run the whole sweep so all curves share one control.

> **Side effect to know:** `run_sweep.py` leaves `firmware/weights/*.h` regenerated from the
> *last* checkpoint it swept. That's normal (it's how each build is produced). If you need the
> firmware tree back at a specific checkpoint afterwards, re-run `model_loader.py` for it (or
> `git checkout firmware/weights/`).

---

## Interpreting the results

Everything below is read off `results/aggregates.csv` (one row per `bit_width × |β|`) and the
plots. Read the curves **relative to the reference (highest-bit) model**, which is drawn on
every plot as the **invariance floor**.

**Under `boostedbeams` the reference collapses to the numerical floor.** With the beams
boosted alongside the particles the float-exact network is *exactly* invariant, so the
`24:24:24` reference `score_drift` drops to ~1e-8 (orders of magnitude below the fixed-beam
curve) at small/moderate `|β|`. **The precision story is then the GAP between a low-bit curve
and this near-zero reference** — pure fixed-point artifact. (Under `fixedbeams` the reference
is the non-zero *fixed-beam floor*; the overlay's gap between the two reference curves
quantifies that floor — see `plots_overlay/` and `FINDINGS_optionA.md`.)

**Large-`|β|` is range, not rounding.** The boosted-beam reference rises again at `|β|≳0.8`:
boosting inflates energies past the `input_t` integer range and the encoding saturates. That
is a genuine dynamic-range precision effect, now isolated — not symmetry breaking.

What each curve tells you, and what "bad" looks like:

| Curve | Reads as | Healthy | Trouble sign |
|---|---|---|---|
| **score_drift** `⟨Δσ²⟩` (log-y) | per-event output instability under boost | smooth, low, parallel to the reference on log-y | a low-bit curve that *peels away* upward from the reference — and especially a sharp upward **knee** = the precision cliff |
| **flip_rate** | fraction of hard tags that cross `w=0` | grows slowly, stays small | a steep rise = scores piling up at the threshold; decisions are no longer boost-stable |
| **auc** | does the drift cost tagging power? | flat near the `|β|=0` value | a drop well below the reference at the same `|β|` = real loss of discrimination |
| **inv_eps_b** `1/ε_B@0.3` | background rejection (physics figure of merit) | flat, tracks the reference | collapse toward 1 = the tagger is failing on boosted events |

**How to locate the cliff.** Scan a fixed bit-width's `score_drift` left-to-right: the cliff is
the `|β|` where the curve bends sharply upward (on log-y, where the slope jumps). As bits drop,
expect the cliff to move to **smaller `|β|`** and eventually coincide with input saturation in
`input_t`/`dot4` (very large boosts overflow the momentum type — a real, intended part of the
effect). The headline claim *fewer bits → larger violation, with a cliff* is **supported** if
lower-bit curves sit progressively above the reference and develop the knee earlier.

**Read the three metrics together — they fail in a specific order.** They are deliberately
redundant because they catch different failure modes at different `|β|`:
- **score_drift** is the earliest, most sensitive probe (moves at the smallest `|β|`).
- **flip_rate** rises once drifted scores reach the `w=0` threshold.
- **auc / inv_eps_b** are *lagging* — they only move once enough jets have drifted to shift the
  whole distribution. If `inv_eps_b` degrades while `score_drift` is still flat, suspect a bug
  (it should be the other way round).

A useful sanity habit: confirm `auc` at `|β|=0` matches the model's known test AUC (the harness
reproduced 0.852 for the 24-bit model). If it doesn't, the encoding or the build is wrong, not
the physics.

See `FINDINGS.md` for the written-up reading of the current (24-bit-only) run.

---

## Metrics (all in `results/aggregates.csv`, plots in `results/plots/`)

| Plot | Metric | What it shows |
|---|---|---|
| `score_drift.png` | `⟨(σ(w₀)−σ(w_β))²⟩` (log-y) | headline equivariance violation (SEAL Fig.2) |
| `logit_drift.png` | `⟨|w₀−w_β|⟩` (log-y) | **operating-point-independent** drift; the fair cross-bit-width comparison (σ-drift over-weights models whose logits straddle 0) |
| `mean_abs_dsigma.png` | `⟨|Δσ|⟩` | same as score_drift, linear |
| `flip_rate.png` | `sign(w₀)≠sign(w_β)` | hard tag crossing `w=0` |
| `auc.png` | AUC of `{w_β vs truth}` | does drift cost tagging power? |
| `inv_eps_b.png` | `1/ε_B` at `ε_S=0.3` | background rejection (heavy ties at low bits inflate it — trust AUC there) |

**Read `score_drift` and `logit_drift` together.** A degraded low-bit model parks its outputs
in the flat tail of σ, suppressing σ-drift even when the logit moves — so a low σ-drift can mean
"less responsive," not "more invariant." `logit_drift` removes that confound. (In the real runs
the two diverge exactly this way; see `FINDINGS.md`.)

Each is paired *within* a bit-width: `f_b(Λx)` vs the same jet's `f_b(x)` (the β=0 row).
