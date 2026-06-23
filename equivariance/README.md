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

## ⚠️ Two facts that shape the measurement (read first)

**1. The firmware hardcodes the beam spurions, so beams cannot be boosted.**
`firmware/nPELICAN.cpp` injects the two beam vectors `(1,0,0,±1)` internally and takes only
the **20 real particles** as its top-level input. So the SEAL "boost every four-vector
including the beams" protocol is *not runnable* through csim without changing the firmware
signature (which would break "csim path == firmware path"). We therefore boost **only the
real particles** (`boost_beams: false`, the only csim-faithful option).

**2. Because the beams stay fixed, the high-bit reference curve is NOT zero — by design.**
With fixed beams, the beam–particle dots `d(beam, Λp) = beam·Λp ≠ beam·p` even in exact
float64, so PELICAN-with-beams is invariant only under the beam-axis-preserving subgroup,
not full Lorentz. The highest-bit-width curve is thus a non-zero **"fixed-beam floor."**
Lower-bit curves rising *above* that floor are the precision artifact this sweep isolates.
(The float64 boost unit test in `gen_boosted_inputs.py` therefore validates only the
**real–real 20×20 dot block** — the genuine Lorentz scalars.)

This reinterprets the task spec's §2/§7 expectation (which assumed boosted beams ⇒ flat
zero floor); it is a property of this firmware, not a bug.

---

## Files

| File | Role |
|---|---|
| `config.yaml` | single source of truth (paths, boost grid, model list) |
| `equiv_common.py` | shared helpers (boost matrix, paths, AUC, `1/ε_B`) |
| `gen_boosted_inputs.py` | build the **canonical, build-independent** boosted dataset + float64 unit test |
| `run_sweep.py` | per model: regen weights/types → gate → run csim → stash logits |
| `compute_metrics.py` | pair `f_b(Λx)` vs `f_b(x)`, write `results.csv`, aggregates, plots |
| `canonical/` | generated: `equiv_pmu.dat`, `equiv_nobj.dat`, `manifest.csv`, `meta.json` |
| `results/` | generated: `logits_bit{bw}.dat`, `results.csv`, `aggregates.csv`, `plots/` |

The input `.dat` is **float text**; the cast onto each build's `input_t` happens in C++
(`copy_data`), so the canonical set is build-independent and reused verbatim across all
bit-widths (only `weights.h` / `types_generated.h` change per build).

---

## Run it

All commands use the PELICAN-nano virtualenv (brevitas/torch/yaml/matplotlib live there):

```bash
cd nPELICAN-fpga/equivariance
PY=../../PELICAN-nano/.venv/bin/python

# 1. Build the canonical boosted set once (reused by every build).
$PY gen_boosted_inputs.py                 # uses config.yaml (n_jets=2000, n_dir=8)

# 2. Sweep every model in config.yaml (regen weights → golden gate → csim → stash).
$PY run_sweep.py

# 3. Metrics + plots (re-derivable from results/results.csv alone).
$PY compute_metrics.py
```

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

**The floor is not zero — and that's correct here.** Because the firmware holds the beam
spurions fixed (see the warnings at the top), even the float-exact network is only invariant
under the beam-axis subgroup, so the reference curve *rises* with `|β|`. Treat it as the
baseline. **The precision story is the GAP between a low-bit curve and the reference**, not the
absolute height of any single curve.

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
