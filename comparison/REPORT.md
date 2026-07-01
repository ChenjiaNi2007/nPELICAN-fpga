# DeepSet vs nanoPELICAN — comparison report

Common task: **binary top-tagging**. **See `HANDOFF.md` for the current state and the runbook
to finish** — a fresh session should read that first.

> **⚠ DATASET CORRECTION (2026-07-01):** the comparison is being moved to **`data/toptag`**
> (nanoPELICAN's training set) for BOTH models. The earlier DeepSet numbers here were on
> `data/sample_data`, where nanoPELICAN is out-of-distribution (0.70 vs its real 0.95). The
> "nanoPELICAN AUC discrepancy" was a **dataset mismatch**, not fixed-point precision —
> `--max-input-bits` was a red herring. The DeepSet is being retrained on `data/toptag`; the
> tables below (from the old sample_data run) will be refreshed.

## Status

- [x] Phase 0 — context; **D2 part = `xcvu13p-flga2577-2-e`**.
- [x] Phase C code — adapter (now leading-pT-20 for 200-wide toptag), hooks, configs (now point
      at `data/toptag`), plot labels, softmax strip.
- [x] Resource/latency (csynth) collected — DeepSet + nanoPELICAN (see table; arch-valid).
- [~] Phase C5 — **retrain DeepSet on `data/toptag`** (blocked on pod GPU TF install — in prog).
- [ ] Phase C6 — ROC overlay on `toptag/test.h5` (after retrain).
- [ ] Phase D — DeepSet equivariance overlay with the toptag model (tooling ready).
- [ ] Refresh stale sample_data numbers below.

## D2 — common FPGA part

`xcvu13p-flga2577-2-e`. nPELICAN's `project.tcl` was `-1`; change to `-2` for the comparison
build. Resource counts (DSP/LUT/FF) are speed-grade-independent, so nPELICAN's prior numbers
stay valid; only timing closure (achievable clock) is re-evaluated.

## Resources & latency (A/B) — `deepset_resources.csv`, `npelican_resources.csv`

Numbers come from the **binary (Phase C) models**, not the stock 5-class DeepSet (QAT learns
task-dependent scales). Columns: `model, nconst, nbits, latency_cycles, clock_ns, latency_ns,
II, DSP, LUT, FF, BRAM, accuracy`. Plot `resource_vs_N.png`: DeepSet O(N) vs nPELICAN O(N²).

**C-synthesis estimates, `xcvu13p-flga2577-2-e`, 5 ns clock (200 MHz target):**

| model | nconst | nbits | DSP | LUT | FF | latency | II | AUC | acc |
|---|---|---|---|---|---|---|---|---|---|
| DeepSet | 20 | 8 | 124 | 665,981 | 343,142 | 144 cyc / 0.72 µs | 40 | 0.9518 | 0.8879 |
| nanoPELICAN | 20 | 18in | 1347 | 230,231 | 63,343 | 14 cyc / 0.070 µs | 1 | TBD | TBD |

Takeaways: nanoPELICAN is **~10× lower latency (70 ns vs 720 ns) and II=1 vs II=40** despite
O(N²); the resource cost trades **DSP↔LUT** — nanoPELICAN spends DSPs on its Minkowski dot
products (1347 DSP / 230k LUT, fits 1 SLR), the 8-bit DeepSet spends LUTs (666k LUT / 124 DSP,
154% of 1 SLR). **CAVEAT (not yet bit-matched):** DeepSet is 8-bit, nanoPELICAN is ~18-bit;
the DSP-vs-LUT split is precision-dependent (wide mults → DSP, narrow → LUT), so part of the
contrast is bit-width, not architecture. A matched-precision row is needed for the headline.
Estimates are csynth; Vivado logic-synth (A.6) gives lower/real LUT/FF.

## Discrimination (C6) — `roc_overlay.png`, `roc_summary.csv`

DeepSet vs nanoPELICAN(float) vs nanoPELICAN(firmware) on one ROC axis; scalar metric is
background rejection `1/ε_B @ ε_S=0.3` on the Top class. _Fill from roc_summary.csv._

## Equivariance under boost (D) — `../nPELICAN-fpga/equivariance/results/plots_*/`

DeepSet is the **non-equivariant baseline**: its score/logit drift should rise monotonically
with `|β|` and sit orders of magnitude above every nPELICAN curve (symmetry-by-construction
vs symmetry-by-luck). nPELICAN stays near the numerical floor (Lorentz-invariant).

## Caveats (state explicitly — from plan §0)

1. **Same part (D2):** all resource numbers on `xcvu13p-flga2577-2-e`.
2. **hls4ml vs hand-HLS:** the two toolchains optimize differently; absolute resource counts
   are toolchain-dependent, not purely architectural.
3. **O(N) vs O(N²):** DeepSet is linear in constituents, nanoPELICAN quadratic (pairwise
   dots) — the headline of the resource-vs-N plot.
4. **Feature asymmetry:** DeepSet sees jet-relative (pT, η_rel, φ_rel) — permutation-equiv
   only, NOT Lorentz; nanoPELICAN sees Minkowski dot products — permutation AND Lorentz
   invariant. This is the physics behind the Phase D contrast.
5. **DeepSet has no beam spurions** → its equivariance curve is mode-independent (drawn on
   both the boostedbeams and fixedbeams overlays).
6. **Binary, not 5-class:** the DeepSet is retrained on this task (`output_dim: 2`); the
   stock 5-class numbers are only a toolchain sanity-check, not the comparison numbers.
