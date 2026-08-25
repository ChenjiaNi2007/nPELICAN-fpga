# Synthesis report index

Canonical home for nPELICAN synthesis reports. **All reports here are xcu250-figd2104-2L-e
at 5 ns**, so the numbers are mutually comparable. Reports on other devices, or from before
Lever 2, live outside the repo in `../../synthesis-archive/` — see that folder's README
before comparing anything against them.

Naming: `{csynth,vsynth}_<config>.rpt`.
`csynth` = Vitis HLS C-synthesis (estimates + Bind Op report). `vsynth` = post-Vivado
synthesis utilization (the number to quote; HLS estimates run high on LUT).

## Reading a report

- **vsynth** — `CLB LUTs` / `CLB Registers` / `DSPs` / `CARRY8` rows in the utilization table.
- **csynth** — the `+ nPELICAN` row of the Synthesis Summary carries slack, latency (cycles),
  II, and DSP/FF/LUT estimates. Vitis 2023.2 has **no** `Utilization Estimates` section; for
  per-operator attribution parse the **Bind Op Report** instead.
- Which multiplier bound where: `grep -oE "mul_[0-9]+s?_[0-9]+n?s?_[0-9]+" <csynth> | sort | uniq -c`.

## Current builds — 20 particles (NPARTICLES2 = 22, 253 pairs)

| Config | vsynth LUT | FF | DSP | CARRY8 | csynth lat | Date | Files |
|---|---|---|---|---|---|---|---|
| pmu-12 monolith (**baseline**) | 69,112 | 24,815 | 1,169 | 7,032 | 13 cyc | 2026-07-10 | `vsynth_monolith_pmu12.rpt`, `clock_sweep/csynth_mono_5ns.rpt` |
| pmu-12, no MAC-DSP pragma | 69,735 | 25,142 | 1,343 | 6,929 | 14 cyc | 2026-07-09 | `vsynth_monolith.rpt`, `csynth_monolith.rpt` |
| pmu-10 | 94,983 | 21,504 | 936 | 9,254 | — | 2026-07-13 | `vsynth_monolith_pmu10.rpt` |
| pmu-9 | 97,174 | 22,514 | 998 | 9,263 | 15 cyc | 2026-07-14 | `vsynth_monolith_pmu9.rpt`, `csynth_monolith_pmu9.rpt` |
| pmu-8 | 98,863 | 21,271 | 977 | 10,618 | 15 cyc | 2026-07-14 | `vsynth_monolith_pmu8.rpt`, `csynth_monolith_pmu8.rpt` |
| w8/a8/i8 + pmu-9 | 100,669 | 26,626 | 918 | 10,531 | 15 cyc | 2026-07-17 | `*_monolith_w8a8i8p9.rpt` |

The width levers all trade the same way: every DSP saved costs ~100+ LUT and CARRY8 goes
**up**. pmu-12 stays the operating point.

## particle_count/ — 16 particles (NPARTICLES2 = 18, 171 pairs)

| Config | vsynth LUT | FF | DSP | CARRY8 | csynth lat | Date |
|---|---|---|---|---|---|---|
| 16p pmu-12, `--bn-frac-bits 12` (**best II=1**) | 49,894 | 15,416 | **769** | 4,446 | 15 cyc, timing met | 2026-08-25 |
| 16p pmu-12 bnf12, **II=3** (experiment) | 48,012 | 15,686 | **313** | 4,268 | 19 cyc, timing met | 2026-08-25 |
| 16p pmu-12 (BN1 on DSP) | **48,419** | 15,487 | 941 | 4,111 | 15 cyc, slack −0.00 ⚠ | 2026-08-22 |
| 16p block-FP W=7 | 84,302 | 17,928 | 751 | 6,043 | 16 cyc | 2026-08-21/22 |

Lowest-LUT builds measured. Particle count is the only lever that moves LUT, DSP **and**
CARRY8 down together, because it removes work instead of relocating it. Pair count scales as
`s(s+1)/2` with `s = NPARTICLES + 2`; dot-stage DSP ≈ `4·(N+2)(N+3)/2`. Validated to <3%
(CARRY8 to 0.1%).

The II=3 row trades throughput for DSP: one event per 15 ns instead of per 5 ns (still
inside the 25 ns bunch spacing), dots time-multiplexed 3:1 onto 57 `dot4` units. It breaks
the II=1 invariant deliberately — an operating point, not the deliverable.

Two caveats on these rows:

1. **`csynth_16p_pmu12.rpt` fails timing** (`Issue Type: Timing`, slack −0.00, 15 cycles vs
   the 20p baseline's 13). Cause is the BN1 constant, not the particle count — see below.
2. **Resources only.** Both reuse the existing 20-particle weights. No AUC/bgRej has been
   measured at 16p, and `nobj_avg = 49` is baked into the checkpoint, so `invnave` is wrong
   for accuracy purposes at 16p. Do not quote these as an accuracy-neutral saving.

### The BN1 constant-width threshold — RESOLVED 2026-08-25

BN1's γ/σ is a *scalar* constant multiplying every dot (`nPELICAN.cpp:202`), so its snapped
literal width binds all `s(s+1)/2` multipliers at once. Measured on this checkpoint
(γ/σ = 0.0398262530696), same weights, one typedef changed:

- **10-bit literal** (653, at the derived `bn_t_gen` F=14) → `mul_6s_11ns_16` ×171, one
  DSP48 each, **and a timing violation** (slack −0.00).
- **8-bit literal** (163, `--bn-frac-bits 12`) → `mul_6s_9ns_14` ×171 in fabric (49 LUT
  each), timing met (est 4.372 ns). **−172 DSP for +1,475 LUT — 8.6 LUT/DSP, the cheapest
  DSP lever measured.** 9 bits is untested.

Two gotchas encoded in those module names:

1. The Bind Op operand width is the literal **plus one zero-extension bit** (signed
   multiplier): 653 → `11ns`, 163 → `9ns`. Do not read the module name as the literal width.
2. Latency did **not** recover: 15 cycles with BN1 on DSP *or* in fabric (the 20p baseline
   runs 13). The 16p latency gap is a separate, open question — only the timing violation
   was BN1's.

Export with `--bn-frac-bits 12` for this checkpoint (F=13 snaps to the same value with a
wasted bit; snap error 3.1e-5, ~4× inside the half-LSB budget). The loader prints the
literal and a measured-verdict line on every export — check it before synthesizing.

The `../../synthesis-archive/stale-reruns/` files record the debugging detour: byte-identical
re-runs that looked like a stale Vitis cache but were honest rebuilds of an unchanged design
(the F=14 cap was a no-op — the derived F already was 14). See that folder's README.

## capacity_sweep/ — hidden width at 20p, pmu-12

| NHIDDEN | csynth DSP | FF | LUT | Date |
|---|---|---|---|---|
| 1 | 1,318 | 49,028 | 175,857 | 2026-07-21 |
| **2** | 1,053 | 51,395 | 199,410 | 2026-07-22 |
| 3 | 1,328 | 55,510 | 236,591 | 2026-07-22 |

csynth estimates (no vsynth pair). DSP is flat in capacity — the dot stage dominates and sets
a floor — so LUT is the real cost of capacity. h=2 is Pareto-optimal (AUC 0.952). See
`../sweep/`.

## clock_sweep/ — 3/4/5 ns, monolith vs split

DSP is 1,049 at every clock in both variants; only LUT/FF/CARRY8 move. The `split` builds are
`nPELICAN_split.cpp`, a per-stage-function copy of the same top function for **resource
attribution only** — its totals are inflated by the function boundaries and are not a
deployable number. `split_tri` adds `-DNPELICAN_SPLIT_TRI`. See `../docs/FUNCTION_SPLIT.md`.

| Build | 3 ns | 4 ns | 5 ns |
|---|---|---|---|
| monolith LUT | 45,509 | 46,176 | 48,962 |
| monolith FF | 33,199 | 21,575 | 21,631 |
| split LUT | 71,949 | 67,270 | 74,219 |
| split_tri LUT (5 ns) | — | — | 59,460 |

Also here: `csynth_split_design_size.rpt` (design-size breakdown, no utilization table) and
the 2026-07-07 `csynth_monolith.rpt` / `csynth_split.rpt` pair.

## Regenerating

```bash
cd nPELICAN-fpga
python model_loader.py --model ../PELICAN-nano/model/<prefix>_best.pt --quant \
    --repo ../PELICAN-nano --out firmware/weights/weights.h      # --out is REQUIRED
vitis_hls -f build_prj.tcl "reset=1 csim=0 synth=1 vsynth=1"     # reset=1 is REQUIRED
```

Both flags are load-bearing:

- The bare `--out` default writes to a **stray top-level `weights/`** that the build does not
  read (`firmware/nPELICAN.h:26` resolves relative to `firmware/`), so the export silently
  does nothing.
- `reset` defaults to **0** in `build_prj.tcl:5`, and headers are not `add_files`'d, so Vitis
  has no dependency edge on `types_generated.h` and will happily re-emit RTL cached from a
  previous run. This has already produced two byte-identical "new" measurements.

Sanity check before believing any report: confirm the loader's printed BN1 binding line, and
confirm the report's internal `Date` is newer than the header you just generated.
