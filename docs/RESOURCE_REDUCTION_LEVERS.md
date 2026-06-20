# Firmware Resource-Reduction Levers

Forward-looking plan for cutting FPGA resource usage of `firmware/nPELICAN.cpp`.
Companion to `resource_log.md` (per-phase synth/csim results) and the
"QAT scale → type contract" / "Firmware invariants" sections of the workspace
`CLAUDE.md`. Written 2026-06-20; update the numbers as new reports land.

## Goal & context

The model is exported QAT → fixed-point and synthesized in Vitis HLS, fully
unrolled at `PIPELINE II=1`. The user is driving resource usage **down**,
primarily by lowering the three QAT bit-width flags (input / weight / activation).
The discovery that motivated this doc: **lowering the flags barely moved DSP**,
because the dominant cost was not flag-controlled. Investigation pinned the
bottlenecks and produced two committed loader fixes plus three remaining levers.

### Authoritative measurement
- **C-sim is remote (online Vitis).** Local clang + open-source `ap_types` is a
  smoke check only (see `resource_log.md`). The user runs csim/csynth online and
  drops the report `.txt` files in the **workspace root** (`Rankin Research/`).
- Synth reports present at time of writing (workspace root):
  `6_6_6_20p.txt` (original), `new6_6_6_20p.txt` (after bias/norm fix),
  `secondnew6_6_6_20p.txt` (after input_t fix), plus `6_6_8_20p.txt`,
  `6_6_12_14p.txt`, `6_6_12_12p.txt`. Naming = `[new prefix]<inbits>_<wbits>_<actbits>_<Nparticles>p.txt`.
- Bit-exactness is a **tolerance gate, not zero-tol** (float BatchNorm is an
  invariant) — see `resource_log.md` "Phase 2 bit-exactness — interpretation".
  Any lever claiming "bit-exact" means it produces mathematically identical
  values, so it cannot worsen the existing gate.

### How to read a report (commands that work)
```bash
# Totals (DSP FF LUT line):
sed -n '62p' <report>.txt
# Top-level util breakdown (Expression vs Instance vs Register):
sed -n '48,75p' <report>.txt
# Multiplier instances by module + DSP each (the dot products live here):
sed -n '76,1100p' <report>.txt | grep -E "\|mul_" \
  | awk -F'|' '{print $3, "DSP="$5}' | sort | uniq -c | sort -rn
```

## Current state (6/6/6, 20 particles → 22 with spurions; `secondnew6_6_6_20p.txt`)

| metric | value | notes |
|--------|-------|-------|
| DSP | **2990** | 97% of one SLR (3072). 1680 = dot products (840 × `mul_24s_24s` × 2 DSP); ~1310 = inferred BN1/normalize/MAC multiplies. |
| FF  | 118,236 | 13% SLR |
| LUT | 403,219 | **93% of one SLR (432000).** 357,515 (89%) is "Expression" = adder trees from the full unroll. Barely moved across fixes → now co-binding. |
| Latency | 18 cycles / 90 ns, II=1 | fully pipelined |

Progress so far (all 6/6/6, 20p):
| report | DSP | FF | LUT | dot multiplier |
|--------|-----|----|----|----------------|
| `6_6_6_20p` (original) | 4677 | 170,899 | 415,830 | `mul_36s_36s_72` ×4 DSP |
| `new6_6_6_20p` | 4670 | 167,277 | 411,487 | `mul_36s_36s_72` ×4 DSP |
| `secondnew6_6_6_20p` | 2990 | 118,236 | 403,219 | `mul_24s_24s_48` ×2 DSP |

Key empirical facts established:
- **Particle count dominates and scales as (N+2)².** From the reports:
  12p→1794, 14p→2383, 20p→4677 DSP; ratios track `(N+2)²` (12p/20p=0.38 vs (14/22)²=0.41).
- **Bit-width flags barely touch DSP**: 6/6/6 → 6/6/**8** at 20p moved DSP 4677→4694 (noise).
  The DSPs live in the dot products, which the flags don't size.

## Already done (committed to `nPELICAN-fpga` main)

1. **`43a1b4c` — bias_t_gen / norm_t derive from learned scales** (was hardcoded
   `F=24` / `<40,1>`). Now `BIAS_F = max(relu_F, out_F)+1`,
   `NORM_F = max(acc2_I+t2_F, acc0_I+t0_F)+1`. They shrink with the flags; also
   fixed a latent 2-bit under-size in norm_t. *Marginal resource effect — these
   types were never the bottleneck (confirmed by `new6_6_6_20p` ≈ `6_6_6_20p`).*
2. **`06eb3b5` — input_t generated from the dot grid.** `input_t` (raw-momentum /
   `dot4` operand) was hand-pinned at `ap_fixed<36,12>`, so the 36×36 dot
   multipliers never shrank with the flags. Now generated in `types_generated.h`:
   `I = _int_bits(|p|max)` (physics range, from `valid.h5` `Pmu`, fallback 2048 GeV),
   `F = ceil(log2|p|max) + dot_F + 3` (dot4 error `4·|p|max·2^-F < ½ dot_t LSB`).
   `nPELICAN.h` keeps the hand typedef as a `#ifndef NPELICAN_INPUT_T_GENERATED`
   fallback for the float path. **This is what cut DSP 4677→2990** (input_t
   `<36,12>`→`<24,12>`, dot mult 4→2 DSP). Verified bit-identical to the old
   value on the 24-bit checkpoint.

## Remaining levers (priority order)

### Lever 1 — Exploit dot symmetry  ★ next, safe, bit-exact
`dot4(p_i, p_j) == dot4(p_j, p_i)`, so `dots[i][j] == dots[j][i]`. `nobjmask`,
BN1 (mean/scale/beta scalars) are elementwise, so **`batch1[i][j]` is symmetric too.**
The dot loop (`nPELICAN.cpp:106-112`, see the existing TODO at line 105) and the
BN1 loop (`:142-148`) both compute the full 22×22.

**Change:** compute the upper triangle (i ≤ j, 253 of 484, incl. diagonal once)
and mirror `dots[j*N+i] = dots[i*N+j]` (and likewise batch1).

**Expected:** dot multipliers 1680 → ~880 DSP (~800 saved); BN1 multiplies and
dot4/BN1 adder LUTs roughly halve. Estimate **DSP ~2990 → ~2050, LUT down
meaningfully** → off the SLR ceiling. No II change, no accuracy change.

**Risk:** low. Values identical by symmetry; the tolerance gate cannot regress.
Watch: keep `PIPELINE II=1` (synthesis report), keep masking/zero exactness.
The 2→2/2→0 paths are NOT touched.

### Lever 2 — Rescale input momenta  (high impact on dot DSP; needs retraining)
`INPUT_I=12` is fixed by the physical |p| range (~2000 GeV) → `input_t` 24-bit →
2 DSP/multiply. Pre-scaling momenta to O(1) (e.g. ÷~1024, a unit change PELICAN
tolerates) drops `INPUT_I` to ~2–3 → `input_t` ≤18 bits → **1 DSP/multiply**.
Combined with Lever 1: dots → ~440 DSP (from 1680).

**Cost/risk:** medium. Requires retraining on scaled inputs (or a scale at the
input port + matching the dot grid) in `PELICAN-nano`, then re-export and a fresh
bit-exactness check. Touches the training repo, not just firmware. Confirm the
input_quant scale still lands the dots on a representable grid.

### Lever 3 — Relax `PIPELINE II=1` / partial roll  (biggest possible cut; breaks an invariant)
Root cause of the magnitude: the whole 22×22 datapath is replicated 484× because
everything is unrolled at II=1. Allowing II>1 (roll the i/j loops, `ARRAY_PARTITION`
cyclic instead of complete) lets HLS **time-share multipliers and adder trees** —
potentially several-fold on **both DSP and LUT**, and the only lever that
substantially cuts the 357K "Expression" LUT.

**Cost/risk:** high. **Breaks the `PIPELINE II=1` invariant in CLAUDE.md** and
lowers throughput (II=4 ≈ ¼ rate). Only pursue if the latency/throughput budget
has room. This is a deliberate user decision, not a silent refactor.

## Not reducible / dead ends (don't re-investigate)
- **2→2 dense MAC is not symmetric** — `T[i][j]` carries `jdotp[i]` vs `jdotp[j]`
  (channels 2/3) which swap under i↔j, so it can't be halved like the dots.
- **bias_t_gen / norm_t / accumulator widths** — already minimized and flag-tracking
  (commit `43a1b4c`). Widening BN-constant precision changed nothing (resource_log).
- **Masking** is already `ap_uint<1>` selects (0 DSP).
- **psloglut encoder is commented out** (`nPELICAN.cpp:124-132`) — not in the
  resource picture; ignore unless re-enabled.
- **Flags below the DSP-packing threshold** save LUT but not DSP (a multiply that
  already fits one DSP48 costs 1 DSP at 6 or 16 bits).

## Suggested sequence
1. Lever 1 now → re-export (`model_loader.py`), user runs online csim + csynth,
   log to `resource_log.md`. Should clear the SLR ceiling.
2. Then choose Lever 2 (if retraining is acceptable) and/or Lever 3 (if throughput
   can be spent) for the next major cut.
