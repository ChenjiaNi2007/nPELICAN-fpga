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
| `secondnew6_6_6_20p` (input_t fix) | 2990 | 118,236 | 403,219 | `mul_24s_24s_48` ×2 DSP |
| `symmetry6_6_6report` (Lever 1) | 2990 | 116,488 | 400,543 | `mul_24s_24s_48` ×2 DSP (still 840) |

Key empirical facts established:
- **Particle count dominates and scales as (N+2)².** From the reports:
  12p→1794, 14p→2383, 20p→4677 DSP; ratios track `(N+2)²` (12p/20p=0.38 vs (14/22)²=0.41).
- **Bit-width flags barely touch DSP**: 6/6/6 → 6/6/**8** at 20p moved DSP 4677→4694 (noise).
  The DSPs live in the dot products, which the flags don't size.
- **HLS CSE already exploits dot symmetry.** `840 = 210·4` (210 = upper triangle of
  20 particles): the dot multiplier count is the *symmetric* count before any manual
  change. Lever 1 (manual upper-triangle) left DSP at 2990 — see Lever 1 section.
- **The dot multiplier is `input_t × input_t`.** Cutting its WIDTH (Lever 2) is the
  only thing left that reduces dot DSP short of relaxing II=1 (Lever 3).

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

## Lever 1 — Exploit dot symmetry  ✗ DONE, INEFFECTIVE (HLS already did it)
Committed `a7b5b06` (dot loop `nPELICAN.cpp:~106`, BN1 loop `:~142` rewritten to
compute the upper triangle j≥i and mirror). **Result: DSP unchanged (2990); only
~1.7k FF / ~2.7k LUT saved** (`symmetry6_6_6report.txt` vs `secondnew6_6_6_20p.txt`).

**Why it didn't help:** the report shows `840 × mul_24s_24s` both before AND after.
`840 = 210 × 4` where 210 = 20·21/2 = the upper triangle of the 20 real particles
(spurion dots fold to constants). **Vitis's common-subexpression elimination already
merged `dot4(p_i,p_j)` and `dot4(p_j,p_i)`** (commutative products) before the change.
Lesson for a fully-unrolled design: **HLS already removes algebraic redundancy like
symmetric/commutative recomputation — manually choosing *what* to compute won't cut
DSP. Only changing operand *width* (Lever 2) or *time-sharing* hardware (Lever 3)
moves DSP.** The symmetry edit is a marginal-but-harmless FF/LUT win; keeping it.

## Remaining levers (priority order)

### Lever 2 — Cap `input_t` width (precision tradeoff)  ★ next; the real DSP lever
Each dot product is `input_t × input_t` = `mul_24s_24s` = **2 DSP** at 6/6/6.
Drop `input_t` to ≤18 bits → `mul_18s_18s` = **1 DSP** → **840 × 2 → 840 × 1 ≈ 840
DSP saved (2990 → ~2150)**, and it narrows the BN1/normalize multiplies in the 1310
"inferred" DSP too. Width reduction is real (CSE can't undo it, unlike Lever 1).

**RESCALING DOES NOT WORK (proven, was a wrong earlier framing).** Scaling momenta
by 1/S drops `INPUT_I` by log2 S but Brevitas relearns an `input_quant` scale ~S²
smaller, so `dot_F` (hence `INPUT_F`) *grows* by the same amount: `INPUT_W` is
invariant. The multiplier width is set by the dot's **dynamic range / relative
precision**, which is scale-invariant — you can't shrink a multiplier by changing
units. See the "dead ends" list.

**What actually works:** the un-capped width exists only to make the fixed-point
dot round to the **same `dot_t` (6-bit) grid point as PyTorch** near rounding
boundaries (`INPUT_F = ceil(log2|p|max) + dot_F + 3`). The dots are *only* dot_t-bit;
the extra width buys bit-exact agreement, not physics. So **cap the width and accept
that some events round one dot_t LSB off from PyTorch** — the same currency the
tolerance gate already spends (the dot4 front-end "caveat D4" is already the dominant
residual). Implemented as loader flag **`--max-input-bits N`** (commit adds it):
shaves `INPUT_F` only (`INPUT_I`/range preserved so momenta never saturate); no-op if
N ≥ the bit-exact width; errors if N ≤ `INPUT_I`.

**Cost/risk:** medium. No retrain needed — works on the current model. **Sweep N
down (e.g. 24→20→18→16) and re-run the online golden gate each time**; pick the
narrowest width with acceptable accuracy. It is a deliberate accuracy tradeoff, not
a free derivation; re-validate the gate whenever the model or dataset changes.

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
- **Algebraic/structural redundancy (symmetric or commutative recompute)** — HLS
  CSE already removes it in the fully-unrolled design (proven by Lever 1). Don't
  chase it; it won't move DSP.
- **Global rescaling of input momenta** — width-invariant (INPUT_I saved = INPUT_F
  paid back; dynamic range / relative dot precision is scale-invariant). Does NOT
  shrink the dot multiplier. Use the `--max-input-bits` cap (Lever 2) instead.

## Suggested sequence
1. ~~Lever 1~~ done (`a7b5b06`), DSP-neutral — see the Lever 1 section.
2. **Lever 2 next** — no retrain: re-export with `--max-input-bits N`, sweep N down
   (24→20→18→16), run the online golden gate + csynth each step, log to
   `resource_log.md`. Target ≤18 → 1 DSP/mul → ~840 DSP saved (2990 → ~2150). Stop
   at the narrowest N with acceptable gate accuracy.
3. Then Lever 3 (if throughput can be spent) for the structural LUT+DSP cut.
