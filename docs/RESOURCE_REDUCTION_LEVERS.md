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

## Current state (6/6/6, 20p; Lever 4 confirmed 2026-07-07, `reports/csynth_monolith.rpt`)

Current operating point: **`--max-input-bits 18`** (Lever 2) + BN2-collapse (Lever 4).

| metric | value | notes |
|--------|-------|-------|
| DSP | **1347** | 44% of one SLR (3072). 1012 dots (253 × 4 mults × 1 DSP, beams-as-inputs — see Lever 5) + ~335 MAC/normalize. |
| FF  | 63,343 | 7% SLR |
| LUT | **332,609 → 229,779** | **53% of one SLR (432000) — now the binding resource.** Per-stage: 51% of it is the 2→2 MAC (Lever 6), 16% dots, 13% BN1. |
| Latency | 14 cycles / 70 ns, II=1 | fully pipelined; slack −0.02 ns (marginal) |

⚠ **CORRECTION 2026-07-10 — the LUT numbers above are csynth ESTIMATES and are
~3.3× pessimistic.** Vivado post-synth on this same baseline
(`reports/vsynth_monolith.rpt`, Vivado 2023.2, xcu250): **69,735 LUT / 25,142 FF
/ 1343 DSP** — i.e. 4.0% of the device's LUTs (16% of one 432k SLR, not 53%) and
10.9% of DSPs. DSP estimates were accurate (1343 vs 1347); LUT/FF were not.
Split-build vsynth (`reports/vsynth_split.rpt`): 77,157 LUT / 43,238 FF /
1575 DSP — the split costs extra (function boundaries block cross-stage optim),
use it for attribution RATIOS only. Consequence: "LUT is the binding resource"
was an estimate artifact; at post-synth reality neither LUT nor DSP is near
binding on xcu250. Relative csynth-to-csynth comparisons between levers remain
valid; absolute headroom claims must use vsynth numbers. (Workspace-root
`vivado_synth.rpt` is the DEEPSET design, 1.196M LUT — don't confuse.)

**Per-stage attribution now exists** (2026-07-07): the split build
(`firmware/nPELICAN_split.cpp`, `build_prj.tcl split=1`, see `FUNCTION_SPLIT.md`)
reports each stage separately — dots 1012 DSP / 45.7k LUT, BN1 0 DSP / 38.0k LUT,
eq2to2 148.6k LUT, aggregations ~21k LUT each (`reports/csynth_split.rpt`,
`resource_log.md`). Levers below cite these numbers. NOTE: `np_bn1` uses ZERO DSP
(6×9-bit const mults → LUT), so the old "~1310 inferred BN1/normalize/MAC" DSP
estimate is obsolete — post-Lever-4 non-dot DSP is ~335, all in the MAC/normalize.

Progress so far (all 6/6/6, 20p):
| report | DSP | FF | LUT | dot multiplier |
|--------|-----|----|----|----------------|
| `6_6_6_20p` (original) | 4677 | 170,899 | 415,830 | `mul_36s_36s_72` ×4 DSP |
| `new6_6_6_20p` | 4670 | 167,277 | 411,487 | `mul_36s_36s_72` ×4 DSP |
| `secondnew6_6_6_20p` (input_t fix) | 2990 | 118,236 | 403,219 | `mul_24s_24s_48` ×2 DSP |
| `symmetry6_6_6report` (Lever 1) | 2990 | 116,488 | 400,543 | `mul_24s_24s_48` ×2 DSP (still 840) |
| `18inputwidthnPELICAN_report` (Lever 2, **operating point**) | **2150** | 122,241 | 332,609 | `mul_18s_18s` ×1 DSP |
| `16inputwidthnPELICAN_report` (Lever 2 @16) | 2150 | 118,521 | 331,231 | `mul_16s_16s` ×1 DSP (DSP-identical to 18) |
| `reports/csynth_monolith.rpt` (Lever 4, **current**) | **1347** | **63,343** | **229,779** | `mac_mulsub_18s_18s` ×1 DSP (1012 total, beams-as-inputs) |

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
- **DSP48 packing threshold = 18 bits.** A DSP48E2 mult is 27×18; both operands ≤18
  → 1 DSP. So `input_t` 18 and 16 both give 1 DSP/mult — identical DSP (2150). Below
  18 there's no further DSP boundary; you only shave peripheral LUT/FF marginally
  (16 vs 18: −0.4% LUT, ~3% FF) while losing more dot4 precision. **18 is the knee.**

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

### Lever 2 — Cap `input_t` width (precision tradeoff)  ✓ DONE; operating point = 18
Each dot product is `input_t × input_t` = `mul_24s_24s` = **2 DSP** at 6/6/6.
Drop `input_t` to ≤18 bits → `mul_18s_18s` = **1 DSP**. **MEASURED** (commit
`f93d819`, `--max-input-bits 18`, `18inputwidthnPELICAN_report`): DSP **2990 → 2150
(−840, exactly as predicted)** AND LUT **400,543 → 332,609 (−17%)** — the narrower
momenta also shrink the dot4 adder trees. At 16 (`16inputwidthnPELICAN_report`): DSP
identical (2150) — both ≤18 fit one DSP48 — so **16 buys nothing over 18 on DSP**
(only −0.4% LUT / ~3% FF) while losing more precision. **18 is the knee; use it.**
Width reduction is real (CSE can't undo it, unlike Lever 1). ⚠ Re-confirm the online
golden gate at 18 (csynth reports don't show it) — log result in `resource_log.md`.

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

#### Sub-12-bit caps (negative F) — MEASURED 2026-07-02, NOT VIABLE
`--max-input-bits` now accepts caps ≤ INPUT_I: INPUT_I stays fixed (range preserved)
and F goes negative (`ap_fixed<W,I>` with I > W is legal; e.g. `<10,12>` = momentum
LSB 2² = 4 GeV). This is hardware-identical to "divide momenta by 2^-F and feed a
W-bit integer" — the binary point is free, so a global ÷8 by itself changes nothing
(see dead-ends list); only the width cut is real. Two interface fixes were required:
`nobj` is now its own `nobj_t = ap_uint<5>` port (an input_t nobj rounds odd counts
to even below 12 bits, corrupting the mask), and note the beam spurions (1,0,0,±1)
are unrepresentable below 12 bits (F<0 → LSB>1) — any real sub-12 design would have
to special-case them.

Local golden-gate sweep (w6a6i6 checkpoint, 200 sample_data events, local g++ oracle;
baseline = uncapped 24-bit input_t, whose 93/200 / max|Δ|=0.375 residual is the known
6/6/6 float-BN bit-faithfulness floor; logit full range ±4):

| input_t | LSB (GeV) | exact | max\|Δ\| | mean\|Δ\| | sign flips vs PyTorch |
|---------|-----------|-------|--------|---------|----------------------|
| 24 (uncapped) | 2⁻¹² | 93/200 | 0.375 | 0.082 | 10 |
| 18 (op point) | 2⁻⁶ | 85/200 | 0.375 | 0.094 | 10 |
| 16 | 2⁻⁴ | 77/200 | 0.5 | 0.11 | 11 |
| 14 | 2⁻² | 41/200 | 0.75 | 0.22 | 12 |
| 12 | 1 | 16/200 | 3.25 | 0.60 | 19 |
| 11 | 2 | 2/200 | 4.5 | 1.29 | 28 |
| 10 | 4 | 1/200 | 4.6 | 1.65 | 32 |

A control experiment (momenta pre-snapped to the coarse grid in the TB data while the
firmware stayed at 18 bits, beams/nobj exact) reproduces the same collapse (10-bit-equiv
4 GeV grid: 34/200 sign flips), proving it is the real-momentum grid itself, not the
beam artifact. The dots-level gate is width-invariant (94/200, 0.375) at every cap, as
it must be. **Conclusion: the knee is 18 and the cliff starts at ~14; ≤12 bits destroys
the dot4 front-end (mean logit error ~0.6–1.7 on a ±4 logit, 10–18% of decisions flip).
Sub-12-bit input widths are a dead end for DSP→fabric savings — do not pursue.**

### Lever 3 — Relax `PIPELINE II=1` / partial roll  (biggest possible cut; breaks an invariant)
Root cause of the magnitude: the whole 22×22 datapath is replicated 484× because
everything is unrolled at II=1. Allowing II>1 (roll the i/j loops, `ARRAY_PARTITION`
cyclic instead of complete) lets HLS **time-share multipliers and adder trees** —
potentially several-fold on **both DSP and LUT**, and the only lever that
substantially cuts the 357K "Expression" LUT.

**Cost/risk:** high. **Breaks the `PIPELINE II=1` invariant in CLAUDE.md** and
lowers throughput (II=4 ≈ ¼ rate). Only pursue if the latency/throughput budget
has room. This is a deliberate user decision, not a silent refactor.

### Lever 4 — Collapse BN2 past the 2→0 aggregation (+ fold BN means)  ✓ DONE & CSYNTH-CONFIRMED
`Tr = BN2(relu)` was computed per `(i,j,h)` = 22·22·NHIDDEN = **968 wide `bn_t_gen`
multiplies**, almost certainly the bulk of the ~1310 non-dot DSP floor (968 BN2 + 210 BN1
+ ~27 normalize ≈ 1310). But `Tr` is consumed ONLY by the linear 2→0 aggregators
(`R_sum`, `R_trace`) and is NOT a quantization point. Because BN2 is affine and the
aggregation linear, the per-channel affine moves PAST the sum (exact identity):
```
R_sum[h]   = Σ_ij BN2_h(Tp_q) = s_h·(Σ_ij Tp_q·mask) + β'_h·nobj²
R_trace[h] = Σ_i  BN2_h(Tp_q) = s_h·(Σ_i Tp_q[i][i]·mask) + β'_h·nobj
```
with `β'_h = β_h − μ_h·s_h` (BN2 mean folded into bias). **968 wide multiplies → NHIDDEN**
(one `s_h·A` per channel × {sum,trace}). Implemented in `nPELICAN.cpp` (the Tr array is
gone; new `accrelu_t`/`accrelurow_t` accumulators sum the raw ReLU output `Tp_q`; loader
emits them, `acc0_t`/`acc0row_t` retired). Same fold applied to BN1 per-element
(`(dots−μ)s+β → dots·s+β'`) to drop the wide per-element mean subtract (LUT).

**Invariant-safe:** this is NOT folding BN into the dense weights — BN1 stays an explicit
elementwise affine before aggregation, and the N-dependent additive term is made explicit as
`β'·count` with `count` (nobj²/nobj) from the runtime `nobj`. Normalize-late preserved (raw
sum → one rescale). **More faithful to PyTorch**, not less (PyTorch sums float `Tr`; the
firmware no longer rounds each `Tr` to `tr_t` first — one rounding at the t0 cast).

**Validation (local clang, Phase-2 firmware is local==Vitis bit-for-bit):** dots-level gate
143/200 exact, max|Δ|=1.1e-5 PASS (was 142/1.1e-5); golden 133/200, max|Δ|=6.26e-4 PASS
(was 133/6.3e-4). **Csynth confirmed 2026-07-07** (`reports/csynth_monolith.rpt`): DSP
2150 → **1347** (−803, as predicted), FF −48%, LUT −31%. Caught one bug
in review: masked off-diagonal `Tp_q` is NOT zero (`T3=jdotp[i]` masked by `[i]` only), so the
raw sums keep `·nobjmask` — the mask is not redundant.

### Lever 5 — Constant beams for deployment (`const_beams=1`)  ✓ IMPLEMENTED (csynth owed)
Revealed by the per-stage report: `np_dots` = **1012 DSP = 253 × 4** — the upper
triangle of all 22 slots *including the beams*. Historical reports showed 840 = 210 × 4
because the beams were compile-time constants and dots against (1,0,0,±1) fold to
`E∓pz` (adds, zero multipliers). Making beams a runtime port (needed only for the
equivariance boost study) silently costs **43 dots × 4 = ~172 DSP (+13%)** plus their
adder-tree LUT. Deployment never boosts the beams, so hardwire them back:

- `-DNPELICAN_CONST_BEAMS` (`build_prj.tcl const_beams=1`, `./build_local.sh
  [split] -DNPELICAN_CONST_BEAMS`): beam rows of `p1` come from constants; the
  `beam_input` port is ignored (HLS may prune it). Implemented in BOTH
  `nPELICAN.cpp` and `nPELICAN_split.cpp` (sync rule).
- **Bit-exact** vs the runtime port driven at |beta|=0 (verified locally
  2026-07-07: all four build combinations byte-identical on the golden gate).
- ⚠ The equivariance harness MUST build without the flag (it boosts the beams).
- **Owed:** remote csynth with `const_beams=1` — expect ≈ −172 DSP in `np_dots`;
  log in `resource_log.md`.

### Lever 6 — Rebalance LUT→DSP in the 2→2 MAC (`mac_dsp=1`)  ✗ MEASURED 2026-07-10: DEAD

Remote csynth (monolith, pmu-12 weights, `reset=1 mac_dsp=1`) is **byte-identical**
to `mac_dsp=0`: 1173 DSP / 61,436 FF / 229,268 LUT / 14 cyc. This is the failure
mode anticipated below: `w1_2to2` are compile-time literals in `weights.h`, so HLS
strength-reduces every MAC multiply into shift-add trees during elaboration — by
scheduling time there are **no `mul` ops left for BIND_OP to bind** (silent no-op;
same reason `config_op mul -impl dsp` won't bite either). And the endgame was
impossible anyway: the 2→2 MAC has 22·22·2·6 = **5808 constant multiplies**; at
1 DSP each that's ~3× the whole SLR's remaining DSP budget (~1900 idle of 3072).
**The LUT→DSP trade direction is dead for any constant-weight stage (this MAC,
BN1, BN2) — don't re-attempt with pragmas.** The flag stays in the tcl/firmware
as documentation; it is a no-op.
LUT is now the binding resource (53% SLR vs DSP 44%), and the split report pins
**51% of all LUT in `np_eq2to2`** (the 2→2 MAC's strength-reduced constant
multiplies + adder trees, 148.6k). ~1700 DSPs sit idle — trade them:

- `-DNPELICAN_MAC_DSP` (`build_prj.tcl mac_dsp=1`): `#pragma HLS BIND_OP
  variable=Tp op=mul impl=dsp` forces the MAC multiplies into DSP48s. In both
  firmware files. Bit-exact by construction (impl choice doesn't change values).
- This is an EXPERIMENT: HLS may pack several 6-bit mults per DSP48 or may
  ignore ops it already strength-reduced — measure, don't assume. Run on the
  split build first so the effect is attributed to `np_eq2to2` alone.
- Combines freely with Lever 5 (independent flags).
- If BIND_OP moves too little (strength-reduced const-mults are no longer "mul"
  ops), the fallback experiment is `config_op mul -impl dsp` in the tcl, or
  rank-1 factoring of the basis terms (`row[i]+col[j]+scalar` regrouping —
  bounded win, Lever-1 caveat applies to the products but not the adder trees).
- **Owed:** remote csynth `split=1 mac_dsp=1` vs plain `split=1`; log the
  `np_eq2to2` LUT/DSP delta in `resource_log.md`.

### Lever 7 — Per-particle block floating point on the momenta  ✓ SWEPT — accuracy win, resource path REJECTED

Origin: the professor's suggestion to "individually quantize parts of the dot product
(p_x, p_y, …)". **Per-COMPONENT is provably zero-sum; per-PARTICLE is worth ~2–3 bits.**
Analysis 2026-08-04, `analysis/blockfp_dots.py`, 3000 `sample_data` valid events,
beams included, against the trained `w6a6i6p12` grid.

> **→ READ 7g FIRST.** The AUC sweep (2026-08-05) has since run. Summary: block-FP is a
> real and free **accuracy** win at equal width (+18% bgRej@0.5 at W=12), but the 8-bit
> DSP-packing endgame in 7d/7e is **dead** — 8-bit mantissas cost 12.4% bgRej, and
> packing is arithmetically 8-bit-or-nothing. The ~4-bit estimate below came from the
> `dot_t` gate metric and was optimistic; the measured figure is 2–3 bits.

#### 7a. Per-component quantization — ✗ DEAD (measure once, never again)

Per-component ranges (237k real particles): E 1946.9 (I=12), px 692.6 (I=11),
py 835.7 (I=11), pz 1874.7 (I=12). So the ceiling is **1 bit on 2 of 4
multipliers** — and in the production config it is **0 bits**, because the trained
`pmu_quant` clips every component at ±512 (`input_t = ap_fixed<12,10>`), giving
I=10 for all four.

Measured on the `dot_t` gate (see 7c for the metric), production clip:

| scheme | W=12 | W=10 | W=8 |
|---|---|---|---|
| uniform (current) | 17.61% | 37.53% | 63.78% |
| per-component I | 17.43% | 37.38% | 63.70% |
| per-component, same total W | 36.36% | 63.89% | 56.13% |

Per-component I buys **0.18 pp** — noise. (In an unclipped I=12 comparison it is
*bit-identical* to uniform; the 0.18 pp here is only because px/py then clip at
±1024 instead of ±512.) Spending the freed bits on extra fractional resolution at
fixed total width is **actively worse** (36% vs 18%), because narrowing px/py's
range to ±256 saturates real momenta.

**Why it cannot work** (same theorem as the global-rescaling dead end, applied per
component): dot error is `Δd ≈ Σ_k |p_k^(j)|·δ_k`. Equalizing each component's
contribution needs `δ_k ∝ 1/max|p_k|`, i.e. `F_k = const − log₂ max|p_k|`. Since
`I_k = log₂ max|p_k|`, the total `W_k = I_k + F_k` is **invariant**. Redistributing
width across components is zero-sum.

#### 7b. Why the dot front-end is expensive (the two facts that reframe it)

1. **The Minkowski dot is catastrophically cancelling.** `max_term/|d_ij|` over i<j:
   median 11.1 (3.5 bits lost), p99 4.6e3 (12.2 bits), p99.9 4.4e4 (15.4 bits),
   max 5.2e6 (**22.3 bits**). 12 bits of momentum are being spent to survive a
   cancellation that eats 3–22.
2. **`dot_t` is far coarser than anyone assumed.** `w6a6i6p12` learns
   `dot_t = ap_fixed<6,10>` → **LSB 16 GeV², clip ±512**. Against the real d_ij
   (median |d| = 6.2): **54.5% of dots quantize to exactly 0**, only **32 of 64**
   levels are ever occupied (72% live in the bottom two cells), 3.0% saturate.

Momentum precision therefore matters *only* for which side of a `dot_t` boundary a
pair lands on; because `Δd ≈ |p|·δp` with |p| up to 512, a global **absolute**
momentum LSB throws dots across boundaries even though the target grid is crude.
That is the mechanism behind the sub-12-bit cliff.

#### 7c. The lever: per-particle po2 exponent + narrow mantissas

`d_ij = 2^(e_i+e_j) · (m_i · g · m_j)`, with `e_i = clamp(⌊log₂ E_i⌋, 0, 10)` and
W-bit signed mantissas `m_i = p_i / 2^{e_i}` at I=2. Every particle then gets full
**relative** precision instead of a shared absolute LSB.

Metric = fraction of d_ij landing in a **different `dot_t` cell** than float-exact
(the dots-level gate already in use):

| mantissa W | uniform (current) | **block-FP / particle** | block-FP / event |
|---|---|---|---|
| 12 | 17.61% | **1.27%** | 12.56% |
| 10 | 37.53% | **4.38%** | 29.50% |
| 9  | 56.99% | **7.61%** | 42.43% |
| 8  | 63.78% | **12.17%** | 55.65% |
| 7  | 56.66% | 18.25% | 59.41% |

**8-bit block-FP mantissas (12.17%) beat the current 12-bit uniform production point
(17.61%)** — strictly better dots on a 4-bit-narrower multiplier. At equal 12 bits
it is a 14× reduction in dot errors. Per-*event* block-FP (one shared exponent)
recovers only a small part of the win: the gain is genuinely per-particle.

Two free simplifications, both measured:
- **Exponent from E alone, not a 4-way max over components** — bit-for-bit the same
  mismatch rate at every W (1.27/4.39/7.63/12.20/18.32%). No 4-way max in hardware,
  just `LZC(E)`. Provably non-overflowing at I=2: `2^e ≤ E < 2^{e+1}` and `E ≥ |p_k|`
  ⟹ `|m_k| < 2`.
- **Exponent clamped to `[0,10]` (4-bit field)** costs nothing (12.17 → 12.20% at
  W=8). Unclamped span is 53 (6 bits) because of a few ~2^-43 padding artifacts.

Storage per particle **drops**: 4×8 + 4 = 36 bits vs 4×12 = 48.

#### 7d. Why this survives the Lever-1 lesson, and the DSP endgame

The normalization is **O(N)** — 22 leading-zero counts + 88 shifts — while the
benefit lands on **O(N²)** = 1012 multipliers. That asymmetry is the whole trade,
and unlike Lever 1 this is a *width* reduction, which CSE cannot undo.

**The DSP endgame — and the trap the pmu8/9/10 runs already fell into.**
⚠ Narrowing operands *by itself* is MEASURED to go the wrong way below the DSP
inference threshold: pmu10 vsynth was −233 DSP but **+25.9k LUT** (~111 LUT per DSP
saved), and pmu9/pmu8 were non-monotonic tool noise (936/998/977 DSP at W=10/9/8).
Below the threshold HLS spills the mults to fabric instead of packing them. **Do not
expect block-FP's narrower mantissas to shrink anything on their own — that path is
already closed by measurement.** Block-FP fixes only the *accuracy* half of why
sub-12-bit failed; it does not by itself fix the resource half.

The resource win has to be taken **explicitly**: at 8-bit mantissas the shared-operand
packing becomes *constructible* (`m_i[k]` is shared across all j) — pack `m_j, m_j'`
into one 27-bit operand at shift 17 and multiply by the 8-bit `m_i` in the 18-bit port,
17 + 8 = 25 ≤ 27, giving **two multiplies per DSP → ~506 dot DSP**. This is arithmetically
unreachable at 12-bit uniform (17 + 12 = 29 > 27). But it must be *hand-written* (pack,
one multiply, slice the two products apart), not left to `BIND_OP` — the Lever 6 lesson
plus the pmu10 spill both say the tool will not do it for you.

So the honest ordering is: block-FP buys the accuracy headroom that would make an
8-bit mantissa *legal*; the DSP saving is a separate, manual piece of work that is only
worth attempting once the AUC sweep says 8 bits holds.

⚠ **UPDATE 2026-08-05 — the sweep says 8 bits does NOT hold (7g).** Block-FP buys 2–3
bits, not 4, so W=8 lands *below* the uniform-12 baseline on background rejection
(−12.4%). And the packing is 8-bit-or-nothing: the shared-operand shift must satisfy
`s ≥ 2W+1` (product width plus a sign guard) with `s + W ≤ 27`, so W=8 → 17+8=25 ✓ but
W=9 → 19+9=28 ✗ and W=10 → 21+10=31 ✗. There is no width that is both accurate enough
and packable. **The 2-mults/DSP endgame in this section is abandoned.**

#### 7e. Risks / owed work

- **The new O(N²) cost is 253 realignment shifters** for `e_i+e_j` (range [0,20]).
  They should be cheap because the output is only 6 bits wide into `dot_t`, but
  **this is the one number that decides whether the lever nets out — csynth it.**
  The resource claims in 7d are analytic, not synthesized.
  ⚠ **SUPERSEDED by 7g:** with the packing endgame dead there is no DSP saving for the
  shifters to be weighed against, so this csynth is no longer on the critical path.
  Run it only if Lever 7 is ever revived as a *resource* play.
- **Resource upside is NOT automatic** (see the 7d warning): on the existing
  sub-threshold evidence, expect narrower mantissas alone to *cost* LUT. Budget for
  the manual DSP-packing work, or treat Lever 7 as an accuracy/robustness result
  rather than a resource result.
- **Against the standing "nothing is resource-bound" finding** (4.0% LUT / 9.5% DSP
  post-vsynth on xcu250), none of this is worth spending training runs on unless
  there is an external budget — a smaller device, an SLR cap, or a latency target.
  Get that budget first; that caveat outranks the whole lever.
- **All numbers above are dot-front-end only.** AUC is not measured; that needs the
  retrain (prototype landed, see below).
- **Breaks bit-exactness with PyTorch** unless `pmu_quant` becomes a block-FP
  fake-quant. **Done on the training side** (2026-08-04, `PELICAN-nano`):
  - `src/layers/blockfp.py` — `BlockFPQuant`, stateless, STE, E-based exponent.
  - `QuantConfig.pmu_block_fp / pmu_exp_min / pmu_exp_max`; model picks it over the
    Brevitas `QuantIdentity` when `--pmu-block-fp` is set with `--pmu-bit-width`.
  - CLI `--pmu-block-fp / --pmu-exp-min / --pmu-exp-max`, wired through
    `train_pelican_nano.py`, `scripts/export_golden.py`, `scripts/check_scales.py`.
  - `tests/test_pmu_blockfp.py` (12 tests; suite 60 passed / 1 skipped).
  - `scripts/sweep_pmu_blockfp.sh` — AUC sweep, `BASELINE=1` also retrains the
    uniform p12 grid for an apples-to-apples comparison. Verified end-to-end on
    `sample_data`; AUC read from the checkpoint's `best_metrics`, not the log.

  Because `BlockFPQuant` is stateless the checkpoint gains no new keys — the
  configuration lives in the run's `args`, which is how `model_loader.py` already
  auto-detects the momentum grid.
- **The loader + firmware side is DONE** (2026-08-08), so a block-FP checkpoint now
  exports and c-sims like any other. See 7i below for the export/vsynth commands and
  the verification evidence.
- **Likely bonus for the equivariance sweep:** block-FP is scale-covariant, so a
  boost is absorbed into the exponent and leaves mantissas nearly unchanged. The
  existing harness can test this directly.
- Masking invariant holds: an all-zero 4-vector gives `e = exp_min`, `m = 0`, output
  exactly 0.

#### 7f. Alternatives measured and rejected

- **Light-cone linear basis (E±p_z)**: *worse* than Cartesian (median rel error
  0.024 vs 0.018 at W=18) and needs 6 mults/pair instead of 4.
- **Cancellation-free curvilinear** `d = 2 p_T^i p_T^j [sinh²(Δy/2) + sin²(Δφ/2)]`:
  on raw dot error it looked as good as block-FP, but **on the `dot_t` gate it is
  worse at every width** (18.51 / 29.05 / 41.37% at W=12/10/8 vs block-FP's
  1.27 / 4.39 / 12.20%). It degrades far more *slowly* than uniform (43% vs 57% at
  W=7), so it is a real effect — just dominated. Given it also costs ~506 trig LUTs
  and a training coordinate change, it is not competitive. Its one unique property
  (manifest invariance under longitudinal boosts, since Δy and Δφ are boost-invariant)
  makes it worth remembering **only** if the equivariance study ever becomes the
  binding constraint rather than resources.

#### 7g. AUC sweep result — THE ANSWER (2026-08-05)

`PELICAN-nano/scripts/sweep_pmu_blockfp.sh`, full `data/toptag`, 16 epochs, seed 42,
`BASELINE=1` (controlled: the uniform-12 row is retrained under identical
seed/epochs/data, *not* the old 35-epoch production number). All figures are the
checkpoint's `best_metrics` (validation).

| grid    | W  | AUC    | acc    | bgRej@0.5 | vs uniform-12 |
|---------|----|--------|--------|-----------|---------------|
| uniform | 12 | 0.9544 | 0.8963 | 39.6      | baseline      |
| blockfp | 12 | 0.9603 | 0.9055 | 46.7      | **+17.9%** bgRej |
| blockfp | 10 | 0.9573 | 0.9051 | 41.7      | +5.3%         |
| blockfp | 9  | 0.9589 | 0.9027 | 39.2      | −1.0%         |
| blockfp | 8  | 0.9527 | 0.8944 | 34.7      | **−12.4%**    |
| blockfp | 7  | 0.9429 | 0.8821 | 26.2      | −33.8%        |

**CONFIRMED — the premise holds.** At *identical* mantissa width, block-FP strictly beats
the uniform grid: +0.0059 AUC and +17.9% background rejection at W=12. The dot's problem
was never total bits, it was that one global scale must cover both a 500 GeV jet core and
a 0.5 GeV constituent. Per-particle exponents make the same 12 bits do more work.

**Worth 2–3 bits, not 4.** W=10 is strictly better than the baseline and W=9 is a wash;
W=8 is where it breaks. The `dot_t` gate metric (7c) was directionally right but
quantitatively optimistic.

**Read bgRej, not AUC.** AUC is non-monotonic (W=9's 0.9589 > W=10's 0.9573), which puts
the AUC noise floor at ≈±0.002 for a 16-epoch single-seed run — that inversion is not
real. bgRej@0.5 is monotone across all five widths and is the better-conditioned metric.

**VERDICT on the DSP endgame (7d/7e): dead.** Packing requires W=8 exactly (see the
`s ≥ 2W+1`, `s+W ≤ 27` update in 7d). W=8 costs 12.4% bgRej, and the pmu10 vsynth
precedent (−233 DSP / **+25.9k LUT**) applies here with 253 realignment shifters added on
top — i.e. strictly worse LUT risk than the experiment that already failed. Do not spend
the manual packing effort.

**REDIRECT — bank it as accuracy, spend it on particle count.** Lever 7 is a *free
accuracy win* at W=10–12 (the mantissa is no wider than today's uniform grid). The
productive use of that surplus is the "fewer particles" decision already listed in
sequence item 7: pair count is quadratic in N, and 253 = 22·23/2 for nobj=20 + 2 beams.

| nobj + beams | pairs | dot DSP vs today |
|---|---|---|
| 20 + 2 | 253 | — |
| 16 + 2 | 171 | −32% |
| 12 + 2 | 105 | −58% |

Quadratic truncation dominates a 2× packing factor, and block-FP has just produced the
accuracy headroom to pay for it. **Next sweep: nobj at block-FP W=10/12**, run down until
AUC/bgRej returns to the uniform-12 baseline.

**Note on export:** the `dot_t` typedef differs between baseline and block-FP
checkpoints, so any csim/vsynth comparison must re-export from the chosen checkpoint,
**not** reuse the existing `weights.h` / `types_generated.h`.

#### 7h. `dot_t` is RANGE-limited, not resolution-limited — CLOSED (2026-08-05)

The learned `input_quant` scale doubled (8.0 → 16.0) under block-FP, i.e. `dot_t` got
*coarser* exactly as the momenta got finer. That looked like the block-FP gain being
discarded at the cast, so two more arms were run at 16 epochs / seed 42 / full toptag:
**B** = `IUNSIGNED=1` at `i6` (the sign bit is provably dead — see below — so this is a
free bit), **C** = plain `i7` (the paid version of the same resolution).

Sorting all **12** runs by CLIP POINT rather than by width shows the actual variable
(clip = 2^(W−1)·scale signed, (2^W−1)·scale unsigned):

| clip | runs | AUC range | bgRej range |
|------|------|-----------|-------------|
| **128**  | 2  | **0.9294 – 0.9332** | **12.9 – 13.1** |
| 256  | 4  | 0.9429 – 0.9592 | 26.2 – 42.0 |
| 512  | 6  | 0.9527 – 0.9603 | 34.7 – 46.7 |

Two independent runs — different arm, different grid, different width — both landed at
clip 128 and both collapsed to nearly the same numbers. That is a threshold, not noise.

Three checks, all agreeing:
- **Doubling resolution at constant range is a wash.** Arm C uniform-12 vs arm A
  uniform-12: identical clip (256), LSB 8→4, AUC 0.9544 → 0.9530.
- **Halving range is catastrophic at any width**, including at 7 bits, which has *more*
  total bits than the arm that worked.
- **The best row in the study has the coarsest LSB.** blockfp-12 at clip 512 / LSB 16 —
  widest range, crudest grid — wins both metrics (0.9603 / 46.7).

Mechanism: dots reach 1.1×10⁴ with p99.9 = 1279, so clipping at 128 destroys the
hard-pair tail that carries jet mass. The 8→16 scale move was the optimizer correctly
*buying range*, not a bottleneck. **Both dot_t levers are dead as accuracy plays.**

**7h-i. `--input-unsigned` — implemented, measured, NOT adopted.** The premise is sound
and verified: on `tb_data/10k_pmu_test.dat`, of 2.28M masked i<j dots only 0.11% are
negative, all particle–particle, **max |d| = 2⁻⁶ — 512× below one `dot_t` LSB** (float32
cancellation noise on near-collinear pairs; every one rounds to 0 anyway). So the sign
bit really does encode nothing, and `ap_ufixed<6,9>` really is the same range at half the
LSB. It still lost, because the scale is **learned**: freeing a bit does not hand you
resolution at constant range, it hands the optimizer a different range/resolution
tradeoff, and on a Pareto tail it took resolution. Arm B's uniform row is the clearest
case — clip 256 → 128, AUC 0.9544 → 0.9294. Flag kept (default off) for reproducing this.

**7h-ii. The fix that IS worth taking: `--input-clip-min`.** 2 of 12 runs fell into the
clip-128 basin — a ~17% failure rate on a free learned parameter over a heavy tail, which
also means single-seed comparisons elsewhere in this study are shakier than they look.
`QuantConfig.input_clip_min` floors the saturation point (Brevitas `scaling_min_val`,
derived as clip_min/threshold and rounded UP to a power of two so the po2 firmware
contract holds). Costs nothing in hardware — `dot_t` is a RESULT width, not a multiplier
operand, so it never touches the 1012 dot DSPs. `sweep_pmu_blockfp.sh` now defaults
`CLIP_MIN=512`.

⚠ **It is NOT training-only.** Brevitas stores the raw runtime stat in
`scaling_impl.value` and applies the clamp on every forward, so every tool that REBUILDS
the model (`model_loader.py`, `check_scales.py`, `export_golden.py`) must replay it or it
reports a scale the model never used. Same silent failure mode as the signedness bug in
`8e1088d`. Both are now read from the checkpoint's own args and asserted.

#### 7i. Export + firmware path for block-FP checkpoints — LANDED (2026-08-08)

`--pmu-block-fp` checkpoints used to be refused by `model_loader.py` (a deliberate guard:
one uniform `input_t` cannot describe a per-particle grid). Both halves are now built.

**Loader** (`model_loader.py`). `BlockFPQuant` is stateless, so the momentum grid leaves
NO trace in the state_dict and `_read_act_quant()` cannot see it — the checkpoint's `args`
are the only record. They are replayed into the rebuilt model and then **asserted** against
it (`pmu_quant` must actually be a `BlockFPQuant` with the recorded W and exponent clamp),
the same discipline `input_unsigned` and `input_clip_min` needed. `types_generated.h` gains:

| typedef | for W=7, exp [0,10] | role |
|---|---|---|
| `mant_t` | `ap_fixed<7,2>` | mantissa — **the dot multiplier operand** |
| `bexp_t` / `bshift_t` | `ap_uint<4>` / `ap_uint<5>` | per-particle exponent; `e1+e2` |
| `mraw_t` | `ap_fixed<30,12>` | `p >> e` held EXACTLY, so the encode rounds only once |
| `mdot_t` | `ap_fixed<16,6>` | EXACT mantissa dot (`<2W,4>` products, +2 bits for the 4-term sum) |
| `dotalign_t` | `ap_fixed<36,26>` | EXACT `mdot << (e1+e2)`; realign never saturates |
| `input_t` | `ap_fixed<25,12>` | raw-momentum PORT into the encoder — **not** a multiplier operand |

**Firmware** (`firmware/np_blockfp.h`, guarded by `NPELICAN_BLOCK_FP`, called from
`nPELICAN.cpp`, `nPELICAN_split.cpp` and `np_dots_only.cpp`). Encode is a priority
cascade on |E| (= one LZC) plus a shift; the dot is
`d = 2^(e_i+e_j)·(m_i·g·m_j)` with the mantissa dot and the realign both exact, so the
front end rounds **once**, at the `dot_t` cast, exactly as PyTorch's `input_quant` does.
Rounding before the shift would quantize on a grid `2^(e_i+e_j)` too fine — that is the
one way to get this wrong. Uniform checkpoints never define the macro and are unaffected.

**`input_t` needs 8 guard bits, not the analytic 3.** The port rounds before the encoder
sees the momenta, and two boundary effects survive the analytic minimum: an energy just
under a power of two can round UP across an exponent boundary and land the particle on a
different mantissa grid entirely, and a momentum near a `mant_t` half-grid point can be
pushed across by the first rounding. Measured on a W=7 checkpoint: at 3 guard bits the
front end contributed 7 mismatching events / 500; at 8 it contributes **0**. Free — the
extra width is wires and a wider shifter, never a multiplier bit
(`NPELICAN_BFP_GUARD_BITS` in `model_loader.py`).

**Verification** (local g++ c-sim, `smoke_bfp7` = W=7 block-FP on `sample_data`, 8 epochs,
plus a `smoke_unif12` control trained on the identical recipe):

| checkpoint | golden gate | dots-injected (bypasses dot4) | front-end contribution |
|---|---|---|---|
| uniform p12 control | 109 / 500 mismatch | 109 | 0 |
| block-FP W=7 | 87 / 500 mismatch | 87 | **0** |

The block-FP dot front end is **bit-exact vs PyTorch on all 500 events**. The residual 87
is a PRE-EXISTING downstream gap on these short 6-bit smoke checkpoints — it shows up
identically with the dots injected, and the uniform control is worse (109), so it is not
block-FP's. ⚠ Chase it before reading any block-FP *accuracy* number off firmware.
Also verified: split build byte-identical to the monolith; `np_dots_only` gate
484/484 on-grid, 0 asymmetric, with and without `const_beams`; and re-exporting three
uniform checkpoints (`w6a6i6p12`, `w6a6i6p14`, `w24a24i14`) through the new loader is
byte-identical except one added `#include "ap_int.h"` — zero regression.

**Commands** (Vitis/Vivado are remote-only; steps 1–2 run anywhere):

```bash
# 1. export from the block-FP checkpoint. dot_t DIFFERS from the uniform baseline,
#    so never reuse an existing weights.h / types_generated.h for a comparison.
python model_loader.py --model ../PELICAN-nano/model/fpga_model_qat_w6a6i6bfp7_best.pt \
    --quant --repo ../PELICAN-nano --out firmware/weights/weights.h
# 2. local bit-exactness gate (no Vitis needed)
python ../PELICAN-nano/scripts/export_golden.py \
    --checkpoint ../PELICAN-nano/model/fpga_model_qat_w6a6i6bfp7_best.pt \
    --testfile ../PELICAN-nano/data/toptag/test.h5 --num 500
./build_local.sh -DRUN_GOLDEN_GATE && ./tb_local
# 3. dots-only front end: csim + csynth, then the netlist-level number. csynth's DSP
#    estimate is unreliable below the DSP48 inference threshold, and a 7-bit mantissa
#    is far below it — vsynth is the trustworthy figure here.
vitis_hls -f build_dots_only.tcl
vivado -mode batch -source vivado_synth_dots.tcl -tclargs _bfp7   # -> vivado_synth_dots_bfp7.rpt
# 4. whole model
vitis_hls -f build_prj.tcl reset=1 csim=1 synth=1 cosim=0 validation=0 export=0 vsynth=0
vivado -mode batch -source vivado_synth.tcl
```

**Read the result against 7g's prediction, which is that this LOSES.** The pmu10
precedent was −233 DSP / **+25.9k LUT**, and block-FP adds 22 encoders and 253 realign
shifters on top. The encoder cost IS counted in `np_dots_only` (it is part of the front
end under block-FP). W=7 is also the worst accuracy row in the sweep (−33.8% bgRej), so
treat this as a resource data point, not an operating point.

## Not reducible / dead ends (don't re-investigate)
- **Per-COMPONENT (E/px/py/pz) bit-width tuning** — zero-sum by the width-invariance
  theorem in Lever 7a; measured bit-identical to uniform. Split per-PARTICLE instead.
- **Widening or unsigning `dot_t` for accuracy** — 12 runs across 3 arms say `dot_t` is
  RANGE-limited, not resolution-limited (Lever 7h). Extra resolution at fixed range is a
  wash; losing range is catastrophic. Floor the clip (`--input-clip-min 512`) and move
  on. Note `dot_t` is a RESULT width, so it never moved DSP in either direction anyway.
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
2. ~~Lever 2~~ done (`f93d819`) — **operating point `--max-input-bits 18`**: DSP
   2990→2150 (−840), LUT −17%. 16 gives no extra DSP (packing threshold). **Still
   owed: confirm the online golden gate at 18 and log it in `resource_log.md`.**
3. ~~Lever 4~~ done & csynth-confirmed (2026-07-07): DSP 2150→1347, FF −48%, LUT −31%.
4. ~~Lever 6~~ MEASURED DEAD 2026-07-10 (no-op; see section — constant mults have
   no `mul` ops to bind, and 5808 mults ≫ DSP budget regardless). Lever 5
   (`const_beams=1`) likely already absorbed by the pmu-12 retrain (−174 DSP ≈ the
   172 beam-port cost); csynth it only if the DSP number matters.
5. **Now — LUT levers that survive the Lever-6 lesson** (adder trees, not mults,
   are the cost; only fewer TERMS or narrower OPERANDS shrink them):
   a. **`w1_2to2` sparsity retrain** (PELICAN-nano, same pattern as the pmu sweep):
      each zeroed entry deletes one 484-position mult-forest + adder-tree slice,
      ~1/12 of the 148.6k eq2to2 LUT upper-bound per entry; currently 0/12 zero.
   b. ~~Vivado reality check~~ DONE for baseline (2026-07-09) AND pmu-12
      (2026-07-10, `reports/vsynth_monolith_pmu12.rpt`): pmu-12 monolith =
      **69,112 LUT / 24,815 FF / 1169 DSP** — the csynth −174 DSP delta is
      netlist-real, LUT/FF flat, calibration ratios identical (LUT 0.30,
      FF 0.40, DSP exact). That run also had `mac_dsp=1`, netlist-confirming
      the Lever 6 no-op. At 4.0% LUT / 9.5% DSP, nothing is resource-bound;
      levers below only matter against an external (SLR/latency/device)
      budget — get that budget before spending training runs.
   c. **Rank-1/structural factoring probe**: 4 of 6 basis channels are rank-≤1
      (jdotp[j], jdotp[i], jmass, δ-terms) so only ~1104 of the 5808 products are
      distinct — BUT fully-unrolled CSE may already share them (Lever-1 lesson)
      and δ-zeros likely constant-fold. Cheap to test on the split build; expect
      the adder trees (the real cost) to survive factoring.
6. **Lever 7 (block-FP momenta)** — ✓ SWEPT 2026-08-05, see **7g**. Outcome: a free
   **accuracy** win (+17.9% bgRej@0.5 at equal 12-bit width), but the 8-bit DSP-packing
   endgame is **dead** (W=8 costs 12.4% bgRej; packing is 8-bit-or-nothing). It is no
   longer a lever that cuts the dot front-end. Remaining work, in order:
   a. ~~Run the AUC sweep~~ — done, 7g.
   b. ~~csynth the 253 realignment shifters~~ — off the critical path now that there is
      no DSP saving to weigh them against.
   c. ~~Sweep block-FP W=10 × `dot_t` 7/8 bits~~ — done, **7h**. `dot_t` is
      range-limited, not resolution-limited; both the unsigned and the wider-`dot_t`
      levers are dead. Use `--input-clip-min 512` from now on (7h-ii) — 2 of 12 runs
      fell into a clip-128 basin and collapsed.
   d. Loader + `dot4` (m, e) firmware work — only if block-FP is adopted for accuracy.
7. **The two big decisions, unchanged:** Lever 3 (relax II=1 — time-multiplexing
   genuinely SHARES the adder trees, the only structural LUT fix) or fewer
   particles ((N+2)² scaling on ~90% of LUT, ~97% of DSP, per the split report).
   **Lever 7 has now made the second one cheaper to take:** block-FP at W=10–12 buys
   +5–18% bgRej at no width cost, which is accuracy that can be spent on truncating N.
   Run the nobj sweep on top of a block-FP checkpoint, not the uniform one (7g).
