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

### Lever 6 — Rebalance LUT→DSP in the 2→2 MAC (`mac_dsp=1`)  ✓ IMPLEMENTED (csynth owed)
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
2. ~~Lever 2~~ done (`f93d819`) — **operating point `--max-input-bits 18`**: DSP
   2990→2150 (−840), LUT −17%. 16 gives no extra DSP (packing threshold). **Still
   owed: confirm the online golden gate at 18 and log it in `resource_log.md`.**
3. ~~Lever 4~~ done & csynth-confirmed (2026-07-07): DSP 2150→1347, FF −48%, LUT −31%.
4. **Now:** remote csynth for Lever 5 (`const_beams=1`, expect −172 DSP, free) and
   the Lever 6 experiment (`split=1 mac_dsp=1`, LUT→DSP trade in `np_eq2to2`).
   Both implemented and locally gated 2026-07-07.
5. **If more LUT is needed after 5+6:** rank-1 factoring of the basis adder trees
   (Lever 6 fallback), training-side sparsity on `w1_2to2` (each zeroed entry
   deletes a 484-element term forest; currently 0/12 are zero), then the two big
   decisions — Lever 3 (relax II=1, throughput cost) or fewer particles ((N+2)²
   scaling on ~90% of LUT and ~97% of DSP, per the split report).
