# Firmware Resource Log

Records synthesis and C-sim results at each phase of the QAT restructure
(see `FIRMWARE_QAT_PLAN.md` for phase definitions).

| phase | date | checkpoint | weights.h commit | LUT | FF | DSP | latency (cycles) | II | timing met | csim vs golden (exact/total, max\|Δ\|) | notes |
|-------|------|------------|-----------------|-----|----|-----|------------------|----|------------|----------------------------------------|-------|
| Phase 0 baseline | 2026-06-12 | fpga_model_qat_best.pt (8ep, 24-bit po2) | 3c688bf | n/a | n/a | n/a | n/a | n/a | n/a | **Vitis csim: 0/200 exact, max\|Δ\|=3.6066896** (remote output archived at tb_data/.ipynb_checkpoints/golden_fw_results-checkpoint.log) | Pre-restructure uniform-type firmware. Local clang + open-source ap_types build exhibits UB (-O0 vs -O2 outputs differ) → local builds are smoke checks only; authoritative csim is remote Vitis. Homebrew gcc-13 too old for current SDK; `brew upgrade gcc` may revive a trustworthy local loop (local differs from Vitis on 200/200 events, max gap 5.64). CSYNTH DID NOT COMPLETE on the remote box — known pre-existing issue with the high particle count (also failed for the previous model); user deprioritized. II/resource regression tracking deferred until synth is viable. |
| Phase 2 retype | 2026-06-13 | fpga_model_qat_best.pt | 1ed473a | TBD | TBD | TBD | TBD | TBD (remote) | TBD | **csim GATE PASS** — dots-level (network) 142/200 zero-tol exact, max\|Δ\|=1.1e-5 (tol 1e-4); golden (incl dot4) 133/200 exact, max\|Δ\|=6.3e-4 (tol 1e-3). Local==Vitis confirmed bit-for-bit (retyped firmware is toolchain-stable; local is now a trustworthy oracle). | Per-stage QAT types wired. Bugs fixed: Tr saturation (t0_t→tr_t), BN eps omitted (weight/sqrt(var+eps)). Tolerance gate (not zero-tol 200/200): float BatchNorm leaves float intermediate segments that fixed-point can't match bit-for-bit at every quantizer boundary (PyTorch's own f32-vs-f64 logits differ ~3.6e-6). dot4 front-end caveat (D4): PyTorch d_ij in lossy float32 → ≤6.3e-4 golden-path residual; dots-level gate isolates+proves the network. csynth still deferred (pre-existing high-particle-count issue). |

## Resource-reduction sweep (6/6/6 QAT, 20 particles → 22 with spurions)

Synth (csynth) results for the resource-reduction effort; see
`RESOURCE_REDUCTION_LEVERS.md` for the levers. Reports archived as `.txt` at the
workspace root. Device SLR limits: DSP 3072, LUT 432000, FF 864000.

| step | commit | report | DSP | FF | LUT | dot multiplier | notes |
|------|--------|--------|-----|-----|-----|----------------|-------|
| input_t hand `<36,12>` (pre-fixes) | `06eb3b5`~ | `6_6_6_20p` | 4677 | 170,899 | 415,830 | `mul_36s_36s_72` ×4 DSP | original; 97%+ over DSP SLR |
| bias/norm derive | `43a1b4c` | `new6_6_6_20p` | 4670 | 167,277 | 411,487 | `mul_36s_36s_72` ×4 DSP | marginal (not the bottleneck) |
| input_t generated `<24,12>` | `06eb3b5` | `secondnew6_6_6_20p` | 2990 | 118,236 | 403,219 | `mul_24s_24s` ×2 DSP | **−36% DSP**; dot mult 4→2 DSP |
| Lever 1 dot symmetry | `a7b5b06` | `symmetry6_6_6report` | 2990 | 116,488 | 400,543 | `mul_24s_24s` ×2 DSP | **DSP-neutral** (HLS CSE already symmetric); ~1.7k FF/2.7k LUT only |
| Lever 2 `--max-input-bits 18` | `f93d819` | `18inputwidthnPELICAN_report` | **2150** | 122,241 | **332,609** | `mul_18s_18s` ×1 DSP | **−840 DSP & −17% LUT** vs symmetry; recommended operating point |
| Lever 2 `--max-input-bits 16` | `f93d819` | `16inputwidthnPELICAN_report` | 2150 | 118,521 | 331,231 | `mul_16s_16s` ×1 DSP | DSP identical to 18 (both ≤18 → 1 DSP48); only ~0.4% LUT / ~3% FF more → not worth the extra precision loss |
| Lever 4 BN2-collapse + BN mean-fold | ≤`d658d10` | `reports/csynth_monolith.rpt` (2026-07-07) | **1347** | **63,343** | **229,779** | `mul_18s_18s`/`mac_mulsub_18s_18s` ×1 DSP | BN2 968 wide mults → NHIDDEN (push affine past 2→0 sum); BN1/BN2 mean folded into bias. **Local gate PASS**: dots-level 143/200 exact max\|Δ\|=1.1e-5 (was 142); golden 133/200 max\|Δ\|=6.26e-4 (was 133/6.3e-4). **Csynth confirmed: −803 DSP / −48% FF / −31% LUT vs Lever 2 @18.** Dot front-end floor is 1012 DSP (253 upper-triangle dots × 4 mults, 1 DSP each — see split report); latency 14 cyc, II=1, slack −0.02 ns. |
| Lever 5 const beams (`const_beams=1`) | TBD | TBD (remote csynth owed) | TBD (expect ≈1175) | TBD | TBD | unchanged | Beams hardwired to fixed spurions → 43 beam dots constant-fold to E∓pz adds (expect −172 DSP in np_dots). **Local gate 2026-07-07: byte-identical** to runtime-beam build at \|β\|=0 (all 4 mono/split × flag combos). NOT for the equivariance harness. |
| Lever 6 MAC→DSP (`mac_dsp=1`) | TBD | TBD (remote csynth owed; run `split=1 mac_dsp=1` vs `split=1`) | TBD | TBD | TBD | unchanged | BIND_OP the 2→2 MAC mults into DSP48s: trade idle DSP (44% SLR) for binding LUT (np_eq2to2 = 51% of LUT). Bit-exact by construction. Record the np_eq2to2 LUT/DSP delta specifically. |

**Operating point: `--max-input-bits 18`.** It crosses the DSP48 packing threshold
(both operands ≤18 fit one 27×18 block), capturing the full dot-DSP win (840 mults
× 2→1 DSP) plus a 17% LUT drop. Below 18 there is no further DSP boundary to cross,
so 16/14/… only shave peripheral LUT/FF marginally while costing more dot4 precision.
DSP is now ~840 dots (1 DSP each, floor for this multiply count) + ~1310 BN1/normalize/
MAC; further DSP cuts need Lever 3 (relax II=1) or fewer particles (N² scaling).

**⚠ Accuracy gate TBD for Lever 2.** These are csynth-only; the online C-sim golden
gate must be confirmed at width 18 (Lever 2 trades dot4 front-end precision). Record
the GOLDEN / DOTS-LEVEL exact-count + max|Δ| here once run. If 18 passes comfortably,
stop there; do not adopt 16.

## Per-stage split build (resource attribution)

`firmware/nPELICAN_split.cpp` + `build_prj.tcl split=1` synthesize the SAME datapath
as per-stage functions (INLINE off) so csynth reports LUT/FF/DSP/latency per stage —
see `FUNCTION_SPLIT.md` for the stage map and the overhead watchlist. The monolith
stays default; the split project dir is `nPELICAN_split_prj`. Local gate 2026-07-03:
split vs monolith **byte-identical** (200 golden events incl. dots-level + stage dump,
and the 10k legacy flow). Record remote csynth numbers here (same weights.h for both):

First results (Vitis 2023.2, xcvu13p-2, 5 ns; weights: 6/6/6 QAT @ `--max-input-bits 18`;
reports in `reports/csynth_{monolith,split}.rpt`):

| date | build | DSP | FF | LUT | latency (cyc) | II | notes |
|------|-------|-----|----|-----|---------------|----|-------|
| 2026-07-07 | monolith (`nPELICAN_prj`) | 1347 | 63,343 | 229,779 | 14 | 1 | baseline; slack −0.02 ns |
| 2026-07-07 | **split total** (`nPELICAN_split_prj`) | 1579 | 108,786 | 292,033 | 22 | 1 | **overhead vs monolith: +232 DSP (+17%) / +45.4k FF (+72%) / +62.3k LUT (+27%) / +8 cyc**; slack −0.02 ns |
| | ├ np_dots | 1012 | 27,763 | 45,700 | 4 | 1 | 253 dots × 4 mults × 1 DSP48 (18×18) — 64% of all DSP |
| | ├ np_bn1 | 0 | 0 | 38,038 | 0 | 1 | 253 `mul_6s_9ns` → pure LUT |
| | ├ np_agg2to2 | 23 | 7,443 | 21,299 | 3 | 1 | adder trees + 23 normalize mults |
| | ├ np_eq2to2 | 529 | 20,136 | 148,562 | 2 | 1 | 2→2 MAC — **51% of all LUT**, top LUT target |
| | ├ np_agg2to0 | 14 | 6,671 | 21,677 | 4 | 1 | sums/trace + collapsed BN2 |
| | ├ np_out2to0 | 1 | 0 | 85 | 0 | 1 | negligible |
| | └ top residual | 0 | 46,773 | 16,672 | — | — | inter-stage boundary registers (FF) + nobjmask/glue (LUT) — this is most of the split overhead |

### Stage-isolation runs (`split_only=<stage>`, one boundary per run)

Marginal per-stage costs measured with the other five stages inlined (monolith
context); see `FUNCTION_SPLIT.md` "Stage isolation". Baseline = monolith row
above (1347 / 63,343 / 229,779, 14 cyc). All six configs verified byte-identical
csim 2026-07-09. Fill from `nPELICAN_split_<stage>_prj/solution/syn/report/csynth.rpt`:

| stage isolated | stage DSP | stage FF | stage LUT | run total DSP/FF/LUT | latency (cyc) | boundary overhead (total − baseline) |
|----------------|-----------|----------|-----------|----------------------|---------------|--------------------------------------|
| dots    | TBD | TBD | TBD | TBD | TBD | TBD |
| bn1     | TBD | TBD | TBD | TBD | TBD | TBD |
| agg2to2 | TBD | TBD | TBD | TBD | TBD | TBD |
| eq2to2  | TBD | TBD | TBD | TBD | TBD | TBD |
| agg2to0 | TBD | TBD | TBD | TBD | TBD | TBD |
| out2to0 | TBD | TBD | TBD | TBD | TBD | TBD |

Reading guide: per-stage DSP sums exactly to the split top (1012+0+23+529+14+1=1579),
so DSP attribution is clean. The split's overhead is NOT inside the stages: ~47k of the
+45k FF delta is the *top residual* (each non-inlined boundary registers its wide
inter-stage bus, +8 pipeline cycles × multi-k-bit buses), and the +232 DSP / part of the
LUT delta is lost cross-stage sharing inside np_eq2to2's MAC (the monolith CSEs across
the BN1/basis/MAC boundary; the split maps more uniform `mac_muladd_6s_4ns_12s` cores).
**Use the split for proportions (where to optimize); use the monolith for real totals
(what to ship). Throughput is unaffected (II=1 in both).**

## Vivado post-synthesis (vsynth) — netlist-level ground truth

`report_utilization` after `synth_design` (Vivado 2023.2), both builds from the same
weights.h as the 2026-07-07 csynth rows. Reports archived at
`reports/vsynth_{monolith,split}.rpt`. Monolith via `build_prj.tcl vsynth=1`; split via
`vivado -mode batch -source vivado_synth_split.tcl` (build_prj.tcl force-disables vsynth
for `split=1`). Part caveat: these ran on `xcu250-figd2104-2L-e` (remote project.tcl was
edited), repo project.tcl says `xcvu13p-flga2577-2-e` — the U250 is the same VU13P
silicon (identical LUT/FF/DSP/BRAM/URAM totals), so counts are directly comparable.

| build (2026-07-09) | LUT | FF | DSP | CARRY8 | SRL | BRAM |
|--------------------|-----|----|-----|--------|-----|------|
| monolith | 69,735 (4.04%) | 25,142 (0.73%) | 1,343 (10.9%) | 6,929 | 25 | 0 |
| split | 77,157 (4.47%) | 43,238 (1.25%) | 1,575 (12.8%) | 7,096 | 3,695 | 0 |
| Δ split−mono | **+7,422 (+10.6%)** | **+18,096 (+72%)** | **+232 (+17.3%)** | +167 | +3,670 | 0 |
| monolith, pmu-12 weights (2026-07-10) | 69,112 (4.00%) | 24,815 (0.72%) | 1,169 (9.5%) | 7,032 | — | 0 |
| Δ pmu-12 − baseline mono | −623 (−0.9%) | −327 (−1.3%) | **−174 (−13%)** | +103 | | |
| monolith, pmu-10 weights (2026-07-13) | 94,983 (5.50%) | 21,504 (0.62%) | 936 (7.6%) | 9,254 | 49 | 0 |
| Δ pmu-10 − pmu-12 | **+25,871 (+37%)** | −3,311 | **−233 (−20%)** | +2,222 | | |

| monolith, pmu-9 weights (2026-07-13) | 97,174 (5.62%) | 22,514 (0.65%) | 998 (8.1%) | 9,263 | 48 | 0 |
| monolith, pmu-8 weights (2026-07-13) | 98,863 (5.72%) | 21,271 (0.62%) | 977 (8.0%) | 10,618 | 64 | 0 |
| Δ pmu-8 − pmu-10 | +3,880 | −233 | **+41** | +1,364 | | |

pmu-9 row (`reports/{vsynth,csynth}_monolith_pmu9.rpt`, `input_t =
ap_fixed<9,8>`): csynth 752 DSP / 50,184 FF / 281,073 LUT, 15 cyc II=1.
**The "csynth DSP is exact" rule BREAKS in the sub-threshold regime**: vsynth
998 DSP vs csynth 752 — Vivado re-inferred ~246 DSPs from arithmetic that HLS
had emitted as generic fabric logic. The −4 rule only holds when csynth binds
DSP48s explicitly; once HLS strength-reduces/fabric-implements mults, Vivado
makes its own independent inference and BOTH estimates (csynth LUT was also
non-monotonic: 281k at pmu-9 > 262k at pmu-8) become unreliable — in this
regime only vsynth numbers mean anything. Full sub-threshold band at vsynth:
LUT 95.0k/97.2k/98.9k and DSP 936/998/977 for 10/9/8 — LUT creeps up as width
shrinks, DSP wobbles ±60. Uniform picture: ~+37–43% LUT over pmu-12 with no
compensating win anywhere.

pmu-8 row (`reports/vsynth_monolith_pmu8.rpt` + `csynth_monolith_pmu8.rpt`,
`input_t = ap_fixed<8,8>`, integer-GeV momenta): csynth 981 DSP / 51,338 FF /
262,165 LUT, 15 cyc **II=1**. The fabric-spill regime is NON-MONOTONIC: vs
pmu-10, LUT rose again (+3.9k) and DSP rose too (+41) — below the DSP-inference
threshold, resource counts are tool-choice noise around different netlists, not
a function you can dial. Calibration held: DSP csynth→vsynth −4 for the third
consecutive build; FF ratio 0.41; LUT ratio 0.38 (csynth is LESS pessimistic
for fabric-arithmetic-heavy netlists than the 0.30 of DSP-heavy ones).
Verdict: pmu-8 is worse than pmu-10 on BOTH resources and worse than
everything ≥9 on accuracy (AUC 0.9075) — the sub-10 branch is closed;
**pmu-12 remains the resource-accuracy optimum**.

pmu-10 row (`reports/vsynth_monolith_pmu10.rpt`, `input_t = ap_fixed<10,9>`): at
10-bit momenta the tools started moving dot-front-end multiplies out of DSP48s
into fabric — DSP −233 but LUT +25.9k and CARRY8 +2.2k, i.e. ~111 LUT paid per
DSP saved, in the *wrong* direction (LUT was the pressured resource). csynth
latency also went 14 → 15 cycles (fabric mults + longer carry chains need an
extra pipeline stage at 5 ns; II=1 is the invariant that matters, latency is
not). Combined with the Phase A* accuracy cliff at 10 bits (AUC 0.9305 vs
0.9519 at 12), **pmu-10 is dominated by pmu-12 on every axis except raw DSP
count — not an operating-point candidate**, just the far end of the
resource-vs-accuracy curve.

pmu-12 row: monolith rebuilt from the Phase A* 12-bit momentum-quantizer weights
(`fpga_model_qat_w6a6i6p12_best.pt`, `input_t = ap_fixed<12,10>`; see
`INPUT_WIDTH_RETRAIN_PLAN.md`), run WITH `mac_dsp=1` — a proven no-op (Lever 6),
so this report doubles as its netlist-level confirmation. Archived at
`reports/vsynth_monolith_pmu12.rpt`. The csynth deltas carry to the netlist
unchanged: DSP −174 exactly as csynth predicted (and the same −4 csynth→vsynth
offset: 1,173 → 1,169), LUT/FF essentially flat, and the calibration ratios
reproduce to two decimals (LUT 0.30, FF 0.40). csynth remains a trustworthy
*relative* estimator; vsynth remains the absolute source of truth.

### csynth estimate vs vsynth actual — calibration

| metric | csynth mono | vsynth mono | actual/est | csynth split | vsynth split | actual/est |
|--------|-------------|-------------|------------|--------------|--------------|------------|
| DSP | 1,347 | 1,343 | **1.00** | 1,579 | 1,575 | **1.00** |
| FF  | 63,343 | 25,142 | 0.40 | 108,786 | 43,238 | 0.40 |
| LUT | 229,779 | 69,735 | 0.30 | 292,033 | 77,157 | 0.26 |

Takeaways:

- **DSP: csynth is exact** (−4 in both builds, identically). Trust csynth DSP numbers
  as-is, including the split's per-stage attribution (1012 dots floor etc.) and the
  +232 lost-sharing overhead — that overhead is real silicon, not an estimation artifact.
- **LUT/FF: csynth overestimates ~3.3× / ~2.5×**, consistently across both builds. Real
  monolith footprint is ~70k LUT / 25k FF — comfortably inside one SLR (432k LUT). The
  levers' csynth *proportions* (np_eq2to2 ≈ 51% of LUT) remain the right optimization
  guide, but absolute LUT pressure is far lower than csynth implied. The DSP count
  (10.9% of device, 44% of one SLR) is now clearly the binding metric, as assumed.
- **Split overhead at the netlist**: FF overhead stays +72% (same ratio as csynth —
  boundary registers are real registers and survive optimization), while LUT overhead
  collapses from +27% (csynth) to +10.6% (Vivado sweeps most of the boundary glue).
  The split's +3,670 SRL16E are inter-stage pipeline delays retimed into shift
  registers (the +8 latency cycles × wide buses). Conclusion unchanged and now
  netlist-confirmed: **split for attribution, monolith for shipping**.
- Ignore the IOB row (1604/676 = 237%): `synth_design` ran non-out-of-context, so every
  top-level port maps to a pad. Add `-mode out_of_context` to the vivado_synth tcl if a
  pad-free report is ever needed. Both reports are Design State = Synthesized; final
  LUT after opt_design/impl is typically lower still.
- These are utilization-only runs (no XDC/timing in the report); timing status remains
  the csynth estimate (II=1, slack −0.02 ns at 5 ns).

## Phase 2 bit-exactness — interpretation

Zero-tolerance 200/200 csim vs the PyTorch quant logits is **not achievable for this
architecture**, by design, because BatchNorm is kept in float (a CLAUDE.md invariant). Between
the learned quantizers (`input_quant`, `post_agg`×2, `act_layer`, `output_quant`) the datapath
has *unquantized* float segments (the two BatchNorms and the N̄-normalized aggregations).
PyTorch evaluates these in float32; the firmware in fixed-point. The two agree at a quantizer
output only when their pre-quant values round to the same grid point — which fails on a minority
of events where float-vs-fixed rounding straddles a boundary, cascading to ≤~1e-5 on the logit.
This is intrinsic, not a width bug: widening the BN-constant precision (F31→F43) changed nothing,
and a float64 golden barely moved the result; PyTorch's own float32-vs-float64 logits already
differ by up to 3.6e-6. Hence the **tolerance gate** (`nPELICAN_tb.cpp`): PASS = max\|Δ\| under
tolerance (1e-4 network / 1e-3 with dot4), with the zero-tolerance exact count reported alongside.

The **dot4 front-end caveat** (plan D4) is the extra golden-path residual: PyTorch computes
`d_ij = E²−|p|²` in lossy float32 (catastrophic cancellation on self-dots), the firmware computes
it exactly from the momenta, so they round to different 2⁻¹⁰ grid points. The **dots-level gate**
(inject PyTorch's quantized dots via `npelican_dots_override`, see `golden_dots.dat`) removes this
variable and verifies the network in isolation (142/200 exact, max 1.1e-5).
