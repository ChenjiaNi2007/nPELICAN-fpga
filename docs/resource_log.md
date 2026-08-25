# Firmware Resource Log

Records synthesis and C-sim results at each phase of the QAT restructure
(see `FIRMWARE_QAT_PLAN.md` for phase definitions).

| phase | date | checkpoint | weights.h commit | LUT | FF | DSP | latency (cycles) | II | timing met | csim vs golden (exact/total, max\|Δ\|) | notes |
|-------|------|------------|-----------------|-----|----|-----|------------------|----|------------|----------------------------------------|-------|
| Phase 0 baseline | 2026-06-12 | fpga_model_qat_best.pt (8ep, 24-bit po2) | 3c688bf | n/a | n/a | n/a | n/a | n/a | n/a | **Vitis csim: 0/200 exact, max\|Δ\|=3.6066896** (remote output archived at tb_data/.ipynb_checkpoints/golden_fw_results-checkpoint.log) | Pre-restructure uniform-type firmware. Local clang + open-source ap_types build exhibits UB (-O0 vs -O2 outputs differ) → local builds are smoke checks only; authoritative csim is remote Vitis. Homebrew gcc-13 too old for current SDK; `brew upgrade gcc` may revive a trustworthy local loop (local differs from Vitis on 200/200 events, max gap 5.64). CSYNTH DID NOT COMPLETE on the remote box — known pre-existing issue with the high particle count (also failed for the previous model); user deprioritized. II/resource regression tracking deferred until synth is viable. |
| Phase 2 retype | 2026-06-13 | fpga_model_qat_best.pt | 1ed473a | TBD | TBD | TBD | TBD | TBD (remote) | TBD | **csim GATE PASS** — dots-level (network) 142/200 zero-tol exact, max\|Δ\|=1.1e-5 (tol 1e-4); golden (incl dot4) 133/200 exact, max\|Δ\|=6.3e-4 (tol 1e-3). Local==Vitis confirmed bit-for-bit (retyped firmware is toolchain-stable; local is now a trustworthy oracle). | Per-stage QAT types wired. Bugs fixed: Tr saturation (t0_t→tr_t), BN eps omitted (weight/sqrt(var+eps)). Tolerance gate (not zero-tol 200/200): float BatchNorm leaves float intermediate segments that fixed-point can't match bit-for-bit at every quantizer boundary (PyTorch's own f32-vs-f64 logits differ ~3.6e-6). dot4 front-end caveat (D4): PyTorch d_ij in lossy float32 → ≤6.3e-4 golden-path residual; dots-level gate isolates+proves the network. csynth still deferred (pre-existing high-particle-count issue). |

## Resource-reduction sweep (6/6/6 QAT, 20 particles → 22 with spurions)

Synth (csynth) results for the resource-reduction effort; see
`RESOURCE_REDUCTION_LEVERS.md` for the levers. Device SLR limits: DSP 3072, LUT 432000,
FF 864000.

### 2026-08-25 — BN1 constant evicted to fabric (`--bn-frac-bits 12`), 16 particles

Same epoch-34 pmu-12 checkpoint and weights as the 2026-08-22 16p build; the ONLY change is
`bn_t_gen` F 14 → 12, which moves the BN1 γ/σ literal from 653 (10 bits, `mul_6s_11ns_16`
×171 on DSP48, slack −0.00) to 163 (8 bits, `mul_6s_9ns_14` ×171 in fabric at 49 LUT each,
timing met at est 4.372 ns).

| build | LUT | FF | DSP | CARRY8 | lat | timing |
|---|---|---|---|---|---|---|
| 16p pmu-12, BN1 on DSP (F=14) | 48,419 | 15,487 | 941 | 4,111 | 15 | slack −0.00 ⚠ |
| 16p pmu-12, `--bn-frac-bits 12` | 49,894 | 15,416 | **769** | 4,446 | 15 | met (4.372 ns) |

Δ = −172 DSP / +1,475 LUT / +335 CARRY8 → **8.6 LUT per DSP saved, the cheapest DSP lever
measured** (width levers run 111–189). Latency did NOT drop to the 20p baseline's 13 —
the 16p 15-cycle schedule is not BN1's doing; open question. Threshold model corrected:
fabric measured at an 8-bit literal, DSP48 + timing violation at 10 bits, 9 untested; Bind
Op module names carry a +1 zero-extension bit (653→`11ns`, 163→`9ns`). Reports:
`reports/particle_count/{c,v}synth_16p_pmu12_bnf12.rpt`.

### 2026-08-25 — II sweep (1/2/3/4), 16p + `--bn-frac-bits 12`

Same bnf12 export, only the initiation interval changed (II>1 breaks the II=1 deliverable
invariant on purpose — these are throughput/resource trade points, not the deliverable; the
II pragma edit must be reverted for the II=1 build).

| build (16p pmu-12 bnf12) | LUT | FF | DSP | CARRY8 | lat | II | evt period | timing est |
|---|---|---|---|---|---|---|---|---|
| II=1 | 49,894 | 15,416 | 769 | 4,446 | 15 (75 ns) | 1 | 5 ns | met (4.372) |
| II=2 | 49,459 | 17,594 | **769** | 4,671 | 17 (85 ns) | 2 | 10 ns | met (4.367) |
| II=3 | 48,012 | 15,686 | **313** | 4,268 | 19 (95 ns) | 3 | 15 ns | met (4.346) |
| II=4 | 47,121 | 15,433 | **257** | 4,272 | 26 (130 ns) | 4 | 20 ns | met (4.367) |

**II=2 is a dead point: zero DSP saved** (769, identical to II=1) while FF rises +2,178 and
CARRY8 +225 — the scheduler keeps the dot stage fully inlined and only pays the pipelining
overhead. Sharing kicks in at II=3: dots restructure into `dot4` function instances (4 DSP
each) time-multiplexed pair-per-cycle — 57 units at II=3 (⌈171/3⌉=57, perfect), **43 at II=4
(⌈171/4⌉=43, perfect → 172 dot DSP)**. The non-dot floor stays ~85 DSP throughout, so
II=4's 257 ≈ 172+85 and the curve is flattening: II=5 would predict ~35·4+85 ≈ 225, only
−32 more for another 5 ns of period. LUT *drops* monotonically past II=2 (muxes are cheaper
than the inlined dot fabric). BN1 stays in fabric at every II (`mul_6s_9ns_14` ×171).
Throughput: every point is inside the 25 ns LHC bunch spacing except II>5 territory; II=4's
20 ns is the last comfortable one at this clock. Reports:
`reports/particle_count/{c,v}synth_16p_pmu12_bnf12_ii{2,3,4}.rpt`.

**Where the report files are (reorganised 2026-08-23):** `../reports/README.md` is the index
for every current, mutually comparable report (all xcu250 @ 5 ns), including the new
`reports/particle_count/` and `reports/capacity_sweep/` subfolders. The `.txt` reports this
section cites were loose at the workspace root; they are now
`../../synthesis-archive/2026-06_xcvu13p_pre-lever2/` under descriptive names (mapping table
in that folder's README) — **xcvu13p, pre-Lever-2, not comparable to anything below.**

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

| monolith, block-FP W=7 (2026-08-09) | 123,543 (7.15%) | 25,288 (0.73%) | 1,090 (8.9%) | 9,034 | 134 | 0 |
| Δ block-FP W=7 − pmu-12 | **+54,431 (+79%)** | +473 | −79 (−6.8%) | +2,002 (+29%) | | |

| monolith, block-FP W=7, **16 particles** (2026-08-21) | 84,302 (4.88%) | 17,928 (0.52%) | 751 (6.1%) | 6,043 | 114 | 0 |
| Δ 16p − 20p (both block-FP W=7) | **−39,241 (−31.8%)** | −7,360 (−29.1%) | **−339 (−31.1%)** | −2,991 (−33.1%) | −20 | 0 |
| Δ 16p block-FP − pmu-12 20p | +15,190 (+22.0%) | −6,887 (−27.8%) | **−418 (−35.8%)** | −989 (−14.1%) | | |
| monolith, **pmu-12, 16 particles** (2026-08-21) | **48,419 (2.80%)** | 15,487 (0.45%) | 941 (7.7%) | 4,111 | 54 | 0 |
| Δ pmu-12 16p − pmu-12 20p | **−20,693 (−29.9%)** | −9,328 (−37.6%) | −228 (−19.5%) | −2,921 (−41.5%) | | |
| Δ pmu-12 16p − block-FP 16p | **−35,883 (−42.6%)** | −2,441 (−13.6%) | **+190 (+25.3%)** | −1,932 (−32.0%) | −60 | 0 |

**pmu-12 @ 16 particles (2026-08-21, `12pmu-16pt-{vivado_synth,csynth}.rpt` at the
workspace root).** Re-exported from `fpga_model_qat_w6a6i6p12_best.pt` (`input_t =
ap_fixed<12,10>`, `dot_t = ap_fixed<6,10>`, no `NPELICAN_BLOCK_FP`). csynth: 945 DSP /
37,844 FF / 164,172 LUT, **15 cycles**, II=1.

**Lowest-LUT build measured to date — 2.80% of the xcu250**, beating the previous best
(pmu-12 20p, 4.00%) by a quarter. Completes the 2×2:

| vsynth | 20 particles | 16 particles |
|--------|--------------|--------------|
| pmu-12      | 1,169 DSP / 69,112 LUT | **941 DSP / 48,419 LUT** |
| block-FP W=7| 1,090 DSP / 123,543 LUT | 751 DSP / 84,302 LUT |

At fixed N=16 the block-FP trade is **+35,883 LUT to save 190 DSP = 189 LUT/DSP** —
the same wrong-direction exchange rate as pmu-10 (111 LUT/DSP) and 7j. Particle count
moves both together; width levers only shuffle between them.

⚠ **DSP came in 20% over the (N+2)² prediction (941 vs 782) — and it is NOT a scaling-law
failure.** The DSP inventory reconciles exactly:

| contribution | count | 20p baseline had |
|---|---|---|
| dot: `mul_12s_12s` ×171 + 3× `mac_mulsub_12s_12s` ×513 | 684 | 1,012 (scales ✓) |
| BN1: `mul_6s_11ns_16` ×171 | **171** | **0 — was fabric** |
| aggregation `am_addmul_16s_16s` ×18 + `mac_muladd_6s_5*` ×54 | 72 | — |
| output-stage tail | 18 | — |
| **total** | **945** | |

The 20p pmu-12 baseline (`6:6:6_pmu12_5ns`) has **no `mul_6s_11ns_16` at all** — its BN1
scale constant fit in ≤10 bits and strength-reduced into LUTs. This re-export snapped
γ/σ to an **11-bit** constant (`bn_t_gen = ap_fixed<21,6>`), crossing the same threshold
documented for pmu-8 above, and 171 multiplies moved fabric → DSP48. Net out the BN1 term
and the dots land at 684 + ~86 tail ≈ 770, against the 782 prediction. **The law holds;
the baseline moved.**

⚠ **TIMING VIOLATION — read before quoting the latency.** The 16p pmu-12 build reports
`Issue Type: Timing`, **slack −0.00 ns**, and **15 cycles** vs the 20p baseline's 13
cycles at a comfortable 4.345 ns estimate. A *smaller* design needing *two more* pipeline
stages and still missing 5 ns is the same BN1 rebinding: a DSP48 has far longer
clock-to-out than a strength-reduced LUT constant multiply, so inserting one into the
dot→BN1→aggregate path lengthened the critical path and forced HLS to add stages. The
pattern holds across every build on file — BN1-on-DSP builds run 15–16 cycles (bfp7 20p
and 16p both 16), the one BN1-in-fabric build runs 13.

**This build should not be shipped as-is.** The fix is to keep the BN1 constant under 11
bits (cap `bn_t_gen` F, or round γ/σ to a 10-bit grid) — which should recover both the
171 DSPs and the two pipeline stages, and is worth testing before the 12p run.

**16-particle row (2026-08-21, `16pt-vivado_synth.rpt` + `16pt-csynth.rpt` at the
workspace root).** First particle-count run ever done at a current operating point —
the only prior N-sweep (`6_6_12_12p.txt` / `6_6_12_14p.txt`, 2026-06-16) was csynth-only,
on xcvu13p, at the pre-Lever-2 `6_6_12` config with `input_t` still pinned at
`ap_fixed<36,12>`, so its 12p/14p/20p DSP numbers (1,794 / 2,383 / 4,677) run ~3.5× the
current point and are not comparable to anything in this table.

⚠ **This build is block-FP W=7, NOT pmu-12.** Identified from the report itself, not from
the build flags: `beam_input` port is 200 bits = 2·4·25 → `input_t` is 25-bit
(`ap_fixed<25,12>`, the bfp7 `types_generated.h`), and the Bind Op report carries
`mul_7s_7s_14` / `mac_mulsub_7s_7s_*`. The like-for-like comparison is therefore the
**20p block-FP row above**; the pmu-12 delta is cross-config and mixes two levers.

csynth: 757 DSP / 42,427 FF / 237,586 LUT, **16 cycles, II=1** (unchanged from 20p).

**(N+2)² scaling confirmed to within a few percent.** Predicted = 20p value × (18/22)² =
×0.6694:

| metric | 20p bfp7 | 16p predicted | 16p measured | error |
|--------|----------|---------------|--------------|-------|
| LUT    | 123,543  | 82,702        | 84,302       | +1.9% |
| FF     | 25,288   | 16,928        | 17,928       | +5.9% |
| DSP    | 1,090    | 730           | 751          | +2.9% |
| CARRY8 | 9,034    | 6,048         | 6,043        | −0.1% |

CARRY8 lands within 0.1% of the law — the adder tree is purely per-pair. The small
positive bias on DSP/LUT/FF is the N-independent tail (output stage, nobj comparators,
`am_addmul` realign) that does not shrink with particle count.

**DSP attribution reproduces exactly.** Under block-FP the bare `mul_7s_7s` falls to
fabric, so dot DSP = 3 mulsub/pair, and BN1 contributes 1 DSP/pair (`mul_6s_11ns_16`,
the 11-bit-constant threshold crossing documented above):

| slots | pairs = s(s+1)/2 | 3× mulsub | BN1 | subtotal | csynth total |
|-------|------------------|-----------|-----|----------|--------------|
| 22 (20p) | 253 | 759 | 253 | 1,012 | 1,097 |
| 18 (16p) | 171 | 513 | 171 | 684   | 757   |

Per-pair module counts scale as predicted: `mul_6s_11ns_16` 253→171 (= s(s+1)/2) and
`mul_6s_6ns_11` 231→153 (= s(s−1)/2, off-diagonal pairs only).

⚠ **Timing tightened, not loosened.** 20p block-FP csynth estimated 4.361 ns against the
5.00 ns target; the 16p summary reports slack 0.01 ns. Report formats differ (full
`Utilization Estimates` vs the 2023.2 Synthesis Summary), so this is not a clean
comparison, but a *smaller* design showing *less* slack is the opposite of the expected
direction and should be checked before the number is used to argue for a faster clock.

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

**Multiplier-inventory receipt (2026-07-14, read from the archived csynth
reports — no re-synthesis needed).** The per-instance tables prove both that
the trained momentum grid reached the datapath and why LUT rises sub-threshold:

| build | dot-mult module | count | per-instance binding |
|---|---|---|---|
| baseline (18-bit) | `mul_18s_18s_36` | 253 | **1 DSP**, 0 LUT |
| pmu-9 | `mul_9s_9s_18` | 420 | **0 DSP, 49 LUT** |
| pmu-8 | `mul_8s_8s_16` | 420 | **0 DSP, 40 LUT** |

The operand widths ARE the trained pmu widths (pseudoweight precision
propagated), and at ≤10-bit operands Vitis's DSP-inference threshold sends
every dot mult to fabric: 420×49 ≈ 20.6k / 420×40 ≈ 16.8k LUT of bare
multipliers plus carry/glue = the observed LUT jump. (Instance counts differ
across netlists because CSE/sharing re-decides per build.)

**The 12→10 exchange-rate accounting (2026-07-14).** Naively 25,871 LUT / 55
LUT-per-fabric-mult "should" equal ~470 freed DSPs, but only 233 were freed —
the ratio is biased on both sides. Numerator: the LUT delta is ~420 dot mults
× ~55 ≈ 23k PLUS the adder/register logic the DSP48s had been absorbing for
free (pre-adder, post-adder, pipeline regs → +2.2k CARRY8 and glue) — a DSP48
implements a mult+add+regs bundle, not a bare multiplier, so evicting one
dumps MORE than 55 LUT into fabric. Denominator: ~420 mults left HLS's DSP
binding, but Vivado re-spends ~190 of the freed DSPs on other arithmetic at
netlist level (measured directly at pmu-9: csynth 752 → vsynth 998), so the
NET DSP drop understates the migration. Same ~420 multipliers on both sides;
the apparent inefficiency is glue (numerator) + re-spend (denominator).
Conclusion the numbers force: total effective resources are strictly worse
below 12 bits — but this is a TOOL DEFAULT, not physics: the dot mults are
variable×variable, so `BIND_OP variable=dots op=mul impl=dsp` would keep
them in DSP48s at 10 bits and reverse the entire LUT penalty, if a sub-12
width were ever wanted despite the accuracy cost. The extra DSPs are NOT dot mults — they're the ~250 BN1 multiplies
(`nPELICAN.cpp:187`, `dots × batch1_2to2[1]`, i.e. 6-bit dot × loader-derived
BN1 scale constant; a real mult because the constant lives in an array, which
blocks constant-folding). In the pmu-9 build that constant's type is 10 bits →
`mul_6s_10ns_15` = 0 DSP / 62 LUT each; in the pmu-8 build the checkpoint's
learned BN constant needed 11 bits → `mul_6s_11ns_16` = **1 DSP / 6 LUT**
each. One bit of loader-derived constant width pushed the whole ~250-instance
population over the DSP threshold, trading ~14k LUT for ~250 DSPs. So the
DSP wobble across 10/9/8 (936/998/977) is marginal multiplier populations
flipping deterministically across the ~11-bit threshold as per-checkpoint
type widths shift by ±1 bit — checkpoint-specific threshold crossings, not
randomness, but still not a dial anyone can usefully turn. If the threshold
story ever needs a hard proof: the dot mults are variable×variable (unlike
the Lever-6 constant MACs), so a `BIND_OP variable=dots op=mul impl=dsp`
run would force them back into DSPs — one cheap confirmatory synth.

**block-FP W=7 row (2026-08-09, `bfp7_reports/` at the workspace root — vsynth
`vivado_synth.rpt`, csynth `nPELICAN_prj/solution/syn/report/nPELICAN_csynth.rpt`;
csynth 1,097 DSP / 58,784 FF / 343,761 LUT, 16 cyc II=1 @ 5 ns).** The multiplier
inventory finishes the story the pmu-9/8 receipt above started, and the answer is
structural: **the Minkowski dot is 1 bare multiply + 3 multiply-SUBTRACTS, and only
the bare multiply is width-addressable.**

| per-pair op | uniform pmu-12 | block-FP W=7 |
|---|---|---|
| bare mul (`E·E`) | 253 × `mul_12s_12s_24` → **253 DSP**, 1,265 LUT | 253 × `mul_7s_7s_14` → **0 DSP, 8,349 LUT** |
| 3 spatial terms | 759 × `mac_mulsub_12s_12s_*` → **759 DSP** | 759 × `mac_mulsub_7s_7s_*` → **759 DSP** |
| dot front end | **1,012 DSP** = 96% of the design's 1,053 | **759 DSP** |

A DSP48E2 computes `(A±D)×B + C` natively, so it performs the subtraction for free —
HLS binds a mulsub for its ACCUMULATE STRUCTURE, not its operand width, and a 7×7
multiply still occupies a whole 27×18 site. Going 12 → 7 bits therefore frees **one in
four** dot DSPs and nothing more. Those 253 are then handed straight back: this
checkpoint's BN1 constant needs 11 bits, so its 253 BN1 multiplies bind as
`mul_6s_11ns_16` = **1 DSP / 6 LUT each** — the identical module and binding recorded
for pmu-8 two paragraphs up, i.e. the same checkpoint-specific 11-bit threshold
crossing, nothing to do with block-FP. Net −79 DSP at vsynth, for +54.4k LUT. The LUT
goes to the fabric mantissa mults (+8.3k), 231 × `mul_6s_6ns_11` (+5.3k), and block-FP's
own machinery: 22 exponent cascades, 88 variable right-shifters on 35-bit `mraw_t`, and
253 realign shifters on 36-bit `dotalign_t` (the +2,002 CARRY8 signature). ⚠ The pmu-12
csynth numbers quoted for comparison are the Jul-17 clock-sweep build (1,053 DSP /
51,395 FF / 199,410 LUT, 13 cyc), NOT the build behind the Jul-10 pmu-12 vsynth (which
csynth'd 1,173) — no csynth→vsynth calibration is claimed across that pair. Full
analysis and verdict: `RESOURCE_REDUCTION_LEVERS.md` §7j.

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

## Clock sweep — is the split overhead a scheduling artifact? (2026-07-16)

Hypothesis (prof): at 5 ns the monolith skips the boundary registers the split is
forced to have; tighten the clock (`build_prj.tcl period=4|3`, per-period project
dirs) and the monolith must "buffer" at the same places, so the split−mono gap
should shrink if it's a scheduling artifact. **Answer: NO — the gap is structural.**
All runs: current 6:6:6 pmu-12 tree, xcu250 (same VU13P silicon), II=1, timing met
at every period (csynth est 4.345 / 3.48–3.49 / 2.56–2.57 ns incl. uncertainty).
Reports archived in `reports/clock_sweep/` (csynth 4/3 ns pairs live on the pod in
`nPELICAN{,_split}_prj_{4,3}ns/`); vsynth via
`vivado -mode batch -source vivado_synth[_split].tcl -tclargs _<N>ns`.

csynth estimates:

| clock | build | DSP | FF | LUT | Δ split−mono |
|-------|-------|-----|-----|------|--------------|
| 5 ns | mono  | 1,053 | 51,395 | 199,410 | |
| 5 ns | split | 1,053 | 89,662 | 304,424 | +38.3k FF / **+105.0k LUT** |
| 4 ns | mono  | 1,053 | 63,250 | 200,210 | |
| 4 ns | split | 1,053 | 120,096 | 304,440 | +56.8k FF / **+104.2k LUT** |
| 3 ns | mono  | 1,053 | 82,673 | 201,041 | |
| 3 ns | split | 1,053 | 136,778 | 304,542 | +54.1k FF / **+103.5k LUT** |

vsynth (netlist ground truth):

| clock | build | LUT | FF | DSP | CARRY8 | SRL16E | Δ split−mono (LUT / FF) |
|-------|-------|-----|-----|-----|--------|--------|--------------------------|
| 5 ns | mono  | 48,962 | 21,631 | 1,049 | 3,668 | 27 | |
| 5 ns | split | 74,219 | 37,593 | 1,049 | 6,328 | 3,425 | **+25.3k (+52%) / +16.0k (+74%)** |
| 4 ns | mono  | 46,176 | 21,575 | 1,049 | 3,774 | 1,555 | |
| 4 ns | split | 67,270 | 47,362 | 1,049 | 6,493 | 3,114 | **+21.1k (+46%) / +25.8k (+120%)** |
| 3 ns | mono  | 45,509 | 33,199 | 1,049 | 3,814 | 1,568 | |
| 3 ns | split | 71,949 | 53,580 | 1,049 | 6,504 | 3,150 | **+26.4k (+58%) / +20.4k (+61%)** |

Takeaways:

- **The combined LUT+FF gap does not shrink with clock: 41.2k → 46.9k → 46.8k**
  (5→4→3 ns). Tightening the clock DID force buffers into the monolith (mono FF
  +54% at 3 ns) and that narrows the FF gap 4→3 ns (25.8k→20.4k), but the LUT gap
  widens in step (21.1k→26.4k) — overhead shifts columns, total is conserved. The
  split penalty is per-boundary interface registers + lost cross-boundary sharing,
  NOT a relaxed-schedule artifact. "Split for attribution, monolith for shipping"
  survives a 5→3 ns sweep.
- **The forced buffers are visible and cheap-to-see in the SRL column**: the mono
  absorbed 4 ns almost entirely as SRL16E (27→1,555, FF flat 21.6k) and only at
  3 ns spilled into real FFs (+11.6k). The split can't do this — its boundary
  registers are architectural.
- **DSP: 1,049 everywhere** (both builds, all three clocks; csynth 1,053 → −4
  rule holds again). Per-stage DSP attribution is exact and clock-invariant.
- **csynth grossly overstates the split penalty**: ΔLUT ~104–105k estimated vs
  21–26k at the netlist (~80% is estimation artifact — Vivado's cross-hierarchy
  flattening recovers the sharing HLS's per-instance estimate can't see). Never
  quote split overhead from csynth.
- csynth LUT is clock-flat in BOTH builds (mono 199–201k, split 304.4–304.5k) —
  csynth LUT simply doesn't see scheduling; only its FF column responds to clock.
- Latency: mono 13 → split 20 cyc at 5 ns (the +7 boundary stages), II=1 in all
  six runs.

### split=2 result — lost symmetry-CSE CONFIRMED as the dominant mechanism (2026-07-17)

`split=2` (triangular symmetric crossings, `NPELICAN_SPLIT_TRI`; FUNCTION_SPLIT.md)
changes ONLY the port representation of `dots`/`batch1` (253-element upper
triangle, `NP_SYMIDX` maps (i,j)/(j,i) to the same element). Byte-identical
csim vs monolith and split=1. Reports:
`reports/clock_sweep/{csynth,vsynth}_split_tri_5ns.rpt`. All @5 ns, same tree:

| build (5 ns) | csynth FF | csynth LUT | vsynth LUT | vsynth FF | vsynth SRL | DSP | lat |
|--------------|-----------|------------|------------|-----------|------------|-----|-----|
| monolith | 51,395 | 199,410 | 48,962 | 21,631 | 27 | 1,049 | 13 |
| split=1  | 89,662 | 304,424 | 74,219 | 37,593 | 3,425 | 1,049 | 20 |
| split=2  | 66,985 | 234,241 | 59,460 | 25,863 | 2,345 | 1,049 | 19 |
| Δ split=1 − mono | +38.3k | +105.0k | **+25,257 (+52%)** | **+15,962 (+74%)** | +3,398 | 0 | +7 |
| Δ split=2 − mono | +15.6k | +34.8k | **+10,498 (+21%)** | **+4,232 (+20%)** | +2,318 | 0 | +6 |

Takeaways:

- **The port-representation change alone closes 58% of the netlist LUT gap,
  73% of the FF gap (64% of LUT+FF combined: 41.2k → 14.7k)** — with zero
  arithmetic change and byte-identical outputs. The split overhead was mostly
  the two symmetric arrays crossing as 484 "independent" scalars: duplicated
  CSE-able MAC products (LUT) + boundary registers for 231 redundant mirror
  elements per array per consumer (FF).
- Direct answer to "HLS is sensitive to how these things are written": yes —
  measurably. Same math, same types, one representational change at two ports,
  −14.8k LUT / −11.7k FF at the netlist.
- **Residual split=2 overhead (+10.5k LUT / +4.2k FF / +6 cyc) is the true
  irreducible boundary cost** of this 6-way split: interface/control for the
  non-symmetric crossings (Tp_q 968 elems dominates), remaining cross-boundary
  sharing (mask products, range narrowing), and per-stage FSMs.
- DSP 1,049 / csynth 1,053 (−4 rule), II=1, timing met — invariant again.
- Attribution guidance updated: split=2 is now the PREFERRED attribution build
  (closest-to-monolith per-stage proportions); split=1 remains for continuity
  with the existing isolation table.

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
