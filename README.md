# nPELICAN-fpga

C++ HLS code for implementing nanoPELICAN on FPGAs.

`clean.sh` removes files from previous builds. Vitis configuration values are set at the
top of `build_prj.tcl`.

## Regenerate-on-retrain workflow

The firmware datapath uses per-stage fixed-point types **derived from the trained model's
learned QAT scales**, so retraining (at any bit width) regenerates the firmware with no manual
header edits. The loop:

1. **Train** (in `../PELICAN-nano/`) with QAT:
   ```bash
   python train_pelican_nano.py --datadir ./data/sample_data --target is_signal \
       --quant --po2-scales --weight-bit-width 24 --act-bit-width 24 --input-bit-width 24 \
       --n-hidden 2 --nobj 20 --nobj-avg 49 --prefix fpga_model_qat
   ```
   Set the bit widths up front; QAT learns each quantizer's po2 scale `2^-k`, which *is* the
   split — the firmware type for a B-bit quantizer is `ap_fixed<B, B-k>` (fractional bits = k,
   integer bits = B-k). `scripts/check_scales.py` prints the learned split per quantizer.

2. **Export weights + types** (regenerates both headers):
   ```bash
   python model_loader.py --model ../PELICAN-nano/model/fpga_model_qat_best.pt --quant \
       --out firmware/weights/weights.h --repo ../PELICAN-nano
   ```
   Writes `firmware/weights/weights.h` and `firmware/weights/types_generated.h`
   (`firmware/nPELICAN.h` `#include`s the latter). The quantization-point types
   (`dot_t, t2_t, relu_t, t0_t, out_t, w1_gen_t, w2_gen_t`) come straight from the learned
   scales; the intermediate/accumulator/MAC types (`bn1out_t, tr_t, acc*_t, mac*_t, bias_t_gen,
   bn_t_gen, norm_t`) are formula-derived from those plus the term counts. `--bn-eps` (default
   1e-5) must match the training BatchNorm eps — the BN scale is `weight/sqrt(var+eps)`.

   **Model dimensions** (`NPARTICLES`, `NHIDDEN`, `NOUT`) are also emitted at the top of
   `types_generated.h` from the checkpoint; `nPELICAN.h` keeps `#ifndef`-guarded hand defaults
   (20/2/1, `NPARTICLES2 = NPARTICLES + 2`) and `weights.h` `static_assert`s that they match.
   A K-class checkpoint (`--n-out K` in PELICAN-nano) exports `NOUT=K` and the firmware emits K
   raw logits via `model_out[NOUT]` (softmax/argmax off-chip); `golden_logits.dat` then carries
   NOUT space-separated values per line. `w2_2to0` element order is row-major `(NOUT, NHIDDEN*2)`
   = `np.ravel(agg_2to0.mixing.weight)`, i.e. `w2_2to0[o*(NHIDDEN*2) + h*2 + a]`.

3. **Export golden vectors** (for the csim gate), in `../PELICAN-nano/`:
   ```bash
   python scripts/export_golden.py
   ```
   Writes `tb_data/golden_{pmu,nobj,logits,dots}.dat` and `golden_stage_dump.txt`.

4. **C-sim** (`vitis_hls -f build_prj.tcl`). By default the testbench runs the legacy 10k flow;
   define `RUN_GOLDEN_GATE` to run the bit-exactness gate instead (see below).

## Bit-exactness gate

The testbench compares firmware logits to the PyTorch quant-model logits and prints:

- `GOLDEN GATE` — momenta path (firmware computes `d_ij` from the four-momenta via `dot4`).
- `DOTS-LEVEL GATE` — network isolated: PyTorch's quantized `d_ij` are injected in place of the
  `dot4` front-end (`tb_data/golden_dots.dat` -> `npelican_dots_override`, csim-only).

Each gate is a **tolerance** gate (PASS = `max|delta|` under tolerance), with the zero-tolerance
exact count reported alongside. Zero-tolerance 200/200 is **not achievable** here by design:
BatchNorm is kept in float, so the datapath has unquantized float segments between the learned
quantizers that fixed-point cannot reproduce bit-for-bit at every quantizer boundary (PyTorch's
own float32-vs-float64 logits already differ by ~3.6e-6). See `docs/resource_log.md` for the
full interpretation and the dot4 front-end caveat. Current result: dots-level 142/200 exact,
`max|delta|`=1.1e-5; golden 133/200 exact, `max|delta|`=6.3e-4.

## Layout

- `firmware/nPELICAN.{h,cpp}` — the datapath. `nPELICAN.h` holds the IO/interface typedefs and
  `#include`s the generated `weights/types_generated.h`; `weights/weights.h` is generated.
- `model_loader.py` — reads a PELICAN-nano checkpoint, writes `weights.h` + `types_generated.h`.
- `nPELICAN_tb.cpp` — testbench (golden + dots-level modes, stage-dump harness).
- `docs/FIRMWARE_QAT_PLAN.md` — the restructure spec; `docs/resource_log.md` — phase results.

## Numerical contracts added 2026-10-06 (5-class / NOUT generalization)

- `nobj` port = particles present in the NPARTICLES slots; clamp raw multiplicities to NPARTICLES
  (testbench `clamp_nobj`, `export_golden.py`). Port width `NOBJ_BITS` follows NPARTICLES2.
- `model_loader.py --bias-guard-bits 8` / `--agg-guard-bits 8` (defaults): bias literals and the
  stored BN1 output keep 8 fractional bits beyond the MAC / post-agg grids so exact half-LSB ties
  cannot form (float PyTorch never ties). New generated type `mac2b_t` (2->2 bias end-add; falls
  back to `mac2_t` for pre-change headers, byte-identical). Set both to 0 for the old rule.
- Gate record for the first NOUT=5 export: `reports/hls4ml5_smoke_gate/GATE.md` (200/200 exact).
- `--jet-quant-split` checkpoints: loader emits `NPELICAN_JET_QUANT_SPLIT` + `dotj_t`/`dotm_t` (jet row/col,
  m_jet^2 dot grids), `jet_t` (jet_input port, pmu_quant_jet) and `dotall_t` (dots container); all alias
  dot_t/input_t otherwise (byte-identical). Gate: `reports/hls4ml5_smoke_gate/GATE_JET_QUANT_SPLIT.md`.

## Synthesizing the 5-class jet+head model (2026-10-07)

`firmware/weights/` holds the export of `PELICAN-nano/model/qat6wj16_h4_jh_e40_s2_best.pt`
(N=16, h=4, w6a6i6 pmu12, jet spurion + 16-wide head, split jet quantizers at 10/16/20 bits;
test accuracy 69.9%, see PELICAN-nano `docs/HLS4ML_5CLASS.md`), with the default DSP-tuned BN1
literal. Matching golden vectors (incl. `tb_data/golden_jet.dat`) are in `tb_data/`.
On the remote: `git pull`, then the usual `vitis_hls -f build_prj.tcl ...`. For a meaningful
csim add `-DRUN_GOLDEN_GATE` to the testbench cflags in `build_prj.tcl` (the `add_files -tb
nPELICAN_tb.cpp -cflags` line); expected `GOLDEN SUMMARY: events=200 exact=199` (the one residual
is the documented BN1-literal tie; `--bn-guard-bits 8` on export gives 200/200 at DSP cost).
The legacy 10k csim flow is not meaningful for this model (toptag inputs, 20 particles, no jet).
To re-export: `python model_loader.py --model ../PELICAN-nano/model/qat6wj16_h4_jh_e40_s2_best.pt
--quant --repo ../PELICAN-nano --out firmware/weights/weights.h`.

## Synthesizing the plain 5-class model, N=16 h8 (2026-10-15) — the (E,px,py,pz) Mixer's partner

`firmware/weights/` now holds the export of `PELICAN-nano/model/h5n16_h8_qat_s1_best.pt` (N=16, h=8,
no jet spurion, no head, w6 a6 i6 pmu12; test 52.22 % / macro AUC 81.54 vs float h8 55.5-55.9 %),
exported with `--bn-guard-bits 8` — the default 10-bit BN1 literal gave 182/200 on the golden gate,
the 18-bit one 200/200 (`reports/hls4ml5_h8_qat_gate/GATE.md`). Golden vectors in `tb_data/` are the
first 200 jets of `data/hls4ml5_n16/test.h5` (`golden_jet.dat` removed: no jet port). The jet+head
export above is restorable with `git checkout 64ddce8 -- firmware/weights tb_data`.

Remote, like-for-like with the MLP-Mixer flow (5 ns, Vivado synth + opt on the U250):

    export NPELICAN_PART=xcu250-figd2104-2L-e      # project.tcl honours it (csynth + vsynth)
    vitis_hls -f build_prj.tcl "reset=1 csim=1 synth=1 cosim=0 validation=0 export=0 vsynth=1 golden=1"

`golden=1` puts `-DRUN_GOLDEN_GATE` on the testbench, so csim prints `GOLDEN SUMMARY: events=200
exact=200`. Reports: `nPELICAN_prj/solution/syn/report/nPELICAN_csynth.rpt` (latency, II, DSP),
`vivado_synth.rpt` (post-synth, comparable with every earlier nanoPELICAN report) and
`vivado_synth_opt.rpt` (+ `_hier`, post-`opt_design`: the number to put next to the Mixer's
86.5k / 81.3k LUT, which are post-opt). Local gate: `./build_local.sh -DRUN_GOLDEN_GATE && ./tb_local`.

If csynth sits at `[SCHED 204-61] Pipelining function 'nPELICAN'` for hours (h8 N=16 is the largest
flattened nanoPELICAN so far: 2,592 2->2 lanes vs 1,936 for the N=20 h4 toptag builds), the
per-stage split build gives the scheduler six small functions instead of one graph. It is gated
200/200 (logits and dots) on this export with `./build_local.sh split -DRUN_GOLDEN_GATE && ./tb_local_split`:

    vitis_hls -f build_prj.tcl "split=1 reset=1 csim=1 synth=1 golden=1"      # -> nPELICAN_split_prj/
    vivado -mode batch -source vivado_synth_split.tcl                          # -> vivado_synth_split{,_opt}.rpt

Split numbers carry stage-boundary registers (more FF, possibly more LUT/latency than the monolith);
label them "split" in any comparison table.

### RESULT (2026-10-21) — plain 5-class N=16 h8, U250 @ 5 ns

Reports: `reports/csynth_hls4ml5_h8n16.rpt`, `reports/vsynth_hls4ml5_h8n16_postsynth.rpt`
(post-`synth_design`) and `reports/vsynth_hls4ml5_h8n16_opt.rpt` (post-`opt_design`, the like-for-like number
against the Mixer; opt_design moved only -264 LUT / -3 FF).
Interface check: `model_input` 768 bits (16 x 4 x 12), `nobj` 5 bits, five 6-bit `model_out_k` ports.
(A first upload under this name was a stale N=32 h2 single-logit build — 1,536-bit input, 2,446 DSP —
kept at the workspace root as `*_STALE_n32_h2_nout1.rpt`; always check the `== Interface` table.)

| | csynth (Vitis 2023.2) | Vivado 2023.2 post-synth | **Vivado post-opt** |
|---|---|---|---|
| latency / II | **16 cyc (80 ns)**, II=1, est. 4.36 ns | — | — |
| LUT | 555,314 est. | 160,690 (160,652 logic + 38 SRL) | **160,426** (160,388 logic + 38 SRL) |
| FF | 76,400 est. | 49,280 | **49,277** |
| CARRY8 | — | 19,444 | 19,444 |
| DSP | 1,035 | 1,029 | **1,029** |
| BRAM | 0 | 0 | 0 |

DSP anatomy (csynth bind report): **684 dot products** (171 pairs x [`mul_12s_12s` + 3 `mac_mulsub`]),
**171 BN1 multiplies** (`mul_6s_19ns`: the `--bn-guard-bits 8` literal is 19 bits and crosses the DSP
threshold — the 10-bit default stays in LUTs, cf. the 16p `--bn-frac-bits 12` row at 769 DSP), 85 of the
2->2 MACs that HLS bound to `mac_muladd_6s_{4,5,6}ns` DSPs, and 95 in the 1/N̄ / BN2 / ncount rescales.
LUT (csynth estimate by source line, indicative only — Vivado lands at 0.29x): ReLU quantizer casts
(`nPELICAN.cpp:392`) 120k, the 2->2 MAC adder trees ~280k, 2->0 sums 48k, dot4 adds/casts 22k, T0 cast
17k, BN1 16k, aggregation adds 14k. Versus the 16p h2 toptag build (49.9k LUT / 769 DSP / 15 cyc), h8
costs 3.2x the LUT: the 2->2 MAC + ReLU scale with NHIDDEN, the dot stage does not.

## Deployment config for the 5-class model: Levers 8 + 9 + const beams (branch `hls4ml5-deploy`, 2026-10-22)

`hls4ml5-deploy` merges `bn1-fold` (Lever 8 BN1 fold + T0 ROM, Lever 9 Winograd dot, const beams,
sticky-bit biases, exact-float32 BN/norm literals) into `lever7-blockfp` (NOUT, jet spurion, head,
jet-quant-split). Same h8 checkpoint, re-exported with the merged loader (no guard-bit flags needed:
`--bias/--norm/--bn-guard-bits` are no-ops now); local gate 200/200 in every configuration, see
`reports/hls4ml5_h8_qat_gate/GATE_DEPLOY.md`. This is the toptag "best operating point" setup
(16p: 370 DSP / 47.3k LUT for h2) applied to the 5-class N=16 h8 export.

    git fetch && git checkout hls4ml5-deploy          # remote box
    export NPELICAN_PART=xcu250-figd2104-2L-e
    vitis_hls -f build_prj.tcl "reset=1 csim=1 synth=1 cosim=0 validation=0 export=0 vsynth=1 golden=1 winograd=1 const_beams=1"

(`bn1_rom=1` is the default.) Expect `GOLDEN SUMMARY: events=200 exact=200`, then
`nPELICAN_prj/solution/syn/report/nPELICAN_csynth.rpt` + `vivado_synth_opt.rpt` as before. Versus the
lever7-blockfp build (1,029 DSP / 160.4k LUT post-opt / 16 cyc): the 684 dot DSPs become 2 per particle
pair + 2 per particle (beam pairs fold to adders), the 171 BN1 multiplies disappear (T0 is a 64-entry
ROM, jmass/jdotp use 4 folded constants), the batch1 array is gone (FF down); the 2->2 MAC + ReLU LUTs
(the bulk) are untouched. Per-stage attribution: `split=1` works with the same flags.

### RESULT (2026-10-22) — deployment config, same checkpoint, U250 @ 5 ns

Reports: `reports/csynth_hls4ml5_h8n16_deploy.rpt`, `reports/vsynth_hls4ml5_h8n16_deploy_opt.rpt`
(post-`opt_design`). Interface check identical to the lever7-blockfp build (768-bit input, five 6-bit logits).

| | lever7-blockfp build (post-opt) | **deployment config** (post-opt) | change |
|---|---|---|---|
| latency / II | 16 cyc, II=1, est. 4.36 ns | **16 cyc**, II=1, est. 4.455 ns | 0 |
| LUT | 160,426 | **128,240** (127,340 logic + 900 SRL) | -20 % |
| FF | 49,277 | **32,191** | -35 % |
| CARRY8 | 19,444 | 12,674 | -35 % |
| DSP | 1,029 (csynth 1,035) | **692** (csynth 676) | -33 % |
| BRAM | 0 | 0 | |

DSP anatomy (csynth bind report, 676): 272 Winograd pair products (136 particle pairs x
[`am_submul` + `ama_addmuladd`]; the 35 beam pairs fold to adders under const beams), 32 per-particle
xi/eta products (`mul_12s_12s`), 0 BN1 (T0 = 168 `sparsemux` ROM lookups, the 3 beam-beam entries are
constants), 219 of the 2->2 MACs bound to `mac_muladd_6s_*` DSPs (85 in the lever7-blockfp build: the
sticky-bias accumulator init makes more mul+add chains DSP candidates), ~150 in the Lever-8 folded-BN1
/ BN2 / 1/N̄ rescales (exact-float32 constants, 2-6 DSP each). LUT (csynth estimate by source line):
ReLU quantizer casts 137k, 2->2 MAC ~150k, 2->0 sums 48k, Winograd dot adds/casts 31k, T0 ROM muxes 46k
(Vivado collapses these to a few LUT each) -> Vivado 128k total. Against the (E,px,py,pz) MLP-Mixer
(81,285 LUT / 28,595 FF / 13,396 CARRY8 / 0 DSP / 21 cyc): 1.58x LUT, 1.13x FF, 0.95x CARRY8, +692 DSP,
5 cycles faster, -2.4 pt accuracy. Untested follow-up lever: `BIND_OP variable=Tp op=mul impl=fabric`
(-219 DSP for a few k LUT).

## Momentum-grid (input bit-width) variants (2026-11-04)

Is the 12-bit momentum grid (`input_t <12,10>`: LSB 0.25 GeV, clip 512 GeV) what caps the 6-bit model under
boost? `variants/` holds one export per input grid of this same firmware — the committed 12-bit baseline,
two inference-only proxies (`<14,12>`, `<16,14>`; resource builds only, `csim=0 golden=0`) and, once the
retrains land, the real 14/16-bit checkpoints with golden vectors — plus `variants/use.sh <name>` to
install one and `build_prj.tcl tag=<name>` for a per-variant project dir (`nPELICAN_prj_<name>`,
`vivado -mode batch -source vivado_synth.tcl -tclargs _<name>` -> `vivado_synth_opt_<name>.rpt`). Full
recipe and table: `variants/README.md`. The loader now replays `pmu_clip_min` (PELICAN-nano
`--pmu-clip-min`, a floor on the momentum clip point) like `input_clip_min`.

### RESULT (2026-11-04) — 14/16-bit momentum input, width proxies, U250 @ 5 ns

Post-opt vs the 12-bit deployment build (128,240 LUT / 32,191 FF / 12,674 CARRY8 / 692 DSP / 16 cyc):
`proxy_in14` (`<14,12>`) **129,592 LUT (+1.05 %) / 32,649 FF (+1.4 %) / 12,676 CARRY8 / 692 DSP / 16 cyc**,
`proxy_in16` (`<16,14>`) **130,346 LUT (+1.64 %) / 32,913 FF (+2.2 %) / 13,116 CARRY8 / 692 DSP / 16 cyc**;
csynth clock estimates 4.37 / 4.46 ns. The wider grid is paid in the dot front-end's post-product LUTs only
(~530 LUT and ~180 FF per input bit); the Winograd pair products stay one DSP48E2 each up to 17-bit inputs.
With the `pmu16f2048` checkpoint this is the boost-robust deployment: 54.8 % rest / 45.8 % at |β| = 0.95 for
1.6 % more LUTs than the 12-bit model. Reports `reports/{csynth,vsynth}_hls4ml5_proxy_in{14,16}*.rpt`;
analysis in `reports/README.md` and `variants/README.md`.
`variants/proxy_in17` (`<17,15>`, 2026-11-05) is the next resource build: 17 bits is the last width at which the
Winograd pre-adder sums (18 bits) fit the DSP48E2's 18-bit port, so it should keep the 692 DSPs; the two 17-bit
retrains are exported and gated 200/200 (`variants/pmu17f4096`: LSB 1/16 @ 4096, `<17,13>`, 55.2 % rest / 48.7 % at
|β| = 0.95; `variants/pmu17f2048`: LSB 1/32 @ 2048, `<17,12>`, 54.5 % / 50.1 %) — build them with `csim=1 golden=1`
like `pmu16f2048`; the csynth instance list must show single-DSP `ama_addmuladd_17s_17s_18s_36s` pair products.
`variants/proxy_in18` (`<18,16>`, 2026-11-05) is the companion build one bit past the port: its 19-bit pre-adder sums
need a second DSP per pair product (expected +272 DSP, ~964 in Vivado), which is the price of the two 18-bit retrains
(`pmu18f4096`: LSB 1/32 @ 4096 = both 17-bit bits at once; `pmu18f2048`: LSB 1/64 @ 2048) — exported here as soon as
they finish training.
