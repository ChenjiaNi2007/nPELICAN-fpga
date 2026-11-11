# Exported model variants (momentum-grid ladder, 2026-11-04)

One directory per export of the SAME firmware: `weights.h` + `types_generated.h` (model_loader.py),
golden vectors (`golden_*.dat`, export_golden.py, first 200 hls4ml 5-class test jets) when a PyTorch
reference exists, `EXPORT.cmd` / `EXPORT.log` (how it was made) and `GATE.txt` (local g++ gate with the
deployment flags). Install one with `variants/use.sh <name>` (copies into `firmware/weights/` and
`tb_data/`), build with `tag=<name>`, restore with `variants/use.sh --restore` before pulling.

Question behind the ladder (professor, 2026-11-04): the 6-bit h8 nanoPELICAN loses 52 -> 35 % accuracy
under boost while its float version is flat; is the 12-bit momentum grid (LSB 0.25 GeV, clip +-512 GeV,
firmware `input_t ap_fixed<12,10>`) the constraint? PyTorch side: `mlpmixer/hls4ml5/pmu_bits_ablation.sh`
(inference-only grids on the 12-bit checkpoint) and `PELICAN-nano/scripts/hls4ml5_pmu_bits.sh`
(retrains at 14/16 bits, with and without `--pmu-clip-min 2048`). Firmware side = this directory.

| name | input_t | source | golden | purpose |
|---|---|---|---|---|
| `pmu12` | `<12,10>` (LSB 0.25 GeV, clip 512) | `h5n16_h8_qat_s1` (= committed `firmware/weights`) | yes, 200/200 | baseline = deployment-config result (128,240 LUT / 692 DSP) |
| `proxy_in14` | `<14,12>` (LSB 0.25, clip 2048) | same checkpoint, `--pmu-extra-int-bits 2` | no (ablation) | resource cost of a 14-bit input grid, available before the retrains finish |
| `proxy_in16` | `<16,14>` (LSB 0.25, clip 8192) | same checkpoint, `--pmu-extra-int-bits 4` | no (ablation) | resource cost of a 16-bit input grid |
| `proxy_in17` | `<17,15>` (LSB 0.25, clip 16384) | same checkpoint, `--pmu-extra-int-bits 5` (2026-11-05) | no (ablation) | resource cost of a 17-bit input grid = the last width with one DSP per Winograd pair product (18-bit pre-adder sums fill the DSP48E2's 18-bit port). **Synthesized 2026-11-05: 131,707 LUT / 33,149 FF / 13,116 CARRY8 / 692 DSP / 16 cyc** (+1.04 % LUT on 16 bits; single-DSP `ama_addmuladd_17s_17s_18s_36s` confirmed) |
| `proxy_in18` | `<18,16>` (LSB 0.25, clip 32768) | same checkpoint, `--pmu-extra-int-bits 6` (2026-11-05) | no (ablation) | resource cost of an 18-bit input grid = the first width PAST the DSP port: the 19-bit Winograd pre-adder sums no longer fit the DSP48E2's 18-bit B port. **Synthesized 2026-11-05: 143,863 LUT / 37,879 FF / 14,608 CARRY8 / 692 DSP / 15 cyc** — NO extra DSPs: Vitis keeps one DSP per pair product (`mul_19s_19s_38_1_1` instances) and moves the pre-adds plus a correction add into fabric, +12,156 LUT (+9.2 %) / +4,730 FF (+14.3 %) / +1,492 CARRY8 over 17 bits = the cost of the two 18-bit retrains (`pmu18f4096`, `pmu18f2048`) |
| `pmu17f4096` | `<17,13>` (LSB 1/16 GeV, clip 4096) | retrained `h5n16_h8_qat_pmu17f4096_s1` (`--pmu-fixed-clip 4096`, 2026-11-05; rest **55.2 %** / 83.3 = the float-momenta model, boost 55.0 -> 48.7) | **200/200** | the 16f2048 grid + one bit of RANGE (clipped jets at \|β\|=0.95: 11.8 -> 1.6 %); best at rest, never below 16f2048 at any β; now the dashed curve of `boost_n16_mixer_vs_npelican.png` |
| `pmu17f2048` | `<17,12>` (LSB 1/32 GeV, clip 2048) | retrained `h5n16_h8_qat_pmu17f2048_s1` (`--pmu-fixed-clip 2048`, 2026-11-05; rest 54.5 % / 83.0, boost 54.5 -> **50.1**) | **200/200** | the 16f2048 grid + one bit of RESOLUTION: flattest quantized curve (54.1 % at 0.8); dot_t `<6,9>` (8 GeV² LSB) in both 17-bit models vs `<6,8>` at 16 bits, T0 ROM still 64 entries |
| `pmu18f4096` | `<18,13>` (LSB 1/32 GeV, clip 4096) | retrained `h5n16_h8_qat_pmu18f4096_s1` (`--pmu-fixed-clip 4096`, 2026-11-05; rest 54.9 % / 83.0, boost 55.1 -> 52.7) | **200/200** | both 17-bit bits at once; near boost-flat (54.7 % at 0.8); dot_t `<6,10>` (16 GeV² LSB), T0 ROM still 64 entries; pays the +9 % LUT / +14 % FF of the 18-bit input that `proxy_in18` measured (no extra DSPs) |
| `pmu18f2048` | `<18,12>` (LSB 1/64 GeV, clip 2048) | retrained `h5n16_h8_qat_pmu18f2048_s1` (`--pmu-fixed-clip 2048`, 2026-11-05; rest **55.1 %** / 83.1, boost 55.3 -> **53.0**) | **200/200** | two bits of RESOLUTION: the flattest quantized curve (54.7 % at 0.8, 53.0 at 0.95 = 1.8 pt under the float-momenta model, +3 on the best 17-bit curve); resolution beats range again, so at >= 17 bits the LSB is the bit to buy; dot_t `<6,10>` |
| `pmu14` | `<14,10>` (LSB 1/16 GeV, clip 512) | retrained `h5n16_h8_qat_pmu14_s1` (clip left to the learned scale; rest 53.4 % / 82.5, boost 53.1 -> 38.2) | 197/200, see note | real 14-bit model: training keeps the 512 GeV clip and spends the bits on the LSB |
| `pmu16` | `<16,10>` (LSB 1/64 GeV, clip 512) | retrained `h5n16_h8_qat_pmu16_s1` (rest 54.0 % / 82.6, boost 53.9 -> 40.4) | 186/200 (<= 1 LSB) | real 16-bit model, same pattern |
| `pmu14f2048` | `<14,12>` (LSB 0.25 GeV, clip 2048) | retrained `h5n16_h8_qat_pmu14f2048_s1` (`--pmu-fixed-clip 2048`; rest 53.9 % / 82.3, boost 53.8 -> 40.4) | **200/200** | 12-bit LSB at 4x the range |
| `pmu16f2048` | `<16,12>` (LSB 1/16 GeV, clip 2048) | retrained `h5n16_h8_qat_pmu16f2048_s1` (`--pmu-fixed-clip 2048`; rest 54.8 % / 83.1, boost 54.8 -> 45.8) | 198/200 (dots-level 200/200) | the best quantized grid found: rest = float-momenta model |
| `pmufloat` | `<24,13>` (loader fallback for float momenta) | retrained without a momentum quantizer (`h5n16_h8_qat_pmufloat_s1`, rest 55.1 % / 83.0, boost-flat 55.2 -> 54.8) | 195/200 (dots-level 200/200) | float-momenta upper bound; the 24-bit input_t only approximates it |

Gate note for the wider grids: a mismatch of 1-3 output LSBs on a few events is the PyTorch reference, not the
firmware. With LSB <= 1/16 GeV the squared momenta (E² ≈ 1.6e5 GeV² for a 400 GeV constituent) need more than
float32's 24 mantissa bits, so PyTorch's float32 dot differs from the exact fixed-point dot by ~1e-2 GeV² and
can land on the other side of a dot-quantizer rounding boundary (pmu14 event 28: m² exact 13.984 -> 12, PyTorch
-> 16). The firmware arithmetic is exact; a float64 reference would gate 200/200. `pmu14c2048` (learned scale
with a `--pmu-clip-min` floor) was dropped: Brevitas applies that floor in eval mode only, so the model trained
on a different grid than it evaluates on.

Resource-wise a proxy is the retrained model of the same input WIDTH: the dot front-end's multipliers and
adders are sized by `input_t`'s width, not by where its binary point sits, and every later stage is
fixed-width (6-bit quantizers). A proxy is NOT deployable (its logits differ from any trained model on the
jets the 12-bit clip touched: 175/200 exact vs the 12-bit golden set), so build it with `csim=0 golden=0`.

Remote recipe (deployment config, U250, 5 ns), per variant:

    git pull                                   # branch hls4ml5-deploy
    export NPELICAN_PART=xcu250-figd2104-2L-e
    name=proxy_in17                            # then proxy_in18; gated (csim=1 golden=1): pmu17f4096 pmu17f2048 pmu18f4096 pmu18f2048
    variants/use.sh $name
    vitis_hls -f build_prj.tcl "reset=1 csim=0 synth=1 cosim=0 validation=0 export=0 vsynth=0 golden=0 winograd=1 const_beams=1 tag=$name"
    vivado -mode batch -source vivado_synth.tcl -tclargs _$name
    variants/use.sh --restore
    # -> nPELICAN_prj_$name/solution/syn/report/nPELICAN_csynth.rpt, vivado_synth_opt_$name.rpt
    # (proxy_in14 / proxy_in16 were run the same way on 2026-11-04; results below)

For the gated variants use `csim=1 golden=1` and expect `GOLDEN SUMMARY: events=200 exact=200`. Check the
csynth `== Interface` table first: `model_input` must be NPARTICLES x 4 x W bits (768 / 896 / 1024 / 1088 / 1152
for W = 12 / 14 / 16 / 17 / 18), five 6-bit `model_out` ports, 5-bit `nobj`. For the 17-bit build also check the
instance list: the pair products must appear as single-DSP `ama_addmuladd_17s_17s_18s_36s` blocks (csynth DSP
total 676, Vivado 692, as for 12-16 bits); any `mul_18s_18s`/two-DSP pattern there means the port limit bit.
For the 18-bit build (`proxy_in18`, synthesized 2026-11-05) the 19-bit pre-adder sums overflow the B port and
Vitis takes the second route: the DSP count stays at 676 / 692, the 272 pair products appear as `mul_19s_19s_38_1_1`
instances (one DSP each, csynth DSP-expression modules 509 -> 237, instance DSPs 167 -> 439) with the pre-adds and
a correction add in fabric, and the cost lands in LUTs/FFs (+9.2 % / +14.3 % over 17 bits) and one cycle less
latency (15).

Status 2026-11-04: all exports done; the momentum grid is the whole QAT boost loss (float momenta = boost-flat
55 %), and `pmu16f2048` is the quantized grid to deploy for a boost-robust model. The two width proxies are
synthesized (deployment config, U250 @ 5 ns, `csim=0 golden=0`; reports in `../reports/`):

| build | `input_t` | `model_input` | csynth (est. clock, DSP, FF, LUT est.) | Vivado post-opt LUT (logic + SRL) | FF | CARRY8 | DSP | latency |
|---|---|---|---|---|---|---|---|---|
| `pmu12` (2026-10-22) | `<12,10>` | 768 b | 4.455 ns, 676, 58,880, 500,777 | 128,240 (127,340 + 900) | 32,191 | 12,674 | 692 | 16 cyc, II=1 |
| `proxy_in14` | `<14,12>` | 896 b | 4.370 ns, 676, 60,476, 504,521 | **129,592** (+1,352, +1.05 %) | 32,649 (+1.4 %) | 12,676 | **692** | 16 cyc, II=1 |
| `proxy_in16` | `<16,14>` | 1024 b | 4.455 ns, 676, 62,074, 508,329 | **130,346** (+2,106, +1.64 %) | 32,913 (+2.2 %) | 13,116 (+3.5 %) | **692** | 16 cyc, II=1 |
| `proxy_in17` (2026-11-05) | `<17,15>` | 1088 b | 4.455 ns, 676, 62,877, 510,217 | **131,707** (+3,467, +2.70 %) | 33,149 (+3.0 %) | 13,116 (+3.5 %) | **692** | 16 cyc, II=1 |
| `proxy_in18` (2026-11-05) | `<18,16>` | 1152 b | 4.455 ns, 676, 57,358, 526,929 | **143,863** (+15,623, +12.2 %) | 37,879 (+17.7 %) | 14,608 (+15.3 %) | **692** | **15 cyc**, II=1 |

(post-synth, before `opt_design`: 129,773 / 130,527 / 131,881 / 144,103 LUT.) So the 16-bit grid that lifts the model from 52.2 to
54.8 % at rest and from 34.9 to 45.8 % at |β| = 0.95 costs 1.6 % more LUTs, 2.2 % more FFs, no DSPs and no
cycles: `pmu16f2048` is 1.60x the 4-feature Mixer's LUTs (was 1.58x) at +0.2 pt accuracy instead of -2.4.
The LUT growth is the dot front-end's post-product arithmetic (saturation compares of the dot cast and the
Winograd correction subtractions, +4 bits per +2 input bits); the pre-adder sums live inside the DSP48E2
(`ama_addmuladd_16s_16s_17s_34s` / `17s_17s_18s_36s`), which holds one pair product per DSP up to 17-bit inputs.
At 18 bits (`proxy_in18`, measured) Vitis does NOT spend a second DSP: it keeps one DSP48E2 per pair product
(`mul_19s_19s_38_1_1`) and moves the pre-adds plus a correction add into fabric, so the step 17 -> 18 costs
+12,156 LUT (+9.2 %), +4,730 FF (+14.3 %) and +1,492 CARRY8 for the same 692 DSPs and one cycle less (15) —
the price of the near-boost-flat `pmu18f4096` / `pmu18f2048` models (52.7 / 53.0 % at |β| = 0.95). Per
input bit over 12 -> 16: ~530 LUT and ~180 FF (the two 2-bit steps are uneven, +1,352 vs +754 LUT, Vivado
having moved adders between LUT6s and CARRY8 chains at 16 bits). A proxy is resource-equivalent to the
retrained variant of the same width (`pmu14`/`pmu14f2048`, `pmu16`/`pmu16f2048`) up to constant-dependent
noise (BN2/bias literals, T0 ROM contents); the gated builds of the retrained checkpoints (`csim=1 golden=1`)
are still owed if an exact deployable figure is needed. The uploaded 16-bit csynth report is cut at 6 MiB
(summary/timing/instances intact, `== Interface` table missing; stored as `csynth_hls4ml5_proxy_in16_TRUNCATED.rpt`).

## Weight / activation / dot widths below 6 bits (2026-11-07, `lowbits_*`)

Same firmware, momentum grid fixed at the best point above (`pmu18f2048`: 18 bits, LSB 1/64 GeV, clip 2048,
`input_t <18,12>`), retrained with the Brevitas widths lowered (`PELICAN-nano/scripts/hls4ml5_lowbits.sh`,
recipe otherwise identical to the 18-bit retrain, seed 1): w = `--weight-bit-width` (2->2 and 2->0 weights),
a = `--act-bit-width` (post-aggregation tensors `t2_t`, ReLU `relu_t`, 2->0 values, logits `out_t`),
i = `--input-bit-width` (the dot quantizer `dot_t`). Every width is read from the checkpoint by
model_loader.py, so each variant carries its own `types_generated.h`. Accuracies: rest = 260k-jet test,
boost = 20k jets x 8 directions (`mlpmixer/hls4ml5/results/pmu_bits/boost_n16_npelican_h8_qat_lowbits_<tag>.csv`).

| name | w / a / i | `dot_t` / `t2_t` / `relu_t` / `out_t` | source checkpoint | rest acc | acc at \|β\| = 0.95 | golden |
|---|---|---|---|---|---|---|
| (`pmu18f2048`) | 6 / 6 / 6 | `<6,10>` / `<6,0>`... | `h5n16_h8_qat_pmu18f2048_s1` | 55.11 % | 53.0 % | 200/200 |
| `lowbits_w4a6i6` | 4 / 6 / 6 | `<6,9>` / `<6,0>` / `<6,1>` / `<6,3>` | `h5n16_h8_qat_p18_w4a6i6_s1` | 54.58 % | 51.8 % | 199/200 (one event 3 LSB off already at the dots level = the float32-dot reference note above; firmware exact) |
| `lowbits_w6a4i6` | 6 / 4 / 6 | `<6,10>` / `<4,2>` / `<4,2>` / `<4,2>` | `h5n16_h8_qat_p18_w6a4i6_s1` | 54.25 % | 52.2 % | 200/200 |
| `lowbits_w4a4i6` | 4 / 4 / 6 | `<6,11>` / `<4,1>` / `<4,2>` / `<4,2>` | `h5n16_h8_qat_p18_w4a4i6_s1` | 54.19 % | 50.1 % | 200/200 |
| `lowbits_w5a5i5` | 5 / 5 / 5 | `<5,10>` / `<5,1>` / `<5,2>` / `<5,3>` | `h5n16_h8_qat_p18_w5a5i5_s1` | 54.33 % | 48.5 % | 200/200 |
| `lowbits_w4a4i4` | 4 / 4 / 4 | `<4,10>` / `<4,1>` / `<4,2>` / `<4,3>` | `h5n16_h8_qat_p18_w4a4i4_s1` | 52.32 % | 47.3 % | 200/200 |

Reading: weights to 4 bits cost 0.5 pt at rest, activations to 4 bits 0.9 pt, both together 0.9 pt at rest
and 2.9 pt at |β| = 0.95; the dot quantizer is the sensitive one (i5 costs ~3 pt of boost flatness, i4 another
2.8 pt at rest), so i stays at 6. `lowbits_w4a4i6` is the candidate low-bit operating point (nominal 2->2 lane
product (4 x 4) / (6 x 6) = 0.44 of the 6-bit cost), `lowbits_w4a6i6` the conservative one (0.67). What moves in
hardware: the 2,592-lane 2->2 dense + ReLU (constant products ~ w x a bits, adder trees ~ a bits), the lane
registers (a bits), and possibly the ~200 6-bit x 6-bit 2->2 products that Vitis binds to DSPs in the 18-bit build
(4-bit products normally stay in fabric). The 272 momentum pair-product DSPs are untouched. Synthesis owed; the
same remote recipe as above with `csim=1 golden=1` (expect `exact=200`, 199 for w4a6i6) and the 18-bit interface
check (`model_input` 1152 bits; five `model_out` ports of the variant's a width).
