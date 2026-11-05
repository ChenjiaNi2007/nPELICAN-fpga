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
| `proxy_in17` | `<17,15>` (LSB 0.25, clip 16384) | same checkpoint, `--pmu-extra-int-bits 5` (2026-11-05) | no (ablation) | resource cost of a 17-bit input grid = the last width with one DSP per Winograd pair product (18-bit pre-adder sums fill the DSP48E2's 18-bit port; 18-bit inputs need a second DSP on each of the 272 pair products). Expected ~130.9k LUT / 692 DSP / 16 cyc |
| `proxy_in18` | `<18,16>` (LSB 0.25, clip 32768) | same checkpoint, `--pmu-extra-int-bits 6` (2026-11-05) | no (ablation) | resource cost of an 18-bit input grid = the first width PAST the DSP port: the 19-bit Winograd pre-adder sums no longer fit the DSP48E2's 18-bit B port, so each of the 272 pair products needs a second DSP (expected ~964 DSP in Vivado / ~948 csynth, +272 on 692/676) plus ~0.5k more LUT; measures what the two 18-bit retrains (`pmu18f4096`, `pmu18f2048`) would cost |
| `pmu17f4096` | `<17,13>` (LSB 1/16 GeV, clip 4096) | retrained `h5n16_h8_qat_pmu17f4096_s1` (`--pmu-fixed-clip 4096`, 2026-11-05; rest **55.2 %** / 83.3 = the float-momenta model, boost 55.0 -> 48.7) | **200/200** | the 16f2048 grid + one bit of RANGE (clipped jets at \|β\|=0.95: 11.8 -> 1.6 %); best at rest, never below 16f2048 at any β; now the dashed curve of `boost_n16_mixer_vs_npelican.png` |
| `pmu17f2048` | `<17,12>` (LSB 1/32 GeV, clip 2048) | retrained `h5n16_h8_qat_pmu17f2048_s1` (`--pmu-fixed-clip 2048`, 2026-11-05; rest 54.5 % / 83.0, boost 54.5 -> **50.1**) | **200/200** | the 16f2048 grid + one bit of RESOLUTION: flattest quantized curve (54.1 % at 0.8); dot_t `<6,9>` (8 GeV² LSB) in both 17-bit models vs `<6,8>` at 16 bits, T0 ROM still 64 entries |
| `pmu18f4096` | `<18,13>` (LSB 1/32 GeV, clip 4096) | retrained `h5n16_h8_qat_pmu18f4096_s1` (`--pmu-fixed-clip 4096`, 2026-11-05; rest 54.9 % / 83.0, boost 55.1 -> 52.7) | **200/200** | both 17-bit bits at once; near boost-flat (54.7 % at 0.8); dot_t `<6,10>` (16 GeV² LSB), T0 ROM still 64 entries; pays the second DSP per pair product that `proxy_in18` measures |
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
For the 18-bit build (`proxy_in18`) the opposite is expected: the 19-bit pre-adder sums overflow the B port, so
the instance list should show a two-DSP multiplier per pair product (a `mul_19s_19s_38_*` or an
`ama_addmuladd_*` + `mul_*` pair) and the DSP total should rise by ~272 (csynth ~948, Vivado ~964); if Vitis
instead keeps the DSP count and grows the LUTs, it has moved one pre-adder into fabric, which is the number to
quote either way.

Status 2026-11-04: all exports done; the momentum grid is the whole QAT boost loss (float momenta = boost-flat
55 %), and `pmu16f2048` is the quantized grid to deploy for a boost-robust model. The two width proxies are
synthesized (deployment config, U250 @ 5 ns, `csim=0 golden=0`; reports in `../reports/`):

| build | `input_t` | `model_input` | csynth (est. clock, DSP, FF, LUT est.) | Vivado post-opt LUT (logic + SRL) | FF | CARRY8 | DSP | latency |
|---|---|---|---|---|---|---|---|---|
| `pmu12` (2026-10-22) | `<12,10>` | 768 b | 4.455 ns, 676, 58,880, 500,777 | 128,240 (127,340 + 900) | 32,191 | 12,674 | 692 | 16 cyc, II=1 |
| `proxy_in14` | `<14,12>` | 896 b | 4.370 ns, 676, 60,476, 504,521 | **129,592** (+1,352, +1.05 %) | 32,649 (+1.4 %) | 12,676 | **692** | 16 cyc, II=1 |
| `proxy_in16` | `<16,14>` | 1024 b | 4.455 ns, 676, 62,074, 508,329 | **130,346** (+2,106, +1.64 %) | 32,913 (+2.2 %) | 13,116 (+3.5 %) | **692** | 16 cyc, II=1 |

(post-synth, before `opt_design`: 129,773 / 130,527 LUT.) So the 16-bit grid that lifts the model from 52.2 to
54.8 % at rest and from 34.9 to 45.8 % at |β| = 0.95 costs 1.6 % more LUTs, 2.2 % more FFs, no DSPs and no
cycles: `pmu16f2048` is 1.60x the 4-feature Mixer's LUTs (was 1.58x) at +0.2 pt accuracy instead of -2.4.
The LUT growth is the dot front-end's post-product arithmetic (saturation compares of the dot cast and the
Winograd correction subtractions, +4 bits per +2 input bits); the pre-adder sums live inside the DSP48E2
(`ama_addmuladd_16s_16s_17s_34s`), which holds one pair product per DSP up to 17-bit inputs — an 18-bit or
wider grid (or the 24-bit `pmufloat` type) would need a second DSP for each of the 272 pair products
(`proxy_in18` measures exactly that step; synthesis owed — the two 18-bit retrains `pmu18f4096` / `pmu18f2048` are
the near-boost-flat models, 52.7 / 53.0 % at |β| = 0.95, that would pay it). Per
input bit over 12 -> 16: ~530 LUT and ~180 FF (the two 2-bit steps are uneven, +1,352 vs +754 LUT, Vivado
having moved adders between LUT6s and CARRY8 chains at 16 bits). A proxy is resource-equivalent to the
retrained variant of the same width (`pmu14`/`pmu14f2048`, `pmu16`/`pmu16f2048`) up to constant-dependent
noise (BN2/bias literals, T0 ROM contents); the gated builds of the retrained checkpoints (`csim=1 golden=1`)
are still owed if an exact deployable figure is needed. The uploaded 16-bit csynth report is cut at 6 MiB
(summary/timing/instances intact, `== Interface` table missing; stored as `csynth_hls4ml5_proxy_in16_TRUNCATED.rpt`).
