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
    variants/use.sh proxy_in14
    vitis_hls -f build_prj.tcl "reset=1 csim=0 synth=1 cosim=0 validation=0 export=0 vsynth=0 golden=0 winograd=1 const_beams=1 tag=proxy_in14"
    vivado -mode batch -source vivado_synth.tcl -tclargs _proxy_in14
    variants/use.sh --restore
    # -> nPELICAN_prj_proxy_in14/solution/syn/report/nPELICAN_csynth.rpt, vivado_synth_opt_proxy_in14.rpt

For the gated variants use `csim=1 golden=1` and expect `GOLDEN SUMMARY: events=200 exact=200`. Check the
csynth `== Interface` table first: `model_input` must be NPARTICLES x 4 x W bits (768 / 896 / 1024 for
W = 12 / 14 / 16), five 6-bit `model_out` ports, 5-bit `nobj`.

Status 2026-11-04 (all exports done; synthesis owed): the momentum grid is the whole QAT boost loss (float momenta
= boost-flat 55 %), and `pmu16f2048` is the quantized grid to deploy for a boost-robust model. Expected firmware
delta vs `pmu12`: `model_input` 768 -> 896 / 1024 bits, Winograd pair products 14x14 / 16x16 (one DSP each),
wider dot adder trees; the 2->2 / ReLU / 2->0 stages are unchanged. Numbers: csynth + `vivado_synth_opt_<name>.rpt`
per variant, to be added to the README RESULT tables.
