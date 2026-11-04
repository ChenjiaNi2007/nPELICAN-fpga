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
| `pmu14`, `pmu16`, `pmu14c2048`, `pmu16c2048` | from the retrained checkpoints | `h5n16_h8_qat_<name>_s1` | yes | the real models (added when training finishes) |

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
