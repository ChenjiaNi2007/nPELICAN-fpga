# Gate record: plain 5-class nanoPELICAN N=16 h8, 6-bit QAT (`h5n16_h8_qat_s1`) — 2026-10-15

Checkpoint `PELICAN-nano/model/h5n16_h8_qat_s1_best.pt` (queue recipe, 20 ep, w6 a6 i6 pmu12,
`--freeze-scales-epoch 12`, seed 1; best = final epoch). Test (260k jets, `eval_hls4ml5.py`):
**52.22 % / macro AUC 81.54** (g/q/W/Z/t 79.8 / 84.3 / 78.1 / 77.4 / 88.1) vs float h8
55.5-55.9 % / 83.1-83.4 -> 6-bit drop 3.4-3.7 pt. This is the information-matched partner of the
(E,px,py,pz) MLP-Mixer (54.6 % QAT / 55.6 % float, 1,690 params) at 167 params.

Learned grids (loader printout): dots `ap_fixed<6,8>` (LSB 4 GeV^2, +-128), momenta
`ap_fixed<12,10>` (LSB 0.25 GeV, clip +-512 GeV; 25/200 golden events carry a component above
the clip, as in PyTorch), t2/relu `<6,2>`, t0 `<6,1>`, logits `<6,3>`.

Golden vectors: `export_golden.py --testfile data/hls4ml5_n16/test.h5 --num 200` (first 200 test
jets; `golden_jet.dat` from the jet+head export deleted — no jet port here).

| export | BN1 literal | local g++ gate (`./build_local.sh -DRUN_GOLDEN_GATE`) |
|---|---|---|
| default (`--bn-guard-bits 0`) | 576 / 2^13 (10 bits, rel. err 7e-4) | **182/200**, 18 events off by 1-2 output LSB (0.125-0.25); only 5 of them have clipped momenta -> not a saturation effect, the coarse literal |
| `--bn-guard-bits 8` (shipped) | 147562 / 2^21 (18 bits, err 1.7e-9) | **200/200 exact**, dots-level 200/200 |

The shipped `firmware/weights/` is the `--bn-guard-bits 8` export (wider BN1 multiplier: DSP /
timing cost per the Lever-8 measurements; same trade-off as the jet+head model, which went
199 -> 200/200 with it). Re-export command:

    ../PELICAN-nano/.venv/bin/python model_loader.py --model ../PELICAN-nano/model/h5n16_h8_qat_s1_best.pt \
        --quant --repo ../PELICAN-nano --out firmware/weights/weights.h --bn-guard-bits 8

Logs: `export_bnguard{0,8}.log`, `gate_bnguard{0,8}.log`.
