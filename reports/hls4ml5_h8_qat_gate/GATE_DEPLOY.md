# Gate record: deployment config (Levers 8 + 9 + const beams) on the 5-class exports — 2026-10-22

Branch `hls4ml5-deploy` = `lever7-blockfp` (NOUT / jet spurion / head / jet-quant-split firmware)
merged with `bn1-fold` (Lever 8 BN1-fold + T0 ROM, Lever 9 Winograd dot, const beams, sticky-bit
biases, exact-float32 BN/norm literals, particle-count widths). The merged datapath is bn1-fold's;
the merged loader keeps every lever7-blockfp export feature. `--bias-guard-bits`, `--norm-guard-bits`
and `--bn-guard-bits` are no-ops now (the sticky-bit rule + Lever 8 make them unnecessary).

Local g++ gate (`./build_local.sh [split] <flags> && ./tb_local[_split]`), 200 golden events each,
logits AND dots level, zero tolerance:

| export | build flags | result |
|---|---|---|
| `h5n16_h8_qat_s1` (plain N=16 h8, NOUT=5) | `-DRUN_GOLDEN_GATE` (BN1 T0 ROM, runtime beams) | 200/200, dots 200/200 |
| same | `+ -DNPELICAN_WINOGRAD_DOT -DNPELICAN_CONST_BEAMS` (**deployment config**) | 200/200, dots 200/200 |
| same | `+ -DNPELICAN_WINOGRAD_DOT` | 200/200, dots 200/200 |
| same | `+ -DNPELICAN_NO_BN1_ROM` (arithmetic T0) | 200/200, dots 200/200 |
| same, split build | `split -DRUN_GOLDEN_GATE` | 200/200, dots 200/200 |
| same, split build | `split + winograd + const beams` | 200/200, dots 200/200 |
| `qat6wj16_h4_jh_e40_s2` (jet spurion + head h4, jet-quant-split) | `-DRUN_GOLDEN_GATE` (dotall_t 16 bits > 10: arithmetic T0 fallback) | 200/200, dots 200/200 |
| same | `+ -DNPELICAN_NO_BN1_ROM` | 200/200, dots 200/200 |
| same, split build | `split -DRUN_GOLDEN_GATE` | 200/200, dots 200/200 |

The jet+head export needed `--bn-guard-bits 8` for 200/200 on lever7-blockfp (199/200 default); with
Lever 8 + sticky biases it is 200/200 with no flags. Winograd is refused (#error) together with
`NPELICAN_JET_QUANT_SPLIT` (not wired for the jet dot populations).

Shipped `firmware/weights/`: the h8 export (`export_deploy_h8.log`): bias_t_gen sticky Fh=9 (BIAS_F=10),
bn1fold_t <25,1>, 64-entry T0 ROM, NPARTICLES 16 / NHIDDEN 8 / NOUT 5. Jet+head export log:
`export_deploy_jet_head.log` (not shipped; re-export from `64ddce8`'s golden set to reproduce).
