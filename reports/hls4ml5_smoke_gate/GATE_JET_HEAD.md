# Jet spurion + nonlinear head firmware gate (SYNTHETIC checkpoint) -- 2026-10-07

Datapath gate for `--add-jet` (NSPURIONS=3, `jet_input[4]` port -> spurion slot 2) and
`--head-hidden K` (2->0 mixes to K channels -> ReLU quant `relu0_t` -> K->NOUT head MAC).
No trained jet/head checkpoint existed yet, so this uses a SYNTHETIC one: accuracy is
irrelevant (AUC 0.78 on 200 file-order events), it only exercises the datapath.

checkpoint: synthetic_jh.pt (scratchpad; mk_jh.py) =
  PELICANNano(4, n_out=5, head_hidden=16, batchnorm='b', activation='relu',
              quant_config=QuantConfig(enabled=True, po2_scales=True, weight_bit_width=6,
              act_bit_width=6, input_bit_width=6, pmu_bit_width=12))
  agg_2to0.mixing.bias and head.bias set to 0.1*randn (so the guard-bit bias paths are
  exercised), then ONE training-mode forward on 256 events of data/hls4ml5j_n16/test.h5
  collated with add_jet=True; args: add_jet=True, head_hidden=16, nobj=16, n_out=5,
  n_hidden=4, datadir='data/hls4ml5j_n16'. Note: the 6-bit input quantizer (scale 2^10)
  saturates the jet dots -- irrelevant for a datapath gate.

export : python model_loader.py --model <synthetic_jh.pt> --quant --repo ../PELICAN-nano --out firmware/weights/weights.h
golden : (PELICAN-nano) .venv/bin/python scripts/export_golden.py --checkpoint <synthetic_jh.pt> --testfile data/hls4ml5j_n16/test.h5 --num 200 --dump-events 3 --outdir ../nPELICAN-fpga/tb_data
         (writes golden_pmu/nobj/jet/logits/dots + golden_stage_dump.txt; dots are (16+3)^2 = 361/line)
gate   : ./build_local.sh -DRUN_GOLDEN_GATE && ./tb_local ; ./build_local.sh split -DRUN_GOLDEN_GATE && ./tb_local_split

generated (types_generated.h):
  #define NSPURIONS 3 ; #define NPELICAN_JET_SPURION 1 ; #define NPELICAN_HEAD 16 ; #define N2TO0_OUT 16
  relu0_t     = ap_fixed<6, 2, AP_RND_CONV, AP_SAT>   (agg_2to0.act_layer, scale 2^-4)
  wh_gen_t    = ap_fixed<6, 1, AP_RND_CONV, AP_SAT>   (head weights, scale 2^-5)
  biash_t_gen = ap_fixed<19, 2, AP_RND_CONV, AP_SAT>  (F = 5+4+G, G=8)
  mach_t      = ap_fixed<25, 8>                       (I = 1+2+ceil(log2 17), F = 17)
  input_t     = ap_fixed<12, 13, ...>                 (trained pmu grid; |Pjet|max 5336 GeV)

## result (local g++ csim)
GOLDEN SUMMARY: events=200 exact=200 mismatch=0 max_abs_delta=0 first_mismatch=-1
GOLDEN GATE: PASS (max_abs_delta=0 vs tol=0.001; 200/200 zero-tolerance exact)
DOTS-LEVEL SUMMARY: events=200 exact=200 mismatch=0 max_abs_delta=0 first_mismatch=-1
DOTS-LEVEL GATE: PASS (max_abs_delta=0 vs tol=0.0001; 200/200 zero-tolerance exact)
split: byte-identical results log and stage dump (cmp) vs the monolith

## meaningfulness checks
- logits vary: 54 distinct 5-logit rows over 200 events.
- event-0 stage dump: dots, T0, Tp, R, Rq all equal PyTorch exactly (fw `Rp:` prints the
  pre-output_quant head MAC Hp, golden `Rp:` the rounded logit -- same convention as the
  no-head `Rp:` line).
- jet is load-bearing: same binary with golden_jet.dat zeroed -> exact=1/200.

## regressions (same session)
- legacy NOUT=1 byte-identity (old SPS weights + goldens): summaries, results logs and
  stage dump `cmp`-identical to the pre-change baseline (monolith + split).
- 5-class smoke ckpt fwsmoke_h5n16_h2_qat_best.pt re-exported with the new loader
  (NSPURIONS=2, no head): 200/200 GOLDEN and DOTS-LEVEL, split identical.
