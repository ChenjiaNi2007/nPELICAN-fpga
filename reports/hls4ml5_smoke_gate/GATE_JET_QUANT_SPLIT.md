# Jet quantizer split (--jet-quant-split) firmware gate (SYNTHETIC checkpoint) -- 2026-10-07

Datapath gate for PELICAN-nano `QuantConfig.jet_quant_split`: with the jet spurion at slot 2,
d[i,j] (i,j != 2) -> `input_quant` (`dot_t`), d[2,j]/d[i,2] -> `input_quant_jet` (`dotj_t`),
d[2,2] = m_jet^2 -> `input_quant_mjet` (`dotm_t`); jet momenta -> `pmu_quant_jet` (`jet_t`).
Firmware: jet row kept in `jet_t pj[4]`; `dot4j(input_t[4], jet_t[4], dotj_t&)` and
`dot4m(jet_t[4], dotm_t&)` round once onto the population grid, then widen exactly into the
`dotall_t dots[]` container (I = max I, F = max F). Without the split every new type aliases
`dot_t`/`input_t` (loader output, or the `NPELICAN_JET_TYPES_GENERATED` fallback in nPELICAN.h).

checkpoint: synthetic_sj.pt (scratchpad; sj/mk_sj.py = jh/mk_jh.py + jet_quant_split=True) =
  PELICANNano(4, n_out=5, head_hidden=16, batchnorm='b', activation='relu',
              quant_config=QuantConfig(enabled=True, po2_scales=True, weight_bit_width=6,
              act_bit_width=6, input_bit_width=6, pmu_bit_width=12, jet_quant_split=True))
  random agg_2to0 / head biases (0.1*randn), ONE training-mode forward on 256 add_jet events of
  data/hls4ml5j_n16/test.h5; args jet_quant_split=True, add_jet=True, nobj=16, n_out=5.
  Learned scales (all differ, so the gate exercises the split):
    input_quant 2^6, input_quant_jet 2^8, input_quant_mjet 2^11, pmu_quant 2^0, pmu_quant_jet 2^1

export : python model_loader.py --model <synthetic_sj.pt> --quant --repo ../PELICAN-nano --out firmware/weights/weights.h
golden : (PELICAN-nano) .venv/bin/python <scratch>/sj/export_golden_split.py --checkpoint <synthetic_sj.pt> \
           --testfile data/hls4ml5j_n16/test.h5 --num 200 --dump-events 3 --outdir ../nPELICAN-fpga/tb_data
gate   : ./build_local.sh -DRUN_GOLDEN_GATE && ./tb_local ; ./build_local.sh split -DRUN_GOLDEN_GATE && ./tb_local_split

TOOLING GAP (PELICAN-nano, not edited here): scripts/export_golden.py captures golden_dots /
the stage-dump `dots:` line with a forward hook on `model.input_quant`. Under the split that
module is fed the Gram matrix with the jet row/col ZEROED, so plain export_golden.py writes
row/col 2 = 0. Result with the plain export: GOLDEN 200/200 (logits are right) but DOTS-LEVEL
6/200 (wrong injected dots). export_golden_split.py is a wrapper that re-fires input_quant's
hooks with the full `_input_quant_split` output (logits byte-identical to the plain export).
Fix upstream: hook the split output (or `forward(covariance_test=True)['inputs']`).

generated (types_generated.h):
  typedef ap_fixed<6, 12, AP_RND_CONV, AP_SAT> dot_t;      // input_quant 2^6
  typedef ap_fixed<6, 14, AP_RND_CONV, AP_SAT> dotj_t;     // input_quant_jet 2^8
  typedef ap_fixed<6, 17, AP_RND_CONV, AP_SAT> dotm_t;     // input_quant_mjet 2^11
  typedef ap_fixed<11, 17, AP_RND_CONV, AP_SAT> dotall_t;  // I = max I, F = max F
  typedef ap_fixed<12, 13, AP_RND_CONV, AP_SAT> jet_t;     // pmu_quant_jet 2^1
  typedef ap_fixed<12, 12, AP_RND_CONV, AP_SAT> input_t;   // pmu_quant 2^0
  #define NPELICAN_JET_QUANT_SPLIT 1 (weights.h #errors if absent); bn1out_t / BN_F sized from dotall_t.

## result (local g++ csim)
GOLDEN SUMMARY: events=200 exact=200 mismatch=0 max_abs_delta=0 first_mismatch=-1
GOLDEN GATE: PASS (max_abs_delta=0 vs tol=0.001; 200/200 zero-tolerance exact)
DOTS-LEVEL SUMMARY: events=200 exact=200 mismatch=0 max_abs_delta=0 first_mismatch=-1
DOTS-LEVEL GATE: PASS (max_abs_delta=0 vs tol=0.0001; 200/200 zero-tolerance exact)
split: byte-identical results log and stage dump (cmp) vs the monolith

## meaningfulness checks
- split is load-bearing: same goldens, header edited to `typedef dot_t dotj_t/dotm_t` -> GOLDEN 22/200.
- 27 distinct 5-logit rows over 200 events.

## regressions (same session)
- legacy NOUT=1 byte-identity (old SPS weights + goldens, pre-split header -> nPELICAN.h fallback):
  summaries, results logs and stage dumps cmp-identical to baseline_nout1 (monolith + split).
- synthetic_jh.pt (jet + head, no split) re-exported with the new loader: aliases emitted,
  GOLDEN 200/200, DOTS-LEVEL 200/200, split identical; results log + stage dump cmp-identical
  to the GATE_JET_HEAD.md run.
