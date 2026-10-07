# Gate on the REAL 6-bit wide-split jet+head checkpoint (qat6wj16_h4_jh_e40_s2: N=16 h=4, w6a6i6 pmu12, jet grids 10/16/20 bits; test acc 69.9%) -- 2026-10-07

## default export (DSP-friendly 13-bit BN1 literal; bias/agg/norm guard bits 8)
GOLDEN SUMMARY: events=200 exact=199 mismatch=1 max_abs_delta=0.1875 first_mismatch=176
GOLDEN GATE: FAIL (max_abs_delta=0.188 vs tol=0.001; 199/200 zero-tolerance exact)
DOTS-LEVEL SUMMARY: events=200 exact=199 mismatch=1 max_abs_delta=0.1875 first_mismatch=176
DOTS-LEVEL GATE: FAIL (max_abs_delta=0.188 vs tol=0.0001; 199/200 zero-tolerance exact)
split == monolith (cmp)
Residual (event 176): a jdotp tie -- exact 2.499983 LSB -> 2 (PyTorch); the 13-bit BN1 scale literal (rel err +6.6e-5,
capped for DSP/timing) gives 2.500067 -> 3. Documented trade-off (--bn-frac-bits cap / --bn-guard-bits).

## export with --bn-guard-bits 8
GOLDEN SUMMARY: events=200 exact=200 mismatch=0 max_abs_delta=0 first_mismatch=-1
GOLDEN GATE: PASS (max_abs_delta=0 vs tol=0.001; 200/200 zero-tolerance exact)
DOTS-LEVEL SUMMARY: events=200 exact=200 mismatch=0 max_abs_delta=0 first_mismatch=-1
DOTS-LEVEL GATE: PASS (max_abs_delta=0 vs tol=0.0001; 200/200 zero-tolerance exact)
split == monolith (cmp)
VERDICT: 200/200 bit-exact with --bn-guard-bits 8 (wider BN1 literal, DSP/timing cost per the Lever-8 measurements)

## history: 51/200 (loader did not replay the per-quantizer jet widths -> 6/6/12-bit jet types) -> 198/200 (fixed)
-> 199/200 (+ --norm-guard-bits 8: invnave2 rel err 1.4e-5 flipped a -9.500062-LSB tie) -> see VERDICT.
NOTE: --bn-frac-bits is a CAP (cannot widen); use --bn-guard-bits to widen.

export : python model_loader.py --model ../PELICAN-nano/model/qat6wj16_h4_jh_e40_s2_best.pt --quant --repo ../PELICAN-nano --out firmware/weights/weights.h [--bn-guard-bits 8]
golden : (PELICAN-nano) .venv/bin/python scripts/export_golden.py --checkpoint model/qat6wj16_h4_jh_e40_s2_best.pt --testfile data/hls4ml5j_n16/test.h5 --num 200 --dump-events 3 --outdir ../nPELICAN-fpga/tb_data
