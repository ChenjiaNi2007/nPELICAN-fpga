# Gate on the REAL 24-bit jet+head checkpoint (q24j16_h4_jh_e40_s1, N=16 h=4, test acc 71.2%) -- 2026-10-07
export : python model_loader.py --model ../PELICAN-nano/model/q24j16_h4_jh_e40_s1_best.pt --quant --repo ../PELICAN-nano --out firmware/weights/weights.h
golden : export_golden.py --checkpoint model/q24j16_h4_jh_e40_s1_best.pt --testfile data/hls4ml5j_n16/test.h5 --num 200

GOLDEN SUMMARY: events=200 exact=0 mismatch=200 max_abs_delta=0.0010199546813964844 first_mismatch=0
GOLDEN GATE: FAIL (max_abs_delta=0.00102 vs tol=0.001; 0/200 zero-tolerance exact)
DOTS-LEVEL SUMMARY: events=200 exact=2 mismatch=198 max_abs_delta=3.4332275390625e-05 first_mismatch=0
DOTS-LEVEL GATE: PASS (max_abs_delta=3.43e-05 vs tol=0.0001; 2/200 zero-tolerance exact)
split == monolith (cmp)

Interpretation: 24-bit grids (output LSB 2^-21) are finer than PyTorch's float32 arithmetic noise, so zero-tolerance
exactness is not attainable against this reference (the toptag 24-bit gate had the same character: 133/200, max 6.3e-4).
DOTS-LEVEL (network isolated): max 3.4e-5 -> datapath verified. GOLDEN (incl. dot4): 1.02e-3, the documented float32
d_ij cancellation caveat, ~4x the toptag value because the jet spurion carries up to 5.3 TeV. The 6-bit synthetic
jet+head gate (GATE_JET_HEAD.md) is 200/200 bit-exact: on deployment grids the float32 noise is below the LSB.
