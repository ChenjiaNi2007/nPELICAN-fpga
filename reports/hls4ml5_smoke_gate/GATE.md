# 5-class (NOUT=5, N=16, h=2, w6a6i6 pmu12) firmware gate -- 2026-10-06
checkpoint: PELICAN-nano/model/fwsmoke_h5n16_h2_qat_best.pt (2-epoch 30k-jet QAT smoke; accuracy irrelevant, exercises the datapath)
export : python model_loader.py --model ../PELICAN-nano/model/fwsmoke_h5n16_h2_qat_best.pt --quant --repo ../PELICAN-nano --out firmware/weights/weights.h
golden : (PELICAN-nano) .venv/bin/python scripts/export_golden.py --checkpoint model/fwsmoke_h5n16_h2_qat_best.pt --testfile data/hls4ml5_n16/test.h5 --num 200 --dump-events 3 --outdir ../nPELICAN-fpga/tb_data
gate   : ./build_local.sh -DRUN_GOLDEN_GATE && ./tb_local ; ./build_local.sh split -DRUN_GOLDEN_GATE && ./tb_local_split

## result (local g++ csim == Vitis csim, proven earlier)
GOLDEN SUMMARY: events=200 exact=200 mismatch=0 max_abs_delta=0 first_mismatch=-1
GOLDEN GATE: PASS (max_abs_delta=0 vs tol=0.001; 200/200 zero-tolerance exact)
DOTS-LEVEL SUMMARY: events=200 exact=200 mismatch=0 max_abs_delta=0 first_mismatch=-1
DOTS-LEVEL GATE: PASS (max_abs_delta=0 vs tol=0.0001; 200/200 zero-tolerance exact)
split: byte-identical results log (cmp)

## history of this gate
54/200  : NOUT generalization alone (raw Nobj up to 137 wrapped in the 5-bit nobj port; bias literals on the activation grid, b2=-0.0053 -> 0)
194/200 : + nobj clamp (TB + exporter), NOBJ_BITS from NPARTICLES2, biases typed on the MAC product grid
198/200 : + --bias-guard-bits 8 (mac2b_t end-add, mac0_t widened): removes exact half-LSB ties from bias snapping
200/200 : + --agg-guard-bits 8 (bn1out_t/acc widths): removes jdotp/jmass sum ties after the 1/Nbar rescale
