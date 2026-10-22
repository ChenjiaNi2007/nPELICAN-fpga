# Pre-exported header sets per particle count

Generated 2026-09-28 by `model_loader.py --quant --nparticles N` from
`PELICAN-nano/model/cap_h2_qatf12_lr0p0025_e20_s1_best.pt` (h=2, 6/6/6, pmu-12), for resource
builds on a synthesis box that has neither the checkpoint nor Brevitas. `weights.h` is
byte-identical in every set; `types_generated.h` differs only in `NPELICAN_NPARTICLES` and the
accumulator headroom (H1/H2) it implies. Select one with `./select_nparticles.sh N`, build with
`vitis_hls -f build_prj.tcl "... nparticles=N"`, and run `./select_nparticles.sh 20` afterwards.
