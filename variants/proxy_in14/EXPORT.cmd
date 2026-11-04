# 2026-11-04 14:15:14  checkpoint: PELICAN-nano/model/h5n16_h8_qat_s1_best.pt (12-bit grid, inference-only +2 integer bits)
python model_loader.py --model <PELICAN-nano>/model/h5n16_h8_qat_s1_best.pt --quant --repo <PELICAN-nano> --out variants/proxy_in14/weights.h --pmu-extra-int-bits 2
