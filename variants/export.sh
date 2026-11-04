#!/usr/bin/env bash
# Export a PELICAN-nano checkpoint into variants/<name>/ (weights.h, types_generated.h, golden_*.dat,
# EXPORT.cmd/.log, GATE.txt) and gate it locally with the deployment flags (Winograd + const beams).
#
#   variants/export.sh <name> <ckpt relative to PELICAN-nano> [--no-golden] [extra model_loader.py flags]
#   e.g. variants/export.sh pmu14c2048 model/h5n16_h8_qat_pmu14c2048_s1_best.pt
#        variants/export.sh proxy_in14 model/h5n16_h8_qat_s1_best.pt --no-golden --pmu-extra-int-bits 2
#
# Golden vectors: the first 200 hls4ml 5-class test jets (data/hls4ml5_n16/test.h5), as for every
# 5-class gate (reports/hls4ml5_h8_qat_gate/). --no-golden is for inference-only ablation exports whose
# PyTorch reference does not exist (resource builds only: csim=0 golden=0).
set -euo pipefail
cd "$(dirname "$0")/.."
NANO=${NANO:-$(cd ../PELICAN-nano && pwd)}
PY=${PY:-$NANO/.venv/bin/python}
name=${1:?name}; ckpt=${2:?checkpoint}; shift 2
golden=1
if [[ "${1:-}" == "--no-golden" ]]; then golden=0; shift; fi
d=variants/$name
mkdir -p "$d"
{
  echo "# $(date '+%F %T')  checkpoint: PELICAN-nano/$ckpt"
  echo "$PY model_loader.py --model $NANO/$ckpt --quant --repo $NANO --out $d/weights.h $*"
  (( golden )) && echo "(cd $NANO && $PY scripts/export_golden.py --checkpoint $ckpt --testfile data/hls4ml5_n16/test.h5 --num 200 --outdir $(pwd)/$d)"
} > "$d/EXPORT.cmd"
"$PY" model_loader.py --model "$NANO/$ckpt" --quant --repo "$NANO" --out "$d/weights.h" "$@" > "$d/EXPORT.log" 2>&1
grep -h "typedef.*input_t;\|typedef.*dot_t;" "$d/types_generated.h"
if (( golden )); then
  (cd "$NANO" && "$PY" scripts/export_golden.py --checkpoint "$ckpt" --testfile data/hls4ml5_n16/test.h5 --num 200 --outdir "$OLDPWD/$d") >> "$d/EXPORT.log" 2>&1
  rm -f "$d"/fw_stage_dump.txt
  variants/use.sh "$name" > /dev/null
  ./build_local.sh -DNPELICAN_WINOGRAD_DOT -DNPELICAN_CONST_BEAMS -DRUN_GOLDEN_GATE > /dev/null
  ./tb_local 2>&1 | grep -E "GOLDEN SUMMARY|GOLDEN GATE|DOTS-LEVEL" | tee "$d/GATE.txt"
  variants/use.sh --restore > /dev/null
else
  echo "no golden vectors (ablation export): build with csim=0 golden=0" | tee "$d/GATE.txt"
fi
echo "exported $d"
