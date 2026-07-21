#!/usr/bin/env bash
# Capacity sweep: how model size (--n-hidden) trades off accuracy vs FPGA resources.
#
# Lives in nPELICAN-fpga/sweep/. Assumes PELICAN-nano is a SIBLING of this repo
# (../PELICAN-nano) unless overridden with the PN env var. Drives the full loop:
#
#   train QAT -> count params + read accuracy -> set firmware NHIDDEN ->
#   export weights/types -> regenerate golden vectors -> local csim gate ->
#   remote Vitis csynth (if available) -> append one row to sweep_results.csv ->
#   render figures.
#
# The quant config + regularization are PINNED to the production recipe
# (PELICAN-nano/scripts/sweep_pmu_width.sh: w/a/i = 6/6/6, pmu = 12) and held fixed
# so every point differs ONLY in --n-hidden. Do not rely on argparse defaults here:
# they are 8/8/8 with pmu=None (== float, UN-quantized momenta), which does not match
# the firmware input_t.
#
# Config via env:
#   NS="1 2 3 4 6"      widths to sweep
#   EPOCHS=8            training epochs  (cos LR needs >=8; set DECAY=flat for >=5)
#   DECAY=cos           --lr-decay-type
#   WBITS=6 ABITS=6 IBITS=6 PMU=12   weight / act / input(dot) / momentum bit widths
#   SEED=42             fixed across widths for comparability
#   DROP=0.05 DROP_OUT=0.05 WD=0.005 NOBJ=20 NOBJ_AVG=49 BATCH=256   production recipe
#   DEVICE="--no-reproducible"       extra train flags (e.g. "--cuda --no-reproducible")
#   GOLDEN_N=200        events for golden-vector regen / csim
#   DATADIR=data/sample_data  h5 dir (globbed for *train/valid/test*.h5), rel. to PELICAN-nano
#   DO_SYNTH=auto       auto|yes|no  (auto = run csynth only if `vitis_hls` on PATH)
#   PN=/path/to/PELICAN-nano   override the training repo location
#   PY=python                  python interpreter (use the QAT venv's python)
#
# Re-runnable: existing checkpoints/rows are overwritten per N.
set -euo pipefail

SWEEP_DIR="$(cd "$(dirname "$0")" && pwd)"
FW="$(cd "$SWEEP_DIR/.." && pwd)"                 # nPELICAN-fpga repo root
WS="$(cd "$FW/.." && pwd)"                        # workspace (parent of both repos)
PN="${PN:-$WS/PELICAN-nano}"                      # training repo (sibling by default)

NS="${NS:-1 2 3 4 6}"
EPOCHS="${EPOCHS:-8}"
DECAY="${DECAY:-cos}"
# Quant config (production recipe: 6/6/6/12), held fixed across the sweep.
WBITS="${WBITS:-6}"
ABITS="${ABITS:-6}"
IBITS="${IBITS:-6}"
PMU="${PMU:-12}"
# Regularization / data recipe, held fixed; SEED fixed for comparability.
SEED="${SEED:-42}"
DROP="${DROP:-0.05}"
DROP_OUT="${DROP_OUT:-0.05}"
WD="${WD:-0.005}"
NOBJ="${NOBJ:-20}"
NOBJ_AVG="${NOBJ_AVG:-49}"
BATCH="${BATCH:-256}"
DEVICE="${DEVICE:---no-reproducible}"   # QAT on GPU needs --no-reproducible (kthvalue calib)
GOLDEN_N="${GOLDEN_N:-200}"
DATADIR="${DATADIR:-data/sample_data}"
DO_SYNTH="${DO_SYNTH:-auto}"
PY="${PY:-python}"

HDR="$FW/firmware/nPELICAN.h"
TOOLS="$SWEEP_DIR/sweep_tools.py"
RESULTS="$SWEEP_DIR/sweep_results.csv"
RPT_DIR="$FW/reports"

if [[ ! -d "$PN" ]]; then
  echo "ERROR: PELICAN-nano not found at '$PN'. Set PN=/path/to/PELICAN-nano." >&2
  exit 1
fi

if [[ ! -f "$RESULTS" ]]; then
  echo "n_hidden,params,accuracy,AUC,BgRej03,csim,LUT,FF,DSP,BRAM,lat_cycles,lat_ns,II" > "$RESULTS"
fi

for N in $NS; do
  PREFIX="nhid${N}"
  CKPT="$PN/model/${PREFIX}_best.pt"
  echo ""
  echo "================ n_hidden = $N ($PREFIX) ================"

  # 1) Train (QAT) at the pinned production config; only --n-hidden varies.
  ( cd "$PN" && "$PY" train_pelican_nano.py \
        --prefix "$PREFIX" --n-hidden "$N" --datadir "$DATADIR" \
        --target is_signal --nobj "$NOBJ" --nobj-avg "$NOBJ_AVG" \
        --num-epoch "$EPOCHS" --lr-decay-type "$DECAY" --batch-size "$BATCH" \
        --quant --po2-scales \
        --weight-bit-width "$WBITS" --act-bit-width "$ABITS" \
        --input-bit-width "$IBITS" --pmu-bit-width "$PMU" \
        --drop-rate "$DROP" --drop-rate-out "$DROP_OUT" --weight-decay "$WD" \
        --seed "$SEED" $DEVICE )

  # 2) Parameter count + accuracy from this checkpoint.
  PARAMS="$("$PY" "$TOOLS" count-params --ckpt "$CKPT" --repo "$PN")"
  ACCROW="$("$PY" "$TOOLS" metrics --prefix "$PREFIX" --logdir "$PN/log")"
  echo "  params=$PARAMS  accuracy,AUC,BgRej03=$ACCROW"

  # 3) Point the firmware at this width (edits `#define NHIDDEN`).
  "$PY" "$TOOLS" set-nhidden --header "$HDR" --n "$N"

  # 4) Export snapped weights + per-quantizer generated types for THIS checkpoint.
  #    MUST target firmware/weights/ — the firmware includes "weights/weights.h"
  #    relative to firmware/, so the repo-root weights/ (loader default) is ignored.
  ( cd "$FW" && "$PY" model_loader.py --model "$CKPT" --quant --repo "$PN" \
        --out firmware/weights/weights.h )

  # 5) Regenerate golden vectors from the PyTorch quant model (checkpoint-specific).
  ( cd "$PN" && "$PY" scripts/export_golden.py \
        --checkpoint "$CKPT" --testfile "$DATADIR/test.h5" \
        --outdir "$FW/tb_data" --num "$GOLDEN_N" )

  # 6) Local functional gate (free, no Vitis): firmware logits vs PyTorch golden.
  CSIM="fail"
  if ( cd "$FW" && ./build_local.sh >/tmp/${PREFIX}_build.log 2>&1 \
        && ./tb_local >/tmp/${PREFIX}_tb.log 2>&1 ); then
    if grep -q "GOLDEN GATE: PASS" "/tmp/${PREFIX}_tb.log"; then CSIM="pass"; else CSIM="ran"; fi
  fi
  echo "  csim=$CSIM  (log: /tmp/${PREFIX}_tb.log)"

  # 7) Resources: Vitis csynth. Skipped cleanly if vitis_hls is absent.
  RES="NA,NA,NA,NA,NA,NA,NA"
  RUN_SYNTH=0
  if [[ "$DO_SYNTH" == "yes" ]]; then RUN_SYNTH=1
  elif [[ "$DO_SYNTH" == "auto" ]] && command -v vitis_hls >/dev/null 2>&1; then RUN_SYNTH=1
  fi
  if [[ "$RUN_SYNTH" == 1 ]]; then
    ( cd "$FW" && vitis_hls -f build_prj.tcl )
    RPT="$FW/nPELICAN_prj/solution/syn/report/nPELICAN_csynth.rpt"
    if [[ -f "$RPT" ]]; then
      cp "$RPT" "$RPT_DIR/csynth_${PREFIX}.rpt"
      RES="$("$PY" "$TOOLS" parse-csynth --rpt "$RPT")"
    fi
    echo "  resources LUT,FF,DSP,BRAM,lat_cyc,lat_ns,II = $RES"
  else
    echo "  resources: SKIPPED (vitis_hls not found / DO_SYNTH=$DO_SYNTH)."
    echo "             Firmware staged (NHIDDEN=$N, weights.h exported); run csynth"
    echo "             on the Vitis box, then backfill with:"
    echo "             $PY $TOOLS parse-csynth --rpt <rpt>"
  fi

  echo "${N},${PARAMS},${ACCROW},${CSIM},${RES}" >> "$RESULTS"
done

echo ""
echo "Results table: $RESULTS"
column -s, -t "$RESULTS" 2>/dev/null || cat "$RESULTS"

# 8) Figures.
"$PY" "$SWEEP_DIR/sweep_plot.py" --csv "$RESULTS" --outdir "$SWEEP_DIR" || \
  echo "(plotting skipped: matplotlib unavailable in $PY)"
