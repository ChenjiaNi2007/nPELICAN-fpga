#!/usr/bin/env bash
# Parallel Vitis HLS csynth over the already-trained n-hidden checkpoints.
#
# Each N needs its OWN working tree: synthesis compiles firmware/nPELICAN.cpp against
# firmware/nPELICAN.h (NHIDDEN) + firmware/weights/{weights,types_generated}.h, and
# writes a fixed nPELICAN_prj/. So we stage a per-N tree under synth_builds/nhid<N>/,
# regenerate the correct weights INTO firmware/weights/ there, set NHIDDEN, and launch
# vitis_hls concurrently. Then we backfill sweep_results.csv and re-plot.
#
# Requires vitis_hls on PATH. Example (Vitis 2023.2):
#   export XILINX_HLS=/tools/Xilinx/Vitis_HLS/2023.2
#   export XILINX_VIVADO=/tools/Xilinx/Vivado/2023.2
#   export PATH=$XILINX_HLS/bin:$XILINX_VIVADO/bin:$PATH
#
# Env:
#   NS="1 2 3 4 6"   widths (must have model/nhid<N>_best.pt already)
#   VSYNTH=0         1 = also run Vivado post-synth (real LUT/FF; slower)
#   JOBS=0           max concurrent synth jobs (0 = all at once)
#   PN=... PY=...    training repo / python (for model_loader export)
set -euo pipefail

SWEEP_DIR="$(cd "$(dirname "$0")" && pwd)"
FW="$(cd "$SWEEP_DIR/.." && pwd)"
WS="$(cd "$FW/.." && pwd)"
PN="${PN:-$WS/PELICAN-nano}"
PY="${PY:-python}"
NS="${NS:-1 2 3 4 6}"
VSYNTH="${VSYNTH:-0}"
JOBS="${JOBS:-0}"

TOOLS="$SWEEP_DIR/sweep_tools.py"
RESULTS="$SWEEP_DIR/sweep_results.csv"
BUILD="$SWEEP_DIR/synth_builds"
RPT_DIR="$FW/reports"

command -v vitis_hls >/dev/null 2>&1 || {
  echo "ERROR: vitis_hls not on PATH. Source your Xilinx env first (see header)." >&2
  exit 1
}
[[ -f "$RESULTS" ]] || { echo "ERROR: $RESULTS not found (run sweep_nhidden.sh first)." >&2; exit 1; }

mkdir -p "$BUILD" "$RPT_DIR"
FLAGS="reset=1 csim=0 synth=1 cosim=0 validation=0 export=0 vsynth=${VSYNTH}"

stage_and_launch() {
  local N="$1" PREFIX="nhid$1" CKPT="$PN/model/nhid$1_best.pt" D="$BUILD/nhid$1"
  [[ -f "$CKPT" ]] || { echo "  [$N] SKIP: no checkpoint $CKPT"; return 0; }
  echo "  [$N] staging $D"
  rm -rf "$D"; mkdir -p "$D"
  # Invariant sources (skip the 544M tb_data + any *_prj); symlink tb_data (unused, csim=0).
  cp -R "$FW/firmware" "$FW/third_party" "$D/"
  cp "$FW/nPELICAN_tb.cpp" "$FW/build_prj.tcl" "$FW/project.tcl" "$FW/vivado_synth.tcl" "$D/"
  ln -s "$FW/tb_data" "$D/tb_data"
  # Correct per-N weights INTO firmware/weights/ (the include location), + NHIDDEN.
  "$PY" "$TOOLS" set-nhidden --header "$D/firmware/nPELICAN.h" --n "$N" >/dev/null
  ( cd "$FW" && "$PY" model_loader.py --model "$CKPT" --quant --repo "$PN" \
        --out "$D/firmware/weights/weights.h" ) >"$D/export.log" 2>&1
  echo "  [$N] launching csynth (log: $D/synth_run.log)"
  ( cd "$D" && vitis_hls -f build_prj.tcl "$FLAGS" >"$D/synth_run.log" 2>&1 ) &
}

# Launch, with optional concurrency throttle.
running=0
for N in $NS; do
  stage_and_launch "$N"
  running=$((running + 1))
  if [[ "$JOBS" -gt 0 && "$running" -ge "$JOBS" ]]; then wait -n 2>/dev/null || wait; running=$((running - 1)); fi
done
wait
echo "All synth jobs finished."

# Backfill + copy reports.
for N in $NS; do
  RPT="$BUILD/nhid$N/nPELICAN_prj/solution/syn/report/nPELICAN_csynth.rpt"
  if [[ -f "$RPT" ]]; then
    cp "$RPT" "$RPT_DIR/csynth_nhid${N}.rpt"
    echo "  [$N] $("$PY" "$TOOLS" backfill --csv "$RESULTS" --n "$N" --rpt "$RPT")"
  else
    echo "  [$N] NO REPORT — check $BUILD/nhid$N/synth_run.log"
  fi
done

echo ""
column -s, -t "$RESULTS" 2>/dev/null || cat "$RESULTS"
"$PY" "$SWEEP_DIR/sweep_plot.py" --csv "$RESULTS" --outdir "$SWEEP_DIR" || true
