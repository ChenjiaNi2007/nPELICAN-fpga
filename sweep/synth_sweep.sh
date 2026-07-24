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
#   VSYNTH=1         also run Vivado post-synth -> real vLUT/vFF/vDSP/vBRAM columns
#                    (needs `vivado` on PATH too; set 0 for HLS estimates only, faster)
#   JOBS=0           max concurrent synth jobs (0 = all at once)
#   PN=... PY=...    training repo / python (for model_loader export)
set -euo pipefail

SWEEP_DIR="$(cd "$(dirname "$0")" && pwd)"
FW="$(cd "$SWEEP_DIR/.." && pwd)"
WS="$(cd "$FW/.." && pwd)"
PN="${PN:-$WS/PELICAN-nano}"
PY="${PY:-python}"
NS="${NS:-1 2 3 4 6}"
VSYNTH="${VSYNTH:-1}"
JOBS="${JOBS:-0}"

TOOLS="$SWEEP_DIR/sweep_tools.py"
RESULTS="$SWEEP_DIR/sweep_results.csv"
BUILD="$SWEEP_DIR/synth_builds"
RPT_DIR="$FW/reports"

# Resolve a relative PY (e.g. ../../PELICAN-nano/.venv/bin/python) to absolute NOW:
# the export step cds into other dirs, where a relative path silently points elsewhere.
if [[ "$PY" == */* ]]; then
  if [[ ! -x "$PY" ]]; then
    echo "ERROR: PY '$PY' not found/executable from $(pwd). Use an absolute path or plain 'python'." >&2
    exit 1
  fi
  PY="$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")"
fi

command -v vitis_hls >/dev/null 2>&1 || {
  echo "ERROR: vitis_hls not on PATH. Source your Xilinx env first (see header)." >&2
  exit 1
}
if [[ "$VSYNTH" == 1 ]] && ! command -v vivado >/dev/null 2>&1; then
  echo "ERROR: VSYNTH=1 but 'vivado' not on PATH. Add \$XILINX_VIVADO/bin, or set VSYNTH=0." >&2
  exit 1
fi
[[ -f "$RESULTS" ]] || { echo "ERROR: $RESULTS not found (run sweep_nhidden.sh first)." >&2; exit 1; }

mkdir -p "$BUILD" "$RPT_DIR"
FLAGS="reset=1 csim=0 synth=1 cosim=0 validation=0 export=0 vsynth=${VSYNTH}"

# model_loader reads PELICAN-nano h5 files read-only; HDF5's file locking can hang
# forever on network filesystems (JupyterHub homes) if any kernel/dead run holds a
# lock. Locking adds nothing for our read-only access — disable it.
export HDF5_USE_FILE_LOCKING=FALSE

# Each N is fully isolated in $BUILD/nhid<N>/, so the per-tree nPELICAN_prj and
# vivado_synth.rpt never collide across parallel jobs. Belt-and-suspenders: the job
# copies its OWN reports into uniquely-named reports/{csynth,vsynth}_nhid<N>.rpt the
# instant it finishes (before any re-run could rm -rf the tree), and stale copies are
# cleared up front so a failed job can't leave an old report to be mis-recorded.
# The CSV backfill stays in the serial post-wait loop (one writer, no corruption).
stage_and_launch() {
  local N="$1" PREFIX="nhid$1" CKPT="$PN/model/nhid$1_best.pt" D="$BUILD/nhid$1"
  [[ -f "$CKPT" ]] || { echo "  [$N] SKIP: no checkpoint $CKPT"; return 0; }
  echo "  [$N] staging $D"
  # A previous tree holds a full nPELICAN_prj (tens of thousands of files); rm -rf
  # on a network filesystem can grind for minutes and look like a stall. Move it
  # aside instantly and delete in the background instead.
  if [[ -e "$D" ]]; then
    mv "$D" "$D.old.$$"
    rm -rf "$D.old.$$" &
  fi
  mkdir -p "$D"
  rm -f "$RPT_DIR/csynth_nhid${N}.rpt" "$RPT_DIR/vsynth_nhid${N}.rpt"   # drop stale
  # Invariant sources (skip the 544M tb_data + any *_prj); symlink tb_data (unused, csim=0).
  cp -R "$FW/firmware" "$D/"
  # third_party (open-source ap_types) is only for the local g++ csim; Vitis HLS
  # supplies its own ap_* headers for synthesis, so copy it only if it exists.
  if [[ -d "$FW/third_party" ]]; then cp -R "$FW/third_party" "$D/"; fi
  cp "$FW/nPELICAN_tb.cpp" "$FW/build_prj.tcl" "$FW/project.tcl" "$FW/vivado_synth.tcl" "$D/"
  ln -s "$FW/tb_data" "$D/tb_data"
  # Correct per-N weights INTO firmware/weights/ (the include location), + NHIDDEN.
  if ! "$PY" "$TOOLS" set-nhidden --header "$D/firmware/nPELICAN.h" --n "$N" \
        >/dev/null 2>"$D/set_nhidden.err"; then
    echo "  [$N] SET-NHIDDEN FAILED ($PY not usable?):" >&2
    cat "$D/set_nhidden.err" >&2
    exit 1
  fi
  # Weight export imports torch+brevitas and may run a calibration pass — a minute
  # or two of silence here is normal; the log shows progress.
  echo "  [$N] exporting weights (log: $D/export.log)"
  if ! ( cd "$FW" && "$PY" model_loader.py --model "$CKPT" --quant --repo "$PN" \
        --out "$D/firmware/weights/weights.h" ) >"$D/export.log" 2>&1; then
    echo "  [$N] EXPORT FAILED — tail of $D/export.log:" >&2
    tail -5 "$D/export.log" >&2
    exit 1
  fi
  echo "  [$N] launching csynth (log: $D/synth_run.log)"
  (
    cd "$D"
    vitis_hls -f build_prj.tcl "$FLAGS" >synth_run.log 2>&1 || true
    # Capture reports immediately into unique per-N files (no cross-job overwrite).
    cp -f nPELICAN_prj/solution/syn/report/nPELICAN_csynth.rpt \
          "$RPT_DIR/csynth_nhid${N}.rpt" 2>/dev/null || true
    [[ "$VSYNTH" == 1 ]] && cp -f vivado_synth.rpt \
          "$RPT_DIR/vsynth_nhid${N}.rpt" 2>/dev/null || true
  ) &
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

# Serial backfill from the per-N report copies captured by each job (single CSV writer).
for N in $NS; do
  RPT="$RPT_DIR/csynth_nhid${N}.rpt"
  if [[ -f "$RPT" ]]; then
    echo "  [$N] HLS   $("$PY" "$TOOLS" backfill --csv "$RESULTS" --n "$N" --rpt "$RPT")"
  else
    echo "  [$N] NO HLS REPORT — check $BUILD/nhid$N/synth_run.log"
  fi
  VRPT="$RPT_DIR/vsynth_nhid${N}.rpt"
  if [[ "$VSYNTH" == 1 && -f "$VRPT" ]]; then
    echo "  [$N] vsyn  $("$PY" "$TOOLS" backfill-vsynth --csv "$RESULTS" --n "$N" --rpt "$VRPT")"
  elif [[ "$VSYNTH" == 1 ]]; then
    echo "  [$N] NO VSYNTH REPORT — check $BUILD/nhid$N/synth_run.log (vivado step)"
  fi
done

echo ""
column -s, -t "$RESULTS" 2>/dev/null || cat "$RESULTS"
"$PY" "$SWEEP_DIR/sweep_plot.py" --csv "$RESULTS" --outdir "$SWEEP_DIR" || true
