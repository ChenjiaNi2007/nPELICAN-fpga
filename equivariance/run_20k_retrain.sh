#!/bin/bash
# Add the 2026-09-09 retrained checkpoints to the 20k boost sweep (see run_20k.sh for the
# directory-swap rationale: CANON_DIR/RESULTS_DIR are hardcoded in equiv_common.py).
# Runs ONLY the new models through csim, then recomputes the boostedbeams aggregates for
# all six nanoPELICAN curves. Firmware export files are backed up and restored, so the
# deployed weights.h / types_generated.h / NHIDDEN are untouched afterwards.
set -euo pipefail
EQDIR="$(cd "$(dirname "$0")" && pwd)"; cd "$EQDIR"
PN=../../PELICAN-nano/.venv/bin/python
[ -d canonical_2k ] && { echo "canonical_2k already exists — resolve first"; exit 1; }
BK=$(mktemp -d); cp ../firmware/weights/weights.h ../firmware/weights/types_generated.h ../firmware/nPELICAN.h "$BK"/ 2>/dev/null || true
mv canonical canonical_2k; mv canonical_20k canonical
mv results results_2k; mv results_20k results
restore() {
  cd "$EQDIR"
  [ -d canonical_2k ] && { mv canonical canonical_20k; mv canonical_2k canonical; }
  [ -d results_2k ] && { mv results results_20k; mv results_2k results; }
  cp "$BK"/weights.h "$BK"/types_generated.h ../firmware/weights/ 2>/dev/null || true
  cp "$BK"/nPELICAN.h ../firmware/ 2>/dev/null || true
  echo "restored directories + firmware export files"
}
trap restore EXIT
echo "=== [1/2] run_sweep: retrained models (csim, boostedbeams) $(date) ==="
$PN run_sweep.py --config "${1:-config_20k_retrain_new.yaml}" --mode boostedbeams
echo "=== [2/2] compute_metrics: all six curves $(date) ==="
$PN compute_metrics.py --config config_20k_retrain_all.yaml --mode boostedbeams
echo "RETRAIN SWEEP DONE $(date)"
