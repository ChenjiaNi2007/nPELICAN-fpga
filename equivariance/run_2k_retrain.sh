#!/bin/bash
# Retrained checkpoints on the committed 2000-jet canonical set (for boost_scoredist).
# Normal directories (canonical/, results/); run AFTER run_20k_retrain.sh has restored them.
set -euo pipefail
EQDIR="$(cd "$(dirname "$0")" && pwd)"; cd "$EQDIR"
[ -d canonical_2k ] && { echo "directories are swapped (20k run in progress) — wait"; exit 1; }
PN=../../PELICAN-nano/.venv/bin/python
BK=$(mktemp -d); cp ../firmware/weights/weights.h ../firmware/weights/types_generated.h ../firmware/nPELICAN.h "$BK"/
trap 'cp "$BK"/weights.h "$BK"/types_generated.h ../firmware/weights/; cp "$BK"/nPELICAN.h ../firmware/; echo restored firmware export files' EXIT
echo "=== run_sweep (2k, boostedbeams) $(date) ==="
$PN run_sweep.py --config config_2k_retrain_new.yaml --mode boostedbeams
echo "=== compute_metrics (2k, all) $(date) ==="
$PN compute_metrics.py --config config_2k_retrain_all.yaml --mode boostedbeams
echo "RETRAIN 2K DONE $(date)"
