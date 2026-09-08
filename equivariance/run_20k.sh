#!/bin/bash
# 20k-jet boost sweep for the boost_auc/boost_bgrej figures (config_20k.yaml).
# CANON_DIR/RESULTS_DIR are hardcoded in equiv_common.py, so this swaps the
# committed 2000-jet canonical/ + results/ aside for the duration and ALWAYS
# restores them on exit; the new set ends up in canonical_20k/ + results_20k/.
set -euo pipefail
EQDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$EQDIR"

PN=../../PELICAN-nano/.venv/bin/python
HGQ=../../PELICAN-nano-hgq2/.venv/bin/python

[ -d canonical_2k ] && { echo "canonical_2k already exists — resolve first"; exit 1; }
mv canonical canonical_2k
mv results results_2k
mkdir -p results

restore() {
  cd "$EQDIR"
  if [ -d canonical_2k ]; then
    [ -d canonical ] && mv canonical canonical_20k
    mv canonical_2k canonical
  fi
  if [ -d results_2k ]; then
    [ -d results ] && mv results results_20k
    mv results_2k results
  fi
}
trap restore EXIT

echo "=== [1/6] gen_boosted_inputs (20k jets) ==="
$PN gen_boosted_inputs.py --config config_20k.yaml

echo "=== [2/6] run_sweep: float + 6:6:6 (csim, boostedbeams) ==="
$PN run_sweep.py --config config_20k.yaml --mode boostedbeams

echo "=== [3/6] mixer (baseline) ==="
KERAS_BACKEND=jax $HGQ run_sweep_mlpmixer.py --mode boostedbeams

echo "=== [4/6] mixer (boost-augmented) ==="
KERAS_BACKEND=jax $HGQ run_sweep_mlpmixer.py --mode boostedbeams \
  --weights ../../mlpmixer/model/mixer_aug_best.weights.h5 \
  --norm ../../mlpmixer/model/norm_aug.npz --label mlpmixer_aug

echo "=== [5/6] compute_metrics ==="
$PN compute_metrics.py --config config_20k.yaml --mode boostedbeams

echo "=== [6/6] overlay_external (mixer aggregates) ==="
$PN overlay_external.py --mode boostedbeams --labels mlpmixer mlpmixer_aug

echo "ALL DONE — results land in results_20k/ after the exit-trap rename"
