#!/bin/bash
# POD-side: regenerate the 20k canonical boosted set (identical jets/boosts to the
# Mac's canonical_20k — guaranteed by the seeded RNG; the merge script verifies the
# manifest md5) and evaluate the trained DeepSet on it at 20k jets.
#
# Run from a REAL terminal (not a notebook cell — cells block-buffer and look hung):
#   cd /home/jovyan/PTQWorkflow/nPELICAN-fpga/equivariance   # adjust to your checkout
#   source "$(conda info --base)/etc/profile.d/conda.sh"
#   conda activate /home/jovyan/envs/fast_jetclass
#   bash run_20k_deepset_pod.sh 2>&1 | tee run_20k_deepset.log
#
# Outputs (after the exit-trap rename):
#   canonical_20k/                                  3.7 GB, kept for reuse, do NOT git add
#   results_20k_pod/logits_boostedbeams_deepset.dat.gz   the transfer artifact (~15 MB)
#   results_20k_pod/manifest.md5                         alignment check for the Mac side
set -euo pipefail
EQDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$EQDIR"

L1REPO=${L1REPO:-../../l1-jet-id}
MODEL_DIR=${MODEL_DIR:-$L1REPO/scripts/trained_deepsets/deepsets_8bit_20const_toptag/kfolding1}
NORM_PKL=${NORM_PKL:-$L1REPO/scripts/data/jetid_toptag/processed/normparams_robust_20const_ptetaphi.pkl}
for f in "$MODEL_DIR" "$NORM_PKL"; do
  [ -e "$f" ] || { echo "missing: $f (override with L1REPO/MODEL_DIR/NORM_PKL env vars)"; exit 1; }
done

[ -d canonical_2k ] && { echo "canonical_2k already exists — resolve first"; exit 1; }
[ -d results_2k ] && { echo "results_2k already exists — resolve first"; exit 1; }
[ -d canonical ] && mv canonical canonical_2k
[ -d results ] && mv results results_2k
mkdir -p results

restore() {
  cd "$EQDIR"
  if [ -d canonical ] && [ -d canonical_2k ]; then mv canonical canonical_20k; fi
  [ -d canonical_2k ] && mv canonical_2k canonical
  if [ -d results ] && [ -d results_2k ]; then mv results results_20k_pod; fi
  [ -d results_2k ] && mv results_2k results
}
trap restore EXIT

# Reuse an existing pod-side 20k canonical set if one is already there.
if [ -d canonical_20k ]; then
  echo "reusing existing canonical_20k/"
  mv canonical_20k canonical
else
  echo "=== [1/2] gen_boosted_inputs (20k jets, ~10-15 min) ==="
  python -u gen_boosted_inputs.py --config config_20k_pod.yaml
fi

echo "=== [2/2] DeepSet evaluation (1.62M rows) ==="
python -u run_sweep_deepset.py \
    --l1-repo "$L1REPO" \
    --model-dir "$MODEL_DIR" \
    --norm-pkl "$NORM_PKL" \
    --nconst 20 --norm robust --mode boostedbeams

md5sum canonical/manifest.csv | awk '{print $1}' > results/manifest.md5
gzip -kf results/logits_boostedbeams_deepset.dat
echo "manifest md5: $(cat results/manifest.md5)"
echo "POD DONE — after the exit-trap rename, commit results_20k_pod/{logits_boostedbeams_deepset.dat.gz,manifest.md5}"
