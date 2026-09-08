#!/bin/bash
# MAC-side: after `git pull` brings the pod's results_20k_pod/ artifacts, fold the
# 20k-jet DeepSet curve into results_20k/ and regenerate the boost figures.
# Aborts unless the pod's canonical manifest md5 matches the local canonical_20k/
# (row-for-row alignment is what makes the pod logits usable against local metrics).
set -euo pipefail
EQDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$EQDIR"

GZ=results_20k_pod/logits_boostedbeams_deepset.dat.gz
[ -f "$GZ" ] || { echo "missing $GZ — did the pod commit land / did you git pull?"; exit 1; }
[ -f results_20k_pod/manifest.md5 ] || { echo "missing results_20k_pod/manifest.md5"; exit 1; }

POD_MD5=$(cat results_20k_pod/manifest.md5)
LOC_MD5=$(md5 -q canonical_20k/manifest.csv)
if [ "$POD_MD5" != "$LOC_MD5" ]; then
  echo "MANIFEST MISMATCH: pod $POD_MD5 vs local $LOC_MD5"
  echo "The pod's 20k jet selection is not the Mac's — do NOT merge. Check that both"
  echo "sides ran with n_jets=20000 seed=1234 and the same test.h5 contents."
  exit 1
fi
echo "manifest md5 match: $POD_MD5"

gunzip -c "$GZ" > results_20k/logits_boostedbeams_deepset.dat

# Swap the 20k dirs in for the hardcoded CANON_DIR/RESULTS_DIR, run the overlay,
# and restore no matter what.
[ -d canonical_2k ] && { echo "canonical_2k already exists — resolve first"; exit 1; }
mv canonical canonical_2k
mv results results_2k
mv canonical_20k canonical
mv results_20k results
restore() {
  cd "$EQDIR"
  [ -d canonical ] && [ -d canonical_2k ] && mv canonical canonical_20k
  [ -d canonical_2k ] && mv canonical_2k canonical
  [ -d results ] && [ -d results_2k ] && mv results results_20k
  [ -d results_2k ] && mv results_2k results
}
trap restore EXIT

../../PELICAN-nano/.venv/bin/python overlay_external.py --mode boostedbeams --labels deepset mlpmixer mlpmixer_aug

restore
trap - EXIT

"../../PELICAN-nano/.venv/bin/python" "../../results/figures/gen_boost_auc.py"
echo "MERGE DONE — boost figures now carry the 20k DeepSet curve"
