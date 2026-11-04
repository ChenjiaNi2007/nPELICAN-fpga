#!/usr/bin/env bash
# Select an exported model variant for a build: copies variants/<name>/{weights.h,types_generated.h}
# into firmware/weights/ and, when the variant carries golden vectors, its golden_*.dat into tb_data/.
# The firmware and testbench read those fixed paths, so a variant is "installed" rather than
# selected by include path; this leaves the working tree DIRTY on purpose. Restore before any
# `git pull` with:   variants/use.sh --restore     (= git checkout -- firmware/weights tb_data)
#
#   variants/use.sh <name>
#   export NPELICAN_PART=xcu250-figd2104-2L-e
#   vitis_hls -f build_prj.tcl "reset=1 csim=1 synth=1 cosim=0 validation=0 export=0 vsynth=0 golden=1 winograd=1 const_beams=1 tag=<name>"
#   vivado -mode batch -source vivado_synth.tcl -tclargs _<name>      # -> vivado_synth_opt_<name>.rpt
#
# Variants without golden_*.dat (the proxy_* resource-only exports) must be built with csim=0 golden=0.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" == "--restore" ]]; then
    git checkout -- firmware/weights tb_data
    echo "restored the committed firmware/weights and tb_data"
    exit 0
fi
name=${1:?usage: variants/use.sh <name> | --restore   (names: $(ls variants | grep -v -e use.sh -e README.md | tr '\n' ' '))}
src=variants/$name
[[ -f $src/weights.h && -f $src/types_generated.h ]] || { echo "ERROR: $src has no weights.h/types_generated.h"; exit 1; }
cp "$src/weights.h" "$src/types_generated.h" firmware/weights/
if [[ -f $src/golden_logits.dat ]]; then
    cp "$src"/golden_*.dat tb_data/
    echo "installed $name (weights + golden vectors): build with golden=1 csim=1 ... tag=$name"
else
    echo "installed $name (weights only, NO golden vectors): build with csim=0 golden=0 ... tag=$name"
fi
grep -h "typedef.*input_t;" firmware/weights/types_generated.h | sed 's/^/  /'
echo "remember: variants/use.sh --restore before git pull"
