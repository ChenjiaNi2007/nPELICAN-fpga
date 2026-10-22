#!/usr/bin/env bash
# Select a pre-exported header set for an N-particle resource build (no checkpoint or
# Brevitas needed on the synthesis box). Copies firmware/weights/exports/N<n>/*.h into
# firmware/weights/. All sets come from cap_h2_qatf12_lr0p0025_e20_s1_best.pt; weights.h is
# identical across N, only types_generated.h (NPELICAN_NPARTICLES + accumulator headroom)
# differs. Run with 20 to restore the default before any 20-particle csim gate.
set -euo pipefail
n="${1:?usage: ./select_nparticles.sh <8|12|16|20|24|32>}"
src="firmware/weights/exports/N${n}"
[[ -d "$src" ]] || { echo "no export for N=${n} (have: $(ls firmware/weights/exports | tr '\n' ' '))"; exit 1; }
cp "$src/weights.h" "$src/types_generated.h" firmware/weights/
grep -m1 NPELICAN_NPARTICLES firmware/weights/types_generated.h
