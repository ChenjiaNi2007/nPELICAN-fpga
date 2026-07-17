#!/usr/bin/env bash
# Local g++ build of the testbench (no Vitis required).
# Uses open-source Xilinx HLS arbitrary-precision types in third_party/ap_types.
# Minimal stubs for hls_stream.h and hls_math.h (unused in the firmware logic)
# are in third_party/stubs/; they must precede the ap_types include path so that
# the ap_*_special.h complex-specialization headers are overridden with empty stubs
# that suppress the macOS/clang ambiguous-'complex' errors from libc++ inline namespaces.
set -euo pipefail

# `./build_local.sh split` builds the per-stage split firmware
# (firmware/nPELICAN_split.cpp, same top function) into ./tb_local_split.
# `./build_local.sh split2` builds the same file with -DNPELICAN_SPLIT_TRI
# (triangular symmetric crossings, build_prj.tcl split=2; FUNCTION_SPLIT.md)
# into ./tb_local_split_tri. Default builds the monolith into ./tb_local.
# Extra g++ flags (e.g. -DRUN_GOLDEN_GATE) can be passed after the optional
# `split`/`split2` argument.
SRC=firmware/nPELICAN.cpp
OUT=tb_local
EXTRA=
if [[ "${1:-}" == "split" ]]; then
    SRC=firmware/nPELICAN_split.cpp
    OUT=tb_local_split
    shift
elif [[ "${1:-}" == "split2" ]]; then
    SRC=firmware/nPELICAN_split.cpp
    OUT=tb_local_split_tri
    EXTRA=-DNPELICAN_SPLIT_TRI
    shift
fi

g++ -std=c++17 -O2 \
    -I third_party/stubs \
    -I third_party/ap_types/include \
    -I . \
    $EXTRA \
    "$@" \
    nPELICAN_tb.cpp "$SRC" \
    -o "$OUT"

echo "Build succeeded: ./$OUT"
