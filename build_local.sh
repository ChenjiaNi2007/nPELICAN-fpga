#!/usr/bin/env bash
# Local g++ build of the testbench (no Vitis required).
# Uses open-source Xilinx HLS arbitrary-precision types in third_party/ap_types.
# Minimal stubs for hls_stream.h and hls_math.h (unused in the firmware logic)
# are in third_party/stubs/; they must precede the ap_types include path so that
# the ap_*_special.h complex-specialization headers are overridden with empty stubs
# that suppress the macOS/clang ambiguous-'complex' errors from libc++ inline namespaces.
set -euo pipefail

g++ -std=c++17 -O2 \
    -I third_party/stubs \
    -I third_party/ap_types/include \
    -I . \
    nPELICAN_tb.cpp firmware/nPELICAN.cpp \
    -o tb_local

echo "Build succeeded: ./tb_local"
