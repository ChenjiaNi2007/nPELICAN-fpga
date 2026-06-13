# Firmware Resource Log

Records synthesis and C-sim results at each phase of the QAT restructure
(see `FIRMWARE_QAT_PLAN.md` for phase definitions).

| phase | date | checkpoint | weights.h commit | LUT | FF | DSP | latency (cycles) | II | timing met | csim vs golden (exact/total, max\|Δ\|) | notes |
|-------|------|------------|-----------------|-----|----|-----|------------------|----|------------|----------------------------------------|-------|
| Phase 0 baseline | 2026-06-12 | fpga_model_qat_best.pt (8ep, 24-bit po2) | 3c688bf | n/a | n/a | n/a | n/a | n/a | n/a | **Vitis csim: 0/200 exact, max\|Δ\|=3.6066896** (remote output archived at tb_data/.ipynb_checkpoints/golden_fw_results-checkpoint.log) | Pre-restructure uniform-type firmware. Local clang + open-source ap_types build exhibits UB (-O0 vs -O2 outputs differ) → local builds are smoke checks only; authoritative csim is remote Vitis. Homebrew gcc-13 too old for current SDK; `brew upgrade gcc` may revive a trustworthy local loop (local differs from Vitis on 200/200 events, max gap 5.64). CSYNTH DID NOT COMPLETE on the remote box — known pre-existing issue with the high particle count (also failed for the previous model); user deprioritized. II/resource regression tracking deferred until synth is viable. |
| Phase 2 retype | 2026-06-13 | fpga_model_qat_best.pt | 1ed473a | TBD | TBD | TBD | TBD | TBD (remote) | TBD | **csim GATE PASS** — dots-level (network) 142/200 zero-tol exact, max\|Δ\|=1.1e-5 (tol 1e-4); golden (incl dot4) 133/200 exact, max\|Δ\|=6.3e-4 (tol 1e-3). Local==Vitis confirmed bit-for-bit (retyped firmware is toolchain-stable; local is now a trustworthy oracle). | Per-stage QAT types wired. Bugs fixed: Tr saturation (t0_t→tr_t), BN eps omitted (weight/sqrt(var+eps)). Tolerance gate (not zero-tol 200/200): float BatchNorm leaves float intermediate segments that fixed-point can't match bit-for-bit at every quantizer boundary (PyTorch's own f32-vs-f64 logits differ ~3.6e-6). dot4 front-end caveat (D4): PyTorch d_ij in lossy float32 → ≤6.3e-4 golden-path residual; dots-level gate isolates+proves the network. csynth still deferred (pre-existing high-particle-count issue). |

## Phase 2 bit-exactness — interpretation

Zero-tolerance 200/200 csim vs the PyTorch quant logits is **not achievable for this
architecture**, by design, because BatchNorm is kept in float (a CLAUDE.md invariant). Between
the learned quantizers (`input_quant`, `post_agg`×2, `act_layer`, `output_quant`) the datapath
has *unquantized* float segments (the two BatchNorms and the N̄-normalized aggregations).
PyTorch evaluates these in float32; the firmware in fixed-point. The two agree at a quantizer
output only when their pre-quant values round to the same grid point — which fails on a minority
of events where float-vs-fixed rounding straddles a boundary, cascading to ≤~1e-5 on the logit.
This is intrinsic, not a width bug: widening the BN-constant precision (F31→F43) changed nothing,
and a float64 golden barely moved the result; PyTorch's own float32-vs-float64 logits already
differ by up to 3.6e-6. Hence the **tolerance gate** (`nPELICAN_tb.cpp`): PASS = max\|Δ\| under
tolerance (1e-4 network / 1e-3 with dot4), with the zero-tolerance exact count reported alongside.

The **dot4 front-end caveat** (plan D4) is the extra golden-path residual: PyTorch computes
`d_ij = E²−|p|²` in lossy float32 (catastrophic cancellation on self-dots), the firmware computes
it exactly from the momenta, so they round to different 2⁻¹⁰ grid points. The **dots-level gate**
(inject PyTorch's quantized dots via `npelican_dots_override`, see `golden_dots.dat`) removes this
variable and verifies the network in isolation (142/200 exact, max 1.1e-5).
