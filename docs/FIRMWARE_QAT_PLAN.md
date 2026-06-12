# FIRMWARE_QAT_PLAN.md — Per-stage fixed-point types from QAT scales

Companion to `CLAUDE.md` (architecture, dataflow, invariants — read first). Goal: restructure
`firmware/nPELICAN.{h,cpp}` and `model_loader.py` so every tensor in the datapath uses a
fixed-point type derived from its learned Brevitas quantization scale, with deliberately
widened accumulators, auto-generated typedefs, and bit-exact C-sim ↔ PyTorch verification.

## 1. Goals / non-goals

Goals: (G1) replace the single `internal_t` with per-stage types matched to learned scales;
(G2) make the type map GENERATED from the checkpoint, not hand-edited; (G3) bit-exact C-sim
agreement with the PyTorch quant model on golden vectors; (G4) resource reduction measured
in the synthesis report (LUT/FF/DSP) at unchanged II.

Non-goals (separate, later work): upper-triangle symmetry exploitation (the dots TODO);
changing NPARTICLES/NHIDDEN; the pseudolog encoder (stays commented out); re-training
(except the optional nobj-avg co-design note in §6).

## 2. Design

### D1 — Type taxonomy in nPELICAN.h
Replace the uniform typedefs with this taxonomy (names final, values generated per D3):

```cpp
// Quantization-point types: ap_fixed<B, B-k, AP_RND_CONV, AP_SAT>, k = learned -log2(scale)
typedef ap_fixed<24,12,AP_RND_CONV,AP_SAT> dot_t;    // input_quant
typedef ap_fixed<24, 4,AP_RND_CONV,AP_SAT> t2_t;     // post_agg 2->2: T basis, batch1, normalized jmass/jdotp
typedef ap_ufixed<24,3,AP_RND_CONV,AP_SAT> relu_t;   // act_layer (signed/unsigned per quantizer!)
typedef ap_fixed<24, 2,AP_RND_CONV,AP_SAT> t0_t;     // post_agg 2->0: Tr, normalized R
typedef ap_fixed<24, 4,AP_RND_CONV,AP_SAT> result_t; // output_quant
typedef ap_fixed<24, 1,AP_RND_CONV,AP_SAT> w1_t;     // 2->2 weights
typedef ap_fixed<24, 4,AP_RND_CONV,AP_SAT> w2_t;     // 2->0 weights
typedef ap_fixed<24, 4,AP_RND_CONV,AP_SAT> bias_t;   // b1, b1_diag, b2 (float-trained, |b|<8)
typedef ap_fixed<24,12,AP_TRN_ZERO,AP_SAT> bn_t;     // BN constants (mean can be O(100))
// Accumulators: summand type widened by ceil(log2(#terms)) integer bits, full fractional
typedef ap_fixed<24+9, 4+9>  acc2_t;   // jmass: sum of NP2^2 t2_t-range terms
typedef ap_fixed<24+5, 4+5>  accrow_t; // jdotp: sum of NP2 terms
typedef ap_fixed<24+9, 2+9>  acc0_t;   // R sum; trace fits in accrow-style width
// MAC temporaries for the two dense layers: let HLS infer, or widen explicitly:
typedef ap_fixed<32, 8>      mac2_t;   // b1 + 6 products of w1_t*t2_t (|prod|<8, 8 terms)
typedef ap_fixed<32, 8>      mac0_t;   // b2 + 2*NHIDDEN products of w2_t*t0_t
```

Rules baked into the above: quantization-point types use AP_RND_CONV (Brevitas rounds to
nearest-even; AP_TRN_ZERO truncation would break bit-exactness at those casts); accumulators
get integer headroom of ⌈log₂(#terms)⌉ over the summand and keep the summand's fractional
bits in full; normalize-late is retained (CLAUDE.md invariant), so the single rescale
`acc × invnave` casts back down to the quantization-point type, rounding once.

### D2 — nPELICAN.cpp retyping
Map every array to its type: `dots`→dot_t; `batch1`→t2_t (cast from the BN affine computed
in a bn-width temp); `jmass`/`jdotp` accumulate in acc2_t/accrow_t, post-normalization values
stored as t2_t; `T`→t2_t; `Tp` accumulates in mac2_t, after ReLU stored as relu_t;
`Tr`→t0_t; `R` accumulates in acc0_t, normalized to t0_t; `Rp` in mac0_t; final cast to
result_t. `nobjmask` becomes ap_uint<1> (it is only ever 0/1; multiply or select). Keep all
loop structure and pragmas; this phase is types only.

### D3 — Generated types (model_loader.py)
Extend the loader to emit `weights/types_generated.h` alongside weights.h:
for each quantizer (input, post_agg 2->2, ReLU, post_agg 2->0, output, two weight layers)
read scale s and signedness from the rebuilt Brevitas model (`quant_weight()` for weights;
act quantizers via named_modules; one calibration forward pass on sample data if an act
scale is uninitialized), compute k = −log₂(s) (assert integer when po2), and emit the
typedef with W = bit width, I = W−k, signed→ap_fixed / unsigned→ap_ufixed, AP_RND_CONV,
AP_SAT. Derive accumulator/MAC typedefs from the quantization-point types + term counts
(NPARTICLES2 from weights). nPELICAN.h `#include`s types_generated.h, keeping the current
hand-written typedefs only as a `#ifndef GENERATED_TYPES` fallback. Also emit the measured
scales as comments in weights.h for the record.

### D4 — Golden-vector verification harness
Add to PELICAN-nano: `scripts/export_golden.py` — loads the quant checkpoint, runs M events
from data/sample_data through the quant model, writes `golden_inputs.dat` (the scaled
4-momenta + nobj, exactly as the testbench feeds nPELICAN) and `golden_logits.dat`.
Add/extend the firmware testbench to read these, run C-sim, and compare logits with a
zero-tolerance (bit-exact) check plus a report of max |Δ|. Acceptance: bit-exact on all M
events; if not bit-exact, the harness must print the first divergent event and stage-level
dumps (dots, jmass, jdotp, T, Tp, Tr, R) to localize the stage.
Known acceptable residual: PyTorch computes d_ij in float then quantizes, firmware computes
d_ij from quantized momenta — if this produces off-by-one-LSB input disagreements, document
it and (option) have export_golden.py emit pre-quantized dots for a second, dots-level
testbench mode that isolates the network from the dot4 front-end.

### D5 — Synthesis regression
After each phase run csim + csynth; record LUT/FF/DSP/latency/II in docs/resource_log.md.
II must stay 1; timing met. Expect resource reduction from D2 (narrower multipliers/adders)
— quantify it.

## 3. Phases

Phase 0: baseline — build current firmware, record csim outputs + synthesis numbers; commit
the golden-vector harness (D4) running against the CURRENT uniform-type firmware to
establish the pre-restructure deviation from PyTorch.
Phase 1: model_loader.py emits types_generated.h (D3) with values matching the current
checkpoint; nPELICAN.h includes it; firmware still compiles unchanged (types equal old ones
initially if you generate them wide; or jump straight to learned values — either, but state
which).
Phase 2: retype nPELICAN.cpp per D2 with the learned-scale types; csim must go bit-exact vs
golden (or document the dot4 front-end caveat per D4); csynth regression per D5.
Phase 3: docs — README section describing the regenerate-on-retrain workflow (train →
model_loader --quant → types+weights regenerate → csim → csynth), and the resource_log
comparison table.

## 4. Acceptance criteria

1. `model_loader.py --quant --repo ../PELICAN-nano --model <ckpt>` regenerates BOTH
   weights.h and types_generated.h with no hand edits required.
2. C-sim bit-exact vs PyTorch quant logits on ≥100 sample events (or documented dot4-LSB
   caveat with the dots-level mode bit-exact).
3. Synthesis: II=1 preserved; resources ≤ baseline; numbers logged.
4. Re-running the whole loop after retraining at different bit widths (e.g. 16/16/16)
   requires zero manual header edits.

## 5. Risks / notes

- Brevitas ReLU quantizer signedness must be read, not assumed; an unsigned grid maps to
  ap_ufixed (no sign bit), changing I by one.
- AP_RND_CONV on every element of fully-unrolled NP2²-sized arrays costs LUTs; if synthesis
  regresses, restrict rounding to true quantization-point casts and keep internal exact-width
  arithmetic (products/sums of fixed-point are exact if the destination is wide enough —
  prefer widening over rounding wherever the value is NOT at a PyTorch quantization point).
- The HLS `if (Tp < 0.)` ReLU compares against a double literal; replace with 0 of the
  proper type while retyping.

## 6. Optional co-design follow-ups (separate tasks, not this refactor)

- Retrain with `--nobj-avg=64`: invnave/invnave2 become exact shifts (2⁻⁶, 2⁻¹²), removing
  the last non-po2 multipliers. Requires retraining + new golden vectors.
- Reduce trained bit widths (e.g. 16 or 12 total) and regenerate — the whole point of G2.
- Upper-triangle dots computation (symmetry) to halve dot4 DSPs.
