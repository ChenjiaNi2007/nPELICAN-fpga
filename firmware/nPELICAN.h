#ifndef NPELICAN_H_
#define NPELICAN_H_

#include "ap_fixed.h"
#include "ap_int.h"
// hls_stream.h removed: hls::stream is not used in nPELICAN.h or nPELICAN.cpp.
// (nnet_helpers.h includes it via its own header for templates unused in this project.)

#include <cmath>

// Generated per-stage fixed-point typedefs (model_loader.py --quant). Phase 2:
// these ARE the datapath types now — nPELICAN.cpp uses dot_t / t2_t / relu_t /
// t0_t / w1_gen_t / w2_gen_t / bias_t_gen / bn_t_gen / norm_t / acc*_t / mac*_t
// from here. The hand-written typedefs below remain only for the standalone
// float-export path (weights.h generated WITHOUT --quant).
//
// Float-reference build (equivariance harness): -DNPELICAN_FLOAT_BUILD swaps the
// generated fixed-point header for types_float.h, which aliases EVERY datapath and
// weight typedef to `double`. The exact firmware algorithm then runs with no
// quantization, giving the un-quantized Lorentz-invariance floor (any gap to a
// fixed-point curve is then purely a quantization artifact — same code, only the
// number type changes).
#ifdef NPELICAN_FLOAT_BUILD
#include "weights/types_float.h"
#else
#include "weights/types_generated.h"
#endif

// Model dimensions. types_generated.h (included above, written by model_loader.py
// from the checkpoint) or a -D flag takes precedence; these are hand defaults.
// weights.h static_asserts that they match the exported checkpoint.
#ifndef NPARTICLES
#define NPARTICLES  20
#endif
// Spurions prepended before the constituents: 2 beams (slots 0,1), plus the full-jet
// 4-momentum at slot 2 when the checkpoint was trained with --add-jet (NSPURIONS=3;
// types_generated.h then also defines NPELICAN_JET_SPURION, which adds the jet_input port).
#ifndef NSPURIONS
#define NSPURIONS 2
#endif
#ifndef NPARTICLES2
#define NPARTICLES2 (NPARTICLES + NSPURIONS)  // + beam (and optional jet) spurions
#endif
#ifndef NHIDDEN
#define NHIDDEN 2       //Number of parallel channels
#endif
#ifndef NOUT
#define NOUT 1          //Output logits: 1 = binary single score; K = K-class (softmax/argmax off-chip)
#endif
// Width of the 2->0 dense output. Without a head it IS the logit count (NOUT). With
// --head-hidden K (types_generated.h: NPELICAN_HEAD K, N2TO0_OUT K) the 2->0 dense mixes
// down to K hidden channels -> ReLU quant (relu0_t) -> K->NOUT head MAC -> logits.
#ifndef N2TO0_OUT
#define N2TO0_OUT NOUT
#endif

// 2->2 bias-add type (bias guard bits, 2026-10-06). Headers from the pre-guard-bit loader
// do not define it; fall back to the product-grid accumulator (legacy arithmetic).
#ifndef NPELICAN_MAC2B_T_GENERATED
typedef mac2_t mac2b_t;   // pre-guard-bit headers: bias add at the product grid (legacy arithmetic)
#endif
#define N_TABLE_PSLOG 1024 //want to cover 10^6 max input 
#define N_TABLE_COS 1024
#define N_TABLE_SINH 1024
#define N_TABLE_COSH 1024
#define TABLE_FRACS 10

// Raw-momentum / IO interface type (NOT a learned quantizer; the input_quant grid
// lives on the dots, typed dot_t). Under --quant this is GENERATED in
// types_generated.h (I from |p|max, F = ceil(log2|p|max)+dot_F+3 so it tracks the dot
// grid and shrinks the 36x36 dot4 multipliers as QAT bits drop). The hand fallback
// below is used only by the float-export path (no --quant), where the generator's
// NPELICAN_INPUT_T_GENERATED guard is absent: I=12 covers |p| up to 2048 GeV; F=24
// keeps the dot4 product error well under half the dot_t LSB. AP_RND_CONV: the
// float->input_t cast lands on the nearest grid point (closest to PyTorch's momenta).
#ifndef NPELICAN_INPUT_T_GENERATED
typedef ap_fixed<36,12,AP_RND_CONV,AP_SAT> input_t;
#endif
// Final logit carries the output_quant grid. Under --quant this is GENERATED in
// types_generated.h as result_t == out_t (the per-checkpoint output_quant grid), so
// model_out[o] = (result_t)Rp[o] rounds exactly once (RND_CONV) and never clamps the
// logit. The hand fallback below (range [-1,1)) is used ONLY by the float-export path
// (no --quant); for a quant checkpoint whose output_quant range exceeds [-1,1) it would
// saturate the logit and corrupt the score ranking, hence the generated override.
// Jet-spurion quantizer split (--jet-quant-split; types_generated.h defines
// NPELICAN_JET_QUANT_SPLIT): d[2,j]/d[i,2] round onto dotj_t (input_quant_jet), d[2,2] onto
// dotm_t (input_quant_mjet), the jet momenta arrive on jet_t (pmu_quant_jet), and the dots
// array is dotall_t (I = max I, F = max F of the three dot grids: holds each exactly).
// Headers from the pre-split loader do not define them: alias to the single-grid types
// (exactly what the new loader emits without the split -> byte-identical datapath).
#ifndef NPELICAN_JET_TYPES_GENERATED
typedef dot_t dotj_t;
typedef dot_t dotm_t;
typedef dot_t dotall_t;
typedef input_t jet_t;
#endif
#if defined(NPELICAN_JET_QUANT_SPLIT) && !defined(NPELICAN_JET_SPURION)
#error "NPELICAN_JET_QUANT_SPLIT requires NPELICAN_JET_SPURION (jet spurion at slot 2)"
#endif
#if defined(NPELICAN_JET_QUANT_SPLIT) && defined(NPELICAN_BLOCK_FP)
#error "NPELICAN_JET_QUANT_SPLIT is not wired for the block-FP front end (PyTorch refuses it too)"
#endif
#ifndef NPELICAN_RESULT_T_GENERATED
typedef ap_fixed<24, 1,AP_RND_CONV,AP_SAT> result_t;
#endif
// --- legacy hand types: used ONLY by the float-export weights.h (no --quant) ---
// Guarded so the float-reference build (types_float.h) can alias them to double;
// the normal fixed-point fallbacks below are used by every other build.
#ifndef NPELICAN_LEGACY_WEIGHT_TYPES
typedef ap_fixed<24,12,AP_TRN_ZERO,AP_SAT> internal_t;
typedef ap_fixed<24,12,AP_TRN_ZERO,AP_SAT> weight_t;
typedef ap_fixed<24, 1,AP_TRN_ZERO,AP_SAT> w1_t;
typedef ap_fixed<24, 4,AP_TRN_ZERO,AP_SAT> w2_t;
typedef ap_fixed<24,12,AP_TRN_ZERO,AP_SAT> bias_t;
#endif

typedef ap_fixed<12,10,AP_TRN_ZERO,AP_SAT> encoder_t;
typedef ap_ufixed<32,16,AP_TRN_ZERO,AP_SAT> psloglut_t;


template<class data_T, int N_TABLE>
static void lut_pslog_init(data_T table_out[N_TABLE])
{
    for (int ii = 0; ii < N_TABLE; ii++) {
        float x = float( ii <<(TABLE_FRACS));
        data_T real_val = (data_T) ((pow(1+x,0.0009)-1)/0.0009);
        table_out[ii] = real_val;
    }
};

// dots carry the input_quant grid → dot_t (was an internal_t/input_t mismatch before).
void dot4(input_t p1[4], input_t p2[4], dot_t& dot);
#ifdef NPELICAN_JET_QUANT_SPLIT
// jet-involving dots keep the jet momenta on their own grid (jet_t): exact products, one
// RND_CONV cast onto the population's learned grid (dotj_t: particle/beam x jet; dotm_t: m_jet^2).
void dot4j(input_t p[4], jet_t q[4], dotj_t& dot);
void dot4m(jet_t q[4], dotm_t& dot);
#endif

// Lever 7: per-particle block-FP encode + (m, e) dot4. Inert unless the exported
// checkpoint was trained with --pmu-block-fp (types_generated.h defines
// NPELICAN_BLOCK_FP and the mant_t/bexp_t/mdot_t/dotalign_t pair). Needs input_t,
// dot_t and NPARTICLES2, so it is included here rather than at the top.
#include "np_blockfp.h"

// nobj is a PARTICLE COUNT (0..NPARTICLES2), not a momentum: it must not share
// input_t. With input_t capped below 12 bits (negative F, momentum LSB > 1 GeV)
// an input_t nobj would round odd counts to even, corrupting the mask and the
// BN2 β'·count terms. ap_uint<5> covers 0..31 and is exact at every input width.
// Width follows NPARTICLES2 (nobj is remapped to 0..NPARTICLES2 inside the top):
// 5 bits up to 31 (N<=29, incl. the default 20), 6 bits to 63 (N=32: NPARTICLES2=34),
// 7 bits beyond. ncount2 = ncount*ncount needs 2*NOBJ_BITS-1 bits (484 < 2^9).
#define NOBJ_BITS ((NPARTICLES2) <= 31 ? 5 : ((NPARTICLES2) <= 63 ? 6 : 7))
typedef ap_uint<NOBJ_BITS> nobj_t;

void nPELICAN(
    input_t model_input[(NPARTICLES)*4],
    input_t beam_input[2*4],            // 2 beam spurions as a top-level input
#ifdef NPELICAN_JET_SPURION
    jet_t jet_input[4],                 // full-jet 4-momentum (E,px,py,pz) -> spurion slot 2
#endif
    nobj_t nobj,
    result_t model_out[NOUT]
);

#endif
