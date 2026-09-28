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

// Lever 9 does NOT apply under Lever 7 block-FP: there the dot operands are per-particle
// (mantissa, exponent) pairs, not uniform input_t momenta, so the Winograd identity would
// need per-pair exponent alignment of every sum (E_i - px_j etc.), losing the exactness
// and the saving. Refuse the combination rather than silently ignoring the flag.
#if defined(NPELICAN_WINOGRAD_DOT) && defined(NPELICAN_BLOCK_FP)
#error "NPELICAN_WINOGRAD_DOT (Lever 9) is incompatible with NPELICAN_BLOCK_FP (Lever 7 block-FP momenta); build without winograd=1"
#endif

// ---- Particle count --------------------------------------------------------------
// NPARTICLES (real constituents) is set, in priority order, by
//   1. -DNPARTICLES=N on the compiler line (build_prj.tcl nparticles=N; build_local.sh
//      passes extra flags through), else
//   2. NPELICAN_NPARTICLES from weights/types_generated.h (model_loader.py --nparticles N;
//      the loader sizes the accumulator headroom H1/H2 for that N), else
//   3. 20 (float-export path / float-reference build: types_float.h defines no N).
// The accumulators in types_generated.h hold N_exported + 2 rows exactly; a build with MORE
// particles than the header was exported for could overflow them silently (they wrap), so
// that combination is a hard error. Fewer particles is safe (over-sized headroom).
#if defined(NPARTICLES)
  #if defined(NPELICAN_NPARTICLES) && (NPARTICLES > NPELICAN_NPARTICLES)
    #error "NPARTICLES exceeds the particle count types_generated.h was exported for; re-run model_loader.py --nparticles N"
  #endif
#elif defined(NPELICAN_NPARTICLES)
  #define NPARTICLES NPELICAN_NPARTICLES
#else
  #define NPARTICLES 20
#endif
#define NPARTICLES2 (NPARTICLES + 2)  //Max number of particles plus number of spurions

// np_bits<N>::value = number of bits needed to hold every value 0..N (N >= 1):
// np_bits<22> = 5 (0..31), np_bits<34> = 6, np_bits<484> = 9, np_bits<1156> = 11.
// C++11 recursive template (Vitis builds with -std=c++0x); compile-time constant only.
template<unsigned long N> struct np_bits {
    enum { value = 1 + np_bits<(N >> 1)>::value };
};
template<> struct np_bits<0> { enum { value = 0 }; };
static_assert(NPARTICLES >= 1, "NPARTICLES must be >= 1");
#define NHIDDEN 2       //Number of parallel channels
#define NOUT 1          //Two classes means one out dimension is sufficient 
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
// model_out[0] = (result_t)Rp rounds exactly once (RND_CONV) and never clamps the
// logit. The hand fallback below (range [-1,1)) is used ONLY by the float-export path
// (no --quant); for a quant checkpoint whose output_quant range exceeds [-1,1) it would
// saturate the logit and corrupt the score ranking, hence the generated override.
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

// Lever 9 (-DNPELICAN_WINOGRAD_DOT; build_prj.tcl winograd=1): Winograd inner-product
// form of the Minkowski dot, 2 multiplies per pair + 2 per particle instead of 4 per
// pair (docs/RESOURCE_REDUCTION_LEVERS.md, Lever 9). With a_i=(E,px,py,pz)_i and
// b_j=(E,-px,-py,-pz)_j:
//   xi[i]  =  E_i*px_i + py_i*pz_i          (per particle, exact in dotxi_t)
//   eta[j] = -E_j*px_j + py_j*pz_j          (per particle, exact in dotxi_t)
//   d_ij   = (E_i-px_j)*(px_i+E_j) + (py_i-pz_j)*(pz_i-py_j) - xi[i] - eta[j]
// Every intermediate is exact (HLS promoted types; nothing narrowed), so the single
// (dot_t) cast sees the same value as dot4 -> bit-identical dots.
// dotxi_t = ap_fixed<2W+1, 2I+1> of input_t<W,I>: holds a two-product sum exactly.
// Under --quant it is GENERATED (types_generated.h, NPELICAN_DOTXI_T_GENERATED); this
// fallback derives it from whatever input_t is in scope (the float-export hand
// input_t <36,12> gives <73,25>). types_float.h aliases it to double.
#ifndef NPELICAN_DOTXI_T_GENERATED
typedef ap_fixed<2*input_t::width+1, 2*input_t::iwidth+1> dotxi_t;
#endif
void dot4_winograd(input_t p1[4], input_t p2[4], dotxi_t xi_i, dotxi_t eta_j, dot_t& dot);

// Lever 7: per-particle block-FP encode + (m, e) dot4. Inert unless the exported
// checkpoint was trained with --pmu-block-fp (types_generated.h defines
// NPELICAN_BLOCK_FP and the mant_t/bexp_t/mdot_t/dotalign_t pair). Needs input_t,
// dot_t and NPARTICLES2, so it is included here rather than at the top.
#include "np_blockfp.h"

// nobj is a PARTICLE COUNT (0..NPARTICLES2), not a momentum: it must not share
// input_t. With input_t capped below 12 bits (negative F, momentum LSB > 1 GeV)
// an input_t nobj would round odd counts to even, corrupting the mask and the
// BN2 β'·count terms. Width is DERIVED from NPARTICLES2 so the remapped count
// (nobj += 2 up to NPARTICLES2) never wraps: ap_uint<5> (0..31) at 20 particles,
// ap_uint<6> at 32. (A fixed ap_uint<5> at 32 particles wrapped nobj=34 to 2 and made
// rows >= 32 unreachable -- HLS deleted their arithmetic.) ncount2 (unmasked pairs,
// 0..NPARTICLES2^2) is ap_uint<np_bits<NPARTICLES2*NPARTICLES2>::value>: 9 bits at 20p.
typedef ap_uint<np_bits<NPARTICLES2>::value> nobj_t;
typedef ap_uint<np_bits<(NPARTICLES2)*(NPARTICLES2)>::value> npairs_t;

void nPELICAN(
    input_t model_input[(NPARTICLES)*4],
    input_t beam_input[2*4],            // 2 beam spurions as a top-level input
    nobj_t nobj,
    result_t model_out[1]
);

#endif
