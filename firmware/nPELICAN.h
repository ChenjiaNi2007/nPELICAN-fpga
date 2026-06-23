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
#include "weights/types_generated.h"

#define NPARTICLES  20
#define NPARTICLES2 22  //Max number of particles plus number of spurions
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
typedef ap_fixed<24,12,AP_TRN_ZERO,AP_SAT> internal_t;
typedef ap_fixed<24,12,AP_TRN_ZERO,AP_SAT> weight_t;
typedef ap_fixed<24, 1,AP_TRN_ZERO,AP_SAT> w1_t;
typedef ap_fixed<24, 4,AP_TRN_ZERO,AP_SAT> w2_t;
typedef ap_fixed<24,12,AP_TRN_ZERO,AP_SAT> bias_t;

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

void nPELICAN(
    input_t model_input[(NPARTICLES)*4],
    input_t nobj,
    result_t model_out[1]
);

#endif
