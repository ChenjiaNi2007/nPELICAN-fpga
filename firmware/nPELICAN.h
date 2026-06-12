#ifndef NPELICAN_H_
#define NPELICAN_H_

#include "ap_fixed.h"
#include "ap_int.h"
// hls_stream.h removed: hls::stream is not used in nPELICAN.h or nPELICAN.cpp.
// (nnet_helpers.h includes it via its own header for templates unused in this project.)

#include <cmath>

#define NPARTICLES  20
#define NPARTICLES2 22  //Max number of particles plus number of spurions
#define NHIDDEN 2       //Number of parallel channels
#define NOUT 1          //Two classes means one out dimension is sufficient 
#define N_TABLE_PSLOG 1024 //want to cover 10^6 max input 
#define N_TABLE_COS 1024
#define N_TABLE_SINH 1024
#define N_TABLE_COSH 1024
#define TABLE_FRACS 10

typedef ap_fixed<24,12,AP_TRN_ZERO,AP_SAT> input_t;    // d_ij — matches learned 2^-12
typedef ap_fixed<24, 4,AP_TRN_ZERO,AP_SAT> result_t;   // logit — learned 2^-20
typedef ap_fixed<24,12,AP_TRN_ZERO,AP_SAT> internal_t; // accumulators — KEEP headroom (Step 4)
typedef ap_fixed<24,12,AP_TRN_ZERO,AP_SAT> weight_t;   // BatchNorm constants only now
typedef ap_fixed<24, 1,AP_TRN_ZERO,AP_SAT> w1_t;       // 2->2 weights — learned 2^-23
typedef ap_fixed<24, 4,AP_TRN_ZERO,AP_SAT> w2_t;       // 2->0 weights — learned 2^-20
typedef ap_fixed<24,12,AP_TRN_ZERO,AP_SAT> bias_t;     // float biases (~ -6.5..0)

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

void dot4(input_t p1[4], input_t p2[4], internal_t& dot);

void nPELICAN(
    input_t model_input[(NPARTICLES)*4],
    input_t nobj,
    result_t model_out[1]
);

#endif
