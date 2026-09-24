#include "../nPELICAN.h"
//model: cap_h2_qatf12_lr0p0025_e20_s1
//nobj: 20

//normalization constants
//nobj avg = 49
norm_t invnave = 0.02040816326530612;
norm_t invnave2 = 0.00041649312786339027;

//first batchnorm [mean, weight/sqrt(var), bias]
bn_t_gen batch1_2to2[3] = {27.966884613037109,  0.058448841566411,  0.951832354068756};

//2to2 linear layer
w1_gen_t w1_2to2[NHIDDEN*6] = {-0.531250000000000, -0.937500000000000,  0.437500000000000,  0.468750000000000,  0.062500000000000,  0.375000000000000,  0.500000000000000,  0.375000000000000, -0.406250000000000, -0.375000000000000, -0.968750000000000, -0.968750000000000};
bias_t_gen b1_2to2[NHIDDEN] = {-0.316359162330627,  0.700386524200439};
bias_t_gen b1_diag_2to2[NHIDDEN] = {-0.000415743183112,  0.284048318862915};

//second batchnorm [channel][mean, weight/sqrt(var), bias]
bn_t_gen batch2_2to0[NHIDDEN][3] = {{ 0.099409341812134, 25.196693420410156, -1.848347663879395}, { 0.344610869884491,  5.781651973724365,  0.596007287502289}};

//2to1 linear layer
w2_gen_t w2_2to0[NHIDDEN*2*NOUT] = {-3.625000000000000,  0.625000000000000, -1.125000000000000, -2.625000000000000};
bias_t_gen b2_2to0[NOUT] = {-0.744833469390869};

//---- learned QAT scales (k = -log2(scale)); see types_generated.h ----
//  input_quant                        scale=2^--3 (8.000000000e+00) signed=True bits=6
//  output_quant                       scale=2^-4 (6.250000000e-02) signed=True bits=6
//  pmu_quant                          scale=2^-1 (5.000000000e-01) signed=True bits=12
//  net2to2.eq_layers.0.post_agg_quant scale=2^-5 (3.125000000e-02) signed=True bits=6
//  net2to2.eq_layers.0.act_layer      scale=2^-5 (3.125000000e-02) signed=True bits=6
//  agg_2to0.post_agg_quant            scale=2^-4 (6.250000000e-02) signed=True bits=6
//  net2to2.eq_layers.0 (weights)      scale=2^-5 (3.125000000e-02) signed=True bits=6
//  agg_2to0 (weights)                 scale=2^-3 (1.250000000e-01) signed=True bits=6
