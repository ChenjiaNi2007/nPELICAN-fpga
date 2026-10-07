#include "../nPELICAN.h"
//model: sps_ft_p12_s1
//nobj: 20

//normalization constants
//nobj avg = 49
norm_t invnave = 0.02040816326530612;
norm_t invnave2 = 0.00041649312786339027;

//first batchnorm [mean, weight/sqrt(var), bias]
bn_t_gen batch1_2to2[3] = {31.690393447875977,  0.042226832479560,  0.673716425895691};

//2to2 linear layer
w1_gen_t w1_2to2[NHIDDEN*6] = {-0.781250000000000, -0.406250000000000,  0.500000000000000,  0.531250000000000, -0.375000000000000,  0.968750000000000,  0.562500000000000,  0.968750000000000, -0.718750000000000, -0.718750000000000, -0.968750000000000, -0.562500000000000};
bias_t_gen b1_2to2[NHIDDEN] = {-0.136733710765839,  0.671883940696716};
bias_t_gen b1_diag_2to2[NHIDDEN] = {-0.459253996610641,  0.026930142194033};

//second batchnorm [channel][mean, weight/sqrt(var), bias]
bn_t_gen batch2_2to0[NHIDDEN][3] = {{ 0.208938807249069, 19.671388626098633, -0.714661180973053}, { 0.316957056522369,  5.034880161285400,  0.284536391496658}};

//2to1 linear layer
w2_gen_t w2_2to0[NHIDDEN*2*NOUT] = {-3.875000000000000,  0.500000000000000, -1.875000000000000, -3.875000000000000};
bias_t_gen b2_2to0[NOUT] = {-1.269961833953857};

//---- learned QAT scales (k = -log2(scale)); see types_generated.h ----
//  input_quant                        scale=2^--4 (1.600000000e+01) signed=True bits=6
//  output_quant                       scale=2^-3 (1.250000000e-01) signed=True bits=6
//  net2to2.eq_layers.0.post_agg_quant scale=2^-5 (3.125000000e-02) signed=True bits=6
//  net2to2.eq_layers.0.act_layer      scale=2^-4 (6.250000000e-02) signed=True bits=6
//  agg_2to0.post_agg_quant            scale=2^-6 (1.562500000e-02) signed=True bits=6
//  net2to2.eq_layers.0 (weights)      scale=2^-5 (3.125000000e-02) signed=True bits=6
//  agg_2to0 (weights)                 scale=2^-3 (1.250000000e-01) signed=True bits=6
