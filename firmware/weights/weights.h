#include "../nPELICAN.h"
//model: fpga_model_qat
//nobj: 20

//normalization constants
//nobj avg = 49
norm_t invnave = 0.02040816326530612;
norm_t invnave2 = 0.00041649312786339027;

//first batchnorm [mean, weight/sqrt(var), bias]
bn_t_gen batch1_2to2[3] = {23.177701950073242,  0.031948961457160,  0.782554686069489};

//2to2 linear layer
w1_gen_t w1_2to2[NHIDDEN*6] = { 0.250000000000000, -0.937500000000000,  0.750000000000000,  0.812500000000000, -0.250000000000000,  0.125000000000000,  0.687500000000000, -0.500000000000000, -1.437500000000000, -1.437500000000000,  0.250000000000000, -1.312500000000000};
bias_t_gen b1_2to2[NHIDDEN] = {-0.205158829689026,  0.191574335098267};
bias_t_gen b1_diag_2to2[NHIDDEN] = {0.041548635810614, 0.323916405439377};

//second batchnorm [channel][mean, weight/sqrt(var), bias]
bn_t_gen batch2_2to0[NHIDDEN][3] = {{ 0.493991792201996,  8.323738098144531,  1.081001043319702}, { 0.108491204679012, 11.582330703735352,  0.595229983329773}};

//2to1 linear layer
w2_gen_t w2_2to0[NHIDDEN*2*NOUT] = { 4.000000000000000, -0.750000000000000, -5.750000000000000, -0.750000000000000};
bias_t_gen b2_2to0[NOUT] = {-0.639774143695831};

//---- learned QAT scales (k = -log2(scale)); see types_generated.h ----
//  input_quant                        scale=2^--2 (4.000000000e+00) signed=True bits=6
//  output_quant                       scale=2^-3 (1.250000000e-01) signed=True bits=6
//  net2to2.eq_layers.0.post_agg_quant scale=2^-3 (1.250000000e-01) signed=True bits=6
//  net2to2.eq_layers.0.act_layer      scale=2^-3 (1.250000000e-01) signed=True bits=6
//  agg_2to0.post_agg_quant            scale=2^-6 (1.562500000e-02) signed=True bits=6
//  net2to2.eq_layers.0 (weights)      scale=2^-4 (6.250000000e-02) signed=True bits=6
//  agg_2to0 (weights)                 scale=2^-2 (2.500000000e-01) signed=True bits=6
