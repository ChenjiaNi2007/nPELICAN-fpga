#include "../nPELICAN.h"
//model: fpga_model_qat
//nobj: 20

//normalization constants
//nobj avg = 49
internal_t invnave = 0.02040816326530612;
internal_t invnave2 = 0.00041649312786339027;

//first batchnorm [mean, weight/sqrt(var), bias]
weight_t batch1_2to2[3] = {5.432260131835938e+01, 3.303462847551368e-03, 1.722870022058487e-01};

//2to2 linear layer
weight_t w1_2to2[NHIDDEN*6] = {-0.582458257675171,  0.008636951446533,  0.044788122177124,  0.041936874389648, -0.472422122955322, -1.931932926177979,  0.018446922302246,  0.167083740234375, -0.122265100479126, -0.213846921920776,  1.175067424774170, -0.266506910324097};
bias_t b1_2to2[NHIDDEN] = {0.030149489641190, 0.093185551464558};
bias_t b1_diag_2to2[NHIDDEN] = { 0.014243968762457, -0.142088621854782};

//second batchnorm [channel][mean, weight/sqrt(var), bias]
weight_t batch2_2to0[NHIDDEN][3] = {{ 1.530051697045565e-02,  1.301911010742188e+02, -2.643904387950897e-01}, { 9.509509056806564e-02,  3.242964553833008e+01, -1.116147935390472e-01}};

//2to1 linear layer
weight_t w2_2to0[NHIDDEN*2*NOUT] = { 0.008559703826904, -2.889031887054443, -0.383216857910156, -1.464580535888672};
bias_t b2_2to0[NOUT] = {-0.339796096086502};

//---- learned QAT scales (k = -log2(scale)); see types_generated.h ----
//  input_quant                        scale=2^-10 (9.765625000e-04) signed=True bits=24
//  output_quant                       scale=2^-23 (1.192092896e-07) signed=True bits=24
//  net2to2.eq_layers.0.post_agg_quant scale=2^-18 (3.814697266e-06) signed=True bits=24
//  net2to2.eq_layers.0.act_layer      scale=2^-22 (2.384185791e-07) signed=True bits=24
//  agg_2to0.post_agg_quant            scale=2^-23 (1.192092896e-07) signed=True bits=24
//  net2to2.eq_layers.0 (weights)      scale=2^-22 (2.384185791e-07) signed=True bits=24
//  agg_2to0 (weights)                 scale=2^-21 (4.768371582e-07) signed=True bits=24
