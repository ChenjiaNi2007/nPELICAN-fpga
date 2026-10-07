#include "../nPELICAN.h"
static_assert(NHIDDEN == 2 && NOUT == 5 && NPARTICLES == 16, "weights.h was exported for NHIDDEN=2 NOUT=5 NPARTICLES=16; firmware/nPELICAN.h or types_generated.h disagree -- re-export or fix the defines");
//model: fwsmoke_h5n16_h2_qat
//nobj: 16

//normalization constants
//nobj avg = 49
norm_t invnave = 0.02040816326530612;
norm_t invnave2 = 0.00041649312786339027;

//first batchnorm [mean, weight/sqrt(var), bias]
bn_t_gen batch1_2to2[3] = { 2.705972671508789e+01,  7.141791454414956e-03, -1.893252730369568e-01};

//2to2 linear layer
w1_gen_t w1_2to2[NHIDDEN*6] = { 0.031250000000000,  0.375000000000000,  0.343750000000000,  0.656250000000000, -0.312500000000000, -0.156250000000000, -0.875000000000000,  0.625000000000000, -0.187500000000000,  0.062500000000000, -0.531250000000000, -0.093750000000000};
bias1_t_gen b1_2to2[NHIDDEN] = {0.046726532280445, 0.208166047930717};
bias1_t_gen b1_diag_2to2[NHIDDEN] = {-0.097096584737301, -0.067727349698544};

//second batchnorm [channel][mean, weight/sqrt(var), bias]
bn_t_gen batch2_2to0[NHIDDEN][3] = {{ 0.020246170461178, 11.057423591613770,  0.045179963111877}, { 0.523547291755676,  1.987672209739685, -0.025686571374536}};

//2to1 linear layer
w2_gen_t w2_2to0[NHIDDEN*2*NOUT] = {-0.250000000000000, -0.312500000000000,  0.375000000000000, -0.750000000000000,  0.312500000000000, -0.125000000000000, -1.312500000000000, -0.937500000000000, -0.000000000000000,  0.750000000000000,  0.625000000000000, -1.437500000000000,  1.187500000000000,  0.375000000000000,  0.062500000000000, -0.500000000000000,  0.812500000000000,  1.562500000000000, -1.937500000000000,  1.250000000000000};
bias2_t_gen b2_2to0[NOUT] = {-0.005278545431793, -0.023010557517409,  0.036803245544434,  0.016502050682902, -0.021758705377579};

//---- learned QAT scales (k = -log2(scale)); see types_generated.h ----
//  input_quant                        scale=2^--7 (1.280000000e+02) signed=True bits=6
//  output_quant                       scale=2^-4 (6.250000000e-02) signed=True bits=6
//  pmu_quant                          scale=2^-0 (1.000000000e+00) signed=True bits=12
//  net2to2.eq_layers.0.post_agg_quant scale=2^-1 (5.000000000e-01) signed=True bits=6
//  net2to2.eq_layers.0.act_layer      scale=2^-2 (2.500000000e-01) signed=True bits=6
//  agg_2to0.post_agg_quant            scale=2^-5 (3.125000000e-02) signed=True bits=6
//  net2to2.eq_layers.0 (weights)      scale=2^-5 (3.125000000e-02) signed=True bits=6
//  agg_2to0 (weights)                 scale=2^-4 (6.250000000e-02) signed=True bits=6
