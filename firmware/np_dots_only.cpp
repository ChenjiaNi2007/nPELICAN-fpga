#include "nPELICAN.h"

// ============================================================================
// Standalone dot-product front-end (Part B: absolute resource cost of JUST the
// dot portion of the model). This is the momentum prep + symmetric Minkowski
// dot4 lifted out of firmware/nPELICAN.cpp (lines ~67-143, == np_dots in
// nPELICAN_split.cpp) into its OWN top function, so csynth reports the dot
// stage's resources in isolation — no BatchNorm, aggregation, dense layers,
// ReLU, or learned weights. There is no weights.h include: the dot front-end
// is parameter-free (a good self-check that the stage is truly isolated).
//
// Output is the full 22x22 = 484 dot matrix, row-major i*NPARTICLES2+j, each
// rounded once onto the input_quant grid (dot_t). Types (input_t, dot_t,
// nobj_t) and sizes come from nPELICAN.h / weights/types_generated.h, so the
// dot4 multiplier width tracks the current checkpoint exactly (presently
// input_t=ap_fixed<18,12>, dot_t=ap_fixed<6,8> LSB=4).
//
// This file is a MEASUREMENT tool, not part of the datapath: nothing here feeds
// the network. It mirrors np_dots byte-for-byte in arithmetic so its numbers are
// directly comparable to the split build's np_dots row. Build with
// -DNPELICAN_CONST_BEAMS to measure the constant-beam saving (Lever 5).
// ============================================================================

// Minkowski dot for (E, px, py, pz): products/sums are exact in HLS's promoted
// fixed-point type, rounded ONCE into dot_t (input_quant grid, AP_RND_CONV).
static void dot4_local(input_t p1[4], input_t p2[4], dot_t& dot) {
    dot = p1[0]*p2[0] - p1[1]*p2[1] - p1[2]*p2[2] - p1[3]*p2[3];
}

void np_dots_only(
    input_t model_input[(NPARTICLES)*4],   // 20 particle 4-vectors (E,px,py,pz)
    input_t beam_input[2*4],               // 2 beam spurions
    nobj_t  nobj,                          // particle count (exact, ap_uint<5>)
    dot_t   dots_out[(NPARTICLES2)*(NPARTICLES2)]   // 484 dots, row-major i*22+j
) {
    #pragma HLS ARRAY_RESHAPE variable=model_input complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=beam_input complete dim=0
    #pragma HLS ARRAY_PARTITION variable=dots_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=model_input,beam_input,dots_out
    #pragma HLS PIPELINE II=1

    // nobj remap + 0/1 mask (same comparators/wiring as the full top). The mask
    // zeros padded particles' momenta in P1Prep; the dot matrix itself is not
    // masked here (in the full model that happens downstream at BN1).
    if (nobj != 0) {
        if (nobj < NPARTICLES) nobj += (NPARTICLES2 - NPARTICLES);
        else                   nobj  = NPARTICLES2;
    }
    ap_uint<1> nobjmask[NPARTICLES2][NPARTICLES2];
    #pragma HLS ARRAY_PARTITION variable=nobjmask complete dim=0
    for (unsigned int i = 0; i < NPARTICLES2; i++)
        for (unsigned int j = 0; j < NPARTICLES2; j++)
            nobjmask[i][j] = (i < nobj && j < nobj) ? 1 : 0;

    // Momentum prep: particles land at p1 index i+2 (masked), beams at 0,1.
    input_t p1[NPARTICLES2][4];
    #pragma HLS ARRAY_PARTITION variable=p1 complete dim=0
    P1Prep: for (unsigned int i = 0; i < NPARTICLES; i++) {
    #pragma HLS unroll
        for (unsigned int k = 0; k < 4; k++) {
        #pragma HLS unroll
            p1[i + (NPARTICLES2 - NPARTICLES)][k] = model_input[i*4+k] * nobjmask[i][0];
        }
    }
#ifdef NPELICAN_CONST_BEAMS
    // Lever 5: beams hardwired to the fixed spurions -> the 43 beam-involving
    // dots fold to E-+pz adds (no multipliers), ~172 DSP recovered. beam_input
    // is ignored. Bit-identical to the runtime-beam path driven at |beta|=0.
    const input_t const_beams[2][4] = {{1, 0, 0, 1}, {1, 0, 0, -1}};
    BeamPrep: for (unsigned int i = 0; i < 2; i++) {
        #pragma HLS unroll
        for (unsigned int k = 0; k < 4; k++) {
            #pragma HLS unroll
            p1[i][k] = const_beams[i][k];
        }
    }
#else
    BeamPrep: for (unsigned int i = 0; i < 2; i++) {
        #pragma HLS unroll
        for (unsigned int k = 0; k < 4; k++) {
            #pragma HLS unroll
            p1[i][k] = beam_input[i*4+k];
        }
    }
#endif

#ifdef NPELICAN_BLOCK_FP
    // Lever 7: per-particle block-FP encode, then mantissa dot + realign. Mirrors
    // np_dots in nPELICAN_split.cpp, so the isolated numbers stay comparable. The
    // encoder (22 priority encoders + 88 shifts) IS counted here — it is part of
    // the dot front end's cost under block-FP. See firmware/np_blockfp.h.
    mant_t p1m[NPARTICLES2][4];
    bexp_t p1e[NPARTICLES2];
    #pragma HLS ARRAY_PARTITION variable=p1m complete dim=0
    #pragma HLS ARRAY_PARTITION variable=p1e complete dim=0
    np_bfp_encode_all(p1, p1m, p1e);
#endif

    // Symmetric dot4: compute the upper triangle (j>=i) and mirror into the
    // lower triangle (pure wiring, no hardware) — halves the dot4 multipliers.
    for (unsigned int i = 0; i < NPARTICLES2; i++) {
        #pragma HLS unroll
        for (unsigned int j = i; j < NPARTICLES2; j++) {
            #pragma HLS unroll
#ifdef NPELICAN_BLOCK_FP
            Dot: np_bfp_dot4(p1m[i], p1e[i], p1m[j], p1e[j], dots_out[i*NPARTICLES2+j]);
#else
            Dot: dot4_local(p1[i], p1[j], dots_out[i*NPARTICLES2+j]);
#endif
            if (j != i) dots_out[j*NPARTICLES2+i] = dots_out[i*NPARTICLES2+j];
        }
    }
}
