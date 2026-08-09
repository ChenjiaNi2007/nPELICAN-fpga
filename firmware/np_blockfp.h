#ifndef NP_BLOCKFP_H_
#define NP_BLOCKFP_H_

// ============================================================================
// Lever 7: per-particle BLOCK FLOATING POINT momenta.
//
// Included from nPELICAN.h AFTER weights/types_generated.h and the NPARTICLES2 /
// input_t definitions. The whole file is inert unless the loader emitted
// NPELICAN_BLOCK_FP, i.e. unless the exported checkpoint was trained with
// --pmu-block-fp. Every uniform-grid checkpoint compiles byte-identically to
// before, and the float-reference build (types_float.h) never defines the macro.
//
// Representation (mirrors PELICAN-nano/src/layers/blockfp.py exactly):
//
//     e_i = clamp(floor(log2 |E_i|), EXP_MIN, EXP_MAX)     one exponent per particle
//     m_ik = p_ik * 2^-e_i   on a signed W-bit, I=2 grid   LSB 2^-(W-2) of 2^e
//     p_ik ~ m_ik * 2^e_i
//
// I=2 provably cannot overflow: for a physical particle E >= |p_k| and
// 2^e <= E < 2^(e+1), so |m_k| < 2 always. (If EXP_MAX clamps a very hard
// particle, |m| can exceed 2 and AP_SAT saturates -- the same behaviour the
// uniform grid already has at its clip point.)
//
// The dot then factorises, which is the entire point:
//
//     d_ij = 2^(e_i + e_j) * (m_i . g . m_j)
//
// so the multipliers are mant_t x mant_t (W x W) instead of input_t x input_t,
// while each particle keeps full RELATIVE precision. The mantissa dot is EXACT
// (mdot_t), the realignment is EXACT (dotalign_t), and the result rounds ONCE
// into dot_t with AP_RND_CONV -- the same single rounding PyTorch applies with
// input_quant. Rounding early, i.e. casting the mantissa dot to dot_t before the
// shift, would quantize on a grid 2^(e_i+e_j) too fine and is the one way to get
// this wrong.
//
// MASKING INVARIANT: a padded particle is zeroed in P1Prep, so |E| = 0 gives
// e = EXP_MIN and m = 0 exactly, hence d = 0 exactly in dot_t. Preserved.
// ============================================================================

#ifdef NPELICAN_BLOCK_FP

// Per-particle encode: (E, px, py, pz) -> (mantissa[4], exponent).
//
// The exponent is a comparison cascade against the EXP_MAX - EXP_MIN constant
// thresholds 2^(EXP_MIN+1) .. 2^EXP_MAX, which HLS maps to a priority encoder
// (equivalently one leading-zero count on the energy word). It reproduces
// clamp(floor(log2|E|), EXP_MIN, EXP_MAX) exactly, including |E| = 0 -> EXP_MIN.
//
// The mantissa shift is taken in mraw_t, which carries EXP_MAX extra fractional
// bits, so p >> e loses nothing off the bottom and the ONLY rounding on this path
// is the final cast to mant_t (AP_RND_CONV, AP_SAT).
static void np_bfp_encode(input_t p[4], mant_t m[4], bexp_t& e) {
#pragma HLS INLINE
    input_t aE = (p[0] < (input_t)0) ? (input_t)(-p[0]) : p[0];

    bexp_t ex = (bexp_t)NPELICAN_BFP_EXP_MIN;
ExpCascade:
    for (int t = NPELICAN_BFP_EXP_MIN + 1; t <= NPELICAN_BFP_EXP_MAX; t++) {
    #pragma HLS unroll
        // 2^t held in a type wide enough for any EXP_MAX, independent of input_t's
        // integer width (a threshold built as input_t(1<<t) would silently saturate
        // if a checkpoint ever trained with EXP_MAX >= input_t's integer bits).
        const ap_ufixed<NPELICAN_BFP_EXP_MAX + 2, NPELICAN_BFP_EXP_MAX + 2> thr =
            (ap_ufixed<NPELICAN_BFP_EXP_MAX + 2, NPELICAN_BFP_EXP_MAX + 2>)(1u << t);
        if (aE >= thr) ex = (bexp_t)t;
    }
    e = ex;

Mantissa:
    for (int k = 0; k < 4; k++) {
    #pragma HLS unroll
        mraw_t mr = (mraw_t)p[k];   // exact widen
        mr >>= ex;                   // exact: mraw_t has EXP_MAX spare fractional bits
        m[k] = (mant_t)mr;           // the single rounding on the encode path
    }
}

static void np_bfp_encode_all(input_t p[NPARTICLES2][4],
                              mant_t m[NPARTICLES2][4],
                              bexp_t e[NPARTICLES2]) {
#pragma HLS INLINE
BfpEncodeAll:
    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
        np_bfp_encode(p[i], m[i], e[i]);
    }
}

// Block-FP Minkowski dot: d = 2^(e1+e2) * (m1 . g . m2), rounded once into dot_t.
static void np_bfp_dot4(mant_t m1[4], bexp_t e1,
                        mant_t m2[4], bexp_t e2, dot_t& dot) {
#pragma HLS INLINE
    // Exact: mant_t products are <2W,4>; the 4-term signed sum needs 2 more integer
    // bits, and |md| < 4 * 2 * 2 = 16 fits mdot_t's I with room to spare.
    mdot_t md = m1[0] * m2[0] - m1[1] * m2[1] - m1[2] * m2[2] - m1[3] * m2[3];

    // Realign BEFORE the dot_t cast. dotalign_t adds 2*EXP_MAX integer bits, so the
    // shift never saturates and never drops a bit.
    bshift_t s = e1 + e2;
    dotalign_t a = md;
    a <<= s;

    dot = a;   // ONE rounding, AP_RND_CONV -- matches PyTorch's input_quant
}

#endif  // NPELICAN_BLOCK_FP
#endif  // NP_BLOCKFP_H_
