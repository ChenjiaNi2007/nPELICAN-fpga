#include <iostream>
#include "nPELICAN.h"
#include "weights/weights.h"

// Lever 8: T0 from the loader-generated BN1 ROM when weights.h provides one —
// never in the float-reference build. Mirrors nPELICAN.cpp.
#if defined(NPELICAN_BN1_ROM) && !defined(NPELICAN_FLOAT_BUILD) && !defined(NPELICAN_NO_BN1_ROM)
#define NP_USE_BN1_ROM 1
#endif

#ifndef __SYNTHESIS__
#include <cstdio>
FILE* npelican_dump_fp = nullptr;
// DOTS-LEVEL test hook (csim only, zero synthesis impact): see nPELICAN.cpp.
dot_t* npelican_dots_override = nullptr;
#endif

// ============================================================================
// SPLIT build of the nPELICAN datapath (docs/FUNCTION_SPLIT.md).
//
// Same top-level function, arithmetic, types, rounding and normalize-late
// placement as firmware/nPELICAN.cpp — the ONLY change is that each datapath
// stage is a separate function with `#pragma HLS INLINE off`, so the csynth
// report attributes LUT/FF/DSP/latency to each stage (grp_np_* instances)
// instead of one flat top. Stage boundaries sit exactly on the quantization
// points, so every crossing array already has its intended narrow type; all
// crossing arrays are completely partitioned on both sides (wires, no BRAM).
// The basis tensor T (pure wiring, 2904 elems) stays INTERNAL to np_eq2to2 so
// it never becomes a function port.
//
// Exactly ONE of {nPELICAN.cpp, nPELICAN_split.cpp} goes into a build
// (build_prj.tcl split=1 / build_local.sh split): they define the same
// symbols. nPELICAN.cpp remains the reference; if this file ever disagrees
// with it (local golden gate compares them byte-for-byte), this file is the
// one that is wrong.
// ============================================================================

// ---------------------------------------------------------------------------
// Stage isolation (one-boundary-at-a-time marginal costs; FUNCTION_SPLIT.md).
// Default: ALL six stages split (full attribution build). Define exactly one
// NPELICAN_SPLIT_ONLY_<STAGE> (build_prj.tcl split_only=<stage>) to keep ONLY
// that stage as a function and force-inline the other five back into the top:
// the isolated stage is then measured in monolith context (true marginal cost),
// and the run's total minus the monolith baseline is that ONE boundary's
// overhead. C-sim output is identical in every configuration — inlining does
// not change arithmetic.
// ---------------------------------------------------------------------------
#if defined(NPELICAN_SPLIT_ONLY_DOTS) || defined(NPELICAN_SPLIT_ONLY_BN1) || \
    defined(NPELICAN_SPLIT_ONLY_AGG2TO2) || defined(NPELICAN_SPLIT_ONLY_EQ2TO2) || \
    defined(NPELICAN_SPLIT_ONLY_AGG2TO0) || defined(NPELICAN_SPLIT_ONLY_OUT2TO0)
  #define NP_ISOLATE 1
#else
  #define NP_ISOLATE 0
#endif
#if !NP_ISOLATE || defined(NPELICAN_SPLIT_ONLY_DOTS)
  #define NP_SPLIT_DOTS 1
#else
  #define NP_SPLIT_DOTS 0
#endif
#if !NP_ISOLATE || defined(NPELICAN_SPLIT_ONLY_BN1)
  #define NP_SPLIT_BN1 1
#else
  #define NP_SPLIT_BN1 0
#endif
#if !NP_ISOLATE || defined(NPELICAN_SPLIT_ONLY_AGG2TO2)
  #define NP_SPLIT_AGG2TO2 1
#else
  #define NP_SPLIT_AGG2TO2 0
#endif
#if !NP_ISOLATE || defined(NPELICAN_SPLIT_ONLY_EQ2TO2)
  #define NP_SPLIT_EQ2TO2 1
#else
  #define NP_SPLIT_EQ2TO2 0
#endif
#if !NP_ISOLATE || defined(NPELICAN_SPLIT_ONLY_AGG2TO0)
  #define NP_SPLIT_AGG2TO0 1
#else
  #define NP_SPLIT_AGG2TO0 0
#endif
#if !NP_ISOLATE || defined(NPELICAN_SPLIT_ONLY_OUT2TO0)
  #define NP_SPLIT_OUT2TO0 1
#else
  #define NP_SPLIT_OUT2TO0 0
#endif

// ---------------------------------------------------------------------------
// split=2: triangular symmetric crossings (NPELICAN_SPLIT_TRI; FUNCTION_SPLIT.md).
// Mechanism test for the split's lost symmetry-CSE: dots and T0 (batch1 before
// Lever 8) are symmetric, but as full 484-element ports the consumers cannot know element
// (i,j) equals (j,i) — the identity lives in the PRODUCER's mirror write, and
// HLS's expression analysis stops at a non-inlined boundary. So the shared
// 2->2 MAC products (w1[h*6+0]*T0[i,j] is the same value for (i,j) and
// (j,i)) get duplicated hardware in split=1. Under this flag the two arrays
// cross as 253-element upper triangles and EVERY access goes through
// NP_SYMIDX, which maps (i,j) and (j,i) to the SAME element — the identity is
// back in the consumer's scope, arithmetic completely unchanged (csim output
// stays byte-identical to the monolith; gate it like any datapath edit).
// If split=2 closes most of the split-vs-monolith LUT gap, lost-CSE is the
// dominant mechanism. Not combinable with split_only (isolation runs stay
// plain split=1). nobjmask is left as a full port on purpose: its symmetric
// products are 1-bit gates on scalars (cheap), and changing one boundary at a
// time is the point.
// ---------------------------------------------------------------------------
#define NP_UT(i, j) ((i)*NPARTICLES2 - ((i)*((i)-1))/2 + ((j)-(i)))
#ifdef NPELICAN_SPLIT_TRI
  #define NP_SYMSZ ((NPARTICLES2)*((NPARTICLES2)+1)/2)
  #define NP_SYMIDX(i, j) ((i) <= (j) ? NP_UT(i, j) : NP_UT(j, i))
#else
  #define NP_SYMSZ ((NPARTICLES2)*(NPARTICLES2))
  #define NP_SYMIDX(i, j) ((i)*NPARTICLES2 + (j))
#endif

void dot4(input_t p1[4], input_t p2[4], dot_t& dot) {
// Input in the form E, px, py, pz. The Minkowski dot is computed in HLS's exact
// promoted type (products/sums of fixed-point are exact) and rounded once into
// dot_t (the input_quant 2^-k grid, AP_RND_CONV).
dot = p1[0]*p2[0]-p1[1]*p2[1]-p1[2]*p2[2]-p1[3]*p2[3];
}

// Stage 1: momentum prep (masked particles + beam spurions) + symmetric dot4
// front-end. Dominant DSP consumer (~NPARTICLES2²/2 multiplies).
void np_dots(
    input_t model_input[(NPARTICLES)*4],
    input_t beam_input[2*4],
    ap_uint<1> nobjmask[NPARTICLES2][NPARTICLES2],
    dot_t dots[NP_SYMSZ]
) {
#if NP_SPLIT_DOTS
    #pragma HLS INLINE off
    #pragma HLS PIPELINE II=1
#else
    #pragma HLS INLINE
#endif
    #pragma HLS ARRAY_PARTITION variable=nobjmask complete dim=0
    #pragma HLS ARRAY_PARTITION variable=dots complete dim=0

    input_t p1[(NPARTICLES2)][4];
    #pragma HLS ARRAY_PARTITION variable=p1 complete dim=0
    P1Prep: for (unsigned int i = 0; i < NPARTICLES; i++) {
    #pragma HLS unroll
      for (unsigned int k = 0; k < 4; k++){
      #pragma HLS unroll
        p1[(i + (NPARTICLES2 - NPARTICLES))][k] = model_input[i*(4)+k]*nobjmask[i][0];
      }
    }
#ifdef NPELICAN_CONST_BEAMS
    //Lever 5 (deployment build): beams hardwired to the fixed spurions —
    //beam dots fold to E∓pz adds, ~172 DSP recovered. Mirrors nPELICAN.cpp;
    //build WITHOUT this flag for the equivariance harness (boosted beams).
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
    //Lever 7: encode each particle once into (mantissa[4], exponent), then dot the
    //mantissas and realign. Mirrors nPELICAN.cpp; see firmware/np_blockfp.h.
    mant_t p1m[(NPARTICLES2)][4];
    bexp_t p1e[(NPARTICLES2)];
    #pragma HLS ARRAY_PARTITION variable=p1m complete dim=0
    #pragma HLS ARRAY_PARTITION variable=p1e complete dim=0
    np_bfp_encode_all(p1, p1m, p1e);
#endif

    //dot4 is symmetric: compute only the upper triangle and mirror (pure wiring).
    //Under NPELICAN_SPLIT_TRI there is nothing to mirror: (j,i) IS element (i,j).
    for(unsigned int i = 0; i < NPARTICLES2; i++){
      #pragma HLS unroll
      for(unsigned int j = i; j < NPARTICLES2; j++){
        #pragma HLS unroll
#ifdef NPELICAN_BLOCK_FP
        Dot: np_bfp_dot4(p1m[i], p1e[i], p1m[j], p1e[j], dots[NP_SYMIDX(i, j)]);
#else
        Dot: dot4(p1[i], p1[j], dots[NP_SYMIDX(i, j)]);
#endif
#ifndef NPELICAN_SPLIT_TRI
        if (j != i) dots[NP_SYMIDX(j, i)] = dots[NP_SYMIDX(i, j)];
#endif
      }
    }
}

// Stage 2 (Lever 8): BN1 T0 lookup + raw masked dot sums. batch1 is never
// materialized (see nPELICAN.cpp for the derivation): T0 = m ? ROM[code(d)] : 0
// (upper triangle, mirrored), and the aggregation sums run on the RAW dots, which
// are exact integers on the dot grid. The BN1 affine is applied once per aggregate
// in np_agg2to2.
void np_bn1(
    dot_t dots[NP_SYMSZ],
    ap_uint<1> nobjmask[NPARTICLES2][NPARTICLES2],
    t2_t T0[NP_SYMSZ],
    accdot2_t &dsum,
    accdotrow_t rowsum[NPARTICLES2]
) {
#if NP_SPLIT_BN1
    #pragma HLS INLINE off
    #pragma HLS PIPELINE II=1
#else
    #pragma HLS INLINE
#endif
    #pragma HLS ARRAY_PARTITION variable=dots complete dim=0
    #pragma HLS ARRAY_PARTITION variable=nobjmask complete dim=0
    #pragma HLS ARRAY_PARTITION variable=T0 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=rowsum complete dim=0
#ifdef NP_USE_BN1_ROM
    #pragma HLS ARRAY_PARTITION variable=bn1_t0_rom complete dim=0
#else
    #pragma HLS ARRAY_PARTITION variable=batch1_2to2 complete dim=0
    //fallback: per-element arithmetic T0, SINGLE rounding straight to t2_t.
    const bn_t_gen bn1_beta = batch1_2to2[2] - batch1_2to2[0]*batch1_2to2[1];
#endif

    //T0 = post_agg quant of batch1 (symmetric: upper triangle + mirror).
    for(unsigned int i = 0; i < NPARTICLES2; i++){
      #pragma HLS unroll
      for(unsigned int j = i; j < NPARTICLES2; j++){
        #pragma HLS unroll
#ifdef NP_USE_BN1_ROM
        ap_uint<NPELICAN_DOT_W> code = dots[NP_SYMIDX(i, j)].range(NPELICAN_DOT_W-1, 0);
        t2_t t0 = nobjmask[i][j] ? bn1_t0_rom[code] : (t2_t)0;
#else
        t2_t t0 = (t2_t)((dots[NP_SYMIDX(i, j)] * batch1_2to2[1] + bn1_beta)*nobjmask[i][j]);
#endif
        T0[NP_SYMIDX(i, j)] = t0;
#ifndef NPELICAN_SPLIT_TRI
        if (j != i) T0[NP_SYMIDX(j, i)] = t0;
#endif
      }
    }

    //raw masked dot sums (exact): Σ_i d_ij·m per column, then Σ_ij d·m = Σ_j rowsum[j]
    //(the same exact integer; a 22-term tree instead of a 484-term one).
    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      rowsum[i] = 0;
    }
    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      for (unsigned int j = 0; j < NPARTICLES2; j++) {
      #pragma HLS unroll
        AggJdot: rowsum[j] += dots[NP_SYMIDX(i, j)]*nobjmask[i][j];
      }
    }
    dsum = 0;
    for (unsigned int j = 0; j < NPARTICLES2; j++) {
    #pragma HLS unroll
      AggMJ: dsum += rowsum[j];
    }
}

// Stage 3 (Lever 8): BN1 affine applied ONCE to each raw aggregate with the 1/N̄
// normalization pre-folded by the loader, then ONE rounding onto the t2 grid:
//   jmass    = t2(dsum·s/N̄² + ncount²·β'/N̄²)
//   jdotp[j] = t2((rowsum[j]·s/N̄ + ncount·β'/N̄)·m_jj)
void np_agg2to2(
    accdot2_t dsum,
    accdotrow_t rowsum[NPARTICLES2],
    ap_uint<1> nobjmask[NPARTICLES2][NPARTICLES2],
    ap_uint<5> ncount,
    ap_uint<9> ncount2,
    t2_t &jmass,
    t2_t jdotp[NPARTICLES2]
) {
#if NP_SPLIT_AGG2TO2
    #pragma HLS INLINE off
    #pragma HLS PIPELINE II=1
#else
    #pragma HLS INLINE
#endif
    #pragma HLS ARRAY_PARTITION variable=rowsum complete dim=0
    #pragma HLS ARRAY_PARTITION variable=nobjmask complete dim=0
    #pragma HLS ARRAY_PARTITION variable=jdotp complete dim=0

    jmass = (t2_t)(dsum*bn1_s_all + ncount2*bn1_b_all);
    for( unsigned int i = 0; i < NPARTICLES2; i++){
    #pragma HLS unroll
      jdotp[i] = (t2_t)((rowsum[i]*bn1_s_row + ncount*bn1_b_row)*nobjmask[i][i]);
    }
}

// Stage 4: basis ops T (INTERNAL — pure wiring) + "dense" 2->2 mix + ReLU +
// act_layer quantization. MAC accumulates in mac2_t (exact product width), so
// the only rounding is the relu_t cast.
void np_eq2to2(
    t2_t T0[NP_SYMSZ],
    t2_t jmass,
    t2_t jdotp[NPARTICLES2],
    ap_uint<1> nobjmask[NPARTICLES2][NPARTICLES2],
    relu_t Tp_q[NPARTICLES2][NPARTICLES2][NHIDDEN]
) {
#if NP_SPLIT_EQ2TO2
    #pragma HLS INLINE off
    #pragma HLS PIPELINE II=1
#else
    #pragma HLS INLINE
#endif
    #pragma HLS ARRAY_PARTITION variable=T0 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=jdotp complete dim=0
    #pragma HLS ARRAY_PARTITION variable=nobjmask complete dim=0
    #pragma HLS ARRAY_PARTITION variable=Tp_q complete dim=0
    #pragma HLS ARRAY_PARTITION variable=w1_2to2 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=b1_2to2 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=b1_diag_2to2 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=b1_diag_total_2to2 complete dim=0

    // T0 = p_i . p_j ; T1 = (J.p_i) d_ij ; T2 = J.p_j ; T3 = J.p_i ; T4 = M_J ; T5 = M_J d_ij
    t2_t T[NPARTICLES2][NPARTICLES2][6];
    #pragma HLS ARRAY_PARTITION variable=T complete dim=0
    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      for (unsigned int j = 0; j < NPARTICLES2; j++) {
    #pragma HLS unroll
        for (unsigned int b = 0; b < 6; b++) {
    #pragma HLS unroll
          T[i][j][b] = 0;
        }
      }
    }

    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      for (unsigned int j = 0; j < NPARTICLES2; j++) {
      #pragma HLS unroll
        LinEq2to2_0: T[i][j][0] = T0[NP_SYMIDX(i, j)];   // post_agg quant of batch1 (np_bn1)
        LinEq2to2_1: T[i][j][4] = jmass*nobjmask[i][j];
        LinEq2to2_4: T[i][j][3] = jdotp[i];
        LinEq2to2_5: T[i][j][2] = jdotp[j];
      }
    }

    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      LinEq2to2_2: T[i][i][5] = jmass*nobjmask[i][i];
      LinEq2to2_3: T[i][i][1] = jdotp[i];
    }

    mac2_t Tp[NPARTICLES2][NPARTICLES2][NHIDDEN];
    #pragma HLS ARRAY_PARTITION variable=Tp complete dim=0
#ifdef NPELICAN_MAC_DSP
    //Lever 6 experiment: force the 2->2 MAC multiplies into DSP48s (LUT is the
    //binding resource; this stage holds 51% of it). Mirrors nPELICAN.cpp.
    #pragma HLS BIND_OP variable=Tp op=mul impl=dsp
#endif

    // MAC accumulators start at 0: mac2_t is the EXACT product-sum type (w1 x t2 grid).
    // The biases (bias_t_gen, sticky-bit encoded: one bit finer than the MAC grid) are added
    // in the final expression right before the relu_t cast.
    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      for (unsigned int j = 0; j < NPARTICLES2; j++) {
      #pragma HLS unroll
        for (unsigned int h = 0; h < NHIDDEN; h++) {
        #pragma HLS unroll
          Tp[i][j][h] = 0;
          }
        }
      }

    // 2->2 weights (frozen element order w1_2to2[h*6+b])
    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      for (unsigned int j = 0; j < NPARTICLES2; j++) {
      #pragma HLS unroll
        for (unsigned int h = 0; h < NHIDDEN; h++) {
        #pragma HLS unroll
          for (unsigned int b = 0; b < 6; b++) {
          #pragma HLS unroll
            Mult2to2: Tp[i][j][h] += w1_2to2[(h*6)+b]*T[i][j][b];
          }
        }
      }
    }

    // + bias (masked exactly as before: b1·m_ij off the diagonal, (b1+b1_diag)·m_ii on it —
    // the diagonal constant is the loader's b1_diag_total_2to2, a select between two
    // constants, no extra add), quantize onto the act_layer grid (relu_t, AP_RND_CONV), then
    // ReLU. Q is monotone with Q(0)=0, so max(0, Q(x)) == Q(max(0, x)): identical to
    // ReLU-then-quantize. The biases are STICKY-BIT encoded (types_generated.h: b_hi on
    // 2^-Fh + one sticky LSB 2^-(Fh+1), Fh >= mac2_F and >= relu_F+1), so this add is
    // narrow yet rounds exactly like the exact float32 bias would.
    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      for (unsigned int j = 0; j < NPARTICLES2; j++) {
      #pragma HLS unroll
        for (unsigned int h = 0; h < NHIDDEN; h++) {
        #pragma HLS unroll
          relu_t q = (relu_t)(Tp[i][j][h]
                              + (i == j ? b1_diag_total_2to2[h] : b1_2to2[h])*nobjmask[i][j]);
          if (q < 0) q = 0;
          Tp_q[i][j][h] = q;
        }
      }
    }
}

// Stage 5: 2->0 aggregation with BN2 collapsed past the sums (see nPELICAN.cpp
// for the derivation): raw sum + trace of Tp_q, then per-channel affine
// s·A + β'·count, then normalize-late — ONE rescale onto the t0 grid.
void np_agg2to0(
    relu_t Tp_q[NPARTICLES2][NPARTICLES2][NHIDDEN],
    ap_uint<1> nobjmask[NPARTICLES2][NPARTICLES2],
    ap_uint<5> ncount,
    ap_uint<9> ncount2,
    t0_t R[NHIDDEN][2]
) {
#if NP_SPLIT_AGG2TO0
    #pragma HLS INLINE off
    #pragma HLS PIPELINE II=1
#else
    #pragma HLS INLINE
#endif
    #pragma HLS ARRAY_PARTITION variable=Tp_q complete dim=0
    #pragma HLS ARRAY_PARTITION variable=nobjmask complete dim=0
    #pragma HLS ARRAY_PARTITION variable=R complete dim=0
    #pragma HLS ARRAY_PARTITION variable=batch2_2to0 complete dim=0

    //folded BN2 bias β'_h = β_h − μ_h·s_h (compile-time constant per channel).
    bn_t_gen bn2_beta[NHIDDEN];
    #pragma HLS ARRAY_PARTITION variable=bn2_beta complete dim=0
    for (unsigned int h = 0; h < NHIDDEN; h++) {
    #pragma HLS unroll
      bn2_beta[h] = batch2_2to0[h][2] - batch2_2to0[h][0]*batch2_2to0[h][1];
    }

    accrelu_t    A_sum[NHIDDEN];
    accrelurow_t A_trace[NHIDDEN];
    #pragma HLS ARRAY_PARTITION variable=A_sum complete dim=0
    #pragma HLS ARRAY_PARTITION variable=A_trace complete dim=0
    for (unsigned int h = 0; h < NHIDDEN; h++) {
    #pragma HLS unroll
      A_sum[h]   = 0;
      A_trace[h] = 0;
    }

    //total sum (Σ_ij Tp_q)
    for (unsigned int h = 0; h < NHIDDEN; h++) {
    #pragma HLS unroll
      for (unsigned int i = 0; i < NPARTICLES2; i++) {
      #pragma HLS unroll
        for (unsigned int j = 0; j < NPARTICLES2; j++) {
        #pragma HLS unroll
            LinEq2to0: A_sum[h] += Tp_q[i][j][h] * nobjmask[i][j];
        }
      }
    }

    //trace (Σ_i Tp_q[i][i])
    for (unsigned int h = 0; h < NHIDDEN; h++) {
    #pragma HLS unroll
      for (unsigned int i = 0; i < NPARTICLES2; i++) {
      #pragma HLS unroll
        A_trace[h] += Tp_q[i][i][h] * nobjmask[i][i];
      }
    }

    //unmasked-entry counts ncount/ncount2: computed once in the top (shared with np_agg2to2).
    for (unsigned int h = 0; h < NHIDDEN; h++) {
    #pragma HLS unroll
      R[h][0] = (t0_t)((batch2_2to0[h][1]*A_sum[h]   + bn2_beta[h]*ncount2) * invnave2);
      R[h][1] = (t0_t)((batch2_2to0[h][1]*A_trace[h] + bn2_beta[h]*ncount ) * invnave);
    }
}

// Stage 6: final 2->0 dense MAC in mac0_t (exact product width). Rp is passed
// out (not just model_out) so the top's csim stage dump can print it; the one
// output_quant rounding (result_t cast) happens in the top, as in the monolith.
void np_out2to0(
    t0_t R[NHIDDEN][2],
    mac0_t Rp[NOUT]
) {
#if NP_SPLIT_OUT2TO0
    #pragma HLS INLINE off
    #pragma HLS PIPELINE II=1
#else
    #pragma HLS INLINE
#endif
    #pragma HLS ARRAY_PARTITION variable=R complete dim=0
    #pragma HLS ARRAY_PARTITION variable=Rp complete dim=0
    #pragma HLS ARRAY_PARTITION variable=w2_2to0 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=b2_2to0 complete dim=0

    // accumulator starts at 0 (mac0_t = exact product-sum type); the sticky-bit-encoded
    // bias b2 is added right before the result_t cast.
    for (unsigned int o = 0; o < NOUT; o++) {
    #pragma HLS unroll
      Rp[o] = 0;
    }

    // 2->0 weights (frozen element order w2_2to0[h*2+a])
    for (unsigned int h = 0; h < NHIDDEN; h++) {
    #pragma HLS unroll
      for (unsigned int a = 0; a < 2; a++) {
      #pragma HLS unroll
        for (unsigned int o = 0; o < NOUT; o++) {
        #pragma HLS unroll
          Mult2to0: Rp[o] += w2_2to0[(h*2)+a*(NOUT)+o]*R[h][a];
        }
      }
    }
}

void nPELICAN(
    input_t model_input[(NPARTICLES)*4],
    input_t beam_input[2*4],            // 2 beam spurions as a top-level input
    nobj_t nobj,                        // particle count — exact at any input width
    result_t model_out[1]
) {
    #pragma HLS ARRAY_RESHAPE variable=model_input complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=beam_input complete dim=0
    #pragma HLS ARRAY_PARTITION variable=model_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=model_input,beam_input,model_out
    #pragma HLS PIPELINE II=1

    if (nobj != 0 ) {
      if (nobj < NPARTICLES) {
        nobj += (NPARTICLES2 - NPARTICLES);
      }
      else {
        nobj = NPARTICLES2;
      }
    }
    //mask + nobj remap stay in the top (comparators/wiring only; shared by 4 stages).
    ap_uint<1> nobjmask[(NPARTICLES2)][(NPARTICLES2)];
    #pragma HLS ARRAY_PARTITION variable=nobjmask complete dim=0
    for(unsigned int i = 0; i < NPARTICLES2; i++){
      for(unsigned int j = 0; j < NPARTICLES2; j++){
        if(i < nobj && j < nobj){
          nobjmask[i][j] = 1;
        }
        else{
          nobjmask[i][j] = 0;
        }
      }
    }

    dot_t dots[NP_SYMSZ];
    #pragma HLS ARRAY_PARTITION variable=dots complete dim=0
    np_dots(model_input, beam_input, nobjmask, dots);

#ifndef __SYNTHESIS__
    // DOTS-LEVEL injection (csim only): same hook/placement as the monolith —
    // after the dot4 front-end, before BN1. The override buffer is always full
    // 484 row-major; under NPELICAN_SPLIT_TRI only the upper triangle is stored
    // (the golden dots are symmetric; every read goes through NP_SYMIDX).
    if (npelican_dots_override) {
      for (unsigned int i = 0; i < NPARTICLES2; i++)
        for (unsigned int j = i; j < NPARTICLES2; j++) {
          dots[NP_SYMIDX(i, j)] = npelican_dots_override[i*NPARTICLES2+j];
#ifndef NPELICAN_SPLIT_TRI
          if (j != i) dots[NP_SYMIDX(j, i)] = npelican_dots_override[j*NPARTICLES2+i];
#endif
        }
    }
#endif

    //unmasked-entry counts (nobj remapped above): shared by np_agg2to2 (BN1 fold, Lever 8)
    //and np_agg2to0 (BN2 collapse, Lever 4).
    ap_uint<5> ncount  = (ap_uint<5>)nobj;     // active rows/cols, 0..22
    ap_uint<9> ncount2 = ncount * ncount;      // unmasked (i,j) pairs, 0..484

    t2_t T0[NP_SYMSZ];
    #pragma HLS ARRAY_PARTITION variable=T0 complete dim=0
    accdot2_t   dsum;
    accdotrow_t rowsum[NPARTICLES2];
    #pragma HLS ARRAY_PARTITION variable=rowsum complete dim=0
    np_bn1(dots, nobjmask, T0, dsum, rowsum);

    t2_t jmass;
    t2_t jdotp[NPARTICLES2];
    #pragma HLS ARRAY_PARTITION variable=jdotp complete dim=0
    np_agg2to2(dsum, rowsum, nobjmask, ncount, ncount2, jmass, jdotp);

    relu_t Tp_q[NPARTICLES2][NPARTICLES2][NHIDDEN];
    #pragma HLS ARRAY_PARTITION variable=Tp_q complete dim=0
    np_eq2to2(T0, jmass, jdotp, nobjmask, Tp_q);

    t0_t R[NHIDDEN][2];
    #pragma HLS ARRAY_PARTITION variable=R complete dim=0
    np_agg2to0(Tp_q, nobjmask, ncount, ncount2, R);

    mac0_t Rp[NOUT];
    #pragma HLS ARRAY_PARTITION variable=Rp complete dim=0
    np_out2to0(R, Rp);

#ifndef __SYNTHESIS__
    // Stage dump (csim only): same format/order/values as the monolith's dump.
    // T and Tr are not materialized in the top anymore (T is internal to
    // np_eq2to2; Tr was already dump-only), so both are reconstructed here with
    // the exact expressions the datapath uses.
    if (npelican_dump_fp) {
        FILE* fp = npelican_dump_fp;

        // dots: 484 values, row-major i*22+j (NP_SYMIDX keeps the dump full-size
        // and byte-identical under NPELICAN_SPLIT_TRI: (j,i) reads element (i,j))
        fprintf(fp, "dots:");
        for (unsigned int i = 0; i < NPARTICLES2; i++)
            for (unsigned int j = 0; j < NPARTICLES2; j++)
                fprintf(fp, " %.17g", (double)dots[NP_SYMIDX(i, j)]);
        fprintf(fp, "\n");

        // batch1: 484 values, row-major (approx, dump-only). Lever 8 never materializes
        // batch1; reconstructed in double as (d·s + β')·m from the batch1_2to2 constants.
        {
            const double s1 = (double)batch1_2to2[1];
            const double b1 = (double)batch1_2to2[2] - (double)batch1_2to2[0]*s1;
            fprintf(fp, "batch1:");
            for (unsigned int i = 0; i < NPARTICLES2; i++)
                for (unsigned int j = 0; j < NPARTICLES2; j++)
                    fprintf(fp, " %.17g", nobjmask[i][j] ? (double)dots[NP_SYMIDX(i, j)]*s1 + b1 : 0.0);
            fprintf(fp, "\n");
        }

        // jmass: 1 value (post-normalization, t2-grid; approx)
        fprintf(fp, "jmass: %.17g\n", (double)jmass);

        // jdotp: 22 values (post-normalization, t2-grid; approx)
        fprintf(fp, "jdotp:");
        for (unsigned int i = 0; i < NPARTICLES2; i++)
            fprintf(fp, " %.17g", (double)jdotp[i]);
        fprintf(fp, "\n");

        // T0..T5 reconstructed for the dump ONLY — np_eq2to2 builds T internally
        // with these exact expressions.
        {
            t2_t T[NPARTICLES2][NPARTICLES2][6];
            for (unsigned int i = 0; i < NPARTICLES2; i++)
                for (unsigned int j = 0; j < NPARTICLES2; j++)
                    for (unsigned int b = 0; b < 6; b++)
                        T[i][j][b] = 0;
            for (unsigned int i = 0; i < NPARTICLES2; i++)
                for (unsigned int j = 0; j < NPARTICLES2; j++) {
                    T[i][j][0] = T0[NP_SYMIDX(i, j)];
                    T[i][j][4] = jmass*nobjmask[i][j];
                    T[i][j][3] = jdotp[i];
                    T[i][j][2] = jdotp[j];
                }
            for (unsigned int i = 0; i < NPARTICLES2; i++) {
                T[i][i][5] = jmass*nobjmask[i][i];
                T[i][i][1] = jdotp[i];
            }
            for (unsigned int b = 0; b < 6; b++) {
                fprintf(fp, "T%u:", b);
                for (unsigned int i = 0; i < NPARTICLES2; i++)
                    for (unsigned int j = 0; j < NPARTICLES2; j++)
                        fprintf(fp, " %.17g", (double)T[i][j][b]);
                fprintf(fp, "\n");
            }
        }

        // Tp: 968 values, order i,j,h with h fastest (act_layer output, relu_t; exact)
        fprintf(fp, "Tp:");
        for (unsigned int i = 0; i < NPARTICLES2; i++)
            for (unsigned int j = 0; j < NPARTICLES2; j++)
                for (unsigned int h = 0; h < NHIDDEN; h++)
                    fprintf(fp, " %.17g", (double)Tp_q[i][j][h]);
        fprintf(fp, "\n");

        // Tr: 968 values, same order (t0-grid; approx). Reconstructed for the dump
        // ONLY — same folded affine the collapsed np_agg2to0 path is derived from.
        {
            bn_t_gen bn2_beta[NHIDDEN];
            for (unsigned int h = 0; h < NHIDDEN; h++)
                bn2_beta[h] = batch2_2to0[h][2] - batch2_2to0[h][0]*batch2_2to0[h][1];
            fprintf(fp, "Tr:");
            for (unsigned int i = 0; i < NPARTICLES2; i++)
                for (unsigned int j = 0; j < NPARTICLES2; j++)
                    for (unsigned int h = 0; h < NHIDDEN; h++) {
                        tr_t trv = (tr_t)((Tp_q[i][j][h]*batch2_2to0[h][1] + bn2_beta[h])*nobjmask[i][j]);
                        fprintf(fp, " %.17g", (double)trv);
                    }
            fprintf(fp, "\n");
        }

        // R: 4 values, order R[0][0] R[0][1] R[1][0] R[1][1] (exact)
        fprintf(fp, "R: %.17g %.17g %.17g %.17g\n",
                (double)R[0][0], (double)R[0][1],
                (double)R[1][0], (double)R[1][1]);

        // Rp: 1 value (output_quant grid; exact)
        fprintf(fp, "Rp: %.17g\n", (double)(Rp[0] + b2_2to0[0]));   // pre-cast logit incl. sticky-encoded b2 (within 2^-(Fh+1) of the exact value; same rounding)
    }
#endif

    model_out[0] = (result_t)(Rp[0] + b2_2to0[0]);   // b2 (sticky-bit encoded) added narrow, ONE rounding
}
