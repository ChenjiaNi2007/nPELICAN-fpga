#include <iostream>
#include "nPELICAN.h"
#include "weights/weights.h"

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
    dot_t dots[(NPARTICLES2)*(NPARTICLES2)]
) {
    #pragma HLS INLINE off
    #pragma HLS PIPELINE II=1
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

    //dot4 is symmetric: compute only the upper triangle and mirror (pure wiring).
    for(unsigned int i = 0; i < NPARTICLES2; i++){
      #pragma HLS unroll
      for(unsigned int j = i; j < NPARTICLES2; j++){
        #pragma HLS unroll
        Dot: dot4(p1[i], p1[j], dots[i*NPARTICLES2+j]);
        if (j != i) dots[j*NPARTICLES2+i] = dots[i*NPARTICLES2+j];
      }
    }
}

// Stage 2: BatchNorm1 scalar affine (mean folded into bias), masked, symmetric.
// batch1 stays WIDE (bn1out_t) — see nPELICAN.cpp for why (PyTorch sums the
// unquantized BN output; only the basis op T0 sees the post_agg quantizer).
void np_bn1(
    dot_t dots[(NPARTICLES2)*(NPARTICLES2)],
    ap_uint<1> nobjmask[NPARTICLES2][NPARTICLES2],
    bn1out_t batch1[(NPARTICLES2)*(NPARTICLES2)]
) {
    #pragma HLS INLINE off
    #pragma HLS PIPELINE II=1
    #pragma HLS ARRAY_PARTITION variable=dots complete dim=0
    #pragma HLS ARRAY_PARTITION variable=nobjmask complete dim=0
    #pragma HLS ARRAY_PARTITION variable=batch1 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=batch1_2to2 complete dim=0

    const bn_t_gen bn1_beta = batch1_2to2[2] - batch1_2to2[0]*batch1_2to2[1];
    for(unsigned int i = 0; i < NPARTICLES2; i++){
      #pragma HLS unroll
      for(unsigned int j = i; j < NPARTICLES2; j++){
        #pragma HLS unroll
        bn1out_t v = (bn1out_t)((dots[i*NPARTICLES2+j] * batch1_2to2[1] + bn1_beta)*nobjmask[i][j]);
        batch1[i*NPARTICLES2+j] = v;
        if (j != i) batch1[j*NPARTICLES2+i] = v;
      }
    }
}

// Stage 3: 2->2 aggregation (parameter-free), normalize-late: raw sums in the
// widened accumulators, then ONE rescale rounding onto the post_agg (t2) grid.
void np_agg2to2(
    bn1out_t batch1[(NPARTICLES2)*(NPARTICLES2)],
    t2_t &jmass,
    t2_t jdotp[NPARTICLES2]
) {
    #pragma HLS INLINE off
    #pragma HLS PIPELINE II=1
    #pragma HLS ARRAY_PARTITION variable=batch1 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=jdotp complete dim=0

    acc2_t   jmass_acc = 0;
    accrow_t jdotp_acc[NPARTICLES2];
    #pragma HLS ARRAY_PARTITION variable=jdotp_acc complete dim=0
    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      jdotp_acc[i] = 0;
    }

    // M_J = sum(batch1); J . p_j = sum over rows i of batch1[i][j]
    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      for (unsigned int j = 0; j < NPARTICLES2; j++) {
      #pragma HLS unroll
        AggMJ:   jmass_acc    += batch1[i*NPARTICLES2+j];
        AggJdot: jdotp_acc[j] += batch1[i*NPARTICLES2+j];
      }
    }

    jmass = (t2_t)(jmass_acc * invnave2);
    for( unsigned int i = 0; i < NPARTICLES2; i++){
    #pragma HLS unroll
      jdotp[i] = (t2_t)(jdotp_acc[i] * invnave);
    }
}

// Stage 4: basis ops T (INTERNAL — pure wiring) + "dense" 2->2 mix + ReLU +
// act_layer quantization. MAC accumulates in mac2_t (exact product width), so
// the only rounding is the relu_t cast.
void np_eq2to2(
    bn1out_t batch1[(NPARTICLES2)*(NPARTICLES2)],
    t2_t jmass,
    t2_t jdotp[NPARTICLES2],
    ap_uint<1> nobjmask[NPARTICLES2][NPARTICLES2],
    relu_t Tp_q[NPARTICLES2][NPARTICLES2][NHIDDEN]
) {
    #pragma HLS INLINE off
    #pragma HLS PIPELINE II=1
    #pragma HLS ARRAY_PARTITION variable=batch1 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=jdotp complete dim=0
    #pragma HLS ARRAY_PARTITION variable=nobjmask complete dim=0
    #pragma HLS ARRAY_PARTITION variable=Tp_q complete dim=0
    #pragma HLS ARRAY_PARTITION variable=w1_2to2 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=b1_2to2 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=b1_diag_2to2 complete dim=0

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
        LinEq2to2_0: T[i][j][0] = (t2_t)batch1[i*NPARTICLES2+j];   // post_agg quant of batch1
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

    // initialize with bias
    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      for (unsigned int j = 0; j < NPARTICLES2; j++) {
      #pragma HLS unroll
        for (unsigned int h = 0; h < NHIDDEN; h++) {
        #pragma HLS unroll
          Tp[i][j][h] = b1_2to2[h]*nobjmask[i][j];
          }
        }
      }

    for (unsigned int i = 0; i < NPARTICLES2; i++){
    #pragma HLS unroll
      for (unsigned int h = 0; h < NHIDDEN; h++) {
      #pragma HLS unroll
        Tp[i][i][h] += b1_diag_2to2[h]*nobjmask[i][i];
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

    // ReLU, then quantize onto the act_layer grid (relu_t, AP_RND_CONV).
    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      for (unsigned int j = 0; j < NPARTICLES2; j++) {
      #pragma HLS unroll
        for (unsigned int h = 0; h < NHIDDEN; h++) {
        #pragma HLS unroll
          if (Tp[i][j][h] < 0){
            Tp[i][j][h] = 0;
            }
          Tp_q[i][j][h] = (relu_t)Tp[i][j][h];
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
    nobj_t nobj,
    t0_t R[NHIDDEN][2]
) {
    #pragma HLS INLINE off
    #pragma HLS PIPELINE II=1
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

    //unmasked-entry counts (nobj already remapped in the top).
    ap_uint<5> ncount  = (ap_uint<5>)nobj;     // active rows/cols, 0..22
    ap_uint<9> ncount2 = ncount * ncount;      // unmasked (i,j) pairs, 0..484

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
    #pragma HLS INLINE off
    #pragma HLS PIPELINE II=1
    #pragma HLS ARRAY_PARTITION variable=R complete dim=0
    #pragma HLS ARRAY_PARTITION variable=Rp complete dim=0
    #pragma HLS ARRAY_PARTITION variable=w2_2to0 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=b2_2to0 complete dim=0

    // initialize with bias
    for (unsigned int o = 0; o < NOUT; o++) {
    #pragma HLS unroll
      Rp[o] = b2_2to0[o];
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

    dot_t dots[(NPARTICLES2)*(NPARTICLES2)];
    #pragma HLS ARRAY_PARTITION variable=dots complete dim=0
    np_dots(model_input, beam_input, nobjmask, dots);

#ifndef __SYNTHESIS__
    // DOTS-LEVEL injection (csim only): same hook/placement as the monolith —
    // after the dot4 front-end, before BN1.
    if (npelican_dots_override) {
      for (unsigned int k = 0; k < NPARTICLES2*NPARTICLES2; k++)
        dots[k] = npelican_dots_override[k];
    }
#endif

    bn1out_t batch1[(NPARTICLES2)*(NPARTICLES2)];
    #pragma HLS ARRAY_PARTITION variable=batch1 complete dim=0
    np_bn1(dots, nobjmask, batch1);

    t2_t jmass;
    t2_t jdotp[NPARTICLES2];
    #pragma HLS ARRAY_PARTITION variable=jdotp complete dim=0
    np_agg2to2(batch1, jmass, jdotp);

    relu_t Tp_q[NPARTICLES2][NPARTICLES2][NHIDDEN];
    #pragma HLS ARRAY_PARTITION variable=Tp_q complete dim=0
    np_eq2to2(batch1, jmass, jdotp, nobjmask, Tp_q);

    t0_t R[NHIDDEN][2];
    #pragma HLS ARRAY_PARTITION variable=R complete dim=0
    np_agg2to0(Tp_q, nobjmask, nobj, R);

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

        // dots: 484 values, row-major i*22+j
        fprintf(fp, "dots:");
        for (unsigned int i = 0; i < NPARTICLES2; i++)
            for (unsigned int j = 0; j < NPARTICLES2; j++)
                fprintf(fp, " %.17g", (double)dots[i*NPARTICLES2+j]);
        fprintf(fp, "\n");

        // batch1: 484 values, row-major (t2-grid; approx)
        fprintf(fp, "batch1:");
        for (unsigned int i = 0; i < NPARTICLES2; i++)
            for (unsigned int j = 0; j < NPARTICLES2; j++)
                fprintf(fp, " %.17g", (double)batch1[i*NPARTICLES2+j]);
        fprintf(fp, "\n");

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
                    T[i][j][0] = (t2_t)batch1[i*NPARTICLES2+j];
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
        fprintf(fp, "Rp: %.17g\n", (double)Rp[0]);
    }
#endif

    model_out[0] = (result_t)Rp[0];
}
