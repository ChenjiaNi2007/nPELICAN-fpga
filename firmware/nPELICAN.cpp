#include <iostream>
// <hls_math.h> removed: no hls math functions are called in this file.
#include "nPELICAN.h"
#include "weights/weights.h"

#ifndef __SYNTHESIS__
#include <cstdio>
FILE* npelican_dump_fp = nullptr;
// DOTS-LEVEL test hook (csim only, zero synthesis impact): when non-null, the dot4
// front-end output is overwritten with these 484 externally-supplied dots (row-major
// i*22+j) so the testbench can feed PyTorch's quantized d_ij directly and isolate the
// network from the float32 d_ij-cancellation caveat (FIRMWARE_QAT_PLAN D4).
dot_t* npelican_dots_override = nullptr;
#endif

// ============================================================================
// Phase 2: per-stage fixed-point types from the learned QAT scales.
// Casts to a quantization-point type (dot_t / t2_t / relu_t / t0_t / result_t)
// carry AP_RND_CONV and sit EXACTLY where PyTorch fake-quantizes; everything
// between those points is computed in exact-widened types (acc*_t, mac*_t) or
// the wide float-constant types (bn_t_gen, bias_t_gen, norm_t). Normalize-late
// is preserved: raw sums accumulate, then ONE rescale rounds down to the grid.
// ============================================================================

psloglut_t psloglut(int index){
  static psloglut_t _table[N_TABLE_PSLOG];
  lut_pslog_init<psloglut_t,N_TABLE_PSLOG>(_table);
  return _table[index];
}

void dot4(input_t p1[4], input_t p2[4], dot_t& dot) {
//#pragma HLS INLINE
//#pragma function instatiate

// Input in the form E, px, py, pz. The Minkowski dot is computed in HLS's exact
// promoted type (products/sums of fixed-point are exact) and rounded once into
// dot_t (the input_quant 2^-k grid, AP_RND_CONV). NOTE: PyTorch quantizes d_ij
// computed from FLOAT momenta; here d_ij comes from input_t momenta, so an
// occasional 1-LSB disagreement at this front-end is the one documented caveat.
dot = p1[0]*p2[0]-p1[1]*p2[1]-p1[2]*p2[2]-p1[3]*p2[3];

}

void nPELICAN(
    input_t model_input[(NPARTICLES)*4],
    input_t nobj,
    result_t model_out[1]
) {
    #pragma HLS ARRAY_RESHAPE variable=model_input complete dim=0
    #pragma HLS ARRAY_PARTITION variable=model_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=model_input,model_out
//    #pragma HLS DATAFLOW
    #pragma HLS PIPELINE II=1

    //pragmas for model weight arrays
    #pragma HLS ARRAY_PARTITION variable=batch1_2to2 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=w1_2to2 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=b1_2to2 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=b1_diag_2to2 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=batch2_2to0 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=w2_2to0 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=b2_2to0 complete dim=0


    if (nobj != 0 ) {
      if (nobj < NPARTICLES) {
        nobj += (NPARTICLES2 - NPARTICLES);
      }
      else {
        nobj = NPARTICLES2;
      }
    }
    //create array mask from number of particles in the event.
    //nobjmask is strictly 0/1, so ap_uint<1>: multiplies become exact selects and
    //padded entries stay EXACTLY 0 in every downstream type.
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
    input_t p1[(NPARTICLES2)][4];
    #pragma HLS ARRAY_PARTITION variable=p1 complete dim=0
    P1Prep: for (unsigned int i = 0; i < NPARTICLES; i++) {
    #pragma HLS unroll
      for (unsigned int k = 0; k < 4; k++){
      #pragma HLS unroll
        p1[(i + (NPARTICLES2 - NPARTICLES))][k] = model_input[i*(4)+k]*nobjmask[i][0];
      }
    }
    //add beam spurions
    p1[0][0]   = 1.; p1[0][1]   = 0.; p1[0][2]   = 0.; p1[0][3]   = 1.;
    p1[1][0] = 1.; p1[1][1] = 0.; p1[1][2] = 0.; p1[1][3] = -1.;

    //fill input array (each dot rounded into dot_t = input_quant grid).
    //dot4 is symmetric (p_i·p_j == p_j·p_i), so compute only the upper triangle
    //(j>=i, incl. diagonal) and mirror the result into the lower triangle. The
    //mirror is pure wiring (no hardware), so this halves the dot4 multipliers —
    //the dominant DSP cost — while producing byte-identical dots (bit-exact).
    for(unsigned int i = 0; i < NPARTICLES2; i++){
      #pragma HLS unroll
      for(unsigned int j = i; j < NPARTICLES2; j++){
        #pragma HLS unroll
        Dot: dot4(p1[i], p1[j], dots[i*NPARTICLES2+j]);
        if (j != i) dots[j*NPARTICLES2+i] = dots[i*NPARTICLES2+j];
      }
    }

#ifndef __SYNTHESIS__
    // DOTS-LEVEL injection (csim only): replace the dot4 result with the supplied
    // PyTorch-quantized dots to test the network in isolation from the front-end.
    if (npelican_dots_override) {
      for (unsigned int k = 0; k < NPARTICLES2*NPARTICLES2; k++)
        dots[k] = npelican_dots_override[k];
    }
#endif

   //psuedolog input encoder
   /*
    for(unsigned int i = 0; i < NPARTICLES2; i++){
      #pragma HLS unroll
      for(unsigned int j = 0; j < NPARTICLES2; j++){
        #pragma HLS unroll
        dots[i*NPARTICLES2+j] = (dot_t) (psloglut(dots[i*NPARTICLES2+j]>>TABLE_FRACS));
      }
    }
    */

    //Do first batchnorm. PyTorch keeps the BN output (batch1) in float and SUMS THE
    //UNQUANTIZED value in the aggregation; only the basis op T0 sees the post_agg
    //quantizer. So batch1 is stored WIDE (bn1out_t, AGG_F frac) — NOT t2_t — otherwise
    //the coarse t2 rounding (F=18) of each summand tips the renormalized jmass/jdotp
    //onto the wrong t2 grid point. T0 below casts batch1 to t2_t once. BN constants are
    //NOT folded (CLAUDE.md invariant).
    //batch1 = BN1(dots) is symmetric too: dots is symmetric, the BN constants
    //(mean/scale/beta) are scalar, and nobjmask[i][j]==nobjmask[j][i]. So compute
    //the upper triangle and mirror — halves the BN1 multiplies (part of the
    //inferred-DSP cost). Bit-exact for the same reason as the dot loop above.
    bn1out_t batch1[(NPARTICLES2)*(NPARTICLES2)];
    #pragma HLS ARRAY_PARTITION variable=batch1 complete dim=0
    for(unsigned int i = 0; i < NPARTICLES2; i++){
      #pragma HLS unroll
      for(unsigned int j = i; j < NPARTICLES2; j++){
        #pragma HLS unroll
        bn1out_t v = (bn1out_t)(((dots[i*NPARTICLES2+j] - batch1_2to2[0]) * batch1_2to2[1] + batch1_2to2[2])*nobjmask[i][j]);
        batch1[i*NPARTICLES2+j] = v;
        if (j != i) batch1[j*NPARTICLES2+i] = v;
      }
    }

    //Aggregation (parameter-free), normalize-late: accumulate raw sums in widened
    //accumulators, then ONE rescale by the (precise) norm_t multipliers.
    acc2_t   jmass_acc = 0;
    accrow_t jdotp_acc[NPARTICLES2];
    #pragma HLS ARRAY_PARTITION variable=jdotp_acc complete dim=0
    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      jdotp_acc[i] = 0;
    }

    // M_J = sum(batch1); J . p_j = sum over rows i of batch1[i][j]
    //TODO: could reform this to only loop over the upper triangle and double off diagonal contributions
    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      for (unsigned int j = 0; j < NPARTICLES2; j++) {
      #pragma HLS unroll
        AggMJ:   jmass_acc    += batch1[i*NPARTICLES2+j];
        AggJdot: jdotp_acc[j] += batch1[i*NPARTICLES2+j];
      }
    }

    //aggregation normalizations: rescale once and round onto the post_agg (t2) grid.
    t2_t jmass = (t2_t)(jmass_acc * invnave2);
    t2_t jdotp[NPARTICLES2];
    #pragma HLS ARRAY_PARTITION variable=jdotp complete dim=0
    for( unsigned int i = 0; i < NPARTICLES2; i++){
    #pragma HLS unroll
      jdotp[i] = (t2_t)(jdotp_acc[i] * invnave);
    }

    //Basis ops T[i][j][0..5] on the post_agg (t2) grid. Each entry is an exact copy of
    //an already-t2-quantized value (batch1 / jmass / jdotp), matching PyTorch's single
    //post_agg_quant over the stacked 6-op tensor.
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

    //TODO: it's possible the following can be simplified to hold fewer arrays
    // T0 = p_i . p_j ; T1 = (J.p_i) d_ij ; T2 = J.p_j ; T3 = J.p_i ; T4 = M_J ; T5 = M_J d_ij
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

    //"dense" 2->2 mix. MAC accumulates in mac2_t (exact product width), so the only
    //rounding is the act_layer quantizer below. Bias is NOT folded into BN/weights.
    mac2_t Tp[NPARTICLES2][NPARTICLES2][NHIDDEN];
    #pragma HLS ARRAY_PARTITION variable=Tp complete dim=0

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

    // ReLU, then quantize onto the act_layer grid (relu_t, AP_RND_CONV). The compare
    // is against a typed 0 (mac2_t), not a double literal.
    relu_t Tp_q[NPARTICLES2][NPARTICLES2][NHIDDEN];
    #pragma HLS ARRAY_PARTITION variable=Tp_q complete dim=0
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

    //second batchnorm: float-constant affine on the relu output. Tr is what PyTorch
    //aggregates in float for the 2->0 ops; it is NOT a quantization point and the BN2
    //scale widens its range (~130x), so it is stored in tr_t (wide I, t0 fractional grid)
    //— NOT t0_t, which (I=1) would saturate it. The 2->0 sums stay clean in acc0(row)_t.
    tr_t Tr[NPARTICLES2][NPARTICLES2][NHIDDEN];
    #pragma HLS ARRAY_PARTITION variable=Tr complete dim=0
    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      for (unsigned int j = 0; j < NPARTICLES2; j++) {
      #pragma HLS unroll
        for (unsigned int h = 0; h < NHIDDEN; h++) {
        #pragma HLS unroll
            Tr[i][j][h] = (tr_t)(((Tp_q[i][j][h] - batch2_2to0[h][0]) * batch2_2to0[h][1] + batch2_2to0[h][2])*nobjmask[i][j]);
        }
      }
    }

    // two aggregators for 2to0: total sum (acc0_t) and trace (acc0row_t)
    acc0_t    R_sum[NHIDDEN];
    acc0row_t R_trace[NHIDDEN];
    #pragma HLS ARRAY_PARTITION variable=R_sum complete dim=0
    #pragma HLS ARRAY_PARTITION variable=R_trace complete dim=0

    for (unsigned int h = 0; h < NHIDDEN; h++) {
    #pragma HLS unroll
      R_sum[h]   = 0;
      R_trace[h] = 0;
    }

    //total sum
    for (unsigned int h = 0; h < NHIDDEN; h++) {
    #pragma HLS unroll
      for (unsigned int i = 0; i < NPARTICLES2; i++) {
      #pragma HLS unroll
        for (unsigned int j = 0; j < NPARTICLES2; j++) {
        #pragma HLS unroll
            LinEq2to0: R_sum[h] += Tr[i][j][h];
        }
      }
    }

    //trace
    for (unsigned int h = 0; h < NHIDDEN; h++) {
    #pragma HLS unroll
      for (unsigned int i = 0; i < NPARTICLES2; i++) {
      #pragma HLS unroll
        R_trace[h] += Tr[i][i][h];
      }
    }

    //normalize-late: rescale once and round onto the post_agg-2to0 grid (t0_t).
    //R[h][0] = normalized total sum; R[h][1] = normalized trace.
    t0_t R[NHIDDEN][2];
    #pragma HLS ARRAY_PARTITION variable=R complete dim=0
    for (unsigned int h = 0; h < NHIDDEN; h++) {
    #pragma HLS unroll
      R[h][0] = (t0_t)(R_sum[h]   * invnave2);
      R[h][1] = (t0_t)(R_trace[h] * invnave);
    }

    //Final 1D output: 2->0 dense MAC in mac0_t (exact product width), then round onto
    //the output_quant grid (result_t == out_t, AP_RND_CONV).
    mac0_t Rp[NOUT];
    #pragma HLS ARRAY_PARTITION variable=Rp complete dim=0

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

#ifndef __SYNTHESIS__
    // Stage dump (csim only): written when npelican_dump_fp is non-null.
    // EXACT-match stages vs the PyTorch golden dump (true quantization points):
    //   dots, T0..T5, Tp, R, Rp.
    // APPROX stages (PyTorch keeps them in float; firmware stores them on the next
    // grid, so expect tiny differences here — they are NOT mismatches):
    //   batch1 (= PyTorch's quantized T0, not raw batch1), jmass, jdotp, Tr.
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

        // T0..T5: six lines, each 484 values row-major T[i][j][b] (exact)
        for (unsigned int b = 0; b < 6; b++) {
            fprintf(fp, "T%u:", b);
            for (unsigned int i = 0; i < NPARTICLES2; i++)
                for (unsigned int j = 0; j < NPARTICLES2; j++)
                    fprintf(fp, " %.17g", (double)T[i][j][b]);
            fprintf(fp, "\n");
        }

        // Tp: 968 values, order i,j,h with h fastest (act_layer output, relu_t; exact)
        fprintf(fp, "Tp:");
        for (unsigned int i = 0; i < NPARTICLES2; i++)
            for (unsigned int j = 0; j < NPARTICLES2; j++)
                for (unsigned int h = 0; h < NHIDDEN; h++)
                    fprintf(fp, " %.17g", (double)Tp_q[i][j][h]);
        fprintf(fp, "\n");

        // Tr: 968 values, same order (t0-grid; approx)
        fprintf(fp, "Tr:");
        for (unsigned int i = 0; i < NPARTICLES2; i++)
            for (unsigned int j = 0; j < NPARTICLES2; j++)
                for (unsigned int h = 0; h < NHIDDEN; h++)
                    fprintf(fp, " %.17g", (double)Tr[i][j][h]);
        fprintf(fp, "\n");

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
