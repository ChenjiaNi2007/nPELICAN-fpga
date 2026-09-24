#include <iostream>
// <hls_math.h> removed: no hls math functions are called in this file.
#include "nPELICAN.h"
#include "weights/weights.h"

// Lever 8: T0 from the loader-generated BN1 ROM when weights.h provides one (dot_t
// narrow enough) — never in the float-reference build (dots are doubles there).
#if defined(NPELICAN_BN1_ROM) && !defined(NPELICAN_FLOAT_BUILD) && !defined(NPELICAN_NO_BN1_ROM)
#define NP_USE_BN1_ROM 1
#endif

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
    input_t beam_input[2*4],            // 2 beam spurions as a top-level input
    nobj_t nobj,                        // particle count — exact at any input width
    result_t model_out[1]
) {
    #pragma HLS ARRAY_RESHAPE variable=model_input complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=beam_input complete dim=0
    #pragma HLS ARRAY_PARTITION variable=model_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=model_input,beam_input,model_out
//    #pragma HLS DATAFLOW
    #pragma HLS PIPELINE II=1

    //pragmas for model weight arrays
    #pragma HLS ARRAY_PARTITION variable=batch1_2to2 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=w1_2to2 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=b1_2to2 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=b1_diag_2to2 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=b1_diag_total_2to2 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=batch2_2to0 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=w2_2to0 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=b2_2to0 complete dim=0
#ifdef NP_USE_BN1_ROM
    #pragma HLS ARRAY_PARTITION variable=bn1_t0_rom complete dim=0
#endif


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
    //beam spurions are now inputs so the test harness can Lorentz-boost them.
    //At |beta|=0 these are driven with (1,0,0,+1)/(1,0,0,-1), which quantize into
    //input_t identically to the previous constants -> beta=0 stays bit-exact.
#ifdef NPELICAN_CONST_BEAMS
    //Lever 5 (deployment build): beams hardwired to the fixed spurions as
    //compile-time constants, so the 43 beam-involving dots fold to E∓pz adds
    //(no multipliers) — recovers the ~172 DSP the runtime port costs (the
    //per-stage report showed np_dots at 253×4 DSP vs the historical 210×4 with
    //constant beams). beam_input is ignored (HLS may prune the port). Bit-exact
    //vs the runtime port driven at |beta|=0. Build WITHOUT this flag for the
    //equivariance harness (it Lorentz-boosts the beams).
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
    //mantissas and realign. p1 is already masked, so a padded particle encodes to
    //e=EXP_MIN, m=0 and its dots stay exactly 0. See firmware/np_blockfp.h.
    mant_t p1m[(NPARTICLES2)][4];
    bexp_t p1e[(NPARTICLES2)];
    #pragma HLS ARRAY_PARTITION variable=p1m complete dim=0
    #pragma HLS ARRAY_PARTITION variable=p1e complete dim=0
    np_bfp_encode_all(p1, p1m, p1e);
#endif

    //fill input array (each dot rounded into dot_t = input_quant grid).
    //dot4 is symmetric (p_i·p_j == p_j·p_i), so compute only the upper triangle
    //(j>=i, incl. diagonal) and mirror the result into the lower triangle. The
    //mirror is pure wiring (no hardware), so this halves the dot4 multipliers —
    //the dominant DSP cost — while producing byte-identical dots (bit-exact).
    for(unsigned int i = 0; i < NPARTICLES2; i++){
      #pragma HLS unroll
      for(unsigned int j = i; j < NPARTICLES2; j++){
        #pragma HLS unroll
#ifdef NPELICAN_BLOCK_FP
        Dot: np_bfp_dot4(p1m[i], p1e[i], p1m[j], p1e[j], dots[i*NPARTICLES2+j]);
#else
        Dot: dot4(p1[i], p1[j], dots[i*NPARTICLES2+j]);
#endif
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

    //unmasked-entry counts (nobj already remapped to the active row/col count incl. spurions).
    //Shared by the BN1 fold (Lever 8) and the BN2 collapse (Lever 4) below.
    ap_uint<5> ncount  = (ap_uint<5>)nobj;     // active rows/cols, 0..22
    ap_uint<9> ncount2 = ncount * ncount;      // unmasked (i,j) pairs, 0..484

    //Lever 8: BN1 PAST THE AGGREGATION + T0 ROM. BN1 (batch1 = (s·d + β')·m, β' = β − μ·s)
    //is consumed only (A) per element as T0 = Q_t2(batch1) and (B) linearly by the 2->2
    //sums. So batch1 is never materialized:
    // (A) T0 is a function of the DOT_W-bit code of d alone -> loader-generated ROM
    //     (bn1_t0_rom, built by running PyTorch's own BN1 + post_agg quantizer on every
    //     code), fully partitioned so each lookup is a mux of constants. The mask select
    //     is REQUIRED: PyTorch zeroes masked batch1 before quantizing (Q(0) = 0), while
    //     ROM[code(0)] = Q(β') != 0.
    // (B) raw masked dot sums (exact, on the dot grid), BN1 applied ONCE per aggregate with
    //     the 1/N̄ normalization pre-folded (Lever-4 algebra):
    //       Σ_ij batch1   = s·Σ_ij d·m  + β'·ncount²
    //       Σ_i  batch1_ij = (s·Σ_i d_ij·m_ij + β'·ncount)·m_jj
    //     All products/sums are exact in HLS promoted types; the ONLY rounding is the final
    //     t2_t cast (AP_RND_CONV, AP_SAT). Normalize-late preserved; BN1 is still an explicit
    //     affine applied BEFORE the post_agg quantizer (not folded into the dense weights),
    //     and its N-dependent additive term is explicit via ncount/ncount2. This removes the
    //     484 intermediate bn1out_t roundings, so it is MORE faithful to PyTorch (which sums
    //     the unrounded batch1).
#ifndef NP_USE_BN1_ROM
    //fallback (dot_t wider than the ROM limit, or the float-reference build): per-element
    //arithmetic T0 with a SINGLE rounding straight to t2_t (no bn1out_t intermediate).
    const bn_t_gen bn1_beta = batch1_2to2[2] - batch1_2to2[0]*batch1_2to2[1];
#endif

    //Aggregation (parameter-free), normalize-late: raw masked dot sums in exact accumulators.
    accdot2_t   dsum = 0;
    accdotrow_t rowsum[NPARTICLES2];
    #pragma HLS ARRAY_PARTITION variable=rowsum complete dim=0
    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      rowsum[i] = 0;
    }

    // M_J / J.p_j raw parts: Σ_i d_ij·m per column, then Σ_ij d·m = Σ_j rowsum[j] (the same
    // exact integer; a 22-term tree instead of a 484-term one). BN1 affine applied after.
    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      for (unsigned int j = 0; j < NPARTICLES2; j++) {
      #pragma HLS unroll
        AggJdot: rowsum[j] += dots[i*NPARTICLES2+j]*nobjmask[i][j];
      }
    }
    for (unsigned int j = 0; j < NPARTICLES2; j++) {
    #pragma HLS unroll
      AggMJ: dsum += rowsum[j];
    }

    //BN1 affine on the aggregates + normalization (pre-folded constants), ONE t2 rounding each.
    t2_t jmass = (t2_t)(dsum*bn1_s_all + ncount2*bn1_b_all);
    t2_t jdotp[NPARTICLES2];
    #pragma HLS ARRAY_PARTITION variable=jdotp complete dim=0
    for( unsigned int i = 0; i < NPARTICLES2; i++){
    #pragma HLS unroll
      jdotp[i] = (t2_t)((rowsum[i]*bn1_s_row + ncount*bn1_b_row)*nobjmask[i][i]);
    }

    //Basis ops T[i][j][0..5] on the post_agg (t2) grid. Each entry is an exact copy of
    //an already-t2-quantized value (T0 / jmass / jdotp), matching PyTorch's single
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

    //T0 = post_agg quant of batch1. Symmetric (dots, BN1 constants and nobjmask are), so
    //look up the upper triangle and mirror (wiring).
    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      for (unsigned int j = i; j < NPARTICLES2; j++) {
      #pragma HLS unroll
#ifdef NP_USE_BN1_ROM
        ap_uint<NPELICAN_DOT_W> code = dots[i*NPARTICLES2+j].range(NPELICAN_DOT_W-1, 0);
        t2_t t0 = nobjmask[i][j] ? bn1_t0_rom[code] : (t2_t)0;
#else
        t2_t t0 = (t2_t)((dots[i*NPARTICLES2+j] * batch1_2to2[1] + bn1_beta)*nobjmask[i][j]);
#endif
        LinEq2to2_0: T[i][j][0] = t0;
        if (j != i) T[j][i][0] = t0;
      }
    }

    //TODO: it's possible the following can be simplified to hold fewer arrays
    // T0 = p_i . p_j ; T1 = (J.p_i) d_ij ; T2 = J.p_j ; T3 = J.p_i ; T4 = M_J ; T5 = M_J d_ij
    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      for (unsigned int j = 0; j < NPARTICLES2; j++) {
      #pragma HLS unroll
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

    //"dense" 2->2 mix. MAC accumulates in mac2_t (exact: product grid + sticky bias bit), so the only
    //rounding is the act_layer quantizer below. Bias is NOT folded into BN/weights.
    mac2_t Tp[NPARTICLES2][NPARTICLES2][NHIDDEN];
    #pragma HLS ARRAY_PARTITION variable=Tp complete dim=0
#ifdef NPELICAN_MAC_DSP
    //Lever 6 experiment: force the 2->2 MAC multiplies into DSP48s. LUT is the
    //binding resource (53% SLR, half of it this MAC) while DSP sits at 44%;
    //impl choice does not change values, so this is bit-exact by construction.
    #pragma HLS BIND_OP variable=Tp op=mul impl=dsp
#endif

    // MAC accumulators start at the bias, masked exactly as before: b1·m_ij off the diagonal,
    // (b1+b1_diag)·m_ii on it (the diagonal constant is the loader's b1_diag_total_2to2, a
    // select between two constants, no extra add). The biases are STICKY-BIT encoded
    // (types_generated.h: b_hi on 2^-Fh + one sticky LSB 2^-(Fh+1), Fh >= mac2_F and
    // >= relu_F+1) and mac2_t carries that sticky bit (F = BIAS_F), so the sum stays exact
    // and the relu_t cast below rounds exactly like the exact float32 bias would.
    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      for (unsigned int j = 0; j < NPARTICLES2; j++) {
      #pragma HLS unroll
        for (unsigned int h = 0; h < NHIDDEN; h++) {
        #pragma HLS unroll
          Tp[i][j][h] = (i == j ? b1_diag_total_2to2[h] : b1_2to2[h]) * nobjmask[i][j];
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

    relu_t Tp_q[NPARTICLES2][NPARTICLES2][NHIDDEN];
    #pragma HLS ARRAY_PARTITION variable=Tp_q complete dim=0
    // Quantize (bias already in the accumulator) onto the act_layer grid (relu_t,
    // AP_RND_CONV), then ReLU. Q is monotone with Q(0)=0, so max(0, Q(x)) == Q(max(0, x)):
    // identical to ReLU-then-quantize.
    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      for (unsigned int j = 0; j < NPARTICLES2; j++) {
      #pragma HLS unroll
        for (unsigned int h = 0; h < NHIDDEN; h++) {
        #pragma HLS unroll
          relu_t q = (relu_t)Tp[i][j][h];
          if (q < 0) q = 0;
          Tp_q[i][j][h] = q;
        }
      }
    }

    //#2: BN2 + 2->0 aggregation COLLAPSED. PyTorch keeps Tr=BN2(relu) in float and the
    //2->0 ops only SUM it (full sum and trace); Tr is never read per-element (it is NOT a
    //quantization point). BN2 is affine and the aggregation linear, so push the per-channel
    //affine PAST the sum (exact identity):
    //   R_sum[h]   = Σ_ij BN2_h(Tp_q) = s_h·(Σ_ij Tp_q)      + β'_h·nobj²
    //   R_trace[h] = Σ_i  BN2_h(Tp_q) = s_h·(Σ_i Tp_q[i][i]) + β'_h·nobj
    //with β'_h = β_h − μ_h·s_h (BN2 mean folded into bias, #4). This replaces 22·22·NHIDDEN
    //wide bn_t_gen multiplies with NHIDDEN, keeps normalize-late, and is MORE faithful to
    //PyTorch (the per-element tr_t rounding is gone — ONE rounding at the t0 cast).
    //  - Tp_q is masked here with nobjmask (ap_uint<1> → a select, 0 DSP): off-diagonal
    //    entries with one masked index are NOT zero (e.g. T3=jdotp[i] is masked by [i] only,
    //    not [i][j]), and the old code zeroed them via BN2's ·mask. The additive β'·count
    //    terms count only unmasked entries (nobj² full / nobj trace), so BN2's N-dependent
    //    bias contribution is preserved.

    //folded BN2 bias β'_h = β_h − μ_h·s_h (compile-time constant per channel).
    bn_t_gen bn2_beta[NHIDDEN];
    #pragma HLS ARRAY_PARTITION variable=bn2_beta complete dim=0
    for (unsigned int h = 0; h < NHIDDEN; h++) {
    #pragma HLS unroll
      bn2_beta[h] = batch2_2to0[h][2] - batch2_2to0[h][0]*batch2_2to0[h][1];
    }

    // raw 2->0 aggregators of the ReLU output: total sum (accrelu_t) and trace (accrelurow_t)
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

    //unmasked-entry counts ncount/ncount2: hoisted above (shared with the BN1 fold).

    //apply the per-channel BN2 affine to the raw aggregate, then normalize-late: ONE rescale
    //rounding onto the post_agg-2to0 grid (t0_t). R[h][0]=normalized sum; R[h][1]=trace. The
    //s·A and β'·count products are exact (HLS), so only the t0 cast rounds.
    t0_t R[NHIDDEN][2];
    #pragma HLS ARRAY_PARTITION variable=R complete dim=0
    for (unsigned int h = 0; h < NHIDDEN; h++) {
    #pragma HLS unroll
      R[h][0] = (t0_t)((batch2_2to0[h][1]*A_sum[h]   + bn2_beta[h]*ncount2) * invnave2);
      R[h][1] = (t0_t)((batch2_2to0[h][1]*A_trace[h] + bn2_beta[h]*ncount ) * invnave);
    }

    //Final 1D output: 2->0 dense MAC in mac0_t (exact: product grid + sticky bias bit), then round onto
    //the output_quant grid (result_t == out_t, AP_RND_CONV).
    mac0_t Rp[NOUT];
    #pragma HLS ARRAY_PARTITION variable=Rp complete dim=0

    // accumulator starts at the sticky-bit-encoded bias b2 (mac0_t carries the sticky bit,
    // F = BIAS_F, so the sum is exact); ONE rounding at the result_t cast.
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
    //   batch1 (reconstructed in double, never materialized after Lever 8), jmass, jdotp, Tr.
    if (npelican_dump_fp) {
        FILE* fp = npelican_dump_fp;

        // dots: 484 values, row-major i*22+j
        fprintf(fp, "dots:");
        for (unsigned int i = 0; i < NPARTICLES2; i++)
            for (unsigned int j = 0; j < NPARTICLES2; j++)
                fprintf(fp, " %.17g", (double)dots[i*NPARTICLES2+j]);
        fprintf(fp, "\n");

        // batch1: 484 values, row-major (approx, dump-only). Lever 8 never materializes
        // batch1; reconstructed in double as (d·s + β')·m from the batch1_2to2 constants.
        {
            const double s1 = (double)batch1_2to2[1];
            const double b1 = (double)batch1_2to2[2] - (double)batch1_2to2[0]*s1;
            fprintf(fp, "batch1:");
            for (unsigned int i = 0; i < NPARTICLES2; i++)
                for (unsigned int j = 0; j < NPARTICLES2; j++)
                    fprintf(fp, " %.17g", nobjmask[i][j] ? (double)dots[i*NPARTICLES2+j]*s1 + b1 : 0.0);
            fprintf(fp, "\n");
        }

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

        // Tr: 968 values, same order (t0-grid; approx). Reconstructed for the dump ONLY —
        // the datapath now folds BN2 past the aggregation (#2), so Tr is never materialized.
        // Uses the same folded affine (Tp_q·s + β') the collapsed path is derived from.
        fprintf(fp, "Tr:");
        for (unsigned int i = 0; i < NPARTICLES2; i++)
            for (unsigned int j = 0; j < NPARTICLES2; j++)
                for (unsigned int h = 0; h < NHIDDEN; h++) {
                    tr_t trv = (tr_t)((Tp_q[i][j][h]*batch2_2to0[h][1] + bn2_beta[h])*nobjmask[i][j]);
                    fprintf(fp, " %.17g", (double)trv);
                }
        fprintf(fp, "\n");

        // R: 4 values, order R[0][0] R[0][1] R[1][0] R[1][1] (exact)
        fprintf(fp, "R: %.17g %.17g %.17g %.17g\n",
                (double)R[0][0], (double)R[0][1],
                (double)R[1][0], (double)R[1][1]);

        // Rp: 1 value (output_quant grid; exact)
        fprintf(fp, "Rp: %.17g\n", (double)Rp[0]);   // pre-cast logit incl. sticky-encoded b2 (accumulator init) (within 2^-(Fh+1) of the exact value; same rounding)
    }
#endif

    model_out[0] = (result_t)Rp[0];   // b2 (sticky-bit encoded) is the accumulator init; ONE rounding
}
