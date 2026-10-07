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
dotall_t* npelican_dots_override = nullptr;   // dotall_t: holds every dot population's grid (== dot_t without the jet split)
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

#ifdef NPELICAN_JET_QUANT_SPLIT
// --jet-quant-split: the jet row/col of the Gram matrix has its own learned grids. The jet
// momenta stay on jet_t (pmu_quant_jet) -- casting them into input_t would re-quantize them
// onto the particle grid. Products/sums are exact in HLS's promoted type; ONE AP_RND_CONV
// cast onto the population's grid (input_quant_jet / input_quant_mjet), as in PyTorch.
void dot4j(input_t p[4], jet_t q[4], dotj_t& dot) {
dot = p[0]*q[0]-p[1]*q[1]-p[2]*q[2]-p[3]*q[3];
}
void dot4m(jet_t q[4], dotm_t& dot) {
dot = q[0]*q[0]-q[1]*q[1]-q[2]*q[2]-q[3]*q[3];
}
#endif

void nPELICAN(
    input_t model_input[(NPARTICLES)*4],
    input_t beam_input[2*4],            // 2 beam spurions as a top-level input
#ifdef NPELICAN_JET_SPURION
    jet_t jet_input[4],                 // full-jet 4-momentum -> spurion slot 2 (never masked)
#endif
    nobj_t nobj,                        // particle count — exact at any input width
    result_t model_out[NOUT]
) {
    #pragma HLS ARRAY_RESHAPE variable=model_input complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=beam_input complete dim=0
    #pragma HLS ARRAY_PARTITION variable=model_out complete dim=0
#ifdef NPELICAN_JET_SPURION
    #pragma HLS ARRAY_RESHAPE variable=jet_input complete dim=0
    #pragma HLS INTERFACE ap_vld port=model_input,beam_input,jet_input,model_out
#else
    #pragma HLS INTERFACE ap_vld port=model_input,beam_input,model_out
#endif
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


    //nobj remap: present-particle count -> active row/col count incl. spurions,
    //i.e. nobj + NSPURIONS (NPARTICLES2 - NPARTICLES == NSPURIONS: 2 beams [+ jet]).
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

    dotall_t dots[(NPARTICLES2)*(NPARTICLES2)];   // == dot_t unless NPELICAN_JET_QUANT_SPLIT
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

#ifdef NPELICAN_JET_SPURION
    //full-jet spurion at slot 2 (PyTorch collate add_jet: beams, jet, constituents).
    //NOT masked: the jet is always present (E>0), exactly like the beams. Taken from the
    //port in both the runtime-beam and NPELICAN_CONST_BEAMS builds.
#ifdef NPELICAN_JET_QUANT_SPLIT
    //split: the jet keeps its own grid (jet_t) in pj; p1[2] is unused (its dots are dot4j/dot4m).
    jet_t pj[4];
    #pragma HLS ARRAY_PARTITION variable=pj complete dim=0
    JetPrep: for (unsigned int k = 0; k < 4; k++) {
      #pragma HLS unroll
      pj[k] = jet_input[k];
      p1[2][k] = 0;
    }
#else
    JetPrep: for (unsigned int k = 0; k < 4; k++) {
      #pragma HLS unroll
      p1[2][k] = jet_input[k];
    }
#endif
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
#elif defined(NPELICAN_JET_QUANT_SPLIT)
        //three populations, compile-time branches after unrolling (j >= i, so i==2 && j==2 is
        //m_jet^2, i==2 || j==2 the jet row/col, else particle/beam pairs). Round into the
        //population grid first, then widen EXACTLY into dotall_t.
        if (i == 2 && j == 2) {
          dotm_t dm;
          DotM: dot4m(pj, dm);
          dots[i*NPARTICLES2+j] = dm;
        } else if (i == 2 || j == 2) {
          dotj_t dj;
          DotJ: dot4j(p1[i == 2 ? j : i], pj, dj);
          dots[i*NPARTICLES2+j] = dj;
        } else {
          dot_t dp;
          Dot: dot4(p1[i], p1[j], dp);
          dots[i*NPARTICLES2+j] = dp;
        }
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
    //#4: fold BN1's mean into the bias ONCE (compile-time constant), dropping the wide
    //(bn_t_gen) per-element subtract: (dots-μ)·s+β == dots·s + (β-μ·s). Mathematically the
    //same affine, still applied elementwise BEFORE aggregation (NOT folded into the dense
    //weights), so the "additive BN term is N-dependent" invariant is untouched. β' rounds at
    //bn_t_gen F (>> bn1out_t F), so the cast to bn1out_t is the same single rounding as before.
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

    //"dense" 2->2 mix. Products accumulate in mac2_t (exact product width); the bias is
    //added once afterwards in mac2b_t (product grid + guard bits), so the only rounding is
    //the act_layer quantizer below. Bias is NOT folded into BN/weights.
    mac2_t Tp[NPARTICLES2][NPARTICLES2][NHIDDEN];
    #pragma HLS ARRAY_PARTITION variable=Tp complete dim=0
#ifdef NPELICAN_MAC_DSP
    //Lever 6 experiment: force the 2->2 MAC multiplies into DSP48s. LUT is the
    //binding resource (53% SLR, half of it this MAC) while DSP sits at 44%;
    //impl choice does not change values, so this is bit-exact by construction.
    #pragma HLS BIND_OP variable=Tp op=mul impl=dsp
#endif

    // initialize to 0: the bias is NOT added here. Products accumulate exactly on the
    // mac2_t product grid; b1/b1_diag carry guard bits below that grid and are added ONCE
    // in mac2b_t just before the relu quantizer (avoids exact rounding ties there).
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

    // Bias add (mac2b_t), ReLU, then quantize onto the act_layer grid (relu_t,
    // AP_RND_CONV). The compare is against an integer 0 (exact), not a double literal.
    relu_t Tp_q[NPARTICLES2][NPARTICLES2][NHIDDEN];
    #pragma HLS ARRAY_PARTITION variable=Tp_q complete dim=0
    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      for (unsigned int j = 0; j < NPARTICLES2; j++) {
      #pragma HLS unroll
        for (unsigned int h = 0; h < NHIDDEN; h++) {
        #pragma HLS unroll
          // bias add at full bias precision (mac2b_t), same nobjmask factors as before
          mac2b_t v = (mac2b_t)Tp[i][j][h] + b1_2to2[h]*nobjmask[i][j];
          if (i == j) v += b1_diag_2to2[h]*nobjmask[i][i];
          if (v < 0){
            v = 0;
            }
          Tp_q[i][j][h] = (relu_t)v;
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

    //unmasked-entry counts (nobj already remapped to the active row/col count incl. spurions).
    ap_uint<NOBJ_BITS> ncount  = (ap_uint<NOBJ_BITS>)nobj;     // active rows/cols, 0..22
    ap_uint<2*NOBJ_BITS-1> ncount2 = ncount * ncount;      // unmasked (i,j) pairs, 0..484

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

    //Final 1D output: 2->0 dense MAC in mac0_t (exact product width), then round onto
    //the output_quant grid (result_t == out_t, AP_RND_CONV).
    //N2TO0_OUT == NOUT without a head; == NPELICAN_HEAD (K hidden channels) with one.
    mac0_t Rp[N2TO0_OUT];
    #pragma HLS ARRAY_PARTITION variable=Rp complete dim=0

    // initialize with bias
    for (unsigned int o = 0; o < N2TO0_OUT; o++) {
    #pragma HLS unroll
      Rp[o] = b2_2to0[o];
    }

    // 2->0 weights. Frozen element order = row-major (N2TO0_OUT, NHIDDEN*2)
    // = np.ravel(agg_2to0.mixing.weight): w2_2to0[o*(NHIDDEN*2) + h*2 + a].
    // N2TO0_OUT=1 reduces to w2_2to0[h*2+a].
    for (unsigned int h = 0; h < NHIDDEN; h++) {
    #pragma HLS unroll
      for (unsigned int a = 0; a < 2; a++) {
      #pragma HLS unroll
        for (unsigned int o = 0; o < N2TO0_OUT; o++) {
        #pragma HLS unroll
          Mult2to0: Rp[o] += w2_2to0[o*(NHIDDEN*2) + (h*2) + a]*R[h][a];
        }
      }
    }

#ifdef NPELICAN_HEAD
    //Nonlinear head (--head-hidden K): ReLU, then the agg_2to0.act_layer quantizer
    //(relu0_t, AP_RND_CONV) on the K channels; then the K->NOUT head MAC in mach_t
    //(exact products, float bias with guard bits), rounded once at the result_t cast.
    #pragma HLS ARRAY_PARTITION variable=w_head complete dim=0
    #pragma HLS ARRAY_PARTITION variable=b_head complete dim=0
    relu0_t Rq[NPELICAN_HEAD];
    #pragma HLS ARRAY_PARTITION variable=Rq complete dim=0
    for (unsigned int k = 0; k < NPELICAN_HEAD; k++) {
    #pragma HLS unroll
      mac0_t v = Rp[k];
      if (v < 0){
        v = 0;
        }
      Rq[k] = (relu0_t)v;
    }
    mach_t Hp[NOUT];
    #pragma HLS ARRAY_PARTITION variable=Hp complete dim=0
    for (unsigned int o = 0; o < NOUT; o++) {
    #pragma HLS unroll
      Hp[o] = b_head[o];
    }
    // head weights: row-major (NOUT, K) = np.ravel(head.weight): w_head[o*K + k]
    for (unsigned int k = 0; k < NPELICAN_HEAD; k++) {
    #pragma HLS unroll
      for (unsigned int o = 0; o < NOUT; o++) {
      #pragma HLS unroll
        MultHead: Hp[o] += w_head[o*NPELICAN_HEAD + k]*Rq[k];
      }
    }
#endif

#ifndef __SYNTHESIS__
    // Stage dump (csim only): written when npelican_dump_fp is non-null.
    // EXACT-match stages vs the PyTorch golden dump (true quantization points):
    //   dots, T0..T5, Tp, R, Rp.
    // APPROX stages (PyTorch keeps them in float; firmware stores them on the next
    // grid, so expect tiny differences here — they are NOT mismatches):
    //   batch1 (= PyTorch's quantized T0, not raw batch1), jmass, jdotp, Tr.
    if (npelican_dump_fp) {
        FILE* fp = npelican_dump_fp;

        // dots: NPARTICLES2^2 values (484 at default), row-major i*NPARTICLES2+j
        fprintf(fp, "dots:");
        for (unsigned int i = 0; i < NPARTICLES2; i++)
            for (unsigned int j = 0; j < NPARTICLES2; j++)
                fprintf(fp, " %.17g", (double)dots[i*NPARTICLES2+j]);
        fprintf(fp, "\n");

        // batch1: NPARTICLES2^2 values, row-major (t2-grid; approx)
        fprintf(fp, "batch1:");
        for (unsigned int i = 0; i < NPARTICLES2; i++)
            for (unsigned int j = 0; j < NPARTICLES2; j++)
                fprintf(fp, " %.17g", (double)batch1[i*NPARTICLES2+j]);
        fprintf(fp, "\n");

        // jmass: 1 value (post-normalization, t2-grid; approx)
        fprintf(fp, "jmass: %.17g\n", (double)jmass);

        // jdotp: NPARTICLES2 values (post-normalization, t2-grid; approx)
        fprintf(fp, "jdotp:");
        for (unsigned int i = 0; i < NPARTICLES2; i++)
            fprintf(fp, " %.17g", (double)jdotp[i]);
        fprintf(fp, "\n");

        // T0..T5: six lines, each NPARTICLES2^2 values row-major T[i][j][b] (exact)
        for (unsigned int b = 0; b < 6; b++) {
            fprintf(fp, "T%u:", b);
            for (unsigned int i = 0; i < NPARTICLES2; i++)
                for (unsigned int j = 0; j < NPARTICLES2; j++)
                    fprintf(fp, " %.17g", (double)T[i][j][b]);
            fprintf(fp, "\n");
        }

        // Tp: NPARTICLES2^2*NHIDDEN values, order i,j,h with h fastest (act_layer output, relu_t; exact)
        fprintf(fp, "Tp:");
        for (unsigned int i = 0; i < NPARTICLES2; i++)
            for (unsigned int j = 0; j < NPARTICLES2; j++)
                for (unsigned int h = 0; h < NHIDDEN; h++)
                    fprintf(fp, " %.17g", (double)Tp_q[i][j][h]);
        fprintf(fp, "\n");

        // Tr: NPARTICLES2^2*NHIDDEN values, same order (t0-grid; approx). Reconstructed for the dump ONLY —
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

        // R: NHIDDEN*2 values, order R[0][0] R[0][1] R[1][0] R[1][1] ... (exact)
        fprintf(fp, "R:");
        for (unsigned int h = 0; h < NHIDDEN; h++)
            for (unsigned int a = 0; a < 2; a++)
                fprintf(fp, " %.17g", (double)R[h][a]);
        fprintf(fp, "\n");

#ifdef NPELICAN_HEAD
        // Rq: K values (agg_2to0.act_layer output, relu0_t; exact)
        fprintf(fp, "Rq:");
        for (unsigned int k = 0; k < NPELICAN_HEAD; k++)
            fprintf(fp, " %.17g", (double)Rq[k]);
        fprintf(fp, "\n");
        // Hp: NOUT values (head MAC, pre output_quant)
        fprintf(fp, "Hp:");
        for (unsigned int o = 0; o < NOUT; o++)
            fprintf(fp, " %.17g", (double)Hp[o]);
        fprintf(fp, "\n");
        // Rp: ALWAYS the final logits (== golden dump Rp), here the head output
        fprintf(fp, "Rp:");
        for (unsigned int o = 0; o < NOUT; o++)
            fprintf(fp, " %.17g", (double)Hp[o]);
        fprintf(fp, "\n");
#else
        // Rp: NOUT values (output_quant grid; exact)
        fprintf(fp, "Rp:");
        for (unsigned int o = 0; o < NOUT; o++)
            fprintf(fp, " %.17g", (double)Rp[o]);
        fprintf(fp, "\n");
#endif
    }
#endif

    for (unsigned int o = 0; o < NOUT; o++) {
    #pragma HLS unroll
#ifdef NPELICAN_HEAD
      model_out[o] = (result_t)Hp[o];
#else
      model_out[o] = (result_t)Rp[o];
#endif
    }
}
