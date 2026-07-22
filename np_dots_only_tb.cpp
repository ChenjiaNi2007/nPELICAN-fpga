// Self-contained testbench for the standalone dot-product front-end
// (firmware/np_dots_only.cpp). No golden files, no nnet helpers: it hardcodes a
// hand-verifiable event so the printed dots can be checked by inspection, and it
// gates the result against the expected values on the current dot_t grid.
//
// Local build (no Vitis):
//   g++ -std=c++17 -O2 -I third_party/stubs -I third_party/ap_types/include -I . \
//       np_dots_only_tb.cpp firmware/np_dots_only.cpp -o tb_dots_only
//   ./tb_dots_only
//
// Also used as the -tb file by build_dots_only.tcl (csim before csynth).
#include <cstdio>
#include "firmware/nPELICAN.h"

// Prototype of the standalone top (defined in firmware/np_dots_only.cpp).
void np_dots_only(input_t model_input[(NPARTICLES)*4],
                  input_t beam_input[2*4],
                  nobj_t  nobj,
                  dot_t   dots_out[(NPARTICLES2)*(NPARTICLES2)]);

int main() {
    // ---- Hand-verifiable event -------------------------------------------
    // 20 identical particles, each 4-vector (E,px,py,pz) = (6,0,0,2).
    //   particle . particle (any pair, incl. diagonal) = 36 - 0 - 0 - 4 = 32
    // Beam spurions (1,0,0,+1) and (1,0,0,-1):
    //   beam0 . particle = E - pz = 6 - 2 = 4
    //   beam1 . particle = E + pz = 6 + 2 = 8
    //   beam0 . beam0    = 1 - 1 = 0 ;  beam1 . beam1 = 1 - 1 = 0
    //   beam0 . beam1    = 1 - (1)(-1) = 2   <-- see note below
    // 0, 4, 8, 32 are exact multiples of the current dot_t LSB (=4) so they pass
    // through unrounded. beam0.beam1 = 2 is exactly half an LSB: AP_RND_CONV
    // (round-half-to-even) sends it to 0. That single value is the intended
    // demonstration that the input_quant grid is real, not an artifact.
    input_t model_input[NPARTICLES*4];
    for (int p = 0; p < NPARTICLES; p++) {
        model_input[p*4+0] = 6;   // E
        model_input[p*4+1] = 0;   // px
        model_input[p*4+2] = 0;   // py
        model_input[p*4+3] = 2;   // pz
    }
    input_t beam_input[8] = {1, 0, 0, 1,   1, 0, 0, -1};
    nobj_t  nobj = 20;            // remaps to 22 -> all 20 particles + 2 beams active

    dot_t dots[NPARTICLES2*NPARTICLES2];
    np_dots_only(model_input, beam_input, nobj, dots);

    // ---- Expected value on the dot_t grid, by (i,j) category -------------
    // index 0,1 = beams; 2..21 = particles.
    auto expected = [](int i, int j) -> double {
        bool bi = (i < 2), bj = (j < 2);
        if ( bi &&  bj) return 0.0;                 // beam-beam: 0, 0, and 2->0
        if (!bi && !bj) return 32.0;                // particle-particle
        int beam = bi ? i : j;                      // beam index for beam-particle
        return (beam == 0) ? 4.0 : 8.0;             // beam0->4, beam1->8
    };

    // ---- Report a few representative entries + a full gate ----------------
    // Smallest positive dot_t step = its LSB (grid resolution), reported for context.
    dot_t lsb; lsb.setBits(1);
    printf("dot_t grid: LSB=%g (ap_fixed<6,8>); input_t=ap_fixed<18,12>\n", (double)lsb);
    printf("Representative dots (i,j -> value):\n");
    printf("  beam0.beam0  (0,0)  = %6g  [expect 0]\n",  (double)dots[0*NPARTICLES2+0]);
    printf("  beam0.beam1  (0,1)  = %6g  [expect 0; raw Minkowski = 2, rounds to 0 on LSB=4]\n",
                                                          (double)dots[0*NPARTICLES2+1]);
    printf("  beam1.beam1  (1,1)  = %6g  [expect 0]\n",  (double)dots[1*NPARTICLES2+1]);
    printf("  beam0.part0  (0,2)  = %6g  [expect 4]\n",  (double)dots[0*NPARTICLES2+2]);
    printf("  beam1.part0  (1,2)  = %6g  [expect 8]\n",  (double)dots[1*NPARTICLES2+2]);
    printf("  part0.part0  (2,2)  = %6g  [expect 32]\n", (double)dots[2*NPARTICLES2+2]);
    printf("  part0.part5  (2,7)  = %6g  [expect 32]\n", (double)dots[2*NPARTICLES2+7]);

    int checked = 0, bad = 0;
    for (int i = 0; i < NPARTICLES2; i++)
        for (int j = 0; j < NPARTICLES2; j++) {
            double got = (double)dots[i*NPARTICLES2+j], exp = expected(i, j);
            checked++;
            if (got != exp) {
                if (bad < 10)
                    printf("  MISMATCH (%d,%d): got %g, expected %g\n", i, j, got, exp);
                bad++;
            }
        }

    // Symmetry check: dots[i][j] must equal dots[j][i] everywhere.
    int asym = 0;
    for (int i = 0; i < NPARTICLES2; i++)
        for (int j = 0; j < NPARTICLES2; j++)
            if (dots[i*NPARTICLES2+j] != dots[j*NPARTICLES2+i]) asym++;

    printf("\nGATE: %s  (%d/%d entries correct, %d asymmetric)\n",
           (bad == 0 && asym == 0) ? "PASS" : "FAIL", checked - bad, checked, asym);
    return (bad == 0 && asym == 0) ? 0 : 1;
}
