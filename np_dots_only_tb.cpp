// Self-contained testbench for the standalone dot-product front-end
// (firmware/np_dots_only.cpp). No golden files, no nnet helpers: it hardcodes a
// simple event and checks each output dot against the exact Minkowski value
// rounded into the CURRENT dot_t grid. Because dot_t (the input_quant grid) is
// regenerated per checkpoint (its scale/LSB varies — e.g. LSB 4 on one
// checkpoint, 16 on another), the reference is computed grid-adaptively rather
// than hardcoded, so this gate is valid for ANY types_generated.h.
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
    // ---- Simple event ----------------------------------------------------
    // 20 identical particles, each 4-vector (E,px,py,pz) = (6,0,0,2); standard
    // beam spurions (1,0,0,+1)/(1,0,0,-1); full occupancy (nobj=20 -> remaps to
    // 22 so all 20 particles + 2 beams are active, mask all ones).
    //   raw Minkowski dots (E1 E2 - px1 px2 - py1 py2 - pz1 pz2):
    //     particle.particle = 36 - 4      = 32
    //     beam0.particle    = 6 - 2       = 4
    //     beam1.particle    = 6 + 2       = 8
    //     beam0.beam0 = beam1.beam1       = 0
    //     beam0.beam1 = 1 - (-1)          = 2
    // Each is then rounded into dot_t (AP_RND_CONV/AP_SAT) — the grid depends on
    // the loaded checkpoint, which is exactly what the reference below applies.
    input_t model_input[NPARTICLES*4];
    for (int p = 0; p < NPARTICLES; p++) {
        model_input[p*4+0] = 6;   // E
        model_input[p*4+1] = 0;   // px
        model_input[p*4+2] = 0;   // py
        model_input[p*4+3] = 2;   // pz
    }
    input_t beam_input[8] = {1, 0, 0, 1,   1, 0, 0, -1};
    nobj_t  nobj = 20;

    dot_t dots[NPARTICLES2*NPARTICLES2];
    np_dots_only(model_input, beam_input, nobj, dots);

    // ---- Grid-adaptive reference -----------------------------------------
    // Rebuild the momentum table the firmware sees (beams at 0,1; particles at
    // 2..21; full occupancy so no masking), then for each (i,j) compute the
    // exact Minkowski dot and round it into dot_t the SAME way the hardware does
    // -> the expected value on whatever grid this checkpoint uses.
    double P[NPARTICLES2][4];
    for (int k = 0; k < 4; k++) {
        P[0][k] = (double)beam_input[0*4+k];
        P[1][k] = (double)beam_input[1*4+k];
    }
    for (int p = 0; p < NPARTICLES; p++)
        for (int k = 0; k < 4; k++)
            P[p+2][k] = (double)model_input[p*4+k];

    auto raw_dot = [&](int i, int j) -> double {
        return P[i][0]*P[j][0] - P[i][1]*P[j][1] - P[i][2]*P[j][2] - P[i][3]*P[j][3];
    };
    auto expected = [&](int i, int j) -> double {
        return (double)(dot_t)raw_dot(i, j);   // exact value rounded onto the current grid
    };

    // ---- Report grid + representative entries (raw -> on-grid) -------------
    dot_t lsb; lsb.setBits(1);
    printf("dot_t grid: LSB=%g  (input_quant scale; checkpoint-dependent)\n", (double)lsb);
    printf("Representative dots (i,j):  raw -> on grid (got | expected)\n");
    int rep[7][2] = {{0,0},{0,1},{1,1},{0,2},{1,2},{2,2},{2,7}};
    const char* lbl[7] = {"beam0.beam0","beam0.beam1","beam1.beam1",
                          "beam0.part0","beam1.part0","part0.part0","part0.part5"};
    for (int r = 0; r < 7; r++) {
        int i = rep[r][0], j = rep[r][1];
        printf("  %-12s (%d,%2d)  raw=%4g -> %6g | %6g\n", lbl[r], i, j,
               raw_dot(i, j), (double)dots[i*NPARTICLES2+j], expected(i, j));
    }

    // ---- Full gate: every entry matches the on-grid reference, symmetric ---
    int checked = 0, bad = 0, asym = 0;
    for (int i = 0; i < NPARTICLES2; i++)
        for (int j = 0; j < NPARTICLES2; j++) {
            double got = (double)dots[i*NPARTICLES2+j], exp = expected(i, j);
            checked++;
            if (got != exp) {
                if (bad < 10)
                    printf("  MISMATCH (%d,%d): got %g, expected %g (raw %g)\n",
                           i, j, got, exp, raw_dot(i, j));
                bad++;
            }
            if (dots[i*NPARTICLES2+j] != dots[j*NPARTICLES2+i]) asym++;
        }

    printf("\nGATE: %s  (%d/%d entries match on-grid reference, %d asymmetric)\n",
           (bad == 0 && asym == 0) ? "PASS" : "FAIL", checked - bad, checked, asym);
    return (bad == 0 && asym == 0) ? 0 : 1;
}
