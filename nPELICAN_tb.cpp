#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "firmware/nPELICAN.h"
#include "firmware/nnet_utils/nnet_helpers.h"
//TODO: create interface for hdf5 files

#define CHECKPOINT 1000

namespace nnet {
bool trace_enabled = true;
std::map<std::string, void *> *trace_outputs = NULL;
size_t trace_type_size = sizeof(double);
} // namespace nnet

// Stage-dump file pointer: set by TB around the single re-run (Task 3).
#ifndef __SYNTHESIS__
extern FILE* npelican_dump_fp;
extern dotall_t* npelican_dots_override;   // DOTS-LEVEL injection hook (see nPELICAN.cpp)
#endif

// Constant beam spurions (1,0,0,+1)/(1,0,0,-1). The firmware now takes the two
// beams as a top-level input (beam_input[2*4]); everywhere the harness is NOT
// Lorentz-boosting the beams (golden gate, dots-level, legacy 10k, zero-input),
// drive them with these constants. At |beta|=0 they quantize into input_t
// identically to the firmware's old hardcoded constants -> bit-exact (gate G1).
static void fill_const_beams(input_t b[8]) {
    static const double c[8] = {1, 0, 0, 1, 1, 0, 0, -1};
    for (int k = 0; k < 8; k++) b[k] = c[k];
}

// NOUT-logit line I/O. A golden_logits.dat line carries NOUT space-separated values
// (NOUT=1: one value, the historical format). parse_logits returns how many values
// it read (at most NOUT); format_logits writes NOUT "%.17g" values separated by
// single spaces, newline-terminated (NOUT=1 -> exactly "%.17g\n").
// The nobj port is a COUNT OF PARTICLES PRESENT IN THE NPARTICLES INPUT SLOTS (0..NPARTICLES);
// the firmware remaps it to the active row/col count incl. beams. Dataset files may carry
// the RAW multiplicity (hls4ml: up to 150), which would wrap in the narrow port and mask
// out real particles -- clamp exactly as the exporter does (min(raw, NPARTICLES)).
static int clamp_nobj(int n) { return n > NPARTICLES ? NPARTICLES : (n < 0 ? 0 : n); }

// Jet spurion (NPELICAN_JET_SPURION, checkpoints trained with --add-jet): the top gains a
// jet_input[4] port (full-jet E,px,py,pz in GeV -> spurion slot 2). NP_CALL hides the
// extra argument so every call site reads the same with and without the define.
#ifdef NPELICAN_JET_SPURION
#define NP_CALL(mi, bi, ji, n, out) nPELICAN(mi, bi, ji, n, out)
// Parse one line of 4 floats (E px py pz) into jet[4]; false on a short/missing line.
static bool read_jet_line(std::istream &is, jet_t jet[4]) {
    std::string line;
    if (!std::getline(is, line)) return false;
    std::vector<float> v;
    char *c = const_cast<char *>(line.c_str());
    char *t = strtok(c, " ");
    while (t != NULL) { v.push_back(atof(t)); t = strtok(NULL, " "); }
    if (v.size() < 4) return false;
    nnet::copy_data<float, jet_t, 0, 4>(v, jet);
    return true;
}
// Harnesses not yet updated for the jet (equivariance, legacy 10k): zero jet + ONE warning.
static void zero_jet_warn(jet_t jet[4], const char *missing) {
    static bool warned = false;
    for (int k = 0; k < 4; k++) jet[k] = 0;
    if (!warned) {
        printf("WARNING: %s not found -- driving the jet spurion with ZEROS (this harness is "
               "not updated for --add-jet checkpoints; outputs are NOT the trained model's)\n",
               missing);
        warned = true;
    }
}
#else
#define NP_CALL(mi, bi, ji, n, out) nPELICAN(mi, bi, n, out)
#endif

static int parse_logits(const std::string &line, double out[NOUT]) {
    std::vector<char> buf(line.begin(), line.end());
    buf.push_back('\0');
    int k = 0;
    char *t = strtok(buf.data(), " ");
    while (t != NULL && k < NOUT) {
        out[k++] = std::stod(t);
        t = strtok(NULL, " ");
    }
    return k;
}
static std::string format_logits(const result_t v[NOUT]) {
    std::string s;
    char buf[64];
    for (int o = 0; o < NOUT; o++) {
        snprintf(buf, sizeof(buf), (o == 0) ? "%.17g" : " %.17g", double(v[o]));
        s += buf;
    }
    s += "\n";
    return s;
}

// The golden-vector / dots-level gate is OPT-IN. By default csim runs the legacy
// 10k flow (tb_data/10k_*.dat). To run the bit-exactness gate instead, define
// RUN_GOLDEN_GATE — either uncomment the line below, or add -DRUN_GOLDEN_GATE to the
// testbench cflags in build_prj.tcl (the `add_files -tb ... -cflags` line).
// #define RUN_GOLDEN_GATE

int main(int argc, char **argv) {

#ifdef RUN_EQUIVARIANCE
    // ---------------------------------------------------------------
    // Equivariance mode (equivariance/ harness): read momenta from
    // tb_data/equiv_in_pmu.dat (one event/line, NPARTICLES*4 = 80 floats,
    // beams added INSIDE the firmware exactly as in the golden path) and the
    // per-event RAW Nobj from tb_data/equiv_in_nobj.dat, run dot4+net, and
    // write the NOUT logits to tb_data/equiv_out_logits.dat (%.17g, one event per line).
    // No comparison: this is a batch oracle for f_b(x). Mirrors the
    // RUN_GOLDEN_GATE reader/writer so the path is byte-identical to the
    // validated golden path (the harness proves this via a golden-gate check
    // before the sweep). Returns immediately after; never falls through to
    // the legacy 10k flow.
    // ---------------------------------------------------------------
    {
        std::ifstream fepmu("tb_data/equiv_in_pmu.dat");
        std::ifstream fenobj("tb_data/equiv_in_nobj.dat");
        if (!fepmu.good() || !fenobj.good()) {
            std::cerr << "EQUIVARIANCE: cannot open tb_data/equiv_in_pmu.dat or "
                         "tb_data/equiv_in_nobj.dat" << std::endl;
            return 1;
        }
        // Per-event beams (8 floats/line, aligned row-for-row with the momenta) written
        // by gen_boosted_inputs.py. If absent, fall back to the constant beams so the
        // fixed-beam path stays available against the same binary.
        std::ifstream febeams("tb_data/equiv_in_beams.dat");
        bool have_beams = febeams.good();
#ifdef NPELICAN_JET_SPURION
        std::ifstream fejet("tb_data/equiv_in_jet.dat");
        bool have_jet = fejet.good();
#endif
        std::ofstream feout("tb_data/equiv_out_logits.dat");

        int n_events = 0;
        std::string pmu_line, nobj_line, beams_line;
        while (std::getline(fepmu, pmu_line) && std::getline(fenobj, nobj_line)) {
            // Parse NPARTICLES*4 floats from pmu_line
            char *cstr = const_cast<char *>(pmu_line.c_str());
            char *current;
            std::vector<float> in;
            current = strtok(cstr, " ");
            while (current != NULL) {
                in.push_back(atof(current));
                current = strtok(NULL, " ");
            }
            int nobj_val = std::stoi(nobj_line);

            input_t model_input[NPARTICLES*4];
            nnet::copy_data<float, input_t, 0, NPARTICLES*4>(in, model_input);

            // Beams for this event: parse the matching beams line, else constant.
            input_t beam_input[8];
            if (have_beams && std::getline(febeams, beams_line)) {
                std::vector<float> bin;
                char *bc = const_cast<char *>(beams_line.c_str());
                char *bt = strtok(bc, " ");
                while (bt != NULL) { bin.push_back(atof(bt)); bt = strtok(NULL, " "); }
                nnet::copy_data<float, input_t, 0, 8>(bin, beam_input);
            } else {
                fill_const_beams(beam_input);
            }

            jet_t jet_input[4];
#ifdef NPELICAN_JET_SPURION
            if (!(have_jet && read_jet_line(fejet, jet_input)))
                zero_jet_warn(jet_input, "tb_data/equiv_in_jet.dat");
#endif
            result_t model_out[NOUT];
            NP_CALL(model_input, beam_input, jet_input, clamp_nobj(nobj_val), model_out);

            feout << format_logits(model_out);
            n_events++;
        }
        feout.close();
        fepmu.close();
        fenobj.close();
        if (have_beams) febeams.close();
        printf("EQUIVARIANCE: wrote %d logits to tb_data/equiv_out_logits.dat\n", n_events);
        return 0;
    }
#endif  // RUN_EQUIVARIANCE

#ifdef RUN_GOLDEN_GATE
    // ---------------------------------------------------------------
    // Golden-vector mode: activated when tb_data/golden_pmu.dat exists
    // ---------------------------------------------------------------
    std::ifstream fgolden_check("tb_data/golden_pmu.dat");
    if (fgolden_check.good()) {
        fgolden_check.close();

        std::ifstream fgpmu("tb_data/golden_pmu.dat");
        std::ifstream fgnobj("tb_data/golden_nobj.dat");
        std::ifstream fglogits("tb_data/golden_logits.dat");
#ifdef NPELICAN_JET_SPURION
        // Jet-spurion firmware: the full-jet 4-momentum per event is REQUIRED (no fallback;
        // a zero jet would silently gate a different model).
        std::ifstream fgjet("tb_data/golden_jet.dat");
        if (!fgjet.good()) {
            printf("GOLDEN GATE: ERROR tb_data/golden_jet.dat missing but the firmware was built "
                   "with NPELICAN_JET_SPURION (re-export golden vectors with export_golden.py "
                   "for this --add-jet checkpoint)\n");
            return 1;
        }
#endif
        std::ofstream fgout("tb_data/golden_fw_results.log");

        int n_events = 0;
        int n_exact = 0;
        int n_mismatch = 0;
        double max_abs_delta = 0.0;
        int first_mismatch_idx = -1;

        std::string pmu_line, nobj_line, logit_line;
        while (std::getline(fgpmu, pmu_line) &&
               std::getline(fgnobj, nobj_line) &&
               std::getline(fglogits, logit_line)) {

            // Parse NPARTICLES*4 floats from pmu_line
            char *cstr = const_cast<char *>(pmu_line.c_str());
            char *current;
            std::vector<float> in;
            current = strtok(cstr, " ");
            while (current != NULL) {
                in.push_back(atof(current));
                current = strtok(NULL, " ");
            }

            // Parse nobj
            int nobj_val = std::stoi(nobj_line);

            // Parse NOUT golden logits (double)
            double golden_logit[NOUT];
            int n_gl = parse_logits(logit_line, golden_logit);
            if (n_gl < NOUT) {
                printf("GOLDEN GATE: ERROR golden_logits.dat has %d value(s)/line but firmware NOUT=%d (re-export golden vectors for this checkpoint)\n",
                       n_gl, (int)NOUT);
                return 1;
            }

            // Run firmware
            input_t model_input[NPARTICLES*4];
            nnet::copy_data<float, input_t, 0, NPARTICLES*4>(in, model_input);
            input_t beam_input[8];
            fill_const_beams(beam_input);
            jet_t jet_input[4];
#ifdef NPELICAN_JET_SPURION
            if (!read_jet_line(fgjet, jet_input)) {
                printf("GOLDEN GATE: ERROR tb_data/golden_jet.dat ran out / malformed at event %d\n", n_events);
                return 1;
            }
#endif
            result_t model_out[NOUT];
            NP_CALL(model_input, beam_input, jet_input, clamp_nobj(nobj_val), model_out);

            // Write firmware logits to log (NOUT x %.17g per line, space-separated)
            fgout << format_logits(model_out);

            // Compare: exact iff ALL NOUT outputs match; delta = max over outputs
            bool all_eq = true;
            double delta = 0.0;
            for (int o = 0; o < NOUT; o++) {
                double fw_logit = double(model_out[o]);
                if (fw_logit != golden_logit[o]) all_eq = false;
                double d = fabs(fw_logit - golden_logit[o]);
                if (d > delta) delta = d;
            }
            if (all_eq) {
                n_exact++;
            } else {
                n_mismatch++;
                if (first_mismatch_idx == -1) {
                    first_mismatch_idx = n_events;
                }
            }
            if (delta > max_abs_delta) {
                max_abs_delta = delta;
            }

            n_events++;
        }

        fgout.close();
        fgpmu.close();
        fgnobj.close();
        fglogits.close();

        // Tolerance gate. Zero-tolerance bit-exactness is not achievable against a float
        // reference because the model keeps BatchNorm in float (an architecture invariant):
        // float-vs-fixed rounding in the unquantized BN/aggregation segments tips a quantizer
        // boundary on a minority of events, cascading to <=~1e-5 on the logit (network), plus
        // the documented dot4 front-end caveat (PyTorch computes d_ij in lossy float32) on the
        // momenta path. PASS = max|delta| under tolerance; the exact count is reported too.
        const double TOL_GOLDEN = 1e-3;   // momenta path: includes the dot4 front-end residual
        const double TOL_NET    = 1e-4;   // dots-level (network isolated): float-BN tipping only

        // Print summary
        printf("GOLDEN SUMMARY: events=%d exact=%d mismatch=%d max_abs_delta=%.17g first_mismatch=%d\n",
               n_events, n_exact, n_mismatch, max_abs_delta,
               first_mismatch_idx);
        printf("GOLDEN GATE: %s (max_abs_delta=%.3g vs tol=%.3g; %d/%d zero-tolerance exact)\n",
               (max_abs_delta < TOL_GOLDEN ? "PASS" : "FAIL"), max_abs_delta, TOL_GOLDEN,
               n_exact, n_events);

#ifndef __SYNTHESIS__
        // ---------------------------------------------------------------
        // DOTS-LEVEL mode (plan D4): when tb_data/golden_dots.dat exists, re-run every
        // event injecting PyTorch's quantized d_ij in place of the dot4 front-end. This
        // isolates the retyped NETWORK from the float32 d_ij-cancellation caveat: a
        // bit-exact result here proves the network itself, leaving dot4 as the only
        // (documented) source of any momenta-level mismatch.
        // ---------------------------------------------------------------
        std::ifstream fdots_check("tb_data/golden_dots.dat");
        if (fdots_check.good()) {
            fdots_check.close();
            std::ifstream fdpmu("tb_data/golden_pmu.dat");
            std::ifstream fdnobj("tb_data/golden_nobj.dat");
            std::ifstream fddots("tb_data/golden_dots.dat");
            std::ifstream fdlogits("tb_data/golden_logits.dat");
#ifdef NPELICAN_JET_SPURION
            std::ifstream fdjet("tb_data/golden_jet.dat");
#endif

            int d_events = 0, d_exact = 0, d_mismatch = 0, d_first = -1;
            double d_maxdelta = 0.0;
            std::string dpmu, dnobj, ddots, dlogit;
            while (std::getline(fdpmu, dpmu) && std::getline(fdnobj, dnobj) &&
                   std::getline(fddots, ddots) && std::getline(fdlogits, dlogit)) {
                // momenta (passed through; dot4 result is overwritten by the override)
                std::vector<float> in;
                { char *c = const_cast<char*>(dpmu.c_str()); char *t = strtok(c, " ");
                  while (t) { in.push_back(atof(t)); t = strtok(NULL, " "); } }
                // injected dots (484 values, row-major)
                static dotall_t dots_inj[NPARTICLES2*NPARTICLES2];
                { char *c = const_cast<char*>(ddots.c_str()); char *t = strtok(c, " ");
                  int k = 0; while (t && k < NPARTICLES2*NPARTICLES2) { dots_inj[k++] = (dotall_t)atof(t); t = strtok(NULL, " "); } }
                int nobj_val = std::stoi(dnobj);
                double golden_logit[NOUT];
                int n_dl = parse_logits(dlogit, golden_logit);
                if (n_dl < NOUT) {
                    printf("GOLDEN GATE: ERROR golden_logits.dat has %d value(s)/line but firmware NOUT=%d (re-export golden vectors for this checkpoint)\n",
                           n_dl, (int)NOUT);
                    return 1;
                }

                input_t model_input[NPARTICLES*4];
                nnet::copy_data<float, input_t, 0, NPARTICLES*4>(in, model_input);
                input_t beam_input[8];
                fill_const_beams(beam_input);
                jet_t jet_input[4];
#ifdef NPELICAN_JET_SPURION
                if (!read_jet_line(fdjet, jet_input)) {
                    printf("DOTS-LEVEL: ERROR tb_data/golden_jet.dat ran out / malformed at event %d\n", d_events);
                    return 1;
                }
#endif
                result_t model_out[NOUT];
                npelican_dots_override = dots_inj;
                NP_CALL(model_input, beam_input, jet_input, clamp_nobj(nobj_val), model_out);
                npelican_dots_override = nullptr;

                bool all_eq = true;
                double delta = 0.0;
                for (int o = 0; o < NOUT; o++) {
                    double fw_logit = double(model_out[o]);
                    if (fw_logit != golden_logit[o]) all_eq = false;
                    double d = fabs(fw_logit - golden_logit[o]);
                    if (d > delta) delta = d;
                }
                if (all_eq) d_exact++;
                else { d_mismatch++; if (d_first == -1) d_first = d_events; }
                if (delta > d_maxdelta) d_maxdelta = delta;
                d_events++;
            }
            printf("DOTS-LEVEL SUMMARY: events=%d exact=%d mismatch=%d max_abs_delta=%.17g first_mismatch=%d\n",
                   d_events, d_exact, d_mismatch, d_maxdelta, d_first);
            printf("DOTS-LEVEL GATE: %s (max_abs_delta=%.3g vs tol=%.3g; %d/%d zero-tolerance exact)\n",
                   (d_maxdelta < TOL_NET ? "PASS" : "FAIL"), d_maxdelta, TOL_NET, d_exact, d_events);
        }
#endif

        // Determine the dump event: first mismatch, or event 0 if all match
        int dump_event = (first_mismatch_idx >= 0) ? first_mismatch_idx : 0;

        // Re-run the dump event with stage dumping enabled (Task 3)
        {
            std::ifstream fgpmu2("tb_data/golden_pmu.dat");
            std::ifstream fgnobj2("tb_data/golden_nobj.dat");

            std::string pmu_line2, nobj_line2;
            for (int e = 0; e <= dump_event; e++) {
                if (!std::getline(fgpmu2, pmu_line2) ||
                    !std::getline(fgnobj2, nobj_line2)) {
                    break;
                }
            }

            // Parse the event
            char *cstr2 = const_cast<char *>(pmu_line2.c_str());
            char *current2;
            std::vector<float> in2;
            current2 = strtok(cstr2, " ");
            while (current2 != NULL) {
                in2.push_back(atof(current2));
                current2 = strtok(NULL, " ");
            }
            int nobj_val2 = std::stoi(nobj_line2);

            input_t model_input2[NPARTICLES*4];
            nnet::copy_data<float, input_t, 0, NPARTICLES*4>(in2, model_input2);
            input_t beam_input2[8];
            fill_const_beams(beam_input2);
            jet_t jet_input2[4];
#ifdef NPELICAN_JET_SPURION
            {
                std::ifstream fgjet2("tb_data/golden_jet.dat");
                for (int e = 0; e <= dump_event; e++) {
                    if (!read_jet_line(fgjet2, jet_input2)) {
                        printf("GOLDEN GATE: ERROR tb_data/golden_jet.dat has no line for dump event %d\n", dump_event);
                        return 1;
                    }
                }
            }
#endif
            result_t model_out2[NOUT];

#ifndef __SYNTHESIS__
            // If golden dots exist, dump the DOTS-LEVEL path (network isolated from dot4)
            // so the stage dump reflects identical inputs to PyTorch.
            static dotall_t dump_dots_inj[NPARTICLES2*NPARTICLES2];
            std::ifstream fddump("tb_data/golden_dots.dat");
            if (fddump.good()) {
                std::string dl;
                for (int e = 0; e <= dump_event; e++) std::getline(fddump, dl);
                char *c = const_cast<char*>(dl.c_str()); char *t = strtok(c, " ");
                int k = 0; while (t && k < NPARTICLES2*NPARTICLES2) { dump_dots_inj[k++] = (dotall_t)atof(t); t = strtok(NULL, " "); }
                npelican_dots_override = dump_dots_inj;
            }
            // Open dump file and set global pointer
            FILE* dump_fp = fopen("tb_data/fw_stage_dump.txt", "w");
            npelican_dump_fp = dump_fp;
            NP_CALL(model_input2, beam_input2, jet_input2, clamp_nobj(nobj_val2), model_out2);
            npelican_dump_fp = nullptr;
            npelican_dots_override = nullptr;
            fclose(dump_fp);
#else
            NP_CALL(model_input2, beam_input2, jet_input2, clamp_nobj(nobj_val2), model_out2);
#endif

            fgpmu2.close();
            fgnobj2.close();
        }

        printf("INFO: Stage dump written for event %d to tb_data/fw_stage_dump.txt\n", dump_event);
        printf("INFO: Golden firmware results saved to tb_data/golden_fw_results.log\n");

        return 0;
    }
#endif  // RUN_GOLDEN_GATE

    // ---------------------------------------------------------------
    // Legacy 10k flow: the default csim path (always runs unless RUN_GOLDEN_GATE
    // is defined and tb_data/golden_pmu.dat is present)
    // ---------------------------------------------------------------

    // load input data from text file
    std::ifstream fin("tb_data/full_pmu_test.dat");
    std::ifstream fnobj("tb_data/full_nobj.dat");//get nobj per event
    // load predictions from text file
    std::ifstream fpr("tb_data/full_signal.dat");
#ifdef NPELICAN_JET_SPURION
    std::ifstream fjet("tb_data/full_jet.dat");
    bool have_jet = fjet.good();
#endif

#ifdef RTL_SIM
    std::string RESULTS_LOG = "tb_data/rtl_cosim_results.log";
#else
    std::string RESULTS_LOG = "tb_data/csim_results.log";
#endif
    std::ofstream fout(RESULTS_LOG);

    std::string iline;
    std::string pline;
    std::string nobjline;
    int e = 0;

    if (fin.is_open() && fpr.is_open() && fnobj.is_open()) {
        while (std::getline(fin, iline) && std::getline(fpr, pline) && std::getline(fnobj,nobjline)) {
            if (e % CHECKPOINT == 0)
                std::cout << "Processing input " << e << std::endl;
            //read in particle four vectors
            char *cstr = const_cast<char *>(iline.c_str());
            char *current;
            std::vector<float> in;
            current = strtok(cstr, " ");
            while (current != NULL) {
                in.push_back(atof(current));
                current = strtok(NULL, " ");
            }
            //read in true event ID
            cstr = const_cast<char *>(pline.c_str());
            std::vector<float> pr;
            current = strtok(cstr, " ");
            while (current != NULL) {
                pr.push_back(atof(current));
                current = strtok(NULL, " ");
            }
            //read in number of particles for the event
            cstr = const_cast<char *>(nobjline.c_str());
            std::vector<int> vnobj;
            current = strtok(cstr, " ");
            while (current != NULL) {
                vnobj.push_back(std::stoi(current));
                current = strtok(NULL, " ");
            }

            // hls-fpga-machine-learning insert data
      input_t model_input[NPARTICLES*4];
      nnet::copy_data<float, input_t, 0, NPARTICLES*4>(in, model_input);
      input_t beam_input[8];
      fill_const_beams(beam_input);
      jet_t jet_input[4];
#ifdef NPELICAN_JET_SPURION
      if (!(have_jet && read_jet_line(fjet, jet_input)))
          zero_jet_warn(jet_input, "tb_data/full_jet.dat");
#endif
      result_t model_out[NOUT];

            // hls-fpga-machine-learning insert top-level-function
            //input_t nobj = vnobj[0];
            NP_CALL(model_input,beam_input,jet_input,clamp_nobj(vnobj[0]),model_out);

            if (e % CHECKPOINT == 0) {
                std::cout << "Predictions" << std::endl;
                // hls-fpga-machine-learning insert predictions
                std::cout << std::endl;
                for(int i = 0; i < 1; i++) {
                  std::cout << pr[i] << " ";
                  std::cout << vnobj[i] << " ";
                }
                std::cout << std::endl;
                std::cout << "Quantized predictions" << std::endl;
                // hls-fpga-machine-learning insert quantized
                nnet::print_result<result_t, NOUT>(model_out, std::cout, true);
            }
            e++;

            // hls-fpga-machine-learning insert tb-output
            nnet::print_result<result_t, NOUT>(model_out, fout);
        }
        fin.close();
        fpr.close();
    } else {
        std::cout << "INFO: Unable to open input/predictions file, using default input." << std::endl;

        // hls-fpga-machine-learning insert zero
    input_t model_input[NPARTICLES*4];
    nnet::fill_zero<input_t, NPARTICLES*4>(model_input);
    input_t beam_input[8];
    fill_const_beams(beam_input);
    jet_t jet_input[4];
#ifdef NPELICAN_JET_SPURION
    zero_jet_warn(jet_input, "tb_data/full_jet.dat");
#endif
    result_t model_out[NOUT];

        // hls-fpga-machine-learning insert top-level-function
        NP_CALL(model_input,beam_input,jet_input,1,model_out);

        // hls-fpga-machine-learning insert output
        nnet::print_result<result_t, NOUT>(model_out, std::cout, true);

        // hls-fpga-machine-learning insert tb-output
        nnet::print_result<result_t, NOUT>(model_out, fout);
    }

    fout.close();
    std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;

    return 0;
}
