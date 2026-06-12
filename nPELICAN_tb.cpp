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
#endif

int main(int argc, char **argv) {

    // ---------------------------------------------------------------
    // Golden-vector mode: activated when tb_data/golden_pmu.dat exists
    // ---------------------------------------------------------------
    std::ifstream fgolden_check("tb_data/golden_pmu.dat");
    if (fgolden_check.good()) {
        fgolden_check.close();

        std::ifstream fgpmu("tb_data/golden_pmu.dat");
        std::ifstream fgnobj("tb_data/golden_nobj.dat");
        std::ifstream fglogits("tb_data/golden_logits.dat");
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

            // Parse NPARTICLES*4 = 80 floats from pmu_line
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

            // Parse golden logit (double)
            double golden_logit = std::stod(logit_line);

            // Run firmware
            input_t model_input[NPARTICLES*4];
            nnet::copy_data<float, input_t, 0, NPARTICLES*4>(in, model_input);
            result_t model_out[1];
            nPELICAN(model_input, nobj_val, model_out);

            double fw_logit = double(model_out[0]);

            // Write firmware logit to log (%.17g, one value per line)
            char buf[64];
            snprintf(buf, sizeof(buf), "%.17g\n", fw_logit);
            fgout << buf;

            // Compare
            double delta = fabs(fw_logit - golden_logit);
            if (fw_logit == golden_logit) {
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

        // Print summary
        printf("GOLDEN SUMMARY: events=%d exact=%d mismatch=%d max_abs_delta=%.17g first_mismatch=%d\n",
               n_events, n_exact, n_mismatch, max_abs_delta,
               first_mismatch_idx);

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
            result_t model_out2[1];

#ifndef __SYNTHESIS__
            // Open dump file and set global pointer
            FILE* dump_fp = fopen("tb_data/fw_stage_dump.txt", "w");
            npelican_dump_fp = dump_fp;
            nPELICAN(model_input2, nobj_val2, model_out2);
            npelican_dump_fp = nullptr;
            fclose(dump_fp);
#else
            nPELICAN(model_input2, nobj_val2, model_out2);
#endif

            fgpmu2.close();
            fgnobj2.close();
        }

        printf("INFO: Stage dump written for event %d to tb_data/fw_stage_dump.txt\n", dump_event);
        printf("INFO: Golden firmware results saved to tb_data/golden_fw_results.log\n");

        return 0;
    }

    // ---------------------------------------------------------------
    // Legacy 10k flow (unchanged): runs when golden files are absent
    // ---------------------------------------------------------------

    // load input data from text file
    std::ifstream fin("tb_data/10k_pmu_test.dat");
    std::ifstream fnobj("tb_data/10k_nobj.dat");//get nobj per event
    // load predictions from text file
    std::ifstream fpr("tb_data/10k_signal.dat");

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
      result_t model_out[1];

            // hls-fpga-machine-learning insert top-level-function
            //input_t nobj = vnobj[0];
            nPELICAN(model_input,vnobj[0],model_out);

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
                nnet::print_result<result_t, 1>(model_out, std::cout, true);
            }
            e++;

            // hls-fpga-machine-learning insert tb-output
            nnet::print_result<result_t, 1>(model_out, fout);
        }
        fin.close();
        fpr.close();
    } else {
        std::cout << "INFO: Unable to open input/predictions file, using default input." << std::endl;

        // hls-fpga-machine-learning insert zero
    input_t model_input[NPARTICLES*4];
    nnet::fill_zero<input_t, NPARTICLES*4>(model_input);
    result_t model_out[1];

        // hls-fpga-machine-learning insert top-level-function
        nPELICAN(model_input,1,model_out);

        // hls-fpga-machine-learning insert output
        nnet::print_result<result_t, 1>(model_out, std::cout, true);

        // hls-fpga-machine-learning insert tb-output
        nnet::print_result<result_t, 1>(model_out, fout);
    }

    fout.close();
    std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;

    return 0;
}
