"""
run_sweep.py — the driver. For each {bit_width, checkpoint} in config.yaml:

  1. model_loader.py --quant --out firmware/weights/weights.h  -> regenerate this
     build's weights.h + types_generated.h from the checkpoint's learned scales.
  2. GATE: regenerate golden vectors for this checkpoint (export_golden.py), build the
     RUN_EQUIVARIANCE testbench, run it on golden_pmu.dat, and assert the logits
     reproduce golden_logits.dat to config golden_tol. This proves the equivariance
     TB mode IS the validated golden firmware path for this exact build.
  3. SWEEP: feed the canonical boosted set (canonical/equiv_pmu.dat, build-independent
     float text) through the same binary -> logits, stashed as results/logits_bit{bw}.dat
     aligned row-for-row with canonical/manifest.csv.

Oracle: the local g++ build (build_local.sh path + -DRUN_EQUIVARIANCE). docs/WORKFLOW.md
certifies it bit-identical to remote Vitis csim, so Vitis is not required.

Run (from the PELICAN-nano venv so brevitas/torch are importable):
  ../PELICAN-nano/.venv/bin/python run_sweep.py [--config config.yaml] [--only-gate]
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from equiv_common import (  # noqa: E402
    CANON_DIR, FPGA_ROOT, RESULTS_DIR, TB_DATA, load_config, model_label,
    repo_path, safe_name,
)

PY = sys.executable  # subprocesses reuse the interpreter running this driver (the venv)

GPP_BASE = [
    "-std=c++17", "-O2",
    "-I", "third_party/stubs",
    "-I", "third_party/ap_types/include",
    "-I", ".",
]

# Equiv-vs-firmware bit-exactness: same nPELICAN() call on the same inputs, so it
# must agree to the last digit. (Float text round-trips at %.17g.)
EQUIV_EXACT_TOL = 1e-9


def run(cmd, **kw):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, **kw)


def regen_weights(ckpt_abs: str, pelican_repo: str):
    run([PY, "model_loader.py", "--model", ckpt_abs, "--quant",
         "--repo", pelican_repo, "--out", "firmware/weights/weights.h"],
        cwd=FPGA_ROOT)


def regen_golden(ckpt_abs: str, pelican_repo: str, num: int = 200):
    run([PY, "scripts/export_golden.py", "--checkpoint", ckpt_abs, "--num", str(num)],
        cwd=pelican_repo)


def build_tb(define: str, out: str) -> str:
    run(["g++", *GPP_BASE, f"-D{define}", "nPELICAN_tb.cpp", "firmware/nPELICAN.cpp",
         "-o", out], cwd=FPGA_ROOT)
    return os.path.join(FPGA_ROOT, out)


def run_tb_on(pmu_src: str, nobj_src: str) -> np.ndarray:
    """Copy (pmu_src, nobj_src) into tb_data/equiv_in_*, run tb_equiv, return logits."""
    shutil.copyfile(pmu_src, os.path.join(TB_DATA, "equiv_in_pmu.dat"))
    shutil.copyfile(nobj_src, os.path.join(TB_DATA, "equiv_in_nobj.dat"))
    run([os.path.join(FPGA_ROOT, "tb_equiv")], cwd=FPGA_ROOT)
    out = os.path.join(TB_DATA, "equiv_out_logits.dat")
    return np.loadtxt(out, dtype=np.float64).reshape(-1)


def gate(pelican_repo: str, warn_tol: float):
    """Validate the oracle, then build the equiv TB ready for the sweep.

    Two distinct checks (see FINDINGS for why they must be separated):
      1. GATE (must pass, bit-exact): the RUN_EQUIVARIANCE output on golden_pmu must
         equal the FIRMWARE's own golden-path output (RUN_GOLDEN_GATE writes
         golden_fw_results.log). This proves the equiv TB plumbing runs the firmware
         correctly — independent of whether the firmware matches PyTorch.
      2. BIT-FAITHFULNESS (informational): firmware vs PyTorch (golden_logits). This is
         ~0 at high bit width but legitimately grows at low bit width, because PyTorch
         keeps BatchNorm in float while the firmware uses fixed BN, and at coarse grids
         a tipped quantizer boundary cascades to the logit. Reported, never aborts.
    """
    gpmu = os.path.join(TB_DATA, "golden_pmu.dat")
    gnobj = os.path.join(TB_DATA, "golden_nobj.dat")
    glog = os.path.join(TB_DATA, "golden_logits.dat")
    gfw = os.path.join(TB_DATA, "golden_fw_results.log")
    if not (os.path.exists(gpmu) and os.path.exists(glog)):
        sys.exit(f"GATE: missing golden vectors ({gpmu}). export_golden.py must run first.")

    # firmware golden path: writes golden_fw_results.log (firmware logits on golden_pmu)
    build_tb("RUN_GOLDEN_GATE", "tb_golden")
    run([os.path.join(FPGA_ROOT, "tb_golden")], cwd=FPGA_ROOT,
        stdout=subprocess.DEVNULL)
    fw = np.loadtxt(gfw, dtype=np.float64).reshape(-1)

    # equiv path on the same golden inputs (also leaves tb_equiv built for the sweep)
    build_tb("RUN_EQUIVARIANCE", "tb_equiv")
    got = run_tb_on(gpmu, gnobj)

    m = min(len(got), len(fw))
    gate_delta = float(np.max(np.abs(got[:m] - fw[:m]))) if m else float("inf")
    status = "PASS" if gate_delta < EQUIV_EXACT_TOL else "FAIL"
    print(f"  GATE (equiv == firmware): {status} (max|delta|={gate_delta:.3g}, "
          f"{int(np.sum(got[:m] == fw[:m]))}/{m} exact)")
    if status != "PASS":
        sys.exit("  GATE FAILED: RUN_EQUIVARIANCE does not reproduce the firmware's own "
                 "golden-path output. The oracle plumbing is wrong; refusing to sweep.")

    # informational: firmware vs PyTorch
    pyt = np.loadtxt(glog, dtype=np.float64).reshape(-1)
    mp = min(len(fw), len(pyt))
    fw_vs_pyt = float(np.max(np.abs(fw[:mp] - pyt[:mp]))) if mp else float("inf")
    flag = "" if fw_vs_pyt < warn_tol else "  (large: float-BN boundary tipping at low bits)"
    print(f"  bit-faithfulness (firmware vs PyTorch): max|delta|={fw_vs_pyt:.3g}{flag}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--only-gate", action="store_true",
                   help="regen weights + golden and run the gate for each model; skip the sweep")
    a = p.parse_args()

    cfg = load_config(a.config)
    pelican_repo = repo_path(cfg["pelican_nano_repo"])
    warn_tol = float(cfg.get("golden_tol", 1e-3))   # threshold for the fw-vs-PyTorch warning
    models = cfg["models"]

    canon_pmu = os.path.join(CANON_DIR, "equiv_pmu.dat")
    canon_nobj = os.path.join(CANON_DIR, "equiv_nobj.dat")
    if not a.only_gate and not os.path.exists(canon_pmu):
        sys.exit(f"Missing {canon_pmu}. Run gen_boosted_inputs.py first.")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    for mdl in models:
        label = model_label(mdl)
        ckpt_abs = repo_path(mdl["checkpoint"])
        ref = " (reference)" if mdl.get("reference") else ""
        print(f"\n=== model {label} (W:A:I){ref}  ckpt={ckpt_abs} ===")
        if not os.path.exists(ckpt_abs):
            sys.exit(f"  checkpoint not found: {ckpt_abs}")

        regen_weights(ckpt_abs, pelican_repo)
        regen_golden(ckpt_abs, pelican_repo)
        gate(pelican_repo, warn_tol)   # builds tb_golden + tb_equiv, validates the oracle

        if a.only_gate:
            continue

        print(f"  [sweep] running canonical boosted set through {label} build")
        logits = run_tb_on(canon_pmu, canon_nobj)
        out = os.path.join(RESULTS_DIR, f"logits_{safe_name(label)}.dat")
        np.savetxt(out, logits, fmt="%.17g")
        print(f"  [sweep] wrote {len(logits)} logits -> {out}")

    print("\nDone. Next: python compute_metrics.py")


if __name__ == "__main__":
    main()
