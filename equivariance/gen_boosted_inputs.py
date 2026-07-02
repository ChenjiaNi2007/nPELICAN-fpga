"""
gen_boosted_inputs.py — build the canonical, build-independent boosted dataset.

Produces ONE physical (float) boosted set, reused across every bit-width so that
all differences between builds are purely a precision effect. Steps (see the task
spec and config.yaml):

  1. Load n_jets test jets via the PELICAN-nano dataloader path (balanced across
     signal/background, seeded). Each event = the RAW 20x4 four-momenta the firmware
     consumes via model_input, PLUS the 2 beam spurions (1,0,0,+/-1) which the firmware
     now takes as a SEPARATE top-level input (beam_input) — Option A.
  2. Sample n_dir isotropic boost directions (seeded, shared across jets/magnitudes),
     and apply Lambda(beta) to the 20 real four-vectors for every (jet, |beta|, dir).
     beta=0 is the identity (the original event), emitted once per jet.
  3. Beams: emit BOTH conventions per row (one gen pass; the momenta are mode-independent
     so run_sweep can sweep both modes from the same canonical set):
       * FIXED  beams = (1,0,0,+/-1) constant  -> equiv_beams_fixedbeams.dat
       * BOOSTED beams = Lambda @ (1,0,0,+/-1)  -> equiv_beams_boostedbeams.dat
     Lambda is the SAME boost applied to the particles. boost_beams in config selects
     which file run_sweep feeds by default; both are always written.
  4. Float64 invariance unit tests (both relative to the term magnitude E_i*E_j, since
     Minkowski dots suffer catastrophic cancellation — a flat absolute 1e-9 is below
     float64 noise at GeV energies; see UNIT_TEST_TOL):
       * real-real 20x20 block — Lorentz scalars, invariant under any boost (always).
       * FULL 22x22 block with BOOSTED beams — the Option-A proof. With the beams
         transformed alongside the particles the entire Gram is invariant; assert
         max relative |d_ij^boost - d_ij| < tol. (Gate G2.) The fixed-beam beam rows
         are NON-invariant by design, so they are NOT gated.
  5. Persist canonical/{equiv_pmu.dat, equiv_nobj.dat, equiv_beams_fixedbeams.dat,
     equiv_beams_boostedbeams.dat, manifest.csv, meta.json}.

Encoding matches scripts/export_golden.py byte-format: one event/line, 80 values
(E px py pz per particle, particle-major) for momenta, 8 values (2 beams) for the
beams files, %.18e. The .dat is FLOAT TEXT — the cast onto each build's input_t
happens in C++ (copy_data), so these files are build-independent and reused verbatim
by run_sweep.py for every bit-width.

Run:  python gen_boosted_inputs.py [--config config.yaml] [--n-jets N] [--n-dir K]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from equiv_common import (  # noqa: E402
    CANON_DIR, NPARTICLES, boost_matrix, load_config, minkowski_gram,
    repo_path, sample_directions,
)

# Boost-invariance unit test tolerance, RELATIVE TO THE TERM MAGNITUDE E_i*E_j.
# The task spec's flat absolute 1e-9 is below float64 noise here: Minkowski dots
# d_ij = E_iE_j - p_i.p_j suffer catastrophic cancellation (both terms ~1e7 at
# large boosts of real GeV energies, while d_ij itself can be tiny for collinear
# massless particles). So the rounding floor scales as eps * E_iE_j, NOT eps * d_ij.
# Normalizing |delta d_ij| by E_i*E_j removes that, leaving ~machine-epsilon noise;
# a genuine boost/metric bug produces O(1) here and is caught with many orders of margin.
UNIT_TEST_TOL = 1e-9

# The two beam spurions in the firmware's convention: (E, px, py, pz) = (1,0,0,+/-1).
# Lightlike (Minkowski norm 0). The firmware adds these as p1[0], p1[1]; here they are
# a top-level input so the harness can boost them (Option A).
BEAMS0 = np.array([[1.0, 0.0, 0.0, 1.0],
                   [1.0, 0.0, 0.0, -1.0]], dtype=np.float64)


def select_jets(testfile: str, n_jets: int, seed: int):
    """Pick n_jets balanced across classes (test.h5 is class-sorted), seeded.

    Returns (sel_idx, Pmu[sel], Nobj[sel], is_signal[sel]) with sel_idx into the file.
    """
    with h5py.File(testfile, "r") as f:
        sig_all = f["is_signal"][:]
        sig_idx = np.where(sig_all == 1)[0]
        bkg_idx = np.where(sig_all == 0)[0]
        rng = np.random.default_rng(seed)
        rng.shuffle(sig_idx)
        rng.shuffle(bkg_idx)
        half = n_jets // 2
        half = min(half, len(sig_idx), len(bkg_idx))
        sel = np.concatenate([sig_idx[:half], bkg_idx[:half]])
        sel.sort()                       # keep file order for reproducible I/O
        Pmu = f["Pmu"][:][sel]           # [n, nconst, 4] float64 GeV (nconst 20 or 200)
        if "Nobj" in f:
            Nobj = f["Nobj"][:][sel].astype(int)
        else:
            # toptag-style files have no Nobj key: real particles are the nonzero rows.
            Nobj = (np.abs(Pmu[:, :, 0]) > 0).sum(axis=1).astype(int)
        sig = sig_all[sel].astype(int)
    if Pmu.shape[1] > NPARTICLES:
        # 200-wide toptag Pmu: keep the leading-NPARTICLES by pT, BEFORE boosting —
        # the jet the firmware sees is defined in the lab frame, and invariance is
        # then measured on that same particle set. Matches PELICAN-nano's --nobj cap
        # and the DeepSet adapter's _select_leading_pt.
        pt2 = Pmu[:, :, 1] ** 2 + Pmu[:, :, 2] ** 2
        order = np.argsort(-pt2, axis=1)[:, :NPARTICLES]
        Pmu = np.take_along_axis(Pmu, order[:, :, None], axis=1)
        Nobj = np.minimum(Nobj, NPARTICLES)
    return sel, Pmu, Nobj, sig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--n-jets", type=int, default=None, help="override config n_jets (smoke test)")
    p.add_argument("--n-dir", type=int, default=None, help="override config n_dir (smoke test)")
    a = p.parse_args()

    cfg = load_config(a.config)
    n_jets = a.n_jets if a.n_jets is not None else int(cfg["n_jets"])
    n_dir = a.n_dir if a.n_dir is not None else int(cfg["n_dir"])
    seed = int(cfg["seed"])
    beta_grid = [float(b) for b in cfg["beta_grid"]]
    boost_beams = bool(cfg.get("boost_beams", False))
    testfile = repo_path(cfg["data_file"])

    # Option A: the firmware takes the beams as a top-level input, so boost_beams is a
    # real toggle. We ALWAYS emit both beams files (fixed and boosted) in one pass; the
    # config flag only records which run_sweep feeds by default. So nothing to abort on.
    print(f"[gen] boost_beams (default sweep mode) = {boost_beams}")
    print(f"[gen] testfile = {testfile}")
    print(f"[gen] n_jets={n_jets}  n_dir={n_dir}  seed={seed}")
    print(f"[gen] beta_grid = {beta_grid}")

    sel, Pmu, Nobj, sig = select_jets(testfile, n_jets, seed)
    n = len(sel)
    print(f"[gen] selected {n} jets ({int((sig==1).sum())} signal / {int((sig==0).sum())} background)")

    # Boost directions: sampled once, shared across all jets and magnitudes.
    rng = np.random.default_rng(seed + 1)
    dirs = sample_directions(n_dir, rng)               # [n_dir, 3]

    # Precompute boost matrices: identity for beta=0, n_dir matrices for beta>0.
    nonzero_betas = [b for b in beta_grid if b > 0.0]
    have_zero = any(b == 0.0 for b in beta_grid)
    L_by_beta = {b: [boost_matrix(b * dirs[d]) for d in range(n_dir)] for b in nonzero_betas}

    os.makedirs(CANON_DIR, exist_ok=True)
    pmu_path = os.path.join(CANON_DIR, "equiv_pmu.dat")
    nobj_path = os.path.join(CANON_DIR, "equiv_nobj.dat")
    beams_fixed_path = os.path.join(CANON_DIR, "equiv_beams_fixedbeams.dat")
    beams_boost_path = os.path.join(CANON_DIR, "equiv_beams_boostedbeams.dat")
    manifest_path = os.path.join(CANON_DIR, "manifest.csv")
    meta_path = os.path.join(CANON_DIR, "meta.json")

    max_unit_err = 0.0       # max ABSOLUTE |d^boost - d|, real-real
    max_unit_rel = 0.0       # max RELATIVE error, real-real (gate)
    max_full_err = 0.0       # max ABSOLUTE, full 22x22 boosted-beams
    max_full_rel = 0.0       # max RELATIVE, full 22x22 boosted-beams (gate G2)
    n_rows = 0

    with open(pmu_path, "w") as fpmu, open(nobj_path, "w") as fnobj, \
         open(beams_fixed_path, "w") as fbf, open(beams_boost_path, "w") as fbb, \
         open(manifest_path, "w", newline="") as fman:
        man = csv.writer(fman)
        man.writerow(["row_idx", "jet_idx", "beta", "dir_idx", "truth_label", "nobj"])

        def emit(pmu20: np.ndarray, beams_fixed: np.ndarray, beams_boost: np.ndarray,
                 jet_idx: int, beta: float, dir_idx: int, truth: int, nobj: int):
            nonlocal n_rows
            vals = pmu20.reshape(-1)                    # 80, particle-major E px py pz
            fpmu.write(" ".join(f"{v:.18e}" for v in vals) + "\n")
            fnobj.write(f"{int(nobj)}\n")
            fbf.write(" ".join(f"{v:.18e}" for v in beams_fixed.reshape(-1)) + "\n")
            fbb.write(" ".join(f"{v:.18e}" for v in beams_boost.reshape(-1)) + "\n")
            man.writerow([n_rows, int(jet_idx), f"{beta:.4g}", int(dir_idx),
                          int(truth), int(nobj)])
            n_rows += 1

        # full-22x22 reference Gram (real particles + FIXED beams), per jet.
        def full_gram(real_pmu: np.ndarray, beams: np.ndarray) -> np.ndarray:
            return minkowski_gram(np.vstack([real_pmu, beams]))

        for j in range(n):
            pmu0 = Pmu[j].astype(np.float64)            # [20,4]
            nobj_j = int(Nobj[j])
            truth_j = int(sig[j])
            real0 = pmu0[:nobj_j]                       # real (non-padded) particles
            gram0 = minkowski_gram(real0)              # [nobj, nobj] reference dots
            full0 = full_gram(real0, BEAMS0)           # [(nobj+2)^2] reference (with beams)

            if have_zero:
                # beta=0: identity; both beam conventions equal (1,0,0,+/-1).
                emit(pmu0, BEAMS0, BEAMS0, sel[j], 0.0, 0, truth_j, nobj_j)

            for beta in nonzero_betas:
                for d in range(n_dir):
                    L = L_by_beta[beta][d]
                    boosted = (L @ pmu0.T).T            # [20,4] boosted particles
                    beams_b = (L @ BEAMS0.T).T          # [2,4]  boosted beams
                    # (1) real-real block (Lorentz scalars; must be invariant)
                    gramb = minkowski_gram(boosted[:nobj_j])
                    if nobj_j > 0:
                        absd = np.abs(gramb - gram0)
                        Eb = np.abs(boosted[:nobj_j, 0])           # boosted energies
                        denom = np.maximum(np.outer(Eb, Eb), 1.0)  # term magnitude E_i*E_j
                        max_unit_err = max(max_unit_err, float(np.max(absd)))
                        max_unit_rel = max(max_unit_rel, float(np.max(absd / denom)))
                    # (2) FULL 22x22 block with BOOSTED beams (Option-A invariance, G2)
                    fullb = full_gram(boosted[:nobj_j], beams_b)
                    absf = np.abs(fullb - full0)
                    Efull = np.abs(np.concatenate([boosted[:nobj_j, 0], beams_b[:, 0]]))
                    denomf = np.maximum(np.outer(Efull, Efull), 1.0)
                    max_full_err = max(max_full_err, float(np.max(absf)))
                    max_full_rel = max(max_full_rel, float(np.max(absf / denomf)))
                    emit(boosted, BEAMS0, beams_b, sel[j], beta, d, truth_j, nobj_j)

    print(f"[gen] float64 invariance — real-real 20x20 block: "
          f"max abs = {max_unit_err:.3e}, max REL = {max_unit_rel:.3e} (tol {UNIT_TEST_TOL:.0e})")
    print(f"[gen] float64 invariance — FULL 22x22 w/ BOOSTED beams (G2): "
          f"max abs = {max_full_err:.3e}, max REL = {max_full_rel:.3e} (tol {UNIT_TEST_TOL:.0e})")
    if max_unit_rel >= UNIT_TEST_TOL:
        sys.exit(f"[gen] FAIL: boost not invariant on the real-real dots "
                 f"(relative {max_unit_rel:.3e} >= {UNIT_TEST_TOL:.0e}). Aborting.")
    if max_full_rel >= UNIT_TEST_TOL:
        sys.exit(f"[gen] FAIL (G2): boosted-beam full 22x22 Gram not invariant "
                 f"(relative {max_full_rel:.3e} >= {UNIT_TEST_TOL:.0e}). Aborting.")
    print("[gen] unit tests PASS (real-real AND full-22x22 boosted-beam invariance)")

    meta = {
        "n_jets_selected": n, "n_dir": n_dir, "seed": seed,
        "beta_grid": beta_grid, "boost_beams": boost_beams,
        "n_rows": n_rows, "directions": dirs.tolist(),
        "selected_file_indices": sel.tolist(),
        "max_full22_rel_invariance_err": max_full_rel,
        "encoding": "pmu: 80 floats (E px py pz per particle, particle-major); "
                    "beams files: 8 floats (2 beams E px py pz); %.18e",
        "note": "build-independent float dataset; reused verbatim across all bit-widths. "
                "Two beams files (fixed/boosted) written; momenta identical for both modes.",
    }
    with open(meta_path, "w") as fm:
        json.dump(meta, fm, indent=2)

    print(f"[gen] wrote {n_rows} rows")
    print(f"[gen]   {pmu_path}")
    print(f"[gen]   {nobj_path}")
    print(f"[gen]   {beams_fixed_path}")
    print(f"[gen]   {beams_boost_path}")
    print(f"[gen]   {manifest_path}")
    print(f"[gen]   {meta_path}")


if __name__ == "__main__":
    main()
