"""run_sweep_deepset.py — DeepSet curve for the equivariance sweep (Phase D).

The DeepSet is NOT the firmware oracle, so it needs its own evaluator. It reuses the
SAME canonical boosted 4-momenta every firmware curve uses (canonical/equiv_pmu.dat,
build-independent text), runs the trained binary DeepSet from Phase C, and emits one
signal logit per manifest row — exactly the contract compute_metrics.py expects:

    results/logits_<mode>_<safe_name('deepset')>.dat   (one value per manifest row)

The point of the comparison: the boost acts on the 4-momenta; the DeepSet's jet-relative
(pT, eta_rel, phi_rel) features are recomputed from the *boosted* constituents (jet axis
recomputed per row), with the TRAIN-fit normalisation reused. Because the DeepSet has no
Lorentz symmetry, its score drifts with |beta| — the headline contrast against nPELICAN.

DeepSet has NO beam spurions, so its output is mode-independent: the same logits are
written for both 'boostedbeams' and 'fixedbeams' (draw the one curve on both overlays).

Run in the l1-jet-id environment (needs tensorflow/keras + fast_jetclass):
  conda activate fast_jetclass
  python run_sweep_deepset.py \
      --l1-repo   ../../l1-jet-id \
      --model-dir ../../l1-jet-id/scripts/trained_deepsets/deepsets_8bit_20const_toptag/kfolding1 \
      --norm-pkl  ../../l1-jet-id/scripts/data/jetid_toptag/processed/normparams_robust_20const_ptetaphi.pkl \
      --nconst 20 --norm robust --mode both
Then:  python compute_metrics.py --mode both   (redraws every curve incl. deepset)
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from equiv_common import CANON_DIR, RESULTS_DIR, safe_name  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--l1-repo", required=True,
                   help="Path to the l1-jet-id repo (added to sys.path for fast_jetclass).")
    p.add_argument("--model-dir", required=True,
                   help="Trained DeepSet kfold dir (keras SavedModel) from Phase C.")
    p.add_argument("--norm-pkl", required=True,
                   help="Train-fit normalisation pkl (root/processed/normparams_...).")
    p.add_argument("--nconst", type=int, default=20)
    p.add_argument("--norm", default="robust")
    p.add_argument("--label", default="deepset")
    p.add_argument("--mode", choices=["boostedbeams", "fixedbeams", "both"],
                   default="both")
    a = p.parse_args()

    sys.path.insert(0, os.path.abspath(a.l1_repo))
    # The same feature conversion used to TRAIN the DeepSet (Phase C, C1).
    from fast_jetclass.data.toptag_data import fourmom_to_ptetaphirel
    from fast_jetclass.data import standardization

    # 1) canonical boosted 4-momenta: 80 floats/row, particle-major (E,px,py,pz).
    pmu_path = os.path.join(CANON_DIR, "equiv_pmu.dat")
    print(f"loading {pmu_path} (large text file, may take ~1 min)...")
    pmu = np.loadtxt(pmu_path, dtype=np.float64).reshape(-1, 20, 4)
    n_rows = pmu.shape[0]
    print(f"  {n_rows} rows x (20, 4)")

    # 2) jet-relative features recomputed from the BOOSTED constituents (the whole point).
    x = fourmom_to_ptetaphirel(pmu)                      # (n_rows, 20, 3)
    if a.nconst <= x.shape[1]:
        x = x[:, : a.nconst, :]
    else:
        x = np.pad(x, ((0, 0), (0, a.nconst - x.shape[1]), (0, 0)))

    # 3) reuse the TRAIN normalisation (do NOT refit).
    with open(a.norm_pkl, "rb") as f:
        norm_params = pickle.load(f)
    x = standardization.apply_standardisation(a.norm, x, norm_params)

    # 4) trained binary DeepSet -> p_top -> single logit (matches nPELICAN convention).
    from tensorflow import keras
    import tensorflow as tf
    model = keras.models.load_model(a.model_dir, compile=False)
    pred = model.predict(x, batch_size=4096, verbose=1)
    if isinstance(model.layers[-1], keras.layers.Dense):
        pred = tf.nn.softmax(pred).numpy()               # last layer is bare Dense
    p_top = np.clip(pred[:, 1].astype(np.float64), 1e-7, 1 - 1e-7)
    logit = np.log(p_top / (1.0 - p_top))                # sigmoid(logit) == p_top

    # 5) write one file per requested mode (identical — DeepSet is mode-independent).
    os.makedirs(RESULTS_DIR, exist_ok=True)
    modes = ["boostedbeams", "fixedbeams"] if a.mode == "both" else [a.mode]
    for mode in modes:
        out = os.path.join(RESULTS_DIR, f"logits_{mode}_{safe_name(a.label)}.dat")
        np.savetxt(out, logit, fmt="%.17g")
        print(f"wrote {len(logit)} logits -> {out}")
    print("\nDone. Next: python compute_metrics.py --mode both")


if __name__ == "__main__":
    main()
