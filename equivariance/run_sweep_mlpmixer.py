"""run_sweep_mlpmixer.py — MLP-Mixer curve for the equivariance sweep.

Mirrors run_sweep_deepset.py: reuses the SAME canonical boosted 4-momenta
(canonical/equiv_pmu.dat) every firmware curve uses, runs the trained MLP-Mixer
(arXiv:2503.03103 port, workspace mlpmixer/), and emits one signal logit per manifest
row — the contract compute_metrics.py / overlay_external.py expect:

    results/logits_<mode>_mlpmixer.dat   (one value per manifest row)

Feature treatment matches training (mlpmixer/common.py): leading-20-by-pT selection
FIRST — which also RE-SORTS by the *boosted* pT, as a real trigger would; the MLP-Mixer
is order-sensitive, so this matters — then jet-relative (pT, eta_rel, phi_rel) from the
boosted constituents, then the TRAIN-fit normalization (never refit).

Like the DeepSet, the MLP-Mixer has no beam spurions, so its output is mode-independent:
identical logits are written for 'boostedbeams' and 'fixedbeams'.

Run in the PELICAN-nano-hgq2 venv (keras 3 + JAX):
  KERAS_BACKEND=jax ../../PELICAN-nano-hgq2/.venv/bin/python run_sweep_mlpmixer.py
Then: overlay_external.py (PELICAN-nano venv, needs matplotlib) redraws the overlays.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from equiv_common import CANON_DIR, RESULTS_DIR, safe_name  # noqa: E402

WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MIXER_DIR = os.path.join(WORKSPACE, "mlpmixer")
sys.path.insert(0, MIXER_DIR)
import common  # noqa: E402  (mlpmixer/common.py)
from model import get_model  # noqa: E402  (mlpmixer/model.py)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=os.path.join(MIXER_DIR, "model", "mixer_best.weights.h5"))
    p.add_argument("--norm", default=common.NORM_FILE,
                   help="norm npz matching the weights (e.g. model/norm_aug.npz for the boost-augmented model)")
    p.add_argument("--label", default="mlpmixer")
    p.add_argument("--mode", choices=["boostedbeams", "fixedbeams", "both"], default="both")
    a = p.parse_args()

    pmu_path = os.path.join(CANON_DIR, "equiv_pmu.dat")
    print(f"loading {pmu_path} (large text file, may take ~1 min)...")
    # Preallocated line-by-line parse: np.loadtxt's intermediate lists blow past
    # this machine's RAM once the canonical set grows beyond the 2000-jet original.
    with open(pmu_path) as f:
        n_rows = sum(1 for _ in f)
    pmu = np.empty((n_rows, 20, 4), dtype=np.float64)
    with open(pmu_path) as f:
        for i, line in enumerate(f):
            pmu[i] = np.array(line.split(), dtype=np.float64).reshape(20, 4)
    print(f"  {pmu.shape[0]} rows x (20, 4)")

    # leading-pT re-sort on the BOOSTED constituents + jet-relative features + train norm
    x = common.pmu_to_features(pmu)
    x = common.apply_norm(x, common.load_norm(a.norm))

    model = get_model(common.NCONST, common.NFEAT)
    model.load_weights(a.weights)
    logit = model.predict(x, batch_size=8192, verbose=1).ravel().astype(np.float64)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    modes = ["boostedbeams", "fixedbeams"] if a.mode == "both" else [a.mode]
    for mode in modes:
        out = os.path.join(RESULTS_DIR, f"logits_{mode}_{safe_name(a.label)}.dat")
        np.savetxt(out, logit, fmt="%.17g")
        print(f"wrote {len(logit)} logits -> {out}")
    print("\nDone. Next: overlay_external.py --mode both")


if __name__ == "__main__":
    main()
