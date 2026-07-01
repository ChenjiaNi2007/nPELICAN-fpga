"""overlay_deepset.py — overlay the DeepSet curve onto the ALREADY-COMPUTED nanoPELICAN
equivariance curves, without rerunning the firmware sweep.

Why this exists: the raw firmware logits (results/logits_*.dat) are gitignored and may be
absent on a given machine, but the per-(config, beta) AGGREGATES from a real run ARE
committed (results/aggregates_{mode}.csv). Those hold every nanoPELICAN curve. The DeepSet
is a pure-TensorFlow model (no torch/brevitas), so all we need locally is its own logits
(run_sweep_deepset.py) — we compute its aggregate here with the SAME definitions as
compute_metrics.aggregate() and plot it on top of the committed nanoPELICAN aggregates.

Run (after run_sweep_deepset.py has written results/logits_<mode>_deepset.dat):
  python overlay_deepset.py --mode both
Outputs: results/plots_deepset_overlay/<mode>_<metric>.png
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from equiv_common import (  # noqa: E402
    CANON_DIR, RESULTS_DIR, inv_eps_b_at, label_sort_key, mann_whitney_auc,
    safe_name, sigmoid,
)

EPS_S = 0.3
DS_LABEL = "deepset"

# metric column -> (axis label, log-y?)
METRICS = {
    "score_drift_mse": (r"$\langle(\sigma_0-\sigma_\beta)^2\rangle$  (equivariance violation)", True),
    "mean_abs_dsigma": (r"$\langle|\sigma_0-\sigma_\beta|\rangle$", True),
    "mean_abs_dlogit": (r"$\langle|w_0-w_\beta|\rangle$", True),
    "flip_rate":       ("flip rate", False),
    "auc":             ("AUC under boost", False),
    "inv_eps_b":       (r"$1/\epsilon_B$ @ $\epsilon_S=0.3$", False),
}


def read_manifest():
    jet, beta, truth = [], [], []
    with open(os.path.join(CANON_DIR, "manifest.csv")) as f:
        for r in csv.DictReader(f):
            jet.append(int(r["jet_idx"]))
            beta.append(float(r["beta"]))
            truth.append(int(r["truth_label"]))
    return np.array(jet), np.array(beta), np.array(truth)


def deepset_aggregate(mode):
    """Compute the DeepSet per-beta aggregate from its logits, identically to
    compute_metrics.aggregate(). Returns (betas, {metric: [values per beta]})."""
    lp = os.path.join(RESULTS_DIR, f"logits_{mode}_{safe_name(DS_LABEL)}.dat")
    if not os.path.exists(lp):
        sys.exit(f"missing {lp} -- run run_sweep_deepset.py --mode {mode} first.")
    jet, beta, truth = read_manifest()
    logit = np.loadtxt(lp, dtype=np.float64).reshape(-1)
    if len(logit) != len(jet):
        sys.exit(f"{lp} has {len(logit)} logits, manifest has {len(jet)} rows.")

    w0_by_jet = {jet[i]: logit[i] for i in range(len(jet)) if beta[i] == 0.0}
    w0 = np.array([w0_by_jet[j] for j in jet])
    sig, sig0 = sigmoid(logit), sigmoid(w0)
    flip = (np.sign(logit) != np.sign(w0)).astype(int)

    betas = sorted(set(beta.tolist()))
    out = {m: [] for m in METRICS}
    ns = []
    for b in betas:
        sel = beta == b
        ns.append(int(sel.sum()))
        out["score_drift_mse"].append(float(np.mean((sig0[sel] - sig[sel]) ** 2)))
        out["mean_abs_dsigma"].append(float(np.mean(np.abs(sig0[sel] - sig[sel]))))
        out["mean_abs_dlogit"].append(float(np.mean(np.abs(w0[sel] - logit[sel]))))
        out["flip_rate"].append(float(np.mean(flip[sel])))
        out["auc"].append(mann_whitney_auc(truth[sel], logit[sel]))
        out["inv_eps_b"].append(inv_eps_b_at(truth[sel], logit[sel], EPS_S))
    return betas, out, ns


def write_deepset_aggregate_csv(betas, ds, ns):
    """Persist the DeepSet per-beta aggregate in the SAME schema as the committed
    nanoPELICAN aggregates_*.csv, so the DeepSet curve is as portable/re-plottable as the
    PELICAN ones (its raw logits are gitignored). Mode-independent (no beam spurions)."""
    path = os.path.join(RESULTS_DIR, "aggregates_deepset.csv")
    cols = ["config", "beta", "n"] + list(METRICS)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, b in enumerate(betas):
            row = {"config": DS_LABEL, "beta": b, "n": ns[i]}
            row.update({m: ds[m][i] for m in METRICS})
            w.writerow(row)
    print(f"wrote {path}")


def read_aggregates(mode):
    """Read committed aggregates_{mode}.csv -> {config: {metric: {beta: value}}}."""
    path = os.path.join(RESULTS_DIR, f"aggregates_{mode}.csv")
    if not os.path.exists(path):
        sys.exit(f"missing {path} (committed nanoPELICAN aggregates). Nothing to overlay onto.")
    curves = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            cfg, b = r["config"], float(r["beta"])
            d = curves.setdefault(cfg, {m: {} for m in METRICS})
            for m in METRICS:
                try:
                    d[m][b] = float(r[m])
                except (KeyError, ValueError):
                    d[m][b] = float("nan")
    return curves


def finite(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    m = np.isfinite(ys)
    return xs[m], ys[m]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["boostedbeams", "fixedbeams", "both"], default="both")
    a = p.parse_args()
    modes = ["boostedbeams", "fixedbeams"] if a.mode == "both" else [a.mode]

    outdir = os.path.join(RESULTS_DIR, "plots_deepset_overlay")
    os.makedirs(outdir, exist_ok=True)

    wrote_csv = False
    for mode in modes:
        npel = read_aggregates(mode)                    # nanoPELICAN curves (committed)
        ds_betas, ds, ds_ns = deepset_aggregate(mode)   # DeepSet curve (computed here)
        if not wrote_csv:                               # mode-independent -> write once
            write_deepset_aggregate_csv(ds_betas, ds, ds_ns)
            wrote_csv = True
        npel_cfgs = sorted(npel.keys(), key=label_sort_key)

        for metric, (ylabel, logy) in METRICS.items():
            plt.figure(figsize=(6.4, 4.6))
            # nanoPELICAN family: thin grey-to-color lines
            for cfg in npel_cfgs:
                bs = sorted(npel[cfg][metric].keys())
                ys = [npel[cfg][metric][b] for b in bs]
                bx, by = finite(bs, ys)
                if len(bx):
                    plt.plot(bx, by, marker=".", lw=1.1, alpha=0.85, label=f"nPELICAN {cfg}")
            # DeepSet: thick red, on top
            bx, by = finite(ds_betas, ds[metric])
            if len(bx):
                plt.plot(bx, by, marker="o", lw=2.6, color="crimson", zorder=10,
                         label="DeepSet (8-bit)")
            if logy:
                plt.yscale("log")
            plt.xlabel(r"boost magnitude $|\beta|$")
            plt.ylabel(ylabel)
            plt.title(f"{metric} vs boost — DeepSet vs nanoPELICAN ({mode})")
            plt.legend(fontsize=7, ncol=2)
            plt.grid(True, which="both", alpha=0.25)
            plt.tight_layout()
            fn = os.path.join(outdir, f"{mode}_{metric}.png")
            plt.savefig(fn, dpi=140)
            plt.close()
            print(f"wrote {fn}")

    print("\nDone. DeepSet overlaid on the committed nanoPELICAN aggregates.")


if __name__ == "__main__":
    main()
