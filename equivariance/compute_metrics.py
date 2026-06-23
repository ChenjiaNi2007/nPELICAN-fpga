"""
compute_metrics.py — pair, aggregate, plot.

Reads canonical/manifest.csv and results/logits_bit{bw}.dat (one per model), pairs
each boosted event with the SAME jet's original at the SAME bit-width (f_b(Lambda x)
vs f_b(x)), and emits:

  results/results.csv      tidy per-(bit_width, beta, dir, jet) logit + sigma + paired w0
  results/aggregates.csv   per-(bit_width, beta) metrics
  results/plots/*.png      the standard SEAL-style curves

All plots/aggregates are re-derivable from results.csv WITHOUT re-running csim.

Metrics (per bit-width b, aggregated over jets & directions within each |beta| bin):
  * score_drift_mse   = <(sigma(w0) - sigma(w_beta))^2>     (headline; SEAL Fig.2)
  * mean_abs_dsigma   = <|sigma(w0) - sigma(w_beta)|>
  * flip_rate         = fraction with sign(w0) != sign(w_beta)
  * auc, inv_eps_b@0.3 of {w_beta vs truth}  (discrimination under boost)
The reference (highest-bit) model is drawn on every plot as the invariance floor.

Run:  python compute_metrics.py [--config config.yaml]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from equiv_common import (  # noqa: E402
    CANON_DIR, RESULTS_DIR, inv_eps_b_at, label_sort_key, load_config,
    mann_whitney_auc, model_label, safe_name, sigmoid,
)

EPS_S = 0.3


def read_manifest():
    path = os.path.join(CANON_DIR, "manifest.csv")
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "row_idx": int(r["row_idx"]), "jet_idx": int(r["jet_idx"]),
                "beta": float(r["beta"]), "dir_idx": int(r["dir_idx"]),
                "truth": int(r["truth_label"]), "nobj": int(r["nobj"]),
            })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    a = p.parse_args()
    cfg = load_config(a.config)

    man = read_manifest()
    n_rows = len(man)
    jet = np.array([m["jet_idx"] for m in man])
    beta = np.array([m["beta"] for m in man])
    dir_idx = np.array([m["dir_idx"] for m in man])
    truth = np.array([m["truth"] for m in man])

    models = cfg["models"]
    ref_label = next((model_label(m) for m in models if m.get("reference")), None)

    plot_dir = os.path.join(RESULTS_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # ---- build tidy results.csv across all available builds ----
    results_rows = []     # dicts
    per_bw = {}           # label -> dict of arrays for plotting

    for mdl in models:
        bw = model_label(mdl)
        logit_path = os.path.join(RESULTS_DIR, f"logits_{safe_name(bw)}.dat")
        if not os.path.exists(logit_path):
            print(f"[metrics] skip {bw}: no {logit_path} (not swept yet)")
            continue
        logit = np.loadtxt(logit_path, dtype=np.float64).reshape(-1)
        if len(logit) != n_rows:
            sys.exit(f"[metrics] {logit_path} has {len(logit)} logits, manifest has {n_rows}")

        # pair each row with the same jet's beta=0 logit (this bit-width)
        w0_by_jet = {}
        for i in range(n_rows):
            if beta[i] == 0.0:
                w0_by_jet[jet[i]] = logit[i]
        w0 = np.array([w0_by_jet[j] for j in jet])

        sig = sigmoid(logit)
        sig0 = sigmoid(w0)
        flip = (np.sign(logit) != np.sign(w0)).astype(int)

        for i in range(n_rows):
            results_rows.append({
                "config": bw, "jet_idx": int(jet[i]), "beta": float(beta[i]),
                "dir_idx": int(dir_idx[i]), "truth": int(truth[i]),
                "logit": float(logit[i]), "w0": float(w0[i]),
                "sigma": float(sig[i]), "sigma0": float(sig0[i]),
                "flip": int(flip[i]),
            })

        per_bw[bw] = {"logit": logit, "w0": w0, "sig": sig, "sig0": sig0, "flip": flip}

    if not per_bw:
        sys.exit("[metrics] no swept logits found. Run run_sweep.py first.")

    results_csv = os.path.join(RESULTS_DIR, "results.csv")
    with open(results_csv, "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=["config", "jet_idx", "beta", "dir_idx",
                                             "truth", "logit", "w0", "sigma", "sigma0", "flip"])
        wcsv.writeheader()
        wcsv.writerows(results_rows)
    print(f"[metrics] wrote {results_csv} ({len(results_rows)} rows)")

    # ---- aggregates per (config, beta) ----
    betas = sorted(set(beta.tolist()))
    agg_rows = []
    for bw, d in sorted(per_bw.items(), key=lambda kv: label_sort_key(kv[0])):
        for b in betas:
            sel = beta == b
            s, s0 = d["sig"][sel], d["sig0"][sel]
            drift = float(np.mean((s0 - s) ** 2))
            mad = float(np.mean(np.abs(s0 - s)))
            # logit-space drift: independent of the sigmoid operating point, so it
            # is NOT suppressed when a degraded low-bit model parks its outputs in the
            # flat tail of sigma. Reading score_drift (Delta-sigma) and logit_drift
            # (Delta-w) together separates "more invariant" from "less responsive".
            dlogit = float(np.mean(np.abs(d["w0"][sel] - d["logit"][sel])))
            flip_rate = float(np.mean(d["flip"][sel]))
            auc = mann_whitney_auc(truth[sel], d["logit"][sel])
            iepsb = inv_eps_b_at(truth[sel], d["logit"][sel], EPS_S)
            agg_rows.append({
                "config": bw, "beta": b, "n": int(sel.sum()),
                "score_drift_mse": drift, "mean_abs_dsigma": mad,
                "mean_abs_dlogit": dlogit,
                "flip_rate": flip_rate, "auc": auc, "inv_eps_b": iepsb,
            })

    agg_csv = os.path.join(RESULTS_DIR, "aggregates.csv")
    with open(agg_csv, "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=["config", "beta", "n", "score_drift_mse",
                                             "mean_abs_dsigma", "mean_abs_dlogit",
                                             "flip_rate", "auc", "inv_eps_b"])
        wcsv.writeheader()
        wcsv.writerows(agg_rows)
    print(f"[metrics] wrote {agg_csv}")

    # ---- plots ----
    def series(bw, key):
        xs = [r["beta"] for r in agg_rows if r["config"] == bw]
        ys = [r[key] for r in agg_rows if r["config"] == bw]
        return np.array(xs), np.array(ys)

    # bit-budget-ordered colour gradient (low precision -> high precision)
    bws = sorted(per_bw.keys(), key=label_sort_key)
    nonref = [b for b in bws if b != ref_label]
    cmap = plt.cm.viridis(np.linspace(0.05, 0.85, max(len(nonref), 1)))
    colour = {b: cmap[i] for i, b in enumerate(nonref)}

    def style(bw):
        if bw == ref_label:
            return dict(color="k", ls="--", marker="o", lw=2.2,
                        label=f"{bw} (reference)", zorder=10)
        return dict(color=colour[bw], marker="o", lw=1.6, label=f"{bw}")

    # 1. score drift (log-y) — the headline equivariance-violation curve
    plt.figure(figsize=(7, 5))
    for bw in bws:
        x, y = series(bw, "score_drift_mse")
        y = np.where(y <= 0, np.nan, y)            # log-y: drop the trivial beta=0 zero
        plt.plot(x, y, **style(bw))
    plt.yscale("log"); plt.xlabel(r"boost magnitude $|\beta|$")
    plt.ylabel(r"$\langle(\sigma(w_0)-\sigma(w_\beta))^2\rangle$")
    plt.title("Score drift vs boost (equivariance violation)")
    plt.grid(True, which="both", alpha=0.3); plt.legend(title="W:A:I bits", fontsize=8, ncol=2)
    plt.tight_layout(); plt.savefig(os.path.join(plot_dir, "score_drift.png"), dpi=130); plt.close()

    # 2. mean |delta sigma|
    plt.figure(figsize=(7, 5))
    for bw in bws:
        x, y = series(bw, "mean_abs_dsigma"); plt.plot(x, y, **style(bw))
    plt.xlabel(r"boost magnitude $|\beta|$"); plt.ylabel(r"$\langle|\Delta\sigma|\rangle$")
    plt.title("Mean score shift vs boost"); plt.grid(True, alpha=0.3); plt.legend(title="W:A:I bits", fontsize=8, ncol=2)
    plt.tight_layout(); plt.savefig(os.path.join(plot_dir, "mean_abs_dsigma.png"), dpi=130); plt.close()

    # 2b. logit-space drift (operating-point independent) — log-y
    plt.figure(figsize=(7, 5))
    for bw in bws:
        x, y = series(bw, "mean_abs_dlogit")
        y = np.where(y <= 0, np.nan, y)
        plt.plot(x, y, **style(bw))
    plt.yscale("log"); plt.xlabel(r"boost magnitude $|\beta|$")
    plt.ylabel(r"$\langle|w_0 - w_\beta|\rangle$")
    plt.title("Logit drift vs boost (operating-point independent)")
    plt.grid(True, which="both", alpha=0.3); plt.legend(title="W:A:I bits", fontsize=8, ncol=2)
    plt.tight_layout(); plt.savefig(os.path.join(plot_dir, "logit_drift.png"), dpi=130); plt.close()

    # 3. decision-flip rate
    plt.figure(figsize=(7, 5))
    for bw in bws:
        x, y = series(bw, "flip_rate"); plt.plot(x, y, **style(bw))
    plt.xlabel(r"boost magnitude $|\beta|$"); plt.ylabel("decision-flip rate")
    plt.title(r"Fraction of jets crossing $w=0$ under boost")
    plt.grid(True, alpha=0.3); plt.legend(title="W:A:I bits", fontsize=8, ncol=2)
    plt.tight_layout(); plt.savefig(os.path.join(plot_dir, "flip_rate.png"), dpi=130); plt.close()

    # 4. AUC vs boost
    plt.figure(figsize=(7, 5))
    for bw in bws:
        x, y = series(bw, "auc"); plt.plot(x, y, **style(bw))
    plt.xlabel(r"boost magnitude $|\beta|$"); plt.ylabel("AUC (w_beta vs truth)")
    plt.title("Discrimination under boost (AUC)"); plt.grid(True, alpha=0.3); plt.legend(title="W:A:I bits", fontsize=8, ncol=2)
    plt.tight_layout(); plt.savefig(os.path.join(plot_dir, "auc.png"), dpi=130); plt.close()

    # 5. 1/eps_B @ eps_S=0.3 vs boost
    plt.figure(figsize=(7, 5))
    for bw in bws:
        x, y = series(bw, "inv_eps_b")
        y = np.array([np.nan if not np.isfinite(v) else v for v in y])
        plt.plot(x, y, **style(bw))
    plt.xlabel(r"boost magnitude $|\beta|$"); plt.ylabel(r"$1/\epsilon_B$ at $\epsilon_S=0.3$")
    plt.title(r"Background rejection under boost"); plt.grid(True, alpha=0.3); plt.legend(title="W:A:I bits", fontsize=8, ncol=2)
    plt.tight_layout(); plt.savefig(os.path.join(plot_dir, "inv_eps_b.png"), dpi=130); plt.close()

    print(f"[metrics] wrote plots -> {plot_dir}/")
    # console summary (config = W:A:I bits)
    print("\nconfig(W:A:I)  beta   drift_mse   flip_rate   auc      1/epsB@0.3")
    for r in agg_rows:
        ie = r["inv_eps_b"]
        ie_s = "  inf" if not np.isfinite(ie) else f"{ie:7.2f}"
        print(f"  {r['config']:>10}  {r['beta']:.2f}  {r['score_drift_mse']:.3e}  "
              f"{r['flip_rate']:.4f}   {r['auc']:.4f}  {ie_s}")


if __name__ == "__main__":
    main()
