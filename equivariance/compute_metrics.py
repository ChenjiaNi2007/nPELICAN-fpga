"""
compute_metrics.py — pair, aggregate, plot (Option A: dual-mode + overlay).

Reads canonical/manifest.csv and the swept logits, pairs each boosted event with the
SAME jet's original at the SAME bit-width (f_b(Lambda x) vs f_b(x)), and emits per-mode
aggregates plus the headline boosted-vs-fixed overlay.

Modes (beam convention, set by run_sweep.py --mode):
  * boostedbeams  — beams transform WITH the particles (Option-A Lorentz invariance):
    the reference curve collapses to the pure fixed-point floor.
  * fixedbeams    — beams held at (1,0,0,+/-1): the reference curve is the non-zero
    "fixed-beam floor" (intended frame-dependence).
  * legacy        — un-tagged logits_<label>.dat from a pre-Option-A run (back-compat).

Per mode it writes:
  results/results_<mode>.csv      tidy per-(config, beta, dir, jet) rows
  results/aggregates_<mode>.csv   per-(config, beta) metrics
  results/plots_<mode>/*.png      the standard SEAL-style curves

Across modes (when both boostedbeams and fixedbeams are present) it writes:
  results/plots_overlay/*.png     boosted (solid) vs fixed (dashed) per metric; the
                                  reference-curve overlay is the headline, and the GAP
                                  between the two reference curves IS the fixed-beam
                                  symmetry-breaking floor (printed + annotated).

All plots/aggregates are re-derivable from the results_<mode>.csv WITHOUT re-running csim.

Run:  python compute_metrics.py [--config config.yaml] [--mode both|boostedbeams|fixedbeams|legacy]
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

# logit-file naming per mode. "legacy" is the pre-Option-A un-tagged convention.
def logit_path(mode: str, label: str) -> str:
    if mode == "legacy":
        return os.path.join(RESULTS_DIR, f"logits_{safe_name(label)}.dat")
    return os.path.join(RESULTS_DIR, f"logits_{mode}_{safe_name(label)}.dat")


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


def aggregate(mode, models, man_arrays):
    """Build (results_rows, per_bw, agg_rows, betas) for one mode. Returns None if no
    logits for this mode are present yet."""
    jet, beta, dir_idx, truth = man_arrays
    n_rows = len(jet)

    results_rows = []
    per_bw = {}
    for mdl in models:
        bw = model_label(mdl)
        lp = logit_path(mode, bw)
        if not os.path.exists(lp):
            print(f"[metrics:{mode}] skip {bw}: no {os.path.basename(lp)} (not swept)")
            continue
        logit = np.loadtxt(lp, dtype=np.float64).reshape(-1)
        if len(logit) != n_rows:
            sys.exit(f"[metrics:{mode}] {lp} has {len(logit)} logits, manifest has {n_rows}")

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
        return None

    betas = sorted(set(beta.tolist()))
    agg_rows = []
    for bw, d in sorted(per_bw.items(), key=lambda kv: label_sort_key(kv[0])):
        for b in betas:
            sel = beta == b
            s, s0 = d["sig"][sel], d["sig0"][sel]
            drift = float(np.mean((s0 - s) ** 2))
            mad = float(np.mean(np.abs(s0 - s)))
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
    return results_rows, per_bw, agg_rows, betas


def write_csvs(mode, results_rows, agg_rows):
    rp = os.path.join(RESULTS_DIR, f"results_{mode}.csv")
    with open(rp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "jet_idx", "beta", "dir_idx",
                                          "truth", "logit", "w0", "sigma", "sigma0", "flip"])
        w.writeheader(); w.writerows(results_rows)
    ap = os.path.join(RESULTS_DIR, f"aggregates_{mode}.csv")
    with open(ap, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "beta", "n", "score_drift_mse",
                                          "mean_abs_dsigma", "mean_abs_dlogit",
                                          "flip_rate", "auc", "inv_eps_b"])
        w.writeheader(); w.writerows(agg_rows)
    print(f"[metrics:{mode}] wrote {rp} ({len(results_rows)} rows) and {ap}")
    return ap


def series(agg_rows, bw, key):
    xs = [r["beta"] for r in agg_rows if r["config"] == bw]
    ys = [r[key] for r in agg_rows if r["config"] == bw]
    return np.array(xs), np.array(ys)


def per_mode_plots(mode, per_bw, agg_rows, ref_label):
    plot_dir = os.path.join(RESULTS_DIR, f"plots_{mode}")
    os.makedirs(plot_dir, exist_ok=True)
    bws = sorted(per_bw.keys(), key=label_sort_key)
    nonref = [b for b in bws if b != ref_label]
    cmap = plt.cm.viridis(np.linspace(0.05, 0.85, max(len(nonref), 1)))
    colour = {b: cmap[i] for i, b in enumerate(nonref)}

    def style(bw):
        if bw == ref_label:
            return dict(color="k", ls="--", marker="o", lw=2.2,
                        label=f"{bw} (reference)", zorder=10)
        return dict(color=colour[bw], marker="o", lw=1.6, label=f"{bw}")

    def curve(key, ylabel, title, fname, logy=False):
        plt.figure(figsize=(7, 5))
        for bw in bws:
            x, y = series(agg_rows, bw, key)
            if logy:
                y = np.where(y <= 0, np.nan, y)
            if key == "inv_eps_b":
                y = np.array([np.nan if not np.isfinite(v) else v for v in y])
            plt.plot(x, y, **style(bw))
        if logy:
            plt.yscale("log")
        plt.xlabel(r"boost magnitude $|\beta|$"); plt.ylabel(ylabel)
        plt.title(f"{title}  [{mode}]")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(title="W:A:I bits", fontsize=8, ncol=2)
        plt.tight_layout(); plt.savefig(os.path.join(plot_dir, fname), dpi=130); plt.close()

    curve("score_drift_mse", r"$\langle(\sigma(w_0)-\sigma(w_\beta))^2\rangle$",
          "Score drift vs boost (equivariance violation)", "score_drift.png", logy=True)
    curve("mean_abs_dsigma", r"$\langle|\Delta\sigma|\rangle$",
          "Mean score shift vs boost", "mean_abs_dsigma.png")
    curve("mean_abs_dlogit", r"$\langle|w_0 - w_\beta|\rangle$",
          "Logit drift vs boost (operating-point independent)", "logit_drift.png", logy=True)
    curve("flip_rate", "decision-flip rate",
          r"Fraction of jets crossing $w=0$ under boost", "flip_rate.png")
    curve("auc", "AUC (w_beta vs truth)", "Discrimination under boost (AUC)", "auc.png")
    curve("inv_eps_b", r"$1/\epsilon_B$ at $\epsilon_S=0.3$",
          "Background rejection under boost", "inv_eps_b.png")
    print(f"[metrics:{mode}] wrote plots -> {plot_dir}/")


# ---- overlay: boosted (solid) vs fixed (dashed), headline = reference curve ----
OVERLAY_METRICS = [
    ("score_drift_mse", r"$\langle(\sigma(w_0)-\sigma(w_\beta))^2\rangle$",
     "Score drift", True),
    ("mean_abs_dlogit", r"$\langle|w_0 - w_\beta|\rangle$", "Logit drift", True),
    ("flip_rate", "decision-flip rate", "Decision-flip rate", False),
    ("auc", "AUC (w_beta vs truth)", "AUC", False),
]


def overlay_plots(agg_by_mode, per_bw_by_mode, ref_label):
    """agg_by_mode: {mode: agg_rows}. Requires both boostedbeams and fixedbeams."""
    if not ({"boostedbeams", "fixedbeams"} <= set(agg_by_mode)):
        print("[overlay] need both boostedbeams and fixedbeams aggregates; skipping overlay.")
        return
    plot_dir = os.path.join(RESULTS_DIR, "plots_overlay")
    os.makedirs(plot_dir, exist_ok=True)
    agg_b = agg_by_mode["boostedbeams"]
    agg_f = agg_by_mode["fixedbeams"]

    # bit-widths present in both modes, bit-budget ordered
    bws = sorted(set(per_bw_by_mode["boostedbeams"]) & set(per_bw_by_mode["fixedbeams"]),
                 key=label_sort_key)
    nonref = [b for b in bws if b != ref_label]
    cmap = plt.cm.viridis(np.linspace(0.05, 0.85, max(len(nonref), 1)))
    colour = {b: cmap[i] for i, b in enumerate(nonref)}
    colour[ref_label] = "k"

    # ---- headline: reference curve, boosted vs fixed, per metric ----
    for key, ylabel, title, logy in OVERLAY_METRICS:
        plt.figure(figsize=(7, 5))
        xf, yf = series(agg_f, ref_label, key)
        xb, yb = series(agg_b, ref_label, key)
        if logy:
            yf_p = np.where(yf <= 0, np.nan, yf)
            yb_p = np.where(yb <= 0, np.nan, yb)
        else:
            yf_p, yb_p = yf, yb
        plt.plot(xf, yf_p, color="C3", ls="--", marker="s", lw=2.2,
                 label=f"{ref_label} fixed beams (floor)")
        plt.plot(xb, yb_p, color="C0", ls="-", marker="o", lw=2.2,
                 label=f"{ref_label} boosted beams (Option A)")
        if logy:
            plt.yscale("log")
        plt.xlabel(r"boost magnitude $|\beta|$"); plt.ylabel(ylabel)
        plt.title(f"{title}: reference — fixed-beam floor vs Option-A invariance")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"reference_overlay_{key}.png"), dpi=130)
        plt.close()

    # ---- full overlay: all bit-widths, boosted (solid) vs fixed (dashed) ----
    for key, ylabel, title, logy in OVERLAY_METRICS:
        plt.figure(figsize=(7.5, 5.5))
        for bw in bws:
            c = colour[bw]
            xf, yf = series(agg_f, bw, key)
            xb, yb = series(agg_b, bw, key)
            if logy:
                yf = np.where(yf <= 0, np.nan, yf)
                yb = np.where(yb <= 0, np.nan, yb)
            lw = 2.4 if bw == ref_label else 1.4
            plt.plot(xf, yf, color=c, ls="--", marker="s", lw=lw, alpha=0.9)
            plt.plot(xb, yb, color=c, ls="-", marker="o", lw=lw, alpha=0.9,
                     label=f"{bw}")
        if logy:
            plt.yscale("log")
        plt.xlabel(r"boost magnitude $|\beta|$"); plt.ylabel(ylabel)
        plt.title(f"{title}: solid=boosted beams, dashed=fixed beams")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(title="W:A:I bits", fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"all_overlay_{key}.png"), dpi=130)
        plt.close()

    print(f"[overlay] wrote plots -> {plot_dir}/")

    # ---- quantify the fixed-beam floor (gap between reference curves) ----
    print("\n=== FIXED-BEAM FLOOR (reference, boosted vs fixed) ===")
    print(f"  metric            |beta|   fixed(floor)   boosted(OptionA)   ratio fixed/boosted")
    for key, _, _, _ in OVERLAY_METRICS:
        xf, yf = series(agg_f, ref_label, key)
        xb, yb = series(agg_b, ref_label, key)
        for i, bval in enumerate(xf):
            if bval == 0.0:
                continue
            f_, b_ = yf[i], yb[i]
            ratio = (f_ / b_) if b_ not in (0.0,) and np.isfinite(b_) and b_ > 0 else float("inf")
            ratio_s = "   inf" if not np.isfinite(ratio) else f"{ratio:8.1f}"
            print(f"  {key:>16}  {bval:5.2f}   {f_:.3e}     {b_:.3e}      {ratio_s}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--mode", choices=["both", "boostedbeams", "fixedbeams", "legacy"],
                   default="both", help="which swept logits to process")
    a = p.parse_args()
    cfg = load_config(a.config)
    models = cfg["models"]
    ref_label = next((model_label(m) for m in models if m.get("reference")), None)

    man = read_manifest()
    jet = np.array([m["jet_idx"] for m in man])
    beta = np.array([m["beta"] for m in man])
    dir_idx = np.array([m["dir_idx"] for m in man])
    truth = np.array([m["truth"] for m in man])
    man_arrays = (jet, beta, dir_idx, truth)

    modes = ["boostedbeams", "fixedbeams"] if a.mode == "both" else [a.mode]

    agg_by_mode = {}
    per_bw_by_mode = {}
    for mode in modes:
        res = aggregate(mode, models, man_arrays)
        if res is None:
            print(f"[metrics:{mode}] no swept logits found; skipping.")
            continue
        results_rows, per_bw, agg_rows, _ = res
        write_csvs(mode, results_rows, agg_rows)
        per_mode_plots(mode, per_bw, agg_rows, ref_label)
        agg_by_mode[mode] = agg_rows
        per_bw_by_mode[mode] = per_bw

        # console summary
        print(f"\n[{mode}] config(W:A:I)  beta   drift_mse   flip_rate   auc      1/epsB@0.3")
        for r in agg_rows:
            ie = r["inv_eps_b"]
            ie_s = "  inf" if not np.isfinite(ie) else f"{ie:7.2f}"
            print(f"  {r['config']:>10}  {r['beta']:.2f}  {r['score_drift_mse']:.3e}  "
                  f"{r['flip_rate']:.4f}   {r['auc']:.4f}  {ie_s}")

    if not agg_by_mode:
        sys.exit("[metrics] no swept logits found. Run run_sweep.py first.")

    if {"boostedbeams", "fixedbeams"} <= set(agg_by_mode):
        overlay_plots(agg_by_mode, per_bw_by_mode, ref_label)


if __name__ == "__main__":
    main()
