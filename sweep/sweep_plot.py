#!/usr/bin/env python3
"""Plot the n-hidden capacity sweep: accuracy-vs-params and resources-vs-params.

Reads sweep_results.csv (produced by sweep_nhidden.sh) and writes PNGs:

  sweep_accuracy.png    params -> AUC & accuracy
  sweep_resources.png   params -> LUT, FF, DSP (skipped if no synth data yet)
  sweep_pareto.png      DSP -> AUC, annotated with n_hidden (skipped if no synth)

Usage:  python sweep_plot.py [--csv sweep_results.csv] [--outdir .]

Rows whose resource columns are "NA" (csynth not yet run) are still used for the
accuracy plot; they are dropped from the resource/Pareto plots.
"""
import argparse
import csv
import os


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in list(r.items()):
            r[k] = v.strip() if isinstance(v, str) else v
    rows.sort(key=lambda r: _f(r.get("params")) or _f(r.get("n_hidden")) or 0)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="sweep_results.csv")
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load(a.csv)
    if not rows:
        raise SystemExit(f"No rows in {a.csv}")
    os.makedirs(a.outdir, exist_ok=True)

    params = [_f(r["params"]) for r in rows]
    nhid = [r["n_hidden"] for r in rows]
    auc = [_f(r.get("AUC")) for r in rows]
    acc = [_f(r.get("accuracy")) for r in rows]

    # ---- accuracy vs params ----
    fig, ax = plt.subplots(figsize=(6, 4))
    if any(v is not None for v in auc):
        ax.plot(params, auc, "o-", label="AUC", color="tab:blue")
    if any(v is not None for v in acc):
        ax.plot(params, acc, "s--", label="accuracy", color="tab:green")
    for x, n in zip(params, nhid):
        if x is not None:
            ax.annotate(f"h={n}", (x, (auc[params.index(x)] or acc[params.index(x)] or 0)),
                        textcoords="offset points", xytext=(0, 6), fontsize=8, ha="center")
    ax.set_xlabel("trainable parameters")
    ax.set_ylabel("score")
    ax.set_title("nPELICAN accuracy vs model size")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p1 = os.path.join(a.outdir, "sweep_accuracy.png")
    fig.savefig(p1, dpi=150)
    print("wrote", p1)

    # ---- resources vs params (only rows with numeric synth data) ----
    res_rows = [r for r in rows if _f(r.get("LUT")) is not None]
    if res_rows:
        rp = [_f(r["params"]) for r in res_rows]
        lut = [_f(r["LUT"]) for r in res_rows]
        ff = [_f(r["FF"]) for r in res_rows]
        dsp = [_f(r["DSP"]) for r in res_rows]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(rp, lut, "o-", label="LUT", color="tab:red")
        ax.plot(rp, ff, "s-", label="FF", color="tab:orange")
        ax.set_xlabel("trainable parameters")
        ax.set_ylabel("LUT / FF")
        ax.grid(True, alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(rp, dsp, "^--", label="DSP", color="tab:purple")
        ax2.set_ylabel("DSP")
        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [ln.get_label() for ln in lines], loc="upper left")
        ax.set_title("nPELICAN FPGA resources vs model size")
        fig.tight_layout()
        p2 = os.path.join(a.outdir, "sweep_resources.png")
        fig.savefig(p2, dpi=150)
        print("wrote", p2)

        # ---- Pareto: AUC vs DSP ----
        pa = [_f(r.get("AUC")) for r in res_rows]
        if any(v is not None for v in pa):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(dsp, pa, "o-", color="tab:blue")
            for x, y, n in zip(dsp, pa, [r["n_hidden"] for r in res_rows]):
                if x is not None and y is not None:
                    ax.annotate(f"h={n}", (x, y), textcoords="offset points",
                                xytext=(5, 5), fontsize=8)
            ax.set_xlabel("DSP")
            ax.set_ylabel("AUC")
            ax.set_title("Accuracy / DSP Pareto front")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            p3 = os.path.join(a.outdir, "sweep_pareto.png")
            fig.savefig(p3, dpi=150)
            print("wrote", p3)
    else:
        print("No numeric resource data yet (csynth not run) -> skipped resource plots.")


if __name__ == "__main__":
    main()
