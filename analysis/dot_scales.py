#!/usr/bin/env python3
"""Characterize the dynamic range of the Minkowski dot products d_ij = p_i . p_j
that feed the firmware input (input_t).

Reproduces exactly what the firmware sees: top NPARTICLES=20 energy-sorted
constituents + 2 constant massless beam spurions (1,0,0,+1)/(1,0,0,-1), dots
computed in float32 (matching PyTorch training dtype), padded entries masked out.

Outputs (to analysis/out/):
  dot_products_distribution.png  - where the dots live on the log2 scale (+ CCDF)
  dot_products_bits.png          - integer vs fractional bits needed per dot,
                                   and the saturation/underflow tradeoff at W=24
  dot_scales_summary.txt         - percentile table and headline numbers

Usage:
  .venv/python analysis/dot_scales.py [--pmu tb_data/10k_pmu_test.dat]
                                      [--nobj tb_data/10k_nobj.dat]
"""

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

NPARTICLES = 20

# current firmware input_t = ap_fixed<24,12>: 11 magnitude bits + sign, 12 frac bits
CUR_W, CUR_I = 24, 12
CUR_SAT = 2.0 ** (CUR_I - 1)      # 2048
CUR_LSB = 2.0 ** -(CUR_W - CUR_I) # 2^-12


def load_events(pmu_path, nobj_path):
    raw = np.loadtxt(pmu_path, dtype=np.float64)
    nobj = np.loadtxt(nobj_path, dtype=np.int64)
    nev = raw.shape[0]
    pmu = raw.reshape(nev, -1, 4)[:, :NPARTICLES, :].astype(np.float32)
    beams = np.array([[1, 0, 0, 1], [1, 0, 0, -1]], dtype=np.float32)
    beams = np.broadcast_to(beams, (nev, 2, 4))
    p = np.concatenate([beams, pmu], axis=1)  # [nev, 22, 4], beams first
    nvalid = np.clip(nobj, 0, NPARTICLES) + 2
    return p, nvalid


def gram(p):
    eta = np.diag(np.array([1, -1, -1, -1], dtype=np.float32))
    return np.einsum("nia,ab,njb->nij", p, eta, p).astype(np.float32)


def masked_upper_pairs(g, nvalid):
    """Return off-diagonal masked dots plus, per dot: category id
    (0 = beam-beam, 1 = beam-particle, 2 = particle-particle),
    event index, and the two constituent indices (0,1 = beams)."""
    nev, n, _ = g.shape
    idx = np.arange(n)
    valid = idx[None, :] < nvalid[:, None]              # [nev, n]
    pair_ok = valid[:, :, None] & valid[:, None, :]
    upper = idx[:, None] < idx[None, :]                 # i < j
    sel = pair_ok & upper[None, :, :]

    is_beam = idx < 2
    nbeams = is_beam[:, None].astype(int) + is_beam[None, :].astype(int)  # 0/1/2
    cat = np.broadcast_to((2 - nbeams)[None, :, :], g.shape)
    # cat: 2 beams -> 0, 1 beam -> 1, 0 beams -> 2
    ev = np.broadcast_to(np.arange(nev)[:, None, None], g.shape)
    ii = np.broadcast_to(idx[None, :, None], g.shape)
    jj = np.broadcast_to(idx[None, None, :], g.shape)
    return g[sel], cat[sel], ev[sel], ii[sel], jj[sel]


def bits_needed(x):
    """Integer/fractional bits to encode each float32 value exactly.
    intbits = magnitude bits above the binary point (0 for |x|<1),
    fracbits = bits below the binary point down to the lowest set mantissa bit."""
    x = np.asarray(x, dtype=np.float32)
    m, e = np.frexp(np.abs(x))                          # x = m * 2^e, m in [0.5,1)
    sig = np.round(m.astype(np.float64) * (1 << 24)).astype(np.int64)
    tz = np.where(sig > 0, np.log2((sig & -sig).astype(np.float64)), 24).astype(np.int64)
    lsb_exp = e.astype(np.int64) - 24 + tz              # exponent of lowest set bit
    intbits = np.maximum(0, e)
    fracbits = np.maximum(0, -lsb_exp)
    intbits = np.where(x == 0, 0, intbits)
    fracbits = np.where(x == 0, 0, fracbits)
    return intbits, fracbits


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmu", default=os.path.join(repo, "tb_data", "10k_pmu_test.dat"))
    ap.add_argument("--nobj", default=os.path.join(repo, "tb_data", "10k_nobj.dat"))
    ap.add_argument("--out", default=os.path.join(here, "out"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    p, nvalid = load_events(args.pmu, args.nobj)
    g = gram(p)
    dots, cat, ev, ii, jj = masked_upper_pairs(g, nvalid)

    pos = dots > 0
    dpos = dots[pos].astype(np.float64)
    log2d = np.log2(dpos)
    cat_pos = cat[pos]
    ev_pos, ii_pos, jj_pos = ev[pos], ii[pos], jj[pos]

    # ---------- summary ----------
    qs = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9, 99.99]
    pct = np.percentile(dpos, qs)
    frac_sat = float(np.mean(dpos >= CUR_SAT))
    frac_zero = float(np.mean(dpos < CUR_LSB / 2))  # rounds to 0 on the current grid
    lines = []
    lines.append(f"events: {p.shape[0]}, masked off-diagonal dots (i<j): {dots.size}")
    lines.append(f"non-positive dots (massless d_ii excluded already): "
                 f"{int(np.sum(~pos))} ({100 * np.mean(~pos):.3f}%)")
    lines.append(f"min positive: {dpos.min():.3e} (2^{np.log2(dpos.min()):.1f})   "
                 f"max: {dpos.max():.3e} (2^{np.log2(dpos.max()):.1f})")
    lines.append(f"dynamic range: {np.log2(dpos.max()) - np.log2(dpos.min()):.1f} octaves")
    lines.append("")
    lines.append("percentile      value        log2")
    for q, v in zip(qs, pct):
        lines.append(f"  {q:7.2f}%   {v:12.5g}   {np.log2(v):7.2f}")
    lines.append("")
    lines.append(f"current input_t = ap_fixed<{CUR_W},{CUR_I}>: "
                 f"saturates at {CUR_SAT:.0f} (2^{CUR_I - 1}), LSB 2^-{CUR_W - CUR_I}")
    lines.append(f"  fraction saturating (>= 2^{CUR_I - 1}): {100 * frac_sat:.4f}%")
    lines.append(f"  fraction rounding to zero (< 2^-{CUR_W - CUR_I - 1} ulp): "
                 f"{100 * frac_zero:.4f}%")
    for thresh_bits in [8, 9, 10, 11, 12, 13, 14, 15]:
        f = float(np.mean(dpos >= 2.0 ** thresh_bits))
        lines.append(f"  fraction of dots >= 2^{thresh_bits}: {100 * f:.4f}%")
    lines.append("")
    names = {0: "beam-beam", 1: "beam-particle", 2: "particle-particle"}
    for c in [0, 1, 2]:
        d = dpos[cat_pos == c]
        if d.size:
            lines.append(f"{names[c]:>18}: n={d.size:>8}  median={np.median(d):10.4g}  "
                         f"p99.9={np.percentile(d, 99.9):10.4g}  max={d.max():10.4g}")

    # structure of the tail: is the high scale a few pairs in a few events?
    sat = dpos >= CUR_SAT
    sat_ev = ev_pos[sat]
    ev_unique, ev_counts = np.unique(sat_ev, return_counts=True)
    lines.append("")
    lines.append(f"tail structure (dots >= 2^{CUR_I - 1} = {CUR_SAT:.0f}):")
    lines.append(f"  saturating dots: {int(sat.sum())} across {ev_unique.size} events "
                 f"({100 * ev_unique.size / p.shape[0]:.2f}% of events)")
    if ev_unique.size:
        lines.append(f"  dots/affected event: median {np.median(ev_counts):.0f}, "
                     f"max {ev_counts.max()}")
        # constituent rank (0 = leading by energy); beams are index 0,1
        lead = np.minimum(ii_pos[sat], jj_pos[sat])
        n_beam_pair = int(np.sum(lead < 2))
        rank = np.maximum(lead - 2, 0)
        lines.append(f"  pairs involving a beam: {n_beam_pair} "
                     f"({100 * n_beam_pair / max(1, sat.sum()):.1f}%)")
        for r in range(4):
            f_ = np.mean(rank <= r)
            lines.append(f"  fraction with lower-index constituent rank <= {r}: "
                         f"{100 * f_:.1f}%")
    summary = "\n".join(lines)
    print(summary)
    with open(os.path.join(args.out, "dot_scales_summary.txt"), "w") as f:
        f.write(summary + "\n")

    # ---------- figure 1: distribution + CCDF ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    bins = np.arange(np.floor(log2d.min()), np.ceil(log2d.max()) + 0.5, 0.5)
    for c, color in zip([2, 1, 0], ["tab:blue", "tab:orange", "tab:green"]):
        ax.hist(log2d[cat_pos == c], bins=bins, histtype="stepfilled",
                alpha=0.55, color=color, label=names[c])
    ax.axvline(CUR_I - 1, color="red", ls="--", lw=1.5,
               label=f"input_t saturation $2^{{{CUR_I - 1}}}$")
    ax.axvline(-(CUR_W - CUR_I), color="purple", ls="--", lw=1.5,
               label=f"input_t LSB $2^{{-{CUR_W - CUR_I}}}$")
    ax.set_xlabel(r"$\log_2\,d_{ij}$")
    ax.set_ylabel("count")
    ax.set_yscale("log")
    ax.set_title(f"Dot products, {p.shape[0]} events (masked, $i<j$)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    srt = np.sort(log2d)
    ccdf = 1.0 - np.arange(1, srt.size + 1) / srt.size
    ax.semilogy(srt, np.maximum(ccdf, 1.0 / srt.size))
    ax.axvline(CUR_I - 1, color="red", ls="--", lw=1.5)
    for tb in [11, 13, 15]:
        f = float(np.mean(dpos >= 2.0 ** tb))
        if f > 0:
            ax.annotate(f"$\\geq 2^{{{tb}}}$: {100 * f:.3g}%",
                        xy=(tb, max(f, 1.0 / srt.size)),
                        xytext=(tb - 9, f * 4), fontsize=9,
                        arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel(r"$\log_2\,d_{ij}$")
    ax.set_ylabel(r"fraction of dots $\geq$ x  (CCDF)")
    ax.set_title("Tail: how many dots live at high scale")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "dot_products_distribution.png"), dpi=150)
    plt.close(fig)

    # ---------- figure 2: integer vs fractional bits ----------
    ibits, fbits = bits_needed(dots[pos].astype(np.float32))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    h = ax.hist2d(ibits, fbits,
                  bins=[np.arange(-0.5, ibits.max() + 1.5),
                        np.arange(-0.5, fbits.max() + 1.5)],
                  norm=LogNorm(), cmap="viridis")
    fig.colorbar(h[3], ax=ax, label="count")
    for w in [16, 24, 32]:
        ax.plot([0, w], [w, 0], ls="--", lw=1, color="red")
        ax.annotate(f"W={w}", xy=(w * 0.62, w * 0.38), color="red", fontsize=9,
                    rotation=-38)
    ax.set_xlabel("integer bits needed (magnitude above binary point)")
    ax.set_ylabel("fractional bits needed (exact float32 encode)")
    ax.set_title("Bits to encode each dot product exactly")

    ax = axes[1]
    b = np.arange(ibits.min() - 0.5, ibits.max() + 1.5)
    ax.hist(ibits, bins=b, color="tab:blue", alpha=0.7)
    ax.set_yscale("log")
    ax.axvline(CUR_I - 1 + 0.0, color="red", ls="--", lw=1.5,
               label=f"input_t magnitude bits ({CUR_I - 1})")
    ax2 = ax.twinx()
    srt_i = np.sort(ibits)
    ax2.plot(srt_i, 100 * (1 - np.arange(1, srt_i.size + 1) / srt_i.size),
             color="k", lw=1.5)
    ax2.set_yscale("log")
    ax2.set_ylabel("% of dots needing more bits (CCDF)")
    ax.set_xlabel("integer bits needed")
    ax.set_ylabel("count")
    ax.set_title("Integer-bit demand")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)

    # tradeoff: at fixed W, sweep I -> fraction saturated vs fraction flushed to 0
    ax = axes[2]
    dabs = dots[pos].astype(np.float64)
    for W, ls in [(16, ":"), (20, "-."), (24, "-"), (28, "--")]:
        Is = np.arange(2, 19)
        sat = [np.mean(dabs >= 2.0 ** (I - 1)) for I in Is]
        und = [np.mean(dabs < 2.0 ** -(W - I + 1)) for I in Is]
        loss = np.array(sat) + np.array(und)
        ax.semilogy(Is, np.maximum(loss, 1e-9), ls=ls, marker="o", ms=3,
                    label=f"W={W}")
    ax.axvline(CUR_I, color="red", ls="--", lw=1, label=f"current I={CUR_I}")
    ax.set_xlabel("integer bits I in ap_fixed<W,I>")
    ax.set_ylabel("fraction saturated + flushed to 0")
    ax.set_title("Range-loss tradeoff vs type choice")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "dot_products_bits.png"), dpi=150)
    plt.close(fig)

    print(f"\nwrote plots to {args.out}/")


if __name__ == "__main__":
    main()
