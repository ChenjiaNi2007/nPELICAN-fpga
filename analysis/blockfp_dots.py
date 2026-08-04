#!/usr/bin/env python3
"""
analysis/blockfp_dots.py -- evidence for Lever 7 (docs/RESOURCE_REDUCTION_LEVERS.md).

Answers the question "should we quantize parts of the dot product individually?"
by measuring, on real events, three things:

  1. per-COMPONENT (E/px/py/pz) quantization  -> zero-sum (Lever 7a)
  2. why the dot front-end is expensive       -> cancellation + coarse dot_t (7b)
  3. per-PARTICLE block floating point        -> ~4 bits (7c)

The metric that matters is NOT dot error in GeV^2 but the fraction of d_ij that
land in a DIFFERENT dot_t cell than float-exact -- the dots-level gate already
used elsewhere in this repo. A scheme with 10x lower raw error that never crosses
a dot_t boundary buys nothing.

Usage:
    python analysis/blockfp_dots.py \
        [--data ../PELICAN-nano/data/sample_data/valid.h5] \
        [--events 3000] [--out analysis/out/blockfp_dots.txt]

Defaults reproduce the tables quoted in Lever 7 (w6a6i6p12 grid).
"""
import argparse
import os
import sys

import h5py
import numpy as np

# Minkowski signature, and the two beam spurions the firmware prepends.
G = np.array([1.0, -1.0, -1.0, -1.0])
BEAMS = np.array([[1.0, 0, 0, 1], [1.0, 0, 0, -1]])


def qfix(x, I, W):
    """ap_fixed<W,I> with round-to-nearest + saturate (AP_RND_CONV/AP_SAT)."""
    F = W - I
    s = 2.0 ** F
    lim = 2.0 ** (I - 1)
    return np.clip(np.rint(x * s) / s, -lim, lim - 2.0 ** -F)


def make_dot_quant(lsb, clip):
    """dot_t grid. w6a6i6p12 learns ap_fixed<6,10>: LSB 16, clip +-512."""
    def q(d):
        return np.clip(np.rint(d / lsb) * lsb, -clip, clip - lsb)
    return q


def dots(p):
    """Minkowski Gram matrix of a (n,4) event."""
    return np.einsum('ik,jk,k->ij', p, p, G)


def events(P, Nobj, n_events, with_beams=True):
    for e in range(n_events):
        n = int(Nobj[e])
        p = P[e, :n]
        if with_beams:
            p = np.concatenate([p, BEAMS])
        yield p


# --------------------------------------------------------------------------
# quantization schemes: each maps (event 4-momenta, width) -> Gram matrix
# --------------------------------------------------------------------------

def sch_uniform(p, W, I=10):
    """Current production: one global ap_fixed<W,I> grid for all components."""
    return dots(qfix(p, I, W))


def sch_percomp(p, W, I_uni=10, I_pc=(12, 11, 11, 12)):
    """Lever 7a: own integer bits per component, same LSB."""
    cols = [qfix(p[:, k], I_pc[k], W - (I_uni - I_pc[k])) for k in range(4)]
    return dots(np.stack(cols, axis=1))


def sch_percomp_samew(p, W, I_pc=(12, 11, 11, 12)):
    """Lever 7a variant: own integer bits, same TOTAL width (finer LSB on px/py)."""
    cols = [qfix(p[:, k], I_pc[k], W) for k in range(4)]
    return dots(np.stack(cols, axis=1))


def sch_blockfp(p, W, exp_min=0, exp_max=10, from_energy=True):
    """Lever 7c: per-particle po2 exponent, W-bit signed mantissas at I=2.

    from_energy=True uses e = floor(log2 E) -- no 4-way max in hardware, and
    provably non-overflowing at I=2 since E >= |p_k| and 2^e <= E < 2^(e+1).
    """
    base = np.abs(p[:, :1]) if from_energy else np.abs(p).max(1, keepdims=True)
    base = np.where(base > 0, base, 1.0)
    e = np.clip(np.floor(np.log2(base)), exp_min, exp_max)
    return dots(qfix(p / 2.0 ** e, 2, W) * 2.0 ** e)


def sch_blockfp_event(p, W, exp_min=0, exp_max=10):
    """Cheaper straw man: ONE exponent per event instead of per particle."""
    mx = np.abs(p).max()
    mx = mx if mx > 0 else 1.0
    e = np.clip(np.floor(np.log2(mx)), exp_min, exp_max)
    return dots(qfix(p / 2.0 ** e, 2, W) * 2.0 ** e)


def sch_lightcone(p, W):
    """Rejected (7f): linear change of basis to E+-pz. 6 mults/pair, worse."""
    pp = qfix(p[:, 0] + p[:, 3], 13, W)
    pm = qfix(p[:, 0] - p[:, 3], 13, W)
    px = qfix(p[:, 1], 11, W)
    py = qfix(p[:, 2], 11, W)
    return 0.5 * (np.outer(pp, pm) + np.outer(pm, pp)) - np.outer(px, px) - np.outer(py, py)


def sch_cancelfree(p, W):
    """Rejected (7f): d = 2 pT_i pT_j [sinh^2(dy/2) + sin^2(dphi/2)], massless.

    Cancellation-free (sum of positive terms) and manifestly invariant under
    longitudinal boosts, but needs ~506 trig LUTs and a training coordinate change.
    """
    pT = np.hypot(p[:, 1], p[:, 2])
    eta = np.arcsinh(p[:, 3] / np.maximum(pT, 1e-30))
    phi = np.arctan2(p[:, 2], p[:, 1])
    pTq, eq, phq = qfix(pT, 12, W), qfix(eta, 5, W), qfix(phi, 3, W)
    dy = eq[:, None] - eq[None, :]
    dp = (phq[:, None] - phq[None, :] + np.pi) % (2 * np.pi) - np.pi
    return 2 * np.outer(pTq, pTq) * (np.sinh(dy / 2) ** 2 + np.sin(dp / 2) ** 2)


# --------------------------------------------------------------------------


def upper(d):
    iu = np.triu_indices(d.shape[0], 1)
    return d[iu]


def gate(scheme, P, Nobj, n_events, W, qdot, ref):
    """Fraction of d_ij landing in a different dot_t cell than float-exact."""
    out = np.concatenate([upper(qdot(scheme(p, W))) for p in events(P, Nobj, n_events)])
    return (out != ref).mean() * 100


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(
        here, '..', '..', 'PELICAN-nano', 'data', 'sample_data', 'valid.h5'))
    ap.add_argument('--events', type=int, default=3000)
    ap.add_argument('--dot-lsb', type=float, default=16.0,
                    help='dot_t LSB; w6a6i6p12 learns ap_fixed<6,10> -> 16')
    ap.add_argument('--dot-clip', type=float, default=512.0)
    ap.add_argument('--out', default=os.path.join(here, 'out', 'blockfp_dots.txt'))
    args = ap.parse_args()

    with h5py.File(args.data, 'r') as f:
        n = min(args.events, len(f['Nobj']))
        P, Nobj = f['Pmu'][:n], f['Nobj'][:n]
        # Section 1 (component ranges) uses the FULL file: ranges are a property of
        # the dataset, not of the event subset the gate is measured on.
        P_all, Nobj_all = f['Pmu'][:], f['Nobj'][:]
    qdot = make_dot_quant(args.dot_lsb, args.dot_clip)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fh = open(args.out, 'w')

    def emit(s=''):
        print(s)
        fh.write(s + '\n')

    emit(f"blockfp_dots.py -- Lever 7 evidence")
    emit(f"data={args.data}  events={n}  dot_t LSB={args.dot_lsb} clip=+-{args.dot_clip}")
    emit()

    # ---- 1. per-component ranges (Lever 7a) ----
    mask = np.arange(P_all.shape[1])[None, :] < Nobj_all[:, None]
    Pm = P_all[mask]
    emit("=== 1. per-component ranges (real particles, FULL file, K=%d) ===" % len(Pm))
    emit(f"{'comp':<4} {'max|.|':>10} {'p99':>9} {'median|.|':>10} {'int bits':>9}")
    for k, nm in enumerate(['E', 'px', 'py', 'pz']):
        a = np.abs(Pm[:, k])
        emit(f"{nm:<4} {a.max():10.2f} {np.percentile(a, 99):9.2f} "
             f"{np.median(a):10.4f} {int(np.ceil(np.log2(a.max()))) + 1:9d}")
    emit("-> ceiling is 1 bit on 2 of 4 multipliers; 0 bits in production")
    emit("   (trained pmu_quant clips all four at +-512 => I=10 for all).")
    emit()

    # ---- 2. cancellation + dot_t occupancy (Lever 7b) ----
    raw, mx = [], []
    for p in events(P, Nobj, n):
        d = dots(p)
        raw.append(upper(d))
        mx.append(upper(np.abs(p[:, None, :] * p[None, :, :]).max(-1)))
    raw, mx = np.concatenate(raw), np.concatenate(mx)
    canc = mx / np.maximum(np.abs(raw), 1e-30)
    emit("=== 2a. cancellation  max_term/|d_ij|  (i<j) ===")
    for q in [50, 90, 99, 99.9, 100]:
        v = np.percentile(canc, q)
        emit(f"  p{q:<5}: {v:.3e}   ({np.log2(v):.1f} bits lost)")
    ref = qdot(raw)
    u = np.unique(ref)
    emit()
    emit("=== 2b. dot_t occupancy vs the real d_ij distribution ===")
    emit(f"  |d| median {np.median(np.abs(raw)):.2f}  p99 {np.percentile(np.abs(raw), 99):.1f}"
         f"  max {np.abs(raw).max():.1f}")
    emit(f"  quantize to exactly 0 : {(ref == 0).mean() * 100:.1f}%")
    emit(f"  saturating            : {(np.abs(ref) >= args.dot_clip - args.dot_lsb).mean() * 100:.2f}%")
    emit(f"  distinct levels used  : {len(u)} of {int(2 * args.dot_clip / args.dot_lsb)}")
    emit()

    # ---- 3. the gate (Levers 7a, 7c, 7f) ----
    emit("=== 3. dot_t cell mismatch vs float-exact (%) -- the metric that matters ===")
    schemes = [
        ('uniform (current)', sch_uniform),
        ('per-component I', sch_percomp),
        ('per-component sameW', sch_percomp_samew),
        ('block-FP / particle', sch_blockfp),
        ('block-FP / event', sch_blockfp_event),
        ('light-cone', sch_lightcone),
        ('cancel-free (pT,y,phi)', sch_cancelfree),
    ]
    widths = [12, 11, 10, 9, 8, 7]
    emit(f"{'scheme':<24}" + ''.join(f"{'W=%d' % W:>9}" for W in widths))
    for nm, fn in schemes:
        row = ''.join(f"{gate(fn, P, Nobj, n, W, qdot, ref):8.2f}%" for W in widths)
        emit(f"{nm:<24}{row}")
    emit()
    emit("Headline: block-FP/particle at W=8 beats uniform at W=12 (production).")
    emit("per-component I is bit-identical to uniform -- zero-sum, see Lever 7a.")
    emit()

    # ---- 4. block-FP design choices ----
    emit("=== 4. block-FP: exponent source and clamp range (mismatch %) ===")
    emit(f"{'W':>3} {'e=LZC(max|p_k|)':>17} {'e=LZC(E)':>10}  | clamp sweep on e=LZC(E)")
    for W in [12, 10, 9, 8, 7]:
        a = gate(lambda p, W: sch_blockfp(p, W, from_energy=False), P, Nobj, n, W, qdot, ref)
        b = gate(sch_blockfp, P, Nobj, n, W, qdot, ref)
        clamps = ''.join(
            f"  [{lo},{hi}]={gate(lambda p, W, lo=lo, hi=hi: sch_blockfp(p, W, lo, hi), P, Nobj, n, W, qdot, ref):.2f}%"
            for lo, hi in [(-4, 10), (0, 10), (2, 10)])
        emit(f"{W:>3} {a:16.2f}% {b:9.2f}% |{clamps}")
    emit()
    emit("-> E-based exponent is free (identical), and [0,10] (4-bit field) costs nothing.")

    fh.close()
    print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == '__main__':
    main()
