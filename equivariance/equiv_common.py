"""
equiv_common.py — shared helpers for the equivariance sweep harness.

Path convention: every relative path in config.yaml is resolved against the
nPELICAN-fpga REPO ROOT (the parent of this equivariance/ directory).
"""
from __future__ import annotations

import os
import numpy as np
import yaml

# Directory layout ------------------------------------------------------------
EQUIV_DIR = os.path.dirname(os.path.abspath(__file__))
FPGA_ROOT = os.path.dirname(EQUIV_DIR)                       # nPELICAN-fpga/
CANON_DIR = os.path.join(EQUIV_DIR, "canonical")
RESULTS_DIR = os.path.join(EQUIV_DIR, "results")
TB_DATA = os.path.join(FPGA_ROOT, "tb_data")

NPARTICLES = 20          # firmware top-level input width (real particles only)


def load_config(path: str | None = None) -> dict:
    if path is None:
        path = os.path.join(EQUIV_DIR, "config.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def repo_path(rel: str) -> str:
    """Resolve a config path (relative to the nPELICAN-fpga repo root)."""
    if os.path.isabs(rel):
        return rel
    return os.path.normpath(os.path.join(FPGA_ROOT, rel))


# Model labels ----------------------------------------------------------------
# A model is identified by a string label. For these checkpoints the natural
# label is the "W:A:I" bit-width triple (weight:act:input), e.g. "6:6:12". Older
# configs used a single integer `bit_width`; we still accept that.
def model_label(mdl: dict) -> str:
    return str(mdl.get("label", mdl.get("bit_width")))


def safe_name(label: str) -> str:
    """Filesystem-safe form of a label (':' -> '_')."""
    return str(label).replace(":", "_").replace(" ", "")


def label_sort_key(label: str):
    """Order labels by total bit budget (sum of W,A,I), then componentwise.

    Puts e.g. 6:6:6 (18) < 6:6:8 (20) < 6:6:12 (24) < 8:8:16 (32) < ... < 24:24:24 (72).
    Falls back gracefully for plain-integer or non-numeric labels.
    """
    s = str(label)
    try:
        parts = [int(x) for x in s.split(":")]
        return (sum(parts), parts)
    except ValueError:
        try:
            v = int(s)
            return (v, [v])
        except ValueError:
            return (10**9, [s])


# Lorentz boost (SEAL eq.7) ---------------------------------------------------
def boost_matrix(beta_vec: np.ndarray) -> np.ndarray:
    """4x4 Lorentz boost acting on (E, px, py, pz) for velocity beta_vec.

    Metric diag(+,-,-,-). |beta| must be < 1. |beta|=0 returns the identity.
    Matches SEAL eq.(7):
       Lambda_00 = gamma ; Lambda_0i = Lambda_i0 = -gamma*beta_i ;
       Lambda_ij = delta_ij + (gamma-1)*beta_i*beta_j/|beta|^2
    """
    b = np.asarray(beta_vec, dtype=np.float64)
    b2 = float(b @ b)
    L = np.eye(4, dtype=np.float64)
    if b2 == 0.0:
        return L
    gamma = 1.0 / np.sqrt(1.0 - b2)
    L[0, 0] = gamma
    L[0, 1:] = -gamma * b
    L[1:, 0] = -gamma * b
    L[1:, 1:] = np.eye(3) + (gamma - 1.0) * np.outer(b, b) / b2
    return L


def sample_directions(n_dir: int, rng: np.random.Generator) -> np.ndarray:
    """n_dir unit vectors uniform on the sphere (isotropic normal, normalized)."""
    v = rng.standard_normal((n_dir, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def minkowski_gram(pmu: np.ndarray) -> np.ndarray:
    """Gram matrix of Minkowski dots d_ij = E_i E_j - p_i . p_j. pmu: [N,4]."""
    metric = np.array([1.0, -1.0, -1.0, -1.0])
    return (pmu * metric) @ pmu.T


# Classification metrics ------------------------------------------------------
def sigmoid(w):
    return 1.0 / (1.0 + np.exp(-np.asarray(w, dtype=np.float64)))


def mann_whitney_auc(y: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC via rank statistic. y in {0,1}. NaN if a class is absent."""
    y = np.asarray(y).astype(int)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    return (ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)


def inv_eps_b_at(y: np.ndarray, scores: np.ndarray, eps_s: float = 0.3) -> float:
    """1 / background-efficiency at the threshold giving signal-efficiency eps_s.

    Standard top-tagging discrimination metric. Higher = better. Returns inf if
    no background survives the threshold (eps_B == 0), NaN if a class is absent.
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    sig = s[y == 1]
    bkg = s[y == 0]
    if len(sig) == 0 or len(bkg) == 0:
        return float("nan")
    # threshold = the score at the (1-eps_s) quantile of signal scores, so that a
    # fraction eps_s of signal lies at/above it.
    thr = np.quantile(sig, 1.0 - eps_s, method="lower")
    eps_b = float((bkg >= thr).mean())
    if eps_b <= 0.0:
        return float("inf")
    return 1.0 / eps_b
