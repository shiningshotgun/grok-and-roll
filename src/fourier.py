"""Fourier basis, projections, restricted/excluded loss, key-freq selection, Gini."""

import math

import numpy as np
import torch

P = 113
N_FREQS = 56  # 1..56


def fourier_basis_1d():
    """Return (P, P) real orthonormal basis: row 0 constant; rows 2k-1,2k cos/sin."""
    basis = np.zeros((P, P), dtype=np.float64)
    x = np.arange(P, dtype=np.float64)
    basis[0, :] = 1.0 / math.sqrt(P)
    for k in range(1, N_FREQS + 1):
        cosv = np.cos(2 * math.pi * k * x / P)
        sinv = np.sin(2 * math.pi * k * x / P)
        cosv = cosv / np.linalg.norm(cosv)
        sinv = sinv / np.linalg.norm(sinv)
        basis[2 * k - 1, :] = cosv
        basis[2 * k, :] = sinv
    return basis


def cos_sin_basis_vectors(k):
    """Return (cos_k, sin_k) 1D unit vectors for frequency k."""
    basis = fourier_basis_1d()
    return basis[2 * k - 1].copy(), basis[2 * k].copy()


def cos_xpy(k):
    cosk, sink = cos_sin_basis_vectors(k)
    # outer products
    M = (np.outer(cosk, cosk) - np.outer(sink, sink)) / math.sqrt(2)
    return M.reshape(-1)  # length P*P


def sin_xpy(k):
    cosk, sink = cos_sin_basis_vectors(k)
    M = (np.outer(sink, cosk) + np.outer(cosk, sink)) / math.sqrt(2)
    return M.reshape(-1)


def project_component(L, d):
    """L: (N_inputs, N_logits); d unit (N_inputs,). Returns outer(d, d@L)."""
    L = np.asarray(L, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    coef = d @ L  # (N_logits,)
    return np.outer(d, coef)  # (N_inputs, N_logits)


def restricted_loss(L, T, labels):
    # bias correction: add back mean over inputs of (L - T) to every row
    diff = L - T
    bias = diff.mean(axis=0)  # (N_logits,)
    corrected = T + bias[None, :]
    return cross_entropy(corrected, labels)


def excluded_loss(L, T, labels):
    return cross_entropy(L - T, labels)


def per_freq_exclusion(L, components_k, labels):
    # components_k: dict k -> (cos_comp, sin_comp) each (N_inputs,N_logits)
    # returns loss of L minus that freq's two components
    res = {}
    for k, (cc, sc) in components_k.items():
        ablated = L - cc - sc
        res[k] = cross_entropy(ablated, labels)
    return res


def cross_entropy(logits, labels):
    # logits: (N, C) numpy; labels: (N,) int
    logits = np.asarray(logits, dtype=np.float64)
    # stable softmax in float32-ish; use float64 for accuracy then cast? spec says float32.
    # We'll compute in float32 to match reported convention.
    lf = torch.from_numpy(logits.astype(np.float32))
    y = torch.from_numpy(np.asarray(labels).astype(np.int64))
    loss = torch.nn.functional.cross_entropy(lf, y)
    return float(loss.item())

 

def select_key_frequencies(mlp_post_act, basis_1d=None):
    """mlp_post_act: (P*P, D_MLP) or (P,P,D_MLP). Returns (list of k, support dict)."""
    if basis_1d is None:
        basis_1d = fourier_basis_1d()
    act = np.asarray(mlp_post_act, dtype=np.float64)
    if act.ndim == 3:
        P1, P2, D = act.shape
        act = act.reshape(P1 * P2, D)
    # reshape to (P,P,D)
    N, D = act.shape
    p = int(round(math.sqrt(N)))
    act = act.reshape(p, p, D)
    # center each neuron over inputs
    actc = act - act.reshape(-1, D).mean(axis=0)[None, None, :]
    # for each neuron compute 2D DFT coeffs and best k
    best_ks = []
    for di in range(D):
        A = actc[:, :, di]  # (p,p)
        # coeffs F[i,j] = sum_{a,b} A[a,b] basis[i,a] basis[j,b]
        F = basis_1d @ A @ basis_1d.T  # (p,p)
        total = float(np.sum(F ** 2))
        best_k = 1
        best_frac = -1.0
        for k in range(1, N_FREQS + 1):
            idxs = [0, 2 * k - 1, 2 * k]
            energy = float(np.sum(F[np.ix_(idxs, idxs)] ** 2))
            frac = energy / (total + 1e-30)
            if frac > best_frac:
                best_frac = frac
                best_k = k
        best_ks.append(best_k)
    # distinct set with support counts
    from collections import Counter
    counts = Counter(best_ks)
    ks_sorted = sorted(counts.keys())
    return ks_sorted, dict(counts)


def gini_coefficient(vec):
    vec = np.asarray(vec, dtype=np.float64).ravel()
    vec = np.abs(vec)
    if vec.sum() == 0:
        return 0.0
    x = np.sort(vec)
    n = len(x)
    cum = np.arange(1, n + 1)
    g = np.sum((2 * cum - n - 1) * x) / (n * np.sum(x))
    return float(g)


def gini_of_weight(W, basis_1d=None):
    """W: (d_model, P) (excluding = column already). Returns gini."""
    if basis_1d is None:
        basis_1d = fourier_basis_1d()
    W = np.asarray(W, dtype=np.float64)  # (d_model, P)
    # transform over token axis: basis (P,P) @ W.T -> (P, d_model)
    coeffs = basis_1d @ W.T  # (P, d_model)
    # L2 norm over model dim per component
    x = np.linalg.norm(coeffs, axis=1)  # (P,)
    return gini_coefficient(x)


def sum_squared_weights(state_dict):
    total = 0.0
    for v in state_dict.values():
        t = torch.as_tensor(v)
        total += float((t.float() ** 2).sum().item())
    return total
