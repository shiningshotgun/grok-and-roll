"""Analysis from saved snapshots: progress measures + ablation."""

import json
import os
import glob

import numpy as np
import torch

from src.model import Transformer, P, D_VOCAB, N_CTX, EQ_TOKEN
from src.fourier import (
    fourier_basis_1d,
    cos_xpy,
    sin_xpy,
    project_component,
    restricted_loss,
    excluded_loss,
    cross_entropy,
    select_key_frequencies,
    gini_of_weight,
    sum_squared_weights,
)


def load_model(epoch, device="cpu"):
    m = Transformer().to(device)
    sd = torch.load(f"snapshots/epoch_{epoch:06d}.pt", map_location=device, weights_only=True)
    m.load_state_dict(sd)
    m.eval()
    return m


def all_inputs(device):
    pairs = [(a, b) for a in range(P) for b in range(P)]
    xs = torch.tensor([[a, b, EQ_TOKEN] for (a, b) in pairs], dtype=torch.long, device=device)
    ys = torch.tensor([(a + b) % P for (a, b) in pairs], dtype=torch.long, device=device)
    return xs, ys, pairs


def logits_grid(model, device):
    xs, ys, pairs = all_inputs(device)
    with torch.no_grad():
        logits = model(xs)[:, -1, :P]  # (P*P, P) drop =
    L = logits.detach().cpu().numpy().astype(np.float64)  # (P*P, P)
    return L, ys.cpu().numpy()


def mlp_post_act_final(model, device):
    xs, ys, pairs = all_inputs(device)
    with torch.no_grad():
        h = model.mlp_post_activation(xs)  # (P*P, D_MLP)
    return h.detach().cpu().numpy()


def run(device="cpu"):
    basis = fourier_basis_1d()
    xs, ys, pairs = all_inputs(device)
    labels = np.array([(a + b) % P for (a, b) in pairs], dtype=np.int64)

    # select K from final snapshot
    final_epochs = sorted(int(p.split("epoch_")[1].split(".pt")[0]) for p in glob.glob("snapshots/epoch_*.pt"))
    final_ep = final_epochs[-1]
    mfinal = load_model(final_ep, device)
    h = mlp_post_act_final(mfinal, device)
    K, counts = select_key_frequencies(h, basis)
    # support counts already in counts; build support dict for K
    support = {k: int(counts.get(k, 0)) for k in K}

    # precompute projection directions per k
    dirs = {}
    for k in range(1, 57):
        dirs[k] = (cos_xpy(k), sin_xpy(k))

    # components per k on final? Need per-snapshot L. Compute per snapshot.
    records = []
    for ep in final_epochs:
        model = load_model(ep, device)
        L, _ = logits_grid(model, device)
        N = P * P
        # T for K
        T = np.zeros_like(L)
        comps_K = {}
        for k in K:
            cc = project_component(L, dirs[k][0])
            sc = project_component(L, dirs[k][1])
            T = T + cc + sc
            comps_K[k] = (cc, sc)
        rl = restricted_loss(L, T, labels)
        el = excluded_loss(L, T, labels)
        # gini
        sd = model.state_dict()
        W_E = sd["W_E"].detach().cpu().numpy()[:, :P]  # (d_model, P)
        W_U = sd["W_U"].detach().cpu().numpy()[:, :P]
        gE = gini_of_weight(W_E, basis)
        gU = gini_of_weight(W_U, basis)
        ssq = sum_squared_weights(sd)
        records.append({
            "epoch": ep,
            "restricted_loss": rl,
            "excluded_loss": el,
            "gini_W_E": gE,
            "gini_W_U": gU,
            "sum_sq_weights": ssq,
        })

    os.makedirs("records", exist_ok=True)
    with open("records/progress_measures.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # ablation on final model
    L, _ = logits_grid(mfinal, device)
    # unablated
    unablated = cross_entropy(L, labels)
    T = np.zeros_like(L)
    for k in K:
        T = T + project_component(L, dirs[k][0]) + project_component(L, dirs[k][1])
    rl = restricted_loss(L, T, labels)
    el = excluded_loss(L, T, labels)
    comps_all = {}
    for k in range(1, 57):
        comps_all[k] = (project_component(L, dirs[k][0]), project_component(L, dirs[k][1]))
    # per-freq exclusion for k in K
    per_freq = {}
    for k in K:
        ablated = L - comps_all[k][0] - comps_all[k][1]
        per_freq[k] = cross_entropy(ablated, labels)
    # non-key: lowest non-key frequencies
    nonkey = [k for k in range(1, 57) if k not in K][:5]
    nonkey_excl = {}
    for k in nonkey:
        ablated = L - comps_all[k][0] - comps_all[k][1]
        nonkey_excl[k] = cross_entropy(ablated, labels)

    ablation = {
        "unablated_loss": unablated,
        "restricted_loss": rl,
        "excluded_loss": el,
        "key_freqs": K,
        "per_key_exclusion": per_freq,
        "nonkey_exclusion": nonkey_excl,
    }
    with open("records/ablation.jsonl", "w") as f:
        f.write(json.dumps(ablation) + "\n")

    # save key freqs info
    with open("records/key_freqs.json", "w") as f:
        json.dump({"K": K, "support": support}, f, indent=2)

    return K, support


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    K, support = run(device=args.device)
    print("K", K)
    print("support", support)
