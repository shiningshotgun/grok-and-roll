"""Generate figure SVGs."""

import json
import os

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


def load_metrics():
    rows = []
    with open("records/metrics.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_progress():
    rows = []
    with open("records/progress_measures.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_ablation():
    with open("records/ablation.jsonl") as f:
        return json.loads(f.readline())


def fig1():
    rows = load_metrics()
    epochs = [r["epoch"] for r in rows]
    tr_loss = [r["train_loss"] for r in rows]
    te_loss = [r["test_loss"] for r in rows]
    tr_acc = [r["train_acc"] for r in rows]
    te_acc = [r["test_acc"] for r in rows]
    if HAS_MPL:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(epochs, tr_loss, label="train loss")
        ax1.plot(epochs, te_loss, label="test loss")
        ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.set_title("Loss (one seed)"); ax1.legend()
        ax2.plot(epochs, tr_acc, label="train acc")
        ax2.plot(epochs, te_acc, label="test acc")
        ax2.set_xlabel("epoch"); ax2.set_ylabel("acc"); ax2.set_title("Accuracy (one seed)"); ax2.legend()
        fig.tight_layout()
        fig.savefig("figures/train_test_curves.svg")
        plt.close(fig)
    else:
        # minimal SVG
        write_minimal_svg("figures/train_test_curves.svg", "train/test curves")


def fig2():
    rows = load_progress()
    epochs = [r["epoch"] for r in rows]
    if HAS_MPL:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes[0, 0].plot(epochs, [r["excluded_loss"] for r in rows], label="excluded loss")
        axes[0, 0].set_title("Excluded loss"); axes[0, 0].set_xlabel("epoch")
        axes[0, 1].plot(epochs, [r["restricted_loss"] for r in rows], label="restricted loss")
        axes[0, 1].set_title("Restricted loss"); axes[0, 1].set_xlabel("epoch")
        axes[1, 0].plot(epochs, [r["gini_W_E"] for r in rows], label="gini W_E")
        axes[1, 0].plot(epochs, [r["gini_W_U"] for r in rows], label="gini W_U")
        axes[1, 0].set_title("Gini"); axes[1, 0].set_xlabel("epoch"); axes[1, 0].legend()
        axes[1, 1].plot(epochs, [r["sum_sq_weights"] for r in rows], label="sum sq weights")
        axes[1, 1].set_title("Sum squared weights"); axes[1, 1].set_xlabel("epoch")
        fig.tight_layout()
        fig.savefig("figures/progress_measures.svg")
        plt.close(fig)
    else:
        write_minimal_svg("figures/progress_measures.svg", "progress measures")


def fig3():
    ab = load_ablation()
    K = ab["key_freqs"]
    per_key = ab["per_key_exclusion"]
    nonkey = ab["nonkey_exclusion"]
    if HAS_MPL:
        K = [int(k) for k in K]
        pk = {int(k): v for k, v in per_key.items()}
        nk = {int(k): v for k, v in nonkey.items()}
        nonkey = sorted(nk.keys())
        labels = [f"k={k}" for k in K] + [f"nk={k}" for k in nonkey]
        vals = [pk[k] for k in K] + [nk[k] for k in nonkey]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, vals)
        ax.axhline(ab["unablated_loss"], color="r", linestyle="--", label="unablated")
        ax.axhline(ab["restricted_loss"], color="g", linestyle="--", label="restricted")
        ax.axhline(ab["excluded_loss"], color="b", linestyle="--", label="excluded")
        ax.set_ylabel("loss"); ax.set_title("Frequency ablation"); ax.legend()
        fig.tight_layout()
        fig.savefig("figures/frequency_ablation.svg")
        plt.close(fig)
    else:
        write_minimal_svg("figures/frequency_ablation.svg", "ablation")


def write_minimal_svg(path, title):
    with open(path, "w") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text x="10" y="50">{title}</text></svg>')


def main():
    os.makedirs("figures", exist_ok=True)
    fig1()
    fig2()
    fig3()
    print("done", HAS_MPL)


if __name__ == "__main__":
    main()
