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

try:
    import PIL  # noqa: F401

    HAS_PIL = True
except Exception:
    HAS_PIL = False


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
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss")
        ax1.set_title("Loss (one seed)")
        ax1.legend()
        ax2.plot(epochs, tr_acc, label="train acc")
        ax2.plot(epochs, te_acc, label="test acc")
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("acc")
        ax2.set_title("Accuracy (one seed)")
        ax2.legend()
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
        axes[0, 0].plot(
            epochs, [r["excluded_loss"] for r in rows], label="excluded loss"
        )
        axes[0, 0].set_title("Excluded loss")
        axes[0, 0].set_xlabel("epoch")
        axes[0, 1].plot(
            epochs, [r["restricted_loss"] for r in rows], label="restricted loss"
        )
        axes[0, 1].set_title("Restricted loss")
        axes[0, 1].set_xlabel("epoch")
        axes[1, 0].plot(epochs, [r["gini_W_E"] for r in rows], label="gini W_E")
        axes[1, 0].plot(epochs, [r["gini_W_U"] for r in rows], label="gini W_U")
        axes[1, 0].set_title("Gini")
        axes[1, 0].set_xlabel("epoch")
        axes[1, 0].legend()
        axes[1, 1].plot(
            epochs, [r["sum_sq_weights"] for r in rows], label="sum sq weights"
        )
        axes[1, 1].set_title("Sum squared weights")
        axes[1, 1].set_xlabel("epoch")
        fig.tight_layout()
        fig.savefig("figures/progress_measures.svg")
        plt.close(fig)
    else:
        write_minimal_svg("figures/progress_measures.svg", "progress measures")


ABLATION_FLOOR = 1e-6  # axis floor for the log-scale loss axis


def _ablation_figure(ab):
    """Ablation bars on a log-scale loss axis, with the exact-zero restricted loss handled explicitly."""
    K = [int(k) for k in ab["key_freqs"]]
    pk = {int(k): v for k, v in ab["per_key_exclusion"].items()}
    nk = {int(k): v for k, v in ab["nonkey_exclusion"].items()}
    nonkey = sorted(nk.keys())
    labels = [f"k={k}" for k in K] + [f"nk={k}" for k in nonkey]
    vals = [pk[k] for k in K] + [nk[k] for k in nonkey]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(labels, vals)
    ax.set_yscale("log")
    ax.set_ylim(bottom=ABLATION_FLOOR, top=30)
    ax.axhline(
        ab["unablated_loss"],
        color="r",
        linestyle="--",
        label=f"unablated {ab['unablated_loss']:.2e}",
    )
    # restricted loss is exactly 0.0: a log axis cannot show 0, so draw it at
    # the axis floor in a distinct style and label it explicitly.
    ax.axhline(
        ABLATION_FLOOR,
        color="g",
        linestyle=":",
        linewidth=3,
        label="restricted: exactly 0.0 (float32 floor)",
    )
    ax.axhline(
        ab["excluded_loss"],
        color="b",
        linestyle="--",
        label=f"excluded {ab['excluded_loss']:.2f}",
    )
    n = len(labels)
    ax.annotate(
        "restricted loss is exactly 0.0 (float32 floor),\n"
        "not a small positive bar: drawn at the axis floor",
        xy=((n - 1) / 2, ABLATION_FLOOR * 1.4),
        xytext=((n - 1) / 2, 2.0),
        ha="center",
        va="top",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="g", lw=1.2),
    )
    ax.set_ylabel("loss (log scale)")
    ax.set_title("Frequency ablation (log scale)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    fig.tight_layout()
    return fig


def fig3(path="figures/frequency_ablation.svg"):
    ab = load_ablation()
    if HAS_MPL:
        fig = _ablation_figure(ab)
        fig.savefig(path)
        plt.close(fig)
    else:
        write_minimal_svg(path, "ablation")


def _grok_window(rows):
    """Epochs where test accuracy first passes 0.5 and 0.99, from the metrics records."""
    start = next(r["epoch"] for r in rows if r["test_acc"] > 0.5)
    end = next(r["epoch"] for r in rows if r["test_acc"] > 0.99)
    return start, end


def fig4(path="figures/grokking_animation.gif"):
    """Hero animation from records/metrics.jsonl: loss and accuracy drawing over epochs."""
    rows = load_metrics()
    if not (HAS_MPL and HAS_PIL):
        fig4_static()
        return
    from matplotlib.animation import PillowWriter

    epochs = [r["epoch"] for r in rows]
    tr_loss = [r["train_loss"] for r in rows]
    te_loss = [r["test_loss"] for r in rows]
    tr_acc = [r["train_acc"] for r in rows]
    te_acc = [r["test_acc"] for r in rows]
    grok_start, grok_end = _grok_window(rows)
    stride = max(1, len(rows) // 250)
    idxs = list(range(0, len(rows), stride))
    if idxs[-1] != len(rows) - 1:
        idxs.append(len(rows) - 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.suptitle("grok-and-roll: a + b mod 113 - one seed, 50,000 epochs", fontsize=11)
    loss_top = max(max(tr_loss), max(te_loss)) * 1.25
    loss_lo = min(min(tr_loss), min(te_loss)) * 0.5
    for ax, ylab in ((ax1, "loss (log scale)"), (ax2, "accuracy")):
        ax.set_xlim(0, epochs[-1])
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylab)
    ax1.set_yscale("log")
    ax1.set_ylim(loss_lo, loss_top)
    ax2.set_ylim(-0.03, 1.08)
    ax1.plot([], [], color="C0", lw=1.6, label="train")
    ax1.plot([], [], color="C1", lw=1.6, label="test")
    ax2.plot([], [], color="C0", lw=1.6, label="train")
    ax2.plot([], [], color="C1", lw=1.6, label="test")
    ax1.legend(loc="upper right", fontsize=8)
    ax2.legend(loc="lower right", fontsize=8)
    epoch_text = fig.text(0.995, 0.965, "", ha="right", va="top", fontsize=10)
    span_drawn = False
    markers = []
    writer = PillowWriter(
        fps=25, metadata={"Title": "grokking: train and test loss and accuracy"}
    )
    prev = 0
    with writer.saving(fig, path, dpi=90):
        for i in idxs:
            ax1.plot(epochs[prev : i + 1], tr_loss[prev : i + 1], color="C0", lw=1.6)
            ax1.plot(epochs[prev : i + 1], te_loss[prev : i + 1], color="C1", lw=1.6)
            ax2.plot(epochs[prev : i + 1], tr_acc[prev : i + 1], color="C0", lw=1.6)
            ax2.plot(epochs[prev : i + 1], te_acc[prev : i + 1], color="C1", lw=1.6)
            prev = i
            if not span_drawn and epochs[i] >= grok_start:
                for ax in (ax1, ax2):
                    ax.axvspan(grok_start, grok_end, color="C1", alpha=0.15, zorder=0)
                ax2.text(
                    (grok_start + grok_end) / 2,
                    0.5,
                    "grokking",
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="C1",
                )
                span_drawn = True
            for h in markers:
                h.remove()
            markers = [
                ax1.plot(epochs[i], tr_loss[i], "o", color="C0", ms=4)[0],
                ax1.plot(epochs[i], te_loss[i], "o", color="C1", ms=4)[0],
                ax2.plot(epochs[i], tr_acc[i], "o", color="C0", ms=4)[0],
                ax2.plot(epochs[i], te_acc[i], "o", color="C1", ms=4)[0],
            ]
            epoch_text.set_text(f"epoch {epochs[i]:,}")
            writer.grab_frame()
    plt.close(fig)


def fig4_static(path="figures/grokking_phases.svg"):
    """Static hero: the four phases as panels, for environments without Pillow."""
    rows = load_metrics()
    if not HAS_MPL:
        write_minimal_svg(path, "grokking phases")
        return
    epochs = [r["epoch"] for r in rows]
    train_done = next(r["epoch"] for r in rows if r["train_acc"] >= 1.0)
    grok_start, grok_end = _grok_window(rows)
    phases = [
        ("memorization", 0, train_done),
        ("plateau", train_done, grok_start),
        ("grokking", grok_start, grok_end),
        ("cleanup", grok_end, epochs[-1]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for ax, (name, lo, hi) in zip(axes.flat, phases):
        ax.plot(epochs, [r["train_loss"] for r in rows], color="C0", lw=1.2)
        ax.plot(epochs, [r["test_loss"] for r in rows], color="C1", lw=1.2)
        ax.set_yscale("log")
        ax.set_xlim(0, epochs[-1])
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss (log scale)")
        ax.axvspan(lo, hi, color="C1", alpha=0.15, zorder=0)
        ax.set_title(name, fontsize=10)
        ax2 = ax.twinx()
        ax2.plot(
            epochs, [r["train_acc"] for r in rows], color="C0", lw=1.0, linestyle=":"
        )
        ax2.plot(
            epochs, [r["test_acc"] for r in rows], color="C1", lw=1.0, linestyle=":"
        )
        ax2.set_ylim(-0.03, 1.08)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_minimal_svg(path, title):
    with open(path, "w") as f:
        f.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text x="10" y="50">{title}</text></svg>'
        )


def main():
    os.makedirs("figures", exist_ok=True)
    fig1()
    fig2()
    fig3()
    fig4()
    print("done", HAS_MPL, HAS_PIL)


if __name__ == "__main__":
    main()
