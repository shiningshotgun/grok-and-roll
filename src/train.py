"""Training loop for modular addition transformer."""

import json
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from src.model import (
    Transformer,
    make_model,
    P,
    D_VOCAB,
    N_CTX,
    EQ_TOKEN,
)

EPOCHS = 50000
LR = 1e-3
WD = 1.0
BETAS = (0.9, 0.98)


def make_data():
    pairs = [(a, b) for a in range(P) for b in range(P)]
    random.seed(0)
    random.shuffle(pairs)
    n_train = int(0.3 * len(pairs))
    train = pairs[:n_train]
    test = pairs[n_train:]
    return train, test


def tokens_of(a, b):
    return [a, b, EQ_TOKEN]


def to_tensors(pairs, device):
    xs = torch.tensor([[a, b, EQ_TOKEN] for (a, b) in pairs], dtype=torch.long, device=device)
    ys = torch.tensor([(a + b) % P for (a, b) in pairs], dtype=torch.long, device=device)
    return xs, ys


def reported_loss_acc(model, xs, ys, device):
    # drop "=" column (index 113), labels 0..112
    with torch.no_grad():
        logits = model(xs)  # (B,3,114)
        last = logits[:, -1, :]  # (B,114)
        kept = last[:, :P]  # (B,113)
        # CE in float32 via log_softmax
        logp = F.log_softmax(kept.float(), dim=-1)
        loss = -logp.gather(1, ys.view(-1, 1)).mean()
        pred = kept.argmax(dim=-1)
        acc = (pred == ys).float().mean()
    return float(loss.item()), float(acc.item())


def training_loss(model, xs, ys):
    # CE over all 114 logits
    logits = model(xs)[:, -1, :]  # (B,114)
    return F.cross_entropy(logits.float(), ys)


def snapshot_path(epoch):
    return os.path.join("snapshots", f"epoch_{epoch:06d}.pt")


def save_snapshot(model, epoch):
    torch.save(model.state_dict(), snapshot_path(epoch))


def main(device="cpu", epochs=EPOCHS):
    train_pairs, test_pairs = make_data()
    xs_train, ys_train = to_tensors(train_pairs, device)
    xs_test, ys_test = to_tensors(test_pairs, device)

    torch.manual_seed(0)
    model = Transformer().to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=BETAS, weight_decay=WD, eps=1e-8)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda step: min((step + 1) / 10.0, 1.0))

    os.makedirs("records", exist_ok=True)
    os.makedirs("snapshots", exist_ok=True)
    metrics_f = open("records/metrics.jsonl", "w")

    # snapshot schedule
    snap_epochs = set(list(range(0, 1001, 100)) + list(range(2000, 50001, 1000)))
    # epoch 0 snapshot before training
    if 0 in snap_epochs:
        save_snapshot(model, 0)

    for epoch in range(epochs + 1):
        if epoch > 0:
            opt.zero_grad()
            loss = training_loss(model, xs_train, ys_train)
            loss.backward()
            opt.step()
            sched.step()

        if epoch % 100 == 0:
            tr_loss, tr_acc = reported_loss_acc(model, xs_train, ys_train, device)
            te_loss, te_acc = reported_loss_acc(model, xs_test, ys_test, device)
            rec = {
                "epoch": epoch,
                "train_loss": tr_loss,
                "test_loss": te_loss,
                "train_acc": tr_acc,
                "test_acc": te_acc,
            }
            metrics_f.write(json.dumps(rec) + "\n")
            metrics_f.flush()

        if epoch in snap_epochs and epoch != 0:
            save_snapshot(model, epoch)

    metrics_f.close()
    return model


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    main(device=args.device, epochs=args.epochs)
