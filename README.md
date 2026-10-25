# grok-and-roll

A 1-layer transformer trained on modular addition (a + b mod 113) memorizes the 30% training split almost immediately, then - long after train accuracy hits 100% while test accuracy stays near chance - it suddenly generalizes. This run reproduces that grokking phenomenon and shows, using the Fourier-basis analysis of Nanda, Chan, Lieberum, Smith & Steinhardt (ICLR 2023), that the generalizing circuit (a Fourier-multiplication algorithm over key frequencies) forms gradually during the memorization plateau and is then cleaned up by weight decay: the network's Fourier content becomes sparse (Gini rises), the norm of the weights drops, and ablating any key frequency damages the final model.

## Sources

- Paper: https://arxiv.org/abs/2301.05217 (ICLR 2023) - specifications for the architecture, optimizer, split, progress measures, and ablation protocol were taken from its text.
- Author walkthrough: https://www.neelnanda.io/grokking-paper - used as corroboration of the paper's narrative and phase description.
- Authors' code: https://github.com/mechanistic-interpretability-grokking/progress-measures-paper - consulted as read-only provenance for configuration details. It carries NO LICENSE file, so no code or checkpoints were copied or downloaded; everything here is independently reimplemented from the written specifications.

## Scope compared with the paper

- Mainline modular-addition study only: training, logit-space Fourier analysis, progress measures, and frequency ablation. (OURS: nothing else was run.)
- One seed (OURS) versus the paper's five-seed Figure 2.
- Neuron/activation-space interventions of Section 4.4 are excluded (OURS).
- Non-modular tasks, no-weight-decay/slingshot analysis, and robustness sweeps are excluded (OURS).
- Device: MPS (OURS; see environment.json) versus the authors' cuda.
- wandb logging replaced with local JSONL records (OURS); metrics cadence densified to every 100 epochs (OURS) from the authors' every-1000.

## Settings

| Setting | Value | Source |
|---|---|---|
| p, d_vocab, n_ctx | 113, 114 ("=" token 113), 3 | verified (paper / authors' code) |
| Architecture | 1-layer ReLU transformer, d_model 128, 4 heads (d_head 32), d_mlp 512, no LayerNorm, untied embed/unembed, learned positional embeddings | verified |
| Init scales | W_E randn(128,114)/sqrt(128); W_pos randn(3,128)/sqrt(128); W_K/Q/V randn(4,32,128)/sqrt(128); W_O randn(128,128)/sqrt(128); W_in randn(512,128)/sqrt(128) with b_in=0; W_out randn(128,512)/sqrt(128) with b_out=0; W_U randn(128,114)/sqrt(114) | verified (authors' constructors) |
| Attention | per-head q.k scores, causal mask via -1e10 upper-triangle penalty, 1/sqrt(d_head) scaling after masking, softmax, concat + W_O; residuals around attention and MLP; logits read at the "=" position | verified |
| Optimizer | AdamW on ALL parameters, lr 1e-3, betas (0.9, 0.98), weight_decay 1.0, eps 1e-8 (PyTorch default; authors' code does not set it) | verified |
| Warmup | LambdaLR, multiplier min((step+1)/10, 1) over the first 10 steps, one optimizer step per epoch (full batch) | verified (index convention OURS) |
| Training loss | cross-entropy over all 114 logits, full-batch mean, log_softmax in float32 | verified |
| Reported loss / accuracy | "=" logit column dropped, 113 kept logits; accuracy = argmax over the 113 kept logits | verified for loss; accuracy convention OURS |
| Epochs | 50000 | verified |
| Split | all 113^2 pairs shuffled with python random.seed(0); first 30% train | verified |
| Init seed | torch.manual_seed(0) before construction | OURS |
| Device | MPS (chosen by 300-epoch pilot: CPU 15.51 ep/s, MPS 15.72 ep/s) | OURS |
| Snapshot schedule | state_dict only, at epochs 0..1000 step 100 and every 1000 from 2000 to 50000 (60 snapshots) | OURS |
| Metrics cadence | every 100 epochs | OURS |
| Gini | 1D Fourier transform over the token axis of W_E and W_U (113 token rows, "=" row excluded), L2 norm over the model dimension per real-basis component, DC included, standard Gini formula G = sum_i (2i - n - 1) x_i / (n * sum_i x_i) | OURS where the paper is silent (DC inclusion, real-basis component count) |
| Logging | local records/*.jsonl; snapshots/ untracked | OURS |
| Seed count | 1 | OURS |

## Reproduce

```bash
# Stage 0-1: scaffold, implementation, unit tests
python3 -m unittest discover -s tests -v
# Stage 2: pilots (device choice + determinism diagnostic)
/usr/bin/time -l python3 -m src.train --device cpu --epochs 300 > logs/pilot_cpu.log 2>&1
/usr/bin/time -l python3 -m src.train --device mps --epochs 300 > logs/pilot_mps.log 2>&1
# full run
/usr/bin/time -l python3 -m src.train --device mps --epochs 50000 > logs/run.log 2>&1
# Stage 3-4: analysis and figures
python3 -m src.analyze --device cpu
python3 -m src.plot_figures
```

## Results (from records only)

- Final metrics (records/metrics.jsonl, epoch 50000): train_loss 7.66e-08, test_loss 2.88e-04, train_acc 1.0000, test_acc 0.99989. The run grokked: train accuracy reached ~1.0 by epoch 200; test accuracy crossed 0.99 at epoch 9600 and was 1.0000 by epoch 10000.
- Key frequencies (records/key_freqs.json, from the final snapshot by the neuron-variance method): K = [14, 15, 35, 42, 52] with per-neuron support {14: 60, 15: 115, 35: 101, 42: 127, 52: 109}.
- Final-model losses (records/ablation.jsonl): unablated 2.02e-04; restricted 0.0; excluded 8.78. Single-frequency exclusion: key frequencies give 0.0085-0.228; lowest five non-key frequencies (1-5) all stay at ~0.00020.
- Progress measures (records/progress_measures.jsonl): excluded loss rises from 4.76 (epoch 0) to 8.78 (epoch 50000) while train/test loss stay flat at the plateau; Gini(W_E) rises 0.035 -> 0.62, Gini(W_U) 0.032 -> 0.67; sum of squared weights falls 1783 -> 973 (peaking ~3658 during memorization).
- Runtime (logs/run.log): 2771.72 s real; peak RSS 424427520 bytes.

## Honest limits

- Single seed: onset epochs and key frequencies could differ across seeds; the paper reports five.
- MPS numerics differ from cuda and from CPU; a 300-epoch CPU-vs-MPS fidelity comparison and a run-to-run hash check (mismatch - MPS nondeterminism) are recorded in records/environment.json.
- No direct comparison against the authors' code was performed (no license; independent reimplementation).
- Choices the paper is silent on (init seed, device, snapshot schedule, metrics cadence, accuracy convention, Gini DC/real-basis handling, logging) are labeled OURS above; the most likely-effect-bearing of these is the device (numerics only) and the init seed (which frequencies get selected could shift).
- Figures were validated programmatically (XML parse, viewBox, element count) but were not pixel-inspected by the agent; see the run report's VISUAL QA line.
