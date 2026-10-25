# Paper versus this study

This page compares this study with the original paper, so the scope of the results is not read wider than it is.

Paper: Nanda, Chan, Lieberum, Smith & Steinhardt, "Progress measures for grokking via mechanistic interpretability", https://arxiv.org/abs/2301.05217 (ICLR 2023; text read at https://arxiv.org/html/2301.05217v3). Section and figure references below are to that text.
Authors' code: https://github.com/mechanistic-interpretability-grokking/progress-measures-paper - unlicensed, consulted read-only; no code or checkpoints were copied.
This study: `README.md`, `records/` and `figures/` in this repository.

## Side by side

| Item | Paper | This study |
|---|---|---|
| Task | Modular addition `a + b mod 113` is the mainline study; the paper also studies other tasks and settings. | Modular addition mod 113 only. Nothing else was run. |
| Model | 1-layer ReLU transformer, d_model 128, 4 heads (d_head 32), d_mlp 512, no LayerNorm, untied embed/unembed, learned positional embeddings. | Same architecture and init scales, independently reimplemented from the paper's text and the authors' readable constructors. |
| Optimizer | AdamW on all parameters, lr 1e-3, betas (0.9, 0.98), weight decay 1.0, linear warmup over the first 10 steps, full batch, 50,000 epochs. | Same settings. eps is PyTorch's default 1e-8 (the authors' code does not set it). The warmup index convention differs by one step, disclosed in `README.md`. |
| Split | All 113^2 pairs shuffled, first 30% in training. | Same procedure and shuffle call (`random.seed(0)`), verified against the authors' code. |
| Seeds | Figure 2 averages five seeds. The mechanistic analysis in Sections 4.1-4.3 dissects individual trained models. | One seed (`torch.manual_seed(0)`; the paper's text is silent on the init seed, so this choice is ours). One run, not an average. |
| Hardware | CUDA (authors' code default device). | MPS on an M1 MacBook Air. MPS is not bitwise deterministic; two same-seed 300-epoch pilots diverged, recorded in `records/environment.json`. |
| Logging | wandb, metrics every 1,000 epochs. | Local JSONL, metrics every 100 epochs (`records/`). |
| Key frequencies | Five key frequencies in the analyzed model: k in {14, 35, 41, 42, 52} (Section 4.1). | Five key frequencies in this model: {14, 15, 35, 42, 52} (`records/key_freqs.json`). Same count, four of five identical; the specific set is a property of the individual trained model. |
| Progress measures | Figure 3 delineates memorization, circuit formation and cleanup: restricted loss declines during the plateau, before test loss declines, with an inflection when grokking begins; excluded loss increases during circuit formation while train and test loss stay flat; Gini coefficients of the embedding/unembedding Fourier norms rise during cleanup; the sum of squared weights falls. | Same four measures, same definitions where the paper states them; Gini conventions (DC inclusion, real-basis component count) are ours where the paper is silent (`README.md`). The same phase directions appeared in this run: restricted loss fell through the plateau, excluded loss rose during circuit formation (4.76 to 8.78 overall), Gini rose during cleanup, weight norm fell. |
| Ablation | Ablating any key frequency significantly raises loss; ablating non-key frequencies does not (Figure 6). | Same qualitative pattern on this model: restricted 0.0 versus unablated 2.02e-04; excluded 8.78; per-key exclusion 0.0085-0.228; the five lowest non-key exclusions all stay near 2.0e-04 (`records/ablation.jsonl`). |
| Excluded scope | Neuron and activation-space interventions (Section 4.4), non-modular tasks, no-weight-decay and slingshot analysis, robustness sweeps. | Not run here. |

## Where the numbers must not be mixed

- The paper's training curves are five-seed averages; every curve in this study is one seed. Onset epochs, spike sizes and post-cleanup oscillations are properties of a single trajectory and must not be compared point-for-point with the paper's averaged curves.
- The key-frequency sets are model-specific. Four of five overlapping is an observation about these two models, not a target the run aimed at.
- The paper's exact loss values come from its own model and device. Our restricted loss of exactly 0.0 is float32 running out of room, not a literally perfect model, and is not comparable to the paper's restricted-loss values beyond "very small".
- Runtime (2,771.72 s) and throughput here describe MPS on this Mac. The paper's hardware differs; no timing comparison is made.
- No direct comparison against the authors' code was performed (no license; independent reimplementation). "Verified" rows above mean verified against the paper's text and the authors' publicly readable constructors, not against executed reference outputs.
