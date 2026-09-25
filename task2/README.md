# Task 2 – Unsupervised Domain Adaptation on PACS (Photo, Art, Cartoon → Sketch)

## Protocol (shared with Task 3, see `../shared/`)
* PACS from the HuggingFace mirror `flwrlabs/pacs` (single parquet, 9,991 images, 7 classes, 4 domains).
  Copy it to `data/pacs/pacs.parquet` (see top-level README).
* `shared/make_pacs_splits.py` → `shared/splits/pacs_sketch_seed6304.json`: stratified 80/20 train/val split
  inside each source domain, seed 6304. All 3,929 Sketch images form the unlabelled adaptation set.
* ResNet-18 (`IMAGENET1K_V1`) + 7-way linear head, full fine-tuning. Resize 256, random 224 crop + flip (train),
  centre crop (eval). **BatchNorm running statistics frozen at ImageNet values**, γ/β trainable
  (`shared/pacs_protocol.py::train_mode_frozen_bn`).
* AdamW lr 1e-4, wd 1e-4, ≤30 source epochs (one epoch = ⌈N_source_train / 24⌉ = 203 steps), early stopping
  after 5 epochs without improvement of **mean source-validation macro-F1**, seed 6304.
* Every step: 8 images from each source domain (24) + 24 unlabelled Sketch images (methods with alignment).
* Sketch labels are read **only** in `evaluate_final.py`; the training loader returns label −1 for Sketch.

## Methods (`methods/`)
| Method | Loss | Notes |
|---|---|---|
| Source-only | CE on source | also the Task 3 ERM baseline (checkpoint reused unchanged) |
| DAN | CE + λ·MMD²(f_s, f_t), λ=1 | 3 RBF kernels, bandwidth = {0.5,1,2}× median pairwise squared distance of the combined batch, **unbiased** U-statistic (`shared/mmd.py`) |
| DANN | CE + domain CE via GRL | discriminator 512→256→ReLU→Dropout(0.5)→2, α(p)=2/(1+e^{−10p})−1, unit weight |
| CDAN | CE + domain CE on vec(f ⊗ p) via GRL | 3584-d input, same discriminator width/schedule/weight; no entropy conditioning, no detach |

Controlled study: λ_MMD ∈ {0.1, 1, 10} for DAN (`configs/dan_lambda*.yaml`).

### MMD estimator (DAN / DAN-DG)
The first DAN-DG and DAN(λ=10) runs used the biased V-statistic and collapsed at step ≈10: with 8 (24) samples per
domain the diagonal k(x,x)=1 terms can only vanish when all features coincide, and AdamW drives the ReLU features
to exactly zero (90% dead units, CE = ln 7). These runs are kept as `*_biased` for the report. All reported DAN /
DAN-DG results use the unbiased U-statistic (`shared/mmd.py`), the standard MMD² estimator, with the same kernels.

### Documented deviation: discriminator learning rate (DANN / CDAN)
With every parameter at the shared AdamW lr 1e-4 the adversarial runs are numerically unstable: once α(p)
reaches ≈0.1 the backbone (11M parameters, sign-like Adam steps, **no BatchNorm renormalisation because the
running statistics are frozen**) drives the feature norm from ≈20 to >10⁶ in a few dozen steps, the domain loss
reaches 10³–10⁴ and the classifier collapses (`results/dann_lr1/`, `results/cdan_lr1/`; the run is kept as
evidence). Gradient-norm clipping does not help because Adam normalises the gradient anyway. Giving the newly
initialised discriminator a 10× learning rate (1e-3, `method.extra_lr_mult: 10`) lets it keep up with the
backbone; the domain loss then stays near ln 2 and features stay bounded. The backbone, head, augmentation,
sampling, schedule, budget and seed are unchanged, and the same multiplier is used for DANN and CDAN.

## Reproduce
```bash
python shared/make_pacs_splits.py
for m in source_only dan dann cdan dan_lambda0.1 dan_lambda10; do python task2/train.py --config task2/configs/$m.yaml; done
python task2/evaluate_final.py --methods source_only dan dann cdan                    # -> results/final_main/
python task2/evaluate_final.py --methods dan_lambda0.1 dan dan_lambda10 --tag lambda_study --baseline dan
```

## Outputs (`results/<method>/`)
`curves_step.csv` (classification / MMD / domain losses, discriminator accuracy, GRL α), `curves_epoch.csv`
(per-domain validation accuracy & macro-F1), `train_summary.json`, `best.pt` (git-ignored).
`results/final_main/`: `table_main.csv` (per-domain source val, mean, target acc/F1, Δ vs Source-only, domain
separability), `per_class_target_acc.csv`, `final_results.json` (confusions), `training_curves.png`,
`discriminator_acc.png`, `per_class_delta.png`.

Domain separability: frozen features of all source-validation images vs. an equal-size seed-6304 sample of
Sketch features, 70/30 split, `LogisticRegression(C=1, class_weight="balanced")`; held-out accuracy.
