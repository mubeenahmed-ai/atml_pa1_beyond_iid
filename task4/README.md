# Task 4 – Open-Set Recognition (CIFAR-10 known, CIFAR-100 unknown)

## Setup
* CIFAR-10 stratified 90/10 train/val split (seed 6304, `data/make_splits.py`); checkpoints chosen by CIFAR-10
  validation accuracy; full CIFAR-10 test set for closed-set accuracy (CSA).
* Unknowns = fixed CIFAR-100 **test** classes: near = bus, pickup_truck, motorcycle, tractor, wolf, fox, leopard,
  camel; far = bottle, bowl, chair, clock, keyboard, mushroom, sunflower, wardrobe (800 images each).
  CIFAR-100 is imported only by `extract_outputs.py` / `evaluate_osr.py`, after every checkpoint, score and
  threshold rule is fixed.
* CIFAR ResNet-18 (`models/resnet_cifar.py`): 3×3 stride-1 stem, no max-pool, 32×32 inputs.
* Recipe: random crop (pad 4) + flip, SGD lr 0.1 / momentum 0.9 / wd 5e-4, cosine, batch 128, 100 epochs, seed 6304,
  fp16 autocast during training (all cached features/logits are extracted in fp32).

## Methods
| Model | Change vs. Vanilla |
|---|---|
| Vanilla | – |
| GCSC | `RandAugment(num_ops=2, magnitude=9)` after crop/flip |
| PROSER | init from Vanilla; +5 dummy classifiers; 50 epochs SGD lr 1e-3; classifier-placeholder loss (β=1) on one half of each batch, manifold-mixup data placeholders after layer2 (λ~Beta(2,2), different classes) trained to the dummy class (γ=0.1) on the other half |
| RPL (optional) | one learnable reciprocal point per class in the 512-d feature space; logits = ‖f−P_k‖²/d; CE + 0.1·MSE(d(f,P_y), R)/2 open-space term; score = −max_k d(f,P_k) |

## Scores (`scores/`, all computed from the same cached arrays)
MSP = 1 − max softmax; MLS = −max logit; Energy = −logsumexp(z); Mahalanobis = min_c (f−μ_c)ᵀΣ⁻¹(f−μ_c) with a shared
diagonal Σ (+1e-6) from unaugmented training features; PROSER placeholder = p(dummy) − max_k p(k) after appending
the max dummy logit.

## Evaluation (`evaluate_osr.py`)
AUROC for Known vs Near / Far / All; threshold τ = 95th percentile of the score on CIFAR-10 **validation**; report
test acceptance, near/far rejection and FPR@95TPR. Failure analysis lists the most confidently accepted near and
far unknowns under the Vanilla-MLS threshold.

## Reproduce
```bash
python task4/data/make_splits.py
for m in vanilla gcsc proser rpl; do python task4/train.py --config task4/configs/$m.yaml; done   # proser needs vanilla first
python task4/extract_outputs.py --models vanilla gcsc proser rpl
python task4/evaluate_osr.py --models vanilla gcsc proser rpl        # -> results/final/
```
