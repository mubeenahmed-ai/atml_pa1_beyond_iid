# Task 3 – Domain Generalisation on PACS (Sketch unseen)

Same protocol, splits, model, optimiser, budget, BN policy and seed as Task 2 (`configs/base.yaml` is identical
except for the results directory). **No Sketch image is loaded by any Task 3 training, selection or diagnostic
script**: all Task 3 methods have `needs_target=False`, so the shared trainer never constructs a target loader,
and `selection/source_validation.py` computes every source-side diagnostic without importing the target loader.
Sketch is read only by `evaluate_sketch.py`, run last.

## Methods (`methods/`)
| Method | Objective |
|---|---|
| ERM | Task 2 Source-only checkpoint copied unchanged (`configs/erm.yaml`) |
| DAN-DG | CE + (λ/3)·Σ_{e<e'} MMD²(F(X_e), F(X_e')) over the 3 source pairs, λ=1, same kernels as Task 2, median per pair |
| SAM | min_θ max_{‖ε‖≤ρ} L_ERM(θ+ε), ρ=0.05, non-adaptive, two passes per batch, frozen BN in both |

Controlled study: ρ ∈ {0.01, 0.05, 0.1} for SAM (`configs/sam_rho*.yaml`). Main comparison keeps ρ=0.05, λ=1.

## Diagnostics
* Source-domain separability: balanced frozen features from the three source validation sets, 70/30 split
  (seed 6304), multinomial logistic regression C=1 → held-out accuracy (chance 33.3%).
* Sharpness proxy: fixed validation batch of 32 images per source (seed 6304), eval mode,
  Δ = L(θ+ε) − L(θ) with ε = 0.05·∇L/‖∇L‖₂ (`evaluation/sharpness.py`).

## Reproduce
```bash
python task3/train.py --config task3/configs/erm.yaml          # copies the Task 2 checkpoint
python task3/train.py --config task3/configs/dan_dg.yaml
python task3/train.py --config task3/configs/sam.yaml
python task3/train.py --config task3/configs/sam_rho0.01.yaml && python task3/train.py --config task3/configs/sam_rho0.1.yaml
python task3/selection/source_validation.py --methods erm dan_dg sam sam_rho0.01 sam_rho0.1   # Sketch-free
python task3/evaluate_sketch.py --methods erm dan_dg sam                                       # loads Sketch
python task3/evaluate_sketch.py --methods sam_rho0.01 sam sam_rho0.1 --tag rho_study --baseline sam
```
