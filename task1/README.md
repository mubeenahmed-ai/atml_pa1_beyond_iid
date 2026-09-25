# Task 1 – Inductive Biases and Feature Representations (STL-10)

Frozen ResNet-50 (`IMAGENET1K_V2`), ViT-B/16 (`IMAGENET1K_V1`) and OpenCLIP ViT-B/32
(`openai`) are compared under controlled interventions on STL-10.

## Reproduce
```bash
conda activate atml-pa1            # see ../requirements.txt
python task1/data/make_subset.py         # stratified 80/20 split + 500-image class-balanced eval subset (seed 6304)
python task1/data/make_cue_conflicts.py  # AdaIN cue conflicts + visual rejection rule (no model involved)
python task1/scripts/run_task1.py        # features, linear heads, all interventions, metrics, figures
```
All settings live in `configs/task1.yaml`. Features are cached in `results/features/`
(git-ignored) so the evaluation stage can be re-run without recomputation.

## Data
STL-10 labelled train (5,000) / test (8,000) partitions are read from the HuggingFace
parquet mirror `tanganke/stl10` (the official 2.6 GB tarball also contains 100k unlabelled
images that are not needed). Class order was verified visually to match torchvision:
`airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck`.
Images are bicubically resized 96→224 to a common RGB canvas; **every intervention is applied
to this canvas before each model's own normalisation**, so all models see identical pixels.

## Design choices (stated before evaluation)
| Choice | Setting | Hypothesis | Metric |
|---|---|---|---|
| Extra colour intervention | fixed hue rotation +120° | ImageNet-supervised models rely partly on natural colour statistics (e.g. sky/sea blue, fur brown), so *changing* colour should hurt more than *removing* it for texture-reliant models; CLIP's language supervision may make it less colour-dependent | Δtop-1, prediction consistency |
| Cue-conflict pairs | {airplane,cat} {car,dog} {ship,horse} {truck,bird} {deer,monkey} {bird,cat}, both directions, AdaIN α=1 | ResNet-50 more texture-driven than ViT-B/16 and CLIP (Geirhos et al.), but coverage may be low because stylised STL-10 images are out-of-distribution | shape bias % and coverage % |
| Rejection rule | pixel std ≥ 0.05, Sobel-edge corr(content, stylised) ≥ 0.30, colour statistics closer to style than content | – | accepted / rejected counts |
| Visualisation | t-SNE, perplexity 30, PCA init, cosine metric, 1000 iterations, seed 6304, one fit per (backbone, intervention) on clean+transformed features | patch shuffling and cue conflict move features far from clean clusters; grayscale and translation do not | visual inspection + cosine stability |

## Outputs (`results/`)
* `splits_seed6304.json` – train/val indices, evaluation image identifiers and labels
* `head_training.json`, `head_history_*.csv` – linear-head training curves
* `table_interventions.csv` – clean / grayscale / hue / patch-shuffle accuracy, macro-F1, mean max confidence, Δ, consistency
* `translation_curve.{csv,json}` – accuracy and consistency vs displacement (per direction in JSON)
* `cue_conflict_manifest.csv`, `cue_conflict_summary.json` – every generated conflict with rejection metrics
* `shape_bias.csv`, `cue_conflict_predictions.csv` – shape/texture/other counts, shape bias, coverage, per-image decisions
* `representation_stability.csv`, `prediction_vs_representation.csv` – cosine stability and its relation to prediction changes
* `figures/` – bar chart, translation curve, stability chart, t-SNE grid; `images/examples/` – intervention & triplet examples

## External code
`models/adain.py` reuses the network definition of naoto0804/pytorch-AdaIN (MIT) and its
released weights (`vgg_normalised.pth`, `decoder.pth`, obtained from the HuggingFace mirror
`BhupatiNadar/adain-style-transfer-models`). Everything else is written for this assignment.
