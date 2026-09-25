# PA1 – Learning Beyond IID: Inductive Biases, Domain Adaptation, Domain Generalisation and Open-Set Recognition

EE-5102/CS-6304 Advanced Topics in Machine Learning (Fall 2026), Programming Assignment 1.
Code for the four tasks lives in `task1/` … `task4/`; the PACS protocol shared by Tasks 2 and 3 is in `shared/`;
genuinely common helpers (seeding, JSON/CSV logging, metrics, plotting defaults) are in `common/`.
Every number in the report traces to a CSV/JSON file under `task*/results/`.

## Environment
```bash
conda create -n atml-pa1 python=3.10 && conda activate atml-pa1
pip install -r requirements.txt          # torch 2.5.1 / torchvision 0.20.1 (CUDA 12.1), open_clip_torch 3.3.0, ...
```
All experiments were run on one RTX 2070 (8 GB). Seed 6304 everywhere (`common/seed.py`).

## Data (not committed)
| Dataset | Source | Location |
|---|---|---|
| STL-10 (labelled train/test) | HF parquet mirror `tanganke/stl10` (`data/train-*.parquet`, `data/test-*.parquet`) | `data/stl10_hf/{train,test}.parquet` |
| PACS | HF mirror `flwrlabs/pacs` (`data/train-00000-of-00001.parquet`) | `data/pacs/pacs.parquet` |
| CIFAR-10 / CIFAR-100 | `torchvision.datasets` (download=True once) | `data/cifar-10-batches-py`, `data/cifar-100-python` |
| AdaIN weights | naoto0804/pytorch-AdaIN release weights, HF mirror `BhupatiNadar/adain-style-transfer-models` | `data/weights/adain_{vgg_normalised,decoder}.pth` |
| CLIP ViT-B/32 `openai` | open_clip (`timm/vit_base_patch32_clip_224.openai`) | `data/weights/clip_vitb32_openai.bin` (optional local copy) |

## Reproducing each task
Each task README (`task*/README.md`) lists the exact commands; in short:
```bash
# Task 1
python task1/data/make_subset.py && python task1/data/make_cue_conflicts.py && python task1/scripts/run_task1.py
# Tasks 2 + 3 (shared PACS protocol)
python shared/make_pacs_splits.py
for m in source_only dan dann cdan dan_lambda0.1 dan_lambda10; do python task2/train.py --config task2/configs/$m.yaml; done
python task2/evaluate_final.py                    # Sketch labels used here only
for m in erm dan_dg sam sam_rho0.01 sam_rho0.1; do python task3/train.py --config task3/configs/$m.yaml; done
python task3/selection/source_validation.py       # Sketch-free diagnostics
python task3/evaluate_sketch.py                   # loads Sketch last
# Task 4
python task4/data/make_splits.py
for m in vanilla gcsc proser rpl; do python task4/train.py --config task4/configs/$m.yaml; done
python task4/extract_outputs.py && python task4/evaluate_osr.py
```

## Leakage checklist
* Task 2: checkpoints chosen by mean source-validation macro-F1; the Sketch loader used in training returns label −1;
  Sketch labels are read only in `task2/evaluate_final.py`.
* Task 3: all methods have `needs_target=False`; no Task 3 training/selection/diagnostic script builds a Sketch
  loader; Task 3 settings (λ_DG=1, ρ=0.05 and the ρ study) were fixed in the configs before any Sketch evaluation
  and were not changed afterwards.
* Task 4: CIFAR-100 is imported only in `extract_outputs.py` / `evaluate_osr.py`; thresholds use CIFAR-10 validation only.

## External code attribution
* `task1/models/adain.py`: network definition and pretrained weights from naoto0804/pytorch-AdaIN (MIT).
* `task4/methods/proser.py`: loss structure follows zhoudw-zdw/CVPR21-Proser (re-implemented).
* `task4/methods/rpl.py`: reciprocal-point loss follows iCGY96/ARPL `RPLoss` (re-implemented).
* Gradient reversal, MMD, SAM and all evaluation code were written for this assignment with standard library calls
  (torch, torchvision, open_clip, scikit-learn).
