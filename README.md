# Fine-Grained Flower Classification: CNN vs Vision Transformer, with Parameter-Efficient Fine-Tuning

A controlled comparison of ResNet-50 (CNN) and ViT-B/16 (Vision Transformer) on the Oxford 102 Flowers dataset, extended with a parameter-efficient (LoRA) ViT variant and a model-agnostic interpretability analysis.

## Overview
- **Task:** fine-grained classification of 102 flower species (Oxford 102 Flowers).
- **Baseline:** ResNet-50, fine-tuned from ImageNet weights.
- **Comparison:** ViT-B/16, fine-tuned from ImageNet weights.
- **Improvement (in progress):** LoRA-tuned ViT-B/16 — aims to match full fine-tuning accuracy while training under 1% of the parameters.
- **Interpretability:** occlusion sensitivity analysis showing which regions each model relies on.

## Results
| Model | Trainable params | Test accuracy | Macro-F1 |
|---|---|---|---|
| ResNet-50 (full fine-tune) | ~25M | 90.8% | 90.5% |
| ViT-B/16 (full fine-tune) | ~86M | 98.9% | 98.8% |
| ViT-B/16 + LoRA | pending | pending | pending |

## Dataset
Oxford 102 Flowers (Nilsback & Zisserman, 2008): 8,189 images, 102 classes, standard 1,020 / 1,020 / 6,149 train/val/test split.
Source: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/

## Repository structure
- `notebooks/` — training and evaluation notebooks
- `src/` — training scripts (e.g. train_peft.py)
- `results/` — metrics, confusion matrices, figures
- `requirements.txt` — exact package versions

## Environment
Python 3.x, Keras 3 / KerasHub, TensorFlow. Trained on Google Colab (T4 GPU). See requirements.txt for exact versions.

## Citation
Nilsback, M-E. and Zisserman, A. "Automated Flower Classification over a Large Number of Classes", ICVGIP, 2008.
