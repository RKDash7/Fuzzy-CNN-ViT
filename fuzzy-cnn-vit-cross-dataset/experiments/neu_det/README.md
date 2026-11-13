# NEU-DET — Primary Experiment

This directory contains the primary Fuzzy CNN–ViT implementation for NEU-DET steel surface defect classification. It is the main experiment in the repository; PCB is documented separately under `validation/pcb/` as cross-dataset validation.

## Dataset and split

- Dataset: NEU-DET
- Total: 1,800 images across six class folders
- Training folder: 1,440 images, split into 1,152 training and 288 held-out test images
- Separate validation folder: 360 images
- Final split: 1,152 train + 360 validation + 288 test

Expected layout:

```text
/content/data/NEU-DET/
├── train/images/<class_name>/
└── validation/images/<class_name>/
```

The script expects the paths above by default. Edit `train_dir` and `val_dir` near the top if needed. Folder names must match across both directories.

## Model and training

The implementation includes a learnable Gaussian fuzzy membership layer, two CNN feature-extraction blocks, convolutional patch embedding, CLS and positional embeddings, a multi-head Transformer block, and a classification head. Inputs are grayscale and resized to 128 × 128.

Default configuration: batch size 32, up to 200 epochs, early-stopping patience 20, AdamW, learning rate 1e-4, weight decay 1e-4, and seed 42. Checkpoint selection uses validation loss; the 288-image test set is reserved for final evaluation.

## Run

From the repository root:

```bash
python experiments/neu_det/neu_det_fuzzy_cnn_vit.py
```

The script saves results under `/content/Fuzzy_CNN_ViT_Results/` by default, including checkpoints, training curves, confusion matrices, CSV metrics/reports, predictions, and an attention visualization.

## Reporting notes

- Do not use the held-out test set for hyperparameter tuning or checkpoint selection.
- The attention overlay represents CLS-token attention to image patches. It is not Grad-CAM and should not be described as a causal explanation.
- For repeated-run comparisons, use controlled seeds and report mean ± standard deviation; repeated evaluations on the same fixed test set are not independent test sets.
- This repository contains code only, not the NEU-DET data or trained weights. Check the dataset's license and citation requirements.
