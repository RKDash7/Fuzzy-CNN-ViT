# NEU-DET Architecture Ablation Study

`architecture_comparison.py` trains and evaluates the nine Fuzzy CNN-ViT configurations used in the architecture-ablation study.

## Dataset layout

The script expects the same NEU-DET directory layout as the main experiment:

```text
DATA_ROOT/
└── NEU-DET/
    ├── train/images/<six class folders>/
    └── validation/images/<six class folders>/
```

The original training folder is stratified into 80% training and 20% held-out testing subsets. The supplied validation folder is used for learning-rate scheduling, checkpoint selection, and early stopping. Test data is evaluated only after restoring the best validation-loss checkpoint.

## Configurations

| Configuration | Transformer layers | Embedding | Heads |
|---|---:|---:|---:|
| 1L-64D-4H | 1 | 64 | 4 |
| 2L-64D-4H | 2 | 64 | 4 |
| 4L-64D-4H | 4 | 64 | 4 |
| 6L-64D-4H | 6 | 64 | 4 |
| 2L-128D-4H | 2 | 128 | 4 |
| 2L-128D-8H | 2 | 128 | 8 |
| 4L-128D-8H | 4 | 128 | 8 |
| 6L-128D-8H | 6 | 128 | 8 |
| 6L-256D-8H | 6 | 256 | 8 |

The MLP hidden dimension defaults to 512 for all configurations, matching the MLP dimension in the supplied single-model NEU-DET implementation. Attention-head count changes the partitioning of the embedding space; it does not by itself change the number of attention projection parameters.

## Installation

From the repository's `fuzzy-cnn-vit-cross-dataset` directory:

```bash
pip install -r requirements.txt
```

## Run all configurations and five seeds

```bash
python experiments/neu_det/architecture_comparison.py \
  --data-root /path/to/data \
  --output-dir experiments/neu_det/results/architecture_comparison \
  --seeds 42 123 202 456 789 \
  --epochs 200 --patience 20
```

For Google Colab, after cloning the repository and extracting the dataset under `/content/data/NEU-DET/`:

```bash
%cd /content/Fuzzy-CNN-ViT/fuzzy-cnn-vit-cross-dataset
!pip install -r requirements.txt
!python experiments/neu_det/architecture_comparison.py \
  --data-root /content/data \
  --output-dir /content/architecture_comparison \
  --seeds 42 123 202 456 789
```

For a short pipeline test (not a publishable experiment):

```bash
python experiments/neu_det/architecture_comparison.py \
  --data-root /path/to/data --seeds 42 --epochs 2 --patience 2 \
  --configs 1L-64D-4H
```

## Outputs

The output directory contains:

- `architecture_comparison_all_runs.csv`: one row per configuration and seed.
- `architecture_comparison_summary.csv`: mean and standard deviation for each metric.
- `architecture_comparison_paper_table.csv`: formatted mean ± standard deviation values.
- `all_training_histories.csv`: epoch-level training/validation history.
- `experiment_configuration.json`: run settings.
- `<configuration>/seed_<seed>/best_validation_checkpoint.pth`: best validation checkpoint for each run.
- Per-run `training_history.csv` and `test_predictions.csv`.

Metrics use **macro averaging** for precision, recall, and F1, consistent with the original NEU-DET evaluation script. Standard deviations in the paper table use the sample standard deviation (`ddof=1`).

## Important reporting note

This script computes new results; it does not hard-code or reproduce the numerical values in an existing manuscript table. Parameter counts are computed from the actual instantiated PyTorch models. In particular, the values produced by this implementation should be checked against any previously reported parameter-count column before publication. Do not copy old mean ± standard deviation values into the new output unless they are independently reproduced from the same configurations, data split, seeds, and training protocol.

The default five seeds refer to independent model initializations and data-augmentation randomness. The stratified train/test split is fixed across configurations using `--split-seed` (default 42), so comparisons are paired on the same held-out test examples.
