# Fuzzy CNN–ViT: Primary Experiment and Cross-Dataset Validation

This repository contains **one proposed-model research project** with two clearly separated dataset workflows:

1. **NEU-DET — primary experiment:** the main Fuzzy CNN–ViT steel-surface-defect classification experiment, including its training, validation, held-out testing, metrics, and attention visualization.
2. **PCB — cross-dataset validation:** a separate validation workflow used to examine the proposed approach on a second defect dataset. It is included as a validation dataset/workflow, **not as an unrelated project** and not mixed into the NEU-DET data or result folders.

The two workflows have separate scripts, data preparation, outputs, and instructions. They share a repository because they support the same research study.

## Repository layout

```text
fuzzy-cnn-vit-cross-dataset/
├── README.md
├── requirements.txt
├── .gitignore
├── experiments/
│   └── neu_det/
│       ├── README.md
│       ├── neu_det_fuzzy_cnn_vit.py
│       └── outputs/
│           └── .gitkeep
└── validation/
    └── pcb/
        ├── README.md
        ├── pcb_defect_classification.py
        └── outputs/
            └── .gitkeep
```

## Dataset roles and separation

| Item | NEU-DET | PCB |
|---|---|---|
| Role in the study | Primary experiment | Cross-dataset validation |
| Data format | Class-folder images | Pascal VOC XML annotations and defect crops |
| Classes | Six | Six |
| Script | `experiments/neu_det/neu_det_fuzzy_cnn_vit.py` | `validation/pcb/pcb_defect_classification.py` |
| Output location | Configured in the NEU-DET script | Configured in the PCB script |

Do not combine the datasets before splitting, and do not mix their result files. Report the NEU-DET primary results and PCB cross-dataset validation results separately in the manuscript.

## Environment setup

Python 3.10 or later is recommended. Install the combined dependencies from the repository root:

```bash
pip install -r requirements.txt
```

For Google Colab, clone this repository and run the desired script from its own directory. Each script currently contains its own dataset and output path configuration; check those paths before execution.

## Run the primary NEU-DET experiment

```bash
python experiments/neu_det/neu_det_fuzzy_cnn_vit.py
```

See [`experiments/neu_det/README.md`](experiments/neu_det/README.md) for the dataset layout, split, architecture, training configuration, and outputs.

## Run PCB cross-dataset validation

```bash
python validation/pcb/pcb_defect_classification.py
```

See [`validation/pcb/README.md`](validation/pcb/README.md) for the PCB dataset source, VOC XML processing, defect classes, outputs, and experimental caveats.



## Data, weights, and results

Datasets, downloaded archives, trained checkpoints, and generated result files are not included in this source repository. Follow the upstream dataset terms and citation requirements before use or redistribution. Do not commit large datasets, private data, or model checkpoints unless appropriate.
