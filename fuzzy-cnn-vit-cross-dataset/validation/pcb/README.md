# PCB — Cross-Dataset Validation

This directory contains the PCB workflow included to validate the research model/framework on a second defect dataset. It is part of the same study and repository as the primary NEU-DET experiment, but its data, preprocessing, script, and outputs remain separate.

## Dataset

The script processes PCB defect images annotated in Pascal VOC XML format and creates object-level defect crops. The six defect classes are:

1. `missing_hole`
2. `mouse_bite`
3. `open_circuit`
4. `short`
5. `spur`
6. `spurious_copper`

Dataset/project source: <https://github.com/Ixiaohuihuihui/Tiny-Defect-Detection-for-PCB>

The supplied script downloads the VOC PCB archive using its configured Dropbox URL. If that URL is unavailable, obtain `VOC_PCB.zip` using the upstream project's instructions and update `DATASET_URL` in `pcb_defect_classification.py`.

## Model implementation

The supplied PCB script uses convolutional feature extraction, patch projection, a learnable CLS token and positional embeddings, multi-head self-attention, and a classification head. It also produces attention visualizations. These are attention maps, not Grad-CAM explanations.

**Architecture consistency:** the supplied PCB script has its own CNN/attention implementation. Before claiming strict cross-dataset validation of the exact NEU-DET Fuzzy CNN–ViT architecture, align and verify the model components, preprocessing, input handling, and training protocol across datasets. Otherwise describe this accurately as validation of the broader proposed framework or approach.

## Run

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
python validation/pcb/pcb_defect_classification.py
```

The script saves the checkpoint to `/content/best_pcb_hybrid_model.pth` and plots under `/content/pcb_outputs/` by default. The script's paths and download settings can be edited in its setup section.

## Experimental protocol caveats — address before final reporting

The supplied implementation is retained as received, but two choices can bias the reported test performance:

1. **Test-set checkpoint selection:** the training loop evaluates test accuracy during training and selects the best checkpoint using that test accuracy. Introduce a separate validation set and select checkpoints using validation metrics only; evaluate the held-out test set once at the end.
2. **Crop-level split:** the current split is over extracted defect crops. If several crops come from one PCB source image, crops from the same source may be present in both train and test. Split by original image filename/group before crop assignment.

The script uses both a weighted sampler and class-weighted cross-entropy; document this choice and compare alternatives if relevant. For robust reporting, run multiple controlled seeds and report mean ± standard deviation while keeping the final test set fixed.

## Outputs and attribution

Expected outputs include confusion matrix, per-class accuracy, precision/recall/F1, confidence histogram, one-vs-rest ROC curves, selected predictions, and attention visualizations. The source repository does not include the dataset, archive, or trained checkpoint. Follow upstream licensing and citation requirements before use or redistribution.
