# ==========================================
# PCB DEFECT CLASSIFICATION FROM VOC XML
# Dataset source:
# https://github.com/Ixiaohuihuihui/Tiny-Defect-Detection-for-PCB
# ==========================================

import os
import shutil
import zipfile
import random
import subprocess
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. SETUP
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

REPO_URL = "https://github.com/Ixiaohuihuihui/Tiny-Defect-Detection-for-PCB.git"
REPO_PATH = "/content/Tiny-Defect-Detection-for-PCB"

DATASET_URL = "https://www.dropbox.com/s/h0f39nyotddibsb/VOC_PCB.zip?dl=1"
ZIP_PATH = "/content/VOC_PCB.zip"
EXTRACT_PATH = "/content/data"
MODEL_PATH = "/content/best_pcb_hybrid_model.pth"

# ==========================================
# 2. DOWNLOAD REPO + DATASET
# ==========================================
for path in [REPO_PATH, EXTRACT_PATH, ZIP_PATH]:
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.isfile(path):
        os.remove(path)

subprocess.run(
    ["git", "clone", "--depth", "1", REPO_URL, REPO_PATH],
    check=True
)
print("GitHub repo cloned:", REPO_PATH)

print("Downloading dataset...")
urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)
print("Dataset downloaded:", ZIP_PATH)

if not zipfile.is_zipfile(ZIP_PATH):
    raise Exception(
        "Downloaded file is not a valid ZIP. "
        "Dropbox may have blocked the direct download. "
        "Try downloading VOC_PCB.zip manually from the GitHub README."
    )

os.makedirs(EXTRACT_PATH, exist_ok=True)

with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
    zip_ref.extractall(EXTRACT_PATH)

print("Dataset extracted:", EXTRACT_PATH)

# ==========================================
# 3. FIND IMAGES + XML FILES RECURSIVELY
# ==========================================
image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

image_files = []
xml_files = []

for root, dirs, files in os.walk(EXTRACT_PATH):
    for f in files:
        path = os.path.join(root, f)
        ext = os.path.splitext(f)[1].lower()

        if ext in image_exts:
            image_files.append(path)
        elif ext == ".xml":
            xml_files.append(path)

if not image_files or not xml_files:
    print("Extracted folder structure:")
    for root, dirs, files in os.walk(EXTRACT_PATH):
        level = root.replace(EXTRACT_PATH, "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        if level >= 4:
            dirs[:] = []

    raise Exception("Could not find image files and XML annotation files.")

print("Total images found:", len(image_files))
print("Total XML files found:", len(xml_files))
print("Example image:", image_files[0])
print("Example XML:", xml_files[0])

image_index = {}

for path in image_files:
    stem = os.path.splitext(os.path.basename(path))[0]
    image_index[stem] = path

# ==========================================
# 4. CLASS NAMES
# ==========================================
CLASS_NAMES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper"
]

class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
idx_to_class = {i: name for name, i in class_to_idx.items()}

# ==========================================
# 5. PARSE VOC XML ANNOTATIONS
# ==========================================
def normalize_label(label):
    label = label.strip().lower()
    label = label.replace(" ", "_")
    label = label.replace("-", "_")
    return label


def find_image(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]

    if stem in image_index:
        return image_index[stem]

    for img_stem, img_path in image_index.items():
        if img_stem == stem or img_stem in stem or stem in img_stem:
            return img_path

    return None


def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename_node = root.find("filename")
    filename = filename_node.text.strip() if filename_node is not None else None

    img_path = find_image(filename) if filename else None

    if img_path is None:
        img_path = find_image(os.path.basename(xml_path))

    if img_path is None:
        return []

    samples = []

    for obj in root.findall("object"):
        name_node = obj.find("name")
        box_node = obj.find("bndbox")

        if name_node is None or box_node is None:
            continue

        label_name = normalize_label(name_node.text)

        if label_name not in class_to_idx:
            continue

        xmin = int(float(box_node.find("xmin").text))
        ymin = int(float(box_node.find("ymin").text))
        xmax = int(float(box_node.find("xmax").text))
        ymax = int(float(box_node.find("ymax").text))

        if xmax <= xmin or ymax <= ymin:
            continue

        samples.append({
            "image": img_path,
            "bbox": [xmin, ymin, xmax, ymax],
            "label": class_to_idx[label_name],
            "label_name": label_name
        })

    return samples


samples = []

for xml_path in xml_files:
    samples.extend(parse_voc_xml(xml_path))

if len(samples) == 0:
    raise Exception("No labeled PCB defects found in XML files.")

print("Total defect crops:", len(samples))
print("Class distribution:", Counter([s["label_name"] for s in samples]))

# ==========================================
# 6. TRAIN / TEST SPLIT
# NOTE: This supplied implementation splits defect crops rather than source images,
# and uses test accuracy for checkpoint selection. See README.md for research caveats.
# ==========================================
labels = [s["label"] for s in samples]
label_counts = Counter(labels)

if min(label_counts.values()) < 2:
    train_samples, test_samples = train_test_split(
        samples,
        test_size=0.2,
        random_state=SEED
    )
else:
    train_samples, test_samples = train_test_split(
        samples,
        test_size=0.2,
        random_state=SEED,
        stratify=labels
    )

print("Train samples:", len(train_samples))
print("Test samples:", len(test_samples))

# ==========================================
# 7. TRANSFORMS
# ==========================================
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((144, 144)),
    transforms.RandomResizedCrop(128, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ==========================================
# 8. DATASET CLASS
# ==========================================
class PCBCropDataset(Dataset):
    def __init__(self, samples, transform=None, padding_ratio=0.35):
        self.samples = samples
        self.transform = transform
        self.padding_ratio = padding_ratio

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        img = Image.open(item["image"]).convert("RGB")
        w, h = img.size

        xmin, ymin, xmax, ymax = item["bbox"]
        bw = xmax - xmin
        bh = ymax - ymin

        pad_x = int(bw * self.padding_ratio)
        pad_y = int(bh * self.padding_ratio)

        xmin = max(0, xmin - pad_x)
        ymin = max(0, ymin - pad_y)
        xmax = min(w, xmax + pad_x)
        ymax = min(h, ymax + pad_y)

        crop = img.crop((xmin, ymin, xmax, ymax))

        if self.transform:
            crop = self.transform(crop)

        return crop, item["label"]


train_data = PCBCropDataset(train_samples, transform=train_transform)
test_data = PCBCropDataset(test_samples, transform=test_transform)

# ==========================================
# 9. DATALOADERS
# ==========================================
train_targets = [s["label"] for s in train_samples]
class_count = Counter(train_targets)

sample_weights = [1.0 / class_count[label] for label in train_targets]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_data,
    batch_size=32,
    sampler=sampler,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_data,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

print("Train distribution:", {idx_to_class[k]: v for k, v in class_count.items()})

# ==========================================
# 10. CNN + ATTENTION MODEL
# ==========================================
class Hybrid_Model(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)

        self.patch = nn.Conv2d(64, 128, 2, 2)

        self.cls = nn.Parameter(torch.randn(1, 1, 128))
        self.pos = nn.Parameter(torch.randn(1, 257, 128))

        self.attn = nn.MultiheadAttention(128, 8, batch_first=True)
        self.norm = nn.LayerNorm(128)

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

        self.last_attn = None

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = self.patch(x)
        x = x.flatten(2).transpose(1, 2)

        B, N, C = x.shape
        cls = self.cls.expand(B, -1, -1)

        x = torch.cat((cls, x), dim=1)
        x = x + self.pos[:, :x.shape[1], :]

        attn_out, attn_weights = self.attn(
            x, x, x,
            need_weights=True,
            average_attn_weights=True
        )

        self.last_attn = attn_weights

        x = self.norm(x + attn_out)
        x = self.dropout(x[:, 0])

        return self.fc(x)


model = Hybrid_Model(num_classes=len(CLASS_NAMES)).to(device)

# ==========================================
# 11. LOSS + OPTIMIZER
# ==========================================
weights = []

for i in range(len(CLASS_NAMES)):
    weights.append(1.0 / class_count[i] if class_count[i] > 0 else 0.0)

weights = torch.tensor(weights, dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

# ==========================================
# 12. TRAINING
# ==========================================
EPOCHS = 50
best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        out = model(x)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(out, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    train_acc = correct / total
    avg_loss = total_loss / len(train_loader)

    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            preds = torch.argmax(out, dim=1)

            test_correct += (preds == y).sum().item()
            test_total += y.size(0)

    test_acc = test_correct / test_total

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), MODEL_PATH)

    print(
        f"Epoch {epoch + 1:03d}/{EPOCHS} | "
        f"Loss={avg_loss:.4f} | "
        f"Train Acc={train_acc:.4f} | "
        f"Test Acc={test_acc:.4f}"
    )

print("Best test accuracy:", best_acc)

# ==========================================
# 13. FINAL EVALUATION
# ==========================================
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)

        out = model(x)
        preds = torch.argmax(out, dim=1).cpu().numpy()

        y_true.extend(y.numpy())
        y_pred.extend(preds)

print("\nClassification Report:\n")
print(
    classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        zero_division=0
    )
)

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))

# ==========================================
# 14. ATTENTION HEATMAP VISUALIZATION
# ==========================================
x_batch, y_batch = next(iter(test_loader))
img = x_batch[0].to(device)
true_label = y_batch[0].item()

with torch.no_grad():
    out = model(img.unsqueeze(0))
    pred_label = torch.argmax(out, dim=1).item()

attn = model.last_attn

cls_attn = attn[0, 0, 1:].detach().cpu().numpy()

size = int(np.sqrt(len(cls_attn)))
attn_map = cls_attn[:size * size].reshape(size, size)

attn_map = attn_map - attn_map.min()
attn_map = attn_map / (attn_map.max() + 1e-8)

heatmap = cv2.resize(attn_map, (128, 128))

orig = img.detach().cpu().squeeze().numpy()
orig = orig - orig.min()
orig = orig / (orig.max() + 1e-8)

orig_rgb = np.stack([orig] * 3, axis=-1)

heatmap_color = cv2.applyColorMap(
    (heatmap * 255).astype(np.uint8),
    cv2.COLORMAP_JET
)

heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0
overlay = 0.6 * orig_rgb + 0.4 * heatmap_color

plt.figure(figsize=(5, 5))
plt.imshow(overlay)
plt.axis("off")
plt.title(f"True: {idx_to_class[true_label]} | Pred: {idx_to_class[pred_label]}")
plt.show()

# ==========================================
# 15. ADDITIONAL EVALUATION PLOTS
# ==========================================
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_curve,
    auc as sklearn_auc
)
from sklearn.preprocessing import label_binarize

OUTPUT_DIR = "/content/pcb_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def collect_predictions(model, loader):
    """Collect labels, predicted labels, confidence, and class probabilities."""
    model.eval()
    all_true, all_pred, all_conf, all_prob = [], [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predictions = probabilities.max(dim=1)

            all_true.extend(labels.cpu().numpy())
            all_pred.extend(predictions.cpu().numpy())
            all_conf.extend(confidence.cpu().numpy())
            all_prob.extend(probabilities.cpu().numpy())

    return (
        np.asarray(all_true),
        np.asarray(all_pred),
        np.asarray(all_conf),
        np.asarray(all_prob)
    )


def denormalize_image(tensor):
    image = tensor.detach().cpu().squeeze().numpy()
    return np.clip(image * 0.5 + 0.5, 0, 1)


def plot_evaluation_results(model, loader):
    y_true, y_pred, y_conf, y_score = collect_predictions(model, loader)
    class_ids = list(range(len(CLASS_NAMES)))
    ticks = np.arange(len(CLASS_NAMES))

    print("\\nClassification report:")
    print(classification_report(
        y_true, y_pred, labels=class_ids,
        target_names=CLASS_NAMES, zero_division=0
    ))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_ids)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(ticks, CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(ticks, CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(
                col, row, str(cm[row, col]),
                ha="center", va="center",
                color="white" if cm[row, col] > cm.max() / 2 else "black"
            )
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.show()

    # Per-class accuracy
    per_class_accuracy = []
    for class_id in class_ids:
        mask = y_true == class_id
        per_class_accuracy.append(
            float((y_pred[mask] == y_true[mask]).mean()) if mask.any() else 0.0
        )

    plt.figure(figsize=(9, 5))
    plt.bar(CLASS_NAMES, per_class_accuracy)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    for idx, value in enumerate(per_class_accuracy):
        plt.text(idx, min(value + 0.02, 1.02), f"{value:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "per_class_accuracy.png"), dpi=150)
    plt.show()

    # Precision / recall / F1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_ids, zero_division=0
    )
    x = np.arange(len(CLASS_NAMES))
    width = 0.25
    plt.figure(figsize=(11, 5))
    plt.bar(x - width, precision, width, label="Precision")
    plt.bar(x, recall, width, label="Recall")
    plt.bar(x + width, f1, width, label="F1-score")
    plt.xticks(x, CLASS_NAMES, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "precision_recall_f1.png"), dpi=150)
    plt.show()

    # Confidence distribution
    correct = y_true == y_pred
    plt.figure(figsize=(8, 5))
    plt.hist(y_conf[correct], bins=20, alpha=0.7, label="Correct")
    plt.hist(y_conf[~correct], bins=20, alpha=0.7, label="Incorrect")
    plt.xlabel("Prediction confidence")
    plt.ylabel("Number of samples")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confidence_histogram.png"), dpi=150)
    plt.show()

    # One-vs-rest ROC curves
    y_binary = label_binarize(y_true, classes=class_ids)
    plt.figure(figsize=(10, 8))
    for class_id in class_ids:
        if len(np.unique(y_binary[:, class_id])) < 2:
            print(f"Skipping ROC for {CLASS_NAMES[class_id]}: test labels contain one class only.")
            continue
        fpr, tpr, _ = roc_curve(y_binary[:, class_id], y_score[:, class_id])
        class_auc = sklearn_auc(fpr, tpr)
        plt.plot(
            fpr, tpr,
            label=f"{CLASS_NAMES[class_id]} (AUC={class_auc:.3f})"
        )
    plt.plot([0, 1], [0, 1], "k--", label="Random classifier")
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve_per_class.png"), dpi=150)
    plt.show()

    return y_true, y_pred, y_conf, y_score


def show_best_correct_predictions(model, loader, max_images=12):
    model.eval()
    items = []

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            probabilities = torch.softmax(logits, dim=1)
            confidence, predictions = probabilities.max(dim=1)

            for idx in range(len(labels)):
                true_label = int(labels[idx])
                pred_label = int(predictions[idx].cpu())
                if pred_label == true_label:
                    items.append((
                        float(confidence[idx].cpu()),
                        images[idx].cpu(),
                        true_label,
                        pred_label
                    ))

    if not items:
        print("No correct predictions found.")
        return

    items.sort(key=lambda item: item[0], reverse=True)
    items = items[:max_images]
    cols = 4
    rows = int(np.ceil(len(items) / cols))
    plt.figure(figsize=(14, 4 * rows))

    for idx, (confidence, image, true_label, pred_label) in enumerate(items):
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(denormalize_image(image), cmap="gray")
        plt.axis("off")
        plt.title(
            f"True: {CLASS_NAMES[true_label]}\\n"
            f"Pred: {CLASS_NAMES[pred_label]}\\nConf: {confidence:.2f}",
            color="green", fontsize=9
        )

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "best_correct_predictions.png"), dpi=150)
    plt.show()


def show_correct_predictions_with_attention(model, loader, max_images=12):
    """Show correctly classified crops with CLS-to-patch attention overlays."""
    model.eval()
    items = []

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            probabilities = torch.softmax(logits, dim=1)
            confidence, predictions = probabilities.max(dim=1)

            # The model stores averaged attention as [B, target_tokens, source_tokens].
            batch_attention = model.last_attn.detach().cpu()

            for idx in range(len(labels)):
                true_label = int(labels[idx])
                pred_label = int(predictions[idx].cpu())
                if pred_label == true_label:
                    items.append({
                        "image": images[idx].cpu(),
                        "true": true_label,
                        "pred": pred_label,
                        "confidence": float(confidence[idx].cpu()),
                        "attention": batch_attention[idx]
                    })
                if len(items) >= max_images:
                    break
            if len(items) >= max_images:
                break

    if not items:
        print("No correct predictions found.")
        return

    cols = 4
    rows = int(np.ceil(len(items) / cols))
    plt.figure(figsize=(14, 4 * rows))

    for idx, item in enumerate(items):
        original = denormalize_image(item["image"])
        original_rgb = np.stack([original] * 3, axis=-1)

        # Row 0 is the CLS query; columns 1 onward correspond to image patches.
        cls_attention = item["attention"][0, 1:].numpy()
        side = int(np.sqrt(len(cls_attention)))
        attention_map = cls_attention[:side * side].reshape(side, side)
        attention_map -= attention_map.min()
        attention_map /= attention_map.max() + 1e-8

        heatmap = cv2.resize(attention_map, (128, 128))
        heatmap_color = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap_color = cv2.cvtColor(
            heatmap_color, cv2.COLOR_BGR2RGB
        ) / 255.0

        overlay = np.clip(0.60 * original_rgb + 0.40 * heatmap_color, 0, 1)

        plt.subplot(rows, cols, idx + 1)
        plt.imshow(overlay)
        plt.axis("off")
        plt.title(
            f"True: {CLASS_NAMES[item['true']]}\\n"
            f"Pred: {CLASS_NAMES[item['pred']]}\\n"
            f"Conf: {item['confidence']:.2f}",
            color="green", fontsize=9
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "correct_predictions_with_attention.png"),
        dpi=150
    )
    plt.show()


# Run the additional plots using the best saved checkpoint.
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
plot_evaluation_results(model, test_loader)
show_best_correct_predictions(model, test_loader, max_images=12)
show_correct_predictions_with_attention(model, test_loader, max_images=12)
