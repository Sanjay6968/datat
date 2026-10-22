"""
dataset.py — image loading, stratified splits, PyTorch datasets for both
             classical (path-based) and GNN (graph-based) pipelines.
"""
import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

CLASSES   = sorted(["blackspot", "canker", "fresh", "grenning"])
CLS2IDX   = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASS = len(CLASSES)

def _resolve_data_root():
    candidates = [
        os.environ.get("DATA_ROOT", ""),
        r"C:\Users\ASUS\Downloads\yy\Orange Dataset",
        os.path.join(os.getcwd(), "Orange Dataset"),
        os.path.join(os.getcwd(), "data", "Orange Dataset"),
        os.path.join(os.getcwd(), "data", "raw"),
        "/content/Orange Dataset",
        "/content/datat/Orange Dataset",
        "/content/data/Orange Dataset",
    ]
    for c in candidates:
        if c and os.path.exists(c) and os.path.exists(os.path.join(c, "train")):
            return c
    return r"C:\Users\ASUS\Downloads\yy\Orange Dataset"

DATA_ROOT = _resolve_data_root()
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR  = os.path.join(DATA_ROOT, "test")

def collect_paths(root: str):
    paths, labels = [], []
    for cls in CLASSES:
        folder = Path(root) / cls
        if not folder.exists():
            continue
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                paths.append(str(f))
                labels.append(CLS2IDX[cls])
    return paths, labels

def get_splits(val_size: float = 0.18, seed: int = 42):
    """Return (tr_paths, val_paths, te_paths, tr_labels, val_labels, te_labels)."""
    tr_paths, tr_labels = collect_paths(TRAIN_DIR)
    te_paths, te_labels = collect_paths(TEST_DIR)
    tr_p, val_p, tr_l, val_l = train_test_split(
        tr_paths, tr_labels, test_size=val_size,
        stratify=tr_labels, random_state=seed,
    )
    return tr_p, val_p, te_paths, tr_l, val_l, te_labels

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

TRAIN_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])
EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

class ImageDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths   = paths
        self.labels  = labels
        self.transform = transform or EVAL_TF

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]
