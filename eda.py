"""
eda.py — Exploratory Data Analysis: class distribution, sample images,
         colour space visualisations, imbalance check.

Run:  python eda.py
Outputs go to  models/eda_*.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

from src.dataset import CLASSES, CLS2IDX, TRAIN_DIR, TEST_DIR, collect_paths

OUT = "models"
os.makedirs(OUT, exist_ok=True)

def plot_distribution():
    tr_p, tr_l = collect_paths(TRAIN_DIR)
    te_p, te_l = collect_paths(TEST_DIR)
    tr_counts  = Counter(tr_l)
    te_counts  = Counter(te_l)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, counts, title in [
        (axes[0], tr_counts, "Train Set"),
        (axes[1], te_counts, "Test Set"),
    ]:
        labels  = [CLASSES[i] for i in sorted(counts)]
        values  = [counts[i]  for i in sorted(counts)]
        bars    = ax.bar(labels, values, color=["#e07b39","#c0392b","#27ae60","#f39c12"])
        ax.set_title(f"Class Distribution — {title}")
        ax.set_ylabel("Image Count")
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width()/2, v + 1, str(v),
                    ha="center", va="bottom", fontsize=10)
    plt.suptitle("Orange Dataset — Class Imbalance Analysis", fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUT, "eda_class_distribution.png")
    plt.savefig(path, dpi=130); plt.close()
    print(f"  Saved -> {path}")
    print(f"  Train counts: { {CLASSES[k]: v for k, v in tr_counts.items()} }")
    imbalance = max(tr_counts.values()) / min(tr_counts.values())
    print(f"  Imbalance ratio (max/min): {imbalance:.2f}x")

def plot_samples(n: int = 4):
    fig, axes = plt.subplots(len(CLASSES), n,
                              figsize=(n * 3, len(CLASSES) * 3))
    for r, cls in enumerate(CLASSES):
        folder = Path(TRAIN_DIR) / cls
        files  = sorted(folder.iterdir())[:n]
        for c, f in enumerate(files):
            img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
            axes[r][c].imshow(img)
            axes[r][c].axis("off")
            if c == 0:
                axes[r][c].set_ylabel(cls, fontsize=12, fontweight="bold")
    plt.suptitle("Sample Images per Class", fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUT, "eda_sample_images.png")
    plt.savefig(path, dpi=120); plt.close()
    print(f"  Saved -> {path}")

def plot_lab_distributions(n_per_class: int = 30):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    ch_names  = ["L* (Lightness)", "a* (Green↔Red)", "b* (Blue↔Yellow)"]
    colors    = ["#e07b39", "#c0392b", "#27ae60", "#f39c12"]

    for cls, color in zip(CLASSES, colors):
        folder = Path(TRAIN_DIR) / cls
        files  = sorted(folder.iterdir())[:n_per_class]
        means  = [[], [], []]
        for f in files:
            img = cv2.imread(str(f))
            if img is None: continue
            lab = cv2.cvtColor(cv2.resize(img, (64, 64)), cv2.COLOR_BGR2LAB)
            for ch in range(3):
                means[ch].append(lab[:, :, ch].mean())
        for ch, ax in enumerate(axes):
            ax.hist(means[ch], bins=15, alpha=0.6, label=cls, color=color)
            ax.set_title(ch_names[ch]); ax.set_xlabel("Mean value")
    axes[0].set_ylabel("Frequency")
    axes[0].legend(fontsize=9)
    plt.suptitle("LAB Colour Channel Distributions per Class", fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUT, "eda_lab_distributions.png")
    plt.savefig(path, dpi=120); plt.close()
    print(f"  Saved -> {path}")

def plot_resolution_scatter():
    tr_p, tr_l = collect_paths(TRAIN_DIR)
    widths, heights, labels = [], [], []
    for p, l in zip(tr_p, tr_l):
        img = cv2.imread(p)
        if img is None: continue
        h, w = img.shape[:2]
        widths.append(w); heights.append(h); labels.append(CLASSES[l])

    fig, ax = plt.subplots(figsize=(7, 5))
    palette = {"blackspot": "#e07b39", "canker": "#c0392b",
               "fresh": "#27ae60", "grenning": "#f39c12"}
    for cls in CLASSES:
        idxs = [i for i, l in enumerate(labels) if l == cls]
        ax.scatter([widths[i] for i in idxs], [heights[i] for i in idxs],
                   label=cls, alpha=0.5, color=palette[cls], s=12)
    ax.set_xlabel("Width (px)"); ax.set_ylabel("Height (px)")
    ax.set_title("Image Resolution Distribution"); ax.legend()
    plt.tight_layout()
    path = os.path.join(OUT, "eda_resolution.png")
    plt.savefig(path, dpi=120); plt.close()
    print(f"  Saved -> {path}")

if __name__ == "__main__":
    print("Running EDA …")
    plot_distribution()
    plot_samples()
    plot_lab_distributions()
    plot_resolution_scatter()
    print("EDA complete. Outputs in ./models/eda_*.png")
