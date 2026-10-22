"""
train_gnn.py — Approach 2: Superpixel Graph Neural Network (GCN) training.

Run:  python train_gnn.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.dataset import get_splits, CLASSES, NUM_CLASS
from src.gnn_model import (
    FruitGCN, build_loaders, train_gnn,
    evaluate_gnn, predict_gnn, NODE_DIM,
)

MODELS_DIR = "models"
CACHE_DIR  = "models/graph_cache"
os.makedirs(MODELS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50
BATCH  = 32
LR     = 1e-3

def plot_history(history, out_path):
    epochs   = [h["epoch"]    for h in history]
    tr_acc   = [h["tr_acc"]   for h in history]
    val_acc  = [h["val_acc"]  for h in history]
    tr_loss  = [h["tr_loss"]  for h in history]
    val_loss = [h["val_loss"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, tr_acc,  label="Train"); ax1.plot(epochs, val_acc,  label="Val")
    ax1.set_title("GCN Accuracy"); ax1.set_xlabel("Epoch"); ax1.legend()
    ax2.plot(epochs, tr_loss, label="Train"); ax2.plot(epochs, val_loss, label="Val")
    ax2.set_title("GCN Loss");     ax2.set_xlabel("Epoch"); ax2.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close()
    print(f"  Saved plot -> {out_path}")

def plot_cm(cm, title, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title); plt.tight_layout()
    plt.savefig(out_path, dpi=120); plt.close()
    print(f"  Saved plot -> {out_path}")

def main():
    print("=" * 55)
    print("  APPROACH 2 — Superpixel GNN (GCN)")
    print(f"  Device: {DEVICE}")
    print("=" * 55)

    tr_p, val_p, te_p, tr_l, val_l, te_l = get_splits()

    print(f"\nDataset split:")
    print(f"  Train : {len(tr_p)}  Val : {len(val_p)}  Test : {len(te_p)}")

    print("\n[1/3] Building superpixel graph loaders …")
    tr_ld, val_ld, te_ld = build_loaders(
        tr_p, val_p, te_p, tr_l, val_l, te_l,
        cache_dir=CACHE_DIR, batch_size=BATCH,
    )

    print("\n[2/3] Training GCN …")
    model   = FruitGCN(in_dim=NODE_DIM, hidden=64, num_classes=NUM_CLASS)
    history = train_gnn(model, tr_ld, val_ld, epochs=EPOCHS, lr=LR, device=DEVICE)

    plot_history(history, os.path.join(MODELS_DIR, "gnn_training_curve.png"))

    ckpt_path = os.path.join(MODELS_DIR, "gnn_model.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Saved model -> {ckpt_path}")

    print("\n[3/3] Evaluating on test set …")
    y_true, y_pred = predict_gnn(model, te_ld, device=DEVICE)
    acc  = accuracy_score(y_true, y_pred)
    cm   = confusion_matrix(y_true, y_pred)
    rep  = classification_report(y_true, y_pred, target_names=CLASSES)
    mf1  = float(classification_report(y_true, y_pred, target_names=CLASSES,
                                        output_dict=True)["macro avg"]["f1-score"])

    print(f"\n{'='*52}\n  GCN — Test Results\n{'='*52}")
    print(rep)
    print("Confusion Matrix:\n", cm)

    plot_cm(cm, "GCN Confusion Matrix",
            os.path.join(MODELS_DIR, "gnn_confusion_matrix.png"))

    print("\n" + "=" * 52)
    print(f"  GCN  Accuracy : {acc:.4f}   Macro-F1 : {mf1:.4f}")
    print("=" * 52)
    print("  Done. Model saved to ./models/")

if __name__ == "__main__":
    main()
