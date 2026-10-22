"""
train_classical.py — Approach 1: train SVM + Random Forest, save models,
                     print evaluation metrics.

Run:  python train_classical.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score

from src.dataset import get_splits, CLASSES
from src.classical_model import (
    build_feature_matrix, make_svm, make_rf,
    train_and_eval, save_model,
)

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def plot_cm(cm, title, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Saved plot -> {out_path}")

def main():
    print("=" * 55)
    print("  APPROACH 1 — Classical CV + ML")
    print("=" * 55)

    tr_p, val_p, te_p, tr_l, val_l, te_l = get_splits()

    all_tr_p = tr_p + val_p
    all_tr_l = tr_l + val_l

    print(f"\nDataset split:")
    print(f"  Train+Val : {len(all_tr_p)} images")
    print(f"  Test      : {len(te_p)} images")

    print("\n[1/4] Extracting train features …")
    X_tr = build_feature_matrix(all_tr_p)
    y_tr = np.array(all_tr_l)

    print("[2/4] Extracting test features …")
    X_te = build_feature_matrix(te_p)
    y_te = np.array(te_l)

    results = {}

    print("\n[3/4] Training SVM (RBF kernel) …")
    svm = make_svm()
    res_svm = train_and_eval(svm, X_tr, y_tr, X_te, y_te, CLASSES, "SVM (RBF)")
    save_model(svm, os.path.join(MODELS_DIR, "svm_model.joblib"))
    plot_cm(res_svm["cm"], "SVM Confusion Matrix",
            os.path.join(MODELS_DIR, "svm_confusion_matrix.png"))
    results["SVM"] = res_svm

    print("\n[4/4] Training Random Forest (300 trees) …")
    rf = make_rf()
    res_rf = train_and_eval(rf, X_tr, y_tr, X_te, y_te, CLASSES, "Random Forest")
    save_model(rf, os.path.join(MODELS_DIR, "rf_model.joblib"))
    plot_cm(res_rf["cm"], "RF Confusion Matrix",
            os.path.join(MODELS_DIR, "rf_confusion_matrix.png"))
    results["RF"] = res_rf

    print("\n" + "=" * 55)
    print("  CLASSICAL MODEL COMPARISON SUMMARY")
    print("=" * 55)
    print(f"  {'Model':<18} {'Accuracy':>10} {'Macro F1':>10}")
    print("  " + "-" * 40)
    for name, r in results.items():
        acc   = r["accuracy"]
        mf1   = r["report"]["macro avg"]["f1-score"]
        print(f"  {name:<18} {acc:>10.4f} {mf1:>10.4f}")
    print("=" * 55)
    print("  Done. Models saved to ./models/")

if __name__ == "__main__":
    main()
