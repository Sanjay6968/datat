"""
main_demo.py — Single-image inference demo using both trained models.

Usage:
  python main_demo.py <path_to_image>

Example:
  python main_demo.py "C:/Users/ASUS/Downloads/yy/Orange Dataset/test/fresh/f (1).png"
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataset import CLASSES, CLS2IDX
from src.classical_model import extract_features, load_model
from src.gnn_model import FruitGCN, image_to_graph, NODE_DIM
from torch_geometric.loader import DataLoader as PyGLoader

MODELS_DIR = "models"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

def load_gnn(num_classes=4) -> FruitGCN:
    m = FruitGCN(in_dim=NODE_DIM, hidden=64, num_classes=num_classes)
    ckpt = os.path.join(MODELS_DIR, "gnn_model.pth")
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    m.eval()
    return m.to(DEVICE)

def predict_classical(image_path: str, model) -> str:
    feat = extract_features(image_path).reshape(1, -1)
    idx  = model.predict(feat)[0]
    proba = model.predict_proba(feat)[0]
    return CLASSES[idx], proba

def predict_gnn_single(image_path: str, model: FruitGCN):

    data   = image_to_graph(image_path, label=0)
    loader = PyGLoader([data], batch_size=1)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out   = model(batch.x, batch.edge_index, batch.batch)
            proba = torch.softmax(out, dim=1).cpu().numpy()[0]
            idx   = proba.argmax()
    return CLASSES[idx], proba

def visualize(image_path: str, cls_pred: str, cls_proba,
              gnn_pred: str, gnn_proba, out_path: str = "demo_result.png"):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].imshow(img)
    axes[0].set_title(f"Input Image\nTrue folder: {os.path.basename(os.path.dirname(image_path))}")
    axes[0].axis("off")

    axes[1].barh(CLASSES, cls_proba, color="steelblue")
    axes[1].axvline(0.5, color="red", linestyle="--", linewidth=1)
    axes[1].set_xlim(0, 1)
    axes[1].set_title(f"Approach 1 — SVM\nPrediction: {cls_pred}", fontweight="bold")
    axes[1].set_xlabel("Probability")

    axes[2].barh(CLASSES, gnn_proba, color="seagreen")
    axes[2].axvline(0.5, color="red", linestyle="--", linewidth=1)
    axes[2].set_xlim(0, 1)
    axes[2].set_title(f"Approach 2 — GCN\nPrediction: {gnn_pred}", fontweight="bold")
    axes[2].set_xlabel("Probability")

    plt.suptitle("Orange Condition Classification — Demo", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Visualization saved -> {out_path}")

def main():
    if len(sys.argv) < 2:

        image_path = r"C:\Users\ASUS\Downloads\yy\Orange Dataset\test\fresh\f (1).png"
        print(f"No image path supplied. Using default: {image_path}")
    else:
        image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"ERROR: File not found: {image_path}"); sys.exit(1)

    print(f"\nImage: {image_path}")
    print("-" * 50)

    svm_path = os.path.join(MODELS_DIR, "svm_model.joblib")
    if not os.path.exists(svm_path):
        print("SVM model not found. Run train_classical.py first."); sys.exit(1)
    svm = load_model(svm_path)
    cls_pred, cls_proba = predict_classical(image_path, svm)
    print(f"[Approach 1 — SVM]      -> {cls_pred}  (conf: {cls_proba.max():.2%})")

    gnn_path = os.path.join(MODELS_DIR, "gnn_model.pth")
    if not os.path.exists(gnn_path):
        print("GNN model not found. Run train_gnn.py first."); sys.exit(1)
    gnn = load_gnn()
    gnn_pred, gnn_proba = predict_gnn_single(image_path, gnn)
    print(f"[Approach 2 — GCN]      -> {gnn_pred}  (conf: {gnn_proba.max():.2%})")

    print("-" * 50)

    visualize(image_path, cls_pred, cls_proba,
              gnn_pred, gnn_proba, out_path="demo_result.png")

    print("\nDone.")

if __name__ == "__main__":
    main()
