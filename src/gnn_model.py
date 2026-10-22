"""
gnn_model.py — Approach 2: Superpixel Graph + Graph Convolutional Network (GCN).

Pipeline:
  Image -> SLIC superpixels (nodes) -> RAG adjacency (edges)
  Node features: mean LAB (3) + mean HSV (3) + LBP hist (10) = 16-d per node
  GCN (2 conv layers) + global mean pool -> 4-class softmax

Why GNN?
  Each image becomes a graph where nodes = image regions (superpixels) and
  edges = spatial adjacency.  The GCN learns how neighboring regions relate
  to each other — e.g. a dark-spot region adjacent to fresh-orange skin is
  a strong cue for 'blackspot', which pure pixel classifiers miss.
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset as PyGDataset
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader as PyGLoader
from skimage.segmentation import slic
from skimage import graph as skgraph
from skimage.feature import local_binary_pattern
from pathlib import Path

N_SEGMENTS  = 75
NODE_DIM    = 16
HIDDEN_DIM  = 64
NUM_CLASSES = 4

def _node_features(img_bgr: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """
    Per-superpixel features:
      [0:3]  mean LAB (L*, a*, b*)
      [3:6]  mean HSV (H, S, V)
      [6:16] LBP uniform histogram (P=8, R=1, 10 bins) normalised
    """
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(float)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(float)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lbp     = local_binary_pattern(gray, P=8, R=1, method="uniform")

    ids   = np.unique(segments)
    feats = np.zeros((len(ids), NODE_DIM), dtype=np.float32)
    for i, sid in enumerate(ids):
        mask = segments == sid
        feats[i, 0:3]  = img_lab[mask].mean(axis=0) / 255.0
        feats[i, 3:6]  = img_hsv[mask].mean(axis=0) / [180., 255., 255.]
        lbp_h, _ = np.histogram(lbp[mask], bins=np.arange(0, 11), range=(0, 10))
        lbp_h    = lbp_h.astype(float)
        s = lbp_h.sum()
        feats[i, 6:16] = lbp_h / s if s > 0 else lbp_h
    return feats

def image_to_graph(image_path: str, label: int) -> Data:
    """Convert one image to a PyG Data object."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Cannot read: {image_path}")
    img_bgr = cv2.resize(img_bgr, (128, 128))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    segments = slic(img_rgb, n_segments=N_SEGMENTS, compactness=10,
                    sigma=1, start_label=0)

    rag     = skgraph.rag_mean_color(img_rgb, segments)
    edges   = list(rag.edges())
    if len(edges) == 0:

        edges = [(0, 0)]
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]

    edge_index = torch.tensor(
        [src + dst, dst + src], dtype=torch.long
    )

    x = torch.from_numpy(_node_features(img_bgr, segments))
    y = torch.tensor([label], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)

class FruitGraphDataset(PyGDataset):
    """Builds & caches superpixel graphs for all images in a split."""

    def __init__(self, paths: list, labels: list, cache_dir: str, split: str):
        self._paths  = paths
        self._labels = labels
        self._cache  = Path(cache_dir) / split
        self._cache.mkdir(parents=True, exist_ok=True)
        super().__init__(root=None)

    def len(self): return len(self._paths)

    def get(self, idx):
        cache_file = self._cache / f"{idx}.pt"
        if cache_file.exists():
            return torch.load(cache_file, weights_only=False)
        data = image_to_graph(self._paths[idx], self._labels[idx])
        torch.save(data, cache_file)
        return data

def build_loaders(tr_p, val_p, te_p, tr_l, val_l, te_l,
                  cache_dir: str, batch_size: int = 32):
    print("  Building graph datasets (cached after first run) …")
    tr_ds  = FruitGraphDataset(tr_p,  tr_l,  cache_dir, "train")
    val_ds = FruitGraphDataset(val_p, val_l, cache_dir, "val")
    te_ds  = FruitGraphDataset(te_p,  te_l,  cache_dir, "test")
    tr_ld  = PyGLoader(tr_ds,  batch_size=batch_size, shuffle=True)
    val_ld = PyGLoader(val_ds, batch_size=batch_size)
    te_ld  = PyGLoader(te_ds,  batch_size=batch_size)
    return tr_ld, val_ld, te_ld

class FruitGCN(nn.Module):
    """
    Two-layer GCN with global mean pooling for graph-level classification.

    GCN update: H' = ReLU(D^{-1/2} A D^{-1/2} H W)
    (handled internally by GCNConv)
    """
    def __init__(self, in_dim=NODE_DIM, hidden=HIDDEN_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv1 = GCNConv(in_dim,  hidden)
        self.conv2 = GCNConv(hidden,  hidden)
        self.conv3 = GCNConv(hidden,  hidden // 2)
        self.head  = nn.Sequential(
            nn.Linear(hidden // 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.head(x)

def train_gnn(model: FruitGCN, tr_ld, val_ld,
              epochs: int = 40, lr: float = 1e-3,
              device: str = "cpu") -> list:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    counts   = torch.tensor([184., 179., 281., 347.])
    weights  = (counts.sum() / (len(counts) * counts)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    history = []
    best_val_acc, best_state = 0.0, None

    for epoch in range(1, epochs + 1):

        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch in tr_ld:
            batch = batch.to(device)
            optimizer.zero_grad()
            out  = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            correct    += (out.argmax(1) == batch.y).sum().item()
            total      += batch.num_graphs
        tr_acc  = correct / total
        tr_loss = total_loss / total

        val_acc, val_loss = evaluate_gnn(model, val_ld, criterion, device)
        scheduler.step(val_loss)

        history.append({"epoch": epoch, "tr_loss": tr_loss, "tr_acc": tr_acc,
                         "val_loss": val_loss, "val_acc": val_acc})
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | tr_loss {tr_loss:.4f} tr_acc {tr_acc:.3f}"
                  f" | val_loss {val_loss:.4f} val_acc {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
        print(f"  Restored best model (val_acc={best_val_acc:.3f})")
    return history

@torch.no_grad()
def evaluate_gnn(model, loader, criterion=None, device="cpu"):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    for batch in loader:
        batch = batch.to(device)
        out   = model(batch.x, batch.edge_index, batch.batch)
        loss  = criterion(out, batch.y)
        total_loss += loss.item() * batch.num_graphs
        correct    += (out.argmax(1) == batch.y).sum().item()
        total      += batch.num_graphs
    return correct / total, total_loss / total

@torch.no_grad()
def predict_gnn(model, loader, device="cpu"):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        out   = model(batch.x, batch.edge_index, batch.batch)
        all_preds.extend(out.argmax(1).cpu().tolist())
        all_labels.extend(batch.y.cpu().tolist())
    return np.array(all_labels), np.array(all_preds)
