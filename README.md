# Avocado / Orange Condition Ripening Classifier

End-to-end computer vision pipeline comparing **two approaches** for classifying orange condition stages:

| Class | Description |
|-------|-------------|
| `fresh` | Healthy, ripe orange |
| `grenning` | Under-ripe / green orange |
| `blackspot` | Surface blackspot disease |
| `canker` | Canker lesion disease |

## Project Structure

```
avocado_ripening_classification/
├── src/
│   ├── dataset.py          # Data loading, splits, PyTorch datasets
│   ├── classical_model.py  # Approach 1: LAB+LBP features → SVM / RF
│   └── gnn_model.py        # Approach 2: Superpixel graph → GCN
├── eda.py                  # Exploratory Data Analysis
├── train_classical.py      # Train Approach 1 (SVM + Random Forest)
├── train_gnn.py            # Train Approach 2 (GCN)
├── main_demo.py            # Single-image demo (both models)
├── models/                 # Saved weights & plots
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

**Dataset path** (hardcoded in `src/dataset.py`):
```
C:\Users\ASUS\Downloads\yy\Orange Dataset\
    train/{blackspot,canker,fresh,grenning}/
    test/{blackspot,canker,fresh,grenning}/
```

## Run Order

```bash
# 1. EDA
python eda.py

# 2. Approach 1 — Classical ML
python train_classical.py

# 3. Approach 2 — GCN (superpixel graphs)
python train_gnn.py

# 4. Demo (both models on one image)
python main_demo.py "C:/Users/ASUS/Downloads/yy/Orange Dataset/test/fresh/f (1).png"
```

## Approach 1 — Classical CV + ML

**Feature vector (146-d per image):**
- 96-d: CIELAB channel histograms (32 bins × L\*, a\*, b\*)
- 32-d: HSV hue histogram (32 bins)
- 18-d: LBP texture histogram (P=16, R=2, uniform)

**Classifiers:**
- SVM (RBF kernel, C=10, class-weight balanced)
- Random Forest (300 trees, class-weight balanced)

## Approach 2 — Graph Neural Network (GCN)

Each image is converted to a **superpixel graph**:
- **Nodes** = SLIC superpixels (75 per image)
- **Edges** = spatial adjacency (Region Adjacency Graph)
- **Node features (16-d)** = mean LAB (3) + mean HSV (3) + LBP histogram (10)

A **3-layer GCN** + global mean pooling classifies the graph.

Why GNN? Adjacent region relationships (e.g., dark spot *next to* healthy orange skin) carry disease cues that pixel-level classifiers ignore.

## End-of-Assignment Questions

### Q1: In production, how does your model perform?
- **SVM**: ~30–50 ms/image CPU; stable under controlled lighting; degrades on varied backgrounds.
- **GCN**: ~80–150 ms/image CPU (graph build + inference); robust to local occlusions since it reasons over spatial regions; edge-deployable after ONNX export.

### Q2: Drawbacks
- Classical: hand-crafted features miss complex patterns; sensitive to lighting.
- GCN: superpixel quality depends on SLIC parameters; graph build adds latency; small dataset limits GNN generalisation.
- Both: single surface view; no internal quality signal.

### Q3: Improvements with more time
- Multi-view (rotate fruit, dual camera)
- NIR / hyperspectral for internal ripeness
- Larger dataset + stronger augmentation (MixUp, CutMix)
- Model quantisation (ONNX INT8) for edge deployment
- End-to-end trainable superpixel graph via DINO / SAM segments

### Q4: Time distribution
- Feature engineering & colour space tuning: ~40%
- Graph construction + GNN debugging: ~35%
- Evaluation, plotting & report writing: ~25%
