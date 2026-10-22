# Produce Condition & Ripening Stage Detection Pipeline
### Comparative Computer Vision Study: Classical CV (CIELAB + LBP + SVM/RF) vs. Graph Neural Network (GCN on Superpixel RAG)

---

## Executive Summary

Automated post-harvest fruit grading is critical for shelf-life optimization and active packaging deployment (such as GreenPod Labs' biochemical storage technologies). This project implements an end-to-end computer vision pipeline comparing two fundamentally different paradigms on produce condition and ripening classification:

1. **Approach 1 (Physics-Inspired Classical Feature Engineering + ML):** Exploiting perceptual color dynamics (**CIELAB** $a^*$ green-to-red axis, **HSV** Hue) and micro-surface roughness (**Uniform Local Binary Patterns**) classified by **RBF-Kernel Support Vector Machines** and **Random Forests**.
2. **Approach 2 (Geometric Deep Learning / Graph Neural Networks):** Transforming images into **Region Adjacency Graphs (RAG)** of **SLIC superpixels**, allowing a 3-layer **Graph Convolutional Network (GCN)** to reason over spatial relationships between healthy tissue and necrotic lesion boundaries.

| Stage / Class | Biological Characteristics | Agricultural Significance |
| :--- | :--- | :--- |
| **`fresh`** | Uniform orange pigmentation, smooth intact cuticle | Optimal market ripeness; immediate packaging/dispatch |
| **`grenning`** | High chlorophyll concentration, irregular green patches | Underripe or affected by citrus greening; requires degreening |
| **`blackspot`** | Necrotic sunken fungal spots (*Guignardia citricarpa*) | Pathological infection; quarantine risk during export |
| **`canker`** | Raised corky eruptive lesions (*Xanthomonas citri*) | Bacterial disease; accelerates internal ethylene and rot |

---

## 1. Deep Dataset Understanding & Bias Analysis

### 1.1 Class Imbalance & Statistical Distribution

The dataset comprises **1,090 total images** partitioned into a **991-image training set** and an independent **99-image test set**.

<p align="center">
  <img src="models/eda_class_distribution.png" width="800" alt="Class Distribution" />
</p>

| Class Name | Train Count | Test Count | Total | Class Share | Imbalance Weight ($\mathcal{W}_c$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`blackspot`** | 184 | 22 | 206 | 18.9% | 1.346 |
| **`canker`** | 179 | 22 | 201 | 18.4% | 1.384 |
| **`fresh`** | 281 | 33 | 314 | 28.8% | 0.882 |
| **`grenning`** | 347 | 22 | 369 | 33.9% | 0.714 |
| **Total** | **991** | **99** | **1,090** | **100%** | **Balanced** |

#### Imbalance Analysis:
- The dataset exhibits a **1.94× imbalance ratio** between the majority class (`grenning`: 347) and minority class (`canker`: 179).
- Without correction, gradient descent or margin maximization favors the majority class. We resolve this by applying **inverse-frequency class weighting**:
  $$\mathcal{W}_c = \frac{N}{C \cdot N_c}$$
  where $N=991$, $C=4$, and $N_c$ is the frequency of class $c$.

---

### 1.2 Visual Inspection & Sample Representation

<p align="center">
  <img src="models/eda_sample_images.png" width="850" alt="Sample Images per Class" />
</p>

- **Color Transition:** `fresh` vs. `grenning` is dominated by the degradation of chlorophyll pigments into carotenoids.
- **Lesion Ambiguity:** `blackspot` and `canker` share dark brown necrotic centroids. However, cankers exhibit raised, corky margins with yellow halos, whereas blackspot exhibits discrete, flat pustules.

---

### 1.3 Color Channel Biases & Illumination Invariance

In standard RGB color space, channel values $R, G, B$ correlate strongly with illuminance ($E$). A shadow or bright reflection shifts all three values simultaneously:

<p align="center">
  <img src="models/eda_lab_distributions.png" width="900" alt="LAB Distributions" />
</p>

#### Mathematical Formulation of CIELAB:
The **CIE $L^*a^*b^*$** color space explicitly decouples perceptual lightness ($L^*$) from chromatic opposition ($a^*$ green $\leftrightarrow$ red; $b^*$ blue $\leftrightarrow$ yellow):

$$L^* = 116 f(Y/Y_n) - 16$$
$$a^* = 500 \left[ f(X/X_n) - f(Y/Y_n) \right]$$
$$b^* = 200 \left[ f(Y/Y_n) - f(Z/Z_n) \right]$$

- As observed in the histogram above, the **$a^*$ channel acts as an ideal zero-bias discriminator**: `grenning` clusters sharply around negative values (green), while `fresh` shifts far into positive values (orange/red).
- The **$L^*$ channel** normalizes uneven ambient illumination and camera flash specularities.

---

### 1.4 Resolution Invariance

<p align="center">
  <img src="models/eda_resolution.png" width="600" alt="Resolution Distribution" />
</p>

- Images range from $200 \times 200$ to $600 \times 600$ pixels.
- All inputs are standardized via bilinear interpolation to fixed dimensions ($128 \times 128$ for classical and graph extraction, $224 \times 224$ for deep learning) ensuring spatial scale consistency.

---

## 2. Approach 1: Classical CV & Machine Learning

### 2.1 Feature Engineering Architecture (146 Dimensions)

Each image is converted into a 146-dimensional hand-crafted descriptor:

```
Input Image (128x128)
  ├── CIELAB Color Space ──> 3-Channel 32-bin Histograms ─────>  96 features
  ├── HSV Color Space    ──> Hue Channel 32-bin Histogram ─────>  32 features
  └── Grayscale          ──> Uniform LBP (P=16, R=2) Hist ─────>  18 features
                                                Total Descriptor: 146 dimensions
```

#### Texture Encoding via Uniform Local Binary Patterns (LBP):
For each pixel $p$ at $(x_c, y_c)$, LBP samples $P=16$ equidistant neighbors on a circle of radius $R=2$:

$$LBP_{P, R} = \sum_{p=0}^{P-1} s(i_p - i_c) 2^p, \quad s(x) = \begin{cases} 1 & x \ge 0 \\ 0 & x < 0 \end{cases}$$

A pattern is deemed "uniform" if it contains at most two bitwise $0 \to 1$ or $1 \to 0$ transitions. This provides rotation-invariant texture descriptors that capture the raised, cratered roughness of `canker` while ignoring lighting gradients.

### 2.2 Classifiers & Results

We evaluate both an **RBF Support Vector Machine** and an ensemble **Random Forest (300 estimators)**:

<p align="center">
  <img src="models/svm_confusion_matrix.png" width="400" alt="SVM Confusion Matrix" />
  <img src="models/rf_confusion_matrix.png" width="400" alt="RF Confusion Matrix" />
</p>

- **SVM (RBF Kernel, $C=10$):** Projects the 146-d vector into a reproducing kernel Hilbert space:
  $$K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2)$$
  Achieves **97.98% Accuracy** and **0.9773 Macro F1**, with only 2 misclassifications across all 99 test images.
- **Random Forest:** Achieves **95.96% Accuracy** and **0.9545 Macro F1**, demonstrating strong robustness against feature collinearity.

---

## 3. Approach 2: Graph Neural Network (GCN on Superpixels)

### 3.1 Why Graph Neural Networks for Produce Inspection?

Standard CNNs treat images as uniform rigid grids. However, fruit diseases manifest as **irregular topological subregions**: a necrotic spot surrounded by healthy rind.

<p align="center">
  <img src="models/gnn_training_curve.png" width="750" alt="GCN Training Dynamics" />
</p>

### 3.2 Graph Construction Pipeline:
1. **SLIC Superpixel Segmentation:** The image is partitioned into $k=75$ compact, color-homogeneous superpixels using k-means in 5D $(L, a, b, x, y)$ space with compactness factor $m=10$.
2. **Region Adjacency Graph (RAG):** Graph nodes $\mathcal{V}$ represent superpixels. Edges $\mathcal{E}$ connect physically adjacent superpixels.
3. **16-Dimensional Node Feature Vectors:**
   - Mean $L^*a^*b^*$ coordinates ($3$-d)
   - Mean $HSV$ coordinates ($3$-d)
   - Normalized 10-bin LBP texture histogram ($10$-d)

### 3.3 GCN Architecture (FruitGCN):
The network executes spatial message passing across 3 layers:

$$\mathbf{H}^{(l+1)} = \sigma \left( \mathbf{\tilde{D}}^{-\frac{1}{2}} \mathbf{\tilde{A}} \mathbf{\tilde{D}}^{-\frac{1}{2}} \mathbf{H}^{(l)} \mathbf{W}^{(l)} \right)$$

- **Layer 1:** `GCNConv(16 -> 64)` + ReLU + Dropout(0.2)
- **Layer 2:** `GCNConv(64 -> 64)` + ReLU
- **Layer 3:** `GCNConv(64 -> 32)` + ReLU
- **Graph Readout:** Permutation-invariant Global Mean Pooling:
  $$\mathbf{h}_G = \frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} \mathbf{h}_i$$
- **Classification Head:** Linear(32 $\to$ 4) + Cross-Entropy Loss with inverse class weights.

<p align="center">
  <img src="models/gnn_confusion_matrix.png" width="420" alt="GCN Confusion Matrix" />
</p>

- **Performance:** **85.86% Test Accuracy** and **0.8287 Macro F1**.
- **Analysis:** GCN separates `fresh` and `grenning` with 100% recall. On subtle `canker` lesions, averaging over $k=75$ superpixels occasionally over-smooths small lesion boundaries, leading to confusion with `blackspot`.

---

## 4. Quantitative Comparison & Evaluation Metrics

| Metric | Approach 1: SVM (RBF) | Approach 1: Random Forest | Approach 2: Superpixel GCN |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | **97.98%** | 95.96% | 85.86% |
| **Macro Precision** | **0.9800** | 0.9545 | 0.8943 |
| **Macro Recall** | **0.9773** | 0.9545 | 0.8398 |
| **Macro F1-Score** | **0.9773** | 0.9545 | 0.8287 |
| **Weighted F1-Score** | **0.9798** | 0.9596 | 0.8441 |
| **Inference Time (CPU)** | **~35 ms** | ~45 ms | ~110 ms |
| **Inference Time (GPU)** | ~30 ms | ~40 ms | **~12 ms** |
| **Model Size** | 318 KB | 2.35 MB | **38 KB** |

### Why Macro F1-Score Decided the Best Model:
While raw accuracy can be inflated by majority classes, **Macro F1-Score** evaluates the unweighted harmonic mean across all categories:

$$\text{Macro F1} = \frac{1}{C} \sum_{c=1}^C \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$

SVM achieved **0.9773 Macro F1**, demonstrating reliable sensitivity on diseased fruit (`blackspot` and `canker`) without generating false alarms on healthy produce.

---

## 5. Live Demonstration: Dual-Model Single-Run Inference

Using [`main_demo.py`](main_demo.py), an input image is evaluated by both trained models simultaneously:

```bash
python main_demo.py "test/fresh/f (1).png"
```

<p align="center">
  <img src="demo_result.png" width="850" alt="Dual-Model Prediction Demo" />
</p>

**Execution Output:**
```text
Image: test/fresh/f (1).png
--------------------------------------------------
[Approach 1 — SVM]      -> fresh  (conf: 99.66%)
[Approach 2 — GCN]      -> fresh  (conf: 99.95%)
--------------------------------------------------
Visualization saved -> demo_result.png
```

---

## 6. Answers to Mandatory Assignment Questions

### Question 1: In production how your model performs?
- **Throughput & Speed:**
  - **Approach 1 (SVM):** Runs in **30–45 ms on CPU**, comfortably supporting industrial grading lines operating at 20–30 items per second without requiring expensive GPU hardware.
  - **Approach 2 (GCN):** Incurs ~110 ms CPU latency due to CPU-based SLIC superpixel clustering (~80 ms) and GCN inference (~30 ms). On edge accelerators (NVIDIA Jetson / TensorRT), GNN message-passing executes in **12 ms**.
- **Environmental Stability:** The CIELAB transformation eliminates ambient illuminance fluctuations. However, severe specular highlights from intense sorting lights require diffuse polarization filters.

### Question 2: What is the drawback of your model?
- **Single-Surface Perspective:** Monocular 2D imaging captures only the camera-facing hemisphere. Internal rot, firmness degradation, or lesions on the hidden side cannot be diagnosed.
- **Fixed Superpixel Granularity:** The GCN pipeline segments images into fixed $k=75$ superpixels. Minor bacterial lesions ($< 5$ pixels) get smoothed into healthy background rind.
- **Continuous Ripening Boundary:** Fruit ripening is a continuous biochemical spectrum; rigid 4-class categorical labels introduce noise on borderline fruit.

### Question 3: If given more time, what would you improve?
1. **Multi-View 360° Roller Capture:** Use synchronized dual-camera setups with roller conveyors to reconstruct the entire 360-degree surface texture.
2. **Deep Vision GNN (ViG):** Replace hand-crafted SLIC superpixels with self-supervised patch tokens (e.g. Vision GNN / DINOv2) trained end-to-end.
3. **Multi-Modal NIR Spectroscopy:** Fuse RGB vision with Near-Infrared sensors (700–1000 nm) to assess internal sugar content (Brix), acidity, and flesh firmness non-destructively.
4. **ONNX INT8 Quantization:** Quantize the models for direct deployment on ultra-low-power industrial microcontrollers.

### Question 4: Where you spend more time, during solving your assignment?
- **Feature Engineering & Color Space Tuning (~40% of time):** Formulating CIELAB $a^*$ and HSV Hue descriptors to reliably isolate chlorophyll decay from healthy orange rind and dark lesions.
- **Graph Formulation & GNN Pipeline (~35% of time):** Constructing superpixel Region Adjacency Graphs (RAG) and configuring message passing on irregular topologies without over-smoothing.
- **Model Evaluation, Benchmarking & Reporting (~25% of time):** Auditing confusion matrices, optimizing hyperparameters, and synthesizing quantitative findings.

---

## 7. Quick Start & Execution Guide

### Local Environment
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run Exploratory Data Analysis
python eda.py

# 3. Train Approach 1 (Classical SVM & Random Forest)
python train_classical.py

# 4. Train Approach 2 (Superpixel GCN)
python train_gnn.py

# 5. Run Live Inference Demo
python main_demo.py "test/fresh/f (1).png"
```

### Google Colab (One-Click)
Open the complete interactive notebook:
 **[Open in Google Colab](https://colab.research.google.com/github/Sanjay6968/datat/blob/main/notebooks/Avocado_Orange_Ripening_Pipeline.ipynb)**
