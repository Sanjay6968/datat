"""
classical_model.py — Approach 1: CIELAB + HSV + LBP features -> SVM + Random Forest.

Feature vector (per image, 146-d):
  96-d : LAB channel histograms (32 bins × 3 channels)
  32-d : HSV hue histogram (32 bins)
  18-d : LBP uniform histogram (P=16, R=2)
"""
import cv2
import numpy as np
import joblib
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def extract_features(image_path: str, size: int = 128) -> np.ndarray:
    """Return 146-d color+texture feature vector for one image."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read: {image_path}")
    img = cv2.resize(img, (size, size))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_feats = np.concatenate([
        cv2.calcHist([lab], [ch], None, [32], [0, 256]).flatten()
        for ch in range(3)
    ])

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_feats = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp  = local_binary_pattern(gray, P=16, R=2, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 19), range=(0, 18))
    lbp_feats = lbp_hist.astype(float)

    return np.concatenate([lab_feats, hsv_feats, lbp_feats])

def build_feature_matrix(paths: list) -> np.ndarray:
    print(f"  Extracting features from {len(paths)} images …")
    feats = []
    for i, p in enumerate(paths):
        feats.append(extract_features(p))
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(paths)}")
    return np.stack(feats)

def make_svm() -> Pipeline:
    """RBF-SVM with class-weight balancing; Platt calibration for probabilities."""
    base = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(base, cv=3, ensemble=False)),
    ])

def make_rf() -> Pipeline:
    """Random Forest — robust to noisy features, gives feature importance."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            max_features="sqrt", random_state=42, n_jobs=-1,
        )),
    ])

def train_and_eval(model: Pipeline, X_tr, y_tr, X_te, y_te,
                   class_names: list, name: str) -> dict:
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    report = classification_report(y_te, y_pred, target_names=class_names,
                                   output_dict=True)
    cm = confusion_matrix(y_te, y_pred)
    print(f"\n{'='*52}\n  {name} — Test Results\n{'='*52}")
    print(classification_report(y_te, y_pred, target_names=class_names))
    print("Confusion Matrix:\n", cm)
    return {"name": name, "accuracy": acc, "report": report,
            "cm": cm, "y_pred": y_pred}

def save_model(model: Pipeline, path: str):
    joblib.dump(model, path)
    print(f"  Saved -> {path}")

def load_model(path: str) -> Pipeline:
    return joblib.load(path)
