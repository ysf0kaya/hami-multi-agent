"""
Sınıflandırma Modeli Eğitimi
-----------------------------
HAMi'den toplanan metriklerle agent kategorisini tahmin eder.

Kategoriler:
  - CPU_YOGUN
  - GPU_YOGUN
  - IO_YOGUN
"""

import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Train")

# ── Ayarlar ───────────────────────────────────────────────────────────────────
DATASET_PATH = Path("/home/ysf/hami-dataset/dataset.csv")
MODEL_DIR    = Path("model/saved")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Sınıflandırma için kullanılacak özellikler
FEATURES = [
    "avg_gpu_mem_mb",
    "max_gpu_mem_mb",
    "avg_gpu_compute_percent",
    "avg_cpu_percent",
    "max_cpu_percent",
    "avg_ram_gb",
    "avg_batch_duration_s",
    "throughput_per_s",
]
TARGET = "category"


def load_data():
    df = pd.read_csv(DATASET_PATH)
    logger.info(f"Dataset yüklendi: {len(df)} satır, {df[TARGET].value_counts().to_dict()}")
    return df


def preprocess(df):
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # Label encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info(f"Kategoriler: {dict(zip(le.classes_, range(len(le.classes_))))}")
    return X_scaled, y_encoded, le, scaler


def train():
    # 1. Veri yükle
    df = load_data()

    # 2. Ön işleme
    X, y, le, scaler = preprocess(df)

    # 3. Train/Test/Val böl
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    logger.info(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # 4. Model eğit
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    logger.info("Model eğitildi ✅")

    # 5. Validation değerlendirme
    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    logger.info(f"Validation Accuracy: {val_acc:.4f}")

    # 6. Test değerlendirme
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)

    logger.info(f"\nTest Accuracy: {test_acc:.4f}")
    logger.info(f"\nClassification Report:\n{classification_report(y_test, test_pred, target_names=le.classes_)}")
    logger.info(f"\nConfusion Matrix:\n{confusion_matrix(y_test, test_pred)}")

    # 7. Feature importance
    importances = pd.Series(model.feature_importances_, index=FEATURES)
    logger.info(f"\nFeature Importance:\n{importances.sort_values(ascending=False)}")

    # 8. Model kaydet
    pickle.dump(model,  open(MODEL_DIR / "model.pkl", "wb"))
    pickle.dump(le,     open(MODEL_DIR / "label_encoder.pkl", "wb"))
    pickle.dump(scaler, open(MODEL_DIR / "scaler.pkl", "wb"))

    logger.info(f"Model kaydedildi → {MODEL_DIR}")
    return model, le, scaler


if __name__ == "__main__":
    train()
