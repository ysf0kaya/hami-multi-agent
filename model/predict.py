"""
Yeni Agent Tahmini
-------------------
Eğitilmiş modeli kullanarak yeni bir agent'ın
HAMi metriklerine bakarak kategorisini tahmin eder.
"""

import pickle
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Predict")

MODEL_DIR = Path("model/saved")

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


def load_model():
    model  = pickle.load(open(MODEL_DIR / "model.pkl", "rb"))
    le     = pickle.load(open(MODEL_DIR / "label_encoder.pkl", "rb"))
    scaler = pickle.load(open(MODEL_DIR / "scaler.pkl", "rb"))
    return model, le, scaler


def predict(metrics: dict) -> dict:
    """
    Verilen metriklerden agent kategorisini tahmin eder.

    Args:
        metrics: HAMi'den toplanan metrik dict'i

    Returns:
        category: tahmin edilen kategori
        confidence: güven skoru
        probabilities: tüm kategoriler için olasılıklar
    """
    model, le, scaler = load_model()

    df = pd.DataFrame([metrics])[FEATURES]
    X_scaled = scaler.transform(df)

    pred_idx = model.predict(X_scaled)[0]
    pred_proba = model.predict_proba(X_scaled)[0]

    category = le.inverse_transform([pred_idx])[0]
    confidence = pred_proba[pred_idx]

    probabilities = {
        cls: round(float(prob), 4)
        for cls, prob in zip(le.classes_, pred_proba)
    }

    result = {
        "category": category,
        "confidence": round(float(confidence), 4),
        "probabilities": probabilities,
    }

    logger.info(f"Tahmin: {category} (güven: {confidence:.2%})")
    logger.info(f"Olasılıklar: {probabilities}")

    return result


if __name__ == "__main__":
    # Test: bilinen bir agent'ın metriklerini kullan
    test_cases = [
        {
            "name": "Yeni LLM Agent (GPU_YOGUN bekleniyor)",
            "metrics": {
                "avg_gpu_mem_mb": 400.0,
                "max_gpu_mem_mb": 400.0,
                "avg_gpu_compute_percent": 40.0,
                "avg_cpu_percent": 8.0,
                "max_cpu_percent": 20.0,
                "avg_ram_gb": 5.7,
                "avg_batch_duration_s": 0.6,
                "throughput_per_s": 1.5,
            }
        },
        {
            "name": "Yeni NLP Agent (CPU_YOGUN bekleniyor)",
            "metrics": {
                "avg_gpu_mem_mb": 50.0,
                "max_gpu_mem_mb": 50.0,
                "avg_gpu_compute_percent": 10.0,
                "avg_cpu_percent": 55.0,
                "max_cpu_percent": 70.0,
                "avg_ram_gb": 5.5,
                "avg_batch_duration_s": 0.15,
                "throughput_per_s": 7.0,
            }
        },
        {
            "name": "Yeni Vision Agent (IO_YOGUN bekleniyor)",
            "metrics": {
                "avg_gpu_mem_mb": 100.0,
                "max_gpu_mem_mb": 100.0,
                "avg_gpu_compute_percent": 12.0,
                "avg_cpu_percent": 10.0,
                "max_cpu_percent": 25.0,
                "avg_ram_gb": 5.8,
                "avg_batch_duration_s": 0.07,
                "throughput_per_s": 14.0,
            }
        },
    ]

    for tc in test_cases:
        print(f"\n{'='*50}")
        print(f"Test: {tc['name']}")
        result = predict(tc["metrics"])
        print(f"Sonuç: {result['category']} (güven: {result['confidence']:.2%})")
