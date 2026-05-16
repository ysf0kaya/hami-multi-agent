"""
Vision Agent — Kategori: IO_YOGUN
----------------------------------
HuggingFace'den vision modelleri yükler ve
batch image classification görevi çalıştırır.
Gerçek görüntü yerine rastgele tensor kullanır.

Desteklenen modeller (MODEL_NAME env ile seçilir):
  - vit-base     : google/vit-base-patch16-224
  - resnet-18    : microsoft/resnet-18
  - resnet-50    : microsoft/resnet-50
  - mobilenet    : google/mobilenet_v2_1.0_224
  - efficientnet : google/efficientnet-b0
"""

import os
import time
import logging
import torch
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from data.collector import MetricCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("VisionAgent")

MODELS = {
    "vit-base":     "google/vit-base-patch16-224",
    "resnet-18":    "microsoft/resnet-18",
    "resnet-50":    "microsoft/resnet-50",
    "mobilenet":    "google/mobilenet_v2_1.0_224",
    "efficientnet": "google/efficientnet-b0",
}

MODEL_KEY  = os.getenv("MODEL_NAME", "resnet-18")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
REPEAT     = int(os.getenv("REPEAT", "5"))
AGENT_ID   = os.getenv("AGENT_ID", f"vision-{MODEL_KEY}")
CATEGORY   = "IO_YOGUN"


def generate_fake_images(count: int, size: int = 224) -> list:
    return [
        np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        for _ in range(count)
    ]


def run():
    model_name = MODELS.get(MODEL_KEY)
    if not model_name:
        raise ValueError(f"Bilinmeyen model: {MODEL_KEY}. Seçenekler: {list(MODELS.keys())}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Model yükleniyor: {model_name} → {device}")

    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    logger.info(f"Model yüklendi ✅ — {model_name}")

    fake_images = generate_fake_images(BATCH_SIZE)
    inputs = extractor(images=fake_images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    collector = MetricCollector(
        agent_id=AGENT_ID,
        category=CATEGORY,
        model_name=model_name,
    )
    collector.start()

    for i in range(REPEAT):
        logger.info(f"Batch {i+1}/{REPEAT} başlıyor...")
        start = time.time()

        with torch.no_grad():
            outputs = model(**inputs)
            _ = outputs.logits.argmax(dim=-1)

        duration = time.time() - start
        collector.record_batch(duration_s=duration)
        logger.info(f"Batch {i+1} tamamlandı — {duration:.2f}s")

    collector.stop()
    summary = collector.save()
    logger.info(f"Sonuç kaydedildi: {summary}")
    logger.info("Agent bekleme moduna geçti...")
    while True:
        time.sleep(60)


if __name__ == "__main__":
    run()
