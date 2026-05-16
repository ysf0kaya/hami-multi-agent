"""
NLP Agent — Kategori: CPU_YOGUN
--------------------------------
HuggingFace'den BERT tabanlı modeller yükler ve
batch text classification görevi çalıştırır.

Desteklenen modeller (MODEL_NAME env ile seçilir):
  - bert-base      : bert-base-uncased
  - distilbert     : distilbert-base-uncased
  - roberta        : cardiffnlp/twitter-roberta-base-sentiment
  - bert-tiny      : prajjwal1/bert-tiny
  - albert-base    : albert-base-v2
"""

import os
import time
import logging
from transformers import pipeline
from data.collector import MetricCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("NLPAgent")

MODELS = {
    "bert-base":   "bert-base-uncased",
    "distilbert":  "distilbert-base-uncased",
    "roberta":     "cardiffnlp/twitter-roberta-base-sentiment",
    "bert-tiny":   "prajjwal1/bert-tiny",
    "albert-base": "albert-base-v2",
}

BATCH_TEXTS_MASK = [
    "The [MASK] is very important for machine learning.",
    "Kubernetes [MASK] container orchestration at scale.",
    "The weather today is [MASK] and sunny.",
    "GPU [MASK] enables faster model training.",
    "Deep learning [MASK] require large datasets.",
    "The [MASK] arrived late and was damaged.",
    "Cloud [MASK] provides scalable infrastructure.",
    "Neural [MASK] can learn complex patterns.",
] * 4  # 32 metin

MODEL_KEY  = os.getenv("MODEL_NAME", "bert-tiny")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
REPEAT     = int(os.getenv("REPEAT", "5"))
AGENT_ID   = os.getenv("AGENT_ID", f"nlp-{MODEL_KEY}")
CATEGORY   = "CPU_YOGUN"


def run():
    model_name = MODELS.get(MODEL_KEY)
    if not model_name:
        raise ValueError(f"Bilinmeyen model: {MODEL_KEY}. Seçenekler: {list(MODELS.keys())}")

    logger.info(f"Model yükleniyor: {model_name}")

    if MODEL_KEY == "roberta":
        pipe = pipeline("text-classification", model=model_name, device=0)
        batch_data = [t.replace("[MASK]", "really") for t in BATCH_TEXTS_MASK]
    else:
        pipe = pipeline("fill-mask", model=model_name, device=0)
        batch_data = BATCH_TEXTS_MASK

    logger.info(f"Model yüklendi ✅ — {model_name}")

    collector = MetricCollector(
        agent_id=AGENT_ID,
        category=CATEGORY,
        model_name=model_name,
    )
    collector.start()

    for i in range(REPEAT):
        logger.info(f"Batch {i+1}/{REPEAT} başlıyor...")
        start = time.time()
        pipe(batch_data, batch_size=BATCH_SIZE)
        duration = time.time() - start
        collector.record_batch(duration_s=duration)
        logger.info(f"Batch {i+1} tamamlandı — {duration:.2f}s")

    collector.stop()
    summary = collector.save()
    logger.info(f"Sonuç kaydedildi: {summary}")
    logger.info("Agent görevi tamamladı, çıkıyor...")
    # sleep yok — pod kapanır, Completed olur


if __name__ == "__main__":
    run()
