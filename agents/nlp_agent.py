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

# ── Model Kataloğu ────────────────────────────────────────────────────────────
MODELS = {
    "bert-base":   "bert-base-uncased",
    "distilbert":  "distilbert-base-uncased",
    "roberta":     "cardiffnlp/twitter-roberta-base-sentiment",
    "bert-tiny":   "prajjwal1/bert-tiny",
    "albert-base": "albert-base-v2",
}

# ── Sabit Batch Verisi ────────────────────────────────────────────────────────
BATCH_TEXTS = [
    "This product is absolutely amazing and works perfectly.",
    "I hate this service, worst experience ever.",
    "The weather today is quite nice and sunny.",
    "Kubernetes simplifies container orchestration greatly.",
    "This movie was boring and too long.",
    "Incredible performance, highly recommended to everyone.",
    "The package arrived late and damaged.",
    "GPU virtualization enables better resource utilization.",
] * 4  # 32 metin

# ── Ayarlar ───────────────────────────────────────────────────────────────────
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
    pipe = pipeline(
        "fill-mask" if "bert" in model_name or "albert" in model_name else "text-classification",
        model=model_name,
        device=0,
        truncation=True,
    )
    logger.info(f"Model yüklendi ✅ — {model_name}")

    # fill-mask için batch veriyi uyarla
    if pipe.task == "fill-mask":
        batch_data = [t.split()[0] + " [MASK] " + " ".join(t.split()[1:])
                      for t in BATCH_TEXTS]
    else:
        batch_data = BATCH_TEXTS

    collector = MetricCollector(
        agent_id=AGENT_ID,
        category=CATEGORY,
        model_name=model_name,
    )
    collector.start()

    for i in range(REPEAT):
        logger.info(f"Batch {i+1}/{REPEAT} başlıyor...")
        start = time.time()

        pipe(batch_data, batch_size=BATCH_SIZE, truncation=True)

        duration = time.time() - start
        collector.record_batch(duration_s=duration)
        logger.info(f"Batch {i+1} tamamlandı — {duration:.2f}s")

    collector.stop()
    summary = collector.save()
    logger.info(f"Sonuç kaydedildi: {summary}")

    logger.info("Agent bekleme moduna geçti...")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    run()
