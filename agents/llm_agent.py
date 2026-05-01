"""
LLM Agent — Kategori: GPU_YOGUN
--------------------------------
HuggingFace'den küçük dil modelleri yükler ve
batch text generation görevi çalıştırır.

Desteklenen modeller (MODEL_NAME env ile seçilir):
  - qwen-0.5b    : Qwen/Qwen1.5-0.5B
  - tinyllama    : TinyLlama/TinyLlama-1.1B-Chat-v1.0
  - distilgpt2   : distilgpt2
  - lamini-gpt   : MBZUAI/LaMini-GPT-124M
  - opt-125m     : facebook/opt-125m
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
logger = logging.getLogger("LLMAgent")

# ── Model Kataloğu ────────────────────────────────────────────────────────────
MODELS = {
    "qwen-0.5b":  "Qwen/Qwen1.5-0.5B",
    "tinyllama":  "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "distilgpt2": "distilgpt2",
    "lamini-gpt": "MBZUAI/LaMini-GPT-124M",
    "opt-125m":   "facebook/opt-125m",
}

# ── Sabit Batch Verisi ────────────────────────────────────────────────────────
BATCH_PROMPTS = [
    "Artificial intelligence is",
    "Kubernetes is a system for",
    "Machine learning helps us",
    "The future of computing is",
    "Neural networks can learn",
    "Deep learning models are",
    "GPU acceleration enables",
    "Cloud computing provides",
] * 4  # 32 prompt

# ── Ayarlar ───────────────────────────────────────────────────────────────────
MODEL_KEY   = os.getenv("MODEL_NAME", "distilgpt2")
BATCH_SIZE  = int(os.getenv("BATCH_SIZE", "32"))
REPEAT      = int(os.getenv("REPEAT", "5"))
AGENT_ID    = os.getenv("AGENT_ID", f"llm-{MODEL_KEY}")
CATEGORY    = "GPU_YOGUN"


def run():
    model_name = MODELS.get(MODEL_KEY)
    if not model_name:
        raise ValueError(f"Bilinmeyen model: {MODEL_KEY}. Seçenekler: {list(MODELS.keys())}")

    logger.info(f"Model yükleniyor: {model_name}")
    pipe = pipeline(
        "text-generation",
        model=model_name,
        device=0,           # GPU
        truncation=True,
    )
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

        pipe(
            BATCH_PROMPTS,
            max_new_tokens=50,
            batch_size=BATCH_SIZE,
            do_sample=True,
            temperature=0.7,
            pad_token_id=pipe.tokenizer.eos_token_id,
        )

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
