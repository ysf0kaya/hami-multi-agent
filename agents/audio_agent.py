"""
Audio Agent — Kategori: KARMA
------------------------------
HuggingFace'den audio modelleri yükler ve
batch speech/audio işleme görevi çalıştırır.
Gerçek ses yerine rastgele numpy array kullanır.

Desteklenen modeller (MODEL_NAME env ile seçilir):
  - whisper-tiny   : openai/whisper-tiny
  - whisper-base   : openai/whisper-base
  - wav2vec2-base  : facebook/wav2vec2-base
"""

import os
import time
import logging
import torch
import numpy as np
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    Wav2Vec2Processor, Wav2Vec2Model,
)
from data.collector import MetricCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("AudioAgent")

# ── Model Kataloğu ────────────────────────────────────────────────────────────
MODELS = {
    "whisper-tiny":  "openai/whisper-tiny",
    "whisper-base":  "openai/whisper-base",
    "wav2vec2-base": "facebook/wav2vec2-base",
}

# ── Ayarlar ───────────────────────────────────────────────────────────────────
MODEL_KEY   = os.getenv("MODEL_NAME", "whisper-tiny")
BATCH_SIZE  = int(os.getenv("BATCH_SIZE", "8"))  # Audio için küçük batch
REPEAT      = int(os.getenv("REPEAT", "5"))
AGENT_ID    = os.getenv("AGENT_ID", f"audio-{MODEL_KEY}")
CATEGORY    = "KARMA"
SAMPLE_RATE = 16000
DURATION_S  = 5  # 5 saniyelik ses


def generate_fake_audio(count: int) -> np.ndarray:
    """Rastgele ses verisi üret (16kHz, 5 saniye)."""
    return np.random.randn(count, SAMPLE_RATE * DURATION_S).astype(np.float32)


def run():
    model_name = MODELS.get(MODEL_KEY)
    if not model_name:
        raise ValueError(f"Bilinmeyen model: {MODEL_KEY}. Seçenekler: {list(MODELS.keys())}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Model yükleniyor: {model_name} → {device}")

    # Model tipine göre yükle
    is_whisper = "whisper" in MODEL_KEY

    if is_whisper:
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    else:
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2Model.from_pretrained(model_name).to(device)

    model.eval()
    logger.info(f"Model yüklendi ✅ — {model_name}")

    # Batch ses verisi hazırla
    fake_audio = generate_fake_audio(BATCH_SIZE)

    collector = MetricCollector(
        agent_id=AGENT_ID,
        category=CATEGORY,
        model_name=model_name,
    )
    collector.start()

    for i in range(REPEAT):
        logger.info(f"Batch {i+1}/{REPEAT} başlıyor...")
        start = time.time()

        if is_whisper:
            inputs = processor(
                fake_audio,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            ).to(device)
            with torch.no_grad():
                _ = model.generate(inputs.input_features)
        else:
            inputs = processor(
                list(fake_audio),
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            ).to(device)
            with torch.no_grad():
                _ = model(**inputs)

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
