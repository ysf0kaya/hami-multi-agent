"""
Audio Agent — Kategori: KARMA
------------------------------
HuggingFace'den audio modelleri yükler ve
batch speech/audio işleme görevi çalıştırır.

Desteklenen modeller:
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

MODELS = {
    "whisper-tiny":  "openai/whisper-tiny",
    "whisper-base":  "openai/whisper-base",
    "wav2vec2-base": "facebook/wav2vec2-base",
}

MODEL_KEY   = os.getenv("MODEL_NAME", "whisper-tiny")
BATCH_SIZE  = int(os.getenv("BATCH_SIZE", "8"))
REPEAT      = int(os.getenv("REPEAT", "5"))
AGENT_ID    = os.getenv("AGENT_ID", f"audio-{MODEL_KEY}")
CATEGORY    = "KARMA"
SAMPLE_RATE = 16000
DURATION_S  = 3  # 3 saniye — daha kısa, daha az bellek


def generate_fake_audio(count: int) -> list:
    """Her biri ayrı numpy array olarak ses verisi üret."""
    return [
        np.random.randn(SAMPLE_RATE * DURATION_S).astype(np.float32)
        for _ in range(count)
    ]


def run():
    model_name = MODELS.get(MODEL_KEY)
    if not model_name:
        raise ValueError(f"Bilinmeyen model: {MODEL_KEY}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Model yükleniyor: {model_name} → {device}")

    is_whisper = "whisper" in MODEL_KEY

    if is_whisper:
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    else:
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2Model.from_pretrained(model_name).to(device)

    model.eval()
    logger.info(f"Model yüklendi ✅ — {model_name}")

    fake_audio_list = generate_fake_audio(BATCH_SIZE)

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
            # Whisper: her ses ayrı işlenir
            for audio in fake_audio_list:
                inputs = processor(
                    audio,
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="pt",
                ).to(device)
                with torch.no_grad():
                    _ = model.generate(inputs.input_features)
        else:
            # Wav2Vec2: batch işleme
            inputs = processor(
                fake_audio_list,
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
        time.sleep(60)


if __name__ == "__main__":
    run()
