"""
Cognitive Agent — Ajan 1
------------------------
TinyLlama veya Qwen-0.5B kullanarak metin anlama ve karar verme yapar.
HAMi üzerinden 1536MB VRAM kotası atanır.

Görevler:
  - text_inference : Metin üretimi
  - sentiment      : Duygu analizi
  - decision       : Karar verme
"""

import os
import time
import logging
import torch
from agents.base_agent import BaseAgent, Task

logger = logging.getLogger("CognitiveAgent")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen1.5-0.5B")


class CognitiveAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_id=os.getenv("AGENT_ID", "cognitive-1"),
            agent_type="COGNITIVE"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            logger.info(f"Model yükleniyor: {MODEL_NAME} → {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
            )
            self.model.eval()
            logger.info(f"Model yüklendi ✅ — device={self.device}")
            if self.device == "cuda":
                used = torch.cuda.memory_allocated() / 1e9
                logger.info(f"GPU bellek kullanımı: {used:.2f} GB")
        except Exception as e:
            logger.warning(f"Model yüklenemedi, mock mod aktif: {e}")
            self.model = None

    def process(self, task: Task) -> Task:
        logger.info(f"Task işleniyor: {task.task_id} — tip={task.task_type}")
        task.agent_id = self.agent_id
        task.status = "running"
        try:
            if task.task_type == "text_inference":
                task.result = self._inference(task.payload)
            elif task.task_type == "sentiment":
                task.result = self._sentiment(task.payload)
            elif task.task_type == "decision":
                task.result = self._decision(task.payload)
            else:
                task.result = {"error": f"Bilinmeyen görev tipi: {task.task_type}"}
                task.status = "failed"
                return task
            task.status = "done"
        except Exception as e:
            task.status = "failed"
            task.result = {"error": str(e)}
            logger.error(f"Task başarısız: {e}")
        return task

    def _inference(self, payload: dict) -> dict:
        prompt = payload.get("prompt", "Hello")
        max_new_tokens = payload.get("max_new_tokens", 50)
        if self.model is None:
            return {
                "prompt": prompt,
                "output": f"[MOCK] {prompt} → Bu bir mock yanıttır.",
                "device": self.device,
                "model": MODEL_NAME,
            }
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = time.time() - start
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {
            "prompt": prompt,
            "output": text,
            "device": self.device,
            "model": MODEL_NAME,
            "elapsed_seconds": round(elapsed, 3),
        }

    def _sentiment(self, payload: dict) -> dict:
        text = payload.get("text", "")
        positive_words = ["iyi", "güzel", "harika", "mükemmel", "good", "great"]
        negative_words = ["kötü", "berbat", "korkunç", "bad", "terrible"]
        text_lower = text.lower()
        pos = sum(1 for w in positive_words if w in text_lower)
        neg = sum(1 for w in negative_words if w in text_lower)
        label = "positive" if pos > neg else "negative" if neg > pos else "neutral"
        return {"text": text, "sentiment": label, "device": self.device}

    def _decision(self, payload: dict) -> dict:
        options = payload.get("options", [])
        context = payload.get("context", "")
        if not options:
            return {"error": "Seçenek listesi boş"}
        chosen = options[len(context) % len(options)]
        return {"context": context, "options": options, "decision": chosen, "device": self.device}


if __name__ == "__main__":
    agent = CognitiveAgent()
    logger.info("Test görevi çalıştırılıyor...")
    test_task = Task(task_type="sentiment", payload={"text": "Bu proje harika gidiyor!"})
    result = agent.process(test_task)
    logger.info(f"Test sonucu: {result.result}")
    logger.info("Agent bekleme moduna geçti — Redis bağlantısı bekleniyor...")
    while True:
        time.sleep(3600)
