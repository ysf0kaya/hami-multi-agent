"""
Processing Agent — Ajan 2
--------------------------
GPU üzerinde yoğun matris ve sinyal işleme yapar.
HAMi üzerinden 1024MB VRAM kotası atanır.

Görevler:
  - matrix_multiply : GPU matris çarpımı
  - signal_classify : Sinyal sınıflandırma (CNN)
  - batch_compute   : Toplu vektör işleme
"""

import os
import time
import logging
import torch
import torch.nn as nn
from agents.base_agent import BaseAgent, Task

logger = logging.getLogger("ProcessingAgent")


class SimpleSignalCNN(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class ProcessingAgent(BaseAgent):
    SIGNAL_CLASSES = ["normal", "anomaly", "noise", "signal_loss"]

    def __init__(self):
        super().__init__(
            agent_id=os.getenv("AGENT_ID", "processing-1"),
            agent_type="PROCESSING"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cnn = SimpleSignalCNN().to(self.device)
        self.cnn.eval()
        logger.info(f"Processing Agent hazır — device={self.device}")

    def process(self, task: Task) -> Task:
        logger.info(f"Task işleniyor: {task.task_id} — tip={task.task_type}")
        task.agent_id = self.agent_id
        task.status = "running"
        try:
            if task.task_type == "matrix_multiply":
                task.result = self._matrix_multiply(task.payload)
            elif task.task_type == "signal_classify":
                task.result = self._signal_classify(task.payload)
            elif task.task_type == "batch_compute":
                task.result = self._batch_compute(task.payload)
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

    def _matrix_multiply(self, payload: dict) -> dict:
        size = payload.get("size", 500)
        iterations = payload.get("iterations", 1)
        a = torch.rand(size, size, device=self.device, dtype=torch.float32)
        b = torch.rand(size, size, device=self.device, dtype=torch.float32)
        if self.device == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            c = torch.mm(a, b)
        if self.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start
        vram_used = torch.cuda.memory_allocated() / 1e6 if self.device == "cuda" else 0
        return {
            "matrix_size": f"{size}x{size}",
            "iterations": iterations,
            "result_sum": float(c.sum().cpu()),
            "device": self.device,
            "elapsed_seconds": round(elapsed, 4),
            "vram_used_mb": round(vram_used, 2),
        }

    def _signal_classify(self, payload: dict) -> dict:
        signal_length = payload.get("signal_length", 256)
        batch_size = payload.get("batch_size", 8)
        signal = torch.rand(batch_size, 1, signal_length, device=self.device)
        start = time.time()
        with torch.no_grad():
            logits = self.cnn(signal)
            predictions = torch.argmax(logits, dim=1)
        elapsed = time.time() - start
        labels = [self.SIGNAL_CLASSES[p.item()] for p in predictions]
        return {
            "batch_size": batch_size,
            "signal_length": signal_length,
            "predictions": labels,
            "device": self.device,
            "elapsed_seconds": round(elapsed, 4),
        }

    def _batch_compute(self, payload: dict) -> dict:
        vector_size = payload.get("vector_size", 512)
        batch_count = payload.get("batch_count", 100)
        vectors = torch.rand(batch_count, vector_size, device=self.device)
        start = time.time()
        norms = torch.norm(vectors, dim=1, keepdim=True)
        normalized = vectors / (norms + 1e-8)
        similarity_matrix = torch.mm(normalized, normalized.T)
        elapsed = time.time() - start
        return {
            "vector_size": vector_size,
            "batch_count": batch_count,
            "mean_similarity": float(similarity_matrix.mean().cpu()),
            "device": self.device,
            "elapsed_seconds": round(elapsed, 4),
        }


if __name__ == "__main__":
    agent = ProcessingAgent()
    logger.info("Test görevi çalıştırılıyor...")
    test_task = Task(task_type="matrix_multiply", payload={"size": 500, "iterations": 3})
    result = agent.process(test_task)
    logger.info(f"Test sonucu: {result.result}")
    logger.info("Agent bekleme moduna geçti — Redis bağlantısı bekleniyor...")
    while True:
        time.sleep(3600)
