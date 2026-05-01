"""
Collector
---------
HAMi ve psutil'den agent metrikleri toplar.
Her agent çalışması sonunda ortalamaları CSV'ye yazar.

Kullanım:
    collector = MetricCollector(agent_id="llm-qwen", category="GPU_YOGUN", model_name="Qwen/Qwen1.5-0.5B")
    collector.start()
    # ... agent batch görevi çalıştırır ...
    collector.stop()
    collector.save()
"""

import os
import csv
import time
import logging
import psutil
import torch
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("Collector")

# Dataset dosya yolu
DATASET_PATH = Path(os.getenv("DATASET_PATH", "/data/dataset.csv"))

# CSV sütun başlıkları
CSV_COLUMNS = [
    "timestamp",
    "agent_id",
    "model_name",
    "category",
    # GPU metrikleri (HAMi/torch üzerinden)
    "avg_gpu_mem_mb",
    "max_gpu_mem_mb",
    "avg_gpu_compute_percent",
    # CPU/RAM metrikleri
    "avg_cpu_percent",
    "max_cpu_percent",
    "avg_ram_gb",
    # Görev metrikleri
    "batch_count",
    "avg_batch_duration_s",
    "throughput_per_s",
    # Sistem bilgisi
    "gpu_name",
    "total_gpu_mem_mb",
]


class MetricCollector:
    def __init__(self, agent_id: str, category: str, model_name: str):
        self.agent_id = agent_id
        self.category = category
        self.model_name = model_name

        # Toplanan ham metrikler
        self._gpu_mem_samples = []
        self._gpu_compute_samples = []
        self._cpu_samples = []
        self._ram_samples = []
        self._batch_durations = []

        self._running = False
        self._start_time = None

        # GPU bilgisi
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            props = torch.cuda.get_device_properties(0)
            self.gpu_name = props.name
            self.total_gpu_mem_mb = round(props.total_memory / 1e6, 2)
        else:
            self.gpu_name = "CPU"
            self.total_gpu_mem_mb = 0

        logger.info(f"Collector hazır — agent={agent_id} category={category} device={self.device}")

    # ── Kontrol ───────────────────────────────────────────────────────────────

    def start(self):
        """Metrik toplamayı başlat."""
        self._running = True
        self._start_time = time.time()
        logger.info(f"Metrik toplama başladı — {self.agent_id}")

    def stop(self):
        """Metrik toplamayı durdur."""
        self._running = False
        logger.info(f"Metrik toplama durduruldu — {self.agent_id}")

    # ── Batch Ölçümü ──────────────────────────────────────────────────────────

    def record_batch(self, duration_s: float):
        """
        Bir batch çalışması sonrasında çağrılır.
        Anlık metrikleri kaydeder ve batch süresini saklar.

        Args:
            duration_s: Batch'in kaç saniye sürdüğü
        """
        if not self._running:
            logger.warning("Collector çalışmıyor, önce start() çağrıyın")
            return

        # GPU metrikleri
        if self.device == "cuda":
            gpu_mem = torch.cuda.memory_allocated(0) / 1e6
            # HAMi VRAM kotası içindeki kullanım oranı
            gpu_compute = round(
                torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory * 100, 2
            )
        else:
            gpu_mem = 0.0
            gpu_compute = 0.0

        # CPU/RAM metrikleri
        cpu = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory().used / 1e9

        # Kaydet
        self._gpu_mem_samples.append(gpu_mem)
        self._gpu_compute_samples.append(gpu_compute)
        self._cpu_samples.append(cpu)
        self._ram_samples.append(ram)
        self._batch_durations.append(duration_s)

        logger.debug(
            f"Batch kaydedildi — gpu_mem={gpu_mem:.1f}MB "
            f"cpu={cpu:.1f}% ram={ram:.2f}GB duration={duration_s:.2f}s"
        )

    # ── Ortalama Hesaplama ────────────────────────────────────────────────────

    def _avg(self, lst):
        return round(sum(lst) / len(lst), 3) if lst else 0.0

    def _max(self, lst):
        return round(max(lst), 3) if lst else 0.0

    def summarize(self) -> dict:
        """Toplanan metriklerden özet istatistik üret."""
        batch_count = len(self._batch_durations)
        avg_duration = self._avg(self._batch_durations)
        throughput = round(batch_count / sum(self._batch_durations), 3) if self._batch_durations else 0.0

        return {
            "timestamp":              datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "agent_id":               self.agent_id,
            "model_name":             self.model_name,
            "category":               self.category,
            "avg_gpu_mem_mb":         self._avg(self._gpu_mem_samples),
            "max_gpu_mem_mb":         self._max(self._gpu_mem_samples),
            "avg_gpu_compute_percent":self._avg(self._gpu_compute_samples),
            "avg_cpu_percent":        self._avg(self._cpu_samples),
            "max_cpu_percent":        self._max(self._cpu_samples),
            "avg_ram_gb":             self._avg(self._ram_samples),
            "batch_count":            batch_count,
            "avg_batch_duration_s":   avg_duration,
            "throughput_per_s":       throughput,
            "gpu_name":               self.gpu_name,
            "total_gpu_mem_mb":       self.total_gpu_mem_mb,
        }

    # ── CSV Kaydetme ──────────────────────────────────────────────────────────

    def save(self):
        """Özet metrikleri dataset.csv'ye ekle."""
        if not self._batch_durations:
            logger.warning("Kaydedilecek veri yok — hiç batch çalıştırılmadı")
            return

        summary = self.summarize()

        # Klasör yoksa oluştur
        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Dosya yoksa başlık satırı ekle
        file_exists = DATASET_PATH.exists()

        with open(DATASET_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(summary)

        logger.info(
            f"Dataset güncellendi → {DATASET_PATH} "
            f"| batch_count={summary['batch_count']} "
            f"| avg_gpu_mem={summary['avg_gpu_mem_mb']}MB "
            f"| avg_cpu={summary['avg_cpu_percent']}%"
        )

        return summary


# ── Kullanım Örneği ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Test: sahte batch çalışması simüle et
    collector = MetricCollector(
        agent_id="test-agent",
        category="GPU_YOGUN",
        model_name="test-model"
    )

    collector.start()

    for i in range(5):
        logger.info(f"Batch {i+1}/5 simüle ediliyor...")
        time.sleep(1)  # Gerçekte model çalışır
        collector.record_batch(duration_s=round(random.uniform(1.0, 3.0), 2))

    collector.stop()
    summary = collector.save()

    logger.info("Test tamamlandı:")
    for k, v in summary.items():
        logger.info(f"  {k}: {v}")
