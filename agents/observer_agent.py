"""
Observer Agent — Ajan 3
------------------------
Sistem metriklerini (CPU, GPU, RAM) izler ve raporlar.
HAMi üzerinden 512MB VRAM kotası atanır.

Görevler:
  - system_metrics : CPU/RAM/GPU anlık durum
  - gpu_metrics    : Detaylı GPU bilgisi
"""

import os
import time
import logging
import psutil
import torch
from agents.base_agent import BaseAgent, Task

logger = logging.getLogger("ObserverAgent")


class ObserverAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_id=os.getenv("AGENT_ID", "observer-1"),
            agent_type="OBSERVER"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Observer Agent hazır — device={self.device}")

    def process(self, task: Task) -> Task:
        logger.info(f"Task işleniyor: {task.task_id} — tip={task.task_type}")
        task.agent_id = self.agent_id
        task.status = "running"

        try:
            if task.task_type == "system_metrics":
                task.result = self._system_metrics()
            elif task.task_type == "gpu_metrics":
                task.result = self._gpu_metrics()
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

    def _system_metrics(self) -> dict:
        cpu_percent = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cpu": {
                "percent": cpu_percent,
                "core_count": psutil.cpu_count(),
            },
            "memory": {
                "total_gb": round(mem.total / 1e9, 2),
                "used_gb": round(mem.used / 1e9, 2),
                "percent": mem.percent,
            },
            "disk": {
                "total_gb": round(disk.total / 1e9, 2),
                "used_gb": round(disk.used / 1e9, 2),
                "percent": disk.percent,
            },
            "gpu": self._gpu_metrics(),
        }

    def _gpu_metrics(self) -> dict:
        if not torch.cuda.is_available():
            return {"available": False}

        props = torch.cuda.get_device_properties(0)
        allocated = torch.cuda.memory_allocated(0)
        reserved  = torch.cuda.memory_reserved(0)
        total     = props.total_memory

        return {
            "available": True,
            "name": props.name,
            "total_mb": round(total / 1e6, 2),
            "allocated_mb": round(allocated / 1e6, 2),
            "reserved_mb": round(reserved / 1e6, 2),
            "free_mb": round((total - allocated) / 1e6, 2),
            "utilization_percent": round(allocated / total * 100, 2),
        }


if __name__ == "__main__":
    agent = ObserverAgent()

    # Basit test
    test_task = Task(
        task_type="system_metrics",
        payload={}
    )
    result = agent.process(test_task)
    logger.info(f"Test sonucu: {result.result}")
