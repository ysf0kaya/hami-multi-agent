"""
Base Agent
----------
Tüm agentların miras aldığı temel sınıf.
Task modeli ve ortak loglama burada tanımlanır.
"""

import uuid
import logging
from pydantic import BaseModel, Field
from typing import Optional, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


class Task(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str
    payload: dict
    result: Optional[Any] = None
    status: str = "pending"
    agent_id: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


class BaseAgent:
    def __init__(self, agent_id: str, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.logger = logging.getLogger(f"{agent_type}:{agent_id}")
        self.logger.info(f"Agent başlatıldı — id={agent_id} type={agent_type}")

    def process(self, task: Task) -> Task:
        """Her agent bu metodu override eder."""
        raise NotImplementedError
