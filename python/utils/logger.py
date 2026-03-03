"""
logger.py — 실험 로거 유틸리티

ML-Agents 학습 중 에피소드 결과를 CSV/NDJSON으로 저장하는 유틸리티.
Unity C# 측에서 Python API로 결과를 전달하거나,
ML-Agents 커스텀 사이드 채널을 통해 데이터를 수신할 때 사용한다.
"""

import csv
import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "python" / "results"


class EpisodeLogger:
    """에피소드 결과를 CSV 파일로 기록한다."""

    FIELDNAMES = [
        "episode_id",
        "termination",      # goal | captured | crash | timeout
        "duration_steps",
        "total_reward",
        "los_breaks",       # LOS 차단 횟수 (Stage2+)
        "goal_distance_final",
        "timestamp",
    ]

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.run_dir = RESULTS_DIR / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / "episodes.csv"
        self._episode_count = 0

        # 파일이 없으면 헤더 작성
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()

    def log(
        self,
        termination: str,
        duration_steps: int,
        total_reward: float = 0.0,
        los_breaks: int = 0,
        goal_distance_final: float = -1.0,
    ):
        """에피소드 결과 한 줄을 기록한다."""
        row = {
            "episode_id": self._episode_count,
            "termination": termination,
            "duration_steps": duration_steps,
            "total_reward": round(total_reward, 4),
            "los_breaks": los_breaks,
            "goal_distance_final": round(goal_distance_final, 4),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow(row)
        self._episode_count += 1

    def summary(self) -> dict:
        """지금까지 기록된 에피소드의 요약 지표를 반환한다."""
        if not self.csv_path.exists():
            return {}
        rows = []
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        n = len(rows)
        if n == 0:
            return {"n_episodes": 0}

        goal = sum(1 for r in rows if r["termination"] == "goal")
        captured = sum(1 for r in rows if r["termination"] == "captured")
        crash = sum(1 for r in rows if r["termination"] == "crash")
        timeout = sum(1 for r in rows if r["termination"] == "timeout")
        return {
            "n_episodes": n,
            "survival_rate": round((timeout + goal) / n * 100, 1),
            "goal_reach_rate": round(goal / n * 100, 1),
            "capture_rate": round(captured / n * 100, 1),
            "crash_rate": round(crash / n * 100, 1),
        }


class TBWriter:
    """TensorBoard SummaryWriter 래퍼 (선택적 의존성)."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self._writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = str(RESULTS_DIR / run_id / "tb")
            self._writer = SummaryWriter(log_dir=log_dir)
        except ImportError:
            print("[TBWriter] tensorboard not available; skipping TB logging.")

    def add_scalar(self, tag: str, value: float, step: int):
        if self._writer:
            self._writer.add_scalar(tag, value, step)

    def flush(self):
        if self._writer:
            self._writer.flush()

    def close(self):
        if self._writer:
            self._writer.close()
