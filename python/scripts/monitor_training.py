"""
monitor_training.py - Lightweight ML-Agents training monitor.

Reads run log files produced by mlagents-learn and prints a concise
status report with actionable recommendations.

Usage examples:
    python python/scripts/monitor_training.py --run-id evader_s1_obstacle_44d_v5_seed42
    python python/scripts/monitor_training.py --run-id evader_s1_obstacle_44d_v5_seed42 --watch --interval 30
    python python/scripts/monitor_training.py --latest --watch
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "python" / "results"


@dataclass
class MonitorSnapshot:
    run_id: str
    step: float | None
    cumulative_reward: float | None
    episode_length: float | None
    entropy: float | None
    beta: float | None
    learning_rate: float | None
    checkpoints: list[dict[str, Any]]
    has_end_time: bool


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _gauge(gauges: dict[str, Any], key: str) -> float | None:
    node = gauges.get(key)
    if not isinstance(node, dict):
        return None
    value = node.get("value")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _choose_latest_run(results_dir: Path) -> str:
    newest_id = None
    newest_time = -1.0

    for d in results_dir.iterdir():
        if not d.is_dir():
            continue
        status_path = d / "run_logs" / "training_status.json"
        timers_path = d / "run_logs" / "timers.json"
        if not status_path.exists() or not timers_path.exists():
            continue
        mtime = max(status_path.stat().st_mtime, timers_path.stat().st_mtime)
        if mtime > newest_time:
            newest_time = mtime
            newest_id = d.name

    if newest_id is None:
        raise FileNotFoundError("No run with run_logs/training_status.json found in python/results")
    return newest_id


def load_snapshot(run_id: str, results_dir: Path = RESULTS_DIR) -> MonitorSnapshot:
    run_dir = results_dir / run_id
    status_path = run_dir / "run_logs" / "training_status.json"
    timers_path = run_dir / "run_logs" / "timers.json"

    if not status_path.exists():
        raise FileNotFoundError(f"Missing file: {status_path}")
    if not timers_path.exists():
        raise FileNotFoundError(f"Missing file: {timers_path}")

    status = _read_json(status_path)
    timers = _read_json(timers_path)

    # Most runs use behavior key "Drone_Evader".
    behavior_key = None
    for key in status.keys():
        if key != "metadata":
            behavior_key = key
            break
    if behavior_key is None:
        raise ValueError("Could not find behavior key in training_status.json")

    checkpoints = status.get(behavior_key, {}).get("checkpoints", [])
    gauges = timers.get("gauges", {})

    snap = MonitorSnapshot(
        run_id=run_id,
        step=_gauge(gauges, f"{behavior_key}.Step.mean"),
        cumulative_reward=_gauge(gauges, f"{behavior_key}.Environment.CumulativeReward.mean"),
        episode_length=_gauge(gauges, f"{behavior_key}.Environment.EpisodeLength.mean"),
        entropy=_gauge(gauges, f"{behavior_key}.Policy.Entropy.mean"),
        beta=_gauge(gauges, f"{behavior_key}.Policy.Beta.mean"),
        learning_rate=_gauge(gauges, f"{behavior_key}.Policy.LearningRate.mean"),
        checkpoints=checkpoints,
        has_end_time="end_time_seconds" in timers.get("metadata", {}),
    )
    return snap


def checkpoint_trend(checkpoints: list[dict[str, Any]], last_n: int = 5) -> float | None:
    if not checkpoints:
        return None
    recent = checkpoints[-last_n:]
    rewards = [cp.get("reward") for cp in recent if isinstance(cp.get("reward"), (int, float))]
    if len(rewards) < 2:
        return None
    return float(rewards[-1] - rewards[0])


def classify(snap: MonitorSnapshot) -> tuple[str, list[str], str]:
    flags: list[str] = []
    trend = checkpoint_trend(snap.checkpoints)

    if snap.cumulative_reward is not None and snap.cumulative_reward < -3.0:
        flags.append("low_reward")
    if snap.episode_length is not None and snap.episode_length > 380:
        flags.append("long_episode_timeout_risk")
    if trend is not None and trend < -1.5:
        flags.append("checkpoint_reward_decline")
    if snap.entropy is not None and snap.entropy < 0.6:
        flags.append("low_entropy")

    # Decision policy tuned for Stage1 obstacle runs.
    if {"low_reward", "long_episode_timeout_risk", "checkpoint_reward_decline"}.issubset(flags):
        decision = "stop_and_revise"
        reason = "Reward is negative, episodes are long, and checkpoint reward trend is declining."
    elif trend is not None and trend > 1.0 and (snap.cumulative_reward is None or snap.cumulative_reward > -1.0):
        decision = "continue"
        reason = "Checkpoint rewards are improving and cumulative reward is not deeply negative."
    else:
        decision = "watch"
        reason = "Signal is mixed; keep monitoring before changing hyperparameters."

    return decision, flags, reason


def print_report(snap: MonitorSnapshot) -> None:
    trend = checkpoint_trend(snap.checkpoints)
    decision, flags, reason = classify(snap)

    print("=" * 72)
    print(f"Run ID           : {snap.run_id}")
    print(f"Training Active? : {'no (ended)' if snap.has_end_time else 'yes (running or open)'}")
    print(f"Step             : {snap.step if snap.step is not None else 'N/A'}")
    print(f"CumReward(mean)  : {snap.cumulative_reward if snap.cumulative_reward is not None else 'N/A'}")
    print(f"EpisodeLen(mean) : {snap.episode_length if snap.episode_length is not None else 'N/A'}")
    print(f"Entropy(mean)    : {snap.entropy if snap.entropy is not None else 'N/A'}")
    print(f"Beta(mean)       : {snap.beta if snap.beta is not None else 'N/A'}")
    print(f"LR(mean)         : {snap.learning_rate if snap.learning_rate is not None else 'N/A'}")
    print(f"Checkpoint trend : {trend if trend is not None else 'N/A'}")
    print(f"Decision         : {decision}")
    print(f"Reason           : {reason}")
    print(f"Flags            : {', '.join(flags) if flags else 'none'}")
    print("=" * 72)


def run_once(args: argparse.Namespace) -> int:
    run_id = args.run_id
    if args.latest:
        run_id = _choose_latest_run(Path(args.results_dir))

    if not run_id:
        raise ValueError("Provide --run-id or use --latest")

    snap = load_snapshot(run_id=run_id, results_dir=Path(args.results_dir))
    print_report(snap)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor ML-Agents run logs and diagnose training state.")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID under python/results")
    parser.add_argument("--latest", action="store_true", help="Auto-select latest run with run_logs")
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR), help="Results directory path")
    parser.add_argument("--watch", action="store_true", help="Watch mode: poll and print periodically")
    parser.add_argument("--interval", type=int, default=30, help="Watch interval seconds")
    args = parser.parse_args()

    if not args.watch:
        return run_once(args)

    while True:
        try:
            run_once(args)
        except Exception as exc:
            print(f"[monitor] {type(exc).__name__}: {exc}")
        time.sleep(max(5, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
