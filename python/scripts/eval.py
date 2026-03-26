"""
eval.py — Evader 평가 스크립트

고정 시드로 N 에피소드를 실행하여 다음 지표를 산출한다:
  - survival_rate    : timeout까지 생존한 에피소드 비율
  - goal_reach_rate  : 목표 지점 도달 비율
  - capture_rate     : 포획된 에피소드 비율
  - crash_rate       : 충돌 종료 비율
  - mean_time_to_capture : 포획 에피소드의 평균 경과 시간
  - los_break_rate   : LOS 차단 성공 비율 (Stage2+, 데이터 있을 경우)

사용법:
    python python/scripts/eval.py --run-id evader_s0_base_seed42 --n-episodes 50 --seed 0
"""

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "python" / "results"

# 에피소드 종료 유형
TERM_GOAL = "goal"
TERM_CAPTURED = "captured"
TERM_CRASH = "crash"
TERM_TIMEOUT = "timeout"


def load_episode_logs(run_id: str) -> list[dict]:
    """
    results/{run_id}/episodes.csv 또는 episodes.ndjson 파일을 읽어 에피소드 목록을 반환한다.

    파일 형식 (CSV 예시):
        episode_id, termination, duration_steps, los_breaks
        0, timeout, 300, 12
        1, captured, 87, 3
        ...

    실제 Unity/ML-Agents 환경과 연동 시 이 함수를 수정하거나 교체한다.
    """
    run_dir = RESULTS_DIR / run_id
    csv_path = run_dir / "episodes.csv"
    ndjson_path = run_dir / "episodes.ndjson"

    episodes = []

    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                episodes.append({
                    "termination": row.get("termination", "timeout").strip(),
                    "duration_steps": int(row.get("duration_steps", 0)),
                    "los_breaks": int(row.get("los_breaks", 0)),
                })
    elif ndjson_path.exists():
        with open(ndjson_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    episodes.append(json.loads(line))
    else:
        print(
            f"[eval] episode log not found at:\n"
            f"  {csv_path}\n  {ndjson_path}\n"
            "[eval] Run the eval inside Unity first and export the log."
        )
        sys.exit(1)

    return episodes


def compute_metrics(episodes: list[dict]) -> dict:
    n = len(episodes)
    if n == 0:
        return {}

    goal_count = sum(1 for e in episodes if e["termination"] == TERM_GOAL)
    captured_count = sum(1 for e in episodes if e["termination"] == TERM_CAPTURED)
    crash_count = sum(1 for e in episodes if e["termination"] == TERM_CRASH)
    timeout_count = sum(1 for e in episodes if e["termination"] == TERM_TIMEOUT)

    captured_durations = [e["duration_steps"] for e in episodes if e["termination"] == TERM_CAPTURED]
    mean_ttc = sum(captured_durations) / len(captured_durations) if captured_durations else None

    total_los = sum(e.get("los_breaks", 0) for e in episodes)

    return {
        "n_episodes": n,
        "survival_rate": round((timeout_count + goal_count) / n * 100, 1),
        "goal_reach_rate": round(goal_count / n * 100, 1),
        "capture_rate": round(captured_count / n * 100, 1),
        "crash_rate": round(crash_count / n * 100, 1),
        "mean_time_to_capture_steps": round(mean_ttc, 1) if mean_ttc else "N/A",
        "total_los_breaks": total_los,
    }


def print_metrics(run_id: str, metrics: dict):
    targets = {
        "survival_rate":   ("M1 목표: ≥70%",  70),
        "goal_reach_rate": ("M1 목표: ≥50%",  50),
        "capture_rate":    ("M1 목표: ≤30%",  None),
        "crash_rate":      ("목표: ≤15%",      None),
    }

    print(f"\n{'='*55}")
    print(f"  Eval Report: {run_id}")
    print(f"  Episodes   : {metrics.get('n_episodes', 0)}")
    print(f"{'='*55}")
    print(f"  {'지표':<30} {'값':>8}  {'목표'}")
    print(f"  {'-'*50}")

    for key, val in metrics.items():
        if key in ("n_episodes", "total_los_breaks"):
            continue
        target_str, _ = targets.get(key, ("", None))
        pct = f"{val}%" if isinstance(val, float) else str(val)
        print(f"  {key:<30} {pct:>8}  {target_str}")

    if metrics.get("total_los_breaks", 0) > 0:
        print(f"  {'total_los_breaks':<30} {metrics['total_los_breaks']:>8}")

    print(f"{'='*55}\n")


def main():
    parser = argparse.ArgumentParser(description="Evader eval metrics")
    parser.add_argument("--run-id", required=True, help="run-id to evaluate")
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0, help="Fixed seed (for documentation)")
    args = parser.parse_args()

    episodes = load_episode_logs(args.run_id)

    # 지정된 에피소드 수로 슬라이스
    if len(episodes) > args.n_episodes:
        episodes = episodes[: args.n_episodes]

    metrics = compute_metrics(episodes)
    print_metrics(args.run_id, metrics)

    # results/{run_id}/metrics.json 에 저장
    out_dir = RESULTS_DIR / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = out_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump({"run_id": args.run_id, "seed": args.seed, **metrics}, f, indent=2)
    print(f"[eval] metrics saved → {metrics_file.relative_to(REPO_ROOT)}")
    print("[eval] /log 커맨드로 EXPERIMENTS.md에 기록하세요.")


if __name__ == "__main__":
    main()
