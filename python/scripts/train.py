"""
train.py — Evader 학습 실행 래퍼

mlagents-learn 커맨드를 래핑하여 run-id 생성, 로그 정리, 결과 저장을 자동화한다.

사용법:
    python python/scripts/train.py --stage 0 --exp base --seed 42
    python python/scripts/train.py --stage 1 --exp obstacle --seed 42 --init-from evader_s0_base_seed42
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "python" / "config"
RESULTS_DIR = REPO_ROOT / "python" / "results"


def build_run_id(stage: int, exp: str, seed: int) -> str:
    return f"evader_s{stage}_{exp}_seed{seed}"


def find_config(stage: int, exp: str) -> Path:
    """stage와 exp 이름으로 config 파일을 탐색한다."""
    # 날짜가 포함된 최신 파일 우선 탐색
    pattern = f"evader_s{stage}_{exp}*.yaml"
    candidates = sorted(CONFIG_DIR.glob(pattern), reverse=True)
    if candidates:
        return candidates[0]
    # 템플릿 폴백
    template = CONFIG_DIR / f"evader_s{stage}_template.yaml"
    if template.exists():
        print(f"[warn] config not found for exp='{exp}', using template: {template.name}")
        return template
    raise FileNotFoundError(
        f"Config not found: {CONFIG_DIR / pattern}\n"
        f"Create it or check the config directory."
    )


def main():
    parser = argparse.ArgumentParser(description="Evader RL training wrapper")
    parser.add_argument("--stage", type=int, required=True, help="Stage number (0-3)")
    parser.add_argument("--exp", type=str, default="base", help="Experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--init-from", type=str, default=None,
        help="run-id to warm-start from (stage transition)"
    )
    parser.add_argument("--force", action="store_true", default=True, help="Force overwrite")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel envs")
    args = parser.parse_args()

    run_id = build_run_id(args.stage, args.exp, args.seed)
    config_path = find_config(args.stage, args.exp)

    print(f"[train] run-id   : {run_id}")
    print(f"[train] config   : {config_path.relative_to(REPO_ROOT)}")
    print(f"[train] seed     : {args.seed}")
    if args.init_from:
        print(f"[train] init-from: {args.init_from}")

    cmd = [
        "mlagents-learn",
        str(config_path),
        f"--run-id={run_id}",
    ]
    if args.force:
        cmd.append("--force")
    if args.init_from:
        cmd.append(f"--initialize-from={args.init_from}")
    if args.num_envs > 1:
        cmd.append(f"--num-envs={args.num_envs}")

    print(f"\n[train] executing: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
