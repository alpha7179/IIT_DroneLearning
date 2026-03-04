"""
run_pathfinding.py — 경로 탐색 알고리즘 벤치마크 CLI 진입점

사용 예:
    python python/scripts/run_pathfinding.py \
        --grid-size 10 \
        --n-trials 100 \
        --modes urban_block random pursuer_centered \
        --seed 42 \
        --out python/results/pathfinding/

출력 파일:
    benchmark_results.json  — 원시 실험 데이터
    stats_summary.json      — 알고리즘별 통계 요약
    grid_comparison.png     — 단일 맵 8알고리즘 시각화
    metrics_bar.png         — 지표 비교 막대 차트
    report.md               — 추천 알고리즘 근거 요약
    (Docs/EXPERIMENTS.md에 자동 추가)
"""

from __future__ import annotations

import argparse
import json
import sys

# Windows 터미널 UTF-8 출력 강제
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

# 레포 루트를 Python 경로에 추가 (어디서 실행하든 동작)
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from python.analysis.algorithms import ALGORITHM_NAMES, run_algorithm
from python.analysis.benchmark import (
    build_report,
    compute_stats,
    records_to_dict,
    run_benchmark,
    stats_to_dict,
)
from python.analysis.grid_env import make_scenario
from python.analysis.visualize import plot_grid_comparison, plot_metrics_bar


# ── CLI 인자 파싱 ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Occlusion-aware 경로 탐색 알고리즘 비교 벤치마크",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--grid-size",  type=int, default=10,
                   help="GridWorld 크기 (N×N)")
    p.add_argument("--n-trials",   type=int, default=100,
                   help="모드별 반복 시나리오 수")
    p.add_argument("--modes",      nargs="+",
                   default=["urban_block", "random", "pursuer_centered"],
                   choices=["urban_block", "random", "pursuer_centered"],
                   help="장애물 배치 모드")
    p.add_argument("--seed",       type=int, default=42,
                   help="시드 시작값")
    p.add_argument("--out",        type=str,
                   default="python/results/pathfinding",
                   help="출력 디렉토리 경로")
    p.add_argument("--no-viz",     action="store_true",
                   help="시각화 생략 (서버 환경)")
    p.add_argument("--viz-mode",   type=str,
                   default="urban_block",
                   choices=["urban_block", "random", "pursuer_centered"],
                   help="grid_comparison.png에 사용할 시나리오 모드")
    p.add_argument("--algorithms", nargs="+", default=None,
                   choices=ALGORITHM_NAMES,
                   help="실행할 알고리즘 목록 (기본: 전체)")
    p.add_argument("--experiments-md", type=str,
                   default="Docs/EXPERIMENTS.md",
                   help="실험 로그 파일 경로 (자동 추가)")
    return p.parse_args()


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    algos = args.algorithms or ALGORITHM_NAMES

    print("=" * 60)
    print("  Pathfinding Benchmark - IIT_DroneLearning")
    print("=" * 60)
    print(f"  Grid size   : {args.grid_size}×{args.grid_size}")
    print(f"  Trials/mode : {args.n_trials}")
    print(f"  Modes       : {args.modes}")
    print(f"  Algorithms  : {algos}")
    print(f"  Seed start  : {args.seed}")
    print(f"  Output      : {out_dir.resolve()}")
    print()

    # ── 1. 벤치마크 실행 ──────────────────────────────────────────────────────
    print("[1/5] 벤치마크 실행 중...")
    records = run_benchmark(
        grid_size=args.grid_size,
        n_trials=args.n_trials,
        seed_start=args.seed,
        modes=args.modes,
        algorithms=algos,
    )
    print(f"      완료: {len(records)} 레코드 수집")

    # ── 2. JSON 저장 ──────────────────────────────────────────────────────────
    print("[2/5] 결과 저장 중...")
    results_path = out_dir / "benchmark_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(records_to_dict(records), f, ensure_ascii=False, indent=2)
    print(f"      {results_path}")

    stats = compute_stats(records)
    stats_path = out_dir / "stats_summary.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_to_dict(stats), f, ensure_ascii=False, indent=2)
    print(f"      {stats_path}")

    # ── 3. 시각화 ─────────────────────────────────────────────────────────────
    if not args.no_viz:
        print("[3/5] 시각화 생성 중...")

        # grid_comparison.png: 단일 시나리오로 8알고리즘 경로 비교
        try:
            env_viz, start_viz, goal_viz, pursuer_viz = make_scenario(
                mode=args.viz_mode,
                size=args.grid_size,
                seed=args.seed,
                min_dist=max(3, args.grid_size // 2),
            )
            viz_results = {
                algo: run_algorithm(
                    name=algo,
                    env=env_viz,
                    start=start_viz,
                    goal=goal_viz,
                    pursuer_pos=pursuer_viz,
                    rrt_seed=args.seed,
                )
                for algo in algos
            }
            plot_grid_comparison(
                env=env_viz,
                start=start_viz,
                goal=goal_viz,
                pursuer_pos=pursuer_viz,
                results=viz_results,
                out_path=out_dir / "grid_comparison.png",
            )
        except Exception as e:
            print(f"      [경고] grid_comparison.png 생성 실패: {e}")

        # metrics_bar.png: 통계 막대 차트
        try:
            plot_metrics_bar(stats, out_path=out_dir / "metrics_bar.png")
        except Exception as e:
            print(f"      [경고] metrics_bar.png 생성 실패: {e}")
    else:
        print("[3/5] 시각화 생략 (--no-viz)")

    # ── 4. report.md 생성 ─────────────────────────────────────────────────────
    print("[4/5] 보고서 생성 중...")
    report_md = build_report(stats, grid_size=args.grid_size, n_trials=args.n_trials)

    # report.md 저장
    report_path = out_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Pathfinding Benchmark Report\n")
        f.write(f"\n- Grid: {args.grid_size}×{args.grid_size}, Trials: {args.n_trials}/mode\n")
        f.write(f"- Modes: {', '.join(args.modes)}\n")
        f.write(f"- Algorithms: {', '.join(algos)}\n")
        f.write(report_md)
    print(f"      {report_path}")

    # ── 5. Docs/EXPERIMENTS.md 자동 추가 ──────────────────────────────────────
    print("[5/5] EXPERIMENTS.md 업데이트 중...")
    exp_md_path = _REPO_ROOT / args.experiments_md
    if exp_md_path.exists():
        with open(exp_md_path, "a", encoding="utf-8") as f:
            f.write(report_md)
        print(f"      {exp_md_path} (추가 완료)")
    else:
        print(f"      [경고] {exp_md_path} 파일이 없습니다. 건너뜁니다.")

    # ── 요약 출력 ─────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  결과 요약 (Composite Score 순위)")
    print("=" * 60)
    for rank, s in enumerate(stats, 1):
        bar = "#" * int(s.composite_score * 20)
        print(
            f"  {rank:2d}. {s.label:<22s} "
            f"score={s.composite_score:.3f} [{bar:<20s}] "
            f"success={s.success_rate*100:.1f}% "
            f"LOS={s.los_exposure_mean:.1f}% "
            f"t={s.exec_time_mean:.1f}ms"
        )
    print()
    if stats:
        best = stats[0]
        print(f"  [추천] {best.label}")
        print(f"    복합 점수 {best.composite_score:.3f} | "
              f"LOS 노출 {best.los_exposure_mean:.1f}% | "
              f"성공률 {best.success_rate*100:.1f}%")
    print()


if __name__ == "__main__":
    main()
