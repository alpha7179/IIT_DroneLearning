"""
benchmark.py — N맵 반복 실험 러너 + 통계 집계

실험 흐름:
  1. 시나리오 생성 (make_scenario)
  2. 각 알고리즘 실행
  3. PathResult + GridWorld.path_los_stats 수집
  4. 알고리즘별 mean/std/min/max 계산
  5. 복합 점수 (composite_score) 산출
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .algorithms import (
    ALGORITHM_LABELS,
    ALGORITHM_NAMES,
    PathResult,
    run_algorithm,
)
from .grid_env import GridWorld, make_scenario


# ── 시험 단위 레코드 ──────────────────────────────────────────────────────────

@dataclass
class TrialRecord:
    trial_id: int
    mode: str
    seed: int
    algorithm: str
    found: bool
    path_cost: float
    path_length: int           # 경유 셀 수
    los_exposure_pct: float    # 노출 셀 비율 (%)
    los_breaks: int            # LOS 상태 전환 횟수
    nodes_expanded: int
    exec_time_ms: float


# ── 벤치마크 실행 ─────────────────────────────────────────────────────────────

def run_benchmark(
    grid_size: int = 10,
    n_trials: int = 100,
    seed_start: int = 0,
    modes: list[str] | None = None,
    algorithms: list[str] | None = None,
) -> list[TrialRecord]:
    """
    N개 맵 × M 알고리즘 조합 실험을 실행하고 원시 기록 목록을 반환한다.

    Args:
        grid_size:   GridWorld 크기 (N×N)
        n_trials:    맵(시나리오) 생성 횟수
        seed_start:  시드 시작값 (각 trial은 seed_start + trial_id 사용)
        modes:       장애물 모드 목록 (None이면 기본 3종 모두)
        algorithms:  알고리즘 목록 (None이면 8종 모두)

    Returns:
        list[TrialRecord]
    """
    if modes is None:
        modes = ["urban_block", "random", "pursuer_centered"]
    if algorithms is None:
        algorithms = ALGORITHM_NAMES

    records: list[TrialRecord] = []
    trial_id = 0

    for mode in modes:
        for t in range(n_trials):
            seed = seed_start + t

            try:
                env, start, goal, pursuer = make_scenario(
                    mode=mode,
                    size=grid_size,
                    seed=seed,
                    min_dist=max(3, grid_size // 2),
                )
            except RuntimeError:
                # 유효 시나리오 생성 실패 — 건너뜀
                continue

            for algo in algorithms:
                result: PathResult = run_algorithm(
                    name=algo,
                    env=env,
                    start=start,
                    goal=goal,
                    pursuer_pos=pursuer,
                    rrt_seed=seed,
                )

                if result.found and result.path:
                    los_stats = env.path_los_stats(result.path, pursuer)
                    exposure_pct = los_stats["exposure_pct"]
                    los_breaks   = los_stats["los_breaks"]
                    path_len     = los_stats["total_cells"]
                    path_cost    = result.cost
                else:
                    exposure_pct = 100.0
                    los_breaks   = 0
                    path_len     = 0
                    path_cost    = float("inf")

                records.append(TrialRecord(
                    trial_id=trial_id,
                    mode=mode,
                    seed=seed,
                    algorithm=algo,
                    found=result.found,
                    path_cost=path_cost,
                    path_length=path_len,
                    los_exposure_pct=exposure_pct,
                    los_breaks=los_breaks,
                    nodes_expanded=result.nodes_expanded,
                    exec_time_ms=result.exec_time_ms,
                ))

            trial_id += 1

    return records


# ── 통계 집계 ─────────────────────────────────────────────────────────────────

@dataclass
class AlgorithmStats:
    name: str
    label: str
    success_rate: float
    path_cost_mean: float
    path_cost_std: float
    los_exposure_mean: float
    los_exposure_std: float
    los_breaks_mean: float
    exec_time_mean: float
    exec_time_std: float
    nodes_expanded_mean: float
    composite_score: float     # 높을수록 좋음 (0~1)
    n_success: int
    n_total: int


def compute_stats(records: list[TrialRecord]) -> list[AlgorithmStats]:
    """
    TrialRecord 목록에서 알고리즘별 통계를 계산한다.

    복합 점수 공식 (가중합, 높을수록 좋음):
        composite = 0.30 × (1 - norm_cost)
                  + 0.25 × (1 - los_exposure/100)
                  + 0.20 × (1 - norm_time)
                  + 0.15 × success_rate
                  + 0.10 × (1 - norm_breaks)

    정규화: 각 지표의 알고리즘 간 최소/최대를 기준으로 Min-Max 정규화.
    실패한 trial은 worst-case 값으로 대체 (cost=inf → 최대값 × 2).
    """
    # 알고리즘별 레코드 그룹화
    groups: dict[str, list[TrialRecord]] = {a: [] for a in ALGORITHM_NAMES}
    for r in records:
        if r.algorithm in groups:
            groups[r.algorithm].append(r)

    # 성공한 trial만 집계 (실패는 worst-case)
    raw: dict[str, dict[str, list[float]]] = {}
    for algo, recs in groups.items():
        if not recs:
            continue
        success = [r for r in recs if r.found]
        n_total = len(recs)
        n_success = len(success)
        success_rate = n_success / n_total if n_total > 0 else 0.0

        costs         = [r.path_cost for r in success] if success else [1e9]
        exposures     = [r.los_exposure_pct for r in success] if success else [100.0]
        breaks        = [r.los_breaks for r in success] if success else [0.0]
        times         = [r.exec_time_ms for r in recs]
        nodes         = [r.nodes_expanded for r in recs]

        raw[algo] = {
            "success_rate":    [success_rate],
            "path_cost":       costs,
            "los_exposure":    exposures,
            "los_breaks":      breaks,
            "exec_time":       times,
            "nodes_expanded":  nodes,
            "n_success":       [n_success],
            "n_total":         [n_total],
        }

    if not raw:
        return []

    algos_present = [a for a in ALGORITHM_NAMES if a in raw]

    def _safe_mean(vals: list[float]) -> float:
        finite = [v for v in vals if math.isfinite(v)]
        return float(np.mean(finite)) if finite else 0.0

    def _safe_std(vals: list[float]) -> float:
        finite = [v for v in vals if math.isfinite(v)]
        return float(np.std(finite)) if len(finite) > 1 else 0.0

    # 정규화 범위 계산 (알고리즘 간 min/max)
    def _minmax_range(key: str) -> tuple[float, float]:
        all_vals = []
        for algo in algos_present:
            all_vals.extend(
                [v for v in raw[algo][key] if math.isfinite(v)]
            )
        if not all_vals:
            return 0.0, 1.0
        lo, hi = min(all_vals), max(all_vals)
        if hi == lo:
            return lo, lo + 1.0
        return lo, hi

    cost_lo, cost_hi   = _minmax_range("path_cost")
    time_lo, time_hi   = _minmax_range("exec_time")
    break_lo, break_hi = _minmax_range("los_breaks")

    def _norm(val: float, lo: float, hi: float) -> float:
        if hi == lo:
            return 0.0
        return max(0.0, min(1.0, (val - lo) / (hi - lo)))

    stats_list: list[AlgorithmStats] = []
    for algo in algos_present:
        d = raw[algo]
        sr   = d["success_rate"][0]
        cost = _safe_mean(d["path_cost"])
        exp  = _safe_mean(d["los_exposure"])
        brk  = _safe_mean(d["los_breaks"])
        t    = _safe_mean(d["exec_time"])

        norm_cost  = _norm(cost, cost_lo, cost_hi)
        norm_time  = _norm(t,    time_lo, time_hi)
        norm_break = _norm(brk,  break_lo, break_hi)

        composite = (
            0.30 * (1.0 - norm_cost)
            + 0.25 * (1.0 - exp / 100.0)
            + 0.20 * (1.0 - norm_time)
            + 0.15 * sr
            + 0.10 * (1.0 - norm_break)
        )

        stats_list.append(AlgorithmStats(
            name=algo,
            label=ALGORITHM_LABELS.get(algo, algo),
            success_rate=sr,
            path_cost_mean=cost,
            path_cost_std=_safe_std(d["path_cost"]),
            los_exposure_mean=exp,
            los_exposure_std=_safe_std(d["los_exposure"]),
            los_breaks_mean=brk,
            exec_time_mean=t,
            exec_time_std=_safe_std(d["exec_time"]),
            nodes_expanded_mean=_safe_mean(d["nodes_expanded"]),
            composite_score=composite,
            n_success=int(d["n_success"][0]),
            n_total=int(d["n_total"][0]),
        ))

    # composite_score 내림차순 정렬
    stats_list.sort(key=lambda s: s.composite_score, reverse=True)
    return stats_list


# ── 보고서 생성 ───────────────────────────────────────────────────────────────

def build_report(stats: list[AlgorithmStats], grid_size: int, n_trials: int) -> str:
    """
    Markdown 보고서 문자열을 생성한다.
    Docs/EXPERIMENTS.md 자동 추가용.
    """
    lines: list[str] = []
    lines.append("\n## 경로 탐색 알고리즘 비교 — 자동 생성")
    lines.append(f"\n- **Grid 크기**: {grid_size}×{grid_size}")
    lines.append(f"- **Trial 수 (모드별)**: {n_trials}")
    lines.append("")
    lines.append("### 종합 순위 (Composite Score 기준)\n")
    lines.append("| 순위 | 알고리즘 | Composite | Success | Cost (μ) | LOS Exp% (μ) | Time ms (μ) |")
    lines.append("|---|---|---|---|---|---|---|")

    for rank, s in enumerate(stats, 1):
        lines.append(
            f"| {rank} | {s.label} | {s.composite_score:.3f} "
            f"| {s.success_rate*100:.1f}% "
            f"| {s.path_cost_mean:.2f} "
            f"| {s.los_exposure_mean:.1f}% "
            f"| {s.exec_time_mean:.2f} |"
        )

    if stats:
        best = stats[0]
        lines.append(f"\n**추천 알고리즘**: `{best.name}` — {best.label}")
        lines.append(
            f"(복합 점수 {best.composite_score:.3f}, "
            f"성공률 {best.success_rate*100:.1f}%, "
            f"LOS 노출 {best.los_exposure_mean:.1f}%)"
        )

    lines.append("")
    return "\n".join(lines)


def records_to_dict(records: list[TrialRecord]) -> list[dict[str, Any]]:
    """JSON 직렬화용 dict 변환."""
    return [
        {
            "trial_id":        r.trial_id,
            "mode":            r.mode,
            "seed":            r.seed,
            "algorithm":       r.algorithm,
            "found":           r.found,
            "path_cost":       r.path_cost if math.isfinite(r.path_cost) else None,
            "path_length":     r.path_length,
            "los_exposure_pct": r.los_exposure_pct,
            "los_breaks":      r.los_breaks,
            "nodes_expanded":  r.nodes_expanded,
            "exec_time_ms":    r.exec_time_ms,
        }
        for r in records
    ]


def stats_to_dict(stats: list[AlgorithmStats]) -> list[dict[str, Any]]:
    """JSON 직렬화용 dict 변환."""
    return [
        {
            "name":               s.name,
            "label":              s.label,
            "success_rate":       s.success_rate,
            "path_cost_mean":     s.path_cost_mean,
            "path_cost_std":      s.path_cost_std,
            "los_exposure_mean":  s.los_exposure_mean,
            "los_exposure_std":   s.los_exposure_std,
            "los_breaks_mean":    s.los_breaks_mean,
            "exec_time_mean":     s.exec_time_mean,
            "exec_time_std":      s.exec_time_std,
            "nodes_expanded_mean": s.nodes_expanded_mean,
            "composite_score":    s.composite_score,
            "n_success":          s.n_success,
            "n_total":            s.n_total,
        }
        for s in stats
    ]
