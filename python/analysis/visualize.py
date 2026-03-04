"""
visualize.py — 경로 탐색 결과 시각화

출력물:
  1. grid_comparison.png  — 단일 맵에서 8 알고리즘 경로 비교 (3×3 서브플롯)
  2. metrics_bar.png      — 4개 지표 막대 차트 (N맵 평균)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")          # GUI 없는 환경 (Colab/서버) 대응
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from .algorithms import ALGORITHM_LABELS, ALGORITHM_NAMES, PathResult
from .benchmark import AlgorithmStats
from .grid_env import GridWorld


# ── 색상 상수 ─────────────────────────────────────────────────────────────────

_COL_OBSTACLE  = "#555555"
_COL_EMPTY     = "#f8f8f8"
_COL_EXPOSED   = "#ffcccc"   # LOS 노출 셀 (빨강 반투명)
_COL_HIDDEN    = "#ccffcc"   # LOS 은폐 셀 (초록 반투명)
_COL_PATH      = "#1a6bb5"   # 경로 선
_COL_START     = "#2ecc71"   # 시작점
_COL_GOAL      = "#e74c3c"   # 목표점
_COL_PURSUER   = "#f39c12"   # 추격자


# ── 그리드 단일 서브플롯 ──────────────────────────────────────────────────────

def _draw_grid(
    ax: plt.Axes,
    env: GridWorld,
    pursuer_pos: tuple[int, int],
    start: tuple[int, int],
    goal: tuple[int, int],
    result: PathResult | None,
    title: str,
) -> None:
    """단일 Axes에 GridWorld + 경로를 그린다."""
    size = env.size

    # ── 배경: LOS 맵 ──────────────────────────────────────────────────────────
    los_map = env.compute_los_map(pursuer_pos)
    img = np.zeros((size, size, 3))

    for r in range(size):
        for c in range(size):
            if env.grid[r, c] == 1:               # 장애물
                img[r, c] = [0.33, 0.33, 0.33]
            elif los_map[r, c] == 1:              # LOS 노출
                img[r, c] = [1.0, 0.8, 0.8]
            else:                                  # LOS 은폐
                img[r, c] = [0.8, 1.0, 0.8]

    ax.imshow(img, origin="upper", interpolation="nearest")

    # ── 격자선 ────────────────────────────────────────────────────────────────
    for x in range(size + 1):
        ax.axhline(x - 0.5, color="#cccccc", linewidth=0.3)
        ax.axvline(x - 0.5, color="#cccccc", linewidth=0.3)

    # ── 경로 ──────────────────────────────────────────────────────────────────
    if result is not None and result.found and len(result.path) >= 2:
        path_r = [p[0] for p in result.path]
        path_c = [p[1] for p in result.path]
        ax.plot(path_c, path_r, color=_COL_PATH, linewidth=1.5, zorder=3)

    # ── 특수 셀 마커 ──────────────────────────────────────────────────────────
    for (r, c), color, marker, size_pt, label in [
        (start,       _COL_START,   "s",  80,  "Start"),
        (goal,        _COL_GOAL,    "*",  120, "Goal"),
        (pursuer_pos, _COL_PURSUER, "^",  80,  "Pursuer"),
    ]:
        ax.scatter(c, r, color=color, marker=marker, s=size_pt, zorder=5)

    ax.set_title(title, fontsize=8, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)


# ── 8알고리즘 그리드 비교도 ───────────────────────────────────────────────────

def plot_grid_comparison(
    env: GridWorld,
    start: tuple[int, int],
    goal: tuple[int, int],
    pursuer_pos: tuple[int, int],
    results: dict[str, PathResult],
    out_path: str | Path,
) -> None:
    """
    3×3 서브플롯으로 8 알고리즘의 경로를 나란히 시각화한다.
    (마지막 칸: 범례 + 환경 요약)

    Args:
        env:         GridWorld 환경
        start:       시작 위치
        goal:        목표 위치
        pursuer_pos: 추격자 위치
        results:     {algo_name: PathResult}
        out_path:    저장 경로 (.png)
    """
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(
        f"Pathfinding Algorithm Comparison — {env.size}×{env.size} Grid",
        fontsize=13, fontweight="bold", y=0.98
    )

    algo_list = ALGORITHM_NAMES  # 고정 순서

    for i, algo in enumerate(algo_list):
        ax = axes[i // 3][i % 3]
        result = results.get(algo)
        label = ALGORITHM_LABELS.get(algo, algo)

        if result and result.found:
            cost_str = f"cost={result.cost:.2f}"
            time_str = f"{result.exec_time_ms:.1f}ms"
            title = f"{label}\n{cost_str} | {time_str}"
        elif result:
            title = f"{label}\n[경로 없음]"
        else:
            title = f"{label}\n[미실행]"

        _draw_grid(ax, env, pursuer_pos, start, goal, result, title)

    # 마지막 칸: 범례 + 요약 텍스트
    ax_legend = axes[2][2]
    ax_legend.axis("off")

    patches = [
        mpatches.Patch(color=_COL_EXPOSED[:-2],  label="LOS 노출 (빨강)"),
        mpatches.Patch(color=_COL_HIDDEN[:-2],   label="LOS 은폐 (초록)"),
        mpatches.Patch(color=_COL_OBSTACLE,       label="장애물"),
        mpatches.Patch(color=_COL_PATH,           label="경로"),
        mpatches.Patch(color=_COL_START,          label="시작 (■)"),
        mpatches.Patch(color=_COL_GOAL,           label="목표 (★)"),
        mpatches.Patch(color=_COL_PURSUER,        label="추격자 (▲)"),
    ]
    ax_legend.legend(handles=patches, loc="center", fontsize=9,
                     frameon=True, framealpha=0.9, title="범례")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Grid comparison saved: {out_path}")


# ── 지표 막대 차트 ────────────────────────────────────────────────────────────

def plot_metrics_bar(
    stats: list[AlgorithmStats],
    out_path: str | Path,
) -> None:
    """
    4개 지표 × 알고리즘 막대 차트를 생성한다.

    서브플롯:
      (1) Composite Score (높을수록 좋음)
      (2) LOS Exposure % (낮을수록 좋음)
      (3) Path Cost (낮을수록 좋음)
      (4) Exec Time ms (낮을수록 좋음)
    """
    if not stats:
        print("[visualize] No stats to plot.")
        return

    labels   = [s.label for s in stats]
    scores   = [s.composite_score for s in stats]
    exposure = [s.los_exposure_mean for s in stats]
    costs    = [s.path_cost_mean for s in stats]
    times    = [s.exec_time_mean for s in stats]

    x = np.arange(len(labels))
    bar_w = 0.6
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(labels)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Pathfinding Algorithm Metrics Comparison", fontsize=13, fontweight="bold")

    def _bar_plot(ax: plt.Axes, values: list[float], title: str,
                  ylabel: str, higher_is_better: bool, fmt: str = ".2f") -> None:
        bars = ax.bar(x, values, bar_w, color=colors, edgecolor="#555")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

        # 값 레이블
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{val:{fmt}}",
                ha="center", va="bottom", fontsize=7
            )

        arrow = "↑ 높을수록 좋음" if higher_is_better else "↓ 낮을수록 좋음"
        ax.text(0.98, 0.98, arrow, transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="#555")

    _bar_plot(axes[0][0], scores,   "Composite Score",    "Score",  True,  ".3f")
    _bar_plot(axes[0][1], exposure, "LOS Exposure (%)",   "Exp %",  False, ".1f")
    _bar_plot(axes[1][0], costs,    "Path Cost (mean)",   "Cost",   False, ".2f")
    _bar_plot(axes[1][1], times,    "Exec Time ms (mean)", "ms",    False, ".2f")

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Metrics bar chart saved: {out_path}")
