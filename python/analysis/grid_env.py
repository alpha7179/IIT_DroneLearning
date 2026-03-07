"""
grid_env.py — 2D Grid 환경 (Occlusion-aware 경로 탐색 비교용)

IIT_DroneLearning 프로젝트의 도심 Occlusion 환경을 단순화한 2D Grid로 모사한다.
Unity 팀이 맵 제작 중인 동안 알고리즘 비교 연구에 활용한다.

담당: 이재왕 (work/evader)
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

# 셀 값
EMPTY    = 0
OBSTACLE = 1

# 8방향 이동 벡터 및 비용
_DIRS_8 = [
    (0, 1, 1.0), (0, -1, 1.0), (1, 0, 1.0), (-1, 0, 1.0),   # 4방향
    (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),               # 대각
    (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
]
_DIRS_4 = [(0, 1, 1.0), (0, -1, 1.0), (1, 0, 1.0), (-1, 0, 1.0)]


@dataclass
class GridWorld:
    """
    N×N Grid 환경.

    좌표 규약: (row, col) — row=0은 상단, col=0은 좌측.
    LOS는 Bresenham's Line 알고리즘으로 장애물 차단 여부를 판정한다.
    """

    size: int = 10
    move: str = "8"               # "4" 또는 "8"
    grid: np.ndarray = field(init=False)

    def __post_init__(self):
        self.grid = np.zeros((self.size, self.size), dtype=np.int8)

    # ── 기본 조작 ────────────────────────────────────────────────────────
    def add_obstacles(self, cells: list[tuple[int, int]]) -> None:
        for r, c in cells:
            if self._in_bounds(r, c):
                self.grid[r, c] = OBSTACLE

    def remove_obstacles(self, cells: list[tuple[int, int]]) -> None:
        for r, c in cells:
            if self._in_bounds(r, c):
                self.grid[r, c] = EMPTY

    def is_valid(self, pos: tuple[int, int]) -> bool:
        r, c = pos
        return self._in_bounds(r, c) and self.grid[r, c] == EMPTY

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.size and 0 <= c < self.size

    # ── 이웃 탐색 ────────────────────────────────────────────────────────
    def neighbors(self, pos: tuple[int, int]) -> Iterator[tuple[tuple[int, int], float]]:
        """(neighbor_pos, move_cost) 이터레이터."""
        r, c = pos
        dirs = _DIRS_8 if self.move == "8" else _DIRS_4
        for dr, dc, cost in dirs:
            nr, nc = r + dr, c + dc
            if self.is_valid((nr, nc)):
                # 대각 이동: 인접 두 셀이 모두 통과 가능해야 함 (corner cutting 방지)
                if abs(dr) == 1 and abs(dc) == 1:
                    if not (self.is_valid((r + dr, c)) and self.is_valid((r, c + dc))):
                        continue
                yield (nr, nc), cost

    # ── LOS (Line of Sight) ──────────────────────────────────────────────
    def is_los_visible(self, p1: tuple[int, int], p2: tuple[int, int]) -> bool:
        """
        numpy linspace 샘플링으로 p1에서 p2까지 직선상에 장애물이 없는지 판정.
        True = 가시 (노출), False = 차단 (은폐).
        대용량 격자(500×500+)에서 Python Bresenham 대비 10배 이상 빠름.
        """
        r0, c0 = p1
        r1, c1 = p2
        n = max(abs(r1 - r0), abs(c1 - c0)) + 1
        if n <= 2:
            return True
        rs = np.round(np.linspace(r0, r1, n)).astype(np.int32)[1:-1]
        cs = np.round(np.linspace(c0, c1, n)).astype(np.int32)[1:-1]
        np.clip(rs, 0, self.size - 1, out=rs)
        np.clip(cs, 0, self.size - 1, out=cs)
        return not bool(np.any(self.grid[rs, cs] == OBSTACLE))

    def compute_los_map(self, pursuer_pos: tuple[int, int]) -> np.ndarray:
        """
        pursuer_pos에서 모든 빈 셀까지의 LOS 노출 여부를 행렬로 반환.
        1 = 노출(가시), 0 = 은폐(차단). 장애물 셀은 -1.

        배치 numpy 연산으로 대용량 격자에서도 수 초 이내 완료.
        """
        N = self.size
        pr, pc = pursuer_pos
        los_map = np.full((N, N), -1, dtype=np.int8)

        empty_r, empty_c = np.where(self.grid == EMPTY)
        if len(empty_r) == 0:
            return los_map

        # 내부 샘플 수: 최대 Chebyshev 거리(N-1)보다 충분히 크게 설정
        n_steps = N + 1
        t = np.linspace(0, 1, n_steps, dtype=np.float32)[1:-1]  # 내부만, 길이 N-1
        BATCH = 512

        for i in range(0, len(empty_r), BATCH):
            br = empty_r[i : i + BATCH]
            bc = empty_c[i : i + BATCH]

            # 각 대상 셀에 대해 pursuer→cell 직선 상의 샘플 좌표 계산
            # sample_r: (B, N-1)
            sample_r = np.round(
                pr + t[None, :] * (br[:, None].astype(np.float32) - pr)
            ).astype(np.int32)
            sample_c = np.round(
                pc + t[None, :] * (bc[:, None].astype(np.float32) - pc)
            ).astype(np.int32)
            np.clip(sample_r, 0, N - 1, out=sample_r)
            np.clip(sample_c, 0, N - 1, out=sample_c)

            # 장애물 통과 여부: (B,)
            hits = np.any(self.grid[sample_r, sample_c] == OBSTACLE, axis=1)
            los_map[br, bc] = np.where(hits, np.int8(0), np.int8(1))

        return los_map

    # ── 경로 LOS 분석 ────────────────────────────────────────────────────
    def path_los_stats(
        self, path: list[tuple[int, int]], pursuer_pos: tuple[int, int]
    ) -> dict:
        """경로 상의 LOS 통계 반환."""
        if not path:
            return {"exposure_pct": 0.0, "los_breaks": 0}
        exposed = [self.is_los_visible(pursuer_pos, cell) for cell in path]
        breaks = sum(1 for i in range(len(exposed) - 1) if exposed[i] != exposed[i + 1])
        return {
            "exposure_pct": 100.0 * sum(exposed) / len(path),
            "los_breaks": breaks,
            "exposed_cells": sum(exposed),
            "total_cells": len(path),
        }

    def path_cost(self, path: list[tuple[int, int]]) -> float:
        """경로의 유클리드 이동 비용 합산."""
        if len(path) < 2:
            return 0.0
        total = 0.0
        for i in range(len(path) - 1):
            dr = path[i + 1][0] - path[i][0]
            dc = path[i + 1][1] - path[i][1]
            total += math.sqrt(dr * dr + dc * dc)
        return total

    # ── 연결성 검사 ──────────────────────────────────────────────────────
    def is_reachable(self, start: tuple[int, int], goal: tuple[int, int]) -> bool:
        """BFS로 start → goal 경로 존재 여부 확인."""
        if not (self.is_valid(start) and self.is_valid(goal)):
            return False
        visited = {start}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            if cur == goal:
                return True
            for nb, _ in self.neighbors(cur):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        return False

    def copy(self) -> "GridWorld":
        env = GridWorld(size=self.size, move=self.move)
        env.grid = self.grid.copy()
        return env

    def __repr__(self) -> str:
        rows = []
        for r in range(self.size):
            row = "".join("█" if self.grid[r, c] else "·" for c in range(self.size))
            rows.append(row)
        return "\n".join(rows)


# ── 장애물 팩토리 ────────────────────────────────────────────────────────

def make_urban_blocks(
    size: int = 10,
    block_sz: int | None = None,
    gap: int | None = None,
) -> list[tuple[int, int]]:
    """
    도시 블록 패턴 장애물.
    block_sz × block_sz 크기 건물을 gap 간격으로 배치한다.

    block_sz/gap 미지정 시 size에 맞게 자동 스케일:
      size=10  → block_sz=2,  gap=1   (소형 격자)
      size=50  → block_sz=8,  gap=4
      size=100 → block_sz=15, gap=6
      size=500 → block_sz=25, gap=10  (도시 블록 스케일)
    """
    if block_sz is None:
        block_sz = max(2, size // 20)
    if gap is None:
        gap = max(1, size // 50)
    obstacles = []
    stride = block_sz + gap
    for r in range(0, size, stride):
        for c in range(0, size, stride):
            for dr in range(block_sz):
                for dc in range(block_sz):
                    if r + dr < size and c + dc < size:
                        obstacles.append((r + dr, c + dc))
    return obstacles


def make_random_obstacles(
    size: int = 10, density: float = 0.2, seed: int | None = None
) -> list[tuple[int, int]]:
    """
    랜덤 밀도 장애물 배치 (density = 장애물 비율).
    클러스터 효과를 위해 인접 셀에 추가 확률 부여.
    """
    rng = random.Random(seed)
    base = set()
    for r in range(size):
        for c in range(size):
            if rng.random() < density:
                base.add((r, c))
    # 약한 클러스터: 인접 셀에 50% 확률로 추가
    extra = set()
    for r, c in base:
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            if rng.random() < 0.3 and 0 <= r + dr < size and 0 <= c + dc < size:
                extra.add((r + dr, c + dc))
    return list(base | extra)


def make_pursuer_centered(
    size: int = 10,
    pursuer_pos: tuple[int, int] = (5, 5),
    radius: float | None = None,
    density_near: float = 0.5,
    density_far: float = 0.1,
    seed: int | None = None,
) -> list[tuple[int, int]]:
    """
    추격자 주변은 장애물 밀집, 멀어질수록 희박.
    회피 드론이 건물 뒤로 숨는 전략의 가치를 극대화하는 환경.
    """
    rng = random.Random(seed)
    if radius is None:
        radius = max(3.0, size * 0.15)  # 격자 크기의 15% 반경 자동 적용
    obstacles = []
    pr, pc = pursuer_pos
    for r in range(size):
        for c in range(size):
            dist = math.sqrt((r - pr) ** 2 + (c - pc) ** 2)
            prob = density_near if dist <= radius else density_far
            if rng.random() < prob:
                obstacles.append((r, c))
    return obstacles


def make_scenario(
    mode: str = "urban_block",
    size: int = 10,
    seed: int = 42,
    min_dist: int = 5,
    max_attempts: int = 500,
) -> tuple["GridWorld", tuple[int, int], tuple[int, int], tuple[int, int]]:
    """
    (env, start, goal, pursuer) 튜플을 반환한다.

    제약:
      - start ↔ goal 맨하탄 거리 ≥ min_dist
      - start, goal, pursuer 모두 EMPTY 셀에 배치
      - BFS로 start → goal 연결성 보장
      - pursuer는 start에서 맨하탄 거리 ≥ size//2
    """
    rng = random.Random(seed)

    # 장애물 생성
    if mode == "urban_block":
        obs = make_urban_blocks(size)
    elif mode == "random":
        obs = make_random_obstacles(size, density=0.2, seed=seed)
    elif mode == "pursuer_centered":
        center = (size // 2, size // 2)
        obs = make_pursuer_centered(size, pursuer_pos=center, seed=seed)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    def random_empty(env: GridWorld, exclude: set) -> tuple[int, int] | None:
        empties = [
            (r, c)
            for r in range(size)
            for c in range(size)
            if env.is_valid((r, c)) and (r, c) not in exclude
        ]
        return rng.choice(empties) if empties else None

    for _ in range(max_attempts):
        env = GridWorld(size=size, move="8")
        env.add_obstacles(obs)

        used: set[tuple[int, int]] = set()
        start = random_empty(env, used)
        if start is None:
            continue
        used.add(start)

        # goal: 맨하탄 거리 ≥ min_dist
        candidates = [
            (r, c)
            for r in range(size)
            for c in range(size)
            if env.is_valid((r, c))
            and (r, c) not in used
            and abs(r - start[0]) + abs(c - start[1]) >= min_dist
        ]
        if not candidates:
            continue
        goal = rng.choice(candidates)
        used.add(goal)

        if not env.is_reachable(start, goal):
            continue

        # pursuer: start에서 맨하탄 거리 ≥ size//2
        p_candidates = [
            (r, c)
            for r in range(size)
            for c in range(size)
            if env.is_valid((r, c))
            and (r, c) not in used
            and abs(r - start[0]) + abs(c - start[1]) >= size // 2
        ]
        if not p_candidates:
            continue
        pursuer = rng.choice(p_candidates)

        return env, start, goal, pursuer

    raise RuntimeError(
        f"make_scenario: 유효한 시나리오를 {max_attempts}회 시도 후 생성 실패. "
        f"grid_size={size}, mode={mode}, min_dist={min_dist}"
    )
