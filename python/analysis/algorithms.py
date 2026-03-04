"""
algorithms.py — 8종 경로 탐색 알고리즘 구현체

비교 대상:
  1. A* (Manhattan 휴리스틱)
  2. A* (Octile / 8방향 최적)
  3. A* + LOS-Penalty (w=0.5)
  4. A* + LOS-Penalty (w=1.0)
  5. Dijkstra (h=0)
  6. Greedy Best-First Search
  7. Theta* (Any-angle, LOS 직선화)
  8. RRT* (샘플링 기반)

공통 반환 타입: PathResult
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Callable

from .grid_env import GridWorld


# ── 공통 반환 타입 ────────────────────────────────────────────────────────────

@dataclass
class PathResult:
    path: list[tuple[int, int]]
    cost: float
    nodes_expanded: int
    exec_time_ms: float
    found: bool


# ── 휴리스틱 함수 ─────────────────────────────────────────────────────────────

def heuristic_manhattan(a: tuple[int, int], b: tuple[int, int]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def heuristic_octile(a: tuple[int, int], b: tuple[int, int]) -> float:
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return max(dr, dc) + (math.sqrt(2) - 1) * min(dr, dc)


def heuristic_zero(a: tuple[int, int], b: tuple[int, int]) -> float:
    return 0.0


# ── A* (공통 코어) ────────────────────────────────────────────────────────────

def astar(
    env: GridWorld,
    start: tuple[int, int],
    goal: tuple[int, int],
    heuristic: Callable = heuristic_octile,
    los_weight: float = 0.0,
    pursuer_pos: tuple[int, int] | None = None,
    greedy: bool = False,
) -> PathResult:
    """
    A* 기반 범용 경로 탐색.

    Args:
        env:          GridWorld 환경
        start:        시작 위치 (r, c)
        goal:         목표 위치 (r, c)
        heuristic:    휴리스틱 함수 h(n, goal)
        los_weight:   LOS 노출 페널티 가중치 (0.0 = 비활성)
        pursuer_pos:  추격자 위치 (los_weight > 0 시 필수)
        greedy:       True = Greedy Best-First (g 무시, f=h만 사용)

    Returns:
        PathResult
    """
    t0 = time.perf_counter()

    if not (env.is_valid(start) and env.is_valid(goal)):
        return PathResult([], float("inf"), 0, 0.0, False)

    if start == goal:
        return PathResult([start], 0.0, 0, 0.0, True)

    # LOS 맵 사전 계산 (los_weight > 0인 경우)
    los_map: dict[tuple[int, int], bool] = {}
    if los_weight > 0 and pursuer_pos is not None:
        for r in range(env.size):
            for c in range(env.size):
                if env.grid[r, c] == 0:
                    los_map[(r, c)] = env.is_los_visible(pursuer_pos, (r, c))

    def los_penalty(pos: tuple[int, int]) -> float:
        if los_weight <= 0 or not los_map:
            return 0.0
        return los_weight if los_map.get(pos, False) else 0.0

    # 우선순위 큐: (f, g, pos)
    # g_cost: 시작점부터 현재 노드까지 실제 비용
    g_cost: dict[tuple[int, int], float] = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    nodes_expanded = 0

    h0 = heuristic(start, goal)
    open_heap: list[tuple[float, float, tuple[int, int]]] = [(h0, 0.0, start)]

    while open_heap:
        f, g, cur = heappop(open_heap)

        if cur == goal:
            # 경로 역추적
            path: list[tuple[int, int]] = []
            node: tuple[int, int] | None = goal
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            exec_ms = (time.perf_counter() - t0) * 1000
            return PathResult(path, g, nodes_expanded, exec_ms, True)

        # 이미 더 좋은 경로가 처리됐으면 스킵
        if g > g_cost.get(cur, float("inf")):
            continue

        nodes_expanded += 1

        for nb, move_cost in env.neighbors(cur):
            # LOS 페널티 포함 이동 비용
            new_g = g + move_cost + los_penalty(nb)

            if new_g < g_cost.get(nb, float("inf")):
                g_cost[nb] = new_g
                parent[nb] = cur
                h = heuristic(nb, goal)
                f_val = h if greedy else (new_g + h)
                heappush(open_heap, (f_val, new_g, nb))

    exec_ms = (time.perf_counter() - t0) * 1000
    return PathResult([], float("inf"), nodes_expanded, exec_ms, False)


# ── Theta* (Any-angle A*) ─────────────────────────────────────────────────────

def theta_star(
    env: GridWorld,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> PathResult:
    """
    Theta*: A*에서 부모 재할당을 통해 Any-angle 직선 경로 생성.
    LOS가 확보되면 그랜드패런트를 직접 부모로 설정한다.
    """
    t0 = time.perf_counter()

    if not (env.is_valid(start) and env.is_valid(goal)):
        return PathResult([], float("inf"), 0, 0.0, False)

    if start == goal:
        return PathResult([start], 0.0, 0, 0.0, True)

    def dist(a: tuple[int, int], b: tuple[int, int]) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    g_cost: dict[tuple[int, int], float] = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    nodes_expanded = 0

    h0 = heuristic_octile(start, goal)
    open_heap: list[tuple[float, tuple[int, int]]] = [(h0, start)]

    while open_heap:
        _, cur = heappop(open_heap)

        if cur == goal:
            path: list[tuple[int, int]] = []
            node: tuple[int, int] | None = goal
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            # 실제 경로 비용 재계산 (직선 거리 합)
            cost = env.path_cost(path)
            exec_ms = (time.perf_counter() - t0) * 1000
            return PathResult(path, cost, nodes_expanded, exec_ms, True)

        g_cur = g_cost.get(cur, float("inf"))
        # 이미 더 좋은 경로가 처리됐으면 스킵
        if g_cur == float("inf"):
            continue

        nodes_expanded += 1

        for nb, _ in env.neighbors(cur):
            # Theta* 핵심: 부모의 부모에서 직선 LOS가 있으면 바로 연결
            p_cur = parent[cur]
            if p_cur is not None and env.is_los_visible(p_cur, nb):
                new_g = g_cost.get(p_cur, float("inf")) + dist(p_cur, nb)
                if new_g < g_cost.get(nb, float("inf")):
                    g_cost[nb] = new_g
                    parent[nb] = p_cur
                    f_val = new_g + heuristic_octile(nb, goal)
                    heappush(open_heap, (f_val, nb))
            else:
                move_cost = dist(cur, nb)
                new_g = g_cur + move_cost
                if new_g < g_cost.get(nb, float("inf")):
                    g_cost[nb] = new_g
                    parent[nb] = cur
                    f_val = new_g + heuristic_octile(nb, goal)
                    heappush(open_heap, (f_val, nb))

    exec_ms = (time.perf_counter() - t0) * 1000
    return PathResult([], float("inf"), nodes_expanded, exec_ms, False)


# ── RRT* ─────────────────────────────────────────────────────────────────────

@dataclass
class _RRTNode:
    pos: tuple[int, int]
    parent: int | None = None      # 인덱스
    cost: float = 0.0


def rrt_star(
    env: GridWorld,
    start: tuple[int, int],
    goal: tuple[int, int],
    max_iter: int = 3000,
    step_size: float = 1.5,
    goal_radius: float = 1.5,
    rewire_radius: float = 3.0,
    seed: int | None = None,
) -> PathResult:
    """
    RRT*: 샘플링 기반 점근적 최적 경로 탐색.

    Grid 셀 좌표를 사용하되 연속 공간으로 확장 후 가장 가까운 유효 셀로 스냅.
    """
    t0 = time.perf_counter()

    if not (env.is_valid(start) and env.is_valid(goal)):
        return PathResult([], float("inf"), 0, 0.0, False)

    if start == goal:
        return PathResult([start], 0.0, 0, 0.0, True)

    rng = random.Random(seed)
    nodes: list[_RRTNode] = [_RRTNode(pos=start, parent=None, cost=0.0)]
    goal_idx: int | None = None

    def dist(a: tuple[int, int], b: tuple[int, int]) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def nearest(sample: tuple[float, float]) -> int:
        best_i, best_d = 0, float("inf")
        for i, n in enumerate(nodes):
            d = math.sqrt((n.pos[0] - sample[0]) ** 2 + (n.pos[1] - sample[1]) ** 2)
            if d < best_d:
                best_d, best_i = d, i
        return best_i

    def steer(
        from_pos: tuple[int, int], to_pos: tuple[float, float]
    ) -> tuple[int, int] | None:
        """from_pos → to_pos 방향으로 step_size만큼 이동 후 가장 가까운 유효 셀 반환."""
        dr = to_pos[0] - from_pos[0]
        dc = to_pos[1] - from_pos[1]
        d = math.sqrt(dr * dr + dc * dc)
        if d < 1e-9:
            return None
        scale = min(step_size, d) / d
        nr = from_pos[0] + dr * scale
        nc = from_pos[1] + dc * scale
        # 정수 좌표로 반올림
        cell = (round(nr), round(nc))
        if env.is_valid(cell):
            return cell
        return None

    def is_collision_free(
        a: tuple[int, int], b: tuple[int, int]
    ) -> bool:
        """Bresenham 직선상에 장애물이 없으면 True."""
        return env.is_los_visible(a, b)

    def near_nodes(new_pos: tuple[int, int]) -> list[int]:
        """rewire_radius 이내 노드 인덱스 목록."""
        result = []
        for i, n in enumerate(nodes):
            if dist(n.pos, new_pos) <= rewire_radius:
                result.append(i)
        return result

    nodes_expanded = 0

    for _ in range(max_iter):
        # 20% 확률로 goal을 직접 샘플링
        if rng.random() < 0.2:
            sample: tuple[float, float] = (float(goal[0]), float(goal[1]))
        else:
            sample = (rng.uniform(0, env.size - 1), rng.uniform(0, env.size - 1))

        near_idx = nearest(sample)
        new_pos = steer(nodes[near_idx].pos, sample)
        if new_pos is None or new_pos == nodes[near_idx].pos:
            continue

        if not is_collision_free(nodes[near_idx].pos, new_pos):
            continue

        nodes_expanded += 1

        # 최적 부모 선택
        neighbors_idx = near_nodes(new_pos)
        best_parent = near_idx
        best_cost = nodes[near_idx].cost + dist(nodes[near_idx].pos, new_pos)

        for idx in neighbors_idx:
            if idx == near_idx:
                continue
            c = nodes[idx].cost + dist(nodes[idx].pos, new_pos)
            if c < best_cost and is_collision_free(nodes[idx].pos, new_pos):
                best_cost = c
                best_parent = idx

        new_node = _RRTNode(pos=new_pos, parent=best_parent, cost=best_cost)
        new_idx = len(nodes)
        nodes.append(new_node)

        # Rewire
        for idx in neighbors_idx:
            if idx == best_parent:
                continue
            potential = best_cost + dist(new_pos, nodes[idx].pos)
            if potential < nodes[idx].cost and is_collision_free(new_pos, nodes[idx].pos):
                nodes[idx].parent = new_idx
                nodes[idx].cost = potential

        # Goal 도달 확인
        if dist(new_pos, goal) <= goal_radius:
            if goal_idx is None or best_cost + dist(new_pos, goal) < nodes[goal_idx].cost:
                goal_idx = new_idx

    exec_ms = (time.perf_counter() - t0) * 1000

    if goal_idx is None:
        return PathResult([], float("inf"), nodes_expanded, exec_ms, False)

    # 경로 역추적
    path: list[tuple[int, int]] = [goal]
    idx: int | None = goal_idx
    while idx is not None:
        path.append(nodes[idx].pos)
        idx = nodes[idx].parent
    path.reverse()
    # goal 셀이 path의 마지막 정확한 위치가 아닐 수 있으므로 정리
    if path[-1] != goal and env.is_valid(goal):
        path.append(goal)
    cost = env.path_cost(path)
    return PathResult(path, cost, nodes_expanded, exec_ms, True)


# ── 공개 API: 이름 → 실행 함수 매핑 ──────────────────────────────────────────

def run_algorithm(
    name: str,
    env: GridWorld,
    start: tuple[int, int],
    goal: tuple[int, int],
    pursuer_pos: tuple[int, int] | None = None,
    rrt_seed: int | None = None,
) -> PathResult:
    """
    알고리즘 이름으로 경로 탐색을 실행한다.

    Args:
        name:         알고리즘 식별자 (ALGORITHM_NAMES 목록 참조)
        env:          GridWorld 환경
        start:        시작 위치
        goal:         목표 위치
        pursuer_pos:  추격자 위치 (LOS 페널티 알고리즘에 필요)
        rrt_seed:     RRT* 재현성 시드

    Returns:
        PathResult
    """
    match name:
        case "astar_manhattan":
            return astar(env, start, goal, heuristic=heuristic_manhattan)
        case "astar_octile":
            return astar(env, start, goal, heuristic=heuristic_octile)
        case "astar_los05":
            return astar(
                env, start, goal,
                heuristic=heuristic_octile,
                los_weight=0.5,
                pursuer_pos=pursuer_pos,
            )
        case "astar_los10":
            return astar(
                env, start, goal,
                heuristic=heuristic_octile,
                los_weight=1.0,
                pursuer_pos=pursuer_pos,
            )
        case "dijkstra":
            return astar(env, start, goal, heuristic=heuristic_zero)
        case "greedy":
            return astar(env, start, goal, heuristic=heuristic_octile, greedy=True)
        case "theta_star":
            return theta_star(env, start, goal)
        case "rrt_star":
            return rrt_star(env, start, goal, seed=rrt_seed)
        case _:
            raise ValueError(f"Unknown algorithm: {name}")


# 지원 알고리즘 목록 (순서 유지)
ALGORITHM_NAMES: list[str] = [
    "astar_manhattan",
    "astar_octile",
    "astar_los05",
    "astar_los10",
    "dijkstra",
    "greedy",
    "theta_star",
    "rrt_star",
]

ALGORITHM_LABELS: dict[str, str] = {
    "astar_manhattan": "A* (Manhattan)",
    "astar_octile":    "A* (Octile)",
    "astar_los05":     "A*+LOS (w=0.5)",
    "astar_los10":     "A*+LOS (w=1.0)",
    "dijkstra":        "Dijkstra",
    "greedy":          "Greedy BFS",
    "theta_star":      "Theta*",
    "rrt_star":        "RRT*",
}
