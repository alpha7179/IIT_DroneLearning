# EXPERIMENTS.md — 회피 드론(Evader) 실험 로그

> 실험 완료 시 반드시 이 파일에 결과를 한 줄로 기록한다.
> `/log` 커맨드를 사용하면 자동으로 추가된다.
>
> **기록 규칙**: 날짜, commit hash, config, run-id, seed, 핵심 지표, 한 줄 코멘트
> **보존 규칙**: 지표가 나쁜 실험도 삭제하지 않는다. 실패 원인 코멘트 필수.

---

## Stage0 — Ray + 힌트 허용 (장애물 없음)

목표: `survival_rate ≥ 70%`, `goal_reach_rate ≥ 50%`, `capture_rate ≤ 30%`

| 날짜 | commit | config | run-id | seed | survival | goal | capture | crash | 코멘트 |
|------|--------|--------|--------|------|----------|------|---------|-------|--------|
| _(예시)_ | `a76f38d` | `evader_s0_template` | `evader_s0_base_seed42` | 42 | 72% | 51% | 25% | 3% | Stage0 첫 수렴 확인 |

---

## Stage1 — 장애물 도입, 힌트 제거

목표: `survival_rate ≥ 50%`, `crash_rate ≤ 15%`

| 날짜 | commit | config | run-id | seed | survival | goal | capture | crash | 코멘트 |
|------|--------|--------|--------|------|----------|------|---------|-------|--------|

---

## Stage2 (M3) — Self-Play / Competitive Learning

목표: `survival_rate ≥ 60%` (RL 추격자 상대)

| 날짜 | commit | config | run-id | seed | survival | goal | capture | los_break | 코멘트 |
|------|--------|--------|--------|------|----------|------|---------|-----------|--------|

---

## Stage3 (M4) — Vision + LOS 은폐 전략

목표: LOS 차단 상황에서 생존 시간 증가

| 날짜 | commit | config | run-id | seed | survival | goal | los_break_rate | 코멘트 |
|------|--------|--------|--------|------|----------|------|----------------|--------|

---

## Stage4 (M5) — 도메인 랜덤화 (Sim2Real)

목표: 랜덤화 강도 증가 시 성능 완만 감소

| 날짜 | commit | config | run-id | seed | survival | goal | rand_strength | 코멘트 |
|------|--------|--------|--------|------|----------|------|---------------|--------|

---

## 센서별 비교 실험 (Phase 5, 11~12주차)

| 실험 ID | 센서 구성 | 전역 맵 | survival | goal | capture | 코멘트 |
|---------|---------|--------|----------|------|---------|--------|
| E1 | Ray만 | O | | | | |
| E2 | 카메라만 | O | | | | |
| E3 | Ray + 카메라 | O | | | | |
| E4 | Ray만 | X | | | | |

---

## 메모 / 이슈

- 보상 해킹 사례, 재현 방법, 대응책 등 자유 형식으로 기록
