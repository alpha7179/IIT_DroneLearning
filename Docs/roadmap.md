# Roadmap

아주대학교 2026-1 파란학기 — 2026년 3월 ~ 6월 (16주)

---

## Evader 마일스톤

| 마일스톤 | 기간 | 핵심 목표 | 완료 기준 |
|---|---|---|---|
| **M0** | Week 1 | 팀 인터페이스 계약 완료 | 에피소드 stable reset/terminate, 학습 커맨드 문서화 |
| **M1** | Week 2~3 | Stage 0 수렴 (Ray + 힌트) | survival ≥ 70%, goal_reach ≥ 50%, capture ≤ 30% |
| **M2** | Week 4~6 | Stage 1 장애물 도입 | survival ≥ 50%, crash ≤ 15% |
| **M3** | Week 7~9 | Stage 2 Self-Play | RL Pursuer 상대 survival ≥ 60% |
| **M4** | Week 10~12 | Stage 3 Vision + LOS 은폐 | LOS break 상황 생존 시간 증가, seed 3종 일관 |
| **M5** | Week 13~15 | Stage 4 도메인 랜덤화 | 랜덤화 강도 증가 시 성능 완만 감소 (붕괴 없음) |
| **M6** | Week 16 | 최종 정리 / 패키징 | ONNX export + eval report + 재현 가이드 + 시연 영상 |

---

## 전체 팀 Phase

| Phase | 기간 | 목표 |
|---|---|---|
| **Phase 1** 환경 구축 | Week 1~3 | Unity 아키텍처, 3D 도심 환경, 6-DOF 물리, ML-Agents 연결 |
| **Phase 2** 단일 에이전트 | Week 4~7 | 센서 설계, 보상 함수, Pursuer / Evader 독립 학습 |
| **Phase 3** MARL 경쟁 학습 | Week 8~11 | Self-Play, LSTM 예측, 정보 비대칭 검증 |
| **Phase 4** Sim2Real 검증 | Week 12~16 | 제어 인터페이스, 일반화 테스트, 최종 보고서 |

---

## 완료 항목

- [x] 팀 인터페이스 계약 초안 (`Docs/CONTRACT.md`)
- [x] Python 학습 파이프라인 골격 (`python/scripts/`, `python/config/`)
- [x] C# 에이전트 골격 (`EvaderAgent.cs`, `EvaderReward.cs`)
- [x] **8종 경로 탐색 알고리즘 벤치마크** (500×500 도시 그리드, 120 trials)
- [x] GitHub Pages 사이트 구축

## 진행 예정

- [ ] Unity 3D 도심 맵 완성 (World 담당)
- [ ] PID 파이프라인 구현 (World 담당)
- [ ] Stage 0 학습 실행 및 M1 달성
- [ ] Stage 1~4 순차 진행

---

## 팀 인터페이스 의존성

| 제공자 | 요청 사항 | 우선순위 |
|---|---|---|
| **World** (이강민) | PID 파이프라인, deterministic reset, capture/collision 판정 API | 필수 (M0) |
| **Sensor** (배민우) | Ray 태그 표준, 역방향 LOS raycast (`is_visible_to_pursuer`) | 필수 (M1) |
| **Pursuer** (박재현) | Stage 0용 Scripted 추격 baseline, RL Pursuer 인터페이스 | 필수 (M1) |
