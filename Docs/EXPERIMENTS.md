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
| 2026-03-20 | `166e7da` | `evader_s0_flat_template` | `evader_s0_flat_seed42` | 42 | - | ~35% | - | - | 평지 Stage0 초도 학습 (300k). timeout 보상 제거, goalShaping 0.02→0.1. 진동하며 최대 0.375 도달. |
| 2026-03-20 | `166e7da` | `evader_s0_flat_template` | `evader_s0_flat_seed42_v2` | 42 | - | **~60%** | - | - | warm-start 추가 300k. 115k에서 0.5 돌파, 최종 0.637(peak 0.686). M1 goal_reach_rate ≥50% **달성**. |
| 2026-03-20 | `d12dfa2` | `evader_s0_flat_template` | `evader_s0_flat_seed42_v3` | 42 | - | **~80~90%** | - | - | warm-start v2→v3 300k. 50k부터 0.9 돌파, peak 0.982(230k), 최종 0.768. |
| 2026-03-21 | `1079846` | `evader_s0_flat_template` | `evader_s0_flat_seed42_v3` | 42 | **100%** | **68%** | **0%** | **0%** | eval 50 eps (EpisodeLogger). survival/goal/M1 전부 달성. timeout 32% → Stage1 전환 준비. |
| 2026-03-27 | `bbcacb5` | `evader_s0_flat_template` | `evader_s0_flat_44d_v6_seed42` | 42 | **100%** | **68%** | **0%** | **0%** | warm-start v5→v6 500k. velAlign/yawAlign dead zone 0.5→0.1m 수정, goalzone 호버링 해소. peak 6.676(390k), final 6.259. 50eps eval. |

---

## Stage1 — 장애물 도입, 힌트 제거

목표: `survival_rate ≥ 50%`, `crash_rate ≤ 15%`

| 날짜 | commit | config | run-id | seed | mean_reward | 코멘트 |
|------|--------|--------|--------|------|-------------|--------|
| 2026-04-06 | `26b746b` | `evader_s1_obstacle_template` | `evader_s1_obstacle_44d_v1_seed42` | 42 | peak +2.669 (350k), final +2.162 (500k) | goalProximityBonus 존재. 290k 최초 양수. 350k peak 후 진동. 호버링 착취 확인 → v2 warm-start로 사용 |
| 2026-04-06 | `e43867d` | `evader_s1_obstacle_template` | `evader_s1_obstacle_44d_v2_seed42` | 42 | - | obstacle-aware vel reward + 체크포인트 시스템 + goalProximityBonus 제거 + timePenalty -0.003 적용. artifact maxStep=470550 |
| 2026-04-07 | `??` | `evader_s1_obstacle_template` | `evader_s1_obstacle_44d_v3_seed42` | 42 | - | v2 warm-start 연장 실험. artifact maxStep=783812 |
| 2026-04-07 | `??` | `evader_s1_obstacle_template` | `evader_s1_obstacle_44d_v4_seed42` | 42 | - | v3 후속 실험(중단/재시작 포함). artifact maxStep=366352 |
| 2026-04-07 | `??` | `evader_s1_obstacle_template` | `evader_s1_obstacle_44d_v5_seed42` | 42 | peak +9.542 (960k), final +5.991 (1M) | v2 warm-start 끝나고 v3→v5로 계속 학습. 900k~1M에서 mean reward 2~10 범위 진동. **이슈**: code 변경 후 scene 재저장 누락으로 Inspector에서 MaxEpisodeSeconds 25(old)로 표시. 실제로는 25로 강제되어 장애물 나비게이션 시간 부족 → timeout 과다발생. 코드는 40으로 변경되었으나 Unity scene 메타데이터 미동기화 문제 → URGENT_FIX_MaxEpisodeSeconds.md 참조. |
| 2026-04-07 | `??` | `evader_s1_obstacle_template` | `evader_s1_obstacle_44d_v6_seed42` | 42 | - | v5 이후 안정화 탐색 라운드. artifact maxStep=303555 |
| 2026-04-07 | `??` | `evader_s1_obstacle_template` | `evader_s1_obstacle_44d_v7_seed42` | 42 | - | Stage1 하드닝/보상 프로파일 강제 반영 라운드. artifact maxStep=884877 |
| 2026-04-08 | `??` | `evader_s1_obstacle_template` | `evader_s1_obstacle_44d_v8exp_seed42` | 42 | - | exploration profile 라운드. artifact maxStep=599895 (최근 resume 명령은 exit=1) |
| 2026-04-08 | `??` | `evader_s1_obstacle_template` | `evader_s1_obstacle_44d_v9guard_seed42` | 42 | - | guard profile 라운드. artifact maxStep=599995 (최근 resume 명령은 exit=1) |
| 2026-04-08 | `??` | `evader_s1_obstacle_template` | `evader_s1_obstacle_44d_v10smooth_seed42` | 42 | - | smoothing profile 라운드. artifact maxStep=299936 |
| 2026-04-08 | `??` | `evader_s1_obstacle_20260408_v11_dewall.yaml` | `evader_s1_obstacle_44d_v11dewall_seed42` | 42 | - | dewall 설정 라운드. artifact maxStep=249991, run_logs metadata 누락 이슈 확인 |
| 2026-04-08 | `??` | `evader_s1_obstacle_20260409_v12_wallfix.yaml` | `evader_s1_obstacle_44d_v12wallfix_seed42` | 42 | - | wallfix 설정 라운드. artifact maxStep=99977, notebook output cell 유실 및 외벽 충돌 다수 관찰 |
| 2026-04-08 | `working-tree` | `evader_s1_obstacle_20260408_v13_wallfocus.yaml` | `evader_s1_obstacle_44d_v13wallfocus_seed42` | 42 | 100k gate final +13.596 (std 20.364) | v12 checkpoint warm-start. target 100k 도달 후 auto-stop 수행, exit=1은 terminate 기반 정상 종료. post-check snapshot_step=99973, artifacts: 24872/49924/74987/99973 |
| 2026-04-08 | `working-tree` | `evader_s1_obstacle_20260408_v13c_rootfix.yaml` | `evader_s1_obstacle_44d_v13crootfix_seed42` | 42 | resume final +11.684 @250k (peak +14.592 @245k, dip +3.163 @215k) | v13c 추가학습(resume) 99943->250000 완료. auto-stop 정상 동작, 진행 안정적. checkpoint exports: 124962/149950/174890/199908/224886 |

참고: `artifact maxStep`은 `python/results/<run-id>/Drone_Evader/Drone_Evader-*.onnx|*.pt` 파일명 기준으로 기록.

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

---

## Stage1-A 분석 노트 (2026-04-06)

### 1. 문제: goalProximityBonus 호버링 착취

**현상**: v1 학습에서 500k 스텝 내내 드론이 goal zone 근처에서 가만히 호버링.
"바로 앞이 goal인데 들어가려 하지 않는다."

**원인 분석**:
```
goal 2m 앞 호버링 시 step당 net reward:
  goalProximityBonus:  +0.005 × (1 - 2/20) = +0.0045
  timePenalty:                               -0.001
  net:                                       +0.0035  ← 양수!

에피소드 350 step 기준:
  goal 2m 호버링 250 step = +0.0035 × 250 = +0.875
  goal 진입 terminal      =                  +1.0   (겨우 조금 나을 뿐)
```
드론 입장에서 "안전한 호버링 ≈ goal 진입"이 되어 리스크 없이 호버링 선택.

**해결**: goalProximityBonus 완전 제거. timePenalty -0.001 → -0.003으로 강화.
```
수정 후 시나리오 (350 step 에피소드):
  goal 도달 (100 step):  shaping + 1.0 - 0.003×100  = +0.7 이상
  어디서든 호버링 타임아웃: 0       - 0.003×350  = -1.05
  충돌 (100 step):        -1.0    - 0.003×100  = -1.3
```
goal 도달 > 호버링 > 충돌 — 명확한 우선순위 복원.

---

### 2. 문제: 장애물 회피 방향 신호 없음

**현상**: 드론이 장애물을 감지해도 어느 방향으로 회피해야 할지 보상 신호가 없음.
기존 velAlignCoeff는 단순히 goal 방향 속도를 보상 → 장애물 있는 방향으로 돌진.

**해결**: Obstacle-Aware Velocity Reward 구현 (EvaderReward.cs)

```
DroneSensorSystem Middle 레이어 인덱스 9~16 사용:
  9:N / 10:NE / 11:E / 12:SE / 13:S / 14:SW / 15:W / 16:NW (드론 로컬 좌표계)

알고리즘:
  1. Middle 8개 레이에서 가중 척력 벡터 계산 (로컬)
     weight = 1 - d/proximityThreshold  (가까울수록 강함)
     repulsion += -rayDir × weight      (장애물 반대 방향)
  2. 장애물 강도 = clamp01(repulsion.magnitude / 2)
  3. 강도 < 0.05: 단순 goal 방향 속도 보상
     강도 >= 0.05: goal 방향과 척력 방향을 Slerp 블렌드 (강도 비율로)
  4. Dot(agentVel, targetDir) / maxObsSpeed × velAlignCoeff
```

**파라미터**: velAlignCoeff 0.003 → 0.005

---

### 3. 문제: 중간 보상 신호 없음 (sparse reward)

**현상**: goal이 50~100m 떨어진 경우 장애물을 피해 도달하면 +1.0이지만,
도달에 실패하면 중간 진행에 대한 보상이 없어 학습 신호가 너무 희박.

**해결**: 에피소드 체크포인트 시스템 (EvaderAgent.cs)

```
에피소드 시작 시 스폰→Goal 직선 위에 2개 체크포인트 생성:
  위치: Lerp(spawnPos, goalPos, t=1/3), Lerp(..., t=2/3)
  Y축: 스폰 고도 유지 (드론 현재 높이 기준)
  장애물 회피: Physics.OverlapSphere(radius=2m) → Building/Wall 태그 감지
              막혀 있으면 8방위 × 4단계 (3m씩) 후보 탐색

도달 판정: 체크포인트 반경 4m 이내 진입 시 +0.3 (순서대로, 1회만)
```

**Inspector 필드**: `_checkpointCount=2`, `_checkpointRadius=4`, `_checkpointReward=0.3`, `_checkpointClearRadius=2`

---

### 4. 실수: timePenalty와 MaxEpisode 동시 변경으로 발산

**현상**: timePenalty -0.001 → -0.005, MaxEpisodeSeconds 35 → 60 동시 변경 시도.
결과: Mean Reward -14.3 (이전 대비 급격 악화).

**원인**: 60초 에피소드에서 step 수 ≈ 700. timePenalty만 계산해도 -0.005×700 = -3.5.
  거기에 MaxEpisode 증가로 crash 기회도 증가 → 복합 발산.

**교훈**: 변수는 한 번에 하나씩 변경. 특히 timePenalty × 에피소드 길이 상호작용에 주의.

즉시 이전 설정으로 롤백 (timePenalty -0.001, MaxEpisode 35초 유지).

---

### 5. 어려웠던 부분: YAML 한글 인코딩 오류

**현상**: YAML 파일에 한글 주석 작성 시 ML-Agents가 cp949로 읽어 UnicodeDecodeError.

**해결**: YAML 파일 내 모든 주석을 ASCII로만 작성. 영어 또는 특수문자만 사용.

**규칙 확정**: `evader_s1_obstacle_template.yaml` 및 이후 모든 YAML 파일 한글 사용 금지.

---

### 6. v1 학습 실행 기록 (500k 스텝)

| 스텝 | Mean Reward | Std | 비고 |
|------|-------------|-----|------|
| 10k  | -3.2 ~ -4.0 | 높음 | crash 과다, 탐색 초기 |
| 100k | -2.8 ~ -3.5 | 높음 | 개선 없음, 호버링 시작 |
| 200k | -1.5 ~ -2.0 | 감소 | 일부 goal 도달 시작 |
| 290k | **+0.203**  | -   | 최초 양수 mean reward |
| 340k | +2.669      | 0.553 | **peak** |
| 350k | +2.553      | -   | peak 근처 유지 |
| 450k | +2.162      | -   | 진동 시작 (goalProximityBonus 착취 고착) |
| 500k | +2.162      | -   | 수렴 실패, v2 warm-start로 활용 |

**결론**: goalProximityBonus가 290k 이후 학습을 견인했으나 350k 이후 호버링 착취 고착.
최적 체크포인트 450k를 v2 warm-start로 사용.

---

### 7. v2 설계 요약 (2026-04-06 구현 완료)

| 항목 | v1 | v2 |
|------|----|----|
| goalProximityBonus | O (착취 원인) | **제거** |
| timePenalty | -0.001 | **-0.003** |
| velAlign | goal 방향만 | **obstacle-aware (척력 블렌드)** |
| 체크포인트 | 없음 | **+0.3 × 2개 / 에피소드** |
| warm-start | v5 (Stage0) | **v1 450k** |

**예상**: 50k 이내 Mean Reward > 0 (v1은 290k에서 최초 달성).

---

### 8. 개선 검토 사항 (v3 이후)

- **Goal Y축 불일치 확인 필요**: Goal 스폰 높이(`cityMaxHeight × 0.5`)와 드론 비행 고도(5~25m) 불일치 시
  trigger가 발동 안 될 수 있음. Heuristic 모드로 수동 확인 권장.
- **체크포인트 반경 조정**: 4m가 너무 좁거나 넓으면 실제 학습 로그 보고 조정.
- **Beta 스케줄**: 초반 탐색(0.02) → 수렴 후(0.005) 스텝 기반 자동 감소 고려.
- **거리 커리큘럼**: v2 수렴 후 EpisodeSpawnCoordinator 최대 거리를 점진 확대 (15m → 30m → 100m).
- **RGB 카메라 오버헤드**: CNN 처리로 학습 속도가 Stage0 대비 느림. 체크포인트 수 / batch_size 튜닝 여지 있음.
