# Agents.md — Evader(Escaper) RL Owner Plan (1 Semester, Unity6 + ML-Agents)

> 이 문서는 Claude Code(claude-sonnet-4-6)가 레포에서 작업할 때 "회피 드론(Evader/Escaper) 담당자(이재왕)"의 기준 문서로 항상 참고한다.
> 목표는 **시행착오를 줄이며** Stage별로 "학습이 붙는" 파이프라인을 구축하고, 최종적으로 Sim2Real 지향의 물리 기반 회피 정책을 확보하는 것이다.

---

## 0) 내 역할(Owner) 정의

### ✅ Scope (내가 책임지는 것)
- Evader/Escaper 에이전트 로직: 관측/행동/보상/종료조건/학습 안정화
- 은폐 대응(회피 관점): Pursuer LOS 차단 전략, 빌딩 활용 은폐/재은폐(단계적으로)
- 학습 파이프라인: `python/config`, `python/scripts`, `python/utils`, `python/results` 구조 정리
- Colab Pro+ 학습 운영(러너 역할) + 실험 기록/재현성 유지
- 최소 평가(Eval) 루틴: 고정 시드에서 survival_rate / time_to_capture / goal_reach_rate 산출

### ❌ Out of scope (내가 함부로 건드리지 말 것)
- 도시 맵(월드) 모델링/레벨 디자인(담당: World)
- 센서 렌더링/카메라 파이프라인/레이 태그 설계(담당: Sensor)
- Pursuer(추격자) 전략/보상 설계(담당: Pursuer)
- 대규모 리팩터/폴더 구조 변경(합의 없이는 금지)

> 단, Evader 학습을 위해 "인터페이스 계약(Contract)"이 깨지면 최소 수정 PR을 제안하고, 필요한 변경은 담당자와 합의 후 진행한다.

---

## 1) Claude Code 작업 규칙 (항상 준수)

1) 변경은 작게: **1 PR = 1 capability**
2) 작업 전/후 반드시 다음 3가지를 남긴다:
   - 무엇을 바꿨는지(Why/What)
   - 어떻게 실행/학습/평가하는지(How)
   - 기대 지표(What metric should move)
3) 파일 범위 제한:
   - 기본: `Assets/**/Evader*`, `python/config/**evader*`, `python/scripts/**`, `docs/**`
4) 실행 커맨드가 없으면 먼저 문서/스크립트를 만든다("추측 실행 금지")
5) PR마다 최소 1개 eval 결과를 남긴다(짧은 eval이라도 OK)

## Git / Branch Rule (Hard Constraint)
- 작업 브랜치: `work/evader` **하나만 사용**한다.
- Claude Code는 작업 시작 시 항상 현재 브랜치가 `work/evader`인지 확인한다.
- 다른 브랜치 생성/체크아웃/리베이스는 사용자가 명시적으로 요청하지 않는 한 수행하지 않는다.
- main과 동기화가 필요하면, "필요하다"는 안내만 하고 사용자가 실행하도록 커맨드를 제안한다.

---

## 2) 핵심 설계 원칙 (Sim2Real + 물리 기반)

### 2.1 Control Principle (Sim2Real + Physics)
- 본 프로젝트의 제어는 **직접 모터/액추에이터 제어가 아니라 PID 기반 제어**를 사용한다.
- RL policy는 low-level actuator가 아니라 **상위 명령(setpoint)** 을 출력한다.

**Default control stack**
RL policy → (thrust_cmd + body rate cmd) → PID (rate/attitude) → Rigidbody Force/Torque

**Notes**
- PID는 Unity FixedUpdate 주기에서 실행된다.
- 액션 스케일/클램프/안전 제한(최대 기울기, 최대 상승률 등)은 환경 계약(Contract)으로 고정한다.

### 2.2 Stage 기반 개발 (한 번에 하나만 올린다)
- Ray + state만으로 Stage0를 먼저 "붙이고"
- 장애물 활용/가림/비전 탐지/기억/랜덤화는 순차적으로 하나씩 추가한다.
- Stage0가 안 붙으면 그 이후는 무의미하므로 Stage0 DoD를 최우선으로 달성한다.

---

## 3) 인터페이스 계약(Contract) — World/Sensor/Pursuer와 합의 필요

> 아래 값은 "초안"이며, Week 1에 팀 합의로 확정한다.

### 3.1 Action Space (Evader)
- Continuous 4D (PID setpoints)
  - a0: thrust_cmd ∈ [0, 1]
  - a1: roll_rate_cmd ∈ [-1, 1]
  - a2: pitch_rate_cmd ∈ [-1, 1]
  - a3: yaw_rate_cmd ∈ [-1, 1]
- Applied via PID → Rigidbody Force/Torque (no direct actuator control).
- 적용 주기: FixedUpdate 기준 (예: 50Hz, dt=0.02)

### 3.2 Observation Space (Evader)
**Always (Stage0~)**
- 자기 상태: position/velocity(로컬), angular velocity, orientation(Up vector or quaternion), altitude, (선택) motor rpm 추정/배터리
- Ray: 전방/측면/하방 거리 + 태그(building/ground/drone/nofly/goal)
- 목표 지점: 상대좌표(goal_rel_pos) + 도달 여부

**Stage0 전용(학습 성립 확인용 힌트)**
- 상대(Pursuer) 상대좌표/상대속도(ground-truth) 허용 → Stage1부터 제거
- is_detected (Pursuer에게 LOS가 열려있는지) 허용 → Stage1부터 노이즈 추가

**은폐 대응용(추후)**
- is_visible_to_pursuer (LOS raycast 결과, 역방향)
- last_known_pursuer_pos, time_since_detected (구조적 기억)
- building_cover_score (주변 빌딩 은폐도 추정, 선택)

**Vision (Stage2~)**
- 84x84 RGB (또는 96x96), 필요시 프레임 스택(2~4)

### 3.3 Termination (공통)
- Captured: Pursuer와 distance < d_catch 를 k-step 연속 만족(스치기 오탐 방지)
- Crash: altitude < h_min, tilt > tilt_max 지속, 충돌
- Timeout: T_max (예: 20~30초) → Evader 생존으로 간주
- Goal Reached: 목표 지점 distance < d_goal 도달 시 성공 종료
- Out-of-bounds / NoFly 진입 시 종료

---

## 4) Evader Reward 설계 (단순→정교)

### Stage0/1 기본 보상(최소안)
- +1.0 : 목표 지점 도달 성공
- -1.0 : 포획(captured) 당함
- step survival bonus: +0.001 (생존할수록 유리)
- goal distance shaping: +k*(d_prev_goal - d_now_goal), k는 매우 작게(0.01~0.05부터)
- timeout 생존: +0.2 (옵션: 도망 성공에 준하는 부분 보상)

### Stage2~ (은폐/LOS 회피)
- LOS 차단 보너스(선택): Pursuer의 LOS가 끊기면 +b (과도한 보상 엔지니어링 주의)
- "빌딩만 끼고 멈추기" 방지: 이동 효율성 고려, 목표 접근 우선순위 유지
- shaping 계수 상한, 속도/각속도 과도 페널티는 매우 작게

---

## 5) 은폐/회피 전략(시행착오 최소 루트)

### 5.1 구조적 기억(권장, 본선)
- LSTM 없이도 다음을 관측에 포함해 은폐 전략을 먼저 달성:
  - is_visible_to_pursuer / last_known_pursuer_pos / time_since_detected
- 정책은 feed-forward 유지 → 디버깅 쉬움, 수렴 안정적

### 5.2 LSTM(실험 분기)
- ML-Agents 문서에서 LSTM은 continuous action에서 잘 안 맞을 수 있으므로,
  - (A) 액션을 이산화한 분기 실험
  - (B) LSTM 튜닝을 별도 브랜치에서만
- 본선 Stage 목표를 LSTM 의존으로 잡지 않는다.

### 5.3 빌딩 활용 은폐 전략
- 빌딩을 방패로 사용하는 회피는 reward shaping 없이 LOS 차단 관측 정보만으로 유도한다.
- 과도한 reward shaping은 "빌딩만 껴서 제자리 회전"하는 국소 최적해를 유발하므로 주의한다.

---

## 6) 학습 파이프라인(Colab Pro+ 표준)

### 6.1 디렉토리 규칙
- `python/config/`: stage별 config YAML
- `python/scripts/`: train/eval/export 도구
- `python/results/`: run 로그(대용량은 git 제외)
- `python/models/`: onnx/ckpt (git 제외, Drive에 저장)

### 6.2 Run-ID / Config 네이밍
- config: `python/config/evader_s{stage}_{yyyymmdd}_{short}.yaml`
- run-id: `evader_s{stage}_{exp}_{seed}`

### 6.3 Eval 표준(필수)
- 고정 시드(예: 10개)로 20~50 에피소드 eval
- 저장 지표:
  - survival_rate (timeout까지 생존 비율)
  - goal_reach_rate (목표 지점 도달 비율)
  - capture_rate (포획 당한 비율)
  - mean_time_to_capture (포획됐을 때 평균 경과 시간)
  - los_break_rate (LOS 차단 성공 비율, Stage2+)

### 6.4 결과 기록(재현성)
- `docs/EXPERIMENTS.md`에 실험 1줄 기록:
  - date, git_commit, config, run-id, seed, 핵심 지표, 코멘트(한 줄)
- 가능하면 TensorBoard 로그는 Drive에 저장(Colab 런타임 종료 대비)

---

## 7) 16주(한 학기) 로드맵 — Evader 관점

### M0 (Week 1) — "학습 가능한 최소 환경" 계약 완료
- Contract(Action/Obs/Termination) 팀 합의/문서화
- Pursuer scripted baseline(단순 추격) 확보 요청
- Eval 스크립트 골격 생성(고정 시드)

**DoD**
- Stage0 환경에서 에피소드가 안정적으로 reset/terminate
- 학습 커맨드 1개가 문서에 존재

### M1 (Week 2~3) — Stage0 수렴(장애물 거의 없음, Ray+state)
- Pursuer 상대좌표 힌트 허용, 목표 지점 힌트 허용
- reward 최소안으로 goal_reach_rate / survival_rate 상승 확인

**DoD**
- eval goal_reach_rate ≥ 50% (고정 시드, 단순 추격자 상대)
- survival_rate(timeout) ≥ 70%
- capture_rate ≤ 30%

### M2 (Week 4~6) — Stage1(장애물 도입 + 힌트 제거)
- 도심 블록 장애물 추가
- Pursuer 상대 힌트를 제거하고 Ray 기반으로 추격자 감지/회피
- "목표 도달 + 충돌 회피" 동시 성립

**DoD**
- eval survival_rate ≥ 50% (장애물 환경, Ray 기반 추격자 상대)
- crash_rate ≤ 15%

### M3 (Week 7~9) — 경쟁 학습(가벼운 Self-play / Opponent Pool)
- Pursuer 고정 정책 ↔ Evader 학습 → 번갈아 업데이트(Iterative)
- Opponent snapshot 저장/재대결

**DoD**
- 상대 정책이 바뀌어도 성능 급락 없이 유지(robustness)
- survival_rate ≥ 60% (RL 추격자 상대)

### M4 (Week 10~12) — Stage2 Vision + LOS 은폐 전략
- 84x84 camera 추가(필요시 xvfb)
- 구조적 기억(is_visible_to_pursuer + Δt) 추가
- los_break_rate 지표 도입, 빌딩 활용 은폐 행동 확인

**DoD**
- LOS 차단 상황에서 생존 시간이 증가
- 성능이 seed 3개 이상에서 일관

### M5 (Week 13~15) — Stage3 Sim2Real 강화(도메인 랜덤화)
- mass/inertia/drag/thrust scale/wind/motor delay 랜덤화
- 센서 노이즈/지연(점진적)

**DoD**
- 랜덤화 강도를 올려도 정책이 붕괴하지 않음(성능 완만 감소)

### Week 16 — 정리/패키징
- 최종 모델(onnx) + eval 리포트 + 재현 가이드
- 논문/발표용 figure(학습곡선, 생존률, 사례 영상)

---

## 8) World/Sensor/Pursuer에 요청해야 하는 "최소 의존성" 체크리스트

### World(물리/환경)
- (필수) rate command → PID → Rigidbody Force/Torque 파이프라인 제공
- (필수) deterministic reset (pos/vel/angvel 초기화)
- (필수) capture/collision 판정 API
- (필수) goal point 생성/리셋 API
- (권장) fixed timestep 고정 안내 + physics solver 설정 문서

### Sensor(레이/비전)
- (필수) Ray 태그 표준(building/ground/drone/nofly/goal)
- (필수) is_visible_to_pursuer 계산용 역방향 LoS 지원(간단 raycast)
- (선택) camera sensor, low-res 설정, headless 대응 가이드

### Pursuer
- Stage0용 scripted baseline(랜덤/단순 추격)
- Stage1 이후 RL pursuer와의 인터페이스(episode sync)

---

## 9) "학습이 안 붙을 때" 디버깅 체크리스트(우선순위)

1) episode가 너무 빨리 끝나지 않는가? (capture/crash/timeout 과다)
2) capture 조건이 너무 빡세거나, 반대로 너무 느슨하지 않은가?
3) 액션 스케일(추력/각속도)이 과도하지 않은가? (항상 전복/상승 폭주)
4) 보상 부호/크기 이상(대부분 0, 또는 shaping만 과대)
5) 관측이 NaN/Inf, 혹은 좌표계가 섞이지 않았는가?
6) fixed timestep / frame skip이 일관되지 않은가?
7) eval과 train 환경이 동일한가? (랜덤화 on/off)

---

## 10) 최종 산출물(한 학기 종료 시)

- Evader 정책(ONNX) + 학습 config + eval 리포트
- Stage별 성능표(survival_rate/goal_reach_rate/capture_rate + los_break_rate)
- Colab 재현 노트북/스크립트
- 시연 영상(은폐 회피 케이스 포함)
