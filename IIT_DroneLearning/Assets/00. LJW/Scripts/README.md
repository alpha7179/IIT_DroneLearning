# 00. LJW / Scripts — Evader 시스템 코드 분석

> 담당자: 이재왕 (work/evader)
> 분석일: 2026-03-26
> 분석자: Claude Code 

---

## 1. 파일 구성

```
Scripts/
├── EvaderAgent.cs          # 회피 드론 RL 에이전트 (핵심)
├── EvaderReward.cs         # 보상 함수 계산기 (분리형)
├── EpisodeLogger.cs        # Eval 전용 에피소드 결과 기록기
├── ScriptedEvader.cs       # 규칙 기반 회피 드론 (Baseline)
└── Editor/
    └── EvaderAgentEditor.cs  # EvaderAgent 커스텀 Inspector
```

---

## 2. 파일별 분석

### 2.1 `EvaderAgent.cs` (265줄)

**역할**: 회피 드론의 ML-Agents RL 에이전트. `DroneAgent`를 상속받아 Evader 전용 로직을 추가/오버라이드.

#### 클래스 계층
```
MonoBehaviour
  └── Agent (ML-Agents)
        └── DroneAgent
              └── EvaderAgent  ← 이 파일
```

#### Inspector 설정 그룹

| 헤더 | 필드 | 기본값 | 설명 |
|---|---|---|---|
| Episode Settings | `_maxEpisodeSeconds` | 25f | 타임아웃 상한 (초) |
| Episode Settings | `_catchDistance` | 1.5f | 포획 판정 거리 (m) |
| Episode Settings | `_goalDistance` | 2.0f | 목표 도달 판정 거리 (m) |
| Stage Control | `_goalOnlyMode` | true | true = Stage0-A (Pursuer 없음) |
| Stage Control | `_currentStage` | 0 | 현재 학습 Stage |
| Spawn Randomization | `_spawnRadius` | 10f | 스폰 반경 |
| Spawn Randomization | `_spawnAltitude` | 5f | 스폰 고도 |
| Spawn Randomization | `_goalRandomizeRadius` | 20f | 목표 지점 랜덤화 반경 |
| Observation Normalization | `_maxDistance` | 50f | 거리 정규화 분모 |
| Observation Normalization | `_maxObsSpeed` | 10f | 속도 정규화 분모 |

#### Observation 벡터 구성 (18차원)

```
[0~2]  localVelocity / _maxObsSpeed          (자기 로컬 속도, 3)
[3~5]  localAngularVelocity / _maxObsSpeed   (자기 각속도, 3)
[6]    position.y / _maxDistance             (고도, 1)
[7~9]  toGoal.normalized                     (목표 방향, 3)
[10]   toGoal.magnitude / _maxDistance       (목표 거리, 1)
[11~13] toPursuer / _maxDistance             (추격자 상대 위치 or 마지막 위치, 3)
[14~16] relativeVelocity / _maxObsSpeed      (추격자 상대 속도, 3)
[17]   isPursuerVisible ? 1f : 0f            (LOS 가시성 플래그, 1)
```

**Stage별 추격자 정보 전략**:
- `goalOnlyMode=true` → `[11~17]` 전부 0 (추격자 개념 없음)
- `goalOnlyMode=false && stage==0` → TargetTransform 직접 참조 (정확한 위치/속도)
- `stage >= 1` → `_lastKnownPursuerPos` 마지막 목격 위치 사용 (메모리 기반)

#### 종료 조건 우선순위
1. 목표 도달 → `+1.0f` 보상, `TermType.Goal`
2. 포획 (`_goalOnlyMode=false`) → `-1.0f` 보상, `TermType.Captured`
3. 타임아웃 → 보상 없음, `TermType.Timeout`
4. 충돌/추락 (외부 호출) → `-1.0f` 보상, `TermType.Crash`

#### 외부 API

```csharp
void SetCrash()
// World/Physics 팀에서 충돌 감지 시 호출 → 에피소드 즉시 종료

void SetPursuerVisibility(bool isVisible, Vector3 pursuerWorldPos)
// Sensor 팀(배민우)에서 매 프레임 LOS 결과 갱신 시 호출
```

#### 설계 특이사항
- `_rb`를 직접 선언하지 않고 `DroneAgent.ResetPhysicsState()`에 물리 초기화 위임
- 속도 취득을 `_dronePhysics.GetVelocity()` API 경유 (Rigidbody 직접 접근 없음)
- GoalTransform/TargetTransform은 DroneAgent 베이스 필드 재사용

---

### 2.2 `EvaderReward.cs` (94줄)

**역할**: 보상 함수를 EvaderAgent에서 분리한 전용 컴포넌트. Inspector 실시간 튜닝 목적.

#### 보상 공식

```
R_step = w_goal × ΔGoalDist    (potential-based shaping)
       + w_survival             (생존 보너스, 기본=0)
       + w_occlusion            (LOS 차단 순간 보너스, 기본=0)
       + w_time                 (시간 페널티, 기본=-0.001)
```

#### Inspector 가중치 기본값

| 필드 | 기본값 | 비고 |
|---|---|---|
| `_goalShapingCoeff` | 0.1f | potential-based shaping 계수 |
| `_survivalRewardPerStep` | 0.0f | Stage0에서는 0 권장 |
| `_occlusionBonus` | 0.0f | Stage1+에서 활성화 |
| `_timePenaltyPerStep` | -0.001f | 매 스텝 음수 패널티 |

#### 설계 포인트
- **Potential-based shaping**: 이전 거리와 현재 거리의 차이(`ΔGoalDist`)로 shaping → 목표 근접 시 양수, 이탈 시 음수
- **Occlusion 보너스**: `_prevPursuerVisible && !isPursuerVisible` — LOS가 **새로 차단된 순간**에만 보너스 지급 (지속 보너스가 아님). 이는 "숨어있기만 하는" 정책 학습 방지 의도
- Episode-end 보상(`±1.0f`)은 EvaderAgent에서 직접 `AddReward` 처리 (이 클래스 범위 밖)

---

### 2.3 `EpisodeLogger.cs` (99줄)

**역할**: Inference Only 모드에서 N 에피소드 실행 후 결과 통계를 CSV로 저장하고 Play 자동 중지.

#### 종료 유형 enum

```csharp
public enum TermType { Goal, Timeout, Crash, Captured }
```

#### 출력 CSV 형식

```csv
episode_id,termination,duration_steps
1,goal,312
2,timeout,500
3,captured,178
...
```

#### 요약 통계 (Console 출력)
- `goal_reach_rate` (목표: ≥50%)
- `timeout_rate`
- `crash_rate`
- `capture_rate`

#### 사용 방법
1. Behavior Type = **Inference Only**, Model = `.onnx` 파일 지정
2. EvaderAgent와 **동일한 GameObject**에 이 컴포넌트 추가
3. Inspector에서 `Max Episodes`, `Output Path` 설정
4. Play → 완료 시 자동 중지 + CSV 저장

#### 설계 포인트
- `_outputPath`가 비어 있으면 `Application.dataPath/../eval_result.csv` (프로젝트 루트) 사용
- `#if UNITY_EDITOR` 가드로 Editor 전용 `isPlaying = false` 처리

---

### 2.4 `ScriptedEvader.cs` (151줄)

**역할**: RL 없이 규칙 기반으로 동작하는 회피 드론. Pursuer 팀의 학습 상대 또는 baseline 비교용.

#### 전략 구성 (우선순위순)

```
1. Goal-seeking    : 목표 방향 단위벡터
2. Obstacle avoid  : 8방향 SphereCast 반발 벡터 합산
3. Pursuer escape  : Pursuer가 _escapeDistance 이내면 반대 방향 가중치 추가
→ 세 벡터 합산 후 normalized → roll/pitch/yaw/throttle로 변환
```

#### 8방향 회피 레이

```csharp
{ Forward, Back, Left, Right,
  Forward+Right, Forward+Left,
  Back+Right, Back+Left }  // 수평면 기준
```

회피 가중치 = `1 - (hit.distance / _avoidDistance)` → 가까울수록 강하게 반발

#### 핵심 파라미터

| 필드 | 기본값 | 설명 |
|---|---|---|
| `_goalSpeed` | 0.6f | 목표 방향 추력 |
| `_avoidRadius` | 1.5f | SphereCast 반경 |
| `_avoidDistance` | 6.0f | 장애물 감지 거리 |
| `_avoidStrength` | 1.2f | 회피 가중치 승수 |
| `_escapeDistance` | 10f | Pursuer 도망 활성 거리 |
| `_escapeStrength` | 0.8f | 도망 가중치 승수 |
| `_targetAltitude` | 5.0f | 고도 유지 목표 (m) |

#### 공개 API

```csharp
void ResetAgent(Vector3 spawnPos)
// 에피소드 리셋 시 외부(PursuerAgent 등)에서 호출
```

#### 고도 제어
`throttle = Clamp(altError / 3f, -1, 1)` — P 제어 (비례 게인 1/3)

#### Gizmos (Editor 전용)
- 청색 와이어 구: 회피 반경 시각화
- 녹색 선: 목표 방향
- 적색 와이어 구: Pursuer 도망 거리 시각화

---

### 2.5 `Editor/EvaderAgentEditor.cs` (108줄)

**역할**: EvaderAgent 전용 커스텀 Inspector. Inspector 가독성 및 안전성 개선.

#### 섹션 구성

| 섹션 | 배경색 | 설명 |
|---|---|---|
| DroneAgent — 수정 가능 | 어두운 파란색 | GoalTransform, TargetTransform, Manual 입력 |
| DroneAgent — 읽기 전용 | 어두운 빨간색 | EvaderAgent가 오버라이드하여 미사용인 DroneAgent 필드 |
| EvaderAgent | 어두운 초록색 | Episode, Stage, Spawn, Obs 정규화 설정 |

#### 읽기 전용 섹션의 의도
DroneAgent의 일부 필드(SpawnRange, SpawnHeight 등)는 EvaderAgent에서 자체 로직으로 **오버라이드되어 실제로 사용되지 않음**. 이를 명시적으로 회색 비활성 상태로 표시해 혼동 방지.

---

## 3. 전체 아키텍처 흐름

```
┌─────────────────────────────────────────────────────────┐
│                    ML-Agents 프레임워크                    │
│  CollectObservations() → OnActionReceived() 매 FixedStep │
└──────────────────────┬──────────────────────────────────┘
                       │
              ┌────────▼────────┐
              │   EvaderAgent    │  (DroneAgent 상속)
              │                 │
              │  ① 관측 수집     │◄── SetPursuerVisibility() ← Sensor팀(BMW)
              │  ② 행동 실행     │──► DronePhysics.SetCommand()
              │  ③ 보상 요청     │──► EvaderReward.ComputeStepReward()
              │  ④ 종료 판정     │──► EpisodeLogger.LogEpisode()
              │  ⑤ SetCrash()   │◄── Physics팀(LGM)
              └─────────────────┘
```

---

## 4. 팀 간 의존성 (인터페이스 계약)

### EvaderAgent가 받는 것

| 제공자 | 메서드 | 내용 |
|---|---|---|
| Sensor팀 (배민우) | `SetPursuerVisibility(bool, Vector3)` | 매 프레임 LOS 결과 |
| Physics팀 (이강민) | `SetCrash()` | 충돌/추락 감지 이벤트 |
| DroneAgent (공용) | `_dronePhysics`, `_sensorSystem` | Awake에서 자동 설정 |
| DroneAgent (공용) | `ResetPhysicsState()` | 물리 초기화 위임 |

### EvaderAgent가 사용하는 DroneAgent 필드 (Inspector)

| 필드 | 용도 |
|---|---|
| `GoalTransform` | 목표 지점 Transform |
| `TargetTransform` | 추격자 드론 Transform |
| `ManualThrottle/Attitude/YawRate` | Heuristic 수동 조종 |

---

## 5. Stage별 설정 가이드

### Stage0-A (순수 목표 도달)
```
_goalOnlyMode  = true
_currentStage  = 0
_occlusionBonus = 0 (EvaderReward)
```

### Stage0-B (추격자 직접 관측)
```
_goalOnlyMode  = false
_currentStage  = 0
TargetTransform = PursuerAgent GameObject
```

### Stage1+ (LOS 메모리 기반)
```
_goalOnlyMode  = false
_currentStage  = 1
_occlusionBonus > 0 (EvaderReward에서 활성화)
Sensor팀 SetPursuerVisibility 연동 필수
```

---

## 6. 코드 리뷰 의견

### 잘 된 점

1. **관심사 분리**: EvaderReward를 별도 컴포넌트로 분리해 보상 튜닝이 코드 수정 없이 Inspector에서 가능
2. **Physics 위임**: `_rb` 직접 접근 없이 `DroneAgent.ResetPhysicsState()` 및 `_dronePhysics` API 경유 → 물리 계층 캡슐화 준수
3. **LOS 메모리 설계**: `_lastKnownPursuerPos` + `_timeSincePursuerDetected`로 Stage1+ 은폐 메모리 구조 준비됨
4. **Occlusion 보너스 타이밍**: LOS 차단 순간에만 보너스 (`justHidden`) → "숨어만 있기" 전략 회피 의도 명확
5. **커스텀 Inspector**: DroneAgent 필드 중 오버라이드된 것을 읽기 전용으로 구분 → Inspector 안전성
6. **OnValidate**: 모든 파라미터에 최솟값 검증 (`Mathf.Max`) → 잘못된 설정 방지

### 개선 가능 포인트

1. **`_timeSincePursuerDetected` 미활용**: 내부 상태로 추적되고 있지만 Observation이나 로직에서 사용되지 않음. Stage1+에서 메모리 신뢰도 감쇠(decay) 관측으로 활용 가능
2. **`SetPursuerVisibility` 호출 주체 미결정**: 주석에 "Sensor 팀에서 호출"이라고 명시되어 있으나, 실제 호출 코드가 아직 구현되지 않은 것으로 보임. Sensor 팀(BMW)과 연동 확인 필요
3. **ScriptedEvader 레이어 마스크 미설정**: `Physics.SphereCast` 호출 시 레이어 마스크 인자가 없어 모든 콜라이더에 반응함. Building/Ground 레이어만 감지하도록 제한 권장
4. **EpisodeLogger 목표 수치**: `goal_reach_rate >= 50%` 목표가 하드코딩됨. Inspector에서 설정 가능하게 하거나 YAML/설정파일로 분리 고려
5. **관측 정규화 일관성**: `localAngVel`은 `_maxObsSpeed`로 나누고 있는데, 각속도의 단위(rad/s)와 선속도(m/s) 단위가 다름. 별도 `_maxAngSpeed` 필드 분리 고려

---

## 7. 관련 파일 참조

| 파일 | 위치 | 관계 |
|---|---|---|
| `DroneAgent.cs` | `Assets/02. Scripts/` | 베이스 클래스 (공용) |
| `DronePhysics.cs` | `Assets/02. Scripts/` | 물리 엔진 (이강민) |
| `DroneSensorSystem.cs` | `Assets/00. BMW/DroneSenser/` | LOS/Ray 센서 (배민우) |
| `evader_s0_flat_v3.onnx` | `Assets/00. LJW/` | Stage0 학습 모델 |
| CLAUDE.md | 저장소 루트 | 팀 작업 기준서 |
