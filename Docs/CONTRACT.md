# CONTRACT.md — 팀 인터페이스 계약서 초안

> **상태**: 초안 (Week 1 팀 회의에서 수치 확정 필요)
> **작성**: 이재왕 (work/evader)
> **목적**: Evader/Pursuer/World/Sensor 간 인터페이스를 사전 정의하여 병렬 개발 중 충돌을 방지한다.
> 변경 시 모든 담당자가 합의하고 이 파일을 PR로 업데이트한다.

---

## 1. 버전 락 (변경 불가)

| 컴포넌트 | 버전 | 확인자 |
|---|---|---|
| Unity | 6000.0.69f1 | 전체 팀 |
| ML-Agents | 4.0.x | 전체 팀 |
| Python | 3.10 | 전체 팀 |
| PyTorch | 2.x | 전체 팀 |

---

## 2. 행동 공간 (Action Space) — 공통

두 에이전트(Pursuer, Evader) 모두 동일한 행동 공간 구조를 사용한다.

| 인덱스 | 이름 | 범위 | 설명 |
|---|---|---|---|
| 0 | `thrust_cmd` | [0, 1] | 추력 명령 (0=없음, 1=최대) |
| 1 | `roll_rate_cmd` | [-1, 1] | 롤 각속도 명령 |
| 2 | `pitch_rate_cmd` | [-1, 1] | 피치 각속도 명령 |
| 3 | `yaw_rate_cmd` | [-1, 1] | 요 각속도 명령 |

- **타입**: Continuous 4D
- **적용 방식**: PID 컨트롤러 → Rigidbody Force/Torque (직접 모터 제어 금지)
- **적용 주기**: FixedUpdate, **50Hz (dt=0.02s)** ← 팀 합의 필요
- **제공자**: World (이강민) — `DroneController.SetPIDSetpoints()`

---

## 3. 관측 공간 (Observation Space)

### 3.1 공통 관측 (두 에이전트 공유)

| 항목 | 차원 | 정규화 기준 | 비고 |
|---|---|---|---|
| 로컬 속도 (vx, vy, vz) | 3 | max_speed=10 m/s | |
| 로컬 각속도 (wx, wy, wz) | 3 | max_speed=10 | |
| 고도 | 1 | max_distance=50 m | |

### 3.2 Evader 전용

| 항목 | 차원 | Stage | 정규화 |
|---|---|---|---|
| 목표 상대 방향 (normalized) | 3 | 0~ | - |
| 목표 거리 | 1 | 0~ | /max_distance |
| [Stage0] Pursuer 상대 위치 | 3 | 0만 | /max_distance |
| [Stage0] Pursuer 상대 속도 | 3 | 0만 | /max_speed |
| Pursuer 가시 여부 | 1 | 0~ | 0 or 1 |
| [Stage1+] 마지막 Pursuer 위치 | 3 | 1~ | /max_distance |

### 3.3 Pursuer 전용

> 상세 내용은 박재현의 AGENT.md Section 3.2 참조.

### 3.4 Ray 관측

- **담당**: Sensor (배민우) — `RayPerceptionSensorComponent` 컴포넌트 사용
- **방향**: 전방/측면/하방 **8방향 이상** (팀 합의 필요)
- **태그 표준**:

| 태그 | 의미 |
|---|---|
| `building` | 건물/벽 |
| `ground` | 지면 |
| `drone` | 상대 드론 (Pursuer 또는 Evader) |
| `nofly` | 비행 금지 구역 |
| `goal` | 목표 지점 |

---

## 4. 종료 조건 (Termination)

두 에이전트 에피소드를 동시에 종료시키는 공통 조건:

| 조건 | 수치 (초안) | 확정 필요 | 제공자 |
|---|---|---|---|
| **Catch**: Pursuer-Evader 거리 | `d < 2.0m`, `k=3 step 연속` | ✅ 합의 필요 | World |
| **Crash**: 고도 | `altitude < 0.5m` | ✅ 합의 필요 | World |
| **Crash**: 기울기 | `tilt > 60°` 지속 | ✅ 합의 필요 | World |
| **Crash**: 충돌 | `OnCollisionEnter` 트리거 | | World |
| **Timeout**: 에피소드 시간 | `T_max = 20s` | ✅ 합의 필요 | World |
| **Goal Reached**: Evader 도달 | `d < 2.0m` | ✅ 합의 필요 | World |
| **Out-of-bounds**: 경계 이탈 | 맵 크기에 따라 결정 | ✅ 합의 필요 | World |

### API 규약

```csharp
// World(이강민)가 제공해야 하는 이벤트/콜백
public interface IDroneEpisodeEvents
{
    void OnCatch();           // Pursuer가 Evader를 잡았을 때
    void OnCrash(string who); // "pursuer" 또는 "evader"
    void OnGoalReached();     // Evader가 Goal Zone 도달
    void OnTimeout();
    void OnOutOfBounds(string who);
}
```

---

## 5. Episode Reset 동기화 순서

에피소드 종료 시 다음 순서로 동기화한다:

```
1. World.ResetEnvironment()    ← 장애물, 물리 상태, Goal Zone 위치 재설정
2. Pursuer.OnEpisodeBegin()    ← Pursuer 위치/속도 초기화
3. Evader.OnEpisodeBegin()     ← Evader 위치/속도 초기화
4. Sensor.ResetBuffers()       ← Ray/Camera 버퍼 초기화
```

- Spawn 위치: **deterministic reset** (고정 시드 eval용) / **random spawn** (학습용) 지원 필수
- 두 드론의 초기 거리: `d_init > 15m` 권장 (즉시 catch 방지)

---

## 6. Reward 신호 흐름

```
Unity C# (EvaderAgent/PursuerAgent)
    └─ AddReward(float)            ← ML-Agents 내부 reward 누적
    └─ EndEpisode()                ← 에피소드 종료 및 Python 전송

Python (ML-Agents 학습 루프)
    └─ reward signal 수신
    └─ YAML config의 reward_signals.extrinsic.strength 배율 적용
    └─ PPO advantage 계산
```

---

## 7. Physics / Timestep 설정

| 항목 | 값 (초안) | 위치 |
|---|---|---|
| Fixed Timestep | **0.02s (50Hz)** | Unity > Project Settings > Time |
| Physics Solver | Default | Unity > Physics Settings |
| Decision Period | **5 steps** (10Hz) | ML-Agents Agent 컴포넌트 설정 |
| Max Linear Velocity | 10 m/s | DroneController 클램프 |
| Max Angular Velocity | 180 deg/s | DroneController 클램프 |

> Decision Period가 클수록 학습 안정성은 높아지지만 반응이 느려진다. 5~10 step을 초안으로 시작.

---

## 8. Colab 학습 환경

| 항목 | 내용 |
|---|---|
| 학습 플랫폼 | Google Colab Pro+ |
| Unity 빌드 | Headless Linux 빌드 (`-batchmode -nographics`) |
| 체크포인트 저장 | `/content/drive/MyDrive/IIT_DroneLearning/checkpoints/` |
| 로그 저장 | `/content/drive/MyDrive/IIT_DroneLearning/runs/` |
| 병렬 환경 수 | `--num-envs 4` 기본값 |

---

## 9. 합의 이력

| 날짜 | 항목 | 변경 내용 | 합의자 |
|---|---|---|---|
| 2026-03-03 | 전체 초안 | 이재왕 작성 | 이재왕 (검토 필요) |

> Week 1 팀 회의에서 수치 확정 후 이 표에 기록한다.
