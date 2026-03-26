# **개발 환경 설정 가이드 (Setup Guide)**

> 본 문서는 IIT 팀의 Unity + ML-Agents 기반 드론 MARL 시뮬레이션 환경의 초기 구축 과정과 확정된 기술 스택을 정리합니다.
> 팀원 간 환경 통일 및 신규 참여자의 온보딩을 위한 기준 문서입니다.

---

## **목차**

1. [확정 버전 스택](#1-확정-버전-스택)
2. [사전 준비](#2-사전-준비-prerequisites)
3. [Python 가상환경 구축](#3-python-가상환경-구축)
4. [ML-Agents 설치](#4-ml-agents-설치)
5. [Unity 프로젝트 설정](#5-unity-프로젝트-설정)
6. [Python ↔ Unity 연동 검증](#6-python--unity-연동-검증)
7. [프로젝트 구조](#7-프로젝트-구조)
8. [구현 완료 코드 명세](#8-구현-완료-코드-명세)
9. [트러블슈팅](#9-트러블슈팅)

---

## **1. 확정 버전 스택**

팀 전원이 아래 버전을 동일하게 사용해야 합니다. 버전 불일치 시 Python ↔ Unity 통신 오류가 발생합니다.

| 항목 | 버전 | 비고 |
|------|------|------|
| **Unity** | `6000.0.69f1` | Unity Hub를 통해 설치 |
| **ML-Agents (Unity Package)** | `4.0.0` (release_23) | GitHub에서 직접 설치 |
| **Python** | `3.10` | 3.11+ 비권장 (호환성 미검증) |
| **mlagents** | `1.2.0.dev0` | PyPI 미배포 → GitHub 설치 필수 |
| **mlagents-envs** | `1.2.0.dev0` | 동일 |
| **PyTorch** | `2.10.0` | |
| **grpcio** | `1.53.2` | 상한선 엄수 (`<=1.53.2`) |
| **protobuf** | `3.20.3` | |
| **numpy** | `1.23.5` | |
| **TensorBoard** | `2.20.0` | |
| **onnx** | `1.15.0` | 학습 모델 export용 |

> ⚠️ **주의:** `mlagents 4.0.0`은 PyPI에 배포되지 않았습니다. `pip install mlagents==4.0.0` 명령어는 동작하지 않으며, 반드시 GitHub release_23 브랜치에서 직접 설치해야 합니다.

---

## **2. 사전 준비 (Prerequisites)**

### **2.1. Unity 설치**

1. [Unity Hub](https://unity.com/download) 설치
2. Unity Hub → **Installs → Add** 클릭
3. `Unity 6000.0.69f1` 선택 후 설치
   - 필수 모듈: **Windows Build Support**, **Visual Studio** (또는 Rider)

### **2.2. Python 3.10 설치**

```bash
# Windows: python.org에서 Python 3.10.x 다운로드 후 설치
# 설치 시 "Add Python to PATH" 반드시 체크

# 버전 확인
python --version
# Python 3.10.x 출력 확인
```

### **2.3. Git 설정**

```bash
git clone https://github.com/alpha7179/IIT_DroneLearning.git
cd IIT_DroneLearning
```

---

## **3. Python 가상환경 구축**

> 팀 내 가상환경 위치 기준: 프로젝트 Unity 폴더와 **별도 경로**에 생성 권장  
> (Unity가 가상환경 폴더를 에셋으로 인식하는 문제 방지)

```bash
# 가상환경 생성 (프로젝트 외부 경로 권장)
# 예시: C:\Users\{username}\Desktop\DroneRL\drone_env
python -m venv drone_env

# 활성화 (Windows)
drone_env\Scripts\activate

# 활성화 (macOS / Linux)
source drone_env/bin/activate

# 활성화 확인: 터미널 앞에 (drone_env) 표시 확인
```

---

## **4. ML-Agents 설치**

### **4.1. 설치 순서 (순서 엄수)**

> `mlagents-envs`를 먼저 설치한 뒤 `mlagents`를 설치해야 합니다.

```bash
# 가상환경 활성화 상태에서 실행

# Step 1. mlagents-envs 먼저 설치
pip install git+https://github.com/Unity-Technologies/ml-agents.git@release_23#subdirectory=ml-agents-envs

# Step 2. mlagents 설치
pip install git+https://github.com/Unity-Technologies/ml-agents.git@release_23#subdirectory=ml-agents

# Step 3. 설치 확인
mlagents-learn --help
# 도움말 출력 시 정상 설치 완료
```

### **4.2. requirements.txt**

```
mlagents @ git+https://github.com/Unity-Technologies/ml-agents.git@release_23#subdirectory=ml-agents
mlagents-envs @ git+https://github.com/Unity-Technologies/ml-agents.git@release_23#subdirectory=ml-agents-envs
torch>=2.0.0
grpcio>=1.11.0,<=1.53.2
protobuf>=3.20.0
numpy>=1.21.0
tensorboard>=2.10.0
onnx>=1.15.0
```

```bash
# requirements.txt로 일괄 설치 시
pip install -r IIT_DroneLearning/python/requirements.txt
```

---

## **5. Unity 프로젝트 설정**

### **5.1. ML-Agents Unity 패키지 설치**

1. Unity Hub에서 `IIT_DroneLearning/IIT_DroneLearning` 폴더를 프로젝트로 열기
2. Unity 상단 메뉴 → **Window → Package Manager**
3. 좌측 상단 **+** 버튼 → **Add package from git URL...**
4. 아래 URL 입력 후 Add:

```
https://github.com/Unity-Technologies/ml-agents.git?path=/com.unity.ml-agents#release_23
```

5. Package Manager에서 `ML Agents 4.0.0` 설치 확인

### **5.2. Input System 설정**

ML-Agents Heuristic 키보드 입력 사용을 위한 설정입니다.

1. **Edit → Project Settings → Player**
2. **Other Settings → Active Input Handling**
3. `Both` 로 변경 후 Unity 재시작

### **5.3. Inspector 설정 (Drone_Tracker 오브젝트 기준)**

씬 내 드론 오브젝트에 아래 컴포넌트를 부착하고 설정합니다.

**Behavior Parameters 컴포넌트:**

| 항목 | 값 |
|------|-----|
| Behavior Name | `DroneTracker` |
| Vector Observation Space Size | `12` |
| Stacked Vectors | `1` |
| Continuous Actions | `0` |
| Discrete Branches | `1` |
| Branch 0 Size | `13` |
| Behavior Type | `Default` (학습) / `Heuristic Only` (수동 테스트) |

**Decision Requester 컴포넌트:**

| 항목 | 값 |
|------|-----|
| Decision Period | `5` |
| Take Actions Between Decisions | `✅ 체크` |

### **5.4. Windows Defender 제외 설정 (Windows 권장)**

Unity 프로젝트 폴더를 Defender 실시간 검사 대상에서 제외하면 컴파일 속도가 크게 향상됩니다.

1. **Windows 보안 → 바이러스 및 위협 방지 → 설정 관리**
2. **제외 → 제외 추가 → 폴더**
3. Unity 프로젝트 루트 폴더 추가

---

## **6. Python ↔ Unity 연동 검증**

### **6.1. YAML 설정 파일**

경로: `IIT_DroneLearning/python/config/drone_test.yaml`

```yaml
behaviors:
  DroneTracker:
    trainer_type: ppo
    hyperparameters:
      batch_size: 64
      buffer_size: 2048
      learning_rate: 3.0e-4
      beta: 0.005
      epsilon: 0.2
      lambd: 0.95
      num_epoch: 3
    network_settings:
      normalize: false
      hidden_units: 128
      num_layers: 2
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
    max_steps: 100000
    time_horizon: 64
    summary_freq: 1000
```

### **6.2. 연동 검증 절차**

```bash
# Step 1. 가상환경 활성화
drone_env\Scripts\activate  # (Windows)

# Step 2. 프로젝트 루트로 이동
cd C:\...\IIT_DroneLearning

# Step 3. mlagents-learn 실행 (Unity 실행 전 먼저 시작)
mlagents-learn IIT_DroneLearning/python/config/drone_test.yaml --run-id=drone_test_v1

# 터미널에 아래 메시지 출력 시 Unity 연결 대기 중:
# [INFO] Listening on port 5004.
# [INFO] Start training by pressing the Play button in the Unity Editor.

# Step 4. Unity Editor에서 Play 버튼 클릭
# → 터미널에 에이전트 연결 로그 출력 시 연동 성공
```

### **6.3. TensorBoard 실행**

```bash
# 별도 터미널에서 실행
tensorboard --logdir IIT_DroneLearning/python/results

# 브라우저에서 확인
# http://localhost:6006
```

---

## **7. 프로젝트 구조**

```
IIT_DroneLearning/
├── IIT_DroneLearning/              # Unity 프로젝트
│   └── Assets/
│       ├── 00_BMW/                 # 배민우: 도시 환경 & 센서
│       ├── 00_LJW/                 # 이재왕: Evader AI & 분석
│       ├── 00_LKM/                 # 이강민: 물리 엔진
│       ├── 00_PJH/                 # 박재현: Tracker AI
│       ├── 01_Scenes/
│       │   └── DroneTest.unity     # 메인 작업 씬
│       ├── 02_Scripts/
│       │   ├── DronePhysics.cs     # 6-DOF 물리 엔진 ✅
│       │   └── DroneAgent.cs       # ML-Agents 에이전트 ✅
│       └── ML-Agents/
│
└── python/                         # Python ML 파이프라인
    ├── config/
    │   └── drone_test.yaml         # PPO 학습 설정 ✅
    ├── models/                     # 학습된 모델 (.onnx, .pt)
    ├── scripts/                    # 학습 & 추론 스크립트
    ├── utils/                      # 데이터 처리 유틸리티
    ├── results/                    # 실험 결과 로그
    └── requirements.txt            # ✅
```

### **브랜치 구조**

| 브랜치 | 담당자 | 역할 |
|--------|--------|------|
| `main` | 전원 | 최종 통합 브랜치 |
| `work/physics` | 이강민 | 드론 물리 엔진 |
| `work/environment` | 배민우 | 도시 환경 & 센서 |
| `work/pursuer` | 박재현 | Tracker AI & 학습 |
| `work/evader` | 이재왕 | Evader AI & 실험 |

---

## **8. 구현 완료 코드 명세**

### **8.1. DronePhysics.cs — 6-DOF 물리 엔진**

경로: `Assets/02_Scripts/DronePhysics.cs`

| 항목 | 상세 |
|------|------|
| **이동 제어** | `ApplyMovement(float x, float y, float z)` — 로컬 좌표 기준 3축 이동 |
| **회전 제어** | `ApplyYaw()`, `ApplyPitch()`, `ApplyRoll()` — 수동 각도 제어 |
| **물리 회전** | `freezeRotation = true` — Rigidbody 물리 회전 잠금, 스크립트 수동 제어 |
| **고도 제한** | `MinAltitude = 1f`, `MaxAltitude = 30f` |
| **공기저항** | `DragCoefficient = 0.95f` — 매 FixedUpdate 속도에 곱산 적용 |
| **기울기 제한** | `MaxPitchAngle = 45f`, `MaxRollAngle = 45f` |
| **외부 접근 메서드** | `GetVelocity()`, `GetRotation()`, `ResetPhysics()` |

**핵심 설계 원칙:** `transform.TransformDirection()`을 사용하여 로컬 좌표 기준으로 이동 벡터를 월드 좌표로 변환합니다. 이를 통해 드론이 어느 방향을 향하든 조작 방향이 직관적으로 유지됩니다.

### **8.2. DroneAgent.cs — ML-Agents 에이전트**

경로: `Assets/02_Scripts/DroneAgent.cs`

**상태 공간 (Observation Space): 12차원**

| 인덱스 | 데이터 | 설명 |
|--------|--------|------|
| 0–2 | 위치 (x, y, z) | 월드 좌표 정규화 값 |
| 3–5 | 속도 (vx, vy, vz) | 현재 이동 속도 벡터 |
| 6–8 | 회전 (Pitch, Yaw, Roll) | 정규화된 오일러 각 |
| 9–11 | 타겟 상대 위치 (dx, dy, dz) | 타겟까지의 상대 벡터 |

**행동 공간 (Action Space): Discrete 1 Branch, Size 13**

| 액션 인덱스 | 동작 | 키 (Heuristic) |
|------------|------|----------------|
| 0 | 정지 (Idle) | — |
| 1 / 2 | +X / -X 이동 | `D` / `A` |
| 3 / 4 | +Z / -Z 이동 | `W` / `S` |
| 5 / 6 | 상승 / 하강 (+Y / -Y) | `E` / `Q` |
| 7 / 8 | Yaw+ / Yaw- | `L` / `J` |
| 9 / 10 | Pitch+ / Pitch- | `I` / `K` |
| 11 / 12 | Roll+ / Roll- | `O` / `U` |

**상속 구조:**
```csharp
// Unity 6 호환을 위해 Awake() 오버라이드 시 반드시 base.Awake() 호출
protected override void Awake()
{
    base.Awake();
    // 초기화 로직
}
```

---

## **9. 트러블슈팅**

본 프로젝트 환경 구축 과정에서 확인된 문제와 해결 방법입니다.

### **❌ Issue 1: `pip install mlagents==4.0.0` 실패**
- **원인:** mlagents 4.0.0 (release_23)은 PyPI에 배포되지 않음
- **해결:** GitHub release_23 브랜치에서 직접 설치 → [Section 4.1 참고](#41-설치-순서-순서-엄수)

### **❌ Issue 2: Python 3.11+ 호환성 오류**
- **원인:** mlagents-envs 내부 의존 패키지 일부가 3.11+ 미지원
- **해결:** Python 3.10 사용 고정

### **❌ Issue 3: `"My Behavior" 에러` — 에이전트 연결 안됨**
- **원인:** 씬 내 기존 에이전트 오브젝트의 Behavior Name 미설정
- **해결:** 기존 씬 제거 후 `DroneTest.unity` 신규 씬 생성, Behavior Name을 YAML의 `behaviors` 키와 동일하게 설정 (`DroneTracker`)

### **❌ Issue 4: Heuristic 키보드 입력 미동작**
- **원인:** Unity 6의 기본 Input System이 New Input System으로 설정됨
- **해결:** **Edit → Project Settings → Player → Active Input Handling → `Both`** 로 변경 후 재시작

### **❌ Issue 5: Unity 컴파일 속도 극단적으로 느림 (Windows)**
- **원인:** Windows Defender 실시간 검사가 Unity 임시 파일 지속 스캔
- **해결:** Unity 프로젝트 폴더를 Defender 제외 목록에 추가

### **❌ Issue 6: Git clone / push 권한 오류**
- **원인:** 기존 경로의 `.git` 폴더 손상 또는 권한 충돌
- **해결:** 새 경로(`Desktop/IIT_Fresh`)에 클론 후 재작업

---

*최초 작성: 2026-03-08 | 담당: 이강민 (work/physics)*  
*본 문서는 환경 변경 시 즉시 업데이트합니다.*
