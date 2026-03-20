# DroneVisualPipeline

드론 FPV 카메라 영상을 ML-Agents 강화학습 Visual Observation으로 연결하고,
학습/분석용 이미지 데이터를 체계적으로 추출하며,
URP 네이티브 Depth Map 보조 센서를 제공하는 파이프라인.

기존 DroneAgent, DroneCameraSystem, DroneSensorSystem 코드를 **일절 수정하지 않고**
드론 GameObject에 컴포넌트를 추가 부착하는 방식으로 동작한다.

## 환경 요구사항

| 항목 | 버전 |
|---|---|
| Unity | 6000.0.69f1 |
| ML-Agents | 4.0.x |
| URP | 17.0.4 |
| Python | 3.10 |

## 시스템 구성

```
드론 GameObject
├── DroneAgent              (기존 — 수정 없음)
├── DronePhysics            (기존 — 수정 없음)
├── DroneSensorSystem       (기존 — 수정 없음)
├── DroneCameraSystem       (기존 — 수정 없음)
│   └── DroneCamera_Solo    (기존 자식 Camera)
│       ├── [Observation Camera]   ← DroneVisionSystem이 AddComponent<Camera>()로 추가
│       └── [Depth Camera]         ← DroneDepthSystem이 AddComponent<Camera>()로 추가
│
├── DroneVisionSystem       (신규) — RGB Visual Observation
├── DroneSnapshotSystem     (신규) — PNG/JPG + 메타데이터 저장
├── DroneDepthSystem        (신규) — Depth Map Visual Observation
├── RenderTextureSensorComponent ×2  (ML-Agents 네이티브, 수동 추가 필요)
```

핵심 설계: Observation 카메라와 Depth 카메라를 `DroneCamera_Solo` GameObject에 `AddComponent<Camera>()`로 추가한다.
동일 Transform을 공유하므로 DroneCameraSystem의 `LateUpdate()` 위치 갱신을 자동 추종하며,
Solo 카메라의 `targetTexture`는 `null`을 유지하여 기존 Display 출력을 보존한다.

## 파일 구조

```
DroneVisualPipeline/
├── Scripts/
│   ├── DroneVisualPipeline.asmdef     Assembly Definition
│   ├── DroneVisionSystem.cs           RGB Visual Observation 브릿지
│   ├── DroneSnapshotSystem.cs         이미지 + 메타데이터 캡처/저장
│   ├── DroneDepthSystem.cs            URP Depth Map 센서
│   ├── CaptureMetadata.cs             직렬화 구조체 (메타데이터, 에피소드 요약, SerializedVector3)
│   └── Shaders/
│       └── DroneDepthVisualize.shader URP HLSL Depth 시각화 셰이더
├── Tests/
│   ├── DroneVisualPipelineTests.asmdef
│   ├── DroneVisionSystemTests.cs
│   ├── DroneSnapshotSystemTests.cs
│   └── DroneDepthSystemTests.cs
└── Docs/
    └── README.md                      ← 이 문서
```

---

## 컴포넌트 상세

### 1. DroneVisionSystem

DroneCameraSystem의 Solo 카메라 렌더링을 RenderTexture에 캡처하여
ML-Agents `RenderTextureSensorComponent`를 통해 Visual Observation으로 전달한다.

#### 데이터 흐름

```
DroneCameraSystem.Camera (Solo)
  → 동일 Transform의 Observation Camera (별도 Camera 컴포넌트)
  → targetTexture = RenderTexture_RGB (84×84, ARGB32)
  → RenderTextureSensorComponent
  → ML-Agents CNN Encoder (simple / nature_cnn / resnet)
  → Policy Network (기존 44차원 Vector Obs와 결합)
```

#### Inspector 필드

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `enableVisualObservation` | bool | true | 활성/비활성 토글. 비활성 시 카메라 off → GPU 절약 |
| `textureWidth` | int [16-256] | 84 | RenderTexture 가로 해상도 |
| `textureHeight` | int [16-256] | 84 | RenderTexture 세로 해상도 |
| `grayscale` | bool | false | true=1채널 흑백, false=RGB 3채널. 플레이 모드 변경 시 경고 |
| `fovOverride` | float [0-170] | 0 | 0 이하 = Solo 카메라 FOV 추종 |
| `farClipOverride` | float | 0 | 0 이하 = Solo 카메라 값 추종 |
| `showDebugFrustum` | bool | false | Scene 뷰 Frustum 시각화 (초록색) |
| `showPreviewOverlay` | bool | false | Game 뷰 좌하단 미리보기 |
| `previewSize` | float [0.05-0.5] | 0.2 | 미리보기 크기 (화면 비율) |
| `sensorName` | string | "DroneRGB" | ML-Agents Behavior Parameters에 표시되는 센서 이름 |

#### OnValidate 실시간 반영

Inspector에서 값을 변경하면 즉시 반영된다 (`_isInitialized` 체크 후 동작):

- `enableVisualObservation` → 카메라 enabled 즉시 토글
- `textureWidth`/`textureHeight` → RenderTexture 재생성
- `fovOverride`/`farClipOverride` → 카메라 파라미터 즉시 반영
- `grayscale` → 플레이 모드에서는 `Debug.LogWarning` (ML-Agents 센서 채널 수 런타임 변경 불가)

#### 공개 프로퍼티

```csharp
Camera ObservationCamera { get; }     // Observation 전용 카메라
RenderTexture RenderTexture { get; }  // 현재 RenderTexture
bool IsInitialized { get; }           // 초기화 완료 여부
```

---

### 2. DroneSnapshotSystem

런타임에 카메라 렌더링을 PNG/JPG 파일 + 메타데이터 JSON으로 저장하는 데이터 추출 컴포넌트.
Sim2Real 데이터셋 구축, 오프라인 분석, 학습 디버깅에 활용한다.

#### 데이터 흐름

```
DroneVisionSystem.RenderTexture (또는 DroneDepthSystem.DepthRenderTexture)
  → AsyncGPUReadback.Request()        GPU→CPU 비동기 전송 (메인 스레드 차단 없음)
  → NativeArray<byte>
  → Texture2D.LoadRawTextureData()    재사용 텍스처 (GC 방지)
  → EncodeToPNG() / EncodeToJPG()
  → Task.Run()                        백그라운드 스레드 파일 저장
  → step_NNNN_rgb.png + step_NNNN_meta.json
```

#### Inspector 필드

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `enableCapture` | bool | false | 캡처 활성/비활성. 기본 비활성 (학습 중 성능 유지) |
| `captureInterval` | int [1-100] | 5 | 매 N 스텝마다 캡처 |
| `captureRGB` | bool | true | RGB 이미지 캡처 |
| `captureDepth` | bool | false | Depth 이미지 캡처 (DroneDepthSystem 필요) |
| `captureWidth` | int [0-1024] | 0 | 0 = DroneVisionSystem RT 해상도 사용 |
| `captureHeight` | int [0-1024] | 0 | 0 = DroneVisionSystem RT 해상도 사용 |
| `imageFormat` | ImageFormat | PNG | PNG 또는 JPG |
| `jpgQuality` | int [1-100] | 85 | JPG 품질 |
| `basePath` | string | "CapturedData" | 저장 기본 경로 (프로젝트 루트 기준) |
| `filePrefix` | string | "" | 비어있으면 DroneAgent.Role 자동 사용 (Evader/Pursuer) |
| `saveMetadata` | bool | true | 메타데이터 JSON 동시 저장 |
| `includeRayDistances` | bool | true | 메타데이터에 26개 레이센서 거리값 포함 |
| `saveEpisodeSummary` | bool | true | 에피소드 종료 시 요약 JSON 저장 |
| `maxAsyncRequests` | int [1-10] | 3 | 최대 동시 비동기 요청 수 |
| `logCaptureEvents` | bool | false | 캡처 시 Console 로그 |
| `showCaptureStatus` | bool | false | Game 뷰 우상단 에피소드/스텝 카운터 표시 |

#### 출력 파일 구조

```
CapturedData/
├── Evader_episode_0001/
│   ├── step_0005_rgb.png
│   ├── step_0005_depth.png          (captureDepth=true일 때)
│   ├── step_0005_meta.json
│   ├── step_0010_rgb.png
│   ├── step_0010_meta.json
│   ├── ...
│   └── episode_summary.json
├── Evader_episode_0002/
│   └── ...
├── Pursuer_episode_0001/
│   └── ...
```

#### 메타데이터 JSON 스키마 (step_NNNN_meta.json)

```json
{
    "timestamp": "2026-03-21T14:30:00.000Z",
    "episode": 42,
    "step": 150,
    "droneRole": "Evader",
    "dronePosition": { "x": 12.5, "y": 8.0, "z": -3.2 },
    "droneRotation": { "x": 0.0, "y": 45.0, "z": 0.0 },
    "droneVelocity": { "x": 1.2, "y": 0.0, "z": -0.8 },
    "targetRelativePosition": { "x": -5.0, "y": 0.5, "z": 10.0 },
    "sensorDistances": [0.0, 0.85, 0.42, ...],
    "reward": -0.001,
    "cumulativeReward": -0.15
}
```

`sensorDistances`는 DroneSensorSystem의 26개 정규화 거리값 (인덱스 0=Top, 1-8=TopMiddle, 9-16=Middle, 17-24=MiddleBottom, 25=Bottom).
`includeRayDistances=false`이면 빈 배열.

#### 에피소드 요약 JSON 스키마 (episode_summary.json)

```json
{
    "episode": 1,
    "droneRole": "Evader",
    "totalSteps": 300,
    "cumulativeReward": -0.45,
    "terminationReason": "captured",
    "capturedImageCount": 60,
    "startTime": "2026-03-21T14:30:00.000Z",
    "endTime": "2026-03-21T14:30:15.000Z"
}
```

#### 공개 API

```csharp
void OnEpisodeBegin()                              // 에피소드 시작 시 호출
void OnEpisodeEnd(string terminationReason)         // 에피소드 종료 시 호출

int EpisodeCount { get; }
int StepCount { get; }
int CapturedImageCount { get; }
string CurrentEpisodePath { get; }
```

#### DroneAgent 데이터 수집 방식

DroneVisualPipeline.asmdef에서 Assembly-CSharp(DroneAgent)를 직접 참조할 수 없으므로,
**리플렉션**으로 DroneAgent 데이터를 수집한다:

```
Start() → GetComponents<MonoBehaviour>() → 타입명 "DroneAgent" 필터링
  → FieldInfo 캐시: reward, cumulativeReward, Role, Target
  → FixedUpdate마다 FieldInfo.GetValue()로 읽기
  → DroneAgent 미존재 시 기본값 0.0f 사용
```

이 패턴은 DroneCameraSystem의 `ResolveRole()` 방식과 동일하다.

---

### 3. DroneDepthSystem

URP 네이티브 `_CameraDepthTexture`를 활용한 Depth Map 보조 센서.
커스텀 HLSL 셰이더로 깊이를 0~1 범위로 정규화하여 ML-Agents Visual Observation으로 등록한다.

#### 데이터 흐름

```
Depth Camera (DroneCamera_Solo에 추가된 별도 Camera)
  → depthTextureMode = DepthTextureMode.Depth
  → URP가 _CameraDepthTexture 자동 생성
  → RenderPipelineManager.endCameraRendering 콜백
  → Graphics.Blit(null, _depthRT, _depthMaterial)
  → DroneDepthVisualize.shader: LinearEyeDepth → _MaxDistance 정규화 → 0~1
  → RenderTexture_Depth (84×84, RFloat)
  → RenderTextureSensorComponent (Grayscale 모드)
  → ML-Agents CNN Encoder
```

URP에서 `Camera.SetReplacementShader()`는 지원되지 않으므로,
`RenderPipelineManager.endCameraRendering` + `Graphics.Blit` 방식을 사용한다.

#### Inspector 필드

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `enableDepthSensor` | bool | true | 활성/비활성 토글 |
| `depthWidth` | int [16-256] | 84 | Depth RT 가로 해상도 |
| `depthHeight` | int [16-256] | 84 | Depth RT 세로 해상도 |
| `depthMode` | DepthOutputMode | Linear | Linear=선형 깊이, Raw=Z-buffer 원본 |
| `maxDepthDistance` | float [0-1000] | 0 | 정규화 최대 거리. 0 = Far Clip Plane 사용 |
| `nearClipOverride` | float | 0 | 0 이하 = DroneCameraSystem 값 추종 |
| `farClipOverride` | float | 0 | 0 이하 = DroneCameraSystem 값 추종 |
| `colorRamp` | DepthColorRamp | Grayscale | Grayscale=흑백, Jet=컬러맵(파랑→빨강) |
| `showDebugFrustum` | bool | false | Scene 뷰 Frustum 시각화 (파란색) |
| `showPreviewOverlay` | bool | false | Game 뷰 좌하단 미리보기 (VisionSystem 위에 배치) |
| `previewSize` | float [0.05-0.5] | 0.2 | 미리보기 크기 |
| `sensorName` | string | "DroneDepth" | ML-Agents 센서 이름 |

#### Depth 셰이더 (DroneDepthVisualize.shader)

URP 호환 HLSL 셰이더. `DeclareDepthTexture.hlsl`의 `SampleSceneDepth()`로 `_CameraDepthTexture`를 샘플링한다.

| 파라미터 | 설명 |
|---|---|
| `_MaxDistance` | 정규화 최대 거리 (미터) |
| `_UseLinear` | 1=LinearEyeDepth 적용, 0=Raw Z-buffer |
| `_UseJetColorRamp` | 1=Jet 컬러맵, 0=Grayscale |

Jet 컬러맵: 0(파랑) → 0.25(시안) → 0.5(초록) → 0.75(노랑) → 1(빨강)

#### 공개 프로퍼티

```csharp
Camera DepthCamera { get; }
RenderTexture DepthRenderTexture { get; }
Material DepthMaterial { get; }
bool IsInitialized { get; }
```

---

### 4. CaptureMetadata.cs

`JsonUtility`와 호환되는 직렬화 구조체 모음.

| 클래스 | 용도 |
|---|---|
| `CaptureMetadata` | 스텝별 메타데이터 (step_NNNN_meta.json) |
| `EpisodeSummary` | 에피소드 요약 (episode_summary.json) |
| `SerializedVector3` | Vector3 직렬화 헬퍼 (JsonUtility는 Vector3 미지원) |

`SerializedVector3`는 `Vector3`와 암시적 변환 연산자를 제공한다:

```csharp
SerializedVector3 sv = transform.position;  // Vector3 → SerializedVector3
Vector3 v = sv;                              // SerializedVector3 → Vector3
```

---

## 사용법

### Step 1: 드론 GameObject에 컴포넌트 추가

1. 드론 GameObject 선택 (DroneAgent, DroneCameraSystem이 이미 부착된 상태)
2. `Add Component` → `DroneVisionSystem` 추가
3. `Add Component` → `RenderTextureSensorComponent` 추가 (ML-Agents 네이티브)
4. (선택) `Add Component` → `DroneDepthSystem` 추가
5. (선택) Depth 사용 시 `RenderTextureSensorComponent` 1개 더 추가 (총 2개)
6. (선택) `Add Component` → `DroneSnapshotSystem` 추가

`DroneVisionSystem`과 `DroneDepthSystem`은 `Start()`에서 자동으로 `RenderTextureSensorComponent`를 찾아
RenderTexture, SensorName, Grayscale을 할당한다.
`sensorName` 기반으로 매칭하므로 2개의 `RenderTextureSensorComponent`가 있어도 충돌하지 않는다.

### Step 2: Inspector 설정

```
DroneVisionSystem:
  enableVisualObservation = true
  textureWidth = 84
  textureHeight = 84
  grayscale = false
  sensorName = "DroneRGB"

DroneDepthSystem (선택):
  enableDepthSensor = true
  depthWidth = 84
  depthHeight = 84
  depthMode = Linear
  sensorName = "DroneDepth"

DroneSnapshotSystem (선택):
  enableCapture = false          ← 학습 중에는 false 권장 (성능)
  captureInterval = 5
  saveMetadata = true
```

### Step 3: ML-Agents YAML Config

`python/config/evader_s0_visual_template.yaml`을 복사하여 사용:

```yaml
behaviors:
  EvaderAgent:
    trainer_type: ppo
    hyperparameters:
      batch_size: 128            # Vector-only 64 → Visual 128
      buffer_size: 12800
    network_settings:
      normalize: true
      hidden_units: 256
      num_layers: 2
      vis_encode_type: simple    # simple | nature_cnn | resnet
    max_steps: 1000000           # 이미지 학습은 수렴이 느림
```

`RenderTextureSensorComponent`가 드론 GameObject에 있으면
ML-Agents가 자동으로 Visual Observation을 감지한다.
YAML에 별도 센서 설정은 필요 없다.

### Step 4: 학습 실행

```bash
mlagents-learn python/config/evader_s0_visual_template.yaml \
  --run-id=evader_visual_seed42 --force
```

### Step 5: 데이터 수집 (Sim2Real)

학습 완료 후 또는 별도 수집 세션에서:

1. Inspector에서 `DroneSnapshotSystem.enableCapture = true`
2. `captureInterval = 1` (매 스텝) 또는 `5` (5스텝마다)
3. Play 모드 실행
4. `CapturedData/` 폴더에 이미지 + 메타데이터 자동 저장

---

## 초기화 순서

컴포넌트 간 의존성을 보장하기 위해 Unity 생명주기를 활용한다:

```
Awake()  → DroneCameraSystem: DroneCamera_Solo 자식 GameObject + Camera 생성
Start()  → DroneVisionSystem: Solo Camera에 Observation Camera 추가, RT 생성, 센서 연동
Start()  → DroneDepthSystem:  Solo Camera에 Depth Camera 추가, 셰이더 로드, RT 생성, 센서 연동
Start()  → DroneSnapshotSystem: 컴포넌트 캐싱, 리플렉션 초기화, 에피소드 폴더 생성
```

`Start()`는 `Awake()` 이후에 호출되므로 DroneCameraSystem의 Solo 카메라가 반드시 존재한다.

---

## 에러 처리

### DroneVisionSystem

| 상황 | 처리 |
|---|---|
| DroneCameraSystem 미존재 | `LogError` + 컴포넌트 비활성화 |
| Solo 카메라 null | `LogError` + 컴포넌트 비활성화 |
| RenderTexture 생성 실패 | `LogError` + 컴포넌트 비활성화 |
| RenderTextureSensorComponent 미존재 | `LogWarning` + 수동 추가 안내 |
| Solo 카메라 targetTexture != null | `LogWarning` (Display 출력 깨질 수 있음) |

### DroneSnapshotSystem

| 상황 | 처리 |
|---|---|
| DroneVisionSystem 미존재 | `LogWarning` + RGB 캡처 비활성 |
| DroneDepthSystem 미존재 (captureDepth=true) | `LogWarning` + captureDepth 자동 false |
| 저장 경로 쓰기 실패 | `LogError` + enableCapture 자동 false |
| AsyncGPUReadback 실패 | `LogWarning` + 해당 프레임 건너뜀 |
| DroneAgent 미존재 | 메타데이터 reward/role 기본값 사용 |

### DroneDepthSystem

| 상황 | 처리 |
|---|---|
| DroneCameraSystem 미존재 | `LogError` + 독립 카메라 폴백 (드론 Transform에 직접 추가) |
| 셰이더 로드 실패 | `LogError` + 컴포넌트 비활성화 |
| RenderTexture 생성 실패 | `LogError` + 컴포넌트 비활성화 |
| RenderTextureSensorComponent 미존재 | `LogWarning` + 수동 추가 안내 |

---

## 성능 고려사항

- Observation/Depth 카메라는 `enableVisualObservation`/`enableDepthSensor`로 개별 비활성화 가능 → GPU 절약
- DroneSnapshotSystem은 `AsyncGPUReadback` + `Task.Run()` 백그라운드 I/O → 메인 스레드 차단 없음
- `maxAsyncRequests`로 동시 GPU readback 수 제한 → 메모리 사용 제어
- `_readbackTexture` 재사용 → FixedUpdate 중 GC 할당 최소화
- 학습 중에는 `enableCapture = false` 권장 (파일 I/O 오버헤드 제거)
- 목표: 단일 드론 < 2ms/frame, 10드론 동시 ≥ 60 FPS

---

## Assembly Definition 의존성

```
DroneVisualPipeline.asmdef
  → DroneCameraSystem       (DroneCameraSystem, Camera 참조)
  → DroneSensorSystem       (GetAllNormalizedDistances() 직접 호출)
  → Unity.ML-Agents         (RenderTextureSensorComponent)

DroneVisualPipelineTests.asmdef
  → DroneVisualPipeline
  → DroneCameraSystem
  → DroneSensorSystem
  → Unity.ML-Agents
  → UnityEngine.TestRunner
  → UnityEditor.TestRunner
```

DroneAgent(Assembly-CSharp)는 직접 참조하지 않는다.
reward, cumulativeReward, Role, Target 등은 리플렉션으로 접근한다.

---

## 테스트

Edit Mode 단위 테스트. DroneSensorSystem 테스트 패턴(Reflection 기반 Awake/Start 수동 호출)을 따른다.

```
DroneVisionSystemTests.cs    — 10개 테스트 (RT 생성, 해상도, 카메라 토글, OnValidate, Solo 무간섭)
DroneSnapshotSystemTests.cs  — 11개 테스트 (에피소드 카운터, 경로 생성, 직렬화, 클램핑)
DroneDepthSystemTests.cs     — 7개 단위 + 4개 속성 테스트 (RT 생성, 카메라 토글, 해상도, OnValidate 안전성)
```

셰이더 로드가 필요한 DroneDepthSystem 테스트는 Edit Mode에서 `Shader.Find()`가 실패할 수 있으므로
`Assert.Inconclusive`로 처리하고 Play Mode 재검증을 안내한다.

실행:

```
Unity Editor → Window → General → Test Runner → Edit Mode → Run All
```

---

## 디버그 도구

### Scene 뷰 Frustum 시각화

- `DroneVisionSystem.showDebugFrustum = true` → 초록색 Frustum
- `DroneDepthSystem.showDebugFrustum = true` → 파란색 Frustum

### Game 뷰 미리보기

- `DroneVisionSystem.showPreviewOverlay = true` → 좌하단 RGB 미리보기
- `DroneDepthSystem.showPreviewOverlay = true` → RGB 위에 Depth 미리보기
- `DroneSnapshotSystem.showCaptureStatus = true` → 우상단 에피소드/스텝 카운터

### Console 로그

모든 컴포넌트는 `[DroneVisionSystem]`, `[DroneSnapshotSystem]`, `[DroneDepthSystem]` 접두사로 로그를 출력한다.
초기화 완료 시 설정 요약 로그가 출력된다:

```
[DroneVisionSystem] 초기화 완료 — 84x84, grayscale=False, sensor='DroneRGB'
[DroneDepthSystem] 초기화 완료 — 84x84, mode=Linear, sensor='DroneDepth'
```
