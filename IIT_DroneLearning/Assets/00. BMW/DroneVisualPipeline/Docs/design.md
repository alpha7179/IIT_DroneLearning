# 설계 문서: 드론 카메라 데이터 추출 & 강화학습 Visual Pipeline

## 개요

본 시스템은 3개의 독립 컴포넌트로 구성된다:

1. **DroneVisionSystem** — DroneCameraSystem의 카메라 영상을 RenderTexture로 캡처하여 ML-Agents Visual Observation으로 전달
2. **DroneSnapshotSystem** — 런타임 카메라 렌더링을 PNG + 메타데이터 JSON으로 저장하는 데이터 추출 파이프라인
3. **DroneDepthSystem** — URP 네이티브 Depth 텍스처를 활용한 Depth Map 보조 센서

모든 컴포넌트는 기존 DroneAgent, DroneCameraSystem, DroneSensorSystem 코드를 수정하지 않고, 드론 GameObject에 추가 부착하는 방식으로 동작한다.

## 아키텍처

```
드론 GameObject
├── DroneAgent                    (기존 — 수정 없음)
├── DronePhysics                  (기존 — 수정 없음)
├── DroneSensorSystem             (기존 — 수정 없음)
├── DroneCameraSystem             (기존 — 수정 없음)
│   ├── DroneCamera_Solo          (기존 Camera)
│   └── DroneCamera_MultiView    (기존 Camera)
│
├── DroneVisionSystem              (신규 — Phase 1)
│   ├── RenderTexture_RGB         (84×84, 학습용)
│   └── RenderTextureSensorComponent (ML-Agents 네이티브)
│
├── DroneSnapshotSystem            (신규 — Phase 2)
│   └── 파일 저장 로직 (AsyncGPUReadback → PNG + JSON)
│
└── DroneDepthSystem              (신규 — Phase 4)
    ├── DepthCamera               (DroneCamera_Solo GO에 AddComponent, RenderPipelineManager 방식)
    ├── RenderTexture_Depth       (84×84, RFloat, 학습용)
    └── RenderTextureSensorComponent (ML-Agents 네이티브)
```

> **Phase 번호 체계**: Phase 1(DroneVisionSystem), Phase 2(DroneSnapshotSystem), Phase 4(DroneDepthSystem). Phase 3(도메인 랜덤화 — 센서 노이즈·조명 변화)은 Stage3 단계에서 별도 구현 예정으로 본 파이프라인 범위에서 제외한다.

### 데이터 흐름

```
Phase 1 (Visual Observation):
  ObservationCamera (DroneCamera_Solo GO에 AddComponent<Camera>, Solo 카메라와 동일 Transform 자동 공유)
    → targetTexture = RenderTexture_RGB (84×84)
    → RenderTextureSensorComponent
    → ML-Agents CNN Encoder
    → Policy Network (Vector Obs 44차원과 결합)
  ※ DroneCameraSystem.Camera (Solo) 의 targetTexture 는 null 유지 (Display 출력 보존)

Phase 2 (Image Capture):
  RenderTexture_RGB (또는 별도 캡처용 RT)
    → AsyncGPUReadback.Request()
    → NativeArray<byte> (GPU → CPU 비동기 전송)
    → Texture2D.LoadRawTextureData()
    → File.WriteAllBytesAsync() → step_NNNN_rgb.png
    → JsonUtility.ToJson() → step_NNNN_meta.json

Phase 4 (Depth Map):
  DepthCamera (DroneCamera_Solo GO에 AddComponent<Camera>, depthTextureMode = Depth)
    → RenderPipelineManager.endCameraRendering + Graphics.Blit(_depthMaterial)
    → 커스텀 셰이더: _CameraDepthTexture → LinearEyeDepth → 정규화(0~1)
    → RenderTexture_Depth (84×84, RFloat)
    → RenderTextureSensorComponent
    → ML-Agents CNN Encoder (Grayscale 모드)
```

## 컴포넌트 상세 설계

### 1. DroneVisionSystem (Phase 1)

DroneCameraSystem의 Solo 카메라 렌더링을 RenderTexture에 캡처하여 ML-Agents Visual Observation으로 전달하는 브릿지 컴포넌트.

```csharp
namespace DroneVisualPipeline
{
    [RequireComponent(typeof(DroneCameraSystem))]
    public class DroneVisionSystem : MonoBehaviour
    {
        #region Unity Inspector 노출 속성

        [Header("Visual Observation 설정")]
        [Tooltip("Visual Observation 활성화 여부\n비활성 시 Observation 카메라를 끄고 GPU 리소스를 절약한다")]
        public bool enableVisualObservation = true;

        [Tooltip("RenderTexture 가로 해상도 (픽셀)\n변경 시 RenderTexture를 재생성한다")]
        [Range(16, 256)]
        public int textureWidth = 84;

        [Tooltip("RenderTexture 세로 해상도 (픽셀)\n변경 시 RenderTexture를 재생성한다")]
        [Range(16, 256)]
        public int textureHeight = 84;

        [Tooltip("Grayscale 모드\ntrue=1채널(흑백), false=RGB 3채널\n※ 런타임 변경 시 RenderTextureSensor 재생성 필요 (에디터 경고 출력)")]
        public bool grayscale = false;

        [Header("카메라 오버라이드")]
        [Tooltip("Observation 카메라 FOV 오버라이드\n0 이하이면 DroneCameraSystem의 FOV를 그대로 사용한다")]
        [Range(0f, 170f)]
        public float fovOverride = 0f;

        [Tooltip("Observation 카메라 Far Clip Plane 오버라이드\n0 이하이면 DroneCameraSystem의 값을 그대로 사용한다")]
        public float farClipOverride = 0f;

        [Header("디버그")]
        [Tooltip("Scene 뷰에서 Observation 카메라의 Frustum을 시각화한다")]
        public bool showDebugFrustum = false;

        [Tooltip("Game 뷰 좌하단에 RenderTexture 미리보기를 표시한다")]
        public bool showPreviewOverlay = false;

        [Tooltip("미리보기 크기 (화면 비율, 0.1 = 10%)")]
        [Range(0.05f, 0.5f)]
        public float previewSize = 0.2f;

        [Header("센서 등록")]
        [Tooltip("ML-Agents 센서 이름 (Behavior Parameters에 표시되는 이름)")]
        public string sensorName = "DroneRGB";

        #endregion

        // 내부 상태
        private RenderTexture _renderTexture;
        private DroneCameraSystem _cameraSystem;
        private Camera _observationCamera;
        private bool _isInitialized;
        private int _lastWidth;
        private int _lastHeight;
    }
}
```

**OnValidate 실시간 반영 정책:**

```csharp
private void OnValidate()
{
    if (!_isInitialized) return;

    // 1. 활성/비활성 즉시 반영
    if (_observationCamera != null)
        _observationCamera.enabled = enableVisualObservation;

    // 2. 해상도 변경 감지 → RenderTexture 재생성
    if (_lastWidth != textureWidth || _lastHeight != textureHeight)
    {
        RecreateRenderTexture();
        _lastWidth = textureWidth;
        _lastHeight = textureHeight;
    }

    // 3. FOV / FarClip 오버라이드 즉시 반영
    if (_observationCamera != null)
    {
        _observationCamera.fieldOfView = fovOverride > 0f
            ? fovOverride
            : _cameraSystem.fieldOfView;
        _observationCamera.farClipPlane = farClipOverride > 0f
            ? farClipOverride
            : _cameraSystem.farClipPlane;
    }

    // 4. Grayscale 변경은 런타임 중 센서 재생성 불가 → 경고
    // (ML-Agents 센서는 초기화 후 채널 수 변경 불가)
}
```

| Inspector 필드 | 실시간 반영 | 비고 |
|---|---|---|
| `enableVisualObservation` | ✅ 즉시 | 카메라 enabled 토글 |
| `textureWidth` / `textureHeight` | ✅ 즉시 | RT 재생성 (에디터 모드), 플레이 모드에서는 다음 에피소드부터 |
| `grayscale` | ⚠️ 에디터만 | 플레이 모드 변경 시 `Debug.LogWarning` 출력 |
| `fovOverride` | ✅ 즉시 | 0 이하 = DroneCameraSystem 값 추종 |
| `farClipOverride` | ✅ 즉시 | 0 이하 = DroneCameraSystem 값 추종 |
| `showDebugFrustum` | ✅ 즉시 | OnDrawGizmos에서 참조 |
| `showPreviewOverlay` | ✅ 즉시 | OnGUI에서 참조 |
| `previewSize` | ✅ 즉시 | OnGUI에서 참조 |
| `sensorName` | ❌ 초기화 시만 | 센서 이름은 Start에서 고정 |

**핵심 설계 결정:**

1. DroneCameraSystem의 Solo 카메라에 직접 `targetTexture`를 설정하면 기존 Display 출력이 깨진다. 따라서 **`DroneCamera_Solo` GameObject에 `AddComponent<Camera>()`로 두 번째 Camera 컴포넌트를 추가**하는 방식을 사용한다. 동일 GameObject에 있으므로 DroneCameraSystem의 `LateUpdate()` 위치 갱신을 별도 추종 로직 없이 자동 공유한다.

2. Observation 카메라는 DroneCameraSystem의 Solo 카메라와 동일한 설정(FOV, clip plane)을 복제하되, `targetTexture`만 RenderTexture_RGB로 설정한다.

3. `enableVisualObservation = false`일 때 Observation 카메라를 비활성화하여 GPU 리소스를 절약한다.

4. **초기화 순서 보장**: DroneVisionSystem은 `Awake()` 대신 `Start()`에서 초기화한다. DroneCameraSystem이 `Awake()`에서 `DroneCamera_Solo` GameObject를 생성하므로, `Start()`에서 접근 시 `DroneCameraSystem.Camera` 참조가 항상 유효하게 보장된다 (Unity 생명주기: 동일 프레임 내 Awake 전체 완료 → Start 실행).

**RenderTextureSensorComponent 연동:**

ML-Agents의 `RenderTextureSensorComponent`는 Inspector에서 RenderTexture를 할당받아 자동으로 Visual Observation을 생성한다. DroneVisionSystem은 `Start()`에서 RenderTexture를 생성하고, 같은 GameObject의 `RenderTextureSensorComponent`에 할당한다.

```
DroneVisionSystem.Start():  // Awake 대신 Start 사용 — DroneCameraSystem.Awake() 완료 후 실행 보장
  1. DroneCameraSystem 참조 획득 (GetComponent)
  2. DroneCameraSystem.Camera.gameObject (DroneCamera_Solo) 에서 AddComponent<Camera>()
     → Observation 전용 Camera 컴포넌트 추가 (동일 Transform 자동 공유)
  3. RenderTexture 생성 (textureWidth × textureHeight, ARGB32)
  4. Observation 카메라 설정 = Solo 카메라 복제 (FOV, nearClip, farClip)
  5. Observation 카메라의 targetTexture = RenderTexture
  6. (검증) Solo 카메라의 targetTexture == null 확인
  7. RenderTextureSensorComponent에 RenderTexture 할당
  8. _isInitialized = true
```

### 2. DroneSnapshotSystem (Phase 2)

런타임에 카메라 렌더링을 PNG 파일 + 메타데이터 JSON으로 저장하는 데이터 추출 컴포넌트.

```csharp
namespace DroneVisualPipeline
{
    public class DroneSnapshotSystem : MonoBehaviour
    {
        #region Unity Inspector 노출 속성

        [Header("캡처 제어")]
        [Tooltip("캡처 활성화 여부\n비활성 시 모든 캡처를 중단한다")]
        public bool enableCapture = false;

        [Tooltip("캡처 간격 (매 N 스텝마다 캡처)\n1=매 스텝, 5=5스텝마다")]
        [Range(1, 100)]
        public int captureInterval = 5;

        [Header("캡처 대상")]
        [Tooltip("RGB 이미지 캡처 활성화")]
        public bool captureRGB = true;

        [Tooltip("Depth 이미지 캡처 활성화\nDroneDepthSystem이 같은 GameObject에 존재해야 한다")]
        public bool captureDepth = false;

        [Header("이미지 설정")]
        [Tooltip("캡처 이미지 가로 해상도 (픽셀)\n0이면 DroneVisionSystem의 RenderTexture 해상도를 그대로 사용")]
        [Range(0, 1024)]
        public int captureWidth = 0;

        [Tooltip("캡처 이미지 세로 해상도 (픽셀)\n0이면 DroneVisionSystem의 RenderTexture 해상도를 그대로 사용")]
        [Range(0, 1024)]
        public int captureHeight = 0;

        [Tooltip("이미지 저장 포맷")]
        public ImageFormat imageFormat = ImageFormat.PNG;

        [Tooltip("JPG 품질 (imageFormat이 JPG일 때만 적용)")]
        [Range(1, 100)]
        public int jpgQuality = 85;

        [Header("저장 경로")]
        [Tooltip("저장 기본 경로 (프로젝트 루트 기준)\n런타임 변경 시 다음 에피소드부터 적용")]
        public string basePath = "CapturedData";

        [Tooltip("파일명 접두사 (드론 식별용)\n비어있으면 DroneRole을 자동 사용")]
        public string filePrefix = "";

        [Header("메타데이터")]
        [Tooltip("메타데이터 JSON 동시 저장 활성화")]
        public bool saveMetadata = true;

        [Tooltip("메타데이터에 26개 레이센서 거리값 포함")]
        public bool includeRayDistances = true;

        [Tooltip("에피소드 종료 시 요약 JSON 저장")]
        public bool saveEpisodeSummary = true;

        [Header("성능")]
        [Tooltip("최대 동시 비동기 요청 수\n높을수록 메모리 사용 증가")]
        [Range(1, 10)]
        public int maxAsyncRequests = 3;

        [Header("디버그")]
        [Tooltip("캡처 시 Console에 파일 경로 로그 출력")]
        public bool logCaptureEvents = false;

        [Tooltip("현재 에피소드/스텝 카운터를 Scene 뷰에 표시")]
        public bool showCaptureStatus = false;

        #endregion

        // 내부 상태
        private int _episodeCount = 0;
        private int _stepCount = 0;
        private string _currentEpisodePath;
        private RenderTexture _captureRT;
        private Texture2D _readbackTexture;  // 재사용하여 GC 방지
        private int _pendingRequests = 0;

        // DroneAgent 리플렉션 참조 (Assembly-CSharp 직접 참조 불가)
        // DroneCameraSystem.ResolveRole() 패턴과 동일하게 적용
        private MonoBehaviour _droneAgent;
        private System.Reflection.FieldInfo _rewardField;
        private System.Reflection.FieldInfo _cumulativeRewardField;
        private System.Reflection.PropertyInfo _velocityProperty;
    }

    public enum ImageFormat { PNG, JPG }
}
```

**OnValidate 실시간 반영 정책:**

```csharp
private void OnValidate()
{
    // 1. captureInterval 범위 클램핑
    captureInterval = Mathf.Clamp(captureInterval, 1, 100);
    maxAsyncRequests = Mathf.Clamp(maxAsyncRequests, 1, 10);
    jpgQuality = Mathf.Clamp(jpgQuality, 1, 100);

    // 2. 캡처 해상도 변경 시 readback 텍스처 재생성 예약
    if (Application.isPlaying && _readbackTexture != null)
    {
        int targetW = captureWidth > 0 ? captureWidth : GetSourceWidth();
        int targetH = captureHeight > 0 ? captureHeight : GetSourceHeight();
        if (_readbackTexture.width != targetW || _readbackTexture.height != targetH)
            RecreateReadbackTexture(targetW, targetH);
    }

    // 3. basePath 유효성 검사 (빈 문자열 방지)
    if (string.IsNullOrWhiteSpace(basePath))
        basePath = "CapturedData";
}
```

| Inspector 필드 | 실시간 반영 | 비고 |
|---|---|---|
| `enableCapture` | ✅ 즉시 | 다음 FixedUpdate부터 캡처 중단/재개 |
| `captureInterval` | ✅ 즉시 | 다음 스텝부터 새 간격 적용 |
| `captureRGB` / `captureDepth` | ✅ 즉시 | 다음 캡처 시점부터 반영 |
| `captureWidth` / `captureHeight` | ✅ 즉시 | Texture2D 재생성 (플레이 모드) |
| `imageFormat` / `jpgQuality` | ✅ 즉시 | 다음 저장부터 반영 |
| `basePath` | ⚠️ 에피소드 경계 | 현재 에피소드 폴더는 유지, 다음 에피소드부터 새 경로 |
| `filePrefix` | ⚠️ 에피소드 경계 | 다음 에피소드부터 반영 |
| `saveMetadata` | ✅ 즉시 | 다음 캡처부터 반영 |
| `includeRayDistances` | ✅ 즉시 | 다음 메타데이터부터 반영 |
| `saveEpisodeSummary` | ✅ 즉시 | 현재 에피소드 종료 시 반영 |
| `maxAsyncRequests` | ✅ 즉시 | 큐 제한 즉시 변경 |
| `logCaptureEvents` | ✅ 즉시 | 다음 캡처부터 반영 |
| `showCaptureStatus` | ✅ 즉시 | OnDrawGizmos에서 참조 |

**DroneAgent 데이터 수집 (리플렉션):**

DroneVisualPipeline.asmdef는 Assembly-CSharp(DroneAgent)를 직접 참조할 수 없으므로, DroneCameraSystem의 `ResolveRole()` 패턴과 동일하게 리플렉션을 사용한다.

```
DroneSnapshotSystem.Start():
  - GetComponents<MonoBehaviour>() 순회 → 타입명 "DroneAgent" 필터링
  - FieldInfo 캐싱: "reward", "cumulativeReward"
  - PropertyInfo 캐싱: 드론 velocity (Rigidbody 또는 DronePhysics 경유)
  - DroneAgent 미존재 또는 필드 없을 시 기본값 0.0f 사용

FixedUpdate (메타데이터 수집 시점):
  - _rewardField.GetValue(_droneAgent) → float reward
  - _cumulativeRewardField.GetValue(_droneAgent) → float cumulativeReward
  - transform.position, rotation → 직접 접근 (MonoBehaviour 공통)
```

**비동기 캡처 파이프라인:**

```
FixedUpdate (매 captureInterval 스텝):
  1. AsyncGPUReadback.Request(_captureRT) 호출
  2. 콜백에서 NativeArray<byte> 수신
  3. _readbackTexture.LoadRawTextureData(data)
  4. byte[] png = _readbackTexture.EncodeToPNG()
  5. Task.Run(() => File.WriteAllBytes(path, png))  // 백그라운드 스레드
  6. 메타데이터 JSON 직렬화 → 동일 백그라운드 스레드에서 저장
```

**메타데이터 스키마:**

```json
{
  "timestamp": "2026-03-20T14:30:00.000Z",
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

**파일 구조:**

```
CapturedData/
├── Evader_episode_0001/
│   ├── step_0000_rgb.png
│   ├── step_0000_depth.png       (captureDepth=true일 때)
│   ├── step_0000_meta.json
│   ├── step_0005_rgb.png
│   ├── step_0005_meta.json
│   ├── ...
│   └── episode_summary.json      (에피소드 종료 시 생성)
├── Evader_episode_0002/
│   └── ...
├── Pursuer_episode_0001/
│   └── ...
```

**에피소드 요약 스키마:**

```json
{
  "episode": 1,
  "droneRole": "Evader",
  "totalSteps": 300,
  "cumulativeReward": -0.45,
  "terminationReason": "captured",
  "capturedImageCount": 60,
  "startTime": "2026-03-20T14:30:00.000Z",
  "endTime": "2026-03-20T14:30:15.000Z"
}
```

### 3. DroneDepthSystem (Phase 4)

URP 네이티브 Depth 텍스처를 활용한 Depth Map 보조 센서.

```csharp
namespace DroneVisualPipeline
{
    [RequireComponent(typeof(DroneCameraSystem))]
    public class DroneDepthSystem : MonoBehaviour
    {
        #region Unity Inspector 노출 속성

        [Header("Depth 센서 설정")]
        [Tooltip("Depth 센서 활성화 여부\n비활성 시 Depth 카메라를 끄고 GPU 리소스를 절약한다")]
        public bool enableDepthSensor = true;

        [Tooltip("Depth RenderTexture 가로 해상도 (픽셀)\n변경 시 RenderTexture를 재생성한다")]
        [Range(16, 256)]
        public int depthWidth = 84;

        [Tooltip("Depth RenderTexture 세로 해상도 (픽셀)\n변경 시 RenderTexture를 재생성한다")]
        [Range(16, 256)]
        public int depthHeight = 84;

        [Header("Depth 렌더링")]
        [Tooltip("Depth 출력 모드\nLinear=선형 깊이(거리 비례), Raw=Z-buffer 원본(비선형)")]
        public DepthOutputMode depthMode = DepthOutputMode.Linear;

        [Tooltip("Depth 정규화 최대 거리 (미터)\n0 이하이면 카메라 Far Clip Plane을 사용한다")]
        [Range(0f, 1000f)]
        public float maxDepthDistance = 0f;

        [Header("카메라 오버라이드")]
        [Tooltip("Depth 카메라 Near Clip Plane 오버라이드\n0 이하이면 DroneCameraSystem의 값을 그대로 사용한다")]
        public float nearClipOverride = 0f;

        [Tooltip("Depth 카메라 Far Clip Plane 오버라이드\n0 이하이면 DroneCameraSystem의 값을 그대로 사용한다")]
        public float farClipOverride = 0f;

        [Header("시각화")]
        [Tooltip("Depth Map 컬러 램프 모드\nGrayscale=흑백, Jet=Jet 컬러맵(가까움=파랑, 먼=빨강)")]
        public DepthColorRamp colorRamp = DepthColorRamp.Grayscale;

        [Header("디버그")]
        [Tooltip("Scene 뷰에서 Depth 카메라의 Frustum을 시각화한다")]
        public bool showDebugFrustum = false;

        [Tooltip("Game 뷰 좌하단에 Depth Map 미리보기를 표시한다")]
        public bool showPreviewOverlay = false;

        [Tooltip("미리보기 크기 (화면 비율, 0.1 = 10%)")]
        [Range(0.05f, 0.5f)]
        public float previewSize = 0.2f;

        [Header("센서 등록")]
        [Tooltip("ML-Agents 센서 이름 (Behavior Parameters에 표시되는 이름)")]
        public string sensorName = "DroneDepth";

        #endregion

        // 내부 상태
        private Camera _depthCamera;
        private RenderTexture _depthRT;
        private Material _depthMaterial;  // 커스텀 Depth 시각화 셰이더
        private DroneCameraSystem _cameraSystem;
        private bool _isInitialized;
        private int _lastWidth;
        private int _lastHeight;
    }

    public enum DepthOutputMode { Linear, Raw }
    public enum DepthColorRamp { Grayscale, Jet }
}
```

**OnValidate 실시간 반영 정책:**

```csharp
private void OnValidate()
{
    if (!_isInitialized) return;

    // 1. 활성/비활성 즉시 반영
    if (_depthCamera != null)
        _depthCamera.enabled = enableDepthSensor;

    // 2. 해상도 변경 감지 → RenderTexture 재생성
    if (_lastWidth != depthWidth || _lastHeight != depthHeight)
    {
        RecreateDepthRenderTexture();
        _lastWidth = depthWidth;
        _lastHeight = depthHeight;
    }

    // 3. Near/Far Clip 오버라이드 즉시 반영
    if (_depthCamera != null)
    {
        _depthCamera.nearClipPlane = nearClipOverride > 0f
            ? nearClipOverride
            : _cameraSystem.nearClipPlane;
        _depthCamera.farClipPlane = farClipOverride > 0f
            ? farClipOverride
            : _cameraSystem.farClipPlane;
    }

    // 4. Depth 셰이더 파라미터 즉시 반영
    if (_depthMaterial != null)
    {
        float maxDist = maxDepthDistance > 0f
            ? maxDepthDistance
            : (_depthCamera != null ? _depthCamera.farClipPlane : 500f);
        _depthMaterial.SetFloat("_MaxDistance", maxDist);
        _depthMaterial.SetInt("_UseLinear", depthMode == DepthOutputMode.Linear ? 1 : 0);
        _depthMaterial.SetInt("_UseJetColorRamp", colorRamp == DepthColorRamp.Jet ? 1 : 0);
    }

    // 5. maxDepthDistance 범위 클램핑
    maxDepthDistance = Mathf.Max(0f, maxDepthDistance);
}
```

| Inspector 필드 | 실시간 반영 | 비고 |
|---|---|---|
| `enableDepthSensor` | ✅ 즉시 | 카메라 enabled 토글 |
| `depthWidth` / `depthHeight` | ✅ 즉시 | RT 재생성 (에디터 모드), 플레이 모드에서는 다음 에피소드부터 |
| `depthMode` | ✅ 즉시 | 셰이더 파라미터 업데이트 |
| `maxDepthDistance` | ✅ 즉시 | 셰이더 `_MaxDistance` 업데이트 |
| `nearClipOverride` | ✅ 즉시 | 0 이하 = DroneCameraSystem 값 추종 |
| `farClipOverride` | ✅ 즉시 | 0 이하 = DroneCameraSystem 값 추종 |
| `colorRamp` | ✅ 즉시 | 셰이더 `_UseJetColorRamp` 업데이트 |
| `showDebugFrustum` | ✅ 즉시 | OnDrawGizmos에서 참조 |
| `showPreviewOverlay` | ✅ 즉시 | OnGUI에서 참조 |
| `previewSize` | ✅ 즉시 | OnGUI에서 참조 |
| `sensorName` | ❌ 초기화 시만 | 센서 이름은 Start에서 고정 |

**Depth 렌더링 방식:**

`Camera.SetReplacementShader()`는 Built-in Render Pipeline 전용 API로 **URP 17.0.4에서 동작하지 않는다**. URP에서는 **`RenderPipelineManager.endCameraRendering` 콜백 + `Graphics.Blit`** 방식을 사용한다:

1. DroneCameraSystem.Camera (Solo) 의 GameObject에 `AddComponent<Camera>()`로 Depth 전용 Camera 추가
   → DroneVisionSystem과 동일 방식, 동일 Transform 자동 공유
2. Depth 카메라 `depthTextureMode = DepthTextureMode.Depth` 설정 → URP가 `_CameraDepthTexture` 자동 생성
3. `RenderPipelineManager.endCameraRendering` 이벤트에서 해당 카메라 완료 시점에 콜백
4. `Graphics.Blit(null, _depthRT, _depthMaterial)` → 커스텀 셰이더가 `_CameraDepthTexture` 샘플링 → 정규화된 선형 깊이를 `_depthRT`에 출력
5. RenderTextureSensorComponent를 통해 ML-Agents Grayscale Visual Observation으로 등록

**커스텀 Depth 셰이더 (URP 호환 HLSL):**

```hlsl
// DroneDepthVisualize.shader (URP 호환)
// Properties: _CameraDepthTexture, _MaxDistance, _UseLinear, _UseJetColorRamp
float depth = SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, sampler_CameraDepthTexture, uv);
float linearDepth = LinearEyeDepth(depth, _ZBufferParams);
float normalized = saturate(linearDepth / _MaxDistance);  // 0~1 정규화
return float4(normalized, normalized, normalized, 1.0);
```

**설계 결정 근거:**
- `Camera.SetReplacementShader()` : Built-in Pipeline 전용, URP 17.0.4 미지원 ❌
- `URP Renderer Feature` : 전역 설정, 드론별 독립 Depth 렌더링 제어 불가 ❌
- `RenderPipelineManager.endCameraRendering` : 카메라별 콜백으로 드론별 독립 제어 가능, URP 네이티브 ✅
- `OnDestroy()`에서 `RenderPipelineManager.endCameraRendering -= OnCameraRendering` 해제 필수 (메모리 누수 방지)
- Unity Perception Package 의존 없이 URP 네이티브 기능만 사용하여 버전 호환성 보장

## YAML Config 설계

### Visual+Vector 혼합 학습 템플릿

```yaml
# evader_s0_visual_template.yaml
behaviors:
  EvaderAgent:
    trainer_type: ppo
    hyperparameters:
      batch_size: 128           # 이미지 학습 시 더 큰 배치 권장
      buffer_size: 12800
      learning_rate: 3.0e-4
      beta: 5.0e-3
      epsilon: 0.2
      lambd: 0.99
      num_epoch: 3
      learning_rate_schedule: linear

    network_settings:
      normalize: true
      hidden_units: 256
      num_layers: 2
      vis_encode_type: simple   # simple | nature_cnn | resnet
      # simple: 2-layer CNN (빠름, 저해상도에 적합)
      # nature_cnn: 3-layer CNN (DQN 논문 기반, 범용)
      # resnet: ResNet 블록 (고해상도, 느림)

    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0

    max_steps: 1000000          # 이미지 학습은 수렴이 느려 스텝 증가
    time_horizon: 64
    summary_freq: 10000
    checkpoint_interval: 100000
    keep_checkpoints: 5
```

## Assembly Definition 구조

```
IIT_DroneLearning/Assets/00. BMW/DroneVisualPipeline/
├── Scripts/
│   ├── DroneVisualPipeline.asmdef
│   │   references: ["DroneCameraSystem", "DroneSensorSystem", "Unity.ML-Agents"]
│   │   rootNamespace: "DroneVisualPipeline"
│   │   // DroneSensorSystem: includeRayDistances=true 시 GetAllNormalizedDistances() 직접 호출용
│   ├── DroneVisionSystem.cs
│   ├── DroneSnapshotSystem.cs
│   ├── DroneDepthSystem.cs
│   ├── CaptureMetadata.cs          (메타데이터 직렬화 구조체)
│   └── Shaders/
│       └── DroneDepthVisualize.shader
├── Tests/
│   ├── DroneVisualPipelineTests.asmdef
│   ├── DroneVisionSystemTests.cs
│   ├── DroneSnapshotSystemTests.cs
│   └── DroneDepthSystemTests.cs
```

## 정확성 속성 (Correctness Properties)

### Property 1: RenderTexture 해상도 일치

*For any* DroneVisionSystem의 textureWidth/textureHeight 설정에 대해, 생성된 RenderTexture의 실제 해상도는 설정값과 정확히 일치해야 한다.

**Validates: 요구사항 1.2**

### Property 2: Visual Observation 비활성화 시 카메라 비활성

*For any* `enableVisualObservation = false` 상태에서, Observation 전용 카메라의 `enabled` 속성은 `false`여야 한다.

**Validates: 요구사항 1.6, 6.3**

### Property 3: 기존 카메라 무간섭

*For any* DroneVisionSystem이 부착된 드론에서, DroneCameraSystem의 Solo 카메라와 MultiView 카메라의 `targetTexture`는 `null`이어야 한다 (기존 Display 출력 유지).

**Validates: 요구사항 1.8, 8.2**

### Property 4: 메타데이터-이미지 파일명 대응

*For any* DroneSnapshotSystem이 저장한 파일 집합에서, 모든 `step_NNNN_rgb.png` 파일에 대해 동일한 `step_NNNN_meta.json` 파일이 존재해야 한다.

**Validates: 요구사항 4.1**

### Property 5: Depth 정규화 범위

*For any* DroneDepthSystem이 렌더링한 Depth Map의 모든 픽셀 값은 0.0 이상 1.0 이하여야 한다.

**Validates: 요구사항 5.3**

### Property 6: 에피소드 경계 스텝 초기화

*For any* 에피소드 전환 시, DroneSnapshotSystem의 내부 스텝 카운터는 0으로 초기화되고, 에피소드 카운터는 1 증가해야 한다.

**Validates: 요구사항 4.3**

### Property 7: 멀티모달 관측 공존

*For any* DroneVisionSystem이 활성화된 DroneAgent에서, CollectObservations()의 Vector Observation 차원은 여전히 44여야 한다 (Visual Observation은 별도 센서로 추가됨).

**Validates: 요구사항 1.4, 8.1**

### Property 8: OnValidate 초기화 전 안전성

*For any* `_isInitialized == false` 상태의 컴포넌트에서, OnValidate() 호출 시 어떠한 내부 상태 변경이나 예외도 발생하지 않아야 한다.

**Validates: 요구사항 6.2**

## 에러 처리

### DroneVisionSystem 에러 처리

| 상황 | 처리 방식 |
|------|----------|
| DroneCameraSystem 미존재 | `Debug.LogError` 출력, 컴포넌트 자동 비활성화 |
| RenderTexture 생성 실패 | `Debug.LogError` 출력, Visual Observation 비활성화 |
| ML-Agents RenderTextureSensorComponent 미존재 | `Debug.LogWarning` 출력, 수동 추가 안내 |

### DroneSnapshotSystem 에러 처리

| 상황 | 처리 방식 |
|------|----------|
| 저장 경로 쓰기 권한 없음 | `Debug.LogError` 출력, 캡처 자동 비활성화 |
| AsyncGPUReadback 실패 | 해당 프레임 건너뜀, `Debug.LogWarning` 출력 |
| 디스크 공간 부족 | 캡처 자동 중단, `Debug.LogError` 출력 |
| DroneAgent 미존재 | 메타데이터에서 reward/episode 필드를 기본값(0)으로 채움 |

### DroneDepthSystem 에러 처리

| 상황 | 처리 방식 |
|------|----------|
| URP Depth Texture 미활성화 | `Debug.LogError` 출력, URP Asset 설정 안내 |
| Depth 셰이더 로드 실패 | `Debug.LogError` 출력, Depth 센서 비활성화 |
| DroneCameraSystem 미존재 | 독립 카메라 생성, `Debug.LogWarning` 출력 |

## 테스트 전략

### 단위 테스트

- DroneVisionSystem: RenderTexture 생성/해상도 검증, 활성/비활성 토글, 기존 카메라 무간섭
- DroneSnapshotSystem: 파일 저장 경로 생성, 메타데이터 직렬화/역직렬화, 에피소드 카운터 동작
- DroneDepthSystem: Depth RenderTexture 생성, 셰이더 로드 검증, 활성/비활성 토글

### 통합 테스트

- DroneVisionSystem + DroneAgent: Visual Observation이 ML-Agents 학습 루프에 정상 전달되는지 확인
- DroneSnapshotSystem + DroneVisionSystem: 캡처된 이미지가 Visual Observation과 동일한 시점인지 확인
- DroneDepthSystem + DroneVisionSystem: RGB + Depth 동시 관측 시 프레임레이트 유지 확인

### 테스트 파일 위치

```
IIT_DroneLearning/Assets/00. BMW/DroneVisualPipeline/Tests/
├── DroneVisualPipelineTests.asmdef
├── DroneVisionSystemTests.cs
├── DroneSnapshotSystemTests.cs
└── DroneDepthSystemTests.cs
```
