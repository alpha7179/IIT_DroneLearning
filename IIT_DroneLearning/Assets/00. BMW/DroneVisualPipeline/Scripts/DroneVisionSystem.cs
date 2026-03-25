using UnityEngine;
using UnityEngine.Rendering.Universal;
using Unity.MLAgents.Sensors;
using DroneCamera;

namespace DroneVisualPipeline
{
    /// <summary>
    /// DroneCameraSystem의 Solo 카메라 렌더링을 RenderTexture에 캡처하여
    /// ML-Agents Visual Observation으로 전달하는 브릿지 컴포넌트.
    ///
    /// 설계 원칙:
    ///   - DroneCamera_Solo GameObject에 AddComponent&lt;Camera&gt;()로 Observation 전용 카메라 추가
    ///   - 동일 Transform 공유 → DroneCameraSystem LateUpdate() 위치 갱신 자동 추종
    ///   - DroneCameraSystem Solo 카메라의 targetTexture == null 유지 (Display 출력 보존)
    ///   - Start()에서 초기화 (Awake 대신) — DroneCameraSystem.Awake() 완료 후 실행 보장
    ///   - 기존 DroneAgent, DroneCameraSystem, DroneSensorSystem 코드 수정 없음
    /// </summary>
    [RequireComponent(typeof(DroneCameraSystem))]
    [DefaultExecutionOrder(-200)]
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

        // ────────────────────────────────────────────
        // 내부 상태
        // ────────────────────────────────────────────

        private RenderTexture _renderTexture;
        private DroneCameraSystem _cameraSystem;
        private Camera _observationCamera;
        private bool _isInitialized;
        private int _lastWidth;
        private int _lastHeight;
        private bool _lastGrayscale;

        // ────────────────────────────────────────────
        // 공개 프로퍼티 (테스트·외부 접근용)
        // ────────────────────────────────────────────

        /// <summary>현재 Observation 전용 카메라 (초기화 후 유효)</summary>
        public Camera ObservationCamera => _observationCamera;

        /// <summary>현재 RenderTexture (초기화 후 유효)</summary>
        public RenderTexture RenderTexture => _renderTexture;

        /// <summary>초기화 완료 여부</summary>
        public bool IsInitialized => _isInitialized;

        // ────────────────────────────────────────────
        // Unity 생명주기
        // ────────────────────────────────────────────

        private void Awake()
        {
            // ML-Agents Agent.Initialize()가 Start() 이전에 센서를 생성하므로,
            // RenderTextureSensorComponent에 임시 RT를 미리 할당하여
            // Texture2D(0,0) 생성 에러를 방지한다.
            var rtSensor = FindSensorComponent(sensorName);
            if (rtSensor != null && rtSensor.RenderTexture == null)
            {
                _renderTexture = CreateRenderTexture(textureWidth, textureHeight);
                rtSensor.RenderTexture = _renderTexture;
                rtSensor.SensorName    = sensorName;
                rtSensor.Grayscale     = grayscale;
            }
        }

        private void Start()
        {
            InitializeSystem();
        }

        private void OnDestroy()
        {
            CleanupRenderTexture();
        }

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
            if (_observationCamera != null && _cameraSystem != null)
            {
                _observationCamera.fieldOfView = fovOverride > 0f
                    ? fovOverride
                    : _cameraSystem.fieldOfView;
                _observationCamera.farClipPlane = farClipOverride > 0f
                    ? farClipOverride
                    : _cameraSystem.farClipPlane;
            }

            // 4. Grayscale 변경 시에만 런타임 경고 (ML-Agents 센서는 채널 수 변경 불가)
            if (Application.isPlaying && grayscale != _lastGrayscale)
            {
                Debug.LogWarning(
                    "[DroneVisionSystem] grayscale 변경은 플레이 모드에서 즉시 적용되지 않습니다. " +
                    "에피소드 재시작 후 적용됩니다.", this);
                _lastGrayscale = grayscale;
            }
        }

        // ────────────────────────────────────────────
        // 디버그 시각화
        // ────────────────────────────────────────────

        private void OnDrawGizmos()
        {
            if (!showDebugFrustum || _observationCamera == null) return;

            Gizmos.color = new Color(0f, 1f, 0.5f, 0.5f);
            Gizmos.matrix = _observationCamera.transform.localToWorldMatrix;
            Gizmos.DrawFrustum(
                Vector3.zero,
                _observationCamera.fieldOfView,
                _observationCamera.farClipPlane,
                _observationCamera.nearClipPlane,
                _observationCamera.aspect);
        }

        private void OnGUI()
        {
            if (!showPreviewOverlay || _renderTexture == null) return;

            int w = (int)(Screen.width * previewSize);
            int h = (int)(Screen.height * previewSize);
            int x = 0;
            int y = Screen.height - h;

            GUI.DrawTexture(new Rect(x, y, w, h), _renderTexture);

            GUI.Label(new Rect(x + 2, y + 2, 120, 20),
                $"RGB {textureWidth}x{textureHeight}");
        }

        // ────────────────────────────────────────────
        // 초기화
        // ────────────────────────────────────────────

        private void InitializeSystem()
        {
            // 1. DroneCameraSystem 참조 획득
            _cameraSystem = GetComponent<DroneCameraSystem>();
            if (_cameraSystem == null)
            {
                Debug.LogError(
                    "[DroneVisionSystem] DroneCameraSystem을 찾을 수 없습니다. 컴포넌트를 비활성화합니다.", this);
                enabled = false;
                return;
            }

            Camera soloCamera = _cameraSystem.Camera;
            if (soloCamera == null)
            {
                Debug.LogError(
                    "[DroneVisionSystem] DroneCameraSystem.Camera (Solo)가 null입니다. 컴포넌트를 비활성화합니다.", this);
                enabled = false;
                return;
            }

            // 2. Observation 전용 child GameObject 생성 후 Camera 추가
            //    (DroneCamera_Solo에는 Camera가 이미 존재하므로 자식 오브젝트에 별도 생성)
            //    동일 parent Transform → LateUpdate 위치 갱신 자동 추종
            var obsGO = new GameObject("DroneVisionCamera");
            obsGO.transform.SetParent(soloCamera.transform, false);
            _observationCamera = obsGO.AddComponent<Camera>();

            // 3. RenderTexture 생성
            _renderTexture = CreateRenderTexture(textureWidth, textureHeight);
            if (_renderTexture == null)
            {
                Debug.LogError("[DroneVisionSystem] RenderTexture 생성에 실패했습니다. Visual Observation 비활성화.", this);
                enabled = false;
                return;
            }

            // 4. Observation 카메라 설정 = Solo 카메라 복제
            _observationCamera.fieldOfView   = fovOverride > 0f ? fovOverride : soloCamera.fieldOfView;
            _observationCamera.nearClipPlane = soloCamera.nearClipPlane;
            _observationCamera.farClipPlane  = farClipOverride > 0f ? farClipOverride : soloCamera.farClipPlane;
            _observationCamera.clearFlags    = soloCamera.clearFlags;
            _observationCamera.backgroundColor = soloCamera.backgroundColor;
            _observationCamera.cullingMask   = soloCamera.cullingMask;
            // Display 8(index=7)로 격리 — Display 1은 기존 MultiView 카메라가 사용
            // targetTexture가 설정되므로 실제 Display 출력은 없지만,
            // 기본값 targetDisplay=0(Display 1)을 유지하면 MultiView와 충돌할 수 있다.
            _observationCamera.targetDisplay = 7;

            // 4-1. URP TAA/포스트프로세싱 비활성화 (단일 프레임 캡처 떨림 방지)
            //      TAA는 시간 누적으로 AA를 수행하므로 개별 프레임에 sub-pixel jitter만 남아
            //      스틸 이미지/ML-Agents Observation에서 shimmer 아티팩트가 발생한다.
            var urpCameraData = _observationCamera.GetUniversalAdditionalCameraData();
            if (urpCameraData != null)
            {
                urpCameraData.antialiasing        = AntialiasingMode.None;
                urpCameraData.renderPostProcessing = false;
            }

            // 5. Observation 카메라의 targetTexture = RenderTexture
            _observationCamera.targetTexture = _renderTexture;

            // 6. Solo 카메라의 targetTexture == null 확인 (Display 출력 보존)
            if (soloCamera.targetTexture != null)
            {
                Debug.LogWarning(
                    "[DroneVisionSystem] DroneCameraSystem Solo 카메라의 targetTexture가 null이 아닙니다. " +
                    "기존 Display 출력이 유지되지 않을 수 있습니다.", this);
            }

            // 7. 활성화 상태 반영
            _observationCamera.enabled = enableVisualObservation;

            // 8. RenderTextureSensorComponent 연동
            var rtSensor = FindSensorComponent(sensorName);
            if (rtSensor == null)
            {
                Debug.LogWarning(
                    "[DroneVisionSystem] RenderTextureSensorComponent가 없습니다. " +
                    "같은 GameObject에 수동으로 추가하고 RenderTexture를 할당해주세요.", this);
            }
            else
            {
                rtSensor.RenderTexture = _renderTexture;
                rtSensor.SensorName    = sensorName;
                rtSensor.Grayscale     = grayscale;
            }

            _lastWidth    = textureWidth;
            _lastHeight   = textureHeight;
            _lastGrayscale = grayscale;
            _isInitialized = true;

            Debug.Log(
                $"[DroneVisionSystem] 초기화 완료 — {textureWidth}x{textureHeight}, " +
                $"grayscale={grayscale}, sensor='{sensorName}'", this);
        }

        // ────────────────────────────────────────────
        // 내부 헬퍼
        // ────────────────────────────────────────────

        private RenderTexture CreateRenderTexture(int width, int height)
        {
            var rt = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32);
            rt.name = $"DroneVision_RT_{width}x{height}";
            rt.Create();
            return rt.IsCreated() ? rt : null;
        }

        private void RecreateRenderTexture()
        {
            CleanupRenderTexture();
            _renderTexture = CreateRenderTexture(textureWidth, textureHeight);

            if (_renderTexture == null)
            {
                Debug.LogError("[DroneVisionSystem] RenderTexture 재생성에 실패했습니다.", this);
                return;
            }

            if (_observationCamera != null)
                _observationCamera.targetTexture = _renderTexture;

            var rtSensor = FindSensorComponent(sensorName);
            if (rtSensor != null)
                rtSensor.RenderTexture = _renderTexture;
        }

        private void CleanupRenderTexture()
        {
            if (_renderTexture != null)
            {
                if (_observationCamera != null)
                    _observationCamera.targetTexture = null;

                _renderTexture.Release();
                if (Application.isPlaying)
                    Destroy(_renderTexture);
                else
                    DestroyImmediate(_renderTexture);

                _renderTexture = null;
            }
        }

        /// <summary>
        /// sensorName과 일치하는 RenderTextureSensorComponent를 반환한다.
        /// DroneVisionSystem / DroneDepthSystem이 동일 GameObject에 공존할 때
        /// GetComponent&lt;T&gt;()가 첫 번째 것만 반환하는 충돌을 방지하기 위해
        /// GetComponents&lt;T&gt;()로 전체를 조회하여 적합한 것을 선택한다.
        /// 우선순위: ① sensorName 일치 ② RenderTexture가 미할당된 것 ③ 첫 번째
        /// </summary>
        private RenderTextureSensorComponent FindSensorComponent(string targetName)
        {
            var sensors = GetComponents<RenderTextureSensorComponent>();
            if (sensors.Length == 0) return null;

            // 1순위: 이미 같은 sensorName이 설정된 것 (재초기화 시 동일 컴포넌트 재사용)
            foreach (var s in sensors)
                if (s.SensorName == targetName) return s;

            // 2순위: RenderTexture가 아직 할당되지 않은 빈 컴포넌트
            foreach (var s in sensors)
                if (s.RenderTexture == null) return s;

            // 3순위: 첫 번째 (폴백)
            return sensors[0];
        }
    }
}
