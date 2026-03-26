using UnityEngine;
using UnityEngine.Rendering;
using Unity.MLAgents.Sensors;
using DroneCamera;

namespace DroneVisualPipeline
{
    /// <summary>
    /// URP 네이티브 Depth 텍스처를 활용한 Depth Map 보조 센서.
    ///
    /// 설계 원칙:
    ///   - Camera.SetReplacementShader() : Built-in Pipeline 전용, URP 미지원 → 사용 금지
    ///   - RenderPipelineManager.endCameraRendering + Graphics.Blit 방식 사용 (URP 네이티브)
    ///   - DroneCamera_Solo GameObject에 AddComponent&lt;Camera&gt;()로 Depth 전용 카메라 추가
    ///   - depthTextureMode = DepthTextureMode.Depth → URP _CameraDepthTexture 자동 생성
    ///   - OnDestroy()에서 이벤트 해제 필수 (메모리 누수 방지)
    ///   - Start()에서 초기화 (DroneCameraSystem.Awake() 완료 후 실행 보장)
    ///   - 기존 DroneAgent, DroneCameraSystem, DroneSensorSystem 코드 수정 없음
    /// </summary>
    [RequireComponent(typeof(DroneCameraSystem))]
    [DefaultExecutionOrder(-200)]
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

        // ────────────────────────────────────────────
        // 내부 상태
        // ────────────────────────────────────────────

        private Camera          _depthCamera;
        private RenderTexture   _depthRT;
        private Material        _depthMaterial;
        private DroneCameraSystem _cameraSystem;
        private bool            _isInitialized;
        private int             _lastWidth;
        private int             _lastHeight;

        // ────────────────────────────────────────────
        // 공개 프로퍼티 (테스트·외부 접근용)
        // ────────────────────────────────────────────

        /// <summary>현재 Depth 전용 카메라 (초기화 후 유효)</summary>
        public Camera DepthCamera => _depthCamera;

        /// <summary>현재 Depth RenderTexture (초기화 후 유효)</summary>
        public RenderTexture DepthRenderTexture => _depthRT;

        /// <summary>현재 Depth 셰이더 Material (초기화 후 유효)</summary>
        public Material DepthMaterial => _depthMaterial;

        /// <summary>초기화 완료 여부</summary>
        public bool IsInitialized => _isInitialized;

        // ────────────────────────────────────────────
        // Unity 생명주기
        // ────────────────────────────────────────────

        private void Awake()
        {
            // ML-Agents Agent.OnEnable() → LazyInitialize() → CreateSensors()가
            // Start()보다 먼저 실행되므로, Awake에서 임시 RT를 미리 할당하여
            // Texture2D(0,0) 생성 에러를 방지한다.
            var rtSensor = FindSensorComponent(sensorName);
            if (rtSensor != null && rtSensor.RenderTexture == null)
            {
                _depthRT = CreateDepthRenderTexture(depthWidth, depthHeight);
                rtSensor.RenderTexture = _depthRT;
                rtSensor.SensorName    = sensorName;
                rtSensor.Grayscale     = true;
            }
        }

        private void Start()
        {
            InitializeSystem();
        }

        private void OnDestroy()
        {
            // RenderPipelineManager 이벤트 해제 (메모리 누수 방지)
            RenderPipelineManager.endCameraRendering -= OnCameraRendering;
            CleanupDepthRenderTexture();
        }

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
                _lastWidth  = depthWidth;
                _lastHeight = depthHeight;
            }

            // 3. Near/Far Clip 오버라이드 즉시 반영
            if (_depthCamera != null && _cameraSystem != null)
            {
                _depthCamera.nearClipPlane = nearClipOverride > 0f
                    ? nearClipOverride
                    : _cameraSystem.nearClipPlane;
                _depthCamera.farClipPlane = farClipOverride > 0f
                    ? farClipOverride
                    : _cameraSystem.farClipPlane;
            }

            // 4. Depth 셰이더 파라미터 즉시 반영
            UpdateShaderParameters();

            // 5. maxDepthDistance 범위 클램핑
            maxDepthDistance = Mathf.Max(0f, maxDepthDistance);
        }

        // ────────────────────────────────────────────
        // 디버그 시각화
        // ────────────────────────────────────────────

        private void OnDrawGizmos()
        {
            if (!showDebugFrustum || _depthCamera == null) return;

            Gizmos.color = new Color(0f, 0.5f, 1f, 0.5f);
            Gizmos.matrix = _depthCamera.transform.localToWorldMatrix;
            Gizmos.DrawFrustum(
                Vector3.zero,
                _depthCamera.fieldOfView,
                _depthCamera.farClipPlane,
                _depthCamera.nearClipPlane,
                _depthCamera.aspect);
        }

        private void OnGUI()
        {
            if (!showPreviewOverlay || _depthRT == null) return;

            int w = (int)(Screen.width  * previewSize);
            int h = (int)(Screen.height * previewSize);
            int x = 0;
            int y = Screen.height - h - (int)(Screen.height * 0.22f); // VisionSystem 위에 표시

            GUI.DrawTexture(new Rect(x, y, w, h), _depthRT);
            GUI.Label(new Rect(x + 2, y + 2, 120, 20),
                $"Depth {depthWidth}x{depthHeight}");
        }

        // ────────────────────────────────────────────
        // RenderPipeline 콜백 (핵심 Depth 렌더링)
        // ────────────────────────────────────────────

        private void OnCameraRendering(ScriptableRenderContext context, Camera cam)
        {
            if (!_isInitialized) return;
            if (cam != _depthCamera) return;
            if (!enableDepthSensor) return;
            if (_depthRT == null || _depthMaterial == null) return;

            // Depth 셰이더 적용: _CameraDepthTexture → 정규화 → _depthRT 출력
            Graphics.Blit(null, _depthRT, _depthMaterial);
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
                    "[DroneDepthSystem] DroneCameraSystem을 찾을 수 없습니다. 독립 카메라로 폴백합니다.", this);
                // 독립 카메라 폴백 (Solo 카메라 없이 드론 Transform에 직접 붙임)
                InitializeDepthCameraStandalone();
            }
            else
            {
                Camera soloCamera = _cameraSystem.Camera;
                if (soloCamera == null)
                {
                    Debug.LogError(
                        "[DroneDepthSystem] DroneCameraSystem.Camera (Solo)가 null입니다.", this);
                    enabled = false;
                    return;
                }

                // 2. Depth 전용 child GameObject 생성 후 Camera 추가
                //    (DroneCamera_Solo에는 Camera가 이미 존재하므로 자식 오브젝트에 별도 생성)
                var depthGO = new GameObject("DroneDepthCamera");
                depthGO.transform.SetParent(soloCamera.transform, false);
                _depthCamera = depthGO.AddComponent<Camera>();
                _depthCamera.fieldOfView   = soloCamera.fieldOfView;
                _depthCamera.nearClipPlane = nearClipOverride > 0f ? nearClipOverride : soloCamera.nearClipPlane;
                _depthCamera.farClipPlane  = farClipOverride > 0f  ? farClipOverride  : soloCamera.farClipPlane;
                _depthCamera.clearFlags    = CameraClearFlags.Depth;
                _depthCamera.cullingMask   = soloCamera.cullingMask;
                // Display 8(index=7)로 격리 — Display 1은 기존 MultiView 카메라가 사용
                // targetTexture 없이 ClearFlags.Depth이므로 설정하지 않으면 Display 1 위에 렌더링됨
                _depthCamera.targetDisplay = 7;
            }

            // 3. depthTextureMode 설정 → URP _CameraDepthTexture 자동 생성
            _depthCamera.depthTextureMode = DepthTextureMode.Depth;
            _depthCamera.enabled = enableDepthSensor;

            // 4. 커스텀 Depth 셰이더 로드 및 Material 생성
            var depthShader = Shader.Find("DroneVisualPipeline/DroneDepthVisualize");
            if (depthShader == null)
            {
                Debug.LogError(
                    "[DroneDepthSystem] DroneDepthVisualize 셰이더를 찾을 수 없습니다. " +
                    "Shaders 폴더에 DroneDepthVisualize.shader가 있는지 확인하세요.", this);
                enabled = false;
                return;
            }
            _depthMaterial = new Material(depthShader);
            _depthMaterial.name = "DroneDepthVisualize_Mat";

            // 5. Depth RenderTexture 생성 (RFloat: 단채널 부동소수점, Depth 저장에 적합)
            _depthRT = CreateDepthRenderTexture(depthWidth, depthHeight);
            if (_depthRT == null)
            {
                Debug.LogError(
                    "[DroneDepthSystem] Depth RenderTexture 생성에 실패했습니다.", this);
                enabled = false;
                return;
            }

            // 6. 셰이더 파라미터 초기 설정
            UpdateShaderParameters();

            // 7. RenderPipelineManager.endCameraRendering 이벤트 등록
            RenderPipelineManager.endCameraRendering += OnCameraRendering;

            // 8. RenderTextureSensorComponent 연동
            // GetComponents<>()로 전체 조회 → sensorName 기반 매칭 (DroneVisionSystem 충돌 방지)
            var rtSensor = FindSensorComponent(sensorName);
            if (rtSensor == null)
            {
                Debug.LogWarning(
                    "[DroneDepthSystem] RenderTextureSensorComponent가 없습니다. " +
                    "같은 GameObject에 수동으로 추가하고 RenderTexture를 할당해주세요.", this);
            }
            else
            {
                rtSensor.RenderTexture = _depthRT;
                rtSensor.SensorName    = sensorName;
                rtSensor.Grayscale     = true;  // Depth는 항상 Grayscale
            }

            _lastWidth  = depthWidth;
            _lastHeight = depthHeight;
            _isInitialized = true;

            Debug.Log(
                $"[DroneDepthSystem] 초기화 완료 — {depthWidth}x{depthHeight}, " +
                $"mode={depthMode}, sensor='{sensorName}'", this);
        }

        private void InitializeDepthCameraStandalone()
        {
            // DroneCameraSystem 없이 드론 Transform에 직접 Camera 추가
            _depthCamera = gameObject.AddComponent<Camera>();
            _depthCamera.fieldOfView   = 60f;
            _depthCamera.nearClipPlane = 0.1f;
            _depthCamera.farClipPlane  = 500f;
            _depthCamera.clearFlags    = CameraClearFlags.Depth;
            _depthCamera.targetDisplay = 7;
        }

        // ────────────────────────────────────────────
        // 내부 헬퍼
        // ────────────────────────────────────────────

        private RenderTexture CreateDepthRenderTexture(int width, int height)
        {
            // RFloat: 단채널 32비트 부동소수점 (Depth 값 저장에 최적)
            var rt = new RenderTexture(width, height, 0, RenderTextureFormat.RFloat);
            rt.name = $"DroneDepth_RT_{width}x{height}";
            rt.Create();
            return rt.IsCreated() ? rt : null;
        }

        private void RecreateDepthRenderTexture()
        {
            CleanupDepthRenderTexture();
            _depthRT = CreateDepthRenderTexture(depthWidth, depthHeight);

            if (_depthRT == null)
            {
                Debug.LogError("[DroneDepthSystem] Depth RenderTexture 재생성에 실패했습니다.", this);
                return;
            }

            var rtSensor = FindSensorComponent(sensorName);
            if (rtSensor != null)
                rtSensor.RenderTexture = _depthRT;
        }

        private void CleanupDepthRenderTexture()
        {
            if (_depthRT != null)
            {
                _depthRT.Release();
                if (Application.isPlaying)
                    Destroy(_depthRT);
                else
                    DestroyImmediate(_depthRT);
                _depthRT = null;
            }
        }

        private void UpdateShaderParameters()
        {
            if (_depthMaterial == null) return;

            float maxDist = maxDepthDistance > 0f
                ? maxDepthDistance
                : (_depthCamera != null ? _depthCamera.farClipPlane : 500f);

            _depthMaterial.SetFloat("_MaxDistance",     maxDist);
            _depthMaterial.SetInt("_UseLinear",         depthMode == DepthOutputMode.Linear ? 1 : 0);
            _depthMaterial.SetInt("_UseJetColorRamp",   colorRamp == DepthColorRamp.Jet ? 1 : 0);
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

            foreach (var s in sensors)
                if (s.SensorName == targetName) return s;

            foreach (var s in sensors)
                if (s.RenderTexture == null) return s;

            return sensors[0];
        }
    }

    /// <summary>Depth 출력 모드</summary>
    public enum DepthOutputMode { Linear, Raw }

    /// <summary>Depth 시각화 컬러 램프 모드</summary>
    public enum DepthColorRamp { Grayscale, Jet }
}
