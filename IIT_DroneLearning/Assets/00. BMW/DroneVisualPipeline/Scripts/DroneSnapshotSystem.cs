using System;
using System.IO;
using System.Reflection;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Rendering;
using DroneCamera;
using DroneSensor;

namespace DroneVisualPipeline
{
    /// <summary>
    /// 런타임 카메라 렌더링을 PNG/JPG + 메타데이터 JSON으로 저장하는 데이터 추출 컴포넌트.
    ///
    /// 설계 원칙:
    ///   - AsyncGPUReadback → 비동기 GPU→CPU 전송 (메인 스레드 차단 없음)
    ///   - Task.Run() → 파일 I/O 백그라운드 스레드 처리
    ///   - DroneAgent 데이터는 리플렉션으로 수집 (Assembly-CSharp 직접 참조 불가)
    ///   - 기존 DroneAgent, DroneCameraSystem, DroneSensorSystem 코드 수정 없음
    /// </summary>
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
        public string basePath = "Assets/00. BMW/DroneVisualPipeline/CapturedData";

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

        [Header("플레이 모드 종료")]
        [Tooltip("플레이 모드 종료 시 캡처 데이터를 자동으로 삭제한다")]
        public bool deleteOnPlayModeExit = false;

        [Header("디버그")]
        [Tooltip("캡처 시 Console에 파일 경로 로그 출력")]
        public bool logCaptureEvents = false;

        [Tooltip("현재 에피소드/스텝 카운터를 Scene 뷰에 표시")]
        public bool showCaptureStatus = false;

        #endregion

        // ────────────────────────────────────────────
        // 내부 상태
        // ────────────────────────────────────────────

        private int    _episodeCount = 0;
        private int    _stepCount    = 0;
        private string _currentEpisodePath;
        private string _resolvedPrefix;
        private string _startTime;

        private Texture2D     _readbackTexture;  // GC 방지를 위해 재사용
        private int           _pendingRequests   = 0;
        private int           _capturedImageCount = 0;
        private float         _lastCumulativeReward = 0f;

        // DroneAgent 리플렉션 캐시 (Assembly-CSharp 직접 참조 불가)
        private MonoBehaviour  _droneAgent;
        private FieldInfo      _rewardField;
        private FieldInfo      _cumulativeRewardField;

        // Rigidbody 캐시 — FixedUpdate GC 방지 (요구사항 7.4)
        private Rigidbody      _rigidbody;

        // DroneSensorSystem 캐시 (includeRayDistances)
        private DroneSensorSystem _sensorSystem;

        // DroneVisionSystem 캐시 (캡처 소스 RT)
        private DroneVisionSystem _visionSystem;

        // DroneDepthSystem 캐시 (Depth 캡처)
        private DroneDepthSystem _depthSystem;

        // ────────────────────────────────────────────
        // 공개 프로퍼티 (테스트·외부 접근용)
        // ────────────────────────────────────────────

        public int  EpisodeCount        => _episodeCount;
        public int  StepCount           => _stepCount;
        public int  CapturedImageCount  => _capturedImageCount;
        public string CurrentEpisodePath => _currentEpisodePath;

        // ────────────────────────────────────────────
        // Unity 생명주기
        // ────────────────────────────────────────────

        private void Start()
        {
            CacheComponents();
            ResolveFilePrefix();
            BeginNewEpisode();
        }

        private void FixedUpdate()
        {
            if (!enableCapture) return;

            _stepCount++;

            if (_stepCount % captureInterval != 0) return;
            if (_pendingRequests >= maxAsyncRequests) return;

            if (captureRGB && _visionSystem != null && _visionSystem.RenderTexture != null)
                RequestCapture(_visionSystem.RenderTexture, isCaptureDepth: false);

            if (captureDepth)
            {
                if (_depthSystem == null)
                {
                    if (logCaptureEvents)
                        Debug.LogWarning("[DroneSnapshotSystem] captureDepth=true이지만 DroneDepthSystem이 없습니다.", this);
                }
                else if (_depthSystem.DepthRenderTexture != null)
                {
                    RequestCapture(_depthSystem.DepthRenderTexture, isCaptureDepth: true);
                }
            }
        }

        private void OnDestroy()
        {
            CleanupReadbackTexture();
        }

        private void OnGUI()
        {
            if (!showCaptureStatus) return;

            string status = enableCapture
                ? $"Ep:{_episodeCount} Step:{_stepCount}\nCaptured:{_capturedImageCount}"
                : $"Ep:{_episodeCount} (캡처 비활성)";

            GUI.Box(new Rect(Screen.width - 165, 10, 155, 45), status);
        }

        private void OnValidate()
        {
            // 1. 값 범위 클램핑
            captureInterval    = Mathf.Clamp(captureInterval, 1, 100);
            maxAsyncRequests   = Mathf.Clamp(maxAsyncRequests, 1, 10);
            jpgQuality         = Mathf.Clamp(jpgQuality, 1, 100);
            captureWidth       = Mathf.Max(0, captureWidth);
            captureHeight      = Mathf.Max(0, captureHeight);

            // 2. 캡처 해상도 변경 시 readback 텍스처 재생성 (플레이 모드)
            if (Application.isPlaying && _readbackTexture != null)
            {
                int targetW = GetSourceWidth();
                int targetH = GetSourceHeight();
                if (_readbackTexture.width != targetW || _readbackTexture.height != targetH)
                    RecreateReadbackTexture(targetW, targetH);
            }

            // 3. basePath 유효성 검사
            if (string.IsNullOrWhiteSpace(basePath))
                basePath = "Assets/00. BMW/DroneVisualPipeline/CapturedData";

            // 4. captureDepth: DroneDepthSystem 미존재 시 경고 (플레이 모드)
            if (Application.isPlaying && captureDepth && _depthSystem == null)
            {
                Debug.LogWarning(
                    "[DroneSnapshotSystem] captureDepth=true이지만 DroneDepthSystem이 없습니다. " +
                    "captureDepth를 자동 비활성화합니다.", this);
                captureDepth = false;
            }
        }

        // ────────────────────────────────────────────
        // 공개 API — ML-Agents / 외부 호출
        // ────────────────────────────────────────────

        /// <summary>
        /// 에피소드 시작 시 호출 — 에피소드 카운터 증가, 스텝 초기화.
        /// DroneAgent의 OnEpisodeBegin() 또는 Start()에서 호출하거나
        /// DroneSnapshotSystem이 자동 탐지한다.
        /// </summary>
        public void OnEpisodeBegin()
        {
            if (_episodeCount > 0)
                SaveEpisodeSummary("unknown");  // 이전 에피소드 요약 저장

            BeginNewEpisode();
        }

        /// <summary>
        /// 에피소드 종료 시 호출 — 요약 JSON 저장.
        /// </summary>
        public void OnEpisodeEnd(string terminationReason = "unknown")
        {
            SaveEpisodeSummary(terminationReason);
        }

        // ────────────────────────────────────────────
        // 비동기 캡처 파이프라인
        // ────────────────────────────────────────────

        private void RequestCapture(RenderTexture source, bool isCaptureDepth)
        {
            if (source == null || !source.IsCreated()) return;

            int episode = _episodeCount;
            int step    = _stepCount;

            // 메타데이터를 캡처 시점에서 수집 (FixedUpdate 프레임)
            CaptureMetadata meta = saveMetadata ? CollectMetadata(episode, step) : null;

            _pendingRequests++;
            AsyncGPUReadback.Request(source, 0, request =>
            {
                _pendingRequests = Mathf.Max(0, _pendingRequests - 1);

                if (request.hasError)
                {
                    Debug.LogWarning(
                        $"[DroneSnapshotSystem] AsyncGPUReadback 실패 — episode={episode}, step={step}", this);
                    return;
                }

                int w = source.width;
                int h = source.height;

                // Readback 텍스처 준비 (재사용)
                if (_readbackTexture == null ||
                    _readbackTexture.width != w || _readbackTexture.height != h)
                    RecreateReadbackTexture(w, h);

                var data = request.GetData<byte>();
                _readbackTexture.LoadRawTextureData(data);
                _readbackTexture.Apply();

                // 인코딩
                byte[] imageBytes = imageFormat == ImageFormat.PNG
                    ? _readbackTexture.EncodeToPNG()
                    : _readbackTexture.EncodeToJPG(jpgQuality);

                string ext      = imageFormat == ImageFormat.PNG ? "png" : "jpg";
                string suffix   = isCaptureDepth ? "depth" : "rgb";
                string imgPath  = Path.Combine(_currentEpisodePath,
                    $"step_{step:D4}_{suffix}.{ext}");
                string metaPath = saveMetadata
                    ? Path.Combine(_currentEpisodePath, $"step_{step:D4}_meta.json")
                    : null;
                string metaJson = meta != null ? JsonUtility.ToJson(meta, true) : null;

                _capturedImageCount++;

                // 백그라운드 스레드에서 파일 저장
                Task.Run(() =>
                {
                    try
                    {
                        File.WriteAllBytes(imgPath, imageBytes);

                        if (metaPath != null && metaJson != null)
                            File.WriteAllText(metaPath, metaJson);

                        if (logCaptureEvents)
                            Debug.Log($"[DroneSnapshotSystem] 저장: {imgPath}", this);
                    }
                    catch (IOException ex)
                    {
                        Debug.LogError(
                            $"[DroneSnapshotSystem] 파일 저장 실패: {ex.Message}\n캡처를 비활성화합니다.", this);
                        enableCapture = false;
                    }
                    catch (Exception ex)
                    {
                        Debug.LogError($"[DroneSnapshotSystem] 저장 오류: {ex.Message}", this);
                    }
                });
            });
        }

        // ────────────────────────────────────────────
        // 메타데이터 수집
        // ────────────────────────────────────────────

        private CaptureMetadata CollectMetadata(int episode, int step)
        {
            float reward           = GetReward();
            float cumulativeReward = GetCumulativeReward();
            _lastCumulativeReward  = cumulativeReward;

            var meta = new CaptureMetadata
            {
                timestamp          = DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffZ"),
                episode            = episode,
                step               = step,
                droneRole          = _resolvedPrefix,
                dronePosition      = transform.position,
                droneRotation      = transform.eulerAngles,
                droneVelocity      = GetVelocity(),
                targetRelativePosition = GetTargetRelativePosition(),
                reward             = reward,
                cumulativeReward   = cumulativeReward,
            };

            if (includeRayDistances && _sensorSystem != null)
                meta.sensorDistances = _sensorSystem.GetAllNormalizedDistances();
            else
                meta.sensorDistances = Array.Empty<float>();

            return meta;
        }

        // ────────────────────────────────────────────
        // 에피소드 관리
        // ────────────────────────────────────────────

        private void BeginNewEpisode()
        {
            _episodeCount++;
            _stepCount            = 0;
            _capturedImageCount   = 0;
            _lastCumulativeReward = 0f;
            _startTime            = DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffZ");

            _currentEpisodePath = Path.Combine(
                Application.dataPath.Replace("Assets", ""),
                basePath,
                $"{_resolvedPrefix}_episode_{_episodeCount:D4}");

            try
            {
                Directory.CreateDirectory(_currentEpisodePath);
            }
            catch (Exception ex)
            {
                Debug.LogError(
                    $"[DroneSnapshotSystem] 에피소드 폴더 생성 실패: {ex.Message}\n캡처를 비활성화합니다.", this);
                enableCapture = false;
            }
        }

        private void SaveEpisodeSummary(string terminationReason)
        {
            if (!saveEpisodeSummary) return;
            if (string.IsNullOrEmpty(_currentEpisodePath)) return;

            var summary = new EpisodeSummary
            {
                episode            = _episodeCount,
                droneRole          = _resolvedPrefix,
                totalSteps         = _stepCount,
                cumulativeReward   = _lastCumulativeReward,
                terminationReason  = terminationReason,
                capturedImageCount = _capturedImageCount,
                startTime          = _startTime,
                endTime            = DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffZ"),
            };

            string path = Path.Combine(_currentEpisodePath, "episode_summary.json");
            string json = JsonUtility.ToJson(summary, true);

            Task.Run(() =>
            {
                try { File.WriteAllText(path, json); }
                catch (Exception ex)
                {
                    Debug.LogError($"[DroneSnapshotSystem] 요약 저장 실패: {ex.Message}", this);
                }
            });
        }

        // ────────────────────────────────────────────
        // 컴포넌트 캐싱 & 리플렉션
        // ────────────────────────────────────────────

        private void CacheComponents()
        {
            _visionSystem  = GetComponent<DroneVisionSystem>();
            _depthSystem   = GetComponent<DroneDepthSystem>();
            _sensorSystem  = GetComponent<DroneSensorSystem>();
            _rigidbody     = GetComponent<Rigidbody>();  // GC 방지를 위해 캐시

            // DroneAgent 리플렉션 (Assembly-CSharp 직접 참조 불가)
            foreach (var comp in GetComponents<MonoBehaviour>())
            {
                if (comp == null || comp.GetType().Name != "DroneAgent") continue;

                _droneAgent = comp;
                // ML-Agents 4.x Agent 베이스 내부 필드명: m_Reward / m_CumulativeReward
                _rewardField = comp.GetType().GetField(
                    "m_Reward", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                _cumulativeRewardField = comp.GetType().GetField(
                    "m_CumulativeReward", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);

                if (_rewardField == null)
                {
                    // ML-Agents 4.x: Agent.GetCumulativeReward() 메서드로 접근 시도
                    Debug.LogWarning(
                        "[DroneSnapshotSystem] DroneAgent에서 'reward' 필드를 찾을 수 없습니다. " +
                        "기본값 0을 사용합니다.", this);
                }
                break;
            }

            if (_visionSystem == null)
                Debug.LogWarning(
                    "[DroneSnapshotSystem] DroneVisionSystem을 찾을 수 없습니다. RGB 캡처를 비활성화합니다.", this);

            if (captureDepth && _depthSystem == null)
            {
                Debug.LogWarning(
                    "[DroneSnapshotSystem] captureDepth=true이지만 DroneDepthSystem이 없습니다. " +
                    "captureDepth를 비활성화합니다.", this);
                captureDepth = false;
            }
        }

        private void ResolveFilePrefix()
        {
            if (!string.IsNullOrWhiteSpace(filePrefix))
            {
                _resolvedPrefix = filePrefix;
                return;
            }

            // DroneAgent.Role 리플렉션으로 prefix 자동 결정
            if (_droneAgent != null)
            {
                var roleField = _droneAgent.GetType().GetField(
                    "Role", BindingFlags.Public | BindingFlags.Instance);
                if (roleField != null)
                {
                    int roleVal = Convert.ToInt32(roleField.GetValue(_droneAgent));
                    _resolvedPrefix = roleVal == 1 ? "Evader" : "Pursuer";
                    return;
                }
            }

            _resolvedPrefix = "Drone";
        }

        // ────────────────────────────────────────────
        // 리플렉션 기반 데이터 수집
        // ────────────────────────────────────────────

        private float GetReward()
        {
            if (_droneAgent == null || _rewardField == null) return 0f;
            try { return Convert.ToSingle(_rewardField.GetValue(_droneAgent)); }
            catch { return 0f; }
        }

        private float GetCumulativeReward()
        {
            if (_droneAgent == null || _cumulativeRewardField == null) return 0f;
            try { return Convert.ToSingle(_cumulativeRewardField.GetValue(_droneAgent)); }
            catch { return 0f; }
        }

        private SerializedVector3 GetVelocity()
        {
            return _rigidbody != null
                ? (SerializedVector3)_rigidbody.linearVelocity
                : new SerializedVector3();
        }

        private SerializedVector3 GetTargetRelativePosition()
        {
            // DroneAgent에서 target Transform 리플렉션 시도
            if (_droneAgent != null)
            {
                var targetField = _droneAgent.GetType().GetField(
                    "Target", BindingFlags.Public | BindingFlags.Instance);
                if (targetField != null)
                {
                    var target = targetField.GetValue(_droneAgent) as Transform;
                    if (target != null)
                        return (SerializedVector3)(target.position - transform.position);
                }
            }
            return new SerializedVector3();
        }

        // ────────────────────────────────────────────
        // 내부 헬퍼
        // ────────────────────────────────────────────

        private int GetSourceWidth()
        {
            if (captureWidth > 0) return captureWidth;
            if (_visionSystem != null && _visionSystem.RenderTexture != null)
                return _visionSystem.RenderTexture.width;
            return 84;
        }

        private int GetSourceHeight()
        {
            if (captureHeight > 0) return captureHeight;
            if (_visionSystem != null && _visionSystem.RenderTexture != null)
                return _visionSystem.RenderTexture.height;
            return 84;
        }

        private void RecreateReadbackTexture(int width, int height)
        {
            CleanupReadbackTexture();
            _readbackTexture = new Texture2D(width, height, TextureFormat.RGBA32, false);
        }

        private void CleanupReadbackTexture()
        {
            if (_readbackTexture != null)
            {
                if (Application.isPlaying)
                    Destroy(_readbackTexture);
                else
                    DestroyImmediate(_readbackTexture);
                _readbackTexture = null;
            }
        }
    }

    /// <summary>이미지 저장 포맷</summary>
    public enum ImageFormat { PNG, JPG }
}
