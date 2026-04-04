using System;
using System.Collections.Generic;
using System.Reflection;
using UnityEngine;
using UnityEngine.Rendering.Universal;

namespace DroneCamera
{
    /// <summary>
    /// 씬 전체의 카메라 레이아웃을 관리하는 매니저 컴포넌트.
    /// 활성 배치 수에 따라 Single/Multi 모드를 자동 전환한다.
    ///
    /// Display 할당 (CameraControlSystem 존재 시):
    ///   Display 1 (targetDisplay=0) — 다중뷰 분할 화면 (Single Batch)
    ///   각 역할 최대 4대까지 해당 섹션 내 세로 균등 분할 (최대 8뷰)
    ///
    ///   1~2대/역할: 1컬럼 세로 스택     3~4대/역할: 2컬럼 그리드
    ///
    ///   각 1대:             각 2대:             각 4대 (2×2 그리드):
    ///   ┌────────┬──────┐  ┌────────┬──────┐  ┌────────┬───┬───┐
    ///   │        │ Pur  │  │        │ Pur0 │  │        │P0 │P1 │
    ///   │TopView ├──────┤  │TopView ├──────┤  │TopView ├───┼───┤
    ///   │  (2/3) │ Eva  │  │  (2/3) │ Pur1 │  │  (2/3) │P2 │P3 │
    ///   │        │      │  │        ├──────┤  │        ├───┼───┤
    ///   └────────┴──────┘  │        │ Eva0 │  │        │E0 │E1 │
    ///                      │        ├──────┤  │        ├───┼───┤
    ///                      │        │ Eva1 │  │        │E2 │E3 │
    ///                      └────────┴──────┘  └────────┴───┴───┘
    ///   Display 3 (targetDisplay=2) — 탑뷰 단독 전체 화면
    ///   Display 4 (targetDisplay=3) — Evader FPV 단독 전체 화면
    ///   Display 5 (targetDisplay=4) — Pursuer FPV 단독 전체 화면
    ///
    ///   Multi Batch 시: Display 1에 BatchTopView 전체 화면, 나머지 비활성.
    ///
    /// TopView/BatchTopView 카메라는 런타임 Awake()에서 생성한다 (씬 미리 배치 불필요).
    /// 드론 카메라는 GameObject 태그("Pursuer"/"Evader")로 자동 탐색·등록한다.
    ///
    /// Assembly-CSharp(CityBatchGenerator, CityDataAPI)는
    /// named assembly에서 직접 참조 불가 → 리플렉션으로 접근.
    /// </summary>
    public class CameraControlSystem : MonoBehaviour
    {
        public enum CameraMode { None, SingleBatch, MultiBatch }

        // ────────────────────────────────────────────
        // Inspector 필드
        // ────────────────────────────────────────────

        [Header("TopView 카메라 설정")]
        [Tooltip("TopView 카메라 높이 (Y 위치)")]
        public float topViewHeight = 100f;

        [Tooltip("TopView Orthographic Size (CityDataAPI 미사용 시 기본값)")]
        public float topViewOrthoSize = 50f;

        [Tooltip("TopView 투영 범위 여백 배수 (기본 1.1)")]
        public float topViewPaddingFactor = 1.1f;

        [Header("드론 역할 태그 설정")]
        [Tooltip("Pursuer 역할 드론에 설정된 Unity 태그")]
        public string pursuerTag = "Pursuer";

        [Tooltip("Evader 역할 드론에 설정된 Unity 태그")]
        public string evaderTag = "Evader";

        [Header("카메라 레이블 설정")]
        [Tooltip("Single Batch 모드에서 각 드론 뷰 우하단에 오브젝트 명 표시")]
        public bool showCameraLabels = true;

        [Tooltip("레이블 폰트 크기")]
        [Range(8, 20)]
        public int labelFontSize = 10;

        [Tooltip("레이블 배경 투명도 (0=투명, 1=불투명)")]
        [Range(0f, 1f)]
        public float labelBackgroundAlpha = 0.55f;

        [Header("BatchTopView 카메라 설정")]
        [Tooltip("BatchTopView 카메라 높이 (Y 위치)")]
        public float batchTopViewHeight = 200f;

        [Tooltip("BatchTopView 투영 범위 여백 배수 (기본 1.1)")]
        public float batchTopViewPaddingFactor = 1.1f;

        // ────────────────────────────────────────────
        // 내부 상태
        // ────────────────────────────────────────────

        // Display 1 (targetDisplay=0): 다중뷰 분할용 TopView
        private Camera _topViewCameraMulti;
        // Display 3 (targetDisplay=2): 탑뷰 단독 전체 화면
        private Camera _topViewCameraSolo;
        // Display 1 (targetDisplay=0): Multi-batch 전체 화면
        private Camera _batchTopViewCamera;

        private readonly List<DroneCameraSystem> _trackerCameras = new List<DroneCameraSystem>();
        private readonly List<DroneCameraSystem> _evaderCameras  = new List<DroneCameraSystem>();

        private CameraMode _currentMode = CameraMode.None;
        private int _lastBatchCount = -1;

        // OnGUI 레이블 렌더링용
        private GUIStyle   _labelStyle;
        private Texture2D  _labelBg;

        // 리플렉션 캐시 (AppDomain 스캔 1회 후 재사용)
        private Type  _batchGenType;
        private Type  _cityGenType;
        private bool  _typesResolved;

        // Display 인덱스 상수
        private const int DisplayMultiView = 0;  // Display 1 — 다중뷰(분할 + BatchTopView)
        private const int DisplayTopView   = 2;  // Display 3 — 탑뷰 단독
        private const int DisplayEvader    = 3;  // Display 4 — Evader FPV 단독
        private const int DisplayPursuer   = 4;  // Display 5 — Pursuer FPV 단독

        // ────────────────────────────────────────────
        // Unity 생명주기
        // ────────────────────────────────────────────

        private void Awake()
        {
            CreateTopViewCameras();
            CreateBatchTopViewCamera();
        }

        private void Start()
        {
            ResolveExternalTypes();
            ScanAndRegisterDroneAgents();
            int n = GetActiveBatchCount();
            _lastBatchCount = n;
            ApplyLayout(n);
        }

        private void Update()
        {
            int n = GetActiveBatchCount();
            if (n == _lastBatchCount) return;
            _lastBatchCount = n;
            ApplyLayout(n);
        }

        private void OnValidate()
        {
            // 인스펙터 변경 시 카메라 파라미터 즉시 반영
            SyncTopViewCameraParams();

            if (_batchTopViewCamera != null)
            {
                _batchTopViewCamera.targetDisplay = DisplayMultiView;
                var p = _batchTopViewCamera.transform.position;
                _batchTopViewCamera.transform.position = new Vector3(p.x, batchTopViewHeight, p.z);
            }

            if (_topViewCameraMulti == null || _batchTopViewCamera == null) return;
            int n = _lastBatchCount >= 0 ? _lastBatchCount : 1;
            ApplyLayout(n);
        }

        // ────────────────────────────────────────────
        // 공개 API
        // ────────────────────────────────────────────

        /// <summary>DroneCameraSystem을 역할별로 등록한다.</summary>
        public void RegisterDroneCamera(DroneCameraSystem cam, DroneRole role)
        {
            if (role == DroneRole.Pursuer)
            {
                if (!_trackerCameras.Contains(cam)) _trackerCameras.Add(cam);
                cam.SetSoloDisplay(DisplayPursuer);
            }
            else
            {
                if (!_evaderCameras.Contains(cam)) _evaderCameras.Add(cam);
                cam.SetSoloDisplay(DisplayEvader);
            }

            // 다중뷰 카메라는 항상 Display 1
            cam.SetMultiViewDisplay(DisplayMultiView);

            int n = _lastBatchCount >= 0 ? _lastBatchCount : GetActiveBatchCount();
            ApplyLayout(n);
        }

        /// <summary>DroneCameraSystem 등록 해제.</summary>
        public void UnregisterDroneCamera(DroneCameraSystem cam)
        {
            _trackerCameras.Remove(cam);
            _evaderCameras.Remove(cam);
        }

        /// <summary>현재 카메라 모드 반환.</summary>
        public CameraMode GetCurrentMode() => _currentMode;

        /// <summary>
        /// 활성 배치 수 반환.
        /// CityBatchGenerator(columns × rows), 없으면 1 반환.
        /// </summary>
        public int GetActiveBatchCount()
        {
            ResolveExternalTypes();
            if (_batchGenType == null) return 1;

            var batchGen = FindFirstObjectByType(_batchGenType) as MonoBehaviour;
            if (batchGen == null) return 1;

            try
            {
                int columns = Convert.ToInt32(_batchGenType.GetField("columns")?.GetValue(batchGen) ?? 1);
                int rows    = Convert.ToInt32(_batchGenType.GetField("rows")   ?.GetValue(batchGen) ?? 1);
                return Mathf.Max(1, columns * rows);
            }
            catch
            {
                return 1;
            }
        }

        /// <summary>
        /// 배치 수에 따라 레이아웃을 적용한다.
        /// 테스트에서 임의의 배치 수로 직접 호출할 수 있도록 public으로 공개.
        /// </summary>
        public void ApplyLayout(int batchCount)
        {
            ApplyDisplayAssignments();

            if (batchCount <= 0)
                ApplyNoneMode();
            else if (batchCount == 1)
                ApplySingleBatchMode();
            else
                ApplyMultiBatchMode();
        }

        // ────────────────────────────────────────────
        // 레이아웃 적용
        // ────────────────────────────────────────────

        private void ApplyNoneMode()
        {
            _currentMode = CameraMode.None;

            SetCamEnabled(_topViewCameraMulti,  false);
            SetCamEnabled(_topViewCameraSolo,   false);
            SetCamEnabled(_batchTopViewCamera,  false);
            foreach (var c in _trackerCameras) c.SetEnabled(false);
            foreach (var c in _evaderCameras)  c.SetEnabled(false);

            Debug.LogWarning("[CameraControlSystem] 활성 배치가 없습니다. 모든 카메라를 비활성화합니다.");
        }

        private void ApplySingleBatchMode()
        {
            _currentMode = CameraMode.SingleBatch;

            Debug.Log($"[CameraControlSystem] SingleBatch — Pursuers:{_trackerCameras.Count}, Evaders:{_evaderCameras.Count}");

            // BatchTopView 비활성화
            SetCamEnabled(_batchTopViewCamera, false);

            // ── Display 1 다중뷰 분할 레이아웃 ──
            // TopView: 왼쪽 2/3, 전체 높이
            SetCamEnabled(_topViewCameraMulti, true);
            _topViewCameraMulti.rect = new Rect(0f, 0f, 2f / 3f, 1f);

            // 드론 FPV: 오른쪽 1/3 상하 분할 (Pursuer 상단 / Evader 하단)
            SetDroneMultiViewports(_trackerCameras, new Rect(2f / 3f, 0.5f, 1f / 3f, 0.5f));
            SetDroneMultiViewports(_evaderCameras,  new Rect(2f / 3f, 0f,   1f / 3f, 0.5f));

            // ── 개별 디스플레이 단독 전체 화면 ──
            // TopView Solo: Display 3
            SetCamEnabled(_topViewCameraSolo, true);
            _topViewCameraSolo.rect = new Rect(0f, 0f, 1f, 1f);

            // 드론 FPV Solo: Display 4, 5
            SetDroneSoloViewports(_trackerCameras, new Rect(0f, 0f, 1f, 1f));
            SetDroneSoloViewports(_evaderCameras,  new Rect(0f, 0f, 1f, 1f));

            UpdateTopViewCamera();
        }

        private void ApplyMultiBatchMode()
        {
            _currentMode = CameraMode.MultiBatch;

            // 개별 TopView 및 드론 FPV 전부 비활성화
            SetCamEnabled(_topViewCameraMulti, false);
            SetCamEnabled(_topViewCameraSolo,  false);
            foreach (var c in _trackerCameras) c.SetEnabled(false);
            foreach (var c in _evaderCameras)  c.SetEnabled(false);

            // BatchTopView — Display 1, 전체 화면
            SetCamEnabled(_batchTopViewCamera, true);
            _batchTopViewCamera.rect = new Rect(0f, 0f, 1f, 1f);
            UpdateBatchTopViewCamera();
        }

        /// <summary>
        /// 다중뷰(Display 1)용 드론 카메라에 분할 Viewport 적용.
        ///
        /// 카메라 수에 따른 그리드 정책 (sectionRect 내부):
        ///   1~2대: 1컬럼 × N행  (세로 스택)
        ///   3~4대: 2컬럼 × ceil(N/2)행  (2×2 그리드)
        ///   5대~: 초과분은 MultiView 비활성화
        ///
        /// 배치 순서: 좌→우, 상→하 (등록 순서 기준)
        /// </summary>
        private const int MaxSlotsPerRole       = 4;
        private const int MultiColumnThreshold  = 3; // 이 이상이면 2컬럼 사용

        private static void SetDroneMultiViewports(List<DroneCameraSystem> cameras, Rect sectionRect)
        {
            int count = Mathf.Min(cameras.Count, MaxSlotsPerRole);
            int cols  = count >= MultiColumnThreshold ? 2 : 1;
            int rows  = Mathf.CeilToInt((float)count / cols);

            float slotW = sectionRect.width  / cols;
            float slotH = sectionRect.height / rows;

            for (int i = 0; i < cameras.Count; i++)
            {
                if (i < count)
                {
                    int row = i / cols;
                    int col = i % cols;
                    // 섹션 상단부터 좌→우, 상→하로 채움 (Unity y=0 은 화면 하단)
                    float slotX = sectionRect.x + col * slotW;
                    float slotY = sectionRect.y + sectionRect.height - (row + 1) * slotH;
                    cameras[i].SetMultiViewEnabled(true);
                    cameras[i].SetViewportRect(new Rect(slotX, slotY, slotW, slotH));
                }
                else
                {
                    cameras[i].SetMultiViewEnabled(false);
                }
            }
        }

        /// <summary>
        /// 단독 디스플레이용 드론 카메라에 Viewport 적용.
        /// [0]번만 활성화하고, 나머지는 비활성화 (다중 등록 정책).
        /// </summary>
        private static void SetDroneSoloViewports(List<DroneCameraSystem> cameras, Rect rect)
        {
            for (int i = 0; i < cameras.Count; i++)
            {
                if (i == 0)
                {
                    cameras[i].SetSoloEnabled(true);
                    cameras[i].SetSoloViewportRect(rect);
                }
                else
                {
                    cameras[i].SetSoloEnabled(false);
                }
            }
        }

        /// <summary>
        /// 모든 카메라에 targetDisplay를 재할당한다.
        /// ApplyLayout 호출 시마다 실행되어 인스펙터 변경도 반영.
        /// </summary>
        private void ApplyDisplayAssignments()
        {
            if (_topViewCameraMulti != null) _topViewCameraMulti.targetDisplay = DisplayMultiView;
            if (_topViewCameraSolo  != null) _topViewCameraSolo.targetDisplay  = DisplayTopView;
            if (_batchTopViewCamera != null) _batchTopViewCamera.targetDisplay = DisplayMultiView;

            foreach (var cam in _trackerCameras)
            {
                cam.SetMultiViewDisplay(DisplayMultiView);
                cam.SetSoloDisplay(DisplayPursuer);
            }
            foreach (var cam in _evaderCameras)
            {
                cam.SetMultiViewDisplay(DisplayMultiView);
                cam.SetSoloDisplay(DisplayEvader);
            }
        }

        // ────────────────────────────────────────────
        // 카메라 생성
        // ────────────────────────────────────────────

        /// <summary>탑뷰 카메라의 그림자 렌더링을 비활성화해 CPU/GPU 부하를 줄인다.</summary>
        private static void DisableShadows(Camera cam)
        {
            var urpData = cam.GetUniversalAdditionalCameraData();
            if (urpData != null) urpData.renderShadows = false;
        }

        private void CreateTopViewCameras()
        {
            // 다중뷰 분할용 (Display 1)
            var goMulti = new GameObject("TopView_Camera");
            goMulti.transform.SetParent(transform, false);
            goMulti.transform.position = new Vector3(0f, topViewHeight, 0f);
            goMulti.transform.rotation = Quaternion.LookRotation(Vector3.down, Vector3.forward);
            _topViewCameraMulti = goMulti.AddComponent<Camera>();
            _topViewCameraMulti.orthographic     = true;
            _topViewCameraMulti.orthographicSize = topViewOrthoSize;
            _topViewCameraMulti.targetDisplay    = DisplayMultiView;
            _topViewCameraMulti.enabled          = false;
            DisableShadows(_topViewCameraMulti);

            // 탑뷰 단독 전체 화면 (Display 3)
            var goSolo = new GameObject("TopView_Camera_Solo");
            goSolo.transform.SetParent(transform, false);
            goSolo.transform.position = new Vector3(0f, topViewHeight, 0f);
            goSolo.transform.rotation = Quaternion.LookRotation(Vector3.down, Vector3.forward);
            _topViewCameraSolo = goSolo.AddComponent<Camera>();
            _topViewCameraSolo.orthographic     = true;
            _topViewCameraSolo.orthographicSize = topViewOrthoSize;
            _topViewCameraSolo.targetDisplay    = DisplayTopView;
            _topViewCameraSolo.enabled          = false;
            DisableShadows(_topViewCameraSolo);
        }

        private void CreateBatchTopViewCamera()
        {
            var go = new GameObject("BatchTopView_Camera");
            go.transform.SetParent(transform, false);
            go.transform.position = new Vector3(0f, batchTopViewHeight, 0f);
            go.transform.rotation = Quaternion.LookRotation(Vector3.down, Vector3.forward);

            _batchTopViewCamera = go.AddComponent<Camera>();
            _batchTopViewCamera.orthographic     = true;
            _batchTopViewCamera.orthographicSize = topViewOrthoSize;
            _batchTopViewCamera.targetDisplay    = DisplayMultiView;
            _batchTopViewCamera.enabled          = false;
            DisableShadows(_batchTopViewCamera);
        }

        // ────────────────────────────────────────────
        // 카메라 파라미터 업데이트
        // ────────────────────────────────────────────

        /// <summary>TopView 두 카메라(Multi/Solo) 위치·OrthoSize 동기화.</summary>
        private void SyncTopViewCameraParams()
        {
            if (_topViewCameraMulti != null)
            {
                _topViewCameraMulti.targetDisplay    = DisplayMultiView;
                _topViewCameraMulti.orthographicSize = topViewOrthoSize;
                var p = _topViewCameraMulti.transform.position;
                _topViewCameraMulti.transform.position = new Vector3(p.x, topViewHeight, p.z);
            }
            if (_topViewCameraSolo != null)
            {
                _topViewCameraSolo.targetDisplay    = DisplayTopView;
                _topViewCameraSolo.orthographicSize = topViewOrthoSize;
                var p = _topViewCameraSolo.transform.position;
                _topViewCameraSolo.transform.position = new Vector3(p.x, topViewHeight, p.z);
            }
        }

        /// <summary>
        /// TopView 두 카메라 위치 및 OrthoSize 갱신.
        /// CityBatchGenerator + CityGenerator 파라미터 기반 자동 계산;
        /// 미존재 시 Inspector 기본값 사용.
        /// </summary>
        private void UpdateTopViewCamera()
        {
            Vector3 pos;
            float orthoSize;

            if (!TryGetSingleCityBounds(out Vector3 center, out float width, out float depth))
            {
                orthoSize = topViewOrthoSize;
                pos       = new Vector3(0f, topViewHeight, 0f);
            }
            else
            {
                orthoSize = Mathf.Max(width, depth) / 2f * topViewPaddingFactor;
                pos       = new Vector3(center.x, topViewHeight, center.z);
            }

            _topViewCameraMulti.orthographicSize = orthoSize;
            _topViewCameraMulti.transform.position = pos;

            if (_topViewCameraSolo != null)
            {
                _topViewCameraSolo.orthographicSize = orthoSize;
                _topViewCameraSolo.transform.position = pos;
            }
        }

        /// <summary>
        /// BatchTopView 카메라 위치 및 OrthoSize 갱신.
        /// CityBatchGenerator 배치 레이아웃 기반 전체 영역 포함.
        /// </summary>
        private void UpdateBatchTopViewCamera()
        {
            ResolveExternalTypes();
            if (_batchGenType == null)
            {
                _batchTopViewCamera.orthographicSize = topViewOrthoSize;
                return;
            }

            var batchGen = FindFirstObjectByType(_batchGenType) as MonoBehaviour;
            if (batchGen == null)
            {
                _batchTopViewCamera.orthographicSize = topViewOrthoSize;
                return;
            }

            try
            {
                int   columns  = Convert.ToInt32 (_batchGenType.GetField("columns")  ?.GetValue(batchGen) ?? 1);
                int   rows     = Convert.ToInt32 (_batchGenType.GetField("rows")     ?.GetValue(batchGen) ?? 1);
                float spacingX = Convert.ToSingle(_batchGenType.GetField("spacingX") ?.GetValue(batchGen) ?? 0f);
                float spacingZ = Convert.ToSingle(_batchGenType.GetField("spacingZ") ?.GetValue(batchGen) ?? 0f);

                float slotW, slotD;
                GetSlotSize(batchGen, out slotW, out slotD);

                float totalW = columns * slotW + (columns - 1) * spacingX;
                float totalD = rows    * slotD + (rows    - 1) * spacingZ;

                Vector3 origin  = batchGen.transform.position;
                float   centerX = origin.x + (columns - 1) * (slotW + spacingX) / 2f + slotW / 2f;
                float   centerZ = origin.z + (rows    - 1) * (slotD + spacingZ) / 2f + slotD / 2f;

                float orthoSize = Mathf.Max(totalW, totalD) / 2f * batchTopViewPaddingFactor;

                _batchTopViewCamera.transform.position = new Vector3(centerX, batchTopViewHeight, centerZ);
                _batchTopViewCamera.orthographicSize   = orthoSize;
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[CameraControlSystem] BatchTopView 크기 계산 실패: {ex.Message}. 기본값 사용.");
                _batchTopViewCamera.orthographicSize = topViewOrthoSize;
            }
        }

        // ────────────────────────────────────────────
        // 드론 자동 탐색 & 등록
        // ────────────────────────────────────────────

        /// <summary>
        /// 씬 내 모든 DroneCameraSystem 컴포넌트를 탐색하고,
        /// GameObject 태그(pursuerTag / evaderTag)를 기준으로 역할을 판별해 등록한다.
        /// DroneCameraSystem.Start()에서 이미 등록한 것은 중복 등록되지 않는다.
        /// </summary>
        private void ScanAndRegisterDroneAgents()
        {
            var allCamSystems = FindObjectsByType<DroneCameraSystem>(FindObjectsSortMode.None);
            foreach (var camSystem in allCamSystems)
            {
                if (_trackerCameras.Contains(camSystem) || _evaderCameras.Contains(camSystem))
                    continue;

                string tag = camSystem.gameObject.tag;
                if (tag == pursuerTag)
                    RegisterDroneCamera(camSystem, DroneRole.Pursuer);
                else if (tag == evaderTag)
                    RegisterDroneCamera(camSystem, DroneRole.Evader);
                else
                    Debug.LogWarning($"[CameraControlSystem] '{camSystem.gameObject.name}'의 태그({tag})가 pursuerTag/evaderTag와 일치하지 않아 건너뜀.");
            }
        }

        // ────────────────────────────────────────────
        // 외부 타입 리플렉션 헬퍼
        // ────────────────────────────────────────────

        private void ResolveExternalTypes()
        {
            if (_typesResolved) return;
            _typesResolved = true;

            foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
            {
                if (_batchGenType == null)
                    _batchGenType = assembly.GetType("ProceduralCityGenerator.CityBatchGenerator");
                if (_cityGenType == null)
                    _cityGenType  = assembly.GetType("ProceduralCityGenerator.CityGenerator");
                if (_batchGenType != null && _cityGenType != null) break;
            }
        }

        private bool TryGetSingleCityBounds(out Vector3 center, out float width, out float depth)
        {
            center = Vector3.zero;
            width  = 0f;
            depth  = 0f;

            ResolveExternalTypes();
            if (_batchGenType == null) return false;

            var batchGen = FindFirstObjectByType(_batchGenType) as MonoBehaviour;
            if (batchGen == null) return false;

            try
            {
                float slotW, slotD;
                GetSlotSize(batchGen, out slotW, out slotD);

                Vector3 origin = batchGen.transform.position;
                center = new Vector3(origin.x + slotW / 2f, 0f, origin.z + slotD / 2f);
                width  = slotW;
                depth  = slotD;
                return true;
            }
            catch
            {
                return false;
            }
        }

        private void GetSlotSize(MonoBehaviour batchGen, out float slotWidth, out float slotDepth)
        {
            slotWidth  = 100f;
            slotDepth  = 100f;

            if (_cityGenType == null) return;

            MonoBehaviour cityGen = null;

            var templateObjField = _batchGenType.GetField("cityTemplateObject");
            if (templateObjField != null)
            {
                var go = templateObjField.GetValue(batchGen) as GameObject;
                if (go != null)
                    cityGen = go.GetComponent(_cityGenType) as MonoBehaviour;
            }

            if (cityGen == null)
            {
                var templateField = _batchGenType.GetField("cityTemplate");
                if (templateField != null)
                    cityGen = templateField.GetValue(batchGen) as MonoBehaviour;
            }

            if (cityGen == null) return;

            float unitDistance    = Convert.ToSingle(_cityGenType.GetField("unitDistance")    ?.GetValue(cityGen) ?? 1f);
            int   buildingWidth   = Convert.ToInt32 (_cityGenType.GetField("buildingWidth")   ?.GetValue(cityGen) ?? 1);
            int   buildingDepth   = Convert.ToInt32 (_cityGenType.GetField("buildingDepth")   ?.GetValue(cityGen) ?? 1);
            int   buildingSpacing = Convert.ToInt32 (_cityGenType.GetField("buildingSpacing") ?.GetValue(cityGen) ?? 1);
            int   maxWidth        = Convert.ToInt32 (_cityGenType.GetField("maxWidth")        ?.GetValue(cityGen) ?? 10);
            int   maxDepth        = Convert.ToInt32 (_cityGenType.GetField("maxDepth")        ?.GetValue(cityGen) ?? 10);

            float cellW = (buildingWidth  + buildingSpacing) * unitDistance;
            float cellD = (buildingDepth  + buildingSpacing) * unitDistance;
            slotWidth   = maxWidth * cellW;
            slotDepth   = maxDepth * cellD;
        }

        // ────────────────────────────────────────────
        // 카메라 레이블 오버레이 (OnGUI)
        // ────────────────────────────────────────────

        private void OnGUI()
        {
            if (!showCameraLabels)                        return;
            if (_currentMode != CameraMode.SingleBatch)   return;
            if (Event.current.type != EventType.Repaint)  return;

            // OnGUI는 각 카메라 렌더링마다 호출될 수 있으므로
            // 현재 렌더링 중인 카메라가 Display 1의 카메라일 때만 라벨을 그린다.
            // Camera.current가 null이면 (에디터 등) 그냥 그린다.
            if (Camera.current != null && Camera.current.targetDisplay != DisplayMultiView)
                return;

            var style = GetLabelStyle();
            DrawRegisteredDroneLabels(style);
        }

        /// <summary>
        /// 등록된 모든 드론 카메라의 MultiViewCamera viewport 우하단에 오브젝트 명을 표시.
        /// 등록 리스트를 직접 사용하여 카메라 개수에 관계없이 모든 뷰에 라벨 표시.
        /// </summary>
        private void DrawRegisteredDroneLabels(GUIStyle style)
        {
            const float labelW  = 110f;
            const float labelH  = 18f;
            const float padding = 3f;

            float sw = Screen.width;
            float sh = Screen.height;

            // 등록된 드론 카메라 리스트에서 직접 그리기
            DrawDroneListLabels(_trackerCameras, "Pursuer", sw, sh, labelW, labelH, padding, style);
            DrawDroneListLabels(_evaderCameras,  "Evader",  sw, sh, labelW, labelH, padding, style);

            // 폴백: 등록 리스트에 없지만 Display 1 우측 1/3에서 렌더링 중인 퍼스펙티브 카메라도 처리
            DrawUnregisteredDroneLabels(sw, sh, labelW, labelH, padding, style);
        }

        private void DrawDroneListLabels(
            List<DroneCameraSystem> cameras,
            string roleTag,
            float sw, float sh,
            float labelW, float labelH, float padding,
            GUIStyle style)
        {
            for (int i = 0; i < cameras.Count; i++)
            {
                var dcs = cameras[i];
                if (dcs == null) continue;
                Camera cam = dcs.MultiViewCamera;
                if (cam == null || !cam.enabled) continue;

                // "오브젝트명 (역할)" 형식으로 표시
                string label = dcs.gameObject.name;

                Rect vp = cam.rect;

                // Viewport → GUI 좌표 변환
                float guiX = (vp.x + vp.width) * sw - labelW - padding;
                float guiY = (1f - vp.y) * sh - labelH - padding;

                GUI.Label(new Rect(guiX, guiY, labelW, labelH), label, style);
            }
        }

        /// <summary>
        /// 등록 리스트에 포함되지 않은 카메라도 라벨을 표시하는 폴백.
        /// Display 1(targetDisplay=0), 퍼스펙티브, 우측 1/3 영역 카메라를 탐색.
        /// </summary>
        private void DrawUnregisteredDroneLabels(
            float sw, float sh,
            float labelW, float labelH, float padding,
            GUIStyle style)
        {
            const float droneAreaStart = 2f / 3f - 0.01f;

            // 이미 라벨을 그린 카메라 수집 (중복 방지)
            var drawn = new HashSet<Camera>();
            foreach (var dcs in _trackerCameras)
                if (dcs?.MultiViewCamera != null) drawn.Add(dcs.MultiViewCamera);
            foreach (var dcs in _evaderCameras)
                if (dcs?.MultiViewCamera != null) drawn.Add(dcs.MultiViewCamera);

            foreach (var cam in Camera.allCameras)
            {
                if (!cam.enabled)                continue;
                if (cam.targetDisplay != 0)      continue;
                if (cam.orthographic)            continue;
                if (cam.rect.x < droneAreaStart) continue;
                if (drawn.Contains(cam))         continue;

                string label = cam.transform.parent != null
                    ? cam.transform.parent.name
                    : cam.gameObject.name;

                Rect vp = cam.rect;
                float guiX = (vp.x + vp.width) * sw - labelW - padding;
                float guiY = (1f - vp.y) * sh - labelH - padding;

                GUI.Label(new Rect(guiX, guiY, labelW, labelH), label, style);
            }
        }

        /// <summary>레이블 GUIStyle을 최초 호출 시 1회 생성하고 캐시한다.</summary>
        private GUIStyle GetLabelStyle()
        {
            // fontSize 또는 alpha 변경 시 스타일 재생성
            if (_labelStyle != null &&
                _labelStyle.fontSize == labelFontSize &&
                _labelBg != null)
                return _labelStyle;

            if (_labelBg != null) Destroy(_labelBg);

            _labelBg = new Texture2D(1, 1);
            _labelBg.SetPixel(0, 0, new Color(0f, 0f, 0f, labelBackgroundAlpha));
            _labelBg.Apply();

            _labelStyle = new GUIStyle(GUI.skin.label)
            {
                fontSize  = labelFontSize,
                alignment = TextAnchor.MiddleCenter,
                padding   = new RectOffset(4, 4, 2, 2),
            };
            _labelStyle.normal.textColor    = Color.white;
            _labelStyle.normal.background   = _labelBg;

            return _labelStyle;
        }

        private void OnDestroy()
        {
            if (_labelBg != null) Destroy(_labelBg);
        }

        // ────────────────────────────────────────────
        // 유틸리티
        // ────────────────────────────────────────────

        private static void SetCamEnabled(Camera cam, bool enabled)
        {
            if (cam != null) cam.enabled = enabled;
        }
    }
}