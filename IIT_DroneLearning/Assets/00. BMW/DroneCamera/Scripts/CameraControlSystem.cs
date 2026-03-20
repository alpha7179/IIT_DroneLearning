using System;
using System.Collections.Generic;
using System.Reflection;
using UnityEngine;

namespace BMW.DroneCamera
{
    /// <summary>
    /// 씬 전체의 카메라 레이아웃을 관리하는 매니저 컴포넌트.
    /// 활성 배치 수에 따라 Single/Multi 모드를 자동 전환한다.
    ///
    /// Display 할당 (CameraControlSystem 존재 시):
    ///   Display 1 (targetDisplay=0) — 다중뷰 분할 화면
    ///     ┌──────────────────────────────┐
    ///     │        TopView  (상단 절반)   │
    ///     ├──────────────┬───────────────┤
    ///     │  Pursuer FPV │   Evader FPV  │
    ///     └──────────────┴───────────────┘
    ///   Display 3 (targetDisplay=2) — 탑뷰 단독 전체 화면
    ///   Display 4 (targetDisplay=3) — Evader FPV 단독 전체 화면
    ///   Display 5 (targetDisplay=4) — Pursuer FPV 단독 전체 화면
    ///
    ///   Multi Batch 시: Display 1에 BatchTopView 전체 화면, 나머지 비활성.
    ///
    /// TopView/BatchTopView 카메라는 런타임 Awake()에서 생성한다 (씬 미리 배치 불필요).
    /// 드론 카메라는 DroneAgent.Role 리플렉션으로 자동 탐색·등록한다.
    ///
    /// Assembly-CSharp(CityBatchGenerator, CityDataAPI, DroneAgent)는
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

        // 리플렉션 캐시 (AppDomain 스캔 1회 후 재사용)
        private Type  _batchGenType;
        private Type  _cityGenType;
        private Type  _droneAgentType;
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

            // BatchTopView 비활성화
            SetCamEnabled(_batchTopViewCamera, false);

            // ── Display 1 다중뷰 분할 레이아웃 ──
            // TopView: 상단 절반
            SetCamEnabled(_topViewCameraMulti, true);
            _topViewCameraMulti.rect = new Rect(0f, 0.5f, 1f, 0.5f);

            // 드론 FPV: 하단 분할 (다중뷰용 MultiViewCamera)
            SetDroneMultiViewports(_trackerCameras, new Rect(0f,   0f, 0.5f, 0.5f));
            SetDroneMultiViewports(_evaderCameras,  new Rect(0.5f, 0f, 0.5f, 0.5f));

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
        /// [0]번만 활성화하고, 나머지는 비활성화 (다중 등록 정책).
        /// </summary>
        private static void SetDroneMultiViewports(List<DroneCameraSystem> cameras, Rect rect)
        {
            for (int i = 0; i < cameras.Count; i++)
            {
                if (i == 0)
                {
                    cameras[i].SetMultiViewEnabled(true);
                    cameras[i].SetViewportRect(rect);
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
        /// 씬 내 모든 DroneAgent 컴포넌트를 리플렉션으로 탐색하고,
        /// 같은 GameObject의 DroneCameraSystem을 역할에 맞게 등록한다.
        /// DroneCameraSystem.Start()에서 등록한 것은 중복 등록되지 않는다.
        /// </summary>
        private void ScanAndRegisterDroneAgents()
        {
            if (_droneAgentType == null) return;

            var roleField = _droneAgentType.GetField(
                "Role", BindingFlags.Public | BindingFlags.Instance);

            var agents = FindObjectsByType(_droneAgentType, FindObjectsSortMode.None);
            foreach (UnityEngine.Object agentObj in agents)
            {
                var mb = agentObj as MonoBehaviour;
                if (mb == null) continue;

                var camSystem = mb.GetComponent<DroneCameraSystem>();
                if (camSystem == null) continue;

                if (_trackerCameras.Contains(camSystem) || _evaderCameras.Contains(camSystem))
                    continue;

                DroneRole role = DroneRole.Pursuer;
                if (roleField != null)
                {
                    int roleValue = Convert.ToInt32(roleField.GetValue(mb));
                    role = roleValue == 1 ? DroneRole.Evader : DroneRole.Pursuer;
                }

                RegisterDroneCamera(camSystem, role);
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
                if (_droneAgentType == null)
                    _droneAgentType = assembly.GetType("DroneAgent");
                if (_batchGenType != null && _cityGenType != null && _droneAgentType != null) break;
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
        // 유틸리티
        // ────────────────────────────────────────────

        private static void SetCamEnabled(Camera cam, bool enabled)
        {
            if (cam != null) cam.enabled = enabled;
        }
    }
}