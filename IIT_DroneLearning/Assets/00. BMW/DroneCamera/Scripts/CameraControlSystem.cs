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
    /// Single Batch 레이아웃 (Viewport Rect):
    ///   ┌──────────────────────────────┐
    ///   │     TopView  (0,0.5,1,0.5)   │
    ///   ├──────────────┬───────────────┤
    ///   │ Tracker FPV  │  Evader FPV   │
    ///   │ (0,0,0.5,.5) │ (.5,0,.5,.5)  │
    ///   └──────────────┴───────────────┘
    ///
    /// Multi Batch 레이아웃:
    ///   전체 화면 BatchTopView (0,0,1,1)
    ///
    /// TopView/BatchTopView 카메라는 런타임 Awake()에서 생성한다 (씬 미리 배치 불필요).
    ///
    /// Orthographic Size 공식:
    ///   TopView    : max(cityWidth, cityDepth) / 2 * paddingFactor
    ///   BatchTopView: max(totalW, totalD) / 2 * paddingFactor
    ///     totalW = columns * slotWidth  + (columns-1) * spacingX
    ///     totalD = rows    * slotDepth  + (rows   -1) * spacingZ
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

        private Camera _topViewCamera;
        private Camera _batchTopViewCamera;

        private readonly List<DroneCameraSystem> _trackerCameras = new List<DroneCameraSystem>();
        private readonly List<DroneCameraSystem> _evaderCameras  = new List<DroneCameraSystem>();

        private CameraMode _currentMode = CameraMode.None;
        private int _lastBatchCount = -1;

        // 리플렉션 캐시 (AppDomain 스캔 1회 후 재사용)
        private Type  _batchGenType;
        private Type  _cityGenType;
        private bool  _typesResolved;

        // ────────────────────────────────────────────
        // Unity 생명주기
        // ────────────────────────────────────────────

        private void Awake()
        {
            CreateTopViewCamera();
            CreateBatchTopViewCamera();
        }

        private void Start()
        {
            ResolveExternalTypes();
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

        // ────────────────────────────────────────────
        // 공개 API
        // ────────────────────────────────────────────

        /// <summary>DroneCameraSystem을 역할별로 등록한다.</summary>
        public void RegisterDroneCamera(DroneCameraSystem cam, DroneRole role)
        {
            if (role == DroneRole.Pursuer)
            {
                if (!_trackerCameras.Contains(cam)) _trackerCameras.Add(cam);
            }
            else
            {
                if (!_evaderCameras.Contains(cam)) _evaderCameras.Add(cam);
            }

            // 현재 레이아웃 재적용 (다중 등록 정책 즉시 반영)
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

            SetCamEnabled(_topViewCamera,      false);
            SetCamEnabled(_batchTopViewCamera, false);
            foreach (var c in _trackerCameras) c.SetEnabled(false);
            foreach (var c in _evaderCameras)  c.SetEnabled(false);

            Debug.LogWarning("[CameraControlSystem] 활성 배치가 없습니다. 모든 카메라를 비활성화합니다.");
        }

        private void ApplySingleBatchMode()
        {
            _currentMode = CameraMode.SingleBatch;

            // BatchTopView 비활성화
            SetCamEnabled(_batchTopViewCamera, false);

            // TopView — 상단 와이드
            SetCamEnabled(_topViewCamera, true);
            _topViewCamera.rect = new Rect(0f, 0.5f, 1f, 0.5f);
            UpdateTopViewCamera();

            // 드론 FPV — 다중 등록 시 [0]만 렌더링
            SetDroneViewports(_trackerCameras, new Rect(0f,   0f, 0.5f, 0.5f));
            SetDroneViewports(_evaderCameras,  new Rect(0.5f, 0f, 0.5f, 0.5f));
        }

        private void ApplyMultiBatchMode()
        {
            _currentMode = CameraMode.MultiBatch;

            // TopView / 드론 FPV 전부 비활성화
            SetCamEnabled(_topViewCamera, false);
            foreach (var c in _trackerCameras) c.SetEnabled(false);
            foreach (var c in _evaderCameras)  c.SetEnabled(false);

            // BatchTopView 전체 화면
            SetCamEnabled(_batchTopViewCamera, true);
            _batchTopViewCamera.rect = new Rect(0f, 0f, 1f, 1f);
            UpdateBatchTopViewCamera();
        }

        /// <summary>
        /// 카메라 리스트에 Viewport 적용.
        /// [0]번만 활성화하고, 나머지는 비활성화 (다중 등록 정책).
        /// </summary>
        private static void SetDroneViewports(List<DroneCameraSystem> cameras, Rect rect)
        {
            for (int i = 0; i < cameras.Count; i++)
            {
                if (i == 0)
                {
                    cameras[i].SetEnabled(true);
                    cameras[i].SetViewportRect(rect);
                }
                else
                {
                    cameras[i].SetEnabled(false);
                }
            }
        }

        // ────────────────────────────────────────────
        // 카메라 생성
        // ────────────────────────────────────────────

        private void CreateTopViewCamera()
        {
            var go = new GameObject("TopView_Camera");
            go.transform.SetParent(transform, false);
            go.transform.position = new Vector3(0f, topViewHeight, 0f);
            go.transform.rotation = Quaternion.LookRotation(Vector3.down, Vector3.forward);

            _topViewCamera = go.AddComponent<Camera>();
            _topViewCamera.orthographic     = true;
            _topViewCamera.orthographicSize = topViewOrthoSize;
            _topViewCamera.enabled          = false;
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
            _batchTopViewCamera.enabled          = false;
        }

        // ────────────────────────────────────────────
        // 카메라 파라미터 업데이트
        // ────────────────────────────────────────────

        /// <summary>
        /// TopView 카메라 위치 및 Orthographic Size 갱신.
        /// CityBatchGenerator + CityGenerator 파라미터 기반 자동 계산;
        /// 미존재 시 Inspector 기본값 사용.
        /// </summary>
        private void UpdateTopViewCamera()
        {
            // 도시 중심 및 크기 계산 시도
            if (!TryGetSingleCityBounds(out Vector3 center, out float width, out float depth))
            {
                // 기본값 유지
                _topViewCamera.orthographicSize = topViewOrthoSize;
                _topViewCamera.transform.position = new Vector3(0f, topViewHeight, 0f);
                return;
            }

            float orthoSize = Mathf.Max(width, depth) / 2f * topViewPaddingFactor;
            _topViewCamera.orthographicSize = orthoSize;
            _topViewCamera.transform.position = new Vector3(center.x, topViewHeight, center.z);
        }

        /// <summary>
        /// BatchTopView 카메라 위치 및 Orthographic Size 갱신.
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

                // 전체 배치 영역
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

        /// <summary>
        /// 단일 도시의 중심 좌표와 크기를 계산.
        /// CityBatchGenerator(1×1) 또는 CityGenerator 단독 사용 시.
        /// </summary>
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

        /// <summary>
        /// CityBatchGenerator가 참조하는 CityGenerator 템플릿으로부터 슬롯 크기를 계산.
        /// </summary>
        private void GetSlotSize(MonoBehaviour batchGen, out float slotWidth, out float slotDepth)
        {
            slotWidth  = 100f;
            slotDepth  = 100f;

            if (_cityGenType == null) return;

            // cityTemplateObject(GameObject) → cityTemplate(CityGenerator) 순으로 탐색
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

            // CityBatchGenerator.CalculateSlotSize 로직 재현
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
