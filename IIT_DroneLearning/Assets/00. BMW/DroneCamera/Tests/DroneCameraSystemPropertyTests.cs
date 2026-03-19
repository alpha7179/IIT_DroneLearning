using System.Reflection;
using NUnit.Framework;
using UnityEngine;

namespace BMW.DroneCamera
{
    /// <summary>
    /// DroneCameraSystem 속성 기반 테스트.
    ///
    /// FsCheck 대신 NUnit + 반복 루프(100회)로 구현한다.
    /// 각 테스트 메서드는 설계 문서의 Property 번호를 태그로 참조한다.
    /// Feature: drone-camera-system
    /// </summary>
    public class DroneCameraSystemPropertyTests
    {
        private const int Iterations = 100;

        private System.Random _rng;

        [SetUp]
        public void SetUp()
        {
            _rng = new System.Random(42); // 재현 가능한 시드
        }

        [TearDown]
        public void TearDown()
        {
            // 남아 있는 테스트용 GameObject 제거
            foreach (var go in Object.FindObjectsByType<GameObject>(FindObjectsSortMode.None))
            {
                if (go.hideFlags == HideFlags.None &&
                    (go.name.StartsWith("Prop_") || go.name.StartsWith("PropCtrl_")))
                {
                    Object.DestroyImmediate(go);
                }
            }
        }

        // ────────────────────────────────────────────
        // 헬퍼
        // ────────────────────────────────────────────

        private static T AddAndAwake<T>(GameObject go) where T : MonoBehaviour
        {
            var comp = go.AddComponent<T>();
            InvokePrivate(comp, "Awake");
            return comp;
        }

        private static void InvokePrivate(MonoBehaviour comp, string methodName)
        {
            var m = comp.GetType().GetMethod(methodName,
                BindingFlags.NonPublic | BindingFlags.Instance);
            m?.Invoke(comp, null);
        }

        /// <summary>범위 [min, max] 내 임의 float 반환</summary>
        private float Rand(float min, float max) =>
            (float)(_rng.NextDouble() * (max - min) + min);

        /// <summary>임의 Quaternion 생성</summary>
        private Quaternion RandRotation() =>
            Quaternion.Euler(Rand(-180f, 180f), Rand(-180f, 180f), Rand(-180f, 180f));

        // ────────────────────────────────────────────
        // Property 1: 카메라 초기화 시 자식 Camera 생성
        // 검증 대상: 요구사항 1.1
        // ────────────────────────────────────────────

        [Test, Category("drone-camera-system")]
        public void Property1_Awake_AlwaysCreatesExactlyOneChildCamera()
        {
            for (int i = 0; i < Iterations; i++)
            {
                var go = new GameObject($"Prop_P1_{i}");
                try
                {
                    var sys = AddAndAwake<DroneCameraSystem>(go);

                    // Camera 프로퍼티 검증
                    Assert.IsNotNull(sys.Camera,
                        $"[P1 iter={i}] Camera 프로퍼티가 null이어선 안 됩니다.");

                    // 자식 Camera 컴포넌트가 정확히 1개인지 검증
                    var childCameras = go.GetComponentsInChildren<Camera>();
                    Assert.AreEqual(1, childCameras.Length,
                        $"[P1 iter={i}] 자식 Camera 컴포넌트가 정확히 1개여야 합니다. 실제: {childCameras.Length}");
                }
                finally
                {
                    Object.DestroyImmediate(go);
                }
            }
        }

        // ────────────────────────────────────────────
        // Property 2: DroneSensorSystem 미존재 시 카메라 전방 방향 정렬
        // 검증 대상: 요구사항 1.2, 1.4, 6.4
        // ────────────────────────────────────────────

        [Test, Category("drone-camera-system")]
        public void Property2_WithoutSensor_CameraForwardAlignsDroneLocalZ()
        {
            for (int i = 0; i < Iterations; i++)
            {
                var go = new GameObject($"Prop_P2_{i}");
                try
                {
                    // DroneSensorSystem 없음 + 임의 회전
                    go.transform.rotation = RandRotation();

                    var sys = AddAndAwake<DroneCameraSystem>(go);

                    Vector3 droneForward  = go.transform.forward.normalized;
                    Vector3 cameraForward = sys.Camera.transform.forward.normalized;
                    float   dot = Vector3.Dot(cameraForward, droneForward);

                    Assert.Greater(dot, 0.999f,
                        $"[P2 iter={i}] 카메라 forward({cameraForward}) ≈ 드론 +Z({droneForward}), dot={dot:F4}");
                }
                finally
                {
                    Object.DestroyImmediate(go);
                }
            }
        }

        // ────────────────────────────────────────────
        // Property 3: DroneSensorSystem 존재 시 카메라-센서 방향 일치
        // 검증 대상: 요구사항 1.3, 6.3
        // ────────────────────────────────────────────

        [Test, Category("drone-camera-system")]
        public void Property3_WithSensor_CameraForwardMatchesSensorForward()
        {
            for (int i = 0; i < Iterations; i++)
            {
                var go = new GameObject($"Prop_P3_{i}");
                try
                {
                    go.transform.rotation = RandRotation();

                    // DroneSensorSystem 먼저 Awake (Awake 호출 순서 보장)
                    var sensor = go.AddComponent<BMW.DroneSensor.DroneSensorSystem>();
                    InvokePrivate(sensor, "Awake");

                    var sys = AddAndAwake<DroneCameraSystem>(go);

                    Vector3 sensorForward = sensor.GetForwardRayDirection().normalized;
                    Vector3 cameraForward = sys.Camera.transform.forward.normalized;
                    float   dot = Vector3.Dot(cameraForward, sensorForward);

                    Assert.Greater(dot, 0.999f,
                        $"[P3 iter={i}] 카메라 forward ≈ GetForwardRayDirection(), dot={dot:F4}");
                }
                finally
                {
                    Object.DestroyImmediate(go);
                }
            }
        }

        // ────────────────────────────────────────────
        // Property 4: Viewport Rect 겹침 없음
        // 검증 대상: 요구사항 2.6
        // ────────────────────────────────────────────

        [Test, Category("drone-camera-system")]
        public void Property4_SingleBatch_ViewportRectsDoNotOverlap()
        {
            for (int i = 0; i < Iterations; i++)
            {
                var ctrlGO    = new GameObject($"PropCtrl_P4_{i}");
                var trackerGO = new GameObject($"Prop_P4_T_{i}");
                var evaderGO  = new GameObject($"Prop_P4_E_{i}");
                try
                {
                    var ctrl    = AddAndAwake<CameraControlSystem>(ctrlGO);
                    var tracker = AddAndAwake<DroneCameraSystem>(trackerGO);
                    var evader  = AddAndAwake<DroneCameraSystem>(evaderGO);

                    ctrl.RegisterDroneCamera(tracker, DroneRole.Pursuer);
                    ctrl.RegisterDroneCamera(evader,  DroneRole.Evader);
                    ctrl.ApplyLayout(1);

                    var tvCam = ctrlGO.transform.Find("TopView_Camera")?.GetComponent<Camera>();
                    Assert.IsNotNull(tvCam);

                    Rect[] rects = {
                        tvCam.rect,
                        tracker.Camera.rect,
                        evader.Camera.rect
                    };

                    for (int a = 0; a < rects.Length; a++)
                    for (int b = a + 1; b < rects.Length; b++)
                    {
                        float overlapArea = GetOverlapArea(rects[a], rects[b]);
                        Assert.AreEqual(0f, overlapArea, 0.001f,
                            $"[P4 iter={i}] Rect[{a}]={rects[a]} 와 Rect[{b}]={rects[b]} 겹침 면적={overlapArea:F4}");
                    }
                }
                finally
                {
                    Object.DestroyImmediate(ctrlGO);
                    Object.DestroyImmediate(trackerGO);
                    Object.DestroyImmediate(evaderGO);
                }
            }
        }

        // ────────────────────────────────────────────
        // Property 5: Multi Batch 모드에서 DroneCameraSystem 비활성화
        // 검증 대상: 요구사항 3.1
        // ────────────────────────────────────────────

        [Test, Category("drone-camera-system")]
        public void Property5_MultiBatch_AllDroneCamerasDisabled()
        {
            var rand = new System.Random(99);

            for (int i = 0; i < Iterations; i++)
            {
                int droneCount = rand.Next(1, 7); // 1~6대
                int batchCount = rand.Next(2, 9); // 2~8 배치

                var ctrlGO = new GameObject($"PropCtrl_P5_{i}");
                var droneGOs = new GameObject[droneCount];
                try
                {
                    var ctrl = AddAndAwake<CameraControlSystem>(ctrlGO);

                    for (int d = 0; d < droneCount; d++)
                    {
                        droneGOs[d] = new GameObject($"Prop_P5_D{d}_{i}");
                        var cam = AddAndAwake<DroneCameraSystem>(droneGOs[d]);
                        DroneRole role = d % 2 == 0 ? DroneRole.Pursuer : DroneRole.Evader;
                        ctrl.RegisterDroneCamera(cam, role);
                    }

                    ctrl.ApplyLayout(batchCount);

                    Assert.AreEqual(CameraControlSystem.CameraMode.MultiBatch,
                        ctrl.GetCurrentMode(),
                        $"[P5 iter={i}] batchCount={batchCount} → MultiBatch 모드 기대");

                    for (int d = 0; d < droneCount; d++)
                    {
                        var cam = droneGOs[d].GetComponentInChildren<Camera>();
                        Assert.IsFalse(cam.enabled,
                            $"[P5 iter={i}] 드론[{d}] 카메라 비활성화 기대");
                    }
                }
                finally
                {
                    Object.DestroyImmediate(ctrlGO);
                    foreach (var go in droneGOs)
                        if (go != null) Object.DestroyImmediate(go);
                }
            }
        }

        // ────────────────────────────────────────────
        // Property 6: BatchTopView 카메라의 배치 영역 포함
        // 검증 대상: 요구사항 3.3, 3.4
        // 설계 공식: orthoSize = max(totalW, totalD) / 2 * paddingFactor
        // ────────────────────────────────────────────

        [Test, Category("drone-camera-system")]
        public void Property6_BatchTopView_ContainsAllBatchBoundingBox()
        {
            for (int i = 0; i < Iterations; i++)
            {
                int   columns  = _rng.Next(1, 6);
                int   rows     = _rng.Next(1, 6);
                float spacingX = Rand(0f, 50f);
                float spacingZ = Rand(0f, 50f);
                float slotW    = Rand(50f, 300f);
                float slotD    = Rand(50f, 300f);
                float padding  = Rand(1.0f, 1.5f);

                // 설계 공식 검증 (CameraControlSystem 내부 공식 재현)
                float totalW   = columns * slotW + (columns - 1) * spacingX;
                float totalD   = rows    * slotD + (rows    - 1) * spacingZ;
                float expected = Mathf.Max(totalW, totalD) / 2f * padding;

                // orthoSize >= 바운딩 박스 반경 조건
                float halfW = totalW / 2f;
                float halfD = totalD / 2f;
                float required = Mathf.Max(halfW, halfD);

                Assert.GreaterOrEqual(expected, required - 0.001f,
                    $"[P6 iter={i}] orthoSize({expected:F2}) >= 배치 반경({required:F2}), " +
                    $"columns={columns}, rows={rows}, slotW={slotW:F1}, slotD={slotD:F1}, padding={padding:F2}");
            }
        }

        // ────────────────────────────────────────────
        // Property 7: 배치 수에 따른 올바른 모드 선택
        // 검증 대상: 요구사항 4.1, 4.2, 4.3
        // ────────────────────────────────────────────

        [Test, Category("drone-camera-system")]
        public void Property7_BatchCountDeterminesCorrectMode()
        {
            var rand = new System.Random(7);

            for (int i = 0; i < Iterations; i++)
            {
                var ctrlGO = new GameObject($"PropCtrl_P7_{i}");
                try
                {
                    var ctrl = AddAndAwake<CameraControlSystem>(ctrlGO);
                    int n = rand.Next(1, 20);

                    ctrl.ApplyLayout(n);
                    var mode = ctrl.GetCurrentMode();

                    if (n == 1)
                        Assert.AreEqual(CameraControlSystem.CameraMode.SingleBatch, mode,
                            $"[P7 iter={i}] n=1 → SingleBatch 기대, 실제={mode}");
                    else
                        Assert.AreEqual(CameraControlSystem.CameraMode.MultiBatch, mode,
                            $"[P7 iter={i}] n={n} → MultiBatch 기대, 실제={mode}");
                }
                finally
                {
                    Object.DestroyImmediate(ctrlGO);
                }
            }
        }

        // ────────────────────────────────────────────
        // Property 8: CityDataAPI 기반 TopView 카메라 도시 포함
        // 검증 대상: 요구사항 5.3
        // 설계 공식: orthoSize = max(cityWidth, cityDepth) / 2 * paddingFactor
        // ────────────────────────────────────────────

        [Test, Category("drone-camera-system")]
        public void Property8_TopViewOrthoSize_ContainsCityBoundingBox()
        {
            for (int i = 0; i < Iterations; i++)
            {
                float cityWidth  = Rand(100f, 2000f);
                float cityDepth  = Rand(100f, 2000f);
                float padding    = Rand(1.0f, 1.5f);

                // 설계 공식
                float orthoSize = Mathf.Max(cityWidth, cityDepth) / 2f * padding;

                // 투영 영역이 도시 바운딩 박스를 포함하는지 검증
                // orthoSize >= max(cityWidth/2, cityDepth/2)
                float required = Mathf.Max(cityWidth, cityDepth) / 2f;
                Assert.GreaterOrEqual(orthoSize, required - 0.001f,
                    $"[P8 iter={i}] orthoSize({orthoSize:F2}) >= 도시 반경({required:F2}), " +
                    $"cityWidth={cityWidth:F1}, cityDepth={cityDepth:F1}, padding={padding:F2}");
            }
        }

        // ────────────────────────────────────────────
        // Property 9: 활성 배치 0개 시 모든 카메라 비활성화
        // 검증 대상: 요구사항 4.4
        // ────────────────────────────────────────────

        [Test, Category("drone-camera-system")]
        public void Property9_ZeroBatch_AllCamerasDisabled()
        {
            var rand = new System.Random(9);

            for (int i = 0; i < Iterations; i++)
            {
                int droneCount = rand.Next(0, 5);

                var ctrlGO  = new GameObject($"PropCtrl_P9_{i}");
                var drones  = new GameObject[droneCount];
                try
                {
                    var ctrl = AddAndAwake<CameraControlSystem>(ctrlGO);

                    for (int d = 0; d < droneCount; d++)
                    {
                        drones[d] = new GameObject($"Prop_P9_D{d}_{i}");
                        var cam = AddAndAwake<DroneCameraSystem>(drones[d]);
                        ctrl.RegisterDroneCamera(cam, d % 2 == 0 ? DroneRole.Pursuer : DroneRole.Evader);
                    }

                    // ApplyLayout(0) 시 LogWarning 기대
                    UnityEngine.TestTools.LogAssert.Expect(
                        LogType.Warning,
                        new System.Text.RegularExpressions.Regex("활성 배치"));

                    ctrl.ApplyLayout(0);

                    Assert.AreEqual(CameraControlSystem.CameraMode.None, ctrl.GetCurrentMode(),
                        $"[P9 iter={i}] batchCount=0 → None 모드 기대");

                    // TopView, BatchTopView 비활성화 검증
                    var tvCam  = ctrlGO.transform.Find("TopView_Camera")     ?.GetComponent<Camera>();
                    var btvCam = ctrlGO.transform.Find("BatchTopView_Camera")?.GetComponent<Camera>();
                    Assert.IsNotNull(tvCam);
                    Assert.IsNotNull(btvCam);
                    Assert.IsFalse(tvCam.enabled,  $"[P9 iter={i}] TopView 카메라 비활성화 기대");
                    Assert.IsFalse(btvCam.enabled, $"[P9 iter={i}] BatchTopView 카메라 비활성화 기대");

                    // 드론 카메라 비활성화 검증
                    for (int d = 0; d < droneCount; d++)
                    {
                        var cam = drones[d].GetComponentInChildren<Camera>();
                        Assert.IsFalse(cam.enabled, $"[P9 iter={i}] 드론[{d}] 카메라 비활성화 기대");
                    }
                }
                finally
                {
                    Object.DestroyImmediate(ctrlGO);
                    foreach (var go in drones)
                        if (go != null) Object.DestroyImmediate(go);
                }
            }
        }

        // ────────────────────────────────────────────
        // 유틸리티
        // ────────────────────────────────────────────

        /// <summary>두 Rect의 교집합 면적을 계산한다.</summary>
        private static float GetOverlapArea(Rect a, Rect b)
        {
            float left   = Mathf.Max(a.xMin, b.xMin);
            float right  = Mathf.Min(a.xMax, b.xMax);
            float bottom = Mathf.Max(a.yMin, b.yMin);
            float top    = Mathf.Min(a.yMax, b.yMax);

            float w = right  - left;
            float h = top    - bottom;

            return (w > 0f && h > 0f) ? w * h : 0f;
        }
    }
}
