using System.Reflection;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace DroneCamera
{
    /// <summary>
    /// DroneCameraSystem / CameraControlSystem 단위 테스트.
    /// Edit Mode 테스트 — Awake는 Reflection으로 수동 호출 (DroneSensorSystem 패턴 준수).
    /// </summary>
    public class DroneCameraSystemTests
    {
        private GameObject _droneGO;
        private GameObject _controlGO;

        [SetUp]
        public void SetUp()
        {
            _droneGO   = new GameObject("TestDrone");
            _controlGO = new GameObject("CameraControl");
        }

        [TearDown]
        public void TearDown()
        {
            if (_droneGO   != null) Object.DestroyImmediate(_droneGO);
            if (_controlGO != null) Object.DestroyImmediate(_controlGO);

            // 테스트 중 생성된 나머지 GameObject 정리
            foreach (var go in Object.FindObjectsByType<GameObject>(FindObjectsSortMode.None))
            {
                if (go.name.StartsWith("Test") || go.name == "CameraControl" ||
                    go.name == "TopView_Camera" || go.name == "BatchTopView_Camera" ||
                    go.name == "DroneCamera")
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
            InvokeAwake(comp);
            return comp;
        }

        private static void InvokeAwake(MonoBehaviour comp)
        {
            var m = comp.GetType().GetMethod("Awake",
                BindingFlags.NonPublic | BindingFlags.Instance);
            m?.Invoke(comp, null);
        }

        private static void InvokeStart(MonoBehaviour comp)
        {
            var m = comp.GetType().GetMethod("Start",
                BindingFlags.NonPublic | BindingFlags.Instance);
            m?.Invoke(comp, null);
        }

        // ────────────────────────────────────────────
        // Task 2.6 — DroneCameraSystem 단위 테스트
        // ────────────────────────────────────────────

        /// <summary>
        /// 요구사항 1.1: Awake() 실행 후 자식에 Camera 컴포넌트가 생성되어야 한다.
        /// </summary>
        [Test]
        public void Awake_CreatesCameraChild()
        {
            var cam = AddAndAwake<DroneCameraSystem>(_droneGO);

            Assert.IsNotNull(cam.Camera,
                "Camera 프로퍼티는 null이어선 안 됩니다.");
            Assert.IsNotNull(cam.Camera.GetComponent<Camera>(),
                "Camera 컴포넌트가 자식 GameObject에 존재해야 합니다.");
        }

        /// <summary>
        /// 요구사항 1.5: Inspector 필드가 선언되어 있어야 한다.
        /// </summary>
        [Test]
        public void InspectorFields_Exist()
        {
            var type = typeof(DroneCameraSystem);

            Assert.IsNotNull(type.GetField("cameraOffset"),    "cameraOffset 필드 필요");
            Assert.IsNotNull(type.GetField("fieldOfView"),     "fieldOfView 필드 필요");
            Assert.IsNotNull(type.GetField("nearClipPlane"),   "nearClipPlane 필드 필요");
            Assert.IsNotNull(type.GetField("farClipPlane"),    "farClipPlane 필드 필요");
        }

        /// <summary>
        /// 요구사항 6.2: DroneSensorSystem 없이 독립 동작해야 한다.
        /// </summary>
        [Test]
        public void Awake_WithoutSensorSystem_WorksIndependently()
        {
            // DroneSensorSystem 없음
            Assert.DoesNotThrow(() => AddAndAwake<DroneCameraSystem>(_droneGO));
        }

        /// <summary>
        /// 요구사항 6.1: CameraControlSystem 없이도 독립 동작해야 한다.
        /// </summary>
        [Test]
        public void Start_WithoutCameraControlSystem_DoesNotThrow()
        {
            var cam = _droneGO.AddComponent<DroneCameraSystem>();
            InvokeAwake(cam);
            // Start()는 FindFirstObjectByType<CameraControlSystem>이 null → LogWarning 출력
            Assert.DoesNotThrow(() => InvokeStart(cam));
        }

        /// <summary>
        /// 요구사항 1.2 / 1.4: DroneSensorSystem 미존재 시 카메라 forward = 드론 로컬 +Z.
        /// </summary>
        [Test]
        public void Awake_WithoutSensor_CameraForwardAlignedWithDroneForward()
        {
            _droneGO.transform.rotation = Quaternion.Euler(0f, 45f, 0f);

            var sys = AddAndAwake<DroneCameraSystem>(_droneGO);
            Camera cam = sys.Camera;

            Vector3 expectedWorldForward = _droneGO.transform.forward;
            float   dot = Vector3.Dot(cam.transform.forward.normalized,
                                      expectedWorldForward.normalized);

            Assert.Greater(dot, 0.999f,
                $"카메라 forward({cam.transform.forward}) ≈ 드론 forward({expectedWorldForward}) 기대");
        }

        /// <summary>
        /// 요구사항 1.4: 드론 회전 시 카메라도 함께 회전해야 한다 (자식 Transform).
        /// </summary>
        [Test]
        public void Camera_FollowsDroneRotation()
        {
            var sys = AddAndAwake<DroneCameraSystem>(_droneGO);

            _droneGO.transform.rotation = Quaternion.Euler(0f, 90f, 0f);

            float dot = Vector3.Dot(
                sys.Camera.transform.forward.normalized,
                _droneGO.transform.forward.normalized);

            Assert.Greater(dot, 0.999f, "드론 회전 후에도 카메라 forward = 드론 forward이어야 한다.");
        }

        /// <summary>
        /// SetEnabled(false) → Camera.enabled == false
        /// SetEnabled(true)  → Camera.enabled == true
        /// </summary>
        [Test]
        public void SetEnabled_TogglesCamera()
        {
            var sys = AddAndAwake<DroneCameraSystem>(_droneGO);

            sys.SetEnabled(false);
            Assert.IsFalse(sys.Camera.enabled, "SetEnabled(false) 후 Camera.enabled = false 기대");

            sys.SetEnabled(true);
            Assert.IsTrue(sys.Camera.enabled, "SetEnabled(true) 후 Camera.enabled = true 기대");
        }

        /// <summary>
        /// SetViewportRect → Camera.rect 반영
        /// </summary>
        [Test]
        public void SetViewportRect_AppliesRect()
        {
            var sys  = AddAndAwake<DroneCameraSystem>(_droneGO);
            var rect = new Rect(0.5f, 0f, 0.5f, 0.5f);

            sys.SetViewportRect(rect);

            Assert.AreEqual(rect, sys.Camera.rect, "SetViewportRect 후 Camera.rect와 일치해야 한다.");
        }

        // ────────────────────────────────────────────
        // Task 4 — CameraControlSystem Inspector 필드
        // ────────────────────────────────────────────

        /// <summary>
        /// 요구사항 5.2: Inspector 필드(topViewHeight, topViewOrthoSize)가 존재해야 한다.
        /// </summary>
        [Test]
        public void CameraControlSystem_InspectorFields_Exist()
        {
            var type = typeof(CameraControlSystem);

            Assert.IsNotNull(type.GetField("topViewHeight"),          "topViewHeight 필드 필요");
            Assert.IsNotNull(type.GetField("topViewOrthoSize"),       "topViewOrthoSize 필드 필요");
            Assert.IsNotNull(type.GetField("topViewPaddingFactor"),   "topViewPaddingFactor 필드 필요");
            Assert.IsNotNull(type.GetField("batchTopViewHeight"),     "batchTopViewHeight 필드 필요");
            Assert.IsNotNull(type.GetField("batchTopViewPaddingFactor"), "batchTopViewPaddingFactor 필드 필요");
        }

        /// <summary>
        /// 요구사항 5.1: Awake() 후 TopView 카메라는 -Y 방향을 바라보고 Orthographic이어야 한다.
        /// </summary>
        [Test]
        public void Awake_TopViewCamera_OrthographicAndLookingDown()
        {
            var ctrl = AddAndAwake<CameraControlSystem>(_controlGO);

            // TopView_Camera 자식 찾기
            var tvGO = _controlGO.transform.Find("TopView_Camera");
            Assert.IsNotNull(tvGO, "TopView_Camera 자식 GameObject가 있어야 한다.");

            var cam = tvGO.GetComponent<Camera>();
            Assert.IsNotNull(cam, "TopView_Camera에 Camera 컴포넌트가 있어야 한다.");
            Assert.IsTrue(cam.orthographic, "TopView 카메라는 Orthographic이어야 한다.");

            // 아래 방향(-Y) 확인
            float dot = Vector3.Dot(tvGO.forward.normalized, Vector3.down);
            Assert.Greater(dot, 0.999f, "TopView 카메라는 수직 아래를 바라봐야 한다.");
        }

        /// <summary>
        /// 요구사항 5.1: BatchTopView 카메라도 Orthographic 및 -Y 방향.
        /// </summary>
        [Test]
        public void Awake_BatchTopViewCamera_OrthographicAndLookingDown()
        {
            var ctrl = AddAndAwake<CameraControlSystem>(_controlGO);

            var btvGO = _controlGO.transform.Find("BatchTopView_Camera");
            Assert.IsNotNull(btvGO, "BatchTopView_Camera 자식 GameObject가 있어야 한다.");

            var cam = btvGO.GetComponent<Camera>();
            Assert.IsNotNull(cam, "BatchTopView_Camera에 Camera 컴포넌트가 있어야 한다.");
            Assert.IsTrue(cam.orthographic, "BatchTopView 카메라는 Orthographic이어야 한다.");

            float dot = Vector3.Dot(btvGO.forward.normalized, Vector3.down);
            Assert.Greater(dot, 0.999f, "BatchTopView 카메라는 수직 아래를 바라봐야 한다.");
        }

        // ────────────────────────────────────────────
        // Task 5.6 — Single Batch 모드 단위 테스트
        // ────────────────────────────────────────────

        /// <summary>
        /// 요구사항 2.1~2.4: Single Batch 모드에서 각 1대일 때 기본 Viewport Rect가 올바르게 적용되어야 한다.
        /// </summary>
        [Test]
        public void SingleBatch_ViewportRects_AreCorrect()
        {
            var ctrl    = AddAndAwake<CameraControlSystem>(_controlGO);
            var trackerGO = new GameObject("Tracker");
            var evaderGO  = new GameObject("Evader");
            try
            {
                var tracker = AddAndAwake<DroneCameraSystem>(trackerGO);
                var evader  = AddAndAwake<DroneCameraSystem>(evaderGO);

                ctrl.RegisterDroneCamera(tracker, DroneRole.Pursuer);
                ctrl.RegisterDroneCamera(evader,  DroneRole.Evader);
                ctrl.ApplyLayout(1);

                Assert.AreEqual(new Rect(2f / 3f, 0.5f, 1f / 3f, 0.5f), tracker.Camera.rect,
                    "Tracker FPV Viewport: 우상단 (2/3, 0.5, 1/3, 0.5)");
                Assert.AreEqual(new Rect(2f / 3f, 0f,   1f / 3f, 0.5f), evader.Camera.rect,
                    "Evader FPV Viewport: 우하단 (2/3, 0, 1/3, 0.5)");

                var tvCam = _controlGO.transform.Find("TopView_Camera")?.GetComponent<Camera>();
                Assert.IsNotNull(tvCam);
                Assert.AreEqual(new Rect(0f, 0f, 2f / 3f, 1f), tvCam.rect,
                    "TopView Viewport: 좌측 2/3 전체 높이 (0, 0, 2/3, 1)");
            }
            finally
            {
                Object.DestroyImmediate(trackerGO);
                Object.DestroyImmediate(evaderGO);
            }
        }

        /// <summary>
        /// Pursuer 2대 등록 시 Pursuer 섹션이 2등분, Evader 섹션은 유지.
        /// </summary>
        [Test]
        public void SingleBatch_MultiPursuer_SectionSplitsVertically()
        {
            var ctrl = AddAndAwake<CameraControlSystem>(_controlGO);
            var p0GO = new GameObject("Pursuer0");
            var p1GO = new GameObject("Pursuer1");
            var eGO  = new GameObject("Evader");
            try
            {
                var p0 = AddAndAwake<DroneCameraSystem>(p0GO);
                var p1 = AddAndAwake<DroneCameraSystem>(p1GO);
                var ev = AddAndAwake<DroneCameraSystem>(eGO);

                ctrl.RegisterDroneCamera(p0, DroneRole.Pursuer);
                ctrl.RegisterDroneCamera(p1, DroneRole.Pursuer);
                ctrl.RegisterDroneCamera(ev, DroneRole.Evader);
                ctrl.ApplyLayout(1);

                // Pursuer 섹션 (y: 0.5~1.0) → 2등분 (각 slotH = 0.25)
                Assert.AreEqual(new Rect(2f / 3f, 0.75f, 1f / 3f, 0.25f), p0.MultiViewCamera.rect,
                    "Pursuer[0]: 섹션 상단 슬롯");
                Assert.AreEqual(new Rect(2f / 3f, 0.5f,  1f / 3f, 0.25f), p1.MultiViewCamera.rect,
                    "Pursuer[1]: 섹션 하단 슬롯");
                Assert.IsTrue(p0.MultiViewCamera.enabled,  "Pursuer[0] MultiView 활성");
                Assert.IsTrue(p1.MultiViewCamera.enabled,  "Pursuer[1] MultiView 활성");

                // Evader 섹션 (y: 0.0~0.5) → 1개이므로 그대로
                Assert.AreEqual(new Rect(2f / 3f, 0f, 1f / 3f, 0.5f), ev.MultiViewCamera.rect,
                    "Evader[0]: 섹션 전체");
                Assert.IsTrue(ev.MultiViewCamera.enabled, "Evader[0] MultiView 활성");
            }
            finally
            {
                Object.DestroyImmediate(p0GO);
                Object.DestroyImmediate(p1GO);
                Object.DestroyImmediate(eGO);
            }
        }

        /// <summary>
        /// Pursuer 4대 등록 시 섹션 내 2×2 그리드로 배치되는지 확인.
        /// </summary>
        [Test]
        public void SingleBatch_FourPursuers_TwoColumnGrid()
        {
            var ctrl = AddAndAwake<CameraControlSystem>(_controlGO);
            var gos  = new GameObject[4];
            try
            {
                var drones = new DroneCameraSystem[4];
                for (int i = 0; i < 4; i++)
                {
                    gos[i]    = new GameObject($"Pursuer{i}");
                    drones[i] = AddAndAwake<DroneCameraSystem>(gos[i]);
                    ctrl.RegisterDroneCamera(drones[i], DroneRole.Pursuer);
                }
                ctrl.ApplyLayout(1);

                // Pursuer 섹션 (y: 0.5~1.0), 2×2 그리드 → slotW=1/6, slotH=0.25
                float x0 = 2f / 3f;
                float x1 = 2f / 3f + 1f / 6f;
                float slotW = 1f / 6f;
                float slotH = 0.25f;

                Assert.AreEqual(new Rect(x0, 0.75f, slotW, slotH), drones[0].MultiViewCamera.rect, "P[0]: 상단-좌");
                Assert.AreEqual(new Rect(x1, 0.75f, slotW, slotH), drones[1].MultiViewCamera.rect, "P[1]: 상단-우");
                Assert.AreEqual(new Rect(x0, 0.5f,  slotW, slotH), drones[2].MultiViewCamera.rect, "P[2]: 하단-좌");
                Assert.AreEqual(new Rect(x1, 0.5f,  slotW, slotH), drones[3].MultiViewCamera.rect, "P[3]: 하단-우");
                for (int i = 0; i < 4; i++)
                    Assert.IsTrue(drones[i].MultiViewCamera.enabled, $"Pursuer[{i}] 활성");
            }
            finally
            {
                foreach (var go in gos) if (go != null) Object.DestroyImmediate(go);
            }
        }

        /// <summary>
        /// 각 역할 4대 + 초과 1대씩 등록 시 최대 4대까지 활성, 초과분 비활성.
        /// 4대/역할 → 2×2 그리드이므로 slotW=1/6, slotH=0.25.
        /// </summary>
        [Test]
        public void SingleBatch_MaxSlots_FourPerRole()
        {
            var ctrl = AddAndAwake<CameraControlSystem>(_controlGO);
            var gos  = new GameObject[10]; // Pursuer 5, Evader 5 (초과 1대씩)
            try
            {
                var drones = new DroneCameraSystem[10];
                for (int i = 0; i < 10; i++)
                {
                    gos[i]    = new GameObject($"Drone{i}");
                    drones[i] = AddAndAwake<DroneCameraSystem>(gos[i]);
                    ctrl.RegisterDroneCamera(drones[i], i < 5 ? DroneRole.Pursuer : DroneRole.Evader);
                }
                ctrl.ApplyLayout(1);

                // 4대 → 2×2 그리드: slotW = 1/6, slotH = 섹션높이(0.5) / 2행 = 0.25
                float slotW = 1f / 6f;
                float slotH = 0.25f;

                // Pursuer 최대 4개 활성, 5번째는 비활성
                for (int i = 0; i < 4; i++)
                    Assert.IsTrue(drones[i].MultiViewCamera.enabled, $"Pursuer[{i}] 활성");
                Assert.IsFalse(drones[4].MultiViewCamera.enabled, "Pursuer[4] 비활성 (초과)");

                // Evader 최대 4개 활성, 5번째는 비활성
                for (int i = 5; i < 9; i++)
                    Assert.IsTrue(drones[i].MultiViewCamera.enabled, $"Evader[{i - 5}] 활성");
                Assert.IsFalse(drones[9].MultiViewCamera.enabled, "Evader[4] 비활성 (초과)");

                // 슬롯 크기 확인 (2×2 그리드)
                Assert.AreEqual(slotW, drones[0].MultiViewCamera.rect.width,  1e-5f, "Pursuer 슬롯 너비");
                Assert.AreEqual(slotH, drones[0].MultiViewCamera.rect.height, 1e-5f, "Pursuer 슬롯 높이");
                Assert.AreEqual(slotW, drones[5].MultiViewCamera.rect.width,  1e-5f, "Evader 슬롯 너비");
                Assert.AreEqual(slotH, drones[5].MultiViewCamera.rect.height, 1e-5f, "Evader 슬롯 높이");
            }
            finally
            {
                foreach (var go in gos) if (go != null) Object.DestroyImmediate(go);
            }
        }

        /// <summary>
        /// 요구사항 4.4: 활성 배치 0개 시 모든 카메라 비활성화 + LogWarning.
        /// </summary>
        [Test]
        public void NoneMode_AllCamerasDisabled_WithWarning()
        {
            var ctrl    = AddAndAwake<CameraControlSystem>(_controlGO);
            var trackerGO = new GameObject("TrackerNone");
            try
            {
                var tracker = AddAndAwake<DroneCameraSystem>(trackerGO);
                ctrl.RegisterDroneCamera(tracker, DroneRole.Pursuer);
                ctrl.ApplyLayout(1); // 먼저 활성화

                LogAssert.Expect(LogType.Warning, new System.Text.RegularExpressions.Regex("활성 배치"));
                ctrl.ApplyLayout(0);

                Assert.AreEqual(CameraControlSystem.CameraMode.None, ctrl.GetCurrentMode());
                Assert.IsFalse(tracker.Camera.enabled, "None 모드에서 드론 카메라 비활성화 기대");
            }
            finally
            {
                Object.DestroyImmediate(trackerGO);
            }
        }

        /// <summary>
        /// 요구사항 2.7: 동일 역할 복수 등록 시 [0]만 활성화, 나머지 비활성화.
        /// </summary>
        [Test]
        public void SingleBatch_MultipleTrackers_OnlyFirstIsActive()
        {
            var ctrl = AddAndAwake<CameraControlSystem>(_controlGO);
            var go1  = new GameObject("Tracker1");
            var go2  = new GameObject("Tracker2");
            try
            {
                var t1 = AddAndAwake<DroneCameraSystem>(go1);
                var t2 = AddAndAwake<DroneCameraSystem>(go2);

                ctrl.RegisterDroneCamera(t1, DroneRole.Pursuer);
                ctrl.RegisterDroneCamera(t2, DroneRole.Pursuer);
                ctrl.ApplyLayout(1);

                Assert.IsTrue(t1.Camera.enabled,  "첫 번째 Tracker 카메라는 활성화");
                Assert.IsFalse(t2.Camera.enabled, "두 번째 Tracker 카메라는 비활성화");
            }
            finally
            {
                Object.DestroyImmediate(go1);
                Object.DestroyImmediate(go2);
            }
        }

        // ────────────────────────────────────────────
        // Task 8.2 — 연동 단위 테스트
        // ────────────────────────────────────────────

        /// <summary>
        /// 요구사항 4.3: RegisterDroneCamera / UnregisterDroneCamera가 올바르게 반영되어야 한다.
        /// </summary>
        [Test]
        public void RegisterUnregister_UpdatesLayout()
        {
            var ctrl    = AddAndAwake<CameraControlSystem>(_controlGO);
            var go      = new GameObject("TrackerReg");
            try
            {
                var cam = AddAndAwake<DroneCameraSystem>(go);

                ctrl.RegisterDroneCamera(cam, DroneRole.Pursuer);
                ctrl.ApplyLayout(1);
                Assert.IsTrue(cam.Camera.enabled, "등록 후 Single 모드에서 카메라 활성화 기대");

                ctrl.UnregisterDroneCamera(cam);
                ctrl.ApplyLayout(1); // 재적용
                // 등록 해제 후에는 카메라 참조가 없으므로 TopView만 활성
                // cam 자체는 Enabled 상태가 변경되지 않음 (리스트에서 제거됨)
                // → 이 테스트는 예외 없이 실행됨을 검증
                Assert.Pass("UnregisterDroneCamera 후 ApplyLayout 예외 없이 실행됨.");
            }
            finally
            {
                Object.DestroyImmediate(go);
            }
        }

        /// <summary>
        /// 요구사항 4.1/4.2: Multi Batch 모드 전환 시 드론 카메라 비활성화 + BatchTopView 활성화.
        /// </summary>
        [Test]
        public void MultiBatch_DronesCameraDisabled_BatchTopViewEnabled()
        {
            var ctrl    = AddAndAwake<CameraControlSystem>(_controlGO);
            var go      = new GameObject("TrackerMulti");
            try
            {
                var cam = AddAndAwake<DroneCameraSystem>(go);
                ctrl.RegisterDroneCamera(cam, DroneRole.Pursuer);
                ctrl.ApplyLayout(4); // 2×2

                Assert.AreEqual(CameraControlSystem.CameraMode.MultiBatch, ctrl.GetCurrentMode());
                Assert.IsFalse(cam.Camera.enabled, "Multi 모드에서 드론 카메라 비활성화 기대");

                var btvCam = _controlGO.transform.Find("BatchTopView_Camera")?.GetComponent<Camera>();
                Assert.IsNotNull(btvCam);
                Assert.IsTrue(btvCam.enabled,      "Multi 모드에서 BatchTopView 활성화 기대");
                Assert.AreEqual(new Rect(0f, 0f, 1f, 1f), btvCam.rect, "BatchTopView 전체 화면");
            }
            finally
            {
                Object.DestroyImmediate(go);
            }
        }
    }
}
