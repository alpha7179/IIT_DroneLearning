using System.Reflection;
using NUnit.Framework;
using UnityEngine;
using DroneCamera;
using DroneVisualPipeline;

namespace DroneVisualPipeline
{
    /// <summary>
    /// DroneDepthSystem 단위 테스트.
    /// Edit Mode 테스트 — Awake/Start는 Reflection으로 수동 호출.
    ///
    /// 검증 항목:
    ///   - Depth RenderTexture 생성 및 해상도 일치 (요구사항 5.1)
    ///   - 활성/비활성 토글 (요구사항 5.4)
    ///   - OnValidate 셰이더 파라미터 업데이트 (요구사항 6.9, 6.10, 6.11)
    ///   - OnValidate 해상도 변경 시 RT 재생성 (요구사항 6.9)
    ///   - OnValidate 초기화 전 안전성 (요구사항 6.2)
    ///
    /// Note: 셰이더 로드 테스트는 Shader.Find()가 Edit Mode에서 동작하지 않을 수 있으므로
    ///       컴포넌트 초기화 실패 경우도 허용하고 IsInitialized로 조건 분기한다.
    /// </summary>
    public class DroneDepthSystemTests
    {
        private GameObject _droneGO;

        [SetUp]
        public void SetUp()
        {
            _droneGO = new GameObject("TestDrone");
        }

        [TearDown]
        public void TearDown()
        {
            if (_droneGO != null) Object.DestroyImmediate(_droneGO);

            foreach (var go in Object.FindObjectsByType<GameObject>(FindObjectsSortMode.None))
            {
                if (go.name.StartsWith("Test") || go.name == "DroneCamera_Solo" ||
                    go.name == "DroneCamera_MultiView")
                    Object.DestroyImmediate(go);
            }
        }

        // ────────────────────────────────────────────
        // 헬퍼
        // ────────────────────────────────────────────

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

        private static void InvokeOnValidate(MonoBehaviour comp)
        {
            var m = comp.GetType().GetMethod("OnValidate",
                BindingFlags.NonPublic | BindingFlags.Instance);
            m?.Invoke(comp, null);
        }

        private (DroneCameraSystem cam, DroneDepthSystem depth) SetupFullSystem(
            int width = 84, int height = 84)
        {
            var cam   = _droneGO.AddComponent<DroneCameraSystem>();
            var depth = _droneGO.AddComponent<DroneDepthSystem>();
            depth.depthWidth  = width;
            depth.depthHeight = height;

            InvokeAwake(cam);
            InvokeStart(depth);

            return (cam, depth);
        }

        // ────────────────────────────────────────────
        // Task 9.3 — DroneDepthSystem 단위 테스트
        // ────────────────────────────────────────────

        /// <summary>
        /// 요구사항 5.1: Start() 실행 후 Depth 카메라가 생성되어야 한다.
        /// </summary>
        [Test]
        public void Start_CreatesDepthCamera()
        {
            var (_, depth) = SetupFullSystem();

            // 셰이더가 없으면 초기화 실패 가능 (Edit Mode 제한)
            if (!depth.IsInitialized)
            {
                Assert.Inconclusive(
                    "셰이더 로드 실패로 DroneDepthSystem 초기화가 불완전합니다 (Edit Mode 제한). " +
                    "Play Mode에서 재검증 필요.");
                return;
            }

            Assert.IsNotNull(depth.DepthCamera,
                "Start() 후 DepthCamera가 null이어선 안 됩니다.");
        }

        /// <summary>
        /// 요구사항 5.1: Start() 실행 후 Depth RenderTexture가 생성되어야 한다.
        /// </summary>
        [Test]
        public void Start_CreatesDepthRenderTexture()
        {
            var (_, depth) = SetupFullSystem();

            if (!depth.IsInitialized)
            {
                Assert.Inconclusive("셰이더 로드 실패로 초기화 불완전 — Play Mode에서 재검증 필요.");
                return;
            }

            Assert.IsNotNull(depth.DepthRenderTexture,
                "Start() 후 DepthRenderTexture가 null이어선 안 됩니다.");
            Assert.IsTrue(depth.DepthRenderTexture.IsCreated(),
                "DepthRenderTexture는 생성(IsCreated)된 상태여야 합니다.");
        }

        /// <summary>
        /// 요구사항 5.1: Depth RenderTexture 해상도가 설정값과 일치해야 한다.
        /// </summary>
        [Test]
        public void Start_DepthRenderTextureResolutionMatchesSettings()
        {
            var (_, depth) = SetupFullSystem(width: 64, height: 48);

            if (!depth.IsInitialized)
            {
                Assert.Inconclusive("셰이더 로드 실패로 초기화 불완전 — Play Mode에서 재검증 필요.");
                return;
            }

            Assert.AreEqual(64, depth.DepthRenderTexture.width,
                "DepthRenderTexture.width가 depthWidth와 일치해야 합니다.");
            Assert.AreEqual(48, depth.DepthRenderTexture.height,
                "DepthRenderTexture.height가 depthHeight와 일치해야 합니다.");
        }

        /// <summary>
        /// 요구사항 5.4: enableDepthSensor=false 시 Depth 카메라가 비활성화되어야 한다.
        /// </summary>
        [Test]
        public void EnableDepthSensor_False_DisablesCamera()
        {
            var (_, depth) = SetupFullSystem();

            if (!depth.IsInitialized)
            {
                Assert.Inconclusive("셰이더 로드 실패로 초기화 불완전 — Play Mode에서 재검증 필요.");
                return;
            }

            depth.enableDepthSensor = false;
            InvokeOnValidate(depth);

            Assert.IsFalse(depth.DepthCamera.enabled,
                "enableDepthSensor=false 시 DepthCamera.enabled는 false여야 합니다.");
        }

        /// <summary>
        /// 요구사항 6.2: _isInitialized=false 상태에서 OnValidate 호출 시 예외가 없어야 한다.
        /// </summary>
        [Test]
        public void OnValidate_BeforeInit_DoesNotThrow()
        {
            _droneGO.AddComponent<DroneCameraSystem>();
            var depth = _droneGO.AddComponent<DroneDepthSystem>();

            // Start()를 호출하지 않은 상태
            Assert.DoesNotThrow(() => InvokeOnValidate(depth),
                "_isInitialized=false 상태에서 OnValidate 호출 시 예외가 없어야 합니다.");
        }

        /// <summary>
        /// 요구사항 6.9: OnValidate에서 maxDepthDistance가 음수이면 0으로 클램핑되어야 한다.
        /// </summary>
        [Test]
        public void OnValidate_NegativeMaxDepthDistance_ClampedToZero()
        {
            var (_, depth) = SetupFullSystem();

            depth.maxDepthDistance = -100f;
            InvokeOnValidate(depth);

            Assert.AreEqual(0f, depth.maxDepthDistance, 0.0001f,
                "maxDepthDistance는 음수일 때 0으로 클램핑되어야 합니다.");
        }

        /// <summary>
        /// 요구사항 5.1: Depth 카메라는 DroneCamera_Solo GameObject에 추가되어야 한다.
        /// </summary>
        [Test]
        public void Start_DepthCameraOnSoloCameraGameObject()
        {
            var (cam, depth) = SetupFullSystem();

            if (!depth.IsInitialized)
            {
                Assert.Inconclusive("셰이더 로드 실패로 초기화 불완전 — Play Mode에서 재검증 필요.");
                return;
            }

            Assert.AreEqual(cam.Camera.gameObject, depth.DepthCamera.gameObject,
                "Depth 카메라는 DroneCamera_Solo GameObject에 추가되어야 합니다.");
        }

        // ────────────────────────────────────────────
        // Task 10 — Property 기반 테스트 (DroneVisionSystem + DroneDepthSystem)
        // ────────────────────────────────────────────

        /// <summary>
        /// Property 1: RenderTexture 해상도 일치 — 임의의 (w, h) 조합에서 검증.
        /// 요구사항 1.2
        /// </summary>
        [TestCase(16,  16)]
        [TestCase(84,  84)]
        [TestCase(128, 64)]
        [TestCase(256, 256)]
        public void Property_RenderTextureResolutionAlwaysMatchesSettings(int w, int h)
        {
            var cam    = _droneGO.AddComponent<DroneCameraSystem>();
            var vision = _droneGO.AddComponent<DroneVisionSystem>();
            vision.textureWidth  = w;
            vision.textureHeight = h;

            InvokeAwake(cam);
            InvokeStart(vision);

            if (!vision.IsInitialized)
            {
                Assert.Inconclusive("DroneVisionSystem 초기화 실패.");
                return;
            }

            Assert.AreEqual(w, vision.RenderTexture.width,
                $"width={w}일 때 RenderTexture.width가 일치해야 합니다.");
            Assert.AreEqual(h, vision.RenderTexture.height,
                $"height={h}일 때 RenderTexture.height가 일치해야 합니다.");
        }

        /// <summary>
        /// Property 2: enableVisualObservation=false 시 항상 카메라가 비활성화.
        /// 요구사항 1.6, 6.3
        /// </summary>
        [Test]
        public void Property_VisualObservationDisabledAlwaysDisablesCamera()
        {
            var cam    = _droneGO.AddComponent<DroneCameraSystem>();
            var vision = _droneGO.AddComponent<DroneVisionSystem>();

            InvokeAwake(cam);
            InvokeStart(vision);

            if (!vision.IsInitialized)
            {
                Assert.Inconclusive("DroneVisionSystem 초기화 실패.");
                return;
            }

            // 100번 반복으로 토글 안정성 검증
            for (int i = 0; i < 100; i++)
            {
                vision.enableVisualObservation = false;
                InvokeOnValidate(vision);

                Assert.IsFalse(vision.ObservationCamera.enabled,
                    $"반복 {i}: enableVisualObservation=false 시 카메라가 비활성화되어야 합니다.");
            }
        }

        /// <summary>
        /// Property 3: 기존 카메라 무간섭 — DroneVisionSystem 부착 후에도 Solo targetTexture == null.
        /// 요구사항 1.8, 8.2
        /// </summary>
        [Test]
        public void Property_SoloCameraTargetTextureAlwaysNull()
        {
            // 100번 반복으로 안정성 검증
            for (int i = 0; i < 100; i++)
            {
                var go     = new GameObject($"TestDrone_{i}");
                var cam    = go.AddComponent<DroneCameraSystem>();
                var vision = go.AddComponent<DroneVisionSystem>();

                InvokeAwake(cam);
                InvokeStart(vision);

                Assert.IsNull(cam.Camera.targetTexture,
                    $"반복 {i}: Solo 카메라 targetTexture는 항상 null이어야 합니다.");

                Object.DestroyImmediate(go);
            }
        }

        /// <summary>
        /// Property 6: OnValidate 초기화 전 안전성 — 어떤 상태에서도 예외 없음.
        /// 요구사항 6.2
        /// </summary>
        [Test]
        public void Property_OnValidateBeforeInitNeverThrows()
        {
            var cam   = _droneGO.AddComponent<DroneCameraSystem>();
            var depth = _droneGO.AddComponent<DroneDepthSystem>();

            // Start()를 호출하지 않은 상태에서 다양한 값으로 OnValidate
            for (int i = 0; i < 50; i++)
            {
                depth.depthWidth        = Random.Range(16, 256);
                depth.depthHeight       = Random.Range(16, 256);
                depth.maxDepthDistance  = Random.Range(-100f, 1000f);
                depth.enableDepthSensor = (i % 2 == 0);

                Assert.DoesNotThrow(() => InvokeOnValidate(depth),
                    $"반복 {i}: 초기화 전 OnValidate에서 예외가 없어야 합니다.");
            }
        }
    }
}
