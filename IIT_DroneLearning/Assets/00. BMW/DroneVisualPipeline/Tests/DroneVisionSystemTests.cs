using System.Reflection;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using DroneCamera;
using DroneVisualPipeline;

namespace DroneVisualPipeline
{
    /// <summary>
    /// DroneVisionSystem 단위 테스트.
    /// Edit Mode 테스트 — Awake/Start는 Reflection으로 수동 호출 (DroneSensorSystem 패턴 준수).
    ///
    /// 검증 항목:
    ///   - RenderTexture 생성 및 해상도 일치 (요구사항 1.1, 1.2)
    ///   - 활성/비활성 토글 (요구사항 1.6, 6.3)
    ///   - 기존 Solo 카메라 targetTexture == null (요구사항 1.8, 8.2)
    ///   - OnValidate 해상도 변경 시 RT 재생성 (요구사항 6.3, 6.4)
    ///   - OnValidate 초기화 전 안전성 (요구사항 6.2)
    /// </summary>
    public class DroneVisionSystemTests
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

        /// <summary>
        /// DroneCameraSystem(Awake) + DroneVisionSystem(Start) 순서 보장 초기화 헬퍼.
        /// </summary>
        private (DroneCameraSystem cam, DroneVisionSystem vision) SetupFullSystem(
            int width = 84, int height = 84, bool grayscale = false)
        {
            var cam    = _droneGO.AddComponent<DroneCameraSystem>();
            var vision = _droneGO.AddComponent<DroneVisionSystem>();
            vision.textureWidth  = width;
            vision.textureHeight = height;
            vision.grayscale     = grayscale;

            // Unity 생명주기 수동 시뮬레이션
            InvokeAwake(cam);    // DroneCameraSystem.Awake() → DroneCamera_Solo 생성
            InvokeStart(vision); // DroneVisionSystem.Start() → Observation 카메라 추가

            return (cam, vision);
        }

        // ────────────────────────────────────────────
        // Task 9.1 — DroneVisionSystem 단위 테스트
        // ────────────────────────────────────────────

        /// <summary>
        /// 요구사항 1.1: Start() 실행 후 RenderTexture가 생성되어야 한다.
        /// </summary>
        [Test]
        public void Start_CreatesRenderTexture()
        {
            var (_, vision) = SetupFullSystem();

            Assert.IsNotNull(vision.RenderTexture,
                "Start() 후 RenderTexture가 null이어선 안 됩니다.");
            Assert.IsTrue(vision.RenderTexture.IsCreated(),
                "RenderTexture는 생성(IsCreated)된 상태여야 합니다.");
        }

        /// <summary>
        /// 요구사항 1.2: 생성된 RenderTexture 해상도가 설정값과 일치해야 한다.
        /// </summary>
        [Test]
        public void Start_RenderTextureResolutionMatchesSettings()
        {
            var (_, vision) = SetupFullSystem(width: 64, height: 48);

            Assert.AreEqual(64, vision.RenderTexture.width,
                "RenderTexture.width가 textureWidth와 일치해야 합니다.");
            Assert.AreEqual(48, vision.RenderTexture.height,
                "RenderTexture.height가 textureHeight와 일치해야 합니다.");
        }

        /// <summary>
        /// 요구사항 1.1: Start() 실행 후 Observation 카메라가 생성되어야 한다.
        /// </summary>
        [Test]
        public void Start_CreatesObservationCamera()
        {
            var (_, vision) = SetupFullSystem();

            Assert.IsNotNull(vision.ObservationCamera,
                "Start() 후 ObservationCamera가 null이어선 안 됩니다.");
        }

        /// <summary>
        /// 요구사항 1.8, 8.2: DroneVisionSystem 부착 후 Solo 카메라 targetTexture는 null이어야 한다.
        /// </summary>
        [Test]
        public void Start_SoloCameraTargetTextureRemainsNull()
        {
            var (cam, vision) = SetupFullSystem();

            Assert.IsNull(cam.Camera.targetTexture,
                "DroneCameraSystem Solo 카메라의 targetTexture는 null을 유지해야 합니다 (Display 출력 보존).");
        }

        /// <summary>
        /// 요구사항 1.6, 6.3: enableVisualObservation=false 시 Observation 카메라가 비활성화되어야 한다.
        /// </summary>
        [Test]
        public void EnableVisualObservation_False_DisablesCamera()
        {
            var (_, vision) = SetupFullSystem();
            vision.enableVisualObservation = false;
            InvokeOnValidate(vision);

            Assert.IsFalse(vision.ObservationCamera.enabled,
                "enableVisualObservation=false 시 ObservationCamera.enabled는 false여야 합니다.");
        }

        /// <summary>
        /// 요구사항 6.3: enableVisualObservation=true 시 Observation 카메라가 활성화되어야 한다.
        /// </summary>
        [Test]
        public void EnableVisualObservation_True_EnablesCamera()
        {
            var (_, vision) = SetupFullSystem();
            vision.enableVisualObservation = false;
            InvokeOnValidate(vision);

            vision.enableVisualObservation = true;
            InvokeOnValidate(vision);

            Assert.IsTrue(vision.ObservationCamera.enabled,
                "enableVisualObservation=true 시 ObservationCamera.enabled는 true여야 합니다.");
        }

        /// <summary>
        /// 요구사항 6.3, 6.4: OnValidate에서 해상도 변경 시 RenderTexture가 재생성되어야 한다.
        /// </summary>
        [Test]
        public void OnValidate_ResolutionChange_RecreatesRenderTexture()
        {
            var (_, vision) = SetupFullSystem(width: 84, height: 84);
            var originalRT = vision.RenderTexture;

            vision.textureWidth  = 128;
            vision.textureHeight = 128;
            InvokeOnValidate(vision);

            Assert.AreNotSame(originalRT, vision.RenderTexture,
                "해상도 변경 후 RenderTexture 인스턴스가 새로 생성되어야 합니다.");
            Assert.AreEqual(128, vision.RenderTexture.width,
                "재생성 후 RenderTexture.width는 새 설정값과 일치해야 합니다.");
            Assert.AreEqual(128, vision.RenderTexture.height,
                "재생성 후 RenderTexture.height는 새 설정값과 일치해야 합니다.");
        }

        /// <summary>
        /// 요구사항 6.2: _isInitialized=false 상태에서 OnValidate 호출 시 예외가 없어야 한다.
        /// </summary>
        [Test]
        public void OnValidate_BeforeInit_DoesNotThrow()
        {
            var vision = _droneGO.AddComponent<DroneCameraSystem>() != null
                ? _droneGO.AddComponent<DroneVisionSystem>()
                : null;

            if (vision == null)
            {
                Assert.Inconclusive("DroneVisionSystem 추가 실패");
                return;
            }

            // Start()를 호출하지 않은 상태 (IsInitialized == false)
            Assert.DoesNotThrow(() => InvokeOnValidate(vision),
                "_isInitialized=false 상태에서 OnValidate 호출 시 예외가 없어야 합니다.");
        }

        /// <summary>
        /// 요구사항 1.1: Observation 카메라가 Solo 카메라와 동일한 Transform을 공유해야 한다.
        /// </summary>
        [Test]
        public void Start_ObservationCameraSharesSoloTransform()
        {
            var (cam, vision) = SetupFullSystem();

            Assert.AreEqual(cam.Camera.gameObject, vision.ObservationCamera.gameObject,
                "Observation 카메라는 DroneCamera_Solo GameObject에 추가되어야 합니다.");
        }

        /// <summary>
        /// 요구사항 1.1: Observation 카메라의 targetTexture가 RenderTexture와 일치해야 한다.
        /// </summary>
        [Test]
        public void Start_ObservationCameraTargetTextureIsRenderTexture()
        {
            var (_, vision) = SetupFullSystem();

            Assert.AreEqual(vision.RenderTexture, vision.ObservationCamera.targetTexture,
                "ObservationCamera.targetTexture는 DroneVisionSystem.RenderTexture와 동일해야 합니다.");
        }
    }
}
