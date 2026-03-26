using System.IO;
using System.Reflection;
using NUnit.Framework;
using UnityEngine;
using DroneCamera;
using DroneVisualPipeline;

namespace DroneVisualPipeline
{
    /// <summary>
    /// DroneSnapshotSystem 단위 테스트.
    /// Edit Mode 테스트 — Start는 Reflection으로 수동 호출.
    ///
    /// 검증 항목:
    ///   - 에피소드 카운터 초기화 및 증가 (요구사항 4.3)
    ///   - 에피소드 폴더 경로 생성 (요구사항 3.3)
    ///   - 메타데이터 직렬화/역직렬화 (요구사항 4.1)
    ///   - OnValidate 클램핑 동작 (요구사항 6.7, 6.8)
    ///   - enableCapture 토글 (요구사항 3.1)
    /// </summary>
    public class DroneSnapshotSystemTests
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
        }

        // ────────────────────────────────────────────
        // 헬퍼
        // ────────────────────────────────────────────

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

        private DroneSnapshotSystem SetupSnapshot(string basePath = "CapturedData_Test")
        {
            var snap = _droneGO.AddComponent<DroneSnapshotSystem>();
            snap.basePath       = basePath;
            snap.filePrefix     = "TestDrone";
            snap.enableCapture  = false;
            snap.saveMetadata   = true;
            snap.saveEpisodeSummary = false; // 테스트 중 파일 저장 방지
            InvokeStart(snap);
            return snap;
        }

        // ────────────────────────────────────────────
        // Task 9.2 — DroneSnapshotSystem 단위 테스트
        // ────────────────────────────────────────────

        /// <summary>
        /// 요구사항 4.3: Start() 후 에피소드 카운터는 1이어야 한다.
        /// </summary>
        [Test]
        public void Start_EpisodeCounterStartsAtOne()
        {
            var snap = SetupSnapshot();

            Assert.AreEqual(1, snap.EpisodeCount,
                "Start() 후 에피소드 카운터는 1이어야 합니다.");
        }

        /// <summary>
        /// 요구사항 4.3: Start() 후 스텝 카운터는 0이어야 한다.
        /// </summary>
        [Test]
        public void Start_StepCounterStartsAtZero()
        {
            var snap = SetupSnapshot();

            Assert.AreEqual(0, snap.StepCount,
                "Start() 후 스텝 카운터는 0이어야 합니다.");
        }

        /// <summary>
        /// 요구사항 4.3: OnEpisodeBegin() 호출 시 에피소드 카운터가 증가해야 한다.
        /// </summary>
        [Test]
        public void OnEpisodeBegin_IncrementsEpisodeCounter()
        {
            var snap = SetupSnapshot();
            int countBefore = snap.EpisodeCount;

            snap.OnEpisodeBegin();

            Assert.AreEqual(countBefore + 1, snap.EpisodeCount,
                "OnEpisodeBegin() 호출 시 에피소드 카운터가 1 증가해야 합니다.");
        }

        /// <summary>
        /// 요구사항 4.3: OnEpisodeBegin() 호출 시 스텝 카운터가 0으로 초기화되어야 한다.
        /// </summary>
        [Test]
        public void OnEpisodeBegin_ResetsStepCounter()
        {
            var snap = SetupSnapshot();

            // 스텝을 임의로 증가 (리플렉션)
            var stepField = typeof(DroneSnapshotSystem).GetField(
                "_stepCount", BindingFlags.NonPublic | BindingFlags.Instance);
            stepField?.SetValue(snap, 50);

            snap.OnEpisodeBegin();

            Assert.AreEqual(0, snap.StepCount,
                "OnEpisodeBegin() 호출 시 스텝 카운터는 0으로 초기화되어야 합니다.");
        }

        /// <summary>
        /// 요구사항 3.3: Start() 후 에피소드 경로가 null이 아니어야 한다.
        /// </summary>
        [Test]
        public void Start_EpisodePathIsNotNull()
        {
            var snap = SetupSnapshot();

            Assert.IsNotNull(snap.CurrentEpisodePath,
                "Start() 후 CurrentEpisodePath가 null이어선 안 됩니다.");
            Assert.IsNotEmpty(snap.CurrentEpisodePath,
                "Start() 후 CurrentEpisodePath가 빈 문자열이어선 안 됩니다.");
        }

        /// <summary>
        /// 요구사항 3.3: 에피소드 경로가 filePrefix와 에피소드 번호를 포함해야 한다.
        /// </summary>
        [Test]
        public void Start_EpisodePathContainsPrefixAndEpisodeNumber()
        {
            var snap = SetupSnapshot();

            StringAssert.Contains("TestDrone",
                snap.CurrentEpisodePath,
                "에피소드 경로에 filePrefix가 포함되어야 합니다.");
            StringAssert.Contains("episode_",
                snap.CurrentEpisodePath,
                "에피소드 경로에 'episode_' 패턴이 포함되어야 합니다.");
        }

        /// <summary>
        /// 요구사항 4.1: CaptureMetadata가 JsonUtility로 직렬화/역직렬화 가능해야 한다.
        /// </summary>
        [Test]
        public void CaptureMetadata_SerializesAndDeserializesCorrectly()
        {
            var meta = new CaptureMetadata
            {
                timestamp        = "2026-03-21T12:00:00.000Z",
                episode          = 42,
                step             = 150,
                droneRole        = "Evader",
                dronePosition    = new SerializedVector3(new Vector3(1f, 2f, 3f)),
                droneRotation    = new SerializedVector3(new Vector3(0f, 45f, 0f)),
                droneVelocity    = new SerializedVector3(new Vector3(1.2f, 0f, -0.8f)),
                reward           = -0.001f,
                cumulativeReward = -0.15f,
                sensorDistances  = new float[] { 0f, 0.5f, 1.0f },
            };

            string json = JsonUtility.ToJson(meta);
            var deserialized = JsonUtility.FromJson<CaptureMetadata>(json);

            Assert.AreEqual(meta.episode,  deserialized.episode,  "episode 필드가 일치해야 합니다.");
            Assert.AreEqual(meta.step,     deserialized.step,     "step 필드가 일치해야 합니다.");
            Assert.AreEqual(meta.droneRole, deserialized.droneRole, "droneRole 필드가 일치해야 합니다.");
            Assert.AreEqual(meta.reward,   deserialized.reward,   0.0001f, "reward 필드가 일치해야 합니다.");
            Assert.AreEqual(3, deserialized.sensorDistances.Length, "sensorDistances 배열 길이가 일치해야 합니다.");
        }

        /// <summary>
        /// 요구사항 6.7, 6.8: OnValidate에서 captureInterval이 [1,100] 범위로 클램핑되어야 한다.
        /// </summary>
        [Test]
        public void OnValidate_CaptureIntervalClamped()
        {
            var snap = _droneGO.AddComponent<DroneSnapshotSystem>();
            snap.captureInterval = -10;
            InvokeOnValidate(snap);
            Assert.AreEqual(1, snap.captureInterval,
                "captureInterval은 최소값 1로 클램핑되어야 합니다.");

            snap.captureInterval = 999;
            InvokeOnValidate(snap);
            Assert.AreEqual(100, snap.captureInterval,
                "captureInterval은 최대값 100으로 클램핑되어야 합니다.");
        }

        /// <summary>
        /// 요구사항 6.7, 6.8: OnValidate에서 maxAsyncRequests가 [1,10] 범위로 클램핑되어야 한다.
        /// </summary>
        [Test]
        public void OnValidate_MaxAsyncRequestsClamped()
        {
            var snap = _droneGO.AddComponent<DroneSnapshotSystem>();
            snap.maxAsyncRequests = 0;
            InvokeOnValidate(snap);
            Assert.AreEqual(1, snap.maxAsyncRequests,
                "maxAsyncRequests는 최소값 1로 클램핑되어야 합니다.");

            snap.maxAsyncRequests = 100;
            InvokeOnValidate(snap);
            Assert.AreEqual(10, snap.maxAsyncRequests,
                "maxAsyncRequests는 최대값 10으로 클램핑되어야 합니다.");
        }

        /// <summary>
        /// 요구사항 6.8: OnValidate에서 basePath가 빈 문자열이면 "CapturedData"로 복구되어야 한다.
        /// </summary>
        [Test]
        public void OnValidate_EmptyBasePathDefaultsToCapturedData()
        {
            var snap = _droneGO.AddComponent<DroneSnapshotSystem>();
            snap.basePath = "";
            InvokeOnValidate(snap);
            Assert.AreEqual("CapturedData", snap.basePath,
                "basePath가 빈 문자열이면 기본 경로로 복구되어야 합니다.");

            snap.basePath = "   ";
            InvokeOnValidate(snap);
            Assert.AreEqual("CapturedData", snap.basePath,
                "basePath가 공백 문자열이면 기본 경로로 복구되어야 합니다.");
        }

        /// <summary>
        /// 요구사항 4.1: EpisodeSummary가 JsonUtility로 직렬화 가능해야 한다.
        /// </summary>
        [Test]
        public void EpisodeSummary_SerializesCorrectly()
        {
            var summary = new EpisodeSummary
            {
                episode          = 1,
                droneRole        = "Evader",
                totalSteps       = 300,
                cumulativeReward = -0.45f,
                terminationReason = "captured",
                capturedImageCount = 60,
                startTime        = "2026-03-21T12:00:00.000Z",
                endTime          = "2026-03-21T12:00:15.000Z",
            };

            string json = JsonUtility.ToJson(summary, true);

            Assert.IsNotEmpty(json, "EpisodeSummary JSON이 비어있어선 안 됩니다.");
            StringAssert.Contains("\"episode\"", json, "JSON에 episode 필드가 포함되어야 합니다.");
            StringAssert.Contains("\"droneRole\"", json, "JSON에 droneRole 필드가 포함되어야 합니다.");
        }
    }
}
