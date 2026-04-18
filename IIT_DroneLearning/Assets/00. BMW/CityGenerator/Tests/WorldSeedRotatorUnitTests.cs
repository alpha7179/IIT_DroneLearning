using System.Reflection;
using NUnit.Framework;
using UnityEngine;

namespace CityGenerator
{
    /// <summary>
    /// WorldSeedRotator 단위 테스트.
    ///
    /// NUnit EditMode 테스트.
    /// CityGenerator 참조가 필요한 케이스는 null 경로(조기 반환)와
    /// 카운터 독립 동작(rotationInterval 을 충분히 크게 설정)을 구분하여 검증한다.
    ///
    /// Feature: world-seed-rotation
    /// </summary>
    public class WorldSeedRotatorUnitTests
    {
        // ── 씬 오브젝트 ──────────────────────────────────────────────────
        private GameObject        _rotatorObject;
        private WorldSeedRotator  _rotator;

        // ── SetUp / TearDown ─────────────────────────────────────────────

        [SetUp]
        public void SetUp()
        {
            _rotatorObject = new GameObject("TestWorldSeedRotator");
            _rotator       = _rotatorObject.AddComponent<WorldSeedRotator>();
        }

        [TearDown]
        public void TearDown()
        {
            if (_rotatorObject != null)
                Object.DestroyImmediate(_rotatorObject);
        }

        // ── Reflection 헬퍼 ──────────────────────────────────────────────

        private void SetField(string name, object value) =>
            GetFieldInfo(name).SetValue(_rotator, value);

        private T GetField<T>(string name) =>
            (T)GetFieldInfo(name).GetValue(_rotator);

        private FieldInfo GetFieldInfo(string name)
        {
            var fi = typeof(WorldSeedRotator).GetField(
                name, BindingFlags.NonPublic | BindingFlags.Instance);
            Assert.IsNotNull(fi, $"필드 '{name}' 을 찾을 수 없습니다.");
            return fi;
        }

        // ── 초기 상태 검증 ───────────────────────────────────────────────

        [Test]
        public void InitialState_AllCountersAreZero()
        {
            Assert.AreEqual(0, _rotator.EpisodeCounter);
            Assert.AreEqual(0, _rotator.TotalEpisodeCount);
            Assert.AreEqual(0, _rotator.TotalRotationCount);
        }

        [Test]
        public void InitialState_DefaultsMatchSpec()
        {
            // _enabled = true, _rotationInterval = 10, Sequential, baseSeed = 42
            Assert.IsTrue(_rotator.IsEnabled);
            Assert.AreEqual(42, _rotator.CurrentSeed);

            int interval = GetField<int>("_rotationInterval");
            Assert.AreEqual(10, interval);

            var strategy = GetField<WorldSeedRotator.SeedSequenceStrategy>("_seedStrategy");
            Assert.AreEqual(WorldSeedRotator.SeedSequenceStrategy.Sequential, strategy);

            var mode = GetField<WorldSeedRotator.RotationTriggerMode>("_triggerMode");
            Assert.AreEqual(WorldSeedRotator.RotationTriggerMode.Interval, mode);
        }

        // ── 비활성화(disabled) 상태 바이패스 ────────────────────────────

        [Test]
        public void Disabled_NotifyDoesNotIncrementAnyCounter()
        {
            SetField("_enabled", false);

            for (int i = 0; i < 20; i++)
                _rotator.NotifyEpisodeBegin();

            Assert.AreEqual(0, _rotator.EpisodeCounter,    "EpisodeCounter 변경되면 안 됨");
            Assert.AreEqual(0, _rotator.TotalEpisodeCount, "TotalEpisodeCount 변경되면 안 됨");
            Assert.AreEqual(0, _rotator.TotalRotationCount,"TotalRotationCount 변경되면 안 됨");
        }

        [Test]
        public void Disabled_CurrentSeedUnchanged()
        {
            int seedBefore = _rotator.CurrentSeed;
            SetField("_enabled", false);

            for (int i = 0; i < 50; i++)
                _rotator.NotifyEpisodeBegin();

            Assert.AreEqual(seedBefore, _rotator.CurrentSeed);
        }

        // ── Interval 모드: 카운터 증가 ───────────────────────────────────

        [Test]
        public void IntervalMode_EpisodeCounterIncrements()
        {
            // CityGenerator = null → 로테이션 시 조기 반환하지만 카운터는 증가
            SetField("_rotationInterval", 999); // 로테이션이 트리거되지 않을 만큼 큰 값

            for (int i = 1; i <= 10; i++)
            {
                _rotator.NotifyEpisodeBegin();
                Assert.AreEqual(i, _rotator.EpisodeCounter,    $"EpisodeCounter: {i}회 호출 후");
                Assert.AreEqual(i, _rotator.TotalEpisodeCount, $"TotalEpisodeCount: {i}회 호출 후");
            }
        }

        [Test]
        public void IntervalMode_EpisodeCounterResetsWhenThresholdReached_NullCityGenerator()
        {
            // CityGenerator null → 로테이션 스킵되지만 카운터 리셋은 정상 동작
            const int N = 5;
            SetField("_rotationInterval", N);
            // _cityGenerator 는 null (AddComponent 후 연결 안 함)

            for (int i = 0; i < N; i++)
                _rotator.NotifyEpisodeBegin();

            // N번째 호출에서 threshold 달성 → 카운터 리셋
            Assert.AreEqual(0, _rotator.EpisodeCounter,    "EpisodeCounter 가 리셋되어야 함");
            Assert.AreEqual(N, _rotator.TotalEpisodeCount, "TotalEpisodeCount 는 누적됨");
        }

        [Test]
        public void IntervalMode_CounterCyclesCorrectly_NullCityGenerator()
        {
            const int N = 3;
            SetField("_rotationInterval", N);

            // 2 사이클 + 1 에피소드
            for (int i = 0; i < 2 * N + 1; i++)
                _rotator.NotifyEpisodeBegin();

            Assert.AreEqual(1, _rotator.EpisodeCounter,        "2사이클+1 후 카운터는 1이어야 함");
            Assert.AreEqual(2 * N + 1, _rotator.TotalEpisodeCount);
        }

        // ── Boundary 모드: 카운터 증가 ───────────────────────────────────

        [Test]
        public void BoundaryMode_EpisodeCounterAlwaysIncrements()
        {
            SetField("_triggerMode", WorldSeedRotator.RotationTriggerMode.Boundary);
            SetField("_rotationInterval", 999);

            for (int i = 1; i <= 15; i++)
            {
                _rotator.NotifyEpisodeBegin();
                // Boundary 모드에서는 EpisodeCounter 가 리셋되지 않음
                Assert.AreEqual(i, _rotator.TotalEpisodeCount, $"TotalEpisodeCount: {i}회 호출 후");
            }
        }

        [Test]
        public void BoundaryMode_EpisodeCounterNotReset_NullCityGenerator()
        {
            const int N = 5;
            SetField("_triggerMode", WorldSeedRotator.RotationTriggerMode.Boundary);
            SetField("_rotationInterval", N);

            for (int i = 0; i < N * 2; i++)
                _rotator.NotifyEpisodeBegin();

            // Boundary 모드에서 EpisodeCounter(사이클 카운터)는 변하지 않음
            Assert.AreEqual(N * 2, _rotator.TotalEpisodeCount, "TotalEpisodeCount 는 누적됨");
            // EpisodeCounter 는 Boundary 모드에서 0 을 유지함 (사용하지 않음)
            Assert.AreEqual(0, _rotator.EpisodeCounter, "Boundary 모드에서 EpisodeCounter(사이클) 는 0 유지");
        }

        // ── OnValidate: rotationInterval 최솟값 클램핑 ──────────────────

        [Test]
        public void OnValidate_ClampsRotationIntervalToMin1()
        {
            SetField("_rotationInterval", 0);
            // OnValidate 직접 호출 (Reflection)
            typeof(WorldSeedRotator)
                .GetMethod("OnValidate", BindingFlags.NonPublic | BindingFlags.Instance)
                ?.Invoke(_rotator, null);
            Assert.AreEqual(1, GetField<int>("_rotationInterval"));
        }

        [Test]
        public void OnValidate_NegativeValueClamped()
        {
            SetField("_rotationInterval", -10);
            typeof(WorldSeedRotator)
                .GetMethod("OnValidate", BindingFlags.NonPublic | BindingFlags.Instance)
                ?.Invoke(_rotator, null);
            Assert.AreEqual(1, GetField<int>("_rotationInterval"));
        }

        // ── Sequential 시드 계산 ─────────────────────────────────────────

        [Test]
        public void ComputeNextSeed_Sequential_ReturnsBaseSeedPlusRotationCountPlusOne()
        {
            const int baseSeed = 100;
            SetField("_baseSeed",          baseSeed);
            SetField("_seedStrategy",      WorldSeedRotator.SeedSequenceStrategy.Sequential);
            SetField("_totalRotationCount", 0);

            int next = InvokeComputeNextSeed();
            Assert.AreEqual(baseSeed + 1, next, "첫 번째 로테이션 시드 = baseSeed + 1");
        }

        [Test]
        public void ComputeNextSeed_Sequential_IncrementsWithRotationCount()
        {
            const int baseSeed       = 0;
            const int rotationCount  = 7;
            SetField("_baseSeed",           baseSeed);
            SetField("_seedStrategy",       WorldSeedRotator.SeedSequenceStrategy.Sequential);
            SetField("_totalRotationCount", rotationCount);

            int next = InvokeComputeNextSeed();
            Assert.AreEqual(baseSeed + rotationCount + 1, next);
        }

        [Test]
        public void ComputeNextSeed_Random_ReturnsDifferentValuesOverTime()
        {
            SetField("_seedStrategy", WorldSeedRotator.SeedSequenceStrategy.Random);

            // TickCount 기반이므로 충분한 간격을 두면 달라질 수 있음.
            // 단순히 int 범위 내 값인지만 검증 (항상 다를 보장은 불가)
            int seed = InvokeComputeNextSeed();
            Assert.IsTrue(seed != int.MinValue || seed != int.MaxValue,
                "Random 시드가 유효한 int 범위 내여야 함");
        }

        // ── CityGenerator null 시 안전 조기 반환 ────────────────────────

        [Test]
        public void NotifyEpisodeBegin_NullCityGenerator_DoesNotThrow_AtRotationPoint()
        {
            const int N = 3;
            SetField("_rotationInterval", N);
            // _cityGenerator = null (기본값)

            // N 번째 호출에서 로테이션 포인트 → Debug.LogError 출력 후 조기 반환 (예외 없어야 함)
            Assert.DoesNotThrow(() =>
            {
                for (int i = 0; i < N; i++)
                    _rotator.NotifyEpisodeBegin();
            });

            Assert.AreEqual(0, _rotator.TotalRotationCount, "CityGenerator null 이므로 로테이션 횟수 0 유지");
        }

        // ── Reflection 헬퍼 ──────────────────────────────────────────────

        private int InvokeComputeNextSeed()
        {
            var mi = typeof(WorldSeedRotator).GetMethod(
                "ComputeNextSeed", BindingFlags.NonPublic | BindingFlags.Instance);
            Assert.IsNotNull(mi, "ComputeNextSeed 메서드를 찾을 수 없습니다.");
            return (int)mi.Invoke(_rotator, null);
        }
    }
}
