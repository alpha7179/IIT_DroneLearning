using System.Reflection;
using NUnit.Framework;
using UnityEngine;

namespace CityGenerator
{
    /// <summary>
    /// WorldSeedRotator 속성 기반 테스트 (Property-Based Tests).
    ///
    /// FsCheck 대신 NUnit + 반복 루프(100회) + System.Random 기반으로 구현한다.
    /// CityGenerator 의존 없이 검증 가능한 속성(카운터 생명주기, 시드 계산, 비활성화 바이패스)에
    /// 집중하며, 실제 도시 재생성이 필요한 속성은 통합 테스트(PlayMode)에서 검증한다.
    ///
    /// Feature: world-seed-rotation
    /// </summary>
    public class WorldSeedRotatorPropertyTests
    {
        private const int Iterations = 100;

        private System.Random    _rng;
        private GameObject       _rotatorObject;
        private WorldSeedRotator _rotator;

        // ── SetUp / TearDown ─────────────────────────────────────────────

        [SetUp]
        public void SetUp()
        {
            _rng           = new System.Random(42);
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
            GetFI(name).SetValue(_rotator, value);

        private T GetField<T>(string name) => (T)GetFI(name).GetValue(_rotator);

        private FieldInfo GetFI(string name)
        {
            var fi = typeof(WorldSeedRotator).GetField(
                name, BindingFlags.NonPublic | BindingFlags.Instance);
            Assert.IsNotNull(fi, $"필드 '{name}' 을 찾을 수 없습니다.");
            return fi;
        }

        private void ResetInternalCounters()
        {
            SetField("_episodeCounter",     0);
            SetField("_totalEpisodeCount",  0);
            SetField("_totalRotationCount", 0);
            SetField("_currentSeed",        GetField<int>("_baseSeed"));
        }

        // ── Property 1: 에피소드 카운터 생명주기 (Interval 모드) ──────────
        //
        // For any N (1~50) and M (1~200) in Interval mode (CityGenerator=null):
        //   TotalEpisodeCount == M
        //   EpisodeCounter    == M % N
        // TotalRotationCount 는 CityGenerator=null 이므로 항상 0.
        //
        // Validates: Requirements 1.1, 1.2, 1.3, 2.1

        [Test]
        public void Property1_IntervalMode_CounterLifecycle()
        {
            for (int iter = 0; iter < Iterations; iter++)
            {
                int n = _rng.Next(1, 51);   // rotationInterval
                int m = _rng.Next(1, 201);  // notification count

                // 상태 초기화
                SetField("_rotationInterval", n);
                SetField("_triggerMode",
                    WorldSeedRotator.RotationTriggerMode.Interval);
                SetField("_enabled", true);
                ResetInternalCounters();

                for (int i = 0; i < m; i++)
                    _rotator.NotifyEpisodeBegin();

                Assert.AreEqual(m,     _rotator.TotalEpisodeCount,
                    $"[iter={iter}] TotalEpisodeCount: N={n}, M={m}");
                Assert.AreEqual(m % n, _rotator.EpisodeCounter,
                    $"[iter={iter}] EpisodeCounter: N={n}, M={m}");
            }
        }

        // ── Property 2: Boundary 모드 카운터 생명주기 ────────────────────
        //
        // For any N (1~50) and M (1~200) in Boundary mode (CityGenerator=null):
        //   TotalEpisodeCount == M
        //   EpisodeCounter    == 0  (Boundary 모드에서 사이클 카운터는 사용하지 않음)
        //
        // Validates: Boundary 모드 카운터 독립성

        [Test]
        public void Property2_BoundaryMode_CounterLifecycle()
        {
            for (int iter = 0; iter < Iterations; iter++)
            {
                int n = _rng.Next(1, 51);
                int m = _rng.Next(1, 201);

                SetField("_rotationInterval", n);
                SetField("_triggerMode",
                    WorldSeedRotator.RotationTriggerMode.Boundary);
                SetField("_enabled", true);
                ResetInternalCounters();

                for (int i = 0; i < m; i++)
                    _rotator.NotifyEpisodeBegin();

                Assert.AreEqual(m, _rotator.TotalEpisodeCount,
                    $"[iter={iter}] TotalEpisodeCount: N={n}, M={m}");
                Assert.AreEqual(0, _rotator.EpisodeCounter,
                    $"[iter={iter}] Boundary 모드에서 EpisodeCounter 는 0 유지: N={n}, M={m}");
            }
        }

        // ── Property 3: 로테이션 간격 최솟값 클램핑 ──────────────────────
        //
        // For any V (-100~100): max(1, V) 가 실제 적용됨.
        //
        // Validates: Requirements 2.4

        [Test]
        public void Property3_RotationInterval_ClampedToMin1()
        {
            var onValidate = typeof(WorldSeedRotator)
                .GetMethod("OnValidate", BindingFlags.NonPublic | BindingFlags.Instance);
            Assert.IsNotNull(onValidate, "OnValidate 메서드를 찾을 수 없습니다.");

            for (int iter = 0; iter < Iterations; iter++)
            {
                int v = _rng.Next(-100, 101);
                SetField("_rotationInterval", v);
                onValidate.Invoke(_rotator, null);

                int applied = GetField<int>("_rotationInterval");
                Assert.AreEqual(Mathf.Max(1, v), applied,
                    $"[iter={iter}] V={v} 에 대한 클램핑 결과 불일치");
            }
        }

        // ── Property 4: Sequential 시드 계산 ─────────────────────────────
        //
        // For any baseSeed (-1000~1000) and rotationCount (0~50):
        //   ComputeNextSeed(Sequential) == baseSeed + rotationCount + 1
        //
        // Validates: Requirements 3.1, 5.2

        [Test]
        public void Property4_Sequential_SeedComputedCorrectly()
        {
            var computeNextSeed = typeof(WorldSeedRotator)
                .GetMethod("ComputeNextSeed", BindingFlags.NonPublic | BindingFlags.Instance);
            Assert.IsNotNull(computeNextSeed, "ComputeNextSeed 를 찾을 수 없습니다.");

            SetField("_seedStrategy", WorldSeedRotator.SeedSequenceStrategy.Sequential);

            for (int iter = 0; iter < Iterations; iter++)
            {
                int baseSeed      = _rng.Next(-1000, 1001);
                int rotationCount = _rng.Next(0, 51);

                SetField("_baseSeed",           baseSeed);
                SetField("_totalRotationCount", rotationCount);

                int result   = (int)computeNextSeed.Invoke(_rotator, null);
                int expected = baseSeed + rotationCount + 1;

                Assert.AreEqual(expected, result,
                    $"[iter={iter}] baseSeed={baseSeed}, rotationCount={rotationCount}");
            }
        }

        // ── Property 5: 비활성화 상태 바이패스 ───────────────────────────
        //
        // For any M (1~200), disabled WorldSeedRotator:
        //   EpisodeCounter == 0, TotalEpisodeCount == 0, TotalRotationCount == 0, CurrentSeed 불변
        //
        // Validates: Requirements 6.2

        [Test]
        public void Property5_Disabled_AllCountersStayZero()
        {
            SetField("_enabled", false);
            SetField("_rotationInterval", 1); // 활성화됐다면 매 에피소드마다 로테이션

            for (int iter = 0; iter < Iterations; iter++)
            {
                int m = _rng.Next(1, 201);
                ResetInternalCounters();

                int seedBefore = _rotator.CurrentSeed;

                for (int i = 0; i < m; i++)
                    _rotator.NotifyEpisodeBegin();

                Assert.AreEqual(0, _rotator.EpisodeCounter,
                    $"[iter={iter}] M={m}: EpisodeCounter 변경되면 안 됨");
                Assert.AreEqual(0, _rotator.TotalEpisodeCount,
                    $"[iter={iter}] M={m}: TotalEpisodeCount 변경되면 안 됨");
                Assert.AreEqual(0, _rotator.TotalRotationCount,
                    $"[iter={iter}] M={m}: TotalRotationCount 변경되면 안 됨");
                Assert.AreEqual(seedBefore, _rotator.CurrentSeed,
                    $"[iter={iter}] M={m}: CurrentSeed 변경되면 안 됨");
            }
        }

        // ── Property 6: Interval 모드 트리거 횟수 일관성 ─────────────────
        //
        // For any N (1~20) and M (1~100) in Interval mode (CityGenerator=null):
        //   실제 로테이션 시도 횟수(카운터 리셋 횟수) == M / N (정수 나눗셈)
        //   단, CityGenerator=null 이므로 TotalRotationCount 는 0으로 유지
        //
        // Validates: Interval 모드 트리거 타이밍

        [Test]
        public void Property6_IntervalMode_TriggerCount_MatchesFloorDivision()
        {
            for (int iter = 0; iter < Iterations; iter++)
            {
                int n = _rng.Next(1, 21);
                int m = _rng.Next(1, 101);

                SetField("_rotationInterval", n);
                SetField("_triggerMode", WorldSeedRotator.RotationTriggerMode.Interval);
                SetField("_enabled", true);
                ResetInternalCounters();

                for (int i = 0; i < m; i++)
                    _rotator.NotifyEpisodeBegin();

                // 카운터 리셋이 일어난 횟수 = m / n
                // EpisodeCounter 로 역산: 현재 카운터 == m % n
                Assert.AreEqual(m % n, _rotator.EpisodeCounter,
                    $"[iter={iter}] EpisodeCounter={_rotator.EpisodeCounter}, N={n}, M={m}, " +
                    $"expected={m % n}");
            }
        }

        // ── Property 7: Boundary 모드 트리거 포인트 일관성 ───────────────
        //
        // For any N (1~20) and M (1~100) in Boundary mode (CityGenerator=null):
        //   TotalEpisodeCount % N == 0 인 시점에서만 로테이션 시도
        //   => M번 호출 후 TotalEpisodeCount == M (절대 카운트 보존)
        //
        // Validates: Boundary 모드 절대 경계값 동작

        [Test]
        public void Property7_BoundaryMode_AbsoluteCountPreserved()
        {
            for (int iter = 0; iter < Iterations; iter++)
            {
                int n = _rng.Next(1, 21);
                int m = _rng.Next(1, 101);

                SetField("_rotationInterval", n);
                SetField("_triggerMode", WorldSeedRotator.RotationTriggerMode.Boundary);
                SetField("_enabled", true);
                ResetInternalCounters();

                for (int i = 0; i < m; i++)
                    _rotator.NotifyEpisodeBegin();

                Assert.AreEqual(m, _rotator.TotalEpisodeCount,
                    $"[iter={iter}] TotalEpisodeCount 는 항상 호출 횟수와 일치해야 함: N={n}, M={m}");
            }
        }
    }
}
