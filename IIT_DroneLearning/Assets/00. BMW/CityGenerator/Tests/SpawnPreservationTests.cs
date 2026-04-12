using System.Collections.Generic;
using System.Reflection;
using NUnit.Framework;
using UnityEngine;

namespace CityGenerator
{
    /// <summary>
    /// 스폰 중첩 버그픽스 보존 속성 테스트.
    ///
    /// 이 테스트들은 수정 전/후 모두 PASS해야 한다 (회귀 없음 확인).
    /// Feature: spawn-overlap-fix
    ///
    /// 보존 대상:
    ///   - 건물이 없는 환경에서의 기존 Fallback/SpawnCenterRandom 동작
    ///   - IsComputed / OnSpawnComputed 이벤트 발생
    ///   - Goal.OnArrival 이벤트 발생
    ///   - CityMetadata 후보 부족 시 SpawnCenterRandom 폴백
    ///   - 드론 간 최소 이격 거리 보장 (건물 없는 환경)
    ///   - ComputeSpawn() 완료 후 스폰 위치 조회 가능
    ///
    /// Validates: Requirements 3.1, 3.2, 3.3, 3.6, 3.7, 3.8
    /// </summary>
    public class SpawnPreservationTests
    {
        private const int Iterations = 50;

        private GameObject _apiGameObject;
        private CityDataAPI _api;
        private GameObject _spawnCenterObject;
        private SpawnCenter _spawnCenter;
        private GameObject _coordinatorObject;
        private EpisodeSpawnCoordinator _coordinator;
        private GameObject _goalObject;
        private Goal _goal;

        private List<GameObject> _evaderObjects;
        private List<GameObject> _pursuerObjects;

        [SetUp]
        public void SetUp()
        {
            _apiGameObject = new GameObject("TestCityDataAPI");
            _api = _apiGameObject.AddComponent<CityDataAPI>();

            _spawnCenterObject = new GameObject("TestSpawnCenter");
            _spawnCenter = _spawnCenterObject.AddComponent<SpawnCenter>();

            _goalObject = new GameObject("TestGoal");
            _goalObject.tag = "Goal";
            _goal = _goalObject.AddComponent<Goal>();

            _coordinatorObject = new GameObject("TestCoordinator");
            _coordinator = _coordinatorObject.AddComponent<EpisodeSpawnCoordinator>();

            _evaderObjects = new List<GameObject>();
            _pursuerObjects = new List<GameObject>();
        }

        [TearDown]
        public void TearDown()
        {
            foreach (var go in _evaderObjects)
                if (go != null) UnityEngine.Object.DestroyImmediate(go);
            foreach (var go in _pursuerObjects)
                if (go != null) UnityEngine.Object.DestroyImmediate(go);
            _evaderObjects.Clear();
            _pursuerObjects.Clear();

            if (_goalObject != null) UnityEngine.Object.DestroyImmediate(_goalObject);
            if (_coordinatorObject != null) UnityEngine.Object.DestroyImmediate(_coordinatorObject);
            if (_spawnCenterObject != null) UnityEngine.Object.DestroyImmediate(_spawnCenterObject);
            if (_apiGameObject != null) UnityEngine.Object.DestroyImmediate(_apiGameObject);
        }

        // ────────────────────────────────────────────
        // 헬퍼
        // ────────────────────────────────────────────

        private void SetPrivateField(object target, string fieldName, object value)
        {
            var field = target.GetType().GetField(fieldName,
                BindingFlags.NonPublic | BindingFlags.Instance);
            Assert.IsNotNull(field, $"필드 '{fieldName}'을 찾을 수 없습니다.");
            field.SetValue(target, value);
        }

        private void CreateDrones(int evaderCount, int pursuerCount)
        {
            for (int i = 0; i < evaderCount; i++)
            {
                var go = new GameObject($"Evader_{i}");
                go.tag = "Evader";
                _evaderObjects.Add(go);
            }
            for (int i = 0; i < pursuerCount; i++)
            {
                var go = new GameObject($"Pursuer_{i}");
                go.tag = "Pursuer";
                _pursuerObjects.Add(go);
            }
        }

        // ────────────────────────────────────────────
        // Preservation 1: 건물 없는 환경 — Fallback 위치가 지정 범위 내에 있어야 함
        //
        // 관찰: buildings == null 또는 빈 리스트인 환경에서
        //   Fallback 전략은 기존과 동일하게 _fallbackRange / _fallbackHeight 범위 내
        //   랜덤 위치를 반환해야 한다.
        //
        // Validates: Requirement 3.1, 3.2
        // ────────────────────────────────────────────
        [Test, Category("spawn-overlap-fix")]
        public void Preservation1_EmptyBuildings_FallbackPositionInRange()
        {
            // 건물 없는 메타데이터 등록 (빈 리스트)
            var metadata = new CityMetadata
            {
                buildings = new List<Building>(),
                validSpawnCandidates = new List<GraphNode>(),
                cityBounds = new Bounds(Vector3.zero, new Vector3(100f, 50f, 100f)),
                cityGraph = new CityGraph(),
                strategicLocations = new List<StrategicLocation>()
            };
            _api.SetCityMetadata(metadata);

            float fallbackRange  = 20f;
            float fallbackHeight = 10f;
            float minSeparation  = 2f;

            CreateDrones(1, 1);
            SetPrivateField(_coordinator, "_strategy", EpisodeSpawnCoordinator.SpawnStrategy.Fallback);
            SetPrivateField(_coordinator, "_fallbackRange", fallbackRange);
            SetPrivateField(_coordinator, "_fallbackHeight", fallbackHeight);
            SetPrivateField(_coordinator, "_minSeparation", minSeparation);
            SetPrivateField(_coordinator, "_maxRetry", 20);

            for (int i = 0; i < Iterations; i++)
            {
                _coordinator.ComputeSpawn();

                Vector3 evaderPos  = _coordinator.GetSpawnPosition(_evaderObjects[0]);
                Vector3 pursuerPos = _coordinator.GetSpawnPosition(_pursuerObjects[0]);

                // Y좌표가 fallbackHeight와 일치해야 한다
                Assert.AreEqual(fallbackHeight, evaderPos.y, 0.001f,
                    $"[P1 iter={i}] Evader Y좌표({evaderPos.y:F2})가 fallbackHeight({fallbackHeight})와 다름.");
                Assert.AreEqual(fallbackHeight, pursuerPos.y, 0.001f,
                    $"[P1 iter={i}] Pursuer Y좌표({pursuerPos.y:F2})가 fallbackHeight({fallbackHeight})와 다름.");

                // XZ좌표가 fallbackRange 내에 있어야 한다
                Assert.LessOrEqual(Mathf.Abs(evaderPos.x), fallbackRange + 0.001f,
                    $"[P1 iter={i}] Evader X({evaderPos.x:F2})가 fallbackRange({fallbackRange}) 초과.");
                Assert.LessOrEqual(Mathf.Abs(evaderPos.z), fallbackRange + 0.001f,
                    $"[P1 iter={i}] Evader Z({evaderPos.z:F2})가 fallbackRange({fallbackRange}) 초과.");
                Assert.LessOrEqual(Mathf.Abs(pursuerPos.x), fallbackRange + 0.001f,
                    $"[P1 iter={i}] Pursuer X({pursuerPos.x:F2})가 fallbackRange({fallbackRange}) 초과.");
                Assert.LessOrEqual(Mathf.Abs(pursuerPos.z), fallbackRange + 0.001f,
                    $"[P1 iter={i}] Pursuer Z({pursuerPos.z:F2})가 fallbackRange({fallbackRange}) 초과.");
            }
        }

        // ────────────────────────────────────────────
        // Preservation 2: ComputeSpawn() 완료 후 IsComputed=true 및 OnSpawnComputed 발생
        //
        // Validates: Requirement 3.7
        // ────────────────────────────────────────────
        [Test, Category("spawn-overlap-fix")]
        public void Preservation2_IsComputed_AndOnSpawnComputedEventFires()
        {
            CreateDrones(1, 1);
            SetPrivateField(_coordinator, "_strategy", EpisodeSpawnCoordinator.SpawnStrategy.Fallback);
            SetPrivateField(_coordinator, "_fallbackRange", 20f);
            SetPrivateField(_coordinator, "_fallbackHeight", 10f);
            SetPrivateField(_coordinator, "_minSeparation", 1f);
            SetPrivateField(_coordinator, "_maxRetry", 20);

            for (int i = 0; i < Iterations; i++)
            {
                bool eventFired = false;
                _coordinator.OnSpawnComputed += () => eventFired = true;

                _coordinator.ComputeSpawn();

                Assert.IsTrue(_coordinator.IsComputed,
                    $"[P2 iter={i}] ComputeSpawn() 완료 후 IsComputed가 true여야 한다.");
                Assert.IsTrue(eventFired,
                    $"[P2 iter={i}] ComputeSpawn() 완료 후 OnSpawnComputed 이벤트가 발생해야 한다.");

                // 이벤트 리스너 해제
                _coordinator.OnSpawnComputed -= () => eventFired = true;
            }
        }

        // ────────────────────────────────────────────
        // Preservation 3: Goal.OnArrival — Evader 도달 판정 이벤트 보존
        //
        // OnTriggerEnter는 PlayMode에서만 동작하므로, NotifyArrival() 직접 호출로
        // 이벤트 연결 자체가 정상 동작함을 검증한다.
        //
        // Validates: Requirement 3.6
        // ────────────────────────────────────────────
        [Test, Category("spawn-overlap-fix")]
        public void Preservation3_Goal_OnArrivalEventFires()
        {
            bool arrivedFired = false;
            _goal.OnArrival += (g) => arrivedFired = true;

            _goal.NotifyArrival();

            Assert.IsTrue(arrivedFired,
                "[P3] Goal.NotifyArrival() 호출 후 OnArrival 이벤트가 발생해야 한다.");
        }

        // ────────────────────────────────────────────
        // Preservation 4: CityMetadata 후보 부족 시 SpawnCenterRandom으로 폴백
        //
        // 관찰: validSpawnCandidates가 필요 수보다 적으면 SpawnCenterRandom으로 폴백하여
        //   ComputeSpawn()이 정상 완료되어야 한다.
        //
        // Validates: Requirement 3.8
        // ────────────────────────────────────────────
        [Test, Category("spawn-overlap-fix")]
        public void Preservation4_CityMetadata_FallsBackToSpawnCenterRandom_WhenCandidatesInsufficient()
        {
            // 후보 노드 1개 (필요 수 = 1E+1P+1G = 3 → 부족)
            var graph = new CityGraph();
            int id = graph.AddNode(new Vector3(0f, 0f, 0f), NodeType.OpenSpace, 0f);
            var metadata = new CityMetadata
            {
                buildings = new List<Building>(),
                validSpawnCandidates = new List<GraphNode> { graph.GetNode(id) },
                cityBounds = new Bounds(Vector3.zero, new Vector3(100f, 50f, 100f)),
                cityGraph = graph,
                strategicLocations = new List<StrategicLocation>()
            };
            _api.SetCityMetadata(metadata);

            // SpawnCenter 범위 설정 (폴백 시 사용)
            var sharedRange = new SpawnCenter.SpawnRange(5f, 20f, 30f);
            SetPrivateField(_spawnCenter, "_sharedRange", sharedRange);
            SetPrivateField(_spawnCenter, "_syncMode", SpawnCenter.SyncMode.Synchronized);
            SetPrivateField(_spawnCenter, "_autoSyncFromCity", false);

            CreateDrones(1, 1);
            SetPrivateField(_coordinator, "_strategy", EpisodeSpawnCoordinator.SpawnStrategy.CityMetadata);
            SetPrivateField(_coordinator, "_minSeparation", 1f);
            SetPrivateField(_coordinator, "_maxRetry", 20);

            _coordinator.ComputeSpawn();

            // 폴백 발생해도 ComputeSpawn()이 정상 완료되어야 한다
            Assert.IsTrue(_coordinator.IsComputed,
                "[P4] CityMetadata 후보 부족 시 폴백 후에도 IsComputed가 true여야 한다.");

            // 드론 위치가 Vector3.zero가 아니어야 한다 (스폰이 수행됨)
            Vector3 evaderPos  = _coordinator.GetSpawnPosition(_evaderObjects[0]);
            Vector3 pursuerPos = _coordinator.GetSpawnPosition(_pursuerObjects[0]);
            Assert.AreNotEqual(Vector3.zero, evaderPos,
                "[P4] 폴백 후 Evader 스폰 위치가 Vector3.zero이면 안 된다.");
            Assert.AreNotEqual(Vector3.zero, pursuerPos,
                "[P4] 폴백 후 Pursuer 스폰 위치가 Vector3.zero이면 안 된다.");
        }

        // ────────────────────────────────────────────
        // Preservation 5: 건물 없는 환경 — 모든 엔티티 쌍의 최소 이격 거리 보장
        //
        // 관찰: 건물이 없는 환경에서 _minSeparation 기반 이격 거리 검사는
        //   기존과 동일하게 동작해야 한다.
        //
        // Validates: Requirement 3.1
        // ────────────────────────────────────────────
        [Test, Category("spawn-overlap-fix")]
        public void Preservation5_EmptyBuildings_MinSeparationPreserved()
        {
            var metadata = new CityMetadata
            {
                buildings = new List<Building>(),
                validSpawnCandidates = new List<GraphNode>(),
                cityBounds = new Bounds(Vector3.zero, new Vector3(200f, 50f, 200f)),
                cityGraph = new CityGraph(),
                strategicLocations = new List<StrategicLocation>()
            };
            _api.SetCityMetadata(metadata);

            float minSeparation = 5f;
            float fallbackRange = 50f;

            CreateDrones(2, 2);
            SetPrivateField(_coordinator, "_strategy", EpisodeSpawnCoordinator.SpawnStrategy.Fallback);
            SetPrivateField(_coordinator, "_fallbackRange", fallbackRange);
            SetPrivateField(_coordinator, "_fallbackHeight", 15f);
            SetPrivateField(_coordinator, "_minSeparation", minSeparation);
            SetPrivateField(_coordinator, "_maxRetry", 50);

            for (int i = 0; i < Iterations; i++)
            {
                _coordinator.ComputeSpawn();

                // 모든 엔티티 위치 수집
                var positions = new List<Vector3>();
                foreach (var go in _evaderObjects)
                    positions.Add(_coordinator.GetSpawnPosition(go));
                foreach (var go in _pursuerObjects)
                    positions.Add(_coordinator.GetSpawnPosition(go));

                // 모든 쌍에 대해 이격 거리 검사
                for (int a = 0; a < positions.Count; a++)
                {
                    for (int b = a + 1; b < positions.Count; b++)
                    {
                        float dist = Vector3.Distance(positions[a], positions[b]);
                        Assert.GreaterOrEqual(dist, minSeparation - 0.01f,
                            $"[P5 iter={i}] 위치[{a}]({positions[a]})~위치[{b}]({positions[b]}) " +
                            $"이격 거리({dist:F2}m)가 _minSeparation({minSeparation}m) 미만임. " +
                            $"건물 없는 환경에서 이격 거리 보장이 깨짐.");
                    }
                }
            }
        }

        // ────────────────────────────────────────────
        // Preservation 6: SpawnCenter Synchronized SyncMode — 모든 엔티티가 공용 범위 사용
        //
        // 관찰: SyncMode.Synchronized 시 골존·드론 모두 _sharedRange를 사용해야 한다.
        //
        // Validates: Requirement 3.4
        // ────────────────────────────────────────────
        [Test, Category("spawn-overlap-fix")]
        public void Preservation6_SyncMode_Synchronized_AllEntitiesInSharedRange()
        {
            float sharedRadius = 30f;
            float sharedMinY   = 5f;
            float sharedMaxY   = 25f;

            var sharedRange = new SpawnCenter.SpawnRange(sharedMinY, sharedMaxY, sharedRadius);
            SetPrivateField(_spawnCenter, "_sharedRange", sharedRange);
            SetPrivateField(_spawnCenter, "_syncMode", SpawnCenter.SyncMode.Synchronized);
            SetPrivateField(_spawnCenter, "_autoSyncFromCity", false);

            var metadata = new CityMetadata
            {
                buildings = new List<Building>(),
                validSpawnCandidates = new List<GraphNode>(),
                cityBounds = new Bounds(Vector3.zero, new Vector3(200f, 50f, 200f)),
                cityGraph = new CityGraph(),
                strategicLocations = new List<StrategicLocation>()
            };
            _api.SetCityMetadata(metadata);

            CreateDrones(1, 1);
            SetPrivateField(_coordinator, "_strategy", EpisodeSpawnCoordinator.SpawnStrategy.SpawnCenterRandom);
            SetPrivateField(_coordinator, "_minSeparation", 1f);
            SetPrivateField(_coordinator, "_maxRetry", 20);

            for (int i = 0; i < Iterations; i++)
            {
                _coordinator.ComputeSpawn();

                Vector3 evaderPos  = _coordinator.GetSpawnPosition(_evaderObjects[0]);
                Vector3 pursuerPos = _coordinator.GetSpawnPosition(_pursuerObjects[0]);
                Vector3 goalPos    = _coordinator.GetGoalPosition();

                // 모든 엔티티 Y좌표가 sharedRange 내에 있어야 한다
                foreach (var (label, pos) in new[]
                {
                    ("Evader", evaderPos), ("Pursuer", pursuerPos), ("Goal", goalPos)
                })
                {
                    Assert.GreaterOrEqual(pos.y, sharedMinY - 0.001f,
                        $"[P6 iter={i}] {label} Y({pos.y:F2})가 sharedMinY({sharedMinY}) 미만.");
                    Assert.LessOrEqual(pos.y, sharedMaxY + 0.001f,
                        $"[P6 iter={i}] {label} Y({pos.y:F2})가 sharedMaxY({sharedMaxY}) 초과.");
                }
            }
        }
    }
}