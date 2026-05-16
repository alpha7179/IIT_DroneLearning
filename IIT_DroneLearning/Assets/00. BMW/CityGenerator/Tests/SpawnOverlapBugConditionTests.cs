using System.Collections.Generic;
using System.Reflection;
using NUnit.Framework;
using UnityEngine;

namespace CityGenerator
{
    /// <summary>
    /// 스폰 중첩 버그 조건 탐색 테스트.
    ///
    /// 이 테스트들은 수정 전 코드에서 FAIL, 수정 후 코드에서 PASS해야 한다.
    /// Feature: spawn-overlap-fix
    ///
    /// Bug Condition: EpisodeSpawnCoordinator가 건물 Bounds 충돌을 검사하지 않아
    ///   드론·골존이 건물 내부에 스폰됨.
    /// Expected Behavior (수정 후):
    ///   모든 스폰 위치·골존이 건물 Bounds 외부에 위치해야 한다.
    ///
    /// Validates: Requirements 2.1, 2.2, 2.3, 2.6
    /// </summary>
    public class SpawnOverlapBugConditionTests
    {
        // 반복 횟수: 확률적 테스트 신뢰도 확보
        private const int Iterations = 30;

        // 씬 오브젝트
        private GameObject _apiGameObject;
        private CityDataAPI _api;
        private GameObject _spawnCenterObject;
        private SpawnCenter _spawnCenter;
        private GameObject _coordinatorObject;
        private EpisodeSpawnCoordinator _coordinator;
        private GameObject _goalObject;
        private Goal _goal;

        // 태그된 드론 오브젝트
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

        /// <summary>
        /// 주어진 위치 목록 중 건물 Bounds와 중첩되는 것이 있으면 true를 반환한다.
        /// </summary>
        private bool AnyInsideBuilding(List<Vector3> positions, List<Building> buildings)
        {
            foreach (var pos in positions)
                foreach (var b in buildings)
                    if (b.bounds.Contains(pos))
                        return true;
            return false;
        }

        /// <summary>
        /// 테스트용 Building을 생성한다.
        /// Building(GameObject, Vector3 position, Vector3 size, float height, GridCell)
        /// bounds = new Bounds(position, size) → min = position - size/2, max = position + size/2
        /// </summary>
        private Building MakeBuilding(Vector3 center, Vector3 size)
        {
            var cell = new GridCell
            {
                x = 0, z = 0,
                hasBuilding = true,
                buildingHeight = size.y,
                worldPosition = center
            };
            return new Building(null, center, size, size.y, cell);
        }

        /// <summary>
        /// CityGraph에 노드를 추가하고 GraphNode를 반환하는 헬퍼.
        /// </summary>
        private GraphNode AddNode(CityGraph graph, float x, float z, float elevation = 0f)
        {
            int id = graph.AddNode(new Vector3(x, 0f, z), NodeType.OpenSpace, elevation);
            return graph.GetNode(id);
        }

        // ────────────────────────────────────────────
        // Bug Condition 1: Fallback 전략 — 드론이 건물 내부에 스폰됨
        //
        // 시나리오:
        //   건물이 Fallback 범위의 81%를 덮음.
        //   현재 코드: 건물 충돌을 무시하고 랜덤 위치를 선택 → 81% 확률로 건물 내부 스폰.
        //   수정 코드: 건물 외부 위치만 선택 → 0% 확률로 건물 내부 스폰.
        //
        // EXPECTED: FAIL before fix (P(fail) ≈ 1 - 0.19^30 ≈ 100%)
        //           PASS after fix
        //
        // Validates: Requirements 1.1, 2.1
        // ────────────────────────────────────────────
        [Test, Category("spawn-overlap-fix")]
        public void BugCondition1_FallbackStrategy_DroneNotSpawnedInsideBuilding()
        {
            // Building: center=(0,15,0), size=(18,30,18)
            // → XZ bounds [-9,9] × [-9,9], Y [0,30]
            // Fallback range=10 → XZ [-10,10] → 81% covered by building
            var building = MakeBuilding(new Vector3(0f, 15f, 0f), new Vector3(18f, 30f, 18f));
            var buildings = new List<Building> { building };

            var metadata = new CityMetadata
            {
                buildings = buildings,
                validSpawnCandidates = new List<GraphNode>(),
                cityBounds = new Bounds(Vector3.zero, new Vector3(100f, 50f, 100f)),
                cityGraph = new CityGraph(),
                strategicLocations = new List<StrategicLocation>()
            };
            _api.SetCityMetadata(metadata);

            CreateDrones(1, 1);
            SetPrivateField(_coordinator, "_strategy", EpisodeSpawnCoordinator.SpawnStrategy.Fallback);
            SetPrivateField(_coordinator, "_fallbackRange", 10f);
            SetPrivateField(_coordinator, "_fallbackHeight", 15f);
            SetPrivateField(_coordinator, "_minSeparation", 0.5f);
            SetPrivateField(_coordinator, "_maxRetry", 50);

            int failures = 0;
            for (int i = 0; i < Iterations; i++)
            {
                _coordinator.ComputeSpawn();

                var positions = new List<Vector3>
                {
                    _coordinator.GetSpawnPosition(_evaderObjects[0]),
                    _coordinator.GetSpawnPosition(_pursuerObjects[0])
                };

                if (AnyInsideBuilding(positions, buildings))
                    failures++;
            }

            // 수정 전: ~81%×30회 ≈ 24회 실패 → Assert 실패 (버그 확인)
            // 수정 후: 0회 실패 → Assert 통과
            Assert.AreEqual(0, failures,
                $"[BugCondition1] {Iterations}회 중 {failures}회에서 드론이 건물 Bounds 내부에 " +
                $"스폰됨. EpisodeSpawnCoordinator.AssignSpawn()에 건물 충돌 검사가 없음.");
        }

        // ────────────────────────────────────────────
        // Bug Condition 2: CityMetadata 전략 — 엔티티가 건물 내부에 스폰됨
        //
        // 시나리오:
        //   후보 노드 10개 중 4개가 건물 Bounds 내부에 위치.
        //   현재 코드: 건물 충돌 무시 → 40% 확률로 건물 내부 노드 선택.
        //   수정 코드: 건물 내부 노드를 거부, 외부 노드만 선택.
        //
        // EXPECTED: FAIL before fix (P(fail) ≈ 1 - 0.6^30 ≈ 100%)
        //           PASS after fix
        //
        // Validates: Requirements 1.3, 2.3
        // ────────────────────────────────────────────
        [Test, Category("spawn-overlap-fix")]
        public void BugCondition2_CityMetadataStrategy_EntityNotSpawnedInsideBuilding()
        {
            // Building: center=(5,5,5), size=(8,10,8) → XZ [1,9] × [1,9]
            var building = MakeBuilding(new Vector3(5f, 5f, 5f), new Vector3(8f, 10f, 8f));
            var buildings = new List<Building> { building };

            var graph = new CityGraph();

            // 건물 내부 노드 4개
            var n1 = AddNode(graph, 2f, 2f);    // inside [1,9]×[1,9]
            var n2 = AddNode(graph, 4f, 4f);
            var n3 = AddNode(graph, 6f, 6f);
            var n4 = AddNode(graph, 8f, 8f);

            // 건물 외부 노드 6개 (충분히 이격)
            var n5  = AddNode(graph, 100f, 100f);
            var n6  = AddNode(graph, 110f, 100f);
            var n7  = AddNode(graph, 120f, 100f);
            var n8  = AddNode(graph, 100f, 110f);
            var n9  = AddNode(graph, 110f, 110f);
            var n10 = AddNode(graph, 120f, 110f);

            var candidates = new List<GraphNode> { n1, n2, n3, n4, n5, n6, n7, n8, n9, n10 };

            var metadata = new CityMetadata
            {
                buildings = buildings,
                validSpawnCandidates = candidates,
                cityBounds = new Bounds(new Vector3(60f, 10f, 60f), new Vector3(200f, 30f, 200f)),
                cityGraph = graph,
                strategicLocations = new List<StrategicLocation>(),
                minBuildingHeight = 5f,
                maxBuildingHeight = 10f
            };
            _api.SetCityMetadata(metadata);

            CreateDrones(1, 1);
            SetPrivateField(_coordinator, "_strategy", EpisodeSpawnCoordinator.SpawnStrategy.CityMetadata);
            SetPrivateField(_coordinator, "_minSeparation", 1f);
            SetPrivateField(_coordinator, "_minSpawnHeight", 5f);
            SetPrivateField(_coordinator, "_maxSpawnHeight", 20f);
            SetPrivateField(_coordinator, "_maxRetry", 50);

            int failures = 0;
            for (int i = 0; i < Iterations; i++)
            {
                _coordinator.ComputeSpawn();

                var positions = new List<Vector3>
                {
                    _coordinator.GetSpawnPosition(_evaderObjects[0]),
                    _coordinator.GetSpawnPosition(_pursuerObjects[0]),
                    _coordinator.GetGoalPosition()
                };

                if (AnyInsideBuilding(positions, buildings))
                    failures++;
            }

            // 수정 전: 40%×30회 ≈ 12회 실패 → Assert 실패 (버그 확인)
            // 수정 후: 0회 실패 → Assert 통과
            Assert.AreEqual(0, failures,
                $"[BugCondition2] {Iterations}회 중 {failures}회에서 엔티티(드론/골존)가 건물 Bounds " +
                $"내부 노드에 스폰됨. SelectCandidateWithSeparation()에 건물 충돌 검사가 없음.");
        }

        // ────────────────────────────────────────────
        // Bug Condition 3: SpawnCenterRandom 전략 — 골존이 건물 내부에 배치됨
        //
        // 시나리오:
        //   SpawnCenter 범위가 건물과 대부분 겹침 (81% 커버).
        //   현재 코드: ComputeGoalPosition()이 건물 충돌 무시 → 건물 내부에 골존 배치.
        //   수정 코드: 건물 외부 위치에 골존 배치.
        //
        // EXPECTED: FAIL before fix (P(fail) ≈ 1 - 0.19^30 ≈ 100%)
        //           PASS after fix
        //
        // Validates: Requirements 1.2, 2.2
        // ────────────────────────────────────────────
        [Test, Category("spawn-overlap-fix")]
        public void BugCondition3_SpawnCenterRandom_GoalNotInsideBuilding()
        {
            // Building: center=(0,15,0), size=(18,30,18) → XZ [-9,9] × [-9,9]
            var building = MakeBuilding(new Vector3(0f, 15f, 0f), new Vector3(18f, 30f, 18f));
            var buildings = new List<Building> { building };

            var metadata = new CityMetadata
            {
                buildings = buildings,
                validSpawnCandidates = new List<GraphNode>(),
                cityBounds = new Bounds(Vector3.zero, new Vector3(100f, 50f, 100f)),
                cityGraph = new CityGraph(),
                strategicLocations = new List<StrategicLocation>()
            };
            _api.SetCityMetadata(metadata);

            // SpawnCenter: Synchronized, Radius=9 → 반경 9m 원 내부 (대부분 건물 내부)
            // SpawnCenter 범위를 Reflection으로 설정
            SetPrivateField(_spawnCenter, "_syncMode", SpawnCenter.SyncMode.Synchronized);
            SetPrivateField(_spawnCenter, "_sharedMinY", 10f);
            SetPrivateField(_spawnCenter, "_sharedMaxY", 20f);
            SetPrivateField(_spawnCenter, "_sharedRadius", 9f);

            CreateDrones(1, 1);
            SetPrivateField(_coordinator, "_strategy", EpisodeSpawnCoordinator.SpawnStrategy.SpawnCenterRandom);
            SetPrivateField(_coordinator, "_minSeparation", 0.5f);
            SetPrivateField(_coordinator, "_maxRetry", 50);

            int failures = 0;
            for (int i = 0; i < Iterations; i++)
            {
                _coordinator.ComputeSpawn();

                Vector3 goalPos = _coordinator.GetGoalPosition();
                if (building.bounds.Contains(goalPos))
                    failures++;
            }

            // 수정 전: 골존이 반경 9m 원 내부 → 대부분 XZ [-9,9] 건물 내부 → 실패
            // 수정 후: 건물 외부 위치에 골존 배치 → 0회 실패
            Assert.AreEqual(0, failures,
                $"[BugCondition3] {Iterations}회 중 {failures}회에서 골존이 건물 Bounds 내부에 " +
                $"배치됨. ComputeGoalPosition()에 건물 충돌 검사가 없음.");
        }
    }
}