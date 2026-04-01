using System;
using System.Collections.Generic;
using System.Reflection;
using NUnit.Framework;
using UnityEngine;

namespace CityGenerator
{
    /// <summary>
    /// EpisodeSpawnCoordinator 속성 기반 테스트.
    ///
    /// FsCheck 대신 NUnit + 반복 루프(100회)로 구현한다.
    /// 각 테스트 메서드는 설계 문서의 Property 번호를 태그로 참조한다.
    /// Feature: city-spawn-refactor
    ///
    /// 테스트 환경 구성:
    ///   - "Evader" / "Pursuer" 태그가 지정된 GameObject 생성
    ///   - CityDataAPI 인스턴스 생성 및 CityMetadata 등록
    ///   - SpawnCenter 인스턴스 생성
    ///   - EpisodeSpawnCoordinator 인스턴스 생성 및 전략 설정
    /// </summary>
    public class EpisodeSpawnCoordinatorPropertyTests
    {
        private const int Iterations = 100;

        private System.Random _rng;

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
            _rng = new System.Random(42);

            // CityDataAPI 싱글톤 생성
            _apiGameObject = new GameObject("TestCityDataAPI");
            _api = _apiGameObject.AddComponent<CityDataAPI>();

            // SpawnCenter 생성
            _spawnCenterObject = new GameObject("TestSpawnCenter");
            _spawnCenter = _spawnCenterObject.AddComponent<SpawnCenter>();

            // Goal 생성
            _goalObject = new GameObject("TestGoal");
            _goalObject.tag = "Goal";
            _goal = _goalObject.AddComponent<Goal>();

            // EpisodeSpawnCoordinator 생성
            _coordinatorObject = new GameObject("TestCoordinator");
            _coordinator = _coordinatorObject.AddComponent<EpisodeSpawnCoordinator>();

            // 드론 오브젝트 리스트 초기화
            _evaderObjects = new List<GameObject>();
            _pursuerObjects = new List<GameObject>();
        }

        [TearDown]
        public void TearDown()
        {
            // 드론 오브젝트 정리
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
        // 헬퍼: 랜덤 유틸리티
        // ────────────────────────────────────────────

        private float Rand(float min, float max) =>
            (float)(_rng.NextDouble() * (max - min) + min);

        private int RandInt(int min, int max) =>
            _rng.Next(min, max + 1);

        // ────────────────────────────────────────────
        // 헬퍼: Reflection을 통한 private 필드 설정
        // ────────────────────────────────────────────

        private void SetPrivateField(object target, string fieldName, object value)
        {
            var field = target.GetType().GetField(fieldName,
                BindingFlags.NonPublic | BindingFlags.Instance);
            Assert.IsNotNull(field, $"필드 '{fieldName}'을 찾을 수 없습니다.");
            field.SetValue(target, value);
        }

        private T GetPrivateField<T>(object target, string fieldName)
        {
            var field = target.GetType().GetField(fieldName,
                BindingFlags.NonPublic | BindingFlags.Instance);
            Assert.IsNotNull(field, $"필드 '{fieldName}'을 찾을 수 없습니다.");
            return (T)field.GetValue(target);
        }

        // ────────────────────────────────────────────
        // 헬퍼: 태그된 드론 GameObject 생성
        // ────────────────────────────────────────────

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

        private void ClearDrones()
        {
            foreach (var go in _evaderObjects)
                if (go != null) UnityEngine.Object.DestroyImmediate(go);
            foreach (var go in _pursuerObjects)
                if (go != null) UnityEngine.Object.DestroyImmediate(go);
            _evaderObjects.Clear();
            _pursuerObjects.Clear();
        }

        // ────────────────────────────────────────────
        // 헬퍼: 유효 스폰 후보 노드를 포함하는 CityMetadata 생성
        // ────────────────────────────────────────────

        /// <summary>
        /// 충분한 수의 유효 스폰 후보 노드를 가진 CityMetadata를 생성한다.
        /// 각 노드는 OpenSpace/Intersection 타입이고, 엣지가 존재하며,
        /// 건물 Bounds 외부에 위치하고, 서로 충분히 떨어져 있다.
        /// </summary>
        private CityMetadata GenerateCityMetadataForSpawn(int requiredCandidates, float minSeparation)
        {
            int width = RandInt(10, 50);
            int depth = RandInt(10, 50);
            float minHeight = Rand(1f, 50f);
            float maxHeight = Rand(minHeight + 1f, 100f);
            int seed = RandInt(0, int.MaxValue);
            CityLayoutMode mode = (CityLayoutMode)RandInt(0, 2);

            float boundsWidth = width * Rand(5f, 10f);
            float boundsDepth = depth * Rand(5f, 10f);
            float boundsHeight = maxHeight + Rand(10f, 50f);
            var center = new Vector3(0f, boundsHeight * 0.5f, 0f);
            var bounds = new Bounds(center, new Vector3(boundsWidth, boundsHeight, boundsDepth));

            // 건물 목록 (소수, 중심에서 떨어진 위치)
            var buildings = new List<Building>();
            int buildingCount = RandInt(0, 5);
            for (int b = 0; b < buildingCount; b++)
            {
                var pos = new Vector3(Rand(-10f, 10f), 0f, Rand(-10f, 10f));
                float bh = Rand(minHeight, maxHeight);
                var sz = new Vector3(Rand(1f, 3f), bh, Rand(1f, 3f));
                var cell = new GridCell { x = b, z = 0, hasBuilding = true, buildingHeight = bh, worldPosition = pos };
                buildings.Add(new Building(null, pos, sz, bh, cell));
            }

            // 그래프 생성 — 충분한 수의 유효 후보 노드 생성
            var graph = new CityGraph();
            var candidates = new List<GraphNode>();

            // 노드를 넓은 영역에 분산 배치하여 이격 거리 조건을 쉽게 만족하도록 함
            float spacing = Mathf.Max(minSeparation * 2f, 20f);
            int gridSize = Mathf.CeilToInt(Mathf.Sqrt(requiredCandidates + 5));

            for (int gx = 0; gx < gridSize; gx++)
            {
                for (int gz = 0; gz < gridSize; gz++)
                {
                    if (candidates.Count >= requiredCandidates + 5) break;

                    NodeType type = _rng.NextDouble() < 0.5 ? NodeType.OpenSpace : NodeType.Intersection;
                    float elevation = Rand(0f, 20f);
                    // 넓은 간격으로 배치 + 약간의 랜덤 오프셋
                    float x = (gx - gridSize / 2f) * spacing + Rand(-2f, 2f);
                    float z = (gz - gridSize / 2f) * spacing + Rand(-2f, 2f);
                    Vector3 pos = new Vector3(x, 0f, z);

                    int nodeId = graph.AddNode(pos, type, elevation);

                    // 이전 노드와 엣지 연결 (최소 1개 엣지 보장)
                    var allNodes = graph.GetAllNodes();
                    if (allNodes.Count > 1)
                    {
                        int prevIdx = allNodes.Count - 2;
                        var prevNode = allNodes[prevIdx];
                        float cost = Vector3.Distance(pos, prevNode.position);
                        graph.AddEdge(nodeId, prevNode.nodeId, cost, PathType.Direct);
                        graph.AddEdge(prevNode.nodeId, nodeId, cost, PathType.Direct);
                    }

                    candidates.Add(graph.GetNode(nodeId));
                }
                if (candidates.Count >= requiredCandidates + 5) break;
            }

            return new CityMetadata
            {
                cityBounds = bounds,
                actualCityWidth = width,
                actualCityDepth = depth,
                minBuildingHeight = minHeight,
                maxBuildingHeight = maxHeight,
                buildings = buildings,
                cityGraph = graph,
                strategicLocations = new List<StrategicLocation>(),
                validSpawnCandidates = candidates,
                usedRandomSeed = seed,
                layoutMode = mode
            };
        }

        /// <summary>
        /// 스폰 위치의 XZ 좌표가 후보 노드 중 하나와 일치하는지 확인한다.
        /// </summary>
        private bool MatchesAnyCandidateXZ(Vector3 spawnPos, List<GraphNode> candidates, float tolerance = 0.01f)
        {
            foreach (var node in candidates)
            {
                if (Mathf.Abs(spawnPos.x - node.position.x) < tolerance &&
                    Mathf.Abs(spawnPos.z - node.position.z) < tolerance)
                {
                    return true;
                }
            }
            return false;
        }

        /// <summary>
        /// 스폰 위치의 XZ 좌표와 일치하는 후보 노드를 찾아 반환한다.
        /// </summary>
        private GraphNode? FindMatchingCandidate(Vector3 spawnPos, List<GraphNode> candidates, float tolerance = 0.01f)
        {
            foreach (var node in candidates)
            {
                if (Mathf.Abs(spawnPos.x - node.position.x) < tolerance &&
                    Mathf.Abs(spawnPos.z - node.position.z) < tolerance)
                {
                    return node;
                }
            }
            return null;
        }

        // ────────────────────────────────────────────
        // Feature: city-spawn-refactor, Property 7: CityMetadata 전략 스폰 위치 유효성
        // **Validates: Requirements 4.2, 4.5, 4.6, 4.10**
        //
        // For any 유효한 CityMetadata가 등록된 상태에서 SpawnStrategy가 CityMetadata로
        // 설정된 EpisodeSpawnCoordinator의 ComputeSpawn() 호출 후,
        // 모든 Evader/Pursuer 스폰 위치의 XZ 좌표가 validSpawnCandidates 노드 중 하나의
        // 위치와 일치하고, Goal 위치의 XZ 좌표도 validSpawnCandidates 노드 중 하나의
        // 위치와 일치해야 한다.
        // ────────────────────────────────────────────

        [Test, Category("city-spawn-refactor")]
        public void Property7_CityMetadata_SpawnPositionValidity()
        {
            for (int i = 0; i < Iterations; i++)
            {
                // 드론 수 랜덤 결정 (1~3 Evader, 1~2 Pursuer)
                int evaderCount = RandInt(1, 3);
                int pursuerCount = RandInt(1, 2);
                int totalDrones = evaderCount + pursuerCount;
                int requiredPositions = totalDrones + 1; // +1 for Goal

                // 드론 오브젝트 생성
                ClearDrones();
                CreateDrones(evaderCount, pursuerCount);

                // 충분한 후보 노드를 가진 CityMetadata 생성
                float minSeparation = Rand(1f, 5f);
                CityMetadata metadata = GenerateCityMetadataForSpawn(requiredPositions + 5, minSeparation);

                // CityDataAPI에 메타데이터 등록
                _api.SetCityMetadata(metadata);

                // EpisodeSpawnCoordinator 설정
                SetPrivateField(_coordinator, "_strategy",
                    EpisodeSpawnCoordinator.SpawnStrategy.CityMetadata);
                SetPrivateField(_coordinator, "_minSeparation", minSeparation);
                SetPrivateField(_coordinator, "_minSpawnHeight", 8f);
                SetPrivateField(_coordinator, "_maxSpawnHeight", 50f);
                SetPrivateField(_coordinator, "_maxRetry", 20);

                // ComputeSpawn 호출
                _coordinator.ComputeSpawn();

                Assert.IsTrue(_coordinator.IsComputed,
                    $"[P7 iter={i}] ComputeSpawn() 후 IsComputed가 true여야 합니다.");

                var candidates = metadata.validSpawnCandidates;

                // Evader 스폰 위치 검증
                for (int e = 0; e < evaderCount; e++)
                {
                    Vector3 pos = _coordinator.GetSpawnPosition(_evaderObjects[e]);
                    Assert.IsTrue(MatchesAnyCandidateXZ(pos, candidates),
                        $"[P7 iter={i}] Evader[{e}] 스폰 위치 XZ({pos.x:F2}, {pos.z:F2})가 " +
                        $"validSpawnCandidates 노드 중 하나와 일치해야 합니다.");
                }

                // Pursuer 스폰 위치 검증
                for (int p = 0; p < pursuerCount; p++)
                {
                    Vector3 pos = _coordinator.GetSpawnPosition(_pursuerObjects[p]);
                    Assert.IsTrue(MatchesAnyCandidateXZ(pos, candidates),
                        $"[P7 iter={i}] Pursuer[{p}] 스폰 위치 XZ({pos.x:F2}, {pos.z:F2})가 " +
                        $"validSpawnCandidates 노드 중 하나와 일치해야 합니다.");
                }

                // Goal 스폰 위치 검증
                Vector3 goalPos = _coordinator.GetGoalPosition();
                Assert.IsTrue(MatchesAnyCandidateXZ(goalPos, candidates),
                    $"[P7 iter={i}] Goal 스폰 위치 XZ({goalPos.x:F2}, {goalPos.z:F2})가 " +
                    $"validSpawnCandidates 노드 중 하나와 일치해야 합니다.");
            }
        }

        // ────────────────────────────────────────────
        // Feature: city-spawn-refactor, Property 8: 스폰 위치 간 최소 이격 거리 불변량
        // **Validates: Requirements 4.7**
        //
        // For any ComputeSpawn() 호출 후 (모든 전략에 대해),
        // 확정된 모든 스폰 위치 쌍(Evader, Pursuer, Goal) 간의 거리가
        // _minSeparation 이상이거나, 재시도 횟수 초과 시 마지막 후보가 사용되어야 한다.
        // ────────────────────────────────────────────

        [Test, Category("city-spawn-refactor")]
        public void Property8_MinSeparationDistance_Invariant()
        {
            for (int i = 0; i < Iterations; i++)
            {
                // 드론 수 랜덤 결정
                int evaderCount = RandInt(1, 3);
                int pursuerCount = RandInt(1, 2);
                int totalDrones = evaderCount + pursuerCount;
                int requiredPositions = totalDrones + 1;

                // 드론 오브젝트 생성
                ClearDrones();
                CreateDrones(evaderCount, pursuerCount);

                // 이격 거리를 작게 설정하여 충분한 후보가 있으면 항상 만족하도록 함
                float minSeparation = Rand(1f, 5f);

                // 넓은 영역에 충분한 후보 노드를 배치한 CityMetadata 생성
                CityMetadata metadata = GenerateCityMetadataForSpawn(requiredPositions + 10, minSeparation);
                _api.SetCityMetadata(metadata);

                // CityMetadata 전략으로 설정
                SetPrivateField(_coordinator, "_strategy",
                    EpisodeSpawnCoordinator.SpawnStrategy.CityMetadata);
                SetPrivateField(_coordinator, "_minSeparation", minSeparation);
                SetPrivateField(_coordinator, "_minSpawnHeight", 8f);
                SetPrivateField(_coordinator, "_maxSpawnHeight", 50f);
                SetPrivateField(_coordinator, "_maxRetry", 20);

                _coordinator.ComputeSpawn();

                Assert.IsTrue(_coordinator.IsComputed,
                    $"[P8 iter={i}] ComputeSpawn() 후 IsComputed가 true여야 합니다.");

                // 모든 스폰 위치 수집
                var allPositions = new List<Vector3>();
                for (int e = 0; e < evaderCount; e++)
                    allPositions.Add(_coordinator.GetSpawnPosition(_evaderObjects[e]));
                for (int p = 0; p < pursuerCount; p++)
                    allPositions.Add(_coordinator.GetSpawnPosition(_pursuerObjects[p]));
                allPositions.Add(_coordinator.GetGoalPosition());

                // 모든 위치 쌍 간 거리 검증
                // 충분한 후보가 있으므로 이격 거리가 보장되어야 한다.
                // (후보가 부족하여 재시도 초과 시 마지막 후보가 사용될 수 있으므로,
                //  이 테스트에서는 충분한 후보를 제공하여 항상 이격 조건을 만족하도록 한다.)
                int violationCount = 0;
                for (int a = 0; a < allPositions.Count; a++)
                {
                    for (int b = a + 1; b < allPositions.Count; b++)
                    {
                        float dist = Vector3.Distance(allPositions[a], allPositions[b]);
                        if (dist < minSeparation - 0.01f) // 부동소수점 오차 허용
                        {
                            violationCount++;
                        }
                    }
                }

                // 충분한 후보가 있는 경우 이격 거리 위반이 없어야 한다
                Assert.AreEqual(0, violationCount,
                    $"[P8 iter={i}] 충분한 후보 노드가 있는 상태에서 " +
                    $"이격 거리({minSeparation:F2}m) 위반이 {violationCount}건 발생했습니다. " +
                    $"위치: {string.Join(", ", allPositions)}");
            }
        }

        // ────────────────────────────────────────────
        // Feature: city-spawn-refactor, Property 9: CityMetadata 전략 스폰 Y좌표 범위
        // **Validates: Requirements 4.8**
        //
        // For any CityMetadata 전략으로 ComputeSpawn() 호출 후,
        // 모든 스폰 위치의 Y좌표가 해당 노드의 elevation + minSpawnHeight 이상이고
        // elevation + maxSpawnHeight 이하여야 한다.
        // ────────────────────────────────────────────

        [Test, Category("city-spawn-refactor")]
        public void Property9_CityMetadata_SpawnYCoordinateRange()
        {
            for (int i = 0; i < Iterations; i++)
            {
                // 드론 수 랜덤 결정
                int evaderCount = RandInt(1, 3);
                int pursuerCount = RandInt(1, 2);
                int totalDrones = evaderCount + pursuerCount;
                int requiredPositions = totalDrones + 1;

                // 드론 오브젝트 생성
                ClearDrones();
                CreateDrones(evaderCount, pursuerCount);

                // 높이 범위 랜덤 설정
                float minSpawnHeight = Rand(5f, 15f);
                float maxSpawnHeight = Rand(minSpawnHeight + 5f, minSpawnHeight + 50f);
                float minSeparation = Rand(1f, 3f);

                // 다양한 elevation을 가진 후보 노드를 포함하는 CityMetadata 생성
                CityMetadata metadata = GenerateCityMetadataForSpawn(requiredPositions + 5, minSeparation);
                _api.SetCityMetadata(metadata);

                // EpisodeSpawnCoordinator 설정
                SetPrivateField(_coordinator, "_strategy",
                    EpisodeSpawnCoordinator.SpawnStrategy.CityMetadata);
                SetPrivateField(_coordinator, "_minSeparation", minSeparation);
                SetPrivateField(_coordinator, "_minSpawnHeight", minSpawnHeight);
                SetPrivateField(_coordinator, "_maxSpawnHeight", maxSpawnHeight);
                SetPrivateField(_coordinator, "_maxRetry", 20);

                _coordinator.ComputeSpawn();

                Assert.IsTrue(_coordinator.IsComputed,
                    $"[P9 iter={i}] ComputeSpawn() 후 IsComputed가 true여야 합니다.");

                var candidates = metadata.validSpawnCandidates;

                // Evader 스폰 위치 Y좌표 검증
                for (int e = 0; e < evaderCount; e++)
                {
                    Vector3 pos = _coordinator.GetSpawnPosition(_evaderObjects[e]);
                    GraphNode? matchedNode = FindMatchingCandidate(pos, candidates);
                    Assert.IsTrue(matchedNode.HasValue,
                        $"[P9 iter={i}] Evader[{e}] 스폰 위치에 대응하는 후보 노드를 찾을 수 없습니다.");

                    float elevation = matchedNode.Value.elevation;
                    float expectedMinY = elevation + minSpawnHeight;
                    float expectedMaxY = elevation + maxSpawnHeight;

                    Assert.GreaterOrEqual(pos.y, expectedMinY - 0.01f,
                        $"[P9 iter={i}] Evader[{e}] Y좌표({pos.y:F2})가 " +
                        $"elevation({elevation:F2}) + minSpawnHeight({minSpawnHeight:F2}) = {expectedMinY:F2} 이상이어야 합니다.");
                    Assert.LessOrEqual(pos.y, expectedMaxY + 0.01f,
                        $"[P9 iter={i}] Evader[{e}] Y좌표({pos.y:F2})가 " +
                        $"elevation({elevation:F2}) + maxSpawnHeight({maxSpawnHeight:F2}) = {expectedMaxY:F2} 이하여야 합니다.");
                }

                // Pursuer 스폰 위치 Y좌표 검증
                for (int p = 0; p < pursuerCount; p++)
                {
                    Vector3 pos = _coordinator.GetSpawnPosition(_pursuerObjects[p]);
                    GraphNode? matchedNode = FindMatchingCandidate(pos, candidates);
                    Assert.IsTrue(matchedNode.HasValue,
                        $"[P9 iter={i}] Pursuer[{p}] 스폰 위치에 대응하는 후보 노드를 찾을 수 없습니다.");

                    float elevation = matchedNode.Value.elevation;
                    float expectedMinY = elevation + minSpawnHeight;
                    float expectedMaxY = elevation + maxSpawnHeight;

                    Assert.GreaterOrEqual(pos.y, expectedMinY - 0.01f,
                        $"[P9 iter={i}] Pursuer[{p}] Y좌표({pos.y:F2})가 " +
                        $"elevation({elevation:F2}) + minSpawnHeight({minSpawnHeight:F2}) = {expectedMinY:F2} 이상이어야 합니다.");
                    Assert.LessOrEqual(pos.y, expectedMaxY + 0.01f,
                        $"[P9 iter={i}] Pursuer[{p}] Y좌표({pos.y:F2})가 " +
                        $"elevation({elevation:F2}) + maxSpawnHeight({maxSpawnHeight:F2}) = {expectedMaxY:F2} 이하여야 합니다.");
                }

                // Goal 스폰 위치 Y좌표 검증
                Vector3 goalPos = _coordinator.GetGoalPosition();
                GraphNode? goalNode = FindMatchingCandidate(goalPos, candidates);
                Assert.IsTrue(goalNode.HasValue,
                    $"[P9 iter={i}] Goal 스폰 위치에 대응하는 후보 노드를 찾을 수 없습니다.");

                float goalElevation = goalNode.Value.elevation;
                float goalExpectedMinY = goalElevation + minSpawnHeight;
                float goalExpectedMaxY = goalElevation + maxSpawnHeight;

                Assert.GreaterOrEqual(goalPos.y, goalExpectedMinY - 0.01f,
                    $"[P9 iter={i}] Goal Y좌표({goalPos.y:F2})가 " +
                    $"elevation({goalElevation:F2}) + minSpawnHeight({minSpawnHeight:F2}) = {goalExpectedMinY:F2} 이상이어야 합니다.");
                Assert.LessOrEqual(goalPos.y, goalExpectedMaxY + 0.01f,
                    $"[P9 iter={i}] Goal Y좌표({goalPos.y:F2})가 " +
                    $"elevation({goalElevation:F2}) + maxSpawnHeight({maxSpawnHeight:F2}) = {goalExpectedMaxY:F2} 이하여야 합니다.");
            }
        }
    }
}
