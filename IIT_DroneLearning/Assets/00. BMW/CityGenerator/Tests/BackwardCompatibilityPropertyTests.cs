using System;
using System.Collections.Generic;
using System.Reflection;
using NUnit.Framework;
using UnityEngine;

namespace CityGenerator
{
    /// <summary>
    /// 하위 호환성 및 이벤트 연동 속성 기반 테스트.
    ///
    /// FsCheck 대신 NUnit + 반복 루프로 구현한다.
    /// 각 테스트 메서드는 설계 문서의 Property 번호를 태그로 참조한다.
    /// Feature: city-spawn-refactor
    ///
    /// 테스트 환경 구성:
    ///   - "Evader" / "Pursuer" 태그가 지정된 GameObject 생성
    ///   - CityDataAPI 인스턴스 생성 및 CityMetadata/SpawnConfiguration 등록
    ///   - SpawnCenter 인스턴스 생성
    ///   - EpisodeSpawnCoordinator 인스턴스 생성 및 전략 설정
    ///   - Goal 인스턴스 생성
    /// </summary>
    public class BackwardCompatibilityPropertyTests
    {
        private const int HeavyIterations = 50;
        private const int LightIterations = 100;

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
        // 헬퍼: 태그된 드론 GameObject 생성/정리
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
        // 헬퍼: SpawnCenter SpawnRange 설정 (Reflection)
        // ────────────────────────────────────────────

        private void SetSpawnCenterRange(float minY, float maxY, float radius, float width, float depth)
        {
            var range = new SpawnCenter.SpawnRange
            {
                MinY = minY,
                MaxY = maxY,
                Radius = radius,
                Width = width,
                Depth = depth
            };
            SetPrivateField(_spawnCenter, "_sharedRange", range);
            SetPrivateField(_spawnCenter, "_goalRange", range);
            SetPrivateField(_spawnCenter, "_pursuerRange", range);
            SetPrivateField(_spawnCenter, "_evaderRange", range);
        }

        // ────────────────────────────────────────────
        // 헬퍼: 유효 스폰 후보 노드를 포함하는 CityMetadata 생성
        // ────────────────────────────────────────────

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

            var buildings = new List<Building>();
            var graph = new CityGraph();
            var candidates = new List<GraphNode>();

            float spacing = Mathf.Max(minSeparation * 2f, 20f);
            int gridSize = Mathf.CeilToInt(Mathf.Sqrt(requiredCandidates + 5));

            for (int gx = 0; gx < gridSize; gx++)
            {
                for (int gz = 0; gz < gridSize; gz++)
                {
                    if (candidates.Count >= requiredCandidates + 5) break;

                    NodeType type = _rng.NextDouble() < 0.5 ? NodeType.OpenSpace : NodeType.Intersection;
                    float elevation = Rand(0f, 20f);
                    float x = (gx - gridSize / 2f) * spacing + Rand(-2f, 2f);
                    float z = (gz - gridSize / 2f) * spacing + Rand(-2f, 2f);
                    Vector3 pos = new Vector3(x, 0f, z);

                    int nodeId = graph.AddNode(pos, type, elevation);

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

        // ────────────────────────────────────────────
        // Feature: city-spawn-refactor, Property 11: SpawnCenterRandom 전략 하위 호환
        // **Validates: Requirements 4.4, 7.1**
        //
        // For any SpawnCenter가 설정된 상태에서 SpawnStrategy가 SpawnCenterRandom으로
        // 설정된 ComputeSpawn() 호출 후, 모든 스폰 위치가 해당 SpawnRange 범위 내에 있어야 한다.
        // ────────────────────────────────────────────

        [Test, Category("city-spawn-refactor")]
        public void Property11_SpawnCenterRandom_BackwardCompatibility()
        {
            for (int i = 0; i < HeavyIterations; i++)
            {
                ClearDrones();
                CreateDrones(1, 1);

                // 랜덤 SpawnRange 설정
                float minY = Rand(0f, 10f);
                float maxY = Rand(minY + 1f, 50f);
                float radius = Rand(5f, 100f);
                float width = Rand(5f, 100f);
                float depth = Rand(5f, 100f);
                SetSpawnCenterRange(minY, maxY, radius, width, depth);

                // SpawnCenter 위치를 원점 근처에 설정
                _spawnCenter.transform.position = new Vector3(
                    Rand(-10f, 10f), Rand(-10f, 10f), Rand(-10f, 10f));

                // SpawnCenterRandom 전략 설정
                SetPrivateField(_coordinator, "_strategy",
                    EpisodeSpawnCoordinator.SpawnStrategy.SpawnCenterRandom);
                SetPrivateField(_coordinator, "_minSeparation", 1f);
                SetPrivateField(_coordinator, "_maxRetry", 20);
                _spawnCenter.AutoSyncFromCity = false;

                _coordinator.ComputeSpawn();

                Assert.IsTrue(_coordinator.IsComputed,
                    $"[P11 iter={i}] ComputeSpawn() 후 IsComputed가 true여야 합니다.");

                Vector3 scCenter = _spawnCenter.GetCenter();
                SpawnCenter.SpawnRange evaderRange = _spawnCenter.GetEvaderSpawnRange();
                SpawnCenter.SpawnRange pursuerRange = _spawnCenter.GetPursuerSpawnRange();

                // Evader 스폰 위치 검증: SpawnRange 범위 내
                Vector3 evaderPos = _coordinator.GetSpawnPosition(_evaderObjects[0]);
                AssertPositionInRange(evaderPos, scCenter, evaderRange, $"[P11 iter={i}] Evader");

                // Pursuer 스폰 위치 검증: SpawnRange 범위 내
                Vector3 pursuerPos = _coordinator.GetSpawnPosition(_pursuerObjects[0]);
                AssertPositionInRange(pursuerPos, scCenter, pursuerRange, $"[P11 iter={i}] Pursuer");
            }
        }

        /// <summary>
        /// 스폰 위치가 SpawnRange 범위 내에 있는지 검증한다.
        /// SpawnCenter는 RangeMode에 따라 Radius 또는 Rectangle 범위를 사용한다.
        /// 여기서는 Rectangle 범위 기준으로 검증한다 (더 넓은 범위).
        /// </summary>
        private void AssertPositionInRange(Vector3 pos, Vector3 center, SpawnCenter.SpawnRange range, string label)
        {
            // Y좌표 범위 검증
            Assert.GreaterOrEqual(pos.y, range.MinY + center.y - 0.01f,
                $"{label} Y좌표({pos.y:F2})가 MinY({range.MinY + center.y:F2}) 이상이어야 합니다.");
            Assert.LessOrEqual(pos.y, range.MaxY + center.y + 0.01f,
                $"{label} Y좌표({pos.y:F2})가 MaxY({range.MaxY + center.y:F2}) 이하여야 합니다.");

            // XZ 범위 검증: Radius 모드에서는 원형 범위, Rectangle 모드에서는 직사각형 범위
            // SpawnCenter.GetRandomPosition()은 RangeMode에 따라 다르게 동작하므로
            // 가장 넓은 범위(Radius)로 검증
            float dx = Mathf.Abs(pos.x - center.x);
            float dz = Mathf.Abs(pos.z - center.z);
            float distXZ = Mathf.Sqrt(dx * dx + dz * dz);

            // Radius 모드: 원형 범위 내
            Assert.LessOrEqual(distXZ, range.Radius + 0.01f,
                $"{label} XZ 거리({distXZ:F2})가 Radius({range.Radius:F2}) 이하여야 합니다.");
        }

        // ────────────────────────────────────────────
        // Feature: city-spawn-refactor, Property 12: CityDataAPI 전략 하위 호환
        // **Validates: Requirements 4.3, 7.2**
        //
        // For any 유효한 SpawnConfiguration이 등록된 상태에서 SpawnStrategy가 CityDataAPI로
        // 설정된 단일 드론 ComputeSpawn() 호출 후, Evader 스폰 위치가
        // SpawnConfiguration.evaderSpawnPosition과 일치하고, Pursuer 스폰 위치가
        // SpawnConfiguration.pursuerSpawnPosition과 일치해야 한다.
        // ────────────────────────────────────────────

        [Test, Category("city-spawn-refactor")]
        public void Property12_CityDataAPI_BackwardCompatibility()
        {
            for (int i = 0; i < HeavyIterations; i++)
            {
                ClearDrones();
                CreateDrones(1, 1);

                // 랜덤 SpawnConfiguration 생성
                Vector3 evaderPos = new Vector3(Rand(-50f, 50f), Rand(5f, 30f), Rand(-50f, 50f));
                Vector3 pursuerPos = new Vector3(Rand(-50f, 50f), Rand(5f, 30f), Rand(-50f, 50f));
                Vector3 targetPos = new Vector3(Rand(-50f, 50f), Rand(5f, 30f), Rand(-50f, 50f));

                var config = new SpawnConfiguration
                {
                    evaderSpawnPosition = evaderPos,
                    pursuerSpawnPosition = pursuerPos,
                    targetPosition = targetPos,
                    evaderSpawnNodeId = RandInt(0, 100),
                    pursuerSpawnNodeId = RandInt(0, 100),
                    targetNodeId = RandInt(0, 100),
                    achievedMinSeparation = Rand(1f, 10f),
                    isValid = true
                };

                _api.SetSpawnConfiguration(config);

                // CityDataAPI 전략 설정
                SetPrivateField(_coordinator, "_strategy",
                    EpisodeSpawnCoordinator.SpawnStrategy.CityDataAPI);
                SetPrivateField(_coordinator, "_minSeparation", 1f);
                SetPrivateField(_coordinator, "_maxRetry", 20);
                _spawnCenter.AutoSyncFromCity = false;

                _coordinator.ComputeSpawn();

                Assert.IsTrue(_coordinator.IsComputed,
                    $"[P12 iter={i}] ComputeSpawn() 후 IsComputed가 true여야 합니다.");

                // Evader 스폰 위치가 SpawnConfiguration과 일치하는지 검증
                Vector3 actualEvader = _coordinator.GetSpawnPosition(_evaderObjects[0]);
                Assert.AreEqual(evaderPos.x, actualEvader.x, 0.01f,
                    $"[P12 iter={i}] Evader X({actualEvader.x:F2})가 config.evaderSpawnPosition.x({evaderPos.x:F2})와 일치해야 합니다.");
                Assert.AreEqual(evaderPos.y, actualEvader.y, 0.01f,
                    $"[P12 iter={i}] Evader Y({actualEvader.y:F2})가 config.evaderSpawnPosition.y({evaderPos.y:F2})와 일치해야 합니다.");
                Assert.AreEqual(evaderPos.z, actualEvader.z, 0.01f,
                    $"[P12 iter={i}] Evader Z({actualEvader.z:F2})가 config.evaderSpawnPosition.z({evaderPos.z:F2})와 일치해야 합니다.");

                // Pursuer 스폰 위치가 SpawnConfiguration과 일치하는지 검증
                Vector3 actualPursuer = _coordinator.GetSpawnPosition(_pursuerObjects[0]);
                Assert.AreEqual(pursuerPos.x, actualPursuer.x, 0.01f,
                    $"[P12 iter={i}] Pursuer X({actualPursuer.x:F2})가 config.pursuerSpawnPosition.x({pursuerPos.x:F2})와 일치해야 합니다.");
                Assert.AreEqual(pursuerPos.y, actualPursuer.y, 0.01f,
                    $"[P12 iter={i}] Pursuer Y({actualPursuer.y:F2})가 config.pursuerSpawnPosition.y({pursuerPos.y:F2})와 일치해야 합니다.");
                Assert.AreEqual(pursuerPos.z, actualPursuer.z, 0.01f,
                    $"[P12 iter={i}] Pursuer Z({actualPursuer.z:F2})가 config.pursuerSpawnPosition.z({pursuerPos.z:F2})와 일치해야 합니다.");
            }
        }

        // ────────────────────────────────────────────
        // Feature: city-spawn-refactor, Property 13: Fallback 전략 하위 호환
        // **Validates: Requirements 7.3**
        //
        // For any SpawnStrategy가 Fallback으로 설정된 ComputeSpawn() 호출 후,
        // 모든 스폰 위치의 XZ 좌표 절대값이 _fallbackRange 이하이고,
        // Y좌표가 _fallbackHeight와 일치해야 한다.
        // ────────────────────────────────────────────

        [Test, Category("city-spawn-refactor")]
        public void Property13_Fallback_BackwardCompatibility()
        {
            for (int i = 0; i < HeavyIterations; i++)
            {
                ClearDrones();
                int evaderCount = RandInt(1, 3);
                int pursuerCount = RandInt(1, 2);
                CreateDrones(evaderCount, pursuerCount);

                // 랜덤 Fallback 파라미터 설정
                float fallbackRange = Rand(5f, 100f);
                float fallbackHeight = Rand(1f, 50f);

                SetPrivateField(_coordinator, "_strategy",
                    EpisodeSpawnCoordinator.SpawnStrategy.Fallback);
                SetPrivateField(_coordinator, "_fallbackRange", fallbackRange);
                SetPrivateField(_coordinator, "_fallbackHeight", fallbackHeight);
                SetPrivateField(_coordinator, "_minSeparation", 0.5f);
                SetPrivateField(_coordinator, "_maxRetry", 20);
                _spawnCenter.AutoSyncFromCity = false;

                _coordinator.ComputeSpawn();

                Assert.IsTrue(_coordinator.IsComputed,
                    $"[P13 iter={i}] ComputeSpawn() 후 IsComputed가 true여야 합니다.");

                // Evader 스폰 위치 검증
                for (int e = 0; e < evaderCount; e++)
                {
                    Vector3 pos = _coordinator.GetSpawnPosition(_evaderObjects[e]);

                    Assert.LessOrEqual(Mathf.Abs(pos.x), fallbackRange + 0.01f,
                        $"[P13 iter={i}] Evader[{e}] |X|({Mathf.Abs(pos.x):F2})가 " +
                        $"fallbackRange({fallbackRange:F2}) 이하여야 합니다.");
                    Assert.LessOrEqual(Mathf.Abs(pos.z), fallbackRange + 0.01f,
                        $"[P13 iter={i}] Evader[{e}] |Z|({Mathf.Abs(pos.z):F2})가 " +
                        $"fallbackRange({fallbackRange:F2}) 이하여야 합니다.");
                    Assert.AreEqual(fallbackHeight, pos.y, 0.01f,
                        $"[P13 iter={i}] Evader[{e}] Y({pos.y:F2})가 " +
                        $"fallbackHeight({fallbackHeight:F2})와 일치해야 합니다.");
                }

                // Pursuer 스폰 위치 검증
                for (int p = 0; p < pursuerCount; p++)
                {
                    Vector3 pos = _coordinator.GetSpawnPosition(_pursuerObjects[p]);

                    Assert.LessOrEqual(Mathf.Abs(pos.x), fallbackRange + 0.01f,
                        $"[P13 iter={i}] Pursuer[{p}] |X|({Mathf.Abs(pos.x):F2})가 " +
                        $"fallbackRange({fallbackRange:F2}) 이하여야 합니다.");
                    Assert.LessOrEqual(Mathf.Abs(pos.z), fallbackRange + 0.01f,
                        $"[P13 iter={i}] Pursuer[{p}] |Z|({Mathf.Abs(pos.z):F2})가 " +
                        $"fallbackRange({fallbackRange:F2}) 이하여야 합니다.");
                    Assert.AreEqual(fallbackHeight, pos.y, 0.01f,
                        $"[P13 iter={i}] Pursuer[{p}] Y({pos.y:F2})가 " +
                        $"fallbackHeight({fallbackHeight:F2})와 일치해야 합니다.");
                }
            }
        }

        // ────────────────────────────────────────────
        // Feature: city-spawn-refactor, Property 14: SpawnConfiguration 라운드트립
        // **Validates: Requirements 7.4**
        //
        // For any 유효한 SpawnConfiguration 객체에 대해,
        // CityDataAPI.SetSpawnConfiguration(config) 호출 후
        // GetSpawnConfiguration()이 동일한 값을 반환하고,
        // HasSpawnConfiguration()이 config.isValid와 동일한 값을 반환해야 한다.
        // ────────────────────────────────────────────

        [Test, Category("city-spawn-refactor")]
        public void Property14_SpawnConfiguration_Roundtrip()
        {
            for (int i = 0; i < LightIterations; i++)
            {
                // 랜덤 SpawnConfiguration 생성
                bool isValid = _rng.NextDouble() > 0.3; // 70% 확률로 유효
                Vector3 evaderPos = new Vector3(Rand(-100f, 100f), Rand(0f, 50f), Rand(-100f, 100f));
                Vector3 pursuerPos = new Vector3(Rand(-100f, 100f), Rand(0f, 50f), Rand(-100f, 100f));
                Vector3 targetPos = new Vector3(Rand(-100f, 100f), Rand(0f, 50f), Rand(-100f, 100f));
                int evaderNodeId = RandInt(0, 1000);
                int pursuerNodeId = RandInt(0, 1000);
                int targetNodeId = RandInt(0, 1000);
                float achievedSep = Rand(0f, 50f);

                var config = new SpawnConfiguration
                {
                    evaderSpawnPosition = evaderPos,
                    pursuerSpawnPosition = pursuerPos,
                    targetPosition = targetPos,
                    evaderSpawnNodeId = evaderNodeId,
                    pursuerSpawnNodeId = pursuerNodeId,
                    targetNodeId = targetNodeId,
                    achievedMinSeparation = achievedSep,
                    isValid = isValid
                };

                // Set → Get 라운드트립
                _api.SetSpawnConfiguration(config);
                SpawnConfiguration retrieved = _api.GetSpawnConfiguration();

                // 위치 값 일치 검증
                Assert.AreEqual(evaderPos, retrieved.evaderSpawnPosition,
                    $"[P14 iter={i}] evaderSpawnPosition이 일치해야 합니다.");
                Assert.AreEqual(pursuerPos, retrieved.pursuerSpawnPosition,
                    $"[P14 iter={i}] pursuerSpawnPosition이 일치해야 합니다.");
                Assert.AreEqual(targetPos, retrieved.targetPosition,
                    $"[P14 iter={i}] targetPosition이 일치해야 합니다.");

                // 노드 ID 일치 검증
                Assert.AreEqual(evaderNodeId, retrieved.evaderSpawnNodeId,
                    $"[P14 iter={i}] evaderSpawnNodeId가 일치해야 합니다.");
                Assert.AreEqual(pursuerNodeId, retrieved.pursuerSpawnNodeId,
                    $"[P14 iter={i}] pursuerSpawnNodeId가 일치해야 합니다.");
                Assert.AreEqual(targetNodeId, retrieved.targetNodeId,
                    $"[P14 iter={i}] targetNodeId가 일치해야 합니다.");

                // 기타 필드 일치 검증
                Assert.AreEqual(achievedSep, retrieved.achievedMinSeparation, 0.001f,
                    $"[P14 iter={i}] achievedMinSeparation이 일치해야 합니다.");
                Assert.AreEqual(isValid, retrieved.isValid,
                    $"[P14 iter={i}] isValid가 일치해야 합니다.");

                // HasSpawnConfiguration()이 config.isValid와 동일한 값을 반환
                Assert.AreEqual(isValid, _api.HasSpawnConfiguration(),
                    $"[P14 iter={i}] HasSpawnConfiguration()이 config.isValid({isValid})와 일치해야 합니다.");
            }
        }

        // ────────────────────────────────────────────
        // Feature: city-spawn-refactor, Property 15: Goal 위치 적용
        // **Validates: Requirements 6.1, 8.6**
        //
        // For any 전략으로 ComputeSpawn() 호출 후, Goal.Current가 존재하면
        // Goal.Current.GetPosition()이 EpisodeSpawnCoordinator.GetGoalPosition()과 일치해야 한다.
        // ────────────────────────────────────────────

        [Test, Category("city-spawn-refactor")]
        public void Property15_GoalPosition_Applied()
        {
            // 여러 전략을 순회하며 검증
            var strategies = new[]
            {
                EpisodeSpawnCoordinator.SpawnStrategy.SpawnCenterRandom,
                EpisodeSpawnCoordinator.SpawnStrategy.Fallback,
                EpisodeSpawnCoordinator.SpawnStrategy.CityMetadata,
            };

            for (int i = 0; i < HeavyIterations; i++)
            {
                ClearDrones();
                CreateDrones(1, 1);

                // 전략 순환 선택
                var strategy = strategies[i % strategies.Length];

                SetPrivateField(_coordinator, "_strategy", strategy);
                SetPrivateField(_coordinator, "_minSeparation", 1f);
                SetPrivateField(_coordinator, "_maxRetry", 20);
                SetPrivateField(_coordinator, "_fallbackRange", 50f);
                SetPrivateField(_coordinator, "_fallbackHeight", 10f);
                SetPrivateField(_coordinator, "_minSpawnHeight", 8f);
                SetPrivateField(_coordinator, "_maxSpawnHeight", 50f);
                _spawnCenter.AutoSyncFromCity = false;

                // CityMetadata 전략인 경우 메타데이터 등록
                if (strategy == EpisodeSpawnCoordinator.SpawnStrategy.CityMetadata)
                {
                    CityMetadata metadata = GenerateCityMetadataForSpawn(10, 1f);
                    _api.SetCityMetadata(metadata);
                }

                _coordinator.ComputeSpawn();

                Assert.IsTrue(_coordinator.IsComputed,
                    $"[P15 iter={i}] ComputeSpawn() 후 IsComputed가 true여야 합니다.");

                // Goal.Current가 존재하면 위치 일치 검증
                Assert.IsNotNull(Goal.Current,
                    $"[P15 iter={i}] Goal.Current가 null이 아니어야 합니다.");

                Vector3 goalFromCoordinator = _coordinator.GetGoalPosition();
                Vector3 goalFromGoal = Goal.Current.GetPosition();

                Assert.AreEqual(goalFromCoordinator.x, goalFromGoal.x, 0.01f,
                    $"[P15 iter={i} strategy={strategy}] Goal.GetPosition().x({goalFromGoal.x:F2})가 " +
                    $"GetGoalPosition().x({goalFromCoordinator.x:F2})와 일치해야 합니다.");
                Assert.AreEqual(goalFromCoordinator.y, goalFromGoal.y, 0.01f,
                    $"[P15 iter={i} strategy={strategy}] Goal.GetPosition().y({goalFromGoal.y:F2})가 " +
                    $"GetGoalPosition().y({goalFromCoordinator.y:F2})와 일치해야 합니다.");
                Assert.AreEqual(goalFromCoordinator.z, goalFromGoal.z, 0.01f,
                    $"[P15 iter={i} strategy={strategy}] Goal.GetPosition().z({goalFromGoal.z:F2})가 " +
                    $"GetGoalPosition().z({goalFromCoordinator.z:F2})와 일치해야 합니다.");
            }
        }

        // ────────────────────────────────────────────
        // Feature: city-spawn-refactor, Property 17: OnSpawnComputed 이벤트 발생
        // **Validates: Requirements 6.1**
        //
        // For any ComputeSpawn() 호출 후, OnSpawnComputed 이벤트가 정확히 한 번 발생해야 한다.
        // ────────────────────────────────────────────

        [Test, Category("city-spawn-refactor")]
        public void Property17_OnSpawnComputed_EventFiredOnce()
        {
            var strategies = new[]
            {
                EpisodeSpawnCoordinator.SpawnStrategy.SpawnCenterRandom,
                EpisodeSpawnCoordinator.SpawnStrategy.Fallback,
                EpisodeSpawnCoordinator.SpawnStrategy.CityMetadata,
                EpisodeSpawnCoordinator.SpawnStrategy.CityDataAPI,
            };

            for (int i = 0; i < LightIterations; i++)
            {
                ClearDrones();
                CreateDrones(1, 1);

                var strategy = strategies[i % strategies.Length];

                SetPrivateField(_coordinator, "_strategy", strategy);
                SetPrivateField(_coordinator, "_minSeparation", 1f);
                SetPrivateField(_coordinator, "_maxRetry", 20);
                SetPrivateField(_coordinator, "_fallbackRange", 50f);
                SetPrivateField(_coordinator, "_fallbackHeight", 10f);
                SetPrivateField(_coordinator, "_minSpawnHeight", 8f);
                SetPrivateField(_coordinator, "_maxSpawnHeight", 50f);
                _spawnCenter.AutoSyncFromCity = false;

                // CityMetadata 전략인 경우 메타데이터 등록
                if (strategy == EpisodeSpawnCoordinator.SpawnStrategy.CityMetadata)
                {
                    CityMetadata metadata = GenerateCityMetadataForSpawn(10, 1f);
                    _api.SetCityMetadata(metadata);
                }

                // CityDataAPI 전략인 경우 SpawnConfiguration 등록
                if (strategy == EpisodeSpawnCoordinator.SpawnStrategy.CityDataAPI)
                {
                    var config = new SpawnConfiguration
                    {
                        evaderSpawnPosition = new Vector3(10f, 10f, 10f),
                        pursuerSpawnPosition = new Vector3(-10f, 10f, -10f),
                        targetPosition = new Vector3(0f, 10f, 0f),
                        isValid = true
                    };
                    _api.SetSpawnConfiguration(config);
                }

                // 이벤트 카운터 설정
                int eventCount = 0;
                Action handler = () => eventCount++;
                _coordinator.OnSpawnComputed += handler;

                try
                {
                    _coordinator.ComputeSpawn();

                    Assert.AreEqual(1, eventCount,
                        $"[P17 iter={i} strategy={strategy}] OnSpawnComputed 이벤트가 " +
                        $"정확히 1번 발생해야 합니다. 실제: {eventCount}번");
                }
                finally
                {
                    _coordinator.OnSpawnComputed -= handler;
                }
            }
        }
    }
}
