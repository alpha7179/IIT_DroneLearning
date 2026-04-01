using System;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;

namespace CityGenerator
{
    /// <summary>
    /// SpawnCenter 도시 메타데이터 동기화 속성 기반 테스트.
    ///
    /// FsCheck 대신 NUnit + 반복 루프(100회)로 구현한다.
    /// Feature: city-spawn-refactor
    /// </summary>
    public class SpawnCenterPropertyTests
    {
        private const int Iterations = 100;

        private System.Random _rng;

        // 씬 오브젝트
        private GameObject _apiGameObject;
        private CityDataAPI _api;
        private GameObject _spawnCenterObject;
        private SpawnCenter _spawnCenter;

        [SetUp]
        public void SetUp()
        {
            _rng = new System.Random(42); // 재현 가능한 시드

            // CityDataAPI 싱글톤 생성
            _apiGameObject = new GameObject("TestCityDataAPI");
            _api = _apiGameObject.AddComponent<CityDataAPI>();

            // SpawnCenter 생성
            _spawnCenterObject = new GameObject("TestSpawnCenter");
            _spawnCenter = _spawnCenterObject.AddComponent<SpawnCenter>();
        }

        [TearDown]
        public void TearDown()
        {
            if (_spawnCenterObject != null)
                UnityEngine.Object.DestroyImmediate(_spawnCenterObject);
            if (_apiGameObject != null)
                UnityEngine.Object.DestroyImmediate(_apiGameObject);
        }

        // ────────────────────────────────────────────
        // 헬퍼
        // ────────────────────────────────────────────

        /// <summary>범위 [min, max] 내 임의 float 반환</summary>
        private float Rand(float min, float max) =>
            (float)(_rng.NextDouble() * (max - min) + min);

        /// <summary>범위 [min, max] 내 임의 int 반환</summary>
        private int RandInt(int min, int max) =>
            _rng.Next(min, max + 1);

        /// <summary>
        /// 유효한 CityMetadata를 랜덤 파라미터로 생성한다.
        /// </summary>
        private CityMetadata GenerateRandomValidCityMetadata()
        {
            int width = RandInt(1, 100);
            int depth = RandInt(1, 100);
            float minHeight = Rand(1f, 250f);
            float maxHeight = Rand(minHeight, 500f);
            int seed = RandInt(0, int.MaxValue);
            CityLayoutMode mode = (CityLayoutMode)RandInt(0, 2);

            // 도시 경계 생성 (양수 크기)
            float boundsWidth = width * Rand(1f, 10f);
            float boundsDepth = depth * Rand(1f, 10f);
            float boundsHeight = maxHeight + Rand(1f, 50f);
            var center = new Vector3(Rand(-100f, 100f), boundsHeight * 0.5f, Rand(-100f, 100f));
            var bounds = new Bounds(center, new Vector3(boundsWidth, boundsHeight, boundsDepth));

            // 건물 목록 생성 (0~50개)
            int buildingCount = RandInt(0, 50);
            var buildings = new List<Building>();
            for (int b = 0; b < buildingCount; b++)
            {
                var pos = new Vector3(Rand(-50f, 50f), 0f, Rand(-50f, 50f));
                float bh = Rand(minHeight, maxHeight);
                var sz = new Vector3(Rand(1f, 5f), bh, Rand(1f, 5f));
                var cell = new GridCell { x = b, z = 0, hasBuilding = true, buildingHeight = bh, worldPosition = pos };
                buildings.Add(new Building(null, pos, sz, bh, cell));
            }

            // 그래프 생성 (최소 1개 노드)
            var graph = new CityGraph();
            int nodeCount = RandInt(1, 20);
            for (int n = 0; n < nodeCount; n++)
            {
                graph.AddNode(
                    new Vector3(Rand(-50f, 50f), 0f, Rand(-50f, 50f)),
                    (NodeType)RandInt(0, 3),
                    Rand(0f, 10f));
            }

            // 전략적 위치 목록 생성
            int stratCount = RandInt(0, 10);
            var strategicLocations = new List<StrategicLocation>();
            for (int s = 0; s < stratCount; s++)
            {
                strategicLocations.Add(new StrategicLocation
                {
                    position = new Vector3(Rand(-50f, 50f), 0f, Rand(-50f, 50f)),
                    locationType = (StrategyType)RandInt(0, 4),
                    dangerScore = Rand(0f, 1f),
                    visibilityScore = Rand(0f, 1f),
                    connectedNodes = new List<int>()
                });
            }

            // 유효 스폰 후보 노드 생성
            var candidates = new List<GraphNode>();
            int candidateCount = RandInt(0, 10);
            var allNodes = graph.GetAllNodes();
            for (int c = 0; c < candidateCount && c < allNodes.Count; c++)
            {
                candidates.Add(allNodes[c]);
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
                strategicLocations = strategicLocations,
                validSpawnCandidates = candidates,
                usedRandomSeed = seed,
                layoutMode = mode
            };
        }

        // ────────────────────────────────────────────
        // Feature: city-spawn-refactor, Property 10: SpawnCenter 도시 메타데이터 동기화
        // **Validates: Requirements 5.3, 5.4, 5.5, 5.7, 8.5**
        //
        // For any 유효한 CityMetadata가 등록된 상태에서
        // SpawnCenter.SyncFromCityMetadata() 호출 후,
        // SpawnRange의 범위가 도시 경계(cityBounds)를 반영하고,
        // MinY/MaxY가 건물 높이 범위를 반영하고,
        // SpawnCenter의 Transform 위치가 cityBounds.center와 일치해야 한다.
        // ────────────────────────────────────────────

        [Test, Category("city-spawn-refactor")]
        public void Property10_SpawnCenter_SyncFromCityMetadata()
        {
            for (int i = 0; i < Iterations; i++)
            {
                CityMetadata metadata = GenerateRandomValidCityMetadata();

                // CityDataAPI에 메타데이터 등록
                _api.SetCityMetadata(metadata);

                // SyncFromCityMetadata 호출
                _spawnCenter.SyncFromCityMetadata();

                Bounds cityBounds = metadata.cityBounds;
                float expectedWidth = cityBounds.extents.x;
                float expectedDepth = cityBounds.extents.z;
                float expectedRadius = Mathf.Max(expectedWidth, expectedDepth);
                float expectedMinY = metadata.minBuildingHeight;
                float expectedMaxY = metadata.maxBuildingHeight;

                // Req 5.3: SpawnRange Width == cityBounds.extents.x
                SpawnCenter.SpawnRange goalRange = _spawnCenter.GetGoalSpawnRange();
                Assert.AreEqual(expectedWidth, goalRange.Width, 0.001f,
                    $"[P10 iter={i}] SpawnRange.Width({goalRange.Width:F3})가 " +
                    $"cityBounds.extents.x({expectedWidth:F3})와 일치해야 합니다.");

                // Req 5.3: SpawnRange Depth == cityBounds.extents.z
                Assert.AreEqual(expectedDepth, goalRange.Depth, 0.001f,
                    $"[P10 iter={i}] SpawnRange.Depth({goalRange.Depth:F3})가 " +
                    $"cityBounds.extents.z({expectedDepth:F3})와 일치해야 합니다.");

                // Req 5.3: SpawnRange Radius == max(Width, Depth)
                Assert.AreEqual(expectedRadius, goalRange.Radius, 0.001f,
                    $"[P10 iter={i}] SpawnRange.Radius({goalRange.Radius:F3})가 " +
                    $"max(Width, Depth)({expectedRadius:F3})와 일치해야 합니다.");

                // Req 5.4: SpawnRange MinY == minBuildingHeight
                Assert.AreEqual(expectedMinY, goalRange.MinY, 0.001f,
                    $"[P10 iter={i}] SpawnRange.MinY({goalRange.MinY:F3})가 " +
                    $"minBuildingHeight({expectedMinY:F3})와 일치해야 합니다.");

                // Req 5.4: SpawnRange MaxY == maxBuildingHeight
                Assert.AreEqual(expectedMaxY, goalRange.MaxY, 0.001f,
                    $"[P10 iter={i}] SpawnRange.MaxY({goalRange.MaxY:F3})가 " +
                    $"maxBuildingHeight({expectedMaxY:F3})와 일치해야 합니다.");

                // Req 5.5: transform.position == cityBounds.center
                Vector3 actualPos = _spawnCenter.transform.position;
                Assert.AreEqual(cityBounds.center.x, actualPos.x, 0.001f,
                    $"[P10 iter={i}] transform.position.x({actualPos.x:F3})가 " +
                    $"cityBounds.center.x({cityBounds.center.x:F3})와 일치해야 합니다.");
                Assert.AreEqual(cityBounds.center.y, actualPos.y, 0.001f,
                    $"[P10 iter={i}] transform.position.y({actualPos.y:F3})가 " +
                    $"cityBounds.center.y({cityBounds.center.y:F3})와 일치해야 합니다.");
                Assert.AreEqual(cityBounds.center.z, actualPos.z, 0.001f,
                    $"[P10 iter={i}] transform.position.z({actualPos.z:F3})가 " +
                    $"cityBounds.center.z({cityBounds.center.z:F3})와 일치해야 합니다.");

                // Desynchronized 모드 범위도 동기화되었는지 추가 검증
                SpawnCenter.SpawnRange pursuerRange = _spawnCenter.GetPursuerSpawnRange();
                SpawnCenter.SpawnRange evaderRange = _spawnCenter.GetEvaderSpawnRange();

                // Synchronized 모드에서는 모든 범위가 동일해야 한다
                Assert.AreEqual(goalRange.Width, pursuerRange.Width, 0.001f,
                    $"[P10 iter={i}] Pursuer SpawnRange.Width가 Goal SpawnRange.Width와 일치해야 합니다.");
                Assert.AreEqual(goalRange.Width, evaderRange.Width, 0.001f,
                    $"[P10 iter={i}] Evader SpawnRange.Width가 Goal SpawnRange.Width와 일치해야 합니다.");
            }
        }
    }
}
