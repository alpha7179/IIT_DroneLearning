using System;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;

namespace CityGenerator
{
    /// <summary>
    /// CityDataAPI 메타데이터 저장/조회 라운드트립 속성 기반 테스트.
    ///
    /// FsCheck 대신 NUnit + 반복 루프(100회)로 구현한다.
    /// Feature: city-spawn-refactor
    /// </summary>
    public class CityDataAPIPropertyTests
    {
        private const int Iterations = 100;

        private System.Random _rng;
        private GameObject _apiGameObject;
        private CityDataAPI _api;

        [SetUp]
        public void SetUp()
        {
            _rng = new System.Random(42); // 재현 가능한 시드

            // CityDataAPI는 MonoBehaviour 싱글톤이므로 GameObject에 부착하여 생성
            _apiGameObject = new GameObject("TestCityDataAPI");
            _api = _apiGameObject.AddComponent<CityDataAPI>();
        }

        [TearDown]
        public void TearDown()
        {
            if (_apiGameObject != null)
            {
                UnityEngine.Object.DestroyImmediate(_apiGameObject);
            }
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
        /// 도시 파라미터 범위: 격자 크기 1~100, 건물 높이 1~500, 건물 수 0~50.
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
        // Feature: city-spawn-refactor, Property 6: CityDataAPI 메타데이터 저장/조회 라운드트립
        // **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**
        // ────────────────────────────────────────────

        [Test, Category("city-spawn-refactor")]
        public void Property6_CityDataAPI_MetadataRoundTrip()
        {
            for (int i = 0; i < Iterations; i++)
            {
                CityMetadata metadata = GenerateRandomValidCityMetadata();

                // Req 3.1: SetCityMetadata 호출
                _api.SetCityMetadata(metadata);

                // Req 3.2: GetCityMetadata()가 동일한 객체를 반환해야 한다
                CityMetadata retrieved = _api.GetCityMetadata();
                Assert.IsNotNull(retrieved,
                    $"[P6 iter={i}] GetCityMetadata()는 null이 아니어야 합니다.");
                Assert.AreSame(metadata, retrieved,
                    $"[P6 iter={i}] GetCityMetadata()는 SetCityMetadata()에 전달한 동일한 객체를 반환해야 합니다.");

                // Req 3.3: HasCityMetadata()가 true를 반환해야 한다
                Assert.IsTrue(_api.HasCityMetadata(),
                    $"[P6 iter={i}] SetCityMetadata() 호출 후 HasCityMetadata()는 true여야 합니다.");

                // Req 3.4: GetValidSpawnCandidates()가 메타데이터의 validSpawnCandidates와 동일한 목록을 반환해야 한다
                var candidates = _api.GetValidSpawnCandidates();
                Assert.IsNotNull(candidates,
                    $"[P6 iter={i}] GetValidSpawnCandidates()는 null이 아니어야 합니다.");
                Assert.AreSame(metadata.validSpawnCandidates, candidates,
                    $"[P6 iter={i}] GetValidSpawnCandidates()는 메타데이터의 validSpawnCandidates와 동일한 참조여야 합니다.");
                Assert.AreEqual(metadata.validSpawnCandidates.Count, candidates.Count,
                    $"[P6 iter={i}] GetValidSpawnCandidates() 개수가 일치해야 합니다.");

                // Req 3.5: GetCityBounds()가 메타데이터의 cityBounds와 동일한 값을 반환해야 한다
                Bounds retrievedBounds = _api.GetCityBounds();
                Assert.AreEqual(metadata.cityBounds.center, retrievedBounds.center,
                    $"[P6 iter={i}] GetCityBounds().center가 메타데이터의 cityBounds.center와 일치해야 합니다.");
                Assert.AreEqual(metadata.cityBounds.size, retrievedBounds.size,
                    $"[P6 iter={i}] GetCityBounds().size가 메타데이터의 cityBounds.size와 일치해야 합니다.");

                // Req 3.6: GetBuildingHeightRange()가 메타데이터의 높이 범위와 동일한 값을 반환해야 한다
                var (min, max) = _api.GetBuildingHeightRange();
                Assert.AreEqual(metadata.minBuildingHeight, min, 0.0001f,
                    $"[P6 iter={i}] GetBuildingHeightRange().min({min})이 메타데이터의 minBuildingHeight({metadata.minBuildingHeight})와 일치해야 합니다.");
                Assert.AreEqual(metadata.maxBuildingHeight, max, 0.0001f,
                    $"[P6 iter={i}] GetBuildingHeightRange().max({max})이 메타데이터의 maxBuildingHeight({metadata.maxBuildingHeight})와 일치해야 합니다.");
            }
        }
    }
}
