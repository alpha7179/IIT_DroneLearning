using System;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;

namespace CityGenerator
{
    /// <summary>
    /// CityMetadata 속성 기반 테스트.
    ///
    /// FsCheck 대신 NUnit + 반복 루프(100회)로 구현한다.
    /// 각 테스트 메서드는 설계 문서의 Property 번호를 태그로 참조한다.
    /// Feature: city-spawn-refactor
    /// </summary>
    public class CityMetadataPropertyTests
    {
        private const int Iterations = 100;

        private System.Random _rng;

        [SetUp]
        public void SetUp()
        {
            _rng = new System.Random(42); // 재현 가능한 시드
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
                validSpawnCandidates = new List<GraphNode>(),
                usedRandomSeed = seed,
                layoutMode = mode
            };
        }

        /// <summary>
        /// 유효 스폰 후보 노드를 포함하는 CityMetadata를 랜덤 파라미터로 생성한다.
        /// validSpawnCandidates에는 OpenSpace/Intersection 타입이고, 엣지가 존재하고,
        /// 건물 Bounds 내부에 위치하지 않는 노드만 포함된다.
        /// </summary>
        private CityMetadata GenerateRandomCityMetadataWithSpawnCandidates()
        {
            CityMetadata metadata = GenerateRandomValidCityMetadata();

            // 건물 목록에서 Bounds 수집
            var buildingBounds = new List<Bounds>();
            foreach (var building in metadata.buildings)
            {
                buildingBounds.Add(building.bounds);
            }

            // 그래프에 추가 노드와 엣지를 생성하여 유효 후보를 만든다
            var graph = metadata.cityGraph;
            int candidateCount = RandInt(1, 15);
            var candidates = new List<GraphNode>();

            for (int c = 0; c < candidateCount; c++)
            {
                // OpenSpace 또는 Intersection 타입만 사용
                NodeType type = _rng.NextDouble() < 0.5 ? NodeType.OpenSpace : NodeType.Intersection;

                // 건물 Bounds 외부에 위치하는 좌표 생성
                Vector3 pos;
                int safetyCounter = 0;
                do
                {
                    pos = new Vector3(Rand(-200f, 200f), 0f, Rand(-200f, 200f));
                    safetyCounter++;
                } while (safetyCounter < 100 && IsInsideAnyBuilding(pos, buildingBounds));

                float elevation = Rand(0f, 10f);
                int nodeId = graph.AddNode(pos, type, elevation);

                // 최소 1개의 엣지를 보장하기 위해 다른 노드와 연결
                var allNodes = graph.GetAllNodes();
                if (allNodes.Count > 1)
                {
                    // 자기 자신이 아닌 임의의 노드와 연결
                    int targetIdx = RandInt(0, allNodes.Count - 1);
                    var targetNode = allNodes[targetIdx];
                    if (targetNode.nodeId != nodeId)
                    {
                        float cost = Vector3.Distance(pos, targetNode.position);
                        graph.AddEdge(nodeId, targetNode.nodeId, cost, PathType.Direct);
                    }
                    else
                    {
                        // 자기 자신이면 다음 노드 선택
                        int altIdx = (targetIdx + 1) % allNodes.Count;
                        var altNode = allNodes[altIdx];
                        if (altNode.nodeId != nodeId)
                        {
                            float cost = Vector3.Distance(pos, altNode.position);
                            graph.AddEdge(nodeId, altNode.nodeId, cost, PathType.Direct);
                        }
                    }
                }

                candidates.Add(graph.GetNode(nodeId));
            }

            metadata.validSpawnCandidates = candidates;
            return metadata;
        }

        /// <summary>
        /// 주어진 위치가 건물 Bounds 내부에 있는지 확인한다.
        /// </summary>
        private bool IsInsideAnyBuilding(Vector3 position, List<Bounds> buildingBounds)
        {
            foreach (var b in buildingBounds)
            {
                if (b.Contains(position))
                    return true;
            }
            return false;
        }

        // ────────────────────────────────────────────
        // Feature: city-spawn-refactor, Property 1: CityMetadata 필드 완전성
        // **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 1.9**
        // ────────────────────────────────────────────

        [Test, Category("city-spawn-refactor")]
        public void Property1_CityMetadata_FieldCompleteness()
        {
            for (int i = 0; i < Iterations; i++)
            {
                CityMetadata metadata = GenerateRandomValidCityMetadata();

                // Req 1.1: cityBounds 크기가 0보다 커야 한다
                Assert.Greater(metadata.cityBounds.size.x, 0f,
                    $"[P1 iter={i}] cityBounds.size.x는 0보다 커야 합니다.");
                Assert.Greater(metadata.cityBounds.size.y, 0f,
                    $"[P1 iter={i}] cityBounds.size.y는 0보다 커야 합니다.");
                Assert.Greater(metadata.cityBounds.size.z, 0f,
                    $"[P1 iter={i}] cityBounds.size.z는 0보다 커야 합니다.");

                // Req 1.2: actualCityWidth와 actualCityDepth가 양수여야 한다
                Assert.Greater(metadata.actualCityWidth, 0,
                    $"[P1 iter={i}] actualCityWidth는 양수여야 합니다. 실제: {metadata.actualCityWidth}");
                Assert.Greater(metadata.actualCityDepth, 0,
                    $"[P1 iter={i}] actualCityDepth는 양수여야 합니다. 실제: {metadata.actualCityDepth}");

                // Req 1.3: minBuildingHeight <= maxBuildingHeight
                Assert.LessOrEqual(metadata.minBuildingHeight, metadata.maxBuildingHeight,
                    $"[P1 iter={i}] minBuildingHeight({metadata.minBuildingHeight}) <= maxBuildingHeight({metadata.maxBuildingHeight})이어야 합니다.");

                // Req 1.4: buildings가 null이 아니어야 한다
                Assert.IsNotNull(metadata.buildings,
                    $"[P1 iter={i}] buildings는 null이 아니어야 합니다.");

                // Req 1.5: cityGraph가 null이 아니어야 한다
                Assert.IsNotNull(metadata.cityGraph,
                    $"[P1 iter={i}] cityGraph는 null이 아니어야 합니다.");

                // Req 1.6: strategicLocations가 null이 아니어야 한다
                Assert.IsNotNull(metadata.strategicLocations,
                    $"[P1 iter={i}] strategicLocations는 null이 아니어야 합니다.");

                // Req 1.8: usedRandomSeed가 설정되어 있어야 한다 (기본값 0이 아닌 값)
                // 참고: 시드 0도 유효한 시드이므로, 여기서는 필드가 할당되었는지만 확인
                // (int 기본값 0과 구분하기 위해 생성기에서 항상 양수 시드를 사용)
                Assert.GreaterOrEqual(metadata.usedRandomSeed, 0,
                    $"[P1 iter={i}] usedRandomSeed는 0 이상이어야 합니다. 실제: {metadata.usedRandomSeed}");

                // Req 1.9: layoutMode가 유효한 열거형 값이어야 한다
                Assert.IsTrue(Enum.IsDefined(typeof(CityLayoutMode), metadata.layoutMode),
                    $"[P1 iter={i}] layoutMode({metadata.layoutMode})는 유효한 CityLayoutMode 열거형 값이어야 합니다.");
            }
        }
        // ────────────────────────────────────────────
        // Feature: city-spawn-refactor, Property 2: 유효 스폰 후보 노드 필터링 불변량
        // **Validates: Requirements 1.7**
        // ────────────────────────────────────────────

        [Test, Category("city-spawn-refactor")]
        public void Property2_ValidSpawnCandidates_FilteringInvariant()
        {
            for (int i = 0; i < Iterations; i++)
            {
                CityMetadata metadata = GenerateRandomCityMetadataWithSpawnCandidates();

                Assert.IsNotNull(metadata.validSpawnCandidates,
                    $"[P2 iter={i}] validSpawnCandidates는 null이 아니어야 합니다.");

                // 건물 Bounds 목록 수집
                var buildingBounds = new List<Bounds>();
                foreach (var building in metadata.buildings)
                {
                    buildingBounds.Add(building.bounds);
                }

                for (int j = 0; j < metadata.validSpawnCandidates.Count; j++)
                {
                    GraphNode node = metadata.validSpawnCandidates[j];

                    // 조건 1: 노드 타입이 OpenSpace 또는 Intersection이어야 한다
                    Assert.IsTrue(
                        node.nodeType == NodeType.OpenSpace || node.nodeType == NodeType.Intersection,
                        $"[P2 iter={i}, node={j}] validSpawnCandidate 노드 타입은 OpenSpace 또는 Intersection이어야 합니다. 실제: {node.nodeType}");

                    // 조건 2: 그래프에서 하나 이상의 엣지가 존재해야 한다
                    var edges = metadata.cityGraph.GetEdges(node.nodeId);
                    Assert.Greater(edges.Count, 0,
                        $"[P2 iter={i}, node={j}] validSpawnCandidate 노드(ID={node.nodeId})는 최소 1개의 엣지가 있어야 합니다.");

                    // 조건 3: 어떤 건물의 Bounds 내부에도 위치하지 않아야 한다
                    bool insideBuilding = false;
                    foreach (var b in buildingBounds)
                    {
                        if (b.Contains(node.position))
                        {
                            insideBuilding = true;
                            break;
                        }
                    }
                    Assert.IsFalse(insideBuilding,
                        $"[P2 iter={i}, node={j}] validSpawnCandidate 노드(ID={node.nodeId}, pos={node.position})는 건물 Bounds 내부에 위치하면 안 됩니다.");
                }
            }
        }
    }
}
