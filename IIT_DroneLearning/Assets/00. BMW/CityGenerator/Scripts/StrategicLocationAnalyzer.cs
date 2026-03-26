using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace CityGenerator
{
    /// <summary>
    /// 전략적 위치를 분석하고 마킹하는 클래스
    /// 도시 그래프와 건물 배치를 기반으로 은폐 지점, 교차로, 막다른 골목, 개방 구역, 우회 경로를 식별합니다.
    /// </summary>
    public static class StrategicLocationAnalyzer
    {
        /// <summary>
        /// 도시 그래프와 건물 배치를 분석하여 전략적 위치를 식별합니다.
        /// </summary>
        /// <param name="graph">도시 그래프</param>
        /// <param name="buildings">건물 배열</param>
        /// <param name="spawnPoint">생성 지점 (위험도 및 가시성 계산용)</param>
        /// <returns>전략적 위치 리스트</returns>
        public static List<StrategicLocation> AnalyzeLocations(
            CityGraph graph,
            Building[] buildings,
            Vector3 spawnPoint)
        {
            if (graph == null || buildings == null)
            {
                Debug.LogWarning("Graph or buildings is null. Cannot analyze strategic locations.");
                return new List<StrategicLocation>();
            }

            List<StrategicLocation> strategicLocations = new List<StrategicLocation>();

            // 각 전략적 위치 타입별로 분석
            strategicLocations.AddRange(FindCoverPoints(graph, buildings));
            strategicLocations.AddRange(FindIntersections(graph));
            strategicLocations.AddRange(FindDeadEnds(graph));
            strategicLocations.AddRange(FindOpenAreas(graph, buildings));
            strategicLocations.AddRange(FindDetourPaths(graph));

            // 각 전략적 위치에 대해 위험도 및 가시성 점수 계산
            for (int i = 0; i < strategicLocations.Count; i++)
            {
                StrategicLocation location = strategicLocations[i];
                location.dangerScore = CalculateDangerScore(location.position, buildings, spawnPoint);
                location.visibilityScore = CalculateVisibilityScore(location.position, spawnPoint, buildings);
                strategicLocations[i] = location;
            }

            Debug.Log($"Strategic locations analyzed: {strategicLocations.Count} locations found");
            Debug.Log($"  - Cover Points: {strategicLocations.Count(l => l.locationType == StrategyType.CoverPoint)}");
            Debug.Log($"  - Intersections: {strategicLocations.Count(l => l.locationType == StrategyType.Intersection)}");
            Debug.Log($"  - Dead Ends: {strategicLocations.Count(l => l.locationType == StrategyType.DeadEnd)}");
            Debug.Log($"  - Open Areas: {strategicLocations.Count(l => l.locationType == StrategyType.OpenArea)}");
            Debug.Log($"  - Detour Paths: {strategicLocations.Count(l => l.locationType == StrategyType.DetourPath)}");

            return strategicLocations;
        }

        /// <summary>
        /// 은폐 지점을 찾습니다 (높은 건물 뒤 또는 건물 사이 좁은 공간).
        /// </summary>
        /// <param name="graph">도시 그래프</param>
        /// <param name="buildings">건물 배열</param>
        /// <returns>은폐 지점 리스트</returns>
        private static List<StrategicLocation> FindCoverPoints(CityGraph graph, Building[] buildings)
        {
            List<StrategicLocation> coverPoints = new List<StrategicLocation>();
            List<GraphNode> allNodes = graph.GetAllNodes();

            foreach (GraphNode node in allNodes)
            {
                // 건물 모서리나 골목 입구는 은폐 지점 후보
                if (node.nodeType == NodeType.BuildingCorner || node.nodeType == NodeType.AlleyEntrance)
                {
                    // 주변에 높은 건물이 있는지 확인
                    if (node.surroundingBuildingHeights != null && node.surroundingBuildingHeights.Length > 0)
                    {
                        float avgHeight = node.surroundingBuildingHeights.Average();
                        
                        // 평균 건물 높이가 일정 수준 이상이면 은폐 지점으로 간주
                        if (avgHeight > 5.0f)
                        {
                            StrategicLocation location = new StrategicLocation
                            {
                                position = node.position,
                                locationType = StrategyType.CoverPoint,
                                dangerScore = 0f, // 나중에 계산
                                visibilityScore = 0f, // 나중에 계산
                                connectedNodes = new List<int> { node.nodeId }
                            };

                            coverPoints.Add(location);
                        }
                    }
                }
            }

            return coverPoints;
        }

        /// <summary>
        /// 교차로를 찾습니다 (여러 경로가 만나는 지점).
        /// </summary>
        /// <param name="graph">도시 그래프</param>
        /// <returns>교차로 리스트</returns>
        private static List<StrategicLocation> FindIntersections(CityGraph graph)
        {
            List<StrategicLocation> intersections = new List<StrategicLocation>();
            List<GraphNode> allNodes = graph.GetAllNodes();

            foreach (GraphNode node in allNodes)
            {
                // 교차로 타입의 노드를 찾음
                if (node.nodeType == NodeType.Intersection)
                {
                    // 연결된 엣지 수 확인
                    List<GraphEdge> edges = graph.GetEdges(node.nodeId);
                    
                    // 3개 이상의 경로가 만나는 지점을 교차로로 간주
                    if (edges.Count >= 3)
                    {
                        StrategicLocation location = new StrategicLocation
                        {
                            position = node.position,
                            locationType = StrategyType.Intersection,
                            dangerScore = 0f, // 나중에 계산
                            visibilityScore = 0f, // 나중에 계산
                            connectedNodes = new List<int> { node.nodeId }
                        };

                        intersections.Add(location);
                    }
                }
            }

            return intersections;
        }

        /// <summary>
        /// 막다른 골목을 찾습니다 (탈출이 어려운 위험 지역).
        /// </summary>
        /// <param name="graph">도시 그래프</param>
        /// <returns>막다른 골목 리스트</returns>
        private static List<StrategicLocation> FindDeadEnds(CityGraph graph)
        {
            List<StrategicLocation> deadEnds = new List<StrategicLocation>();
            List<GraphNode> allNodes = graph.GetAllNodes();

            foreach (GraphNode node in allNodes)
            {
                // 연결된 엣지 수 확인
                List<GraphEdge> edges = graph.GetEdges(node.nodeId);
                
                // 1개의 경로만 연결된 노드를 막다른 골목으로 간주
                if (edges.Count == 1)
                {
                    StrategicLocation location = new StrategicLocation
                    {
                        position = node.position,
                        locationType = StrategyType.DeadEnd,
                        dangerScore = 0f, // 나중에 계산
                        visibilityScore = 0f, // 나중에 계산
                        connectedNodes = new List<int> { node.nodeId }
                    };

                    deadEnds.Add(location);
                }
            }

            return deadEnds;
        }

        /// <summary>
        /// 개방 구역을 찾습니다 (가림 없이 노출되는 위험 지역).
        /// </summary>
        /// <param name="graph">도시 그래프</param>
        /// <param name="buildings">건물 배열</param>
        /// <returns>개방 구역 리스트</returns>
        private static List<StrategicLocation> FindOpenAreas(CityGraph graph, Building[] buildings)
        {
            List<StrategicLocation> openAreas = new List<StrategicLocation>();
            List<GraphNode> allNodes = graph.GetAllNodes();

            foreach (GraphNode node in allNodes)
            {
                // 개방 공간 타입의 노드를 확인
                if (node.nodeType == NodeType.OpenSpace)
                {
                    // 주변 건물이 적거나 없는 경우 개방 구역으로 간주
                    if (node.surroundingBuildingHeights == null || node.surroundingBuildingHeights.Length <= 2)
                    {
                        StrategicLocation location = new StrategicLocation
                        {
                            position = node.position,
                            locationType = StrategyType.OpenArea,
                            dangerScore = 0f, // 나중에 계산
                            visibilityScore = 0f, // 나중에 계산
                            connectedNodes = new List<int> { node.nodeId }
                        };

                        openAreas.Add(location);
                    }
                }
            }

            return openAreas;
        }

        /// <summary>
        /// 우회 경로를 찾습니다 (추적자를 피할 수 있는 대안 경로).
        /// </summary>
        /// <param name="graph">도시 그래프</param>
        /// <returns>우회 경로 리스트</returns>
        private static List<StrategicLocation> FindDetourPaths(CityGraph graph)
        {
            List<StrategicLocation> detourPaths = new List<StrategicLocation>();
            List<GraphNode> allNodes = graph.GetAllNodes();

            foreach (GraphNode node in allNodes)
            {
                // 연결된 엣지 확인
                List<GraphEdge> edges = graph.GetEdges(node.nodeId);
                
                // 우회 또는 은폐 경로 타입의 엣지가 있는지 확인
                bool hasDetourPath = edges.Any(e => e.pathType == PathType.Detour || e.pathType == PathType.Concealed);
                
                // 2개 이상의 경로가 있고 우회 경로가 포함된 경우
                if (edges.Count >= 2 && hasDetourPath)
                {
                    StrategicLocation location = new StrategicLocation
                    {
                        position = node.position,
                        locationType = StrategyType.DetourPath,
                        dangerScore = 0f, // 나중에 계산
                        visibilityScore = 0f, // 나중에 계산
                        connectedNodes = new List<int> { node.nodeId }
                    };

                    detourPaths.Add(location);
                }
            }

            return detourPaths;
        }

        /// <summary>
        /// 위치의 위험도 점수를 계산합니다 (0.0 ~ 1.0).
        /// 생성 지점에 가까울수록, 주변 건물이 적을수록 위험도가 높습니다.
        /// </summary>
        /// <param name="position">위치</param>
        /// <param name="buildings">건물 배열</param>
        /// <param name="spawnPoint">생성 지점</param>
        /// <returns>위험도 점수 (0.0 = 안전, 1.0 = 매우 위험)</returns>
        private static float CalculateDangerScore(Vector3 position, Building[] buildings, Vector3 spawnPoint)
        {
            // 생성 지점까지의 거리 (정규화)
            float distanceToSpawn = Vector3.Distance(position, spawnPoint);

            // 최대 거리를 건물 분포에서 동적으로 계산 (하드코딩 100f 제거)
            float maxDistance = 0f;
            foreach (Building b in buildings)
            {
                if (b != null)
                {
                    float d = Vector3.Distance(spawnPoint, b.position);
                    if (d > maxDistance) maxDistance = d;
                }
            }
            if (maxDistance < 1f) maxDistance = 100f; // 건물 없을 때 폴백

            float distanceScore = 1f - Mathf.Clamp01(distanceToSpawn / maxDistance);

            // 주변 건물 수 계산 (반경 10 단위 내)
            float searchRadius = 10f;
            int nearbyBuildingCount = 0;
            
            foreach (Building building in buildings)
            {
                if (building != null && Vector3.Distance(position, building.position) <= searchRadius)
                {
                    nearbyBuildingCount++;
                }
            }

            // 건물이 적을수록 위험도 증가
            float coverScore = 1f - Mathf.Clamp01(nearbyBuildingCount / 5f);

            // 최종 위험도 점수 (거리 60%, 은폐 40%)
            float dangerScore = (distanceScore * 0.6f) + (coverScore * 0.4f);

            return Mathf.Clamp01(dangerScore);
        }

        /// <summary>
        /// 위치의 가시성 점수를 계산합니다 (0.0 ~ 1.0).
        /// 생성 지점에서 보일 확률을 나타냅니다.
        /// </summary>
        /// <param name="position">위치</param>
        /// <param name="spawnPoint">생성 지점</param>
        /// <param name="buildings">건물 배열</param>
        /// <returns>가시성 점수 (0.0 = 보이지 않음, 1.0 = 완전히 보임)</returns>
        private static float CalculateVisibilityScore(Vector3 position, Vector3 spawnPoint, Building[] buildings)
        {
            // 레이캐스트를 사용하여 가시성 확인
            Vector3 direction = position - spawnPoint;
            float distance = direction.magnitude;
            
            if (distance < 0.1f)
            {
                return 1.0f; // 매우 가까우면 완전히 보임
            }

            direction.Normalize();

            // Unity의 Physics.Raycast를 사용하여 건물에 의한 가림 확인
            RaycastHit hit;
            if (Physics.Raycast(spawnPoint, direction, out hit, distance))
            {
                // 건물에 가려짐
                return 0.2f; // 부분적으로 보일 수 있음
            }

            // 가려지지 않음
            // 거리에 따라 가시성 감소
            float visibilityByDistance = 1f - Mathf.Clamp01(distance / 100f);
            
            return Mathf.Clamp01(visibilityByDistance);
        }
    }
}
