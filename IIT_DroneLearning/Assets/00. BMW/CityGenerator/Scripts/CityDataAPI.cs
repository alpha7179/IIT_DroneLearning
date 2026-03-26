using System.Collections.Generic;
using UnityEngine;

namespace CityGenerator
{
    /// <summary>
    /// 런타임에 도시 데이터를 쿼리하기 위한 싱글톤 API 클래스
    /// CityGraph 및 SpatialIndex에 대한 전역 접근을 제공합니다.
    /// </summary>
    public class CityDataAPI : MonoBehaviour
    {
        // 싱글톤 인스턴스
        private static CityDataAPI instance;

        /// <summary>
        /// CityDataAPI의 싱글톤 인스턴스에 접근합니다.
        /// </summary>
        public static CityDataAPI Instance
        {
            get
            {
                if (instance == null)
                {
                    // 씬에서 기존 인스턴스 찾기
                    instance = FindFirstObjectByType<CityDataAPI>();

                    // 없으면 새로 생성
                    if (instance == null)
                    {
                        GameObject go = new GameObject("CityDataAPI");
                        instance = go.AddComponent<CityDataAPI>();
                        // DontDestroyOnLoad은 Play Mode에서만 유효 (Edit Mode에서 호출 시 예외 발생)
                        if (Application.isPlaying)
                        {
                            DontDestroyOnLoad(go);
                        }
                        Debug.Log("CityDataAPI singleton instance created.");
                    }
                }
                return instance;
            }
        }

        // 도시 그래프 참조
        private CityGraph graph;

        // 공간 인덱스 참조
        private SpatialIndex spatialIndex;

        // 노드 캐시 (빠른 접근을 위한 Dictionary)
        private Dictionary<int, GraphNode> nodeCache;

        // 전략적 위치 캐시 (타입별 인덱싱으로 O(k) 조회 지원)
        private Dictionary<StrategyType, List<StrategicLocation>> strategicLocationsByType;

        // 스폰/타겟 포인트 구성
        private SpawnConfiguration spawnConfiguration;

        /// <summary>
        /// Unity Awake 메서드
        /// 싱글톤 인스턴스를 설정합니다.
        /// </summary>
        private void Awake()
        {
            // 싱글톤 패턴 구현
            if (instance == null)
            {
                instance = this;
                if (Application.isPlaying)
                {
                    DontDestroyOnLoad(gameObject);
                }
                Debug.Log("CityDataAPI initialized.");
            }
            else if (instance != this)
            {
                // 이미 인스턴스가 존재하면 자신을 파괴
                Debug.LogWarning("Duplicate CityDataAPI instance detected. Destroying duplicate.");
                Destroy(gameObject);
            }
        }

        /// <summary>
        /// CityDataAPI를 초기화합니다.
        /// CityGraph 및 SpatialIndex 참조를 저장하고 노드 캐시를 구축합니다.
        /// </summary>
        /// <param name="cityGraph">도시 그래프</param>
        /// <param name="spatialIndex">공간 인덱스</param>
        public void Initialize(CityGraph cityGraph, SpatialIndex spatialIndex, List<StrategicLocation> strategicLocations = null)
        {
            if (cityGraph == null)
            {
                Debug.LogError("CityDataAPI.Initialize: cityGraph is null.");
                return;
            }

            if (spatialIndex == null)
            {
                Debug.LogError("CityDataAPI.Initialize: spatialIndex is null.");
                return;
            }

            this.graph = cityGraph;
            this.spatialIndex = spatialIndex;

            // 노드 캐시 초기화
            InitializeNodeCache();

            // 전략적 위치 캐시 구축 (dangerScore/visibilityScore 보존을 위해 타입별 인덱싱)
            InitializeStrategicLocationsCache(strategicLocations);

            Debug.Log($"CityDataAPI initialized with {graph.NodeCount} nodes.");
        }

        /// <summary>
        /// 노드 캐시 Dictionary를 초기화합니다.
        /// 모든 노드를 Dictionary에 저장하여 O(1) 접근을 가능하게 합니다.
        /// </summary>
        private void InitializeNodeCache()
        {
            nodeCache = new Dictionary<int, GraphNode>();

            if (graph == null)
            {
                Debug.LogWarning("CityDataAPI.InitializeNodeCache: graph is null.");
                return;
            }

            // 모든 노드를 캐시에 저장
            List<GraphNode> allNodes = graph.GetAllNodes();
            foreach (GraphNode node in allNodes)
            {
                nodeCache[node.nodeId] = node;
            }

            Debug.Log($"Node cache initialized with {nodeCache.Count} nodes.");
        }

        /// <summary>
        /// 전략적 위치 캐시를 타입별 Dictionary로 구축합니다.
        /// dangerScore/visibilityScore가 포함된 원본 StrategicLocation 데이터를 보존합니다.
        /// </summary>
        private void InitializeStrategicLocationsCache(List<StrategicLocation> locations)
        {
            strategicLocationsByType = new Dictionary<StrategyType, List<StrategicLocation>>();

            if (locations == null || locations.Count == 0)
            {
                return;
            }

            foreach (StrategicLocation loc in locations)
            {
                if (!strategicLocationsByType.TryGetValue(loc.locationType, out List<StrategicLocation> list))
                {
                    list = new List<StrategicLocation>();
                    strategicLocationsByType[loc.locationType] = list;
                }
                list.Add(loc);
            }

            Debug.Log($"CityDataAPI: 전략적 위치 캐시 구축 완료. 총 {locations.Count}개");
        }

        /// <summary>
        /// API가 초기화되었는지 확인합니다.
        /// </summary>
        /// <returns>초기화 여부</returns>
        public bool IsInitialized()
        {
            return graph != null && spatialIndex != null && nodeCache != null;
        }

        /// <summary>
        /// 노드 ID로 노드를 직접 조회합니다. O(1) 캐시 접근.
        /// </summary>
        /// <param name="nodeId">노드 ID</param>
        /// <returns>해당 노드. 없으면 default 반환</returns>
        public GraphNode GetNodeById(int nodeId)
        {
            if (!IsInitialized())
            {
                Debug.LogError("CityDataAPI.GetNodeById: API is not initialized.");
                return default;
            }

            if (nodeCache.TryGetValue(nodeId, out GraphNode node))
                return node;

            try { return graph.GetNode(nodeId); }
            catch (System.ArgumentException)
            {
                Debug.LogWarning($"CityDataAPI.GetNodeById: Node {nodeId} not found.");
                return default;
            }
        }

        /// <summary>
        /// 특정 위치에서 가장 가까운 노드를 반환합니다.
        /// SpatialIndex.FindNearest를 사용하여 O(log n) 성능을 제공합니다.
        /// </summary>
        /// <param name="position">검색할 위치</param>
        /// <returns>가장 가까운 노드. 노드가 없으면 기본값 반환</returns>
        public GraphNode GetNodeAtPosition(Vector3 position)
        {
            if (!IsInitialized())
            {
                Debug.LogError("CityDataAPI.GetNodeAtPosition: API is not initialized.");
                return default;
            }

            // SpatialIndex를 사용하여 가장 가까운 노드 ID 찾기
            int nearestNodeId = spatialIndex.FindNearest(position);

            // 노드 ID가 유효하지 않은 경우 (-1)
            if (nearestNodeId == -1)
            {
                Debug.LogWarning($"CityDataAPI.GetNodeAtPosition: No node found near position {position}.");
                return default;
            }

            // 노드 캐시에서 노드 정보 가져오기
            if (nodeCache.TryGetValue(nearestNodeId, out GraphNode node))
            {
                return node;
            }

            // 캐시에 없으면 그래프에서 직접 가져오기
            try
            {
                return graph.GetNode(nearestNodeId);
            }
            catch (System.ArgumentException ex)
            {
                Debug.LogError($"CityDataAPI.GetNodeAtPosition: {ex.Message}");
                return default;
            }
        }

        /// <summary>
        /// 특정 위치를 중심으로 반경 내의 모든 노드를 반환합니다.
        /// SpatialIndex.QueryRange를 사용하여 O(log n) 성능을 제공합니다.
        /// </summary>
        /// <param name="center">검색 중심 위치</param>
        /// <param name="radius">검색 반경</param>
        /// <returns>반경 내의 노드 리스트</returns>
        public List<GraphNode> GetNodesInRadius(Vector3 center, float radius)
        {
            List<GraphNode> result = new List<GraphNode>();

            if (!IsInitialized())
            {
                Debug.LogError("CityDataAPI.GetNodesInRadius: API is not initialized.");
                return result;
            }

            if (radius < 0)
            {
                Debug.LogWarning($"CityDataAPI.GetNodesInRadius: Invalid radius {radius}. Must be non-negative.");
                return result;
            }

            // SpatialIndex를 사용하여 반경 내의 노드 ID 리스트 가져오기
            List<int> nodeIds = spatialIndex.QueryRange(center, radius);

            // 각 노드 ID에 대해 노드 정보 가져오기
            foreach (int nodeId in nodeIds)
            {
                if (nodeCache.TryGetValue(nodeId, out GraphNode node))
                {
                    result.Add(node);
                }
                else
                {
                    // 캐시에 없으면 그래프에서 직접 가져오기
                    try
                    {
                        GraphNode graphNode = graph.GetNode(nodeId);
                        result.Add(graphNode);
                    }
                    catch (System.ArgumentException)
                    {
                        Debug.LogWarning($"CityDataAPI.GetNodesInRadius: Node {nodeId} not found in graph.");
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// 특정 노드의 인접 노드 리스트를 반환합니다.
        /// O(1) 시간 복잡도로 인접 노드를 가져옵니다.
        /// </summary>
        /// <param name="nodeId">노드 ID</param>
        /// <returns>인접 노드 리스트</returns>
        public List<GraphNode> GetNeighborNodes(int nodeId)
        {
            List<GraphNode> neighbors = new List<GraphNode>();

            if (!IsInitialized())
            {
                Debug.LogError("CityDataAPI.GetNeighborNodes: API is not initialized.");
                return neighbors;
            }

            // 노드가 존재하는지 확인
            if (!graph.HasNode(nodeId))
            {
                Debug.LogWarning($"CityDataAPI.GetNeighborNodes: Node {nodeId} does not exist.");
                return neighbors;
            }

            // 그래프에서 해당 노드의 엣지 리스트 가져오기 (O(1) - Dictionary 접근)
            List<GraphEdge> edges = graph.GetEdges(nodeId);

            // 각 엣지의 끝 노드를 인접 노드로 추가
            foreach (GraphEdge edge in edges)
            {
                if (nodeCache.TryGetValue(edge.endNodeId, out GraphNode neighborNode))
                {
                    neighbors.Add(neighborNode);
                }
                else
                {
                    // 캐시에 없으면 그래프에서 직접 가져오기
                    try
                    {
                        GraphNode node = graph.GetNode(edge.endNodeId);
                        neighbors.Add(node);
                    }
                    catch (System.ArgumentException)
                    {
                        Debug.LogWarning($"CityDataAPI.GetNeighborNodes: Neighbor node {edge.endNodeId} not found.");
                    }
                }
            }

            return neighbors;
        }

        /// <summary>
        /// 두 노드 간의 최단 경로를 Dijkstra 알고리즘으로 계산합니다.
        /// </summary>
        /// <param name="startNodeId">시작 노드 ID</param>
        /// <param name="endNodeId">끝 노드 ID</param>
        /// <returns>최단 경로의 노드 ID 리스트 (시작 노드부터 끝 노드까지). 경로가 없으면 빈 리스트 반환</returns>
        public List<int> GetShortestPath(int startNodeId, int endNodeId)
        {
            List<int> path = new List<int>();

            if (!IsInitialized())
            {
                Debug.LogError("CityDataAPI.GetShortestPath: API is not initialized.");
                return path;
            }

            // 시작 노드와 끝 노드가 존재하는지 확인
            if (!graph.HasNode(startNodeId))
            {
                Debug.LogWarning($"CityDataAPI.GetShortestPath: Start node {startNodeId} does not exist.");
                return path;
            }

            if (!graph.HasNode(endNodeId))
            {
                Debug.LogWarning($"CityDataAPI.GetShortestPath: End node {endNodeId} does not exist.");
                return path;
            }

            // 시작 노드와 끝 노드가 같으면 시작 노드만 반환
            if (startNodeId == endNodeId)
            {
                path.Add(startNodeId);
                return path;
            }

            // Dijkstra 알고리즘 구현
            // 거리 저장 Dictionary (노드 ID -> 최단 거리)
            Dictionary<int, float> distances = new Dictionary<int, float>();
            // 이전 노드 저장 Dictionary (경로 재구성용)
            Dictionary<int, int> previousNodes = new Dictionary<int, int>();
            // 방문 여부 HashSet
            HashSet<int> visited = new HashSet<int>();
            // 우선순위 큐 (거리가 짧은 노드부터 처리)
            SortedSet<(float distance, int nodeId)> priorityQueue = new SortedSet<(float, int)>();

            // 모든 노드의 거리를 무한대로 초기화
            List<GraphNode> allNodes = graph.GetAllNodes();
            foreach (GraphNode node in allNodes)
            {
                distances[node.nodeId] = float.MaxValue;
            }

            // 시작 노드의 거리를 0으로 설정
            distances[startNodeId] = 0f;
            priorityQueue.Add((0f, startNodeId));

            // Dijkstra 알고리즘 메인 루프
            while (priorityQueue.Count > 0)
            {
                // 가장 거리가 짧은 노드 선택
                var current = priorityQueue.Min;
                priorityQueue.Remove(current);
                int currentNodeId = current.nodeId;
                float currentDistance = current.distance;

                // 이미 방문한 노드는 스킵
                if (visited.Contains(currentNodeId))
                {
                    continue;
                }

                visited.Add(currentNodeId);

                // 목표 노드에 도달하면 종료
                if (currentNodeId == endNodeId)
                {
                    break;
                }

                // 현재 노드의 모든 인접 노드 확인
                List<GraphEdge> edges = graph.GetEdges(currentNodeId);
                foreach (GraphEdge edge in edges)
                {
                    int neighborId = edge.endNodeId;

                    // 이미 방문한 노드는 스킵
                    if (visited.Contains(neighborId))
                    {
                        continue;
                    }

                    // 새로운 거리 계산
                    float newDistance = currentDistance + edge.travelCost;

                    // 더 짧은 경로를 발견하면 업데이트
                    if (newDistance < distances[neighborId])
                    {
                        // 기존 항목 제거 (있다면)
                        priorityQueue.Remove((distances[neighborId], neighborId));

                        // 새로운 거리 및 이전 노드 업데이트
                        distances[neighborId] = newDistance;
                        previousNodes[neighborId] = currentNodeId;

                        // 우선순위 큐에 추가
                        priorityQueue.Add((newDistance, neighborId));
                    }
                }
            }

            // 경로가 존재하지 않으면 빈 리스트 반환
            if (!previousNodes.ContainsKey(endNodeId))
            {
                Debug.LogWarning($"CityDataAPI.GetShortestPath: No path found from {startNodeId} to {endNodeId}.");
                return path;
            }

            // 경로 재구성 (끝 노드에서 시작 노드로 역추적)
            List<int> reversePath = new List<int>();
            int currentNode = endNodeId;
            while (currentNode != startNodeId)
            {
                reversePath.Add(currentNode);
                currentNode = previousNodes[currentNode];
            }
            reversePath.Add(startNodeId);

            // 경로를 뒤집어서 시작 노드부터 끝 노드 순서로 변경
            reversePath.Reverse();
            path = reversePath;

            return path;
        }

        /// <summary>
        /// 특정 위치 주변의 은폐 지점(Cover Points)을 반환합니다.
        /// 반경 내의 노드 중 CoverPoint 전략 마커를 가진 노드를 찾습니다.
        /// </summary>
        /// <param name="position">검색 중심 위치</param>
        /// <param name="radius">검색 반경</param>
        /// <returns>은폐 지점 리스트</returns>
        public List<StrategicLocation> GetCoverPoints(Vector3 position, float radius)
        {
            List<StrategicLocation> coverPoints = new List<StrategicLocation>();

            if (!IsInitialized())
            {
                Debug.LogError("CityDataAPI.GetCoverPoints: API is not initialized.");
                return coverPoints;
            }

            if (radius < 0)
            {
                Debug.LogWarning($"CityDataAPI.GetCoverPoints: Invalid radius {radius}. Must be non-negative.");
                return coverPoints;
            }

            // strategicLocationsByType 캐시 사용: dangerScore/visibilityScore가 보존된 원본 데이터 활용
            if (strategicLocationsByType != null &&
                strategicLocationsByType.TryGetValue(StrategyType.CoverPoint, out List<StrategicLocation> allCoverPoints))
            {
                foreach (StrategicLocation loc in allCoverPoints)
                {
                    if (Vector3.Distance(position, loc.position) <= radius)
                    {
                        coverPoints.Add(loc);
                    }
                }
                return coverPoints;
            }

            // 폴백: 캐시 미비 시 노드 스캔 (dangerScore = 0)
            List<GraphNode> nodesInRadius = GetNodesInRadius(position, radius);
            foreach (GraphNode node in nodesInRadius)
            {
                if (node.strategicMarkers != null && node.strategicMarkers.Contains(StrategyType.CoverPoint))
                {
                    coverPoints.Add(new StrategicLocation
                    {
                        position = node.position,
                        locationType = StrategyType.CoverPoint,
                        dangerScore = 0f,
                        visibilityScore = node.isVisibleFromSpawn ? 1f : 0f,
                        connectedNodes = new List<int> { node.nodeId }
                    });
                }
            }

            return coverPoints;
        }

        /// <summary>
        /// 특정 위치에서 가장 가까운 전략적 위치를 반환합니다.
        /// 지정된 전략 타입의 노드 중 가장 가까운 노드를 찾습니다.
        /// </summary>
        /// <param name="position">검색 시작 위치</param>
        /// <param name="type">찾고자 하는 전략적 위치 타입</param>
        /// <returns>가장 가까운 전략적 위치. 찾지 못하면 기본값 반환</returns>
        public StrategicLocation GetNearestStrategicLocation(Vector3 position, StrategyType type)
        {
            if (!IsInitialized())
            {
                Debug.LogError("CityDataAPI.GetNearestStrategicLocation: API is not initialized.");
                return default;
            }

            // strategicLocationsByType 캐시 사용: O(k) (k = 해당 타입 수), dangerScore/visibilityScore 보존
            if (strategicLocationsByType != null &&
                strategicLocationsByType.TryGetValue(type, out List<StrategicLocation> locations) &&
                locations.Count > 0)
            {
                StrategicLocation nearest = default;
                float minDist = float.MaxValue;

                foreach (StrategicLocation loc in locations)
                {
                    float dist = Vector3.Distance(position, loc.position);
                    if (dist < minDist)
                    {
                        minDist = dist;
                        nearest = loc;
                    }
                }
                return nearest;
            }

            // 폴백: 캐시 미비 시 전체 노드 스캔 (dangerScore = 0)
            List<GraphNode> allNodes = graph.GetAllNodes();
            GraphNode nearestNode = default;
            float minDistance = float.MaxValue;
            bool found = false;

            foreach (GraphNode node in allNodes)
            {
                if (node.strategicMarkers != null && node.strategicMarkers.Contains(type))
                {
                    float distance = Vector3.Distance(position, node.position);
                    if (distance < minDistance)
                    {
                        minDistance = distance;
                        nearestNode = node;
                        found = true;
                    }
                }
            }

            if (!found)
            {
                Debug.LogWarning($"CityDataAPI.GetNearestStrategicLocation: No strategic location of type {type} found.");
                return default;
            }

            return new StrategicLocation
            {
                position = nearestNode.position,
                locationType = type,
                dangerScore = 0f,
                visibilityScore = nearestNode.isVisibleFromSpawn ? 1f : 0f,
                connectedNodes = new List<int> { nearestNode.nodeId }
            };
        }

        /// <summary>
        /// 스폰/타겟 포인트 구성을 설정합니다. CityGenerator가 생성 후 호출합니다.
        /// </summary>
        public void SetSpawnConfiguration(SpawnConfiguration config)
        {
            spawnConfiguration = config;
        }

        /// <summary>
        /// 스폰/타겟 포인트가 유효하게 설정되었는지 확인합니다.
        /// </summary>
        public bool HasSpawnConfiguration() => spawnConfiguration.isValid;

        /// <summary>
        /// 도망 드론의 스폰 위치를 반환합니다.
        /// </summary>
        public Vector3 GetEvaderSpawnPosition() => spawnConfiguration.evaderSpawnPosition;

        /// <summary>
        /// 추적 드론의 스폰 위치를 반환합니다.
        /// </summary>
        public Vector3 GetPursuerSpawnPosition() => spawnConfiguration.pursuerSpawnPosition;

        /// <summary>
        /// 타겟 포인트 위치를 반환합니다.
        /// </summary>
        public Vector3 GetTargetPosition() => spawnConfiguration.targetPosition;

        /// <summary>
        /// 스폰/타겟 포인트 전체 구성을 반환합니다.
        /// </summary>
        public SpawnConfiguration GetSpawnConfiguration() => spawnConfiguration;

        /// <summary>
        /// 두 위치 간의 가시성을 확인합니다.
        /// Unity의 레이캐스트를 사용하여 건물에 의한 가림 현상을 체크합니다.
        /// </summary>
        /// <param name="from">시작 위치</param>
        /// <param name="to">목표 위치</param>
        /// <returns>두 위치가 서로 보이면 true, 건물에 의해 가려지면 false</returns>
        public bool IsPositionVisible(Vector3 from, Vector3 to)
        {
            if (!IsInitialized())
            {
                Debug.LogError("CityDataAPI.IsPositionVisible: API is not initialized.");
                return false;
            }

            // 두 위치 간의 방향 벡터 계산
            Vector3 direction = to - from;
            float distance = direction.magnitude;

            // 거리가 0이면 같은 위치이므로 가시성 true
            if (distance < 0.001f)
            {
                return true;
            }

            // 레이캐스트를 사용하여 두 위치 사이에 장애물이 있는지 확인
            // LayerMask를 사용하지 않으면 모든 콜라이더를 체크합니다
            RaycastHit hit;
            if (Physics.Raycast(from, direction.normalized, out hit, distance))
            {
                // 레이캐스트가 무언가에 충돌했으면 가려진 것
                // 충돌 지점이 목표 위치보다 가까우면 가시성 false
                if (hit.distance < distance - 0.001f) // 부동소수점 오차 고려
                {
                    return false;
                }
            }

            // 레이캐스트가 아무것도 충돌하지 않았거나, 목표 위치에 도달했으면 가시성 true
            return true;
        }

        /// <summary>
        /// Unity OnDestroy 메서드
        /// 싱글톤 인스턴스를 정리합니다.
        /// </summary>
        private void OnDestroy()
        {
            if (instance == this)
            {
                instance = null;
            }
        }
    }
}
