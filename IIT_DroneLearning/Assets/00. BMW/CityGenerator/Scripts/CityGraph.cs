using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;

namespace CityGenerator
{
    /// <summary>
    /// 도시 구조를 표현하는 그래프 자료구조 클래스
    /// 노드-엣지 기반 그래프로 AI 경로 계획 및 전략 수립을 지원합니다.
    /// </summary>
    public class CityGraph
    {
        // 노드 저장소: 노드 ID를 키로 하는 Dictionary
        private Dictionary<int, GraphNode> nodes;
        
        // 엣지 저장소: 인접 리스트 방식 (노드 ID -> 연결된 엣지 리스트)
        private Dictionary<int, List<GraphEdge>> adjacencyList;
        
        // 다음 노드 ID (자동 증가)
        private int nextNodeId;

        /// <summary>
        /// CityGraph 생성자
        /// </summary>
        public CityGraph()
        {
            nodes = new Dictionary<int, GraphNode>();
            adjacencyList = new Dictionary<int, List<GraphEdge>>();
            nextNodeId = 0;
        }

        #region 노드 관리

        /// <summary>
        /// 그래프에 새로운 노드를 추가합니다.
        /// </summary>
        /// <param name="position">노드의 월드 좌표</param>
        /// <param name="type">노드 타입</param>
        /// <param name="elevation">고도 정보</param>
        /// <returns>생성된 노드의 ID</returns>
        public int AddNode(Vector3 position, NodeType type, float elevation)
        {
            int nodeId = nextNodeId++;
            
            GraphNode node = new GraphNode
            {
                nodeId = nodeId,
                position = position,
                nodeType = type,
                elevation = elevation,
                surroundingBuildingHeights = new float[0],
                strategicMarkers = new List<StrategyType>(),
                isVisibleFromSpawn = false
            };

            nodes[nodeId] = node;
            adjacencyList[nodeId] = new List<GraphEdge>();

            return nodeId;
        }

        /// <summary>
        /// 특정 ID의 노드를 가져옵니다.
        /// </summary>
        /// <param name="nodeId">노드 ID</param>
        /// <returns>GraphNode 구조체</returns>
        public GraphNode GetNode(int nodeId)
        {
            if (nodes.TryGetValue(nodeId, out GraphNode node))
            {
                return node;
            }
            
            throw new ArgumentException($"Node with ID {nodeId} does not exist.");
        }

        /// <summary>
        /// 모든 노드를 리스트로 반환합니다.
        /// </summary>
        /// <returns>모든 노드의 리스트</returns>
        public List<GraphNode> GetAllNodes()
        {
            return new List<GraphNode>(nodes.Values);
        }

        /// <summary>
        /// 노드가 존재하는지 확인합니다.
        /// </summary>
        /// <param name="nodeId">노드 ID</param>
        /// <returns>존재 여부</returns>
        public bool HasNode(int nodeId)
        {
            return nodes.ContainsKey(nodeId);
        }

        /// <summary>
        /// 노드 정보를 업데이트합니다.
        /// </summary>
        /// <param name="nodeId">노드 ID</param>
        /// <param name="node">업데이트할 노드 정보</param>
        public void UpdateNode(int nodeId, GraphNode node)
        {
            if (nodes.ContainsKey(nodeId))
            {
                nodes[nodeId] = node;
            }
        }

        /// <summary>
        /// 그래프의 총 노드 수를 반환합니다.
        /// </summary>
        public int NodeCount => nodes.Count;

        #endregion

        #region 엣지 관리

        /// <summary>
        /// 두 노드 사이에 엣지를 추가합니다.
        /// </summary>
        /// <param name="startNodeId">시작 노드 ID</param>
        /// <param name="endNodeId">끝 노드 ID</param>
        /// <param name="cost">이동 비용</param>
        /// <param name="pathType">경로 타입</param>
        public void AddEdge(int startNodeId, int endNodeId, float cost, PathType pathType)
        {
            if (!nodes.ContainsKey(startNodeId))
            {
                throw new ArgumentException($"Start node {startNodeId} does not exist.");
            }
            
            if (!nodes.ContainsKey(endNodeId))
            {
                throw new ArgumentException($"End node {endNodeId} does not exist.");
            }

            GraphEdge edge = new GraphEdge
            {
                startNodeId = startNodeId,
                endNodeId = endNodeId,
                travelCost = cost,
                visibilityScore = 0.5f, // 기본값
                pathType = pathType
            };

            adjacencyList[startNodeId].Add(edge);
        }

        /// <summary>
        /// 특정 노드에서 나가는 모든 엣지를 가져옵니다.
        /// </summary>
        /// <param name="nodeId">노드 ID</param>
        /// <returns>엣지 리스트</returns>
        public List<GraphEdge> GetEdges(int nodeId)
        {
            if (adjacencyList.TryGetValue(nodeId, out List<GraphEdge> edges))
            {
                return new List<GraphEdge>(edges);
            }
            
            return new List<GraphEdge>();
        }

        /// <summary>
        /// 그래프의 총 엣지 수를 반환합니다.
        /// </summary>
        public int EdgeCount
        {
            get
            {
                int count = 0;
                foreach (var edgeList in adjacencyList.Values)
                {
                    count += edgeList.Count;
                }
                return count;
            }
        }

        #endregion

        #region 그래프 구축

        /// <summary>
        /// 격자 기반으로 그래프를 구축합니다.
        /// </summary>
        /// <param name="grid">격자 셀 2D 배열</param>
        /// <param name="unitDistance">단위 거리</param>
        public void BuildFromGrid(GridCell[,] grid, float unitDistance)
        {
            if (grid == null || grid.Length == 0)
            {
                Debug.LogWarning("Grid is null or empty. Cannot build graph.");
                return;
            }

            int width = grid.GetLength(0);
            int depth = grid.GetLength(1);

            // 노드 ID를 저장할 2D 배열 (격자 좌표 -> 노드 ID 매핑)
            int[,] nodeIdMap = new int[width, depth];
            
            // 초기화: -1은 노드가 없음을 의미
            for (int x = 0; x < width; x++)
            {
                for (int z = 0; z < depth; z++)
                {
                    nodeIdMap[x, z] = -1;
                }
            }

            // 1단계: 모든 격자 셀에 대해 노드 생성
            for (int x = 0; x < width; x++)
            {
                for (int z = 0; z < depth; z++)
                {
                    GridCell cell = grid[x, z];
                    
                    // 노드 타입 결정
                    NodeType nodeType = DetermineNodeType(grid, x, z, width, depth);
                    
                    // 노드 추가
                    int nodeId = AddNode(cell.worldPosition, nodeType, 0f);
                    nodeIdMap[x, z] = nodeId;
                    
                    // 주변 건물 높이 계산
                    float[] surroundingHeights = CalculateSurroundingBuildingHeights(grid, x, z, width, depth);
                    
                    // 노드 정보 업데이트
                    GraphNode node = GetNode(nodeId);
                    node.surroundingBuildingHeights = surroundingHeights;
                    UpdateNode(nodeId, node);
                }
            }

            // 2단계: 인접 노드 간 엣지 연결
            for (int x = 0; x < width; x++)
            {
                for (int z = 0; z < depth; z++)
                {
                    int currentNodeId = nodeIdMap[x, z];
                    if (currentNodeId == -1) continue;

                    // 중복 엣지 방지: 우(+X) 및 위(+Z) 방향만 처리하여 각 인접 쌍을 한 번씩만 연결
                    // 4방향(상하좌우) 처리 시 (A,B) 쌍이 양쪽에서 각각 처리되어 엣지가 2배로 생성되는 버그 방지
                    int[] dx = { 1, 0 };
                    int[] dz = { 0, 1 };

                    for (int i = 0; i < 2; i++)
                    {
                        int nx = x + dx[i];
                        int nz = z + dz[i];

                        // 경계 체크
                        if (nx >= 0 && nx < width && nz >= 0 && nz < depth)
                        {
                            int neighborNodeId = nodeIdMap[nx, nz];
                            if (neighborNodeId != -1)
                            {
                                // 이동 비용 계산 (유클리드 거리)
                                Vector3 pos1 = nodes[currentNodeId].position;
                                Vector3 pos2 = nodes[neighborNodeId].position;
                                float cost = Vector3.Distance(pos1, pos2);

                                // 경로 타입 결정
                                PathType pathType = DeterminePathType(grid, x, z, nx, nz);

                                // 양방향 엣지 추가 (각 쌍당 정확히 1회)
                                AddEdge(currentNodeId, neighborNodeId, cost, pathType);
                                AddEdge(neighborNodeId, currentNodeId, cost, pathType);
                            }
                        }
                    }
                }
            }

            Debug.Log($"Graph built: {NodeCount} nodes, {EdgeCount} edges");
        }

        /// <summary>
        /// 격자 셀의 노드 타입을 결정합니다.
        /// </summary>
        private NodeType DetermineNodeType(GridCell[,] grid, int x, int z, int width, int depth)
        {
            GridCell cell = grid[x, z];
            
            // 현재 셀에 건물이 있으면 BuildingCorner
            if (cell.hasBuilding)
            {
                return NodeType.BuildingCorner;
            }

            // 인접 셀 확인
            int adjacentBuildings = 0;
            int adjacentOpenSpaces = 0;

            int[] dx = { 0, 0, -1, 1 };
            int[] dz = { -1, 1, 0, 0 };

            for (int i = 0; i < 4; i++)
            {
                int nx = x + dx[i];
                int nz = z + dz[i];

                if (nx >= 0 && nx < width && nz >= 0 && nz < depth)
                {
                    if (grid[nx, nz].hasBuilding)
                    {
                        adjacentBuildings++;
                    }
                    else
                    {
                        adjacentOpenSpaces++;
                    }
                }
            }

            // 교차로: 3개 이상의 개방 공간과 인접
            if (adjacentOpenSpaces >= 3)
            {
                return NodeType.Intersection;
            }

            // 골목 입구: 1-2개의 건물과 인접
            if (adjacentBuildings >= 1 && adjacentBuildings <= 2)
            {
                return NodeType.AlleyEntrance;
            }

            // 기본값: 개방 공간
            return NodeType.OpenSpace;
        }

        /// <summary>
        /// 주변 건물 높이를 계산합니다.
        /// </summary>
        private float[] CalculateSurroundingBuildingHeights(GridCell[,] grid, int x, int z, int width, int depth)
        {
            List<float> heights = new List<float>();

            // 3x3 영역 확인
            for (int dx = -1; dx <= 1; dx++)
            {
                for (int dz = -1; dz <= 1; dz++)
                {
                    if (dx == 0 && dz == 0) continue; // 자기 자신 제외

                    int nx = x + dx;
                    int nz = z + dz;

                    if (nx >= 0 && nx < width && nz >= 0 && nz < depth)
                    {
                        GridCell cell = grid[nx, nz];
                        if (cell.hasBuilding)
                        {
                            heights.Add(cell.buildingHeight);
                        }
                    }
                }
            }

            return heights.ToArray();
        }

        /// <summary>
        /// 경로 타입을 결정합니다.
        /// </summary>
        private PathType DeterminePathType(GridCell[,] grid, int x1, int z1, int x2, int z2)
        {
            GridCell cell1 = grid[x1, z1];
            GridCell cell2 = grid[x2, z2];

            // 두 셀 모두 건물이 없으면 직선 경로
            if (!cell1.hasBuilding && !cell2.hasBuilding)
            {
                return PathType.Direct;
            }

            // 한 셀에만 건물이 있으면 우회 경로
            if (cell1.hasBuilding != cell2.hasBuilding)
            {
                return PathType.Detour;
            }

            // 두 셀 모두 건물이 있으면 은폐 경로
            return PathType.Concealed;
        }

        #endregion

        #region 직렬화

        /// <summary>
        /// 그래프를 JSON 형식으로 직렬화합니다.
        /// </summary>
        /// <returns>JSON 문자열</returns>
        public string SerializeToJson()
        {
            GraphData data = new GraphData
            {
                nodes = GetAllNodes(),
                edges = GetAllEdges()
            };

            return JsonUtility.ToJson(data, true);
        }

        /// <summary>
        /// JSON 문자열에서 그래프를 역직렬화합니다.
        /// </summary>
        /// <param name="json">JSON 문자열</param>
        public void DeserializeFromJson(string json)
        {
            if (string.IsNullOrEmpty(json))
            {
                Debug.LogWarning("JSON string is null or empty.");
                return;
            }

            try
            {
                GraphData data = JsonUtility.FromJson<GraphData>(json);

                // 기존 데이터 초기화
                nodes.Clear();
                adjacencyList.Clear();
                nextNodeId = 0;

                // 노드 복원
                foreach (GraphNode node in data.nodes)
                {
                    nodes[node.nodeId] = node;
                    adjacencyList[node.nodeId] = new List<GraphEdge>();
                    
                    if (node.nodeId >= nextNodeId)
                    {
                        nextNodeId = node.nodeId + 1;
                    }
                }

                // 엣지 복원
                foreach (GraphEdge edge in data.edges)
                {
                    if (adjacencyList.ContainsKey(edge.startNodeId))
                    {
                        adjacencyList[edge.startNodeId].Add(edge);
                    }
                }

                Debug.Log($"Graph deserialized: {NodeCount} nodes, {EdgeCount} edges");
            }
            catch (Exception e)
            {
                Debug.LogError($"Failed to deserialize graph from JSON: {e.Message}");
            }
        }

        /// <summary>
        /// 그래프를 바이너리 형식으로 직렬화합니다.
        /// </summary>
        /// <returns>바이트 배열</returns>
        public byte[] SerializeToBinary()
        {
            using (MemoryStream stream = new MemoryStream())
            using (BinaryWriter writer = new BinaryWriter(stream))
            {
                // 노드 수 쓰기
                writer.Write(nodes.Count);

                // 각 노드 쓰기
                foreach (var node in nodes.Values)
                {
                    WriteNode(writer, node);
                }

                // 엣지 수 쓰기
                int edgeCount = EdgeCount;
                writer.Write(edgeCount);

                // 각 엣지 쓰기
                foreach (var edgeList in adjacencyList.Values)
                {
                    foreach (var edge in edgeList)
                    {
                        WriteEdge(writer, edge);
                    }
                }

                return stream.ToArray();
            }
        }

        /// <summary>
        /// 바이너리 데이터에서 그래프를 역직렬화합니다.
        /// </summary>
        /// <param name="data">바이트 배열</param>
        public void DeserializeFromBinary(byte[] data)
        {
            if (data == null || data.Length == 0)
            {
                Debug.LogWarning("Binary data is null or empty.");
                return;
            }

            try
            {
                using (MemoryStream stream = new MemoryStream(data))
                using (BinaryReader reader = new BinaryReader(stream))
                {
                    // 기존 데이터 초기화
                    nodes.Clear();
                    adjacencyList.Clear();
                    nextNodeId = 0;

                    // 노드 수 읽기
                    int nodeCount = reader.ReadInt32();

                    // 각 노드 읽기
                    for (int i = 0; i < nodeCount; i++)
                    {
                        GraphNode node = ReadNode(reader);
                        nodes[node.nodeId] = node;
                        adjacencyList[node.nodeId] = new List<GraphEdge>();
                        
                        if (node.nodeId >= nextNodeId)
                        {
                            nextNodeId = node.nodeId + 1;
                        }
                    }

                    // 엣지 수 읽기
                    int edgeCount = reader.ReadInt32();

                    // 각 엣지 읽기
                    for (int i = 0; i < edgeCount; i++)
                    {
                        GraphEdge edge = ReadEdge(reader);
                        if (adjacencyList.ContainsKey(edge.startNodeId))
                        {
                            adjacencyList[edge.startNodeId].Add(edge);
                        }
                    }

                    Debug.Log($"Graph deserialized from binary: {NodeCount} nodes, {EdgeCount} edges");
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"Failed to deserialize graph from binary: {e.Message}");
            }
        }

        /// <summary>
        /// 그래프를 파일에 저장합니다 (JSON 형식).
        /// </summary>
        /// <param name="filePath">파일 경로</param>
        public void SaveToFile(string filePath)
        {
            try
            {
                string json = SerializeToJson();
                File.WriteAllText(filePath, json);
                Debug.Log($"Graph saved to {filePath}");
            }
            catch (Exception e)
            {
                Debug.LogError($"Failed to save graph to file: {e.Message}");
            }
        }

        /// <summary>
        /// 파일에서 그래프를 로드합니다 (JSON 형식).
        /// </summary>
        /// <param name="filePath">파일 경로</param>
        public void LoadFromFile(string filePath)
        {
            try
            {
                if (!File.Exists(filePath))
                {
                    Debug.LogWarning($"File does not exist: {filePath}");
                    return;
                }

                string json = File.ReadAllText(filePath);
                DeserializeFromJson(json);
            }
            catch (Exception e)
            {
                Debug.LogError($"Failed to load graph from file: {e.Message}");
            }
        }

        /// <summary>
        /// 그래프를 바이너리 파일에 저장합니다.
        /// </summary>
        /// <param name="filePath">파일 경로</param>
        public void SaveToBinaryFile(string filePath)
        {
            try
            {
                byte[] data = SerializeToBinary();
                File.WriteAllBytes(filePath, data);
                Debug.Log($"Graph saved to binary file {filePath}");
            }
            catch (Exception e)
            {
                Debug.LogError($"Failed to save graph to binary file: {e.Message}");
            }
        }

        /// <summary>
        /// 바이너리 파일에서 그래프를 로드합니다.
        /// </summary>
        /// <param name="filePath">파일 경로</param>
        public void LoadFromBinaryFile(string filePath)
        {
            try
            {
                if (!File.Exists(filePath))
                {
                    Debug.LogWarning($"File does not exist: {filePath}");
                    return;
                }

                byte[] data = File.ReadAllBytes(filePath);
                DeserializeFromBinary(data);
            }
            catch (Exception e)
            {
                Debug.LogError($"Failed to load graph from binary file: {e.Message}");
            }
        }

        #endregion

        #region 헬퍼 메서드

        /// <summary>
        /// 모든 엣지를 리스트로 반환합니다.
        /// </summary>
        private List<GraphEdge> GetAllEdges()
        {
            List<GraphEdge> allEdges = new List<GraphEdge>();
            foreach (var edgeList in adjacencyList.Values)
            {
                allEdges.AddRange(edgeList);
            }
            return allEdges;
        }

        /// <summary>
        /// 노드를 바이너리로 쓰기
        /// </summary>
        private void WriteNode(BinaryWriter writer, GraphNode node)
        {
            writer.Write(node.nodeId);
            writer.Write(node.position.x);
            writer.Write(node.position.y);
            writer.Write(node.position.z);
            writer.Write((int)node.nodeType);
            writer.Write(node.elevation);
            
            // 주변 건물 높이
            writer.Write(node.surroundingBuildingHeights.Length);
            foreach (float height in node.surroundingBuildingHeights)
            {
                writer.Write(height);
            }
            
            // 전략적 마커
            writer.Write(node.strategicMarkers.Count);
            foreach (StrategyType marker in node.strategicMarkers)
            {
                writer.Write((int)marker);
            }
            
            writer.Write(node.isVisibleFromSpawn);
        }

        /// <summary>
        /// 바이너리에서 노드 읽기
        /// </summary>
        private GraphNode ReadNode(BinaryReader reader)
        {
            GraphNode node = new GraphNode
            {
                nodeId = reader.ReadInt32(),
                position = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle()),
                nodeType = (NodeType)reader.ReadInt32(),
                elevation = reader.ReadSingle()
            };

            // 주변 건물 높이
            int heightCount = reader.ReadInt32();
            node.surroundingBuildingHeights = new float[heightCount];
            for (int i = 0; i < heightCount; i++)
            {
                node.surroundingBuildingHeights[i] = reader.ReadSingle();
            }

            // 전략적 마커
            int markerCount = reader.ReadInt32();
            node.strategicMarkers = new List<StrategyType>();
            for (int i = 0; i < markerCount; i++)
            {
                node.strategicMarkers.Add((StrategyType)reader.ReadInt32());
            }

            node.isVisibleFromSpawn = reader.ReadBoolean();

            return node;
        }

        /// <summary>
        /// 엣지를 바이너리로 쓰기
        /// </summary>
        private void WriteEdge(BinaryWriter writer, GraphEdge edge)
        {
            writer.Write(edge.startNodeId);
            writer.Write(edge.endNodeId);
            writer.Write(edge.travelCost);
            writer.Write(edge.visibilityScore);
            writer.Write((int)edge.pathType);
        }

        /// <summary>
        /// 바이너리에서 엣지 읽기
        /// </summary>
        private GraphEdge ReadEdge(BinaryReader reader)
        {
            return new GraphEdge
            {
                startNodeId = reader.ReadInt32(),
                endNodeId = reader.ReadInt32(),
                travelCost = reader.ReadSingle(),
                visibilityScore = reader.ReadSingle(),
                pathType = (PathType)reader.ReadInt32()
            };
        }

        #endregion

        #region 직렬화용 데이터 구조

        /// <summary>
        /// JSON 직렬화를 위한 데이터 구조
        /// </summary>
        [Serializable]
        private class GraphData
        {
            public List<GraphNode> nodes;
            public List<GraphEdge> edges;
        }

        #endregion
    }
}
