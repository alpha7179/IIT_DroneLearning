using System.Collections.Generic;
using UnityEngine;

namespace ProceduralCityGenerator
{
    /// <summary>
    /// CityDataAPI의 그래프 탐색 기능 사용 예제
    /// GetNeighborNodes 및 GetShortestPath 메서드 사용법을 보여줍니다.
    /// </summary>
    public class CityDataAPIUsageExample : MonoBehaviour
    {
        [Header("Example Settings")]
        [Tooltip("시작 위치")]
        public Vector3 startPosition = new Vector3(0, 0, 0);
        
        [Tooltip("목표 위치")]
        public Vector3 targetPosition = new Vector3(10, 0, 10);

        [Header("Visibility Check")]
        [Tooltip("가시성 확인 시작 위치")]
        public Vector3 visibilityFrom = new Vector3(0, 5, 0);
        
        [Tooltip("가시성 확인 목표 위치")]
        public Vector3 visibilityTo = new Vector3(20, 5, 20);

        [Header("Visualization")]
        [Tooltip("경로를 시각화할지 여부")]
        public bool visualizePath = true;
        
        [Tooltip("경로 선 색상")]
        public Color pathColor = Color.green;
        
        [Tooltip("인접 노드 연결선 색상")]
        public Color neighborColor = Color.yellow;
        
        [Tooltip("가시성 선 색상 (보임)")]
        public Color visibleColor = Color.green;
        
        [Tooltip("가시성 선 색상 (가려짐)")]
        public Color blockedColor = Color.red;

        private List<Vector3> pathPositions = new List<Vector3>();
        private List<Vector3> neighborPositions = new List<Vector3>();
        private Vector3 currentNodePosition;
        private bool isVisible = false;
        private bool hasVisibilityCheck = false;

        /// <summary>
        /// 예제 실행: 최단 경로 찾기
        /// </summary>
        [ContextMenu("Find Shortest Path")]
        public void FindShortestPath()
        {
            if (!CityDataAPI.Instance.IsInitialized())
            {
                Debug.LogError("CityDataAPI is not initialized. Generate a city first.");
                return;
            }

            // 1. 시작 위치에서 가장 가까운 노드 찾기
            GraphNode startNode = CityDataAPI.Instance.GetNodeAtPosition(startPosition);
            if (startNode.nodeId == 0 && startNode.position == Vector3.zero)
            {
                Debug.LogWarning("No node found near start position.");
                return;
            }

            // 2. 목표 위치에서 가장 가까운 노드 찾기
            GraphNode targetNode = CityDataAPI.Instance.GetNodeAtPosition(targetPosition);
            if (targetNode.nodeId == 0 && targetNode.position == Vector3.zero)
            {
                Debug.LogWarning("No node found near target position.");
                return;
            }

            Debug.Log($"Start Node: ID {startNode.nodeId} at {startNode.position}");
            Debug.Log($"Target Node: ID {targetNode.nodeId} at {targetNode.position}");

            // 3. 최단 경로 계산 (Dijkstra 알고리즘)
            List<int> pathNodeIds = CityDataAPI.Instance.GetShortestPath(startNode.nodeId, targetNode.nodeId);

            if (pathNodeIds.Count == 0)
            {
                Debug.LogWarning("No path found between start and target nodes.");
                return;
            }

            // 4. 경로 노드들의 위치 저장 (시각화용)
            pathPositions.Clear();
            foreach (int nodeId in pathNodeIds)
            {
                // 노드 위치 가져오기
                List<GraphNode> neighbors = CityDataAPI.Instance.GetNeighborNodes(nodeId);
                if (neighbors.Count > 0)
                {
                    GraphNode node = CityDataAPI.Instance.GetNodeAtPosition(neighbors[0].position);
                    pathPositions.Add(node.position);
                }
            }

            Debug.Log($"Path found with {pathNodeIds.Count} nodes. Total segments: {pathPositions.Count - 1}");
        }

        /// <summary>
        /// 예제 실행: 인접 노드 찾기
        /// </summary>
        [ContextMenu("Find Neighbor Nodes")]
        public void FindNeighborNodes()
        {
            if (!CityDataAPI.Instance.IsInitialized())
            {
                Debug.LogError("CityDataAPI is not initialized. Generate a city first.");
                return;
            }

            // 1. 시작 위치에서 가장 가까운 노드 찾기
            GraphNode currentNode = CityDataAPI.Instance.GetNodeAtPosition(startPosition);
            if (currentNode.nodeId == 0 && currentNode.position == Vector3.zero)
            {
                Debug.LogWarning("No node found near start position.");
                return;
            }

            currentNodePosition = currentNode.position;
            Debug.Log($"Current Node: ID {currentNode.nodeId} at {currentNode.position}, Type: {currentNode.nodeType}");

            // 2. 인접 노드 가져오기 (O(1) 성능)
            List<GraphNode> neighbors = CityDataAPI.Instance.GetNeighborNodes(currentNode.nodeId);
            
            Debug.Log($"Found {neighbors.Count} neighbor nodes:");
            neighborPositions.Clear();

            foreach (GraphNode neighbor in neighbors)
            {
                float distance = Vector3.Distance(currentNode.position, neighbor.position);
                Debug.Log($"  - Neighbor ID {neighbor.nodeId} at {neighbor.position}, " +
                          $"Distance: {distance:F2}, Type: {neighbor.nodeType}");
                neighborPositions.Add(neighbor.position);
            }
        }

        /// <summary>
        /// 예제 실행: 두 위치 간 가시성 확인
        /// 레이캐스트를 사용하여 건물에 의한 가림 현상을 체크합니다.
        /// </summary>
        [ContextMenu("Check Visibility")]
        public void CheckVisibility()
        {
            if (!CityDataAPI.Instance.IsInitialized())
            {
                Debug.LogError("CityDataAPI is not initialized. Generate a city first.");
                return;
            }

            // 두 위치 간의 가시성 확인
            isVisible = CityDataAPI.Instance.IsPositionVisible(visibilityFrom, visibilityTo);
            hasVisibilityCheck = true;

            float distance = Vector3.Distance(visibilityFrom, visibilityTo);
            
            if (isVisible)
            {
                Debug.Log($"<color=green>Visibility Check: VISIBLE</color>");
                Debug.Log($"From {visibilityFrom} to {visibilityTo} (Distance: {distance:F2}m)");
                Debug.Log("No buildings are blocking the line of sight.");
            }
            else
            {
                Debug.Log($"<color=red>Visibility Check: BLOCKED</color>");
                Debug.Log($"From {visibilityFrom} to {visibilityTo} (Distance: {distance:F2}m)");
                Debug.Log("One or more buildings are blocking the line of sight.");
            }
        }

        /// <summary>
        /// 경로 및 인접 노드를 Scene 뷰에 시각화
        /// </summary>
        private void OnDrawGizmos()
        {
            if (!visualizePath)
                return;

            // 최단 경로 시각화
            if (pathPositions.Count > 1)
            {
                Gizmos.color = pathColor;
                for (int i = 0; i < pathPositions.Count - 1; i++)
                {
                    Gizmos.DrawLine(pathPositions[i], pathPositions[i + 1]);
                    Gizmos.DrawSphere(pathPositions[i], 0.3f);
                }
                Gizmos.DrawSphere(pathPositions[pathPositions.Count - 1], 0.3f);
            }

            // 인접 노드 시각화
            if (neighborPositions.Count > 0)
            {
                Gizmos.color = neighborColor;
                foreach (Vector3 neighborPos in neighborPositions)
                {
                    Gizmos.DrawLine(currentNodePosition, neighborPos);
                    Gizmos.DrawWireSphere(neighborPos, 0.2f);
                }
                Gizmos.DrawSphere(currentNodePosition, 0.4f);
            }

            // 가시성 확인 시각화
            if (hasVisibilityCheck)
            {
                Gizmos.color = isVisible ? visibleColor : blockedColor;
                Gizmos.DrawLine(visibilityFrom, visibilityTo);
                Gizmos.DrawWireSphere(visibilityFrom, 0.5f);
                Gizmos.DrawWireSphere(visibilityTo, 0.5f);
            }

            // 시작 및 목표 위치 표시
            Gizmos.color = Color.blue;
            Gizmos.DrawWireSphere(startPosition, 0.5f);
            Gizmos.color = Color.red;
            Gizmos.DrawWireSphere(targetPosition, 0.5f);
        }
    }
}
