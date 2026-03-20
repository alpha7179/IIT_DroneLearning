using System.Collections.Generic;
using UnityEngine;

namespace CityGenerator
{
    /// <summary>
    /// 공간 쿼리를 최적화하기 위한 Quadtree 기반 공간 인덱스
    /// O(log n) 성능으로 위치 기반 노드 검색을 지원합니다.
    /// </summary>
    public class SpatialIndex
    {
        private QuadtreeNode root;
        private Bounds worldBounds;
        private int maxDepth;
        private int maxNodesPerCell;

        /// <summary>
        /// SpatialIndex 생성자
        /// </summary>
        /// <param name="bounds">전체 월드 경계</param>
        /// <param name="maxDepth">최대 트리 깊이 (기본값: 8)</param>
        /// <param name="maxNodesPerCell">셀당 최대 노드 수 (기본값: 4)</param>
        public SpatialIndex(Bounds bounds, int maxDepth = 8, int maxNodesPerCell = 4)
        {
            this.worldBounds = bounds;
            this.maxDepth = maxDepth;
            this.maxNodesPerCell = maxNodesPerCell;
            this.root = new QuadtreeNode(bounds);
        }

        /// <summary>
        /// 노드를 Quadtree에 삽입합니다.
        /// </summary>
        /// <param name="nodeId">노드 ID</param>
        /// <param name="position">노드 위치</param>
        public void Insert(int nodeId, Vector3 position)
        {
            InsertRecursive(root, nodeId, position, 0);
        }

        /// <summary>
        /// 재귀적으로 노드를 삽입합니다.
        /// </summary>
        private void InsertRecursive(QuadtreeNode node, int nodeId, Vector3 position, int depth)
        {
            // 위치가 경계 내에 있는지 확인
            if (!node.bounds.Contains(position))
            {
                return;
            }

            // 리프 노드인 경우
            if (node.isLeaf)
            {
                // 노드 추가
                node.nodes.Add((nodeId, position));

                // 최대 노드 수를 초과하고 최대 깊이에 도달하지 않았으면 분할
                // Split()이 내부에서 기존 노드를 자식으로 재분배하므로 별도 처리 불필요
                if (node.nodes.Count > maxNodesPerCell && depth < maxDepth)
                {
                    node.Split();
                }
            }
            else
            {
                // 리프가 아니면 적절한 자식 노드에 삽입
                int childIndex = node.GetChildIndexPublic(position);
                if (childIndex >= 0 && childIndex < 4 && node.children != null)
                {
                    InsertRecursive(node.children[childIndex], nodeId, position, depth + 1);
                }
            }
        }

        /// <summary>
        /// 특정 반경 내의 노드들을 검색합니다.
        /// </summary>
        /// <param name="center">중심 위치</param>
        /// <param name="radius">검색 반경</param>
        /// <returns>반경 내 노드 ID 리스트</returns>
        public List<int> QueryRange(Vector3 center, float radius)
        {
            List<int> result = new List<int>();
            QueryRangeRecursive(root, center, radius, result);
            return result;
        }

        /// <summary>
        /// 재귀적으로 반경 내 노드를 검색합니다.
        /// </summary>
        private void QueryRangeRecursive(QuadtreeNode node, Vector3 center, float radius, List<int> result)
        {
            if (node == null)
            {
                return;
            }

            // 경계와 검색 범위가 겹치는지 확인
            if (!BoundsIntersectsSphere(node.bounds, center, radius))
            {
                return;
            }

            // 리프 노드인 경우 노드들을 확인
            if (node.isLeaf)
            {
                foreach (var n in node.nodes)
                {
                    float distance = Vector3.Distance(n.position, center);
                    if (distance <= radius)
                    {
                        result.Add(n.nodeId);
                    }
                }
            }
            else
            {
                // 자식 노드들을 재귀적으로 검색
                if (node.children != null)
                {
                    foreach (var child in node.children)
                    {
                        QueryRangeRecursive(child, center, radius, result);
                    }
                }
            }
        }

        /// <summary>
        /// 경계와 구가 겹치는지 확인합니다.
        /// </summary>
        private bool BoundsIntersectsSphere(Bounds bounds, Vector3 center, float radius)
        {
            // 경계 내에서 구의 중심에 가장 가까운 점을 찾습니다
            Vector3 closestPoint = bounds.ClosestPoint(center);
            float distance = Vector3.Distance(closestPoint, center);
            return distance <= radius;
        }

        /// <summary>
        /// 특정 위치에서 가장 가까운 노드를 찾습니다.
        /// </summary>
        /// <param name="position">검색 위치</param>
        /// <returns>가장 가까운 노드 ID</returns>
        public int FindNearest(Vector3 position)
        {
            int nearestNodeId = -1;
            float nearestDistance = float.MaxValue;
            FindNearestRecursive(root, position, ref nearestNodeId, ref nearestDistance);
            return nearestNodeId;
        }

        /// <summary>
        /// 재귀적으로 가장 가까운 노드를 찾습니다.
        /// </summary>
        private void FindNearestRecursive(QuadtreeNode node, Vector3 position, ref int nearestNodeId, ref float nearestDistance)
        {
            if (node == null)
            {
                return;
            }

            // 경계까지의 최소 거리가 현재 최근접 거리보다 크면 탐색 중단
            Vector3 closestPoint = node.bounds.ClosestPoint(position);
            float minPossibleDistance = Vector3.Distance(closestPoint, position);
            
            if (minPossibleDistance > nearestDistance)
            {
                return;
            }

            // 리프 노드인 경우 노드들을 확인
            if (node.isLeaf)
            {
                foreach (var n in node.nodes)
                {
                    float distance = Vector3.Distance(n.position, position);
                    if (distance < nearestDistance)
                    {
                        nearestDistance = distance;
                        nearestNodeId = n.nodeId;
                    }
                }
            }
            else
            {
                // 자식 노드들을 거리 순으로 정렬하여 탐색
                if (node.children != null)
                {
                    // 각 자식 노드까지의 거리 계산
                    List<(int index, float distance)> childDistances = new List<(int, float)>();
                    for (int i = 0; i < node.children.Length; i++)
                    {
                        Vector3 childClosestPoint = node.children[i].bounds.ClosestPoint(position);
                        float dist = Vector3.Distance(childClosestPoint, position);
                        childDistances.Add((i, dist));
                    }

                    // 거리 순으로 정렬
                    childDistances.Sort((a, b) => a.distance.CompareTo(b.distance));

                    // 가까운 순서대로 탐색
                    foreach (var (index, _) in childDistances)
                    {
                        FindNearestRecursive(node.children[index], position, ref nearestNodeId, ref nearestDistance);
                    }
                }
            }
        }

        /// <summary>
        /// Quadtree를 초기화합니다.
        /// </summary>
        public void Clear()
        {
            root = new QuadtreeNode(worldBounds);
        }

        /// <summary>
        /// Quadtree의 내부 노드 클래스
        /// 공간을 4개의 사분면으로 분할하여 노드를 저장합니다.
        /// </summary>
        internal class QuadtreeNode
        {
            /// <summary>
            /// 이 노드가 담당하는 공간 경계
            /// </summary>
            public Bounds bounds;

            /// <summary>
            /// 이 노드에 저장된 (nodeId, position) 쌍의 리스트
            /// </summary>
            public List<(int nodeId, Vector3 position)> nodes;

            /// <summary>
            /// 자식 노드 배열 (NW, NE, SW, SE 순서)
            /// null이면 리프 노드입니다.
            /// </summary>
            public QuadtreeNode[] children;

            /// <summary>
            /// 리프 노드 여부
            /// </summary>
            public bool isLeaf;

            /// <summary>
            /// QuadtreeNode 생성자
            /// </summary>
            /// <param name="bounds">이 노드가 담당할 공간 경계</param>
            public QuadtreeNode(Bounds bounds)
            {
                this.bounds = bounds;
                this.nodes = new List<(int nodeId, Vector3 position)>();
                this.children = null;
                this.isLeaf = true;
            }

            /// <summary>
            /// 노드를 4개의 사분면으로 분할합니다.
            /// NW (북서), NE (북동), SW (남서), SE (남동) 순서로 자식 노드를 생성합니다.
            /// </summary>
            public void Split()
            {
                // 이미 분할되었으면 무시
                if (!isLeaf)
                {
                    return;
                }

                // 현재 경계의 중심과 크기 계산
                Vector3 center = bounds.center;
                Vector3 size = bounds.size;
                Vector3 quarterSize = size / 2f;

                // 4개의 자식 노드 생성
                children = new QuadtreeNode[4];

                // NW (북서): x-, z+
                Vector3 nwCenter = center + new Vector3(-quarterSize.x / 2f, 0, quarterSize.z / 2f);
                children[0] = new QuadtreeNode(new Bounds(nwCenter, quarterSize));

                // NE (북동): x+, z+
                Vector3 neCenter = center + new Vector3(quarterSize.x / 2f, 0, quarterSize.z / 2f);
                children[1] = new QuadtreeNode(new Bounds(neCenter, quarterSize));

                // SW (남서): x-, z-
                Vector3 swCenter = center + new Vector3(-quarterSize.x / 2f, 0, -quarterSize.z / 2f);
                children[2] = new QuadtreeNode(new Bounds(swCenter, quarterSize));

                // SE (남동): x+, z-
                Vector3 seCenter = center + new Vector3(quarterSize.x / 2f, 0, -quarterSize.z / 2f);
                children[3] = new QuadtreeNode(new Bounds(seCenter, quarterSize));

                // 리프 노드가 아님을 표시
                isLeaf = false;

                // 기존 노드들을 자식 노드로 재분배
                List<(int nodeId, Vector3 position)> oldNodes = new List<(int nodeId, Vector3 position)>(nodes);
                nodes.Clear();

                foreach (var node in oldNodes)
                {
                    // 각 노드를 적절한 자식 노드에 할당
                    int childIndex = GetChildIndex(node.position);
                    if (childIndex >= 0 && childIndex < 4)
                    {
                        children[childIndex].nodes.Add(node);
                    }
                }
            }

            /// <summary>
            /// 주어진 위치가 속하는 자식 노드의 인덱스를 반환합니다.
            /// </summary>
            /// <param name="position">위치</param>
            /// <returns>자식 노드 인덱스 (0: NW, 1: NE, 2: SW, 3: SE)</returns>
            private int GetChildIndex(Vector3 position)
            {
                Vector3 center = bounds.center;

                // x축 기준: 왼쪽(0) 또는 오른쪽(1)
                bool isRight = position.x >= center.x;
                // z축 기준: 위쪽(0) 또는 아래쪽(1)
                bool isBottom = position.z < center.z;

                // 인덱스 계산
                // NW (0): !isRight && !isBottom
                // NE (1): isRight && !isBottom
                // SW (2): !isRight && isBottom
                // SE (3): isRight && isBottom
                if (!isRight && !isBottom) return 0; // NW
                if (isRight && !isBottom) return 1;  // NE
                if (!isRight && isBottom) return 2;  // SW
                if (isRight && isBottom) return 3;   // SE

                return -1; // 오류 (발생하지 않아야 함)
            }

            /// <summary>
            /// 주어진 위치가 속하는 자식 노드의 인덱스를 반환합니다 (공개 메서드).
            /// </summary>
            /// <param name="position">위치</param>
            /// <returns>자식 노드 인덱스 (0: NW, 1: NE, 2: SW, 3: SE)</returns>
            public int GetChildIndexPublic(Vector3 position)
            {
                return GetChildIndex(position);
            }
        }
    }
}
