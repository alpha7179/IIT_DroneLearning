using System.Collections.Generic;
using UnityEngine;

namespace ProceduralCityGenerator
{
    /// <summary>
    /// MinimapRenderer 실시간 업데이트 기능 사용 예제
    /// Task 15.4: 미니맵 실시간 업데이트 테스트
    /// 
    /// 이 예제는 다음을 보여줍니다:
    /// 1. 동적 마커 추가/제거 (Requirements 19.1, 19.2)
    /// 2. 경로 표시 (Requirement 19.2)
    /// 3. 마커 위치 업데이트 (Requirements 19.3, 19.4)
    /// 4. 성능 최적화 (Requirement 19.5)
    /// </summary>
    public class MinimapRealtimeUpdateExample : MonoBehaviour
    {
        [Header("References")]
        [Tooltip("MinimapRenderer 컴포넌트 참조")]
        public MinimapRenderer minimapRenderer;

        [Header("Simulation Settings")]
        [Tooltip("도망자 드론 위치")]
        public Transform evaderDrone;

        [Tooltip("추적자 드론 위치")]
        public Transform pursuerDrone;

        [Tooltip("목표 지점 위치")]
        public Transform targetPoint;

        [Tooltip("경로 업데이트 간격 (초)")]
        [Range(0.1f, 2.0f)]
        public float updateInterval = 0.5f;

        [Tooltip("경로 표시 활성화")]
        public bool showPath = true;

        [Tooltip("경로 색상")]
        public Color pathColor = Color.yellow;

        // 내부 상태
        private Vector3 lastEvaderPosition;
        private Vector3 lastPursuerPosition;
        private float timeSinceLastUpdate;
        private List<Vector3> evaderPath = new List<Vector3>();

        /// <summary>
        /// 초기화
        /// </summary>
        private void Start()
        {
            if (minimapRenderer == null)
            {
                Debug.LogError("MinimapRealtimeUpdateExample: MinimapRenderer가 할당되지 않았습니다.");
                return;
            }

            if (!minimapRenderer.IsInitialized)
            {
                Debug.LogError("MinimapRealtimeUpdateExample: MinimapRenderer가 초기화되지 않았습니다.");
                return;
            }

            // 초기 마커 추가
            if (evaderDrone != null)
            {
                minimapRenderer.AddDynamicMarker(evaderDrone.position, MarkerType.EvaderDrone);
                lastEvaderPosition = evaderDrone.position;
                evaderPath.Add(evaderDrone.position);
            }

            if (pursuerDrone != null)
            {
                minimapRenderer.AddDynamicMarker(pursuerDrone.position, MarkerType.PursuerDrone);
                lastPursuerPosition = pursuerDrone.position;
            }

            if (targetPoint != null)
            {
                minimapRenderer.AddDynamicMarker(targetPoint.position, MarkerType.TargetPoint);
            }

            Debug.Log("MinimapRealtimeUpdateExample: 초기화 완료");
        }

        /// <summary>
        /// 프레임마다 마커 위치 업데이트
        /// </summary>
        private void Update()
        {
            if (minimapRenderer == null || !minimapRenderer.IsInitialized)
            {
                return;
            }

            timeSinceLastUpdate += Time.deltaTime;

            // 지정된 간격마다 업데이트
            if (timeSinceLastUpdate >= updateInterval)
            {
                UpdateMarkers();
                timeSinceLastUpdate = 0f;
            }
        }

        /// <summary>
        /// 마커 위치 업데이트
        /// </summary>
        private void UpdateMarkers()
        {
            // 도망자 드론 위치 업데이트
            if (evaderDrone != null)
            {
                Vector3 currentPosition = evaderDrone.position;
                
                // 위치가 변경되었으면 업데이트
                if (Vector3.Distance(currentPosition, lastEvaderPosition) > 0.1f)
                {
                    minimapRenderer.UpdateMarkerPosition(
                        lastEvaderPosition,
                        currentPosition,
                        MarkerType.EvaderDrone
                    );

                    // 경로에 추가
                    evaderPath.Add(currentPosition);

                    // 경로가 너무 길면 오래된 점 제거 (최대 50개 점)
                    if (evaderPath.Count > 50)
                    {
                        evaderPath.RemoveAt(0);
                    }

                    lastEvaderPosition = currentPosition;

                    // 경로 표시
                    if (showPath && evaderPath.Count >= 2)
                    {
                        minimapRenderer.DrawPath(evaderPath, pathColor);
                    }
                }
            }

            // 추적자 드론 위치 업데이트
            if (pursuerDrone != null)
            {
                Vector3 currentPosition = pursuerDrone.position;
                
                // 위치가 변경되었으면 업데이트
                if (Vector3.Distance(currentPosition, lastPursuerPosition) > 0.1f)
                {
                    minimapRenderer.UpdateMarkerPosition(
                        lastPursuerPosition,
                        currentPosition,
                        MarkerType.PursuerDrone
                    );

                    lastPursuerPosition = currentPosition;
                }
            }
        }

        /// <summary>
        /// 경로 초기화
        /// </summary>
        public void ClearPath()
        {
            evaderPath.Clear();
            
            if (evaderDrone != null)
            {
                evaderPath.Add(evaderDrone.position);
            }

            Debug.Log("MinimapRealtimeUpdateExample: 경로 초기화 완료");
        }

        /// <summary>
        /// 마커 제거
        /// </summary>
        public void RemoveAllMarkers()
        {
            if (minimapRenderer == null || !minimapRenderer.IsInitialized)
            {
                return;
            }

            if (evaderDrone != null)
            {
                minimapRenderer.RemoveDynamicMarker(lastEvaderPosition);
            }

            if (pursuerDrone != null)
            {
                minimapRenderer.RemoveDynamicMarker(lastPursuerPosition);
            }

            if (targetPoint != null)
            {
                minimapRenderer.RemoveDynamicMarker(targetPoint.position);
            }

            Debug.Log("MinimapRealtimeUpdateExample: 모든 마커 제거 완료");
        }

        /// <summary>
        /// 마커 재추가
        /// </summary>
        public void ReaddMarkers()
        {
            if (minimapRenderer == null || !minimapRenderer.IsInitialized)
            {
                return;
            }

            if (evaderDrone != null)
            {
                minimapRenderer.AddDynamicMarker(evaderDrone.position, MarkerType.EvaderDrone);
                lastEvaderPosition = evaderDrone.position;
            }

            if (pursuerDrone != null)
            {
                minimapRenderer.AddDynamicMarker(pursuerDrone.position, MarkerType.PursuerDrone);
                lastPursuerPosition = pursuerDrone.position;
            }

            if (targetPoint != null)
            {
                minimapRenderer.AddDynamicMarker(targetPoint.position, MarkerType.TargetPoint);
            }

            Debug.Log("MinimapRealtimeUpdateExample: 마커 재추가 완료");
        }

        /// <summary>
        /// 계획된 경로 표시
        /// </summary>
        /// <param name="plannedPath">계획된 경로 (월드 좌표 리스트)</param>
        /// <param name="color">경로 색상</param>
        public void ShowPlannedPath(List<Vector3> plannedPath, Color color)
        {
            if (minimapRenderer == null || !minimapRenderer.IsInitialized)
            {
                return;
            }

            if (plannedPath == null || plannedPath.Count < 2)
            {
                Debug.LogWarning("MinimapRealtimeUpdateExample: 경로는 최소 2개의 점이 필요합니다.");
                return;
            }

            minimapRenderer.DrawPath(plannedPath, color);
            Debug.Log($"MinimapRealtimeUpdateExample: 계획된 경로 표시 완료 ({plannedPath.Count}개 점)");
        }

        /// <summary>
        /// 성능 테스트: 여러 마커 동시 추가
        /// </summary>
        /// <param name="count">추가할 마커 수</param>
        public void PerformanceTest_AddMultipleMarkers(int count)
        {
            if (minimapRenderer == null || !minimapRenderer.IsInitialized)
            {
                return;
            }

            System.Diagnostics.Stopwatch stopwatch = System.Diagnostics.Stopwatch.StartNew();

            for (int i = 0; i < count; i++)
            {
                Vector3 randomPosition = new Vector3(
                    Random.Range(-50f, 50f),
                    0,
                    Random.Range(-50f, 50f)
                );

                minimapRenderer.AddDynamicMarker(randomPosition, MarkerType.EvaderDrone);
            }

            stopwatch.Stop();
            double elapsedMs = stopwatch.Elapsed.TotalMilliseconds;
            double avgPerMarker = elapsedMs / count;

            Debug.Log($"[Performance Test] {count}개 마커 추가: {elapsedMs:F3}ms (평균: {avgPerMarker:F3}ms/마커)");

            if (avgPerMarker > 1.0)
            {
                Debug.LogWarning($"[Performance Warning] 평균 마커 추가 시간이 1ms를 초과했습니다: {avgPerMarker:F3}ms");
            }
        }

        /// <summary>
        /// 성능 테스트: 경로 그리기
        /// </summary>
        /// <param name="pointCount">경로 점 개수</param>
        public void PerformanceTest_DrawPath(int pointCount)
        {
            if (minimapRenderer == null || !minimapRenderer.IsInitialized)
            {
                return;
            }

            // 랜덤 경로 생성
            List<Vector3> path = new List<Vector3>(pointCount);
            for (int i = 0; i < pointCount; i++)
            {
                path.Add(new Vector3(
                    i * 5f,
                    0,
                    Mathf.Sin(i * 0.5f) * 20f
                ));
            }

            System.Diagnostics.Stopwatch stopwatch = System.Diagnostics.Stopwatch.StartNew();
            minimapRenderer.DrawPath(path, Color.cyan);
            stopwatch.Stop();

            double elapsedMs = stopwatch.Elapsed.TotalMilliseconds;

            Debug.Log($"[Performance Test] {pointCount}개 점 경로 그리기: {elapsedMs:F3}ms");

            if (elapsedMs > 1.0)
            {
                Debug.LogWarning($"[Performance Warning] 경로 그리기 시간이 1ms를 초과했습니다: {elapsedMs:F3}ms");
            }
        }

        /// <summary>
        /// OnGUI: 간단한 테스트 UI
        /// </summary>
        private void OnGUI()
        {
            GUILayout.BeginArea(new Rect(10, 10, 300, 400));
            GUILayout.Label("Minimap Realtime Update Example", GUI.skin.box);

            if (GUILayout.Button("Clear Path"))
            {
                ClearPath();
            }

            if (GUILayout.Button("Remove All Markers"))
            {
                RemoveAllMarkers();
            }

            if (GUILayout.Button("Readd Markers"))
            {
                ReaddMarkers();
            }

            GUILayout.Space(10);
            GUILayout.Label("Performance Tests:", GUI.skin.box);

            if (GUILayout.Button("Add 10 Markers"))
            {
                PerformanceTest_AddMultipleMarkers(10);
            }

            if (GUILayout.Button("Add 20 Markers"))
            {
                PerformanceTest_AddMultipleMarkers(20);
            }

            if (GUILayout.Button("Draw Path (15 points)"))
            {
                PerformanceTest_DrawPath(15);
            }

            if (GUILayout.Button("Draw Path (30 points)"))
            {
                PerformanceTest_DrawPath(30);
            }

            GUILayout.Space(10);
            GUILayout.Label($"Update Interval: {updateInterval:F2}s");
            GUILayout.Label($"Evader Path Points: {evaderPath.Count}");

            GUILayout.EndArea();
        }
    }
}
