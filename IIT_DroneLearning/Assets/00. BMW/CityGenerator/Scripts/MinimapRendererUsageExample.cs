using UnityEngine;
using UnityEngine.UI;

namespace ProceduralCityGenerator
{
    /// <summary>
    /// MinimapRenderer 사용 예제
    /// UI 캔버스에 미니맵을 표시하는 방법을 보여줍니다.
    /// Task 10.2: 동적 마커 추가 및 제거 기능 시연
    /// </summary>
    public class MinimapRendererUsageExample : MonoBehaviour
    {
        [Header("References")]
        [SerializeField]
        [Tooltip("도시 생성기 참조")]
        private CityGenerator cityGenerator;

        [SerializeField]
        [Tooltip("미니맵을 표시할 MinimapRenderer 컴포넌트")]
        private MinimapRenderer minimapRenderer;

        [Header("Settings")]
        [SerializeField]
        [Tooltip("미니맵 해상도")]
        private MinimapResolution minimapResolution = MinimapResolution.Resolution512;

        [Header("Dynamic Marker Demo")]
        [SerializeField]
        [Tooltip("동적 마커 데모 활성화")]
        private bool enableDynamicMarkerDemo = true;

        [SerializeField]
        [Tooltip("경로 표시 데모 활성화")]
        private bool enablePathDemo = true;

        [SerializeField]
        [Tooltip("도망자 드론 이동 속도")]
        private float evaderSpeed = 5f;

        [SerializeField]
        [Tooltip("추적자 드론 이동 속도")]
        private float pursuerSpeed = 3f;

        // 동적 마커 위치
        private Vector3 evaderPosition;
        private Vector3 pursuerPosition;
        private Vector3 targetPosition;
        private Bounds cityBounds;

        // 경로 추적용
        private System.Collections.Generic.List<Vector3> evaderPath = new System.Collections.Generic.List<Vector3>();
        private System.Collections.Generic.List<Vector3> pursuerPath = new System.Collections.Generic.List<Vector3>();
        private float pathUpdateInterval = 0.5f; // 0.5초마다 경로 업데이트
        private float lastPathUpdateTime = 0f;

        private void Start()
        {
            // 도시 생성기가 설정되어 있는지 확인
            if (cityGenerator == null)
            {
                Debug.LogError("MinimapRendererUsageExample: CityGenerator가 설정되지 않았습니다.");
                return;
            }

            if (minimapRenderer == null)
            {
                Debug.LogError("MinimapRendererUsageExample: MinimapRenderer가 설정되지 않았습니다.");
                return;
            }

            // 도시 생성
            Debug.Log("MinimapRendererUsageExample: 도시 생성 중...");
            cityGenerator.GenerateCity();

            // 미니맵 생성 및 렌더러 초기화
            InitializeMinimap();

            // 동적 마커 데모 시작
            if (enableDynamicMarkerDemo)
            {
                StartDynamicMarkerDemo();
            }
        }

        /// <summary>
        /// 동적 마커 데모를 시작합니다.
        /// Requirements: 19.1, 19.2
        /// </summary>
        private void StartDynamicMarkerDemo()
        {
            if (!minimapRenderer.IsInitialized)
            {
                Debug.LogWarning("MinimapRendererUsageExample: 미니맵이 초기화되지 않아 동적 마커 데모를 시작할 수 없습니다.");
                return;
            }

            // 초기 위치 설정
            evaderPosition = new Vector3(cityBounds.min.x + 10f, 0, cityBounds.min.z + 10f);
            pursuerPosition = new Vector3(cityBounds.max.x - 10f, 0, cityBounds.max.z - 10f);
            targetPosition = cityBounds.center;

            // 동적 마커 추가
            minimapRenderer.AddDynamicMarker(evaderPosition, MarkerType.EvaderDrone);
            minimapRenderer.AddDynamicMarker(pursuerPosition, MarkerType.PursuerDrone);
            minimapRenderer.AddDynamicMarker(targetPosition, MarkerType.TargetPoint);

            Debug.Log("MinimapRendererUsageExample: 동적 마커 데모 시작");
        }

        /// <summary>
        /// Update 메서드: 동적 마커 위치 업데이트
        /// </summary>
        private void Update()
        {
            if (!enableDynamicMarkerDemo || !minimapRenderer.IsInitialized)
            {
                return;
            }

            // 이전 위치 저장
            Vector3 oldEvaderPosition = evaderPosition;
            Vector3 oldPursuerPosition = pursuerPosition;

            // 도망자 드론: 목표 지점을 향해 이동
            Vector3 evaderDirection = (targetPosition - evaderPosition).normalized;
            evaderPosition += evaderDirection * evaderSpeed * Time.deltaTime;

            // 추적자 드론: 도망자를 향해 이동
            Vector3 pursuerDirection = (evaderPosition - pursuerPosition).normalized;
            pursuerPosition += pursuerDirection * pursuerSpeed * Time.deltaTime;

            // 경계 내로 제한
            evaderPosition = ClampToBounds(evaderPosition);
            pursuerPosition = ClampToBounds(pursuerPosition);

            // 마커 위치 업데이트
            minimapRenderer.UpdateMarkerPosition(oldEvaderPosition, evaderPosition, MarkerType.EvaderDrone);
            minimapRenderer.UpdateMarkerPosition(oldPursuerPosition, pursuerPosition, MarkerType.PursuerDrone);

            // 경로 추적 및 표시
            if (enablePathDemo && Time.time - lastPathUpdateTime >= pathUpdateInterval)
            {
                // 경로에 현재 위치 추가
                evaderPath.Add(evaderPosition);
                pursuerPath.Add(pursuerPosition);

                // 경로가 너무 길어지면 오래된 점 제거 (최대 20개 점 유지)
                if (evaderPath.Count > 20)
                {
                    evaderPath.RemoveAt(0);
                }
                if (pursuerPath.Count > 20)
                {
                    pursuerPath.RemoveAt(0);
                }

                // 경로 그리기 (도망자: 파란색, 추적자: 빨간색)
                if (evaderPath.Count >= 2)
                {
                    minimapRenderer.DrawPath(evaderPath, Color.cyan);
                }
                if (pursuerPath.Count >= 2)
                {
                    minimapRenderer.DrawPath(pursuerPath, new Color(1f, 0.5f, 0.5f)); // 연한 빨간색
                }

                lastPathUpdateTime = Time.time;
            }

            // 도망자가 목표에 도달하면 새로운 목표 설정
            if (Vector3.Distance(evaderPosition, targetPosition) < 5f)
            {
                // 이전 목표 마커 제거
                minimapRenderer.RemoveDynamicMarker(targetPosition);

                // 새로운 랜덤 목표 설정
                targetPosition = new Vector3(
                    Random.Range(cityBounds.min.x, cityBounds.max.x),
                    0,
                    Random.Range(cityBounds.min.z, cityBounds.max.z)
                );

                // 새로운 목표 마커 추가
                minimapRenderer.AddDynamicMarker(targetPosition, MarkerType.TargetPoint);

                Debug.Log($"MinimapRendererUsageExample: 새로운 목표 설정 - {targetPosition}");
            }
        }

        /// <summary>
        /// 위치를 도시 경계 내로 제한합니다.
        /// </summary>
        /// <param name="position">제한할 위치</param>
        /// <returns>제한된 위치</returns>
        private Vector3 ClampToBounds(Vector3 position)
        {
            return new Vector3(
                Mathf.Clamp(position.x, cityBounds.min.x, cityBounds.max.x),
                position.y,
                Mathf.Clamp(position.z, cityBounds.min.z, cityBounds.max.z)
            );
        }

        /// <summary>
        /// 미니맵을 생성하고 렌더러를 초기화합니다.
        /// </summary>
        private void InitializeMinimap()
        {
            // 도시 경계 계산
            cityBounds = CalculateCityBounds();

            // MinimapGenerator 생성
            MinimapGenerator generator = new MinimapGenerator((int)minimapResolution, cityBounds);

            // 미니맵 텍스처 생성
            // Note: 실제 구현에서는 cityGenerator에서 건물 배열과 그래프를 가져와야 합니다.
            // 이 예제에서는 간단히 빈 배열로 시연합니다.
            Building[] buildings = new Building[0]; // 실제로는 cityGenerator.GetBuildings() 같은 메서드 필요
            Texture2D minimapTexture = generator.GenerateMinimap(buildings);

            // MinimapRenderer 초기화
            minimapRenderer.Initialize(minimapTexture, generator.PixelsPerMeter, cityBounds);

            Debug.Log("MinimapRendererUsageExample: 미니맵 초기화 완료");
        }

        /// <summary>
        /// 도시 경계를 계산합니다.
        /// </summary>
        /// <returns>도시 경계 Bounds</returns>
        private Bounds CalculateCityBounds()
        {
            // 실제 구현에서는 cityGenerator에서 경계를 가져와야 합니다.
            // 이 예제에서는 파라미터를 기반으로 간단히 계산합니다.
            
            // 도시 크기 계산 (중간값 사용)
            float cityWidth = (cityGenerator.minWidth + cityGenerator.maxWidth) / 2f;
            float cityDepth = (cityGenerator.minDepth + cityGenerator.maxDepth) / 2f;
            
            // 건물 크기와 간격을 고려한 실제 크기
            float actualWidth = cityWidth * (cityGenerator.buildingWidth + cityGenerator.buildingSpacing) * cityGenerator.unitDistance;
            float actualDepth = cityDepth * (cityGenerator.buildingDepth + cityGenerator.buildingSpacing) * cityGenerator.unitDistance;
            
            // 최대 건물 높이
            float maxHeight = cityGenerator.maxBuildingHeight * cityGenerator.unitDistance;
            
            // 중심점 (원점 기준)
            Vector3 center = new Vector3(actualWidth / 2f, maxHeight / 2f, actualDepth / 2f);
            Vector3 size = new Vector3(actualWidth, maxHeight, actualDepth);
            
            return new Bounds(center, size);
        }

        /// <summary>
        /// Inspector에서 미니맵 재생성 버튼
        /// </summary>
        [ContextMenu("Regenerate Minimap")]
        private void RegenerateMinimap()
        {
            if (cityGenerator != null && minimapRenderer != null)
            {
                cityGenerator.GenerateCity();
                InitializeMinimap();
            }
        }
    }
}
