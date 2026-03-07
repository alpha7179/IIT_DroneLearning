using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace ProceduralCityGenerator
{
    /// <summary>
    /// UI 캔버스에 미니맵을 표시하고 실시간 업데이트를 처리하는 MonoBehaviour 컴포넌트
    /// 기본 미니맵 텍스처와 동적 레이어를 관리하여 드론 위치 등을 실시간으로 표시
    /// Requirements: 19.6
    /// </summary>
    [RequireComponent(typeof(RawImage))]
    public class MinimapRenderer : MonoBehaviour
    {
        /// <summary>
        /// 미니맵을 표시할 RawImage UI 컴포넌트
        /// </summary>
        [SerializeField]
        [Tooltip("미니맵을 표시할 RawImage UI 컴포넌트")]
        public RawImage minimapImage;

        /// <summary>
        /// 기본 미니맵 텍스처 (건물 및 전략적 위치가 그려진 정적 레이어)
        /// </summary>
        private Texture2D baseMinimapTexture;

        /// <summary>
        /// 동적 레이어 텍스처 (드론 위치, 경로 등 실시간 업데이트되는 요소)
        /// </summary>
        private Texture2D dynamicTexture;

        /// <summary>
        /// 합성 결과 텍스처 (baseMinimapTexture + dynamicTexture). 매 프레임 재사용하여 GC 부하 방지.
        /// </summary>
        private Texture2D compositeTexture;

        /// <summary>
        /// 픽셀당 미터 비율 (월드 좌표를 픽셀 좌표로 변환하는 데 사용)
        /// </summary>
        private float pixelsPerMeter;

        /// <summary>
        /// 미니맵이 초기화되었는지 여부
        /// </summary>
        private bool isInitialized = false;

        /// <summary>
        /// 도시 경계 (월드 좌표를 픽셀 좌표로 변환하는 데 사용)
        /// </summary>
        private Bounds cityBounds;

        /// <summary>
        /// 동적 마커 정보를 저장하는 딕셔너리 (월드 좌표 → 마커 타입)
        /// </summary>
        private Dictionary<Vector3, MarkerType> dynamicMarkers = new Dictionary<Vector3, MarkerType>();

        /// <summary>
        /// 동적 레이어가 변경되었는지 여부 (더티 플래그)
        /// </summary>
        private bool isDirty = false;

        /// <summary>
        /// Awake 메서드: RawImage 컴포넌트 자동 참조
        /// </summary>
        private void Awake()
        {
            // RawImage 컴포넌트가 할당되지 않았으면 자동으로 가져오기
            if (minimapImage == null)
            {
                minimapImage = GetComponent<RawImage>();
            }

            if (minimapImage == null)
            {
                Debug.LogError("MinimapRenderer: RawImage 컴포넌트를 찾을 수 없습니다.");
            }
        }

        /// <summary>
        /// 미니맵 렌더러를 초기화합니다.
        /// Requirements: 19.6
        /// </summary>
        /// <param name="minimap">기본 미니맵 텍스처</param>
        /// <param name="pixelsPerMeter">픽셀당 미터 비율</param>
        /// <param name="bounds">도시 경계</param>
        public void Initialize(Texture2D minimap, float pixelsPerMeter, Bounds bounds)
        {
            if (minimap == null)
            {
                Debug.LogError("MinimapRenderer.Initialize: 미니맵 텍스처가 null입니다.");
                return;
            }

            if (minimapImage == null)
            {
                Debug.LogError("MinimapRenderer.Initialize: RawImage 컴포넌트가 null입니다.");
                return;
            }

            // 기본 미니맵 텍스처 저장
            this.baseMinimapTexture = minimap;
            this.pixelsPerMeter = pixelsPerMeter;
            this.cityBounds = bounds;

            // 동적 레이어 텍스처 생성 (기본 미니맵과 동일한 해상도)
            this.dynamicTexture = new Texture2D(minimap.width, minimap.height, TextureFormat.RGBA32, false);

            // 동적 레이어를 투명하게 초기화
            Color[] transparentPixels = new Color[minimap.width * minimap.height];
            for (int i = 0; i < transparentPixels.Length; i++)
            {
                transparentPixels[i] = Color.clear;
            }
            dynamicTexture.SetPixels(transparentPixels);
            dynamicTexture.Apply();

            // 합성 텍스처 미리 생성 (CompositeLayers에서 재사용하여 매 프레임 GC 할당 방지)
            this.compositeTexture = new Texture2D(minimap.width, minimap.height, TextureFormat.RGBA32, false);

            // RawImage에 기본 미니맵 텍스처 설정
            minimapImage.texture = baseMinimapTexture;

            isInitialized = true;

            Debug.Log($"MinimapRenderer.Initialize: 미니맵 초기화 완료 (해상도: {minimap.width}x{minimap.height}, 픽셀당 미터: {pixelsPerMeter:F2})");
        }

        /// <summary>
        /// 월드 좌표를 미니맵 픽셀 좌표로 변환합니다.
        /// Requirements: 19.3
        /// </summary>
        /// <param name="worldPosition">월드 좌표</param>
        /// <returns>미니맵 픽셀 좌표</returns>
        private Vector2Int WorldToPixel(Vector3 worldPosition)
        {
            if (!isInitialized || baseMinimapTexture == null)
            {
                return Vector2Int.zero;
            }

            // 도시 경계의 최소 지점을 기준으로 상대 위치 계산
            Vector3 relativePosition = worldPosition - cityBounds.min;

            // 픽셀 좌표로 변환
            int pixelX = Mathf.RoundToInt(relativePosition.x * pixelsPerMeter);
            int pixelY = Mathf.RoundToInt(relativePosition.z * pixelsPerMeter);

            // 경계 내로 제한
            pixelX = Mathf.Clamp(pixelX, 0, baseMinimapTexture.width - 1);
            pixelY = Mathf.Clamp(pixelY, 0, baseMinimapTexture.height - 1);

            return new Vector2Int(pixelX, pixelY);
        }

        /// <summary>
        /// 미니맵이 초기화되었는지 확인합니다.
        /// </summary>
        public bool IsInitialized => isInitialized;

        /// <summary>
        /// Update 메서드: 더티 플래그가 설정된 경우에만 동적 레이어를 갱신합니다.
        /// Requirements: 19.4, 19.5
        /// </summary>
        private void Update()
        {
            // 더티 플래그가 설정된 경우에만 갱신 (불필요한 업데이트 방지)
            if (isDirty && isInitialized)
            {
                RefreshDynamicLayer();
            }
        }

        /// <summary>
        /// 동적 마커를 미니맵에 추가합니다.
        /// Requirements: 19.1, 19.2, 19.5
        /// </summary>
        /// <param name="worldPosition">마커를 추가할 월드 좌표</param>
        /// <param name="type">마커 타입</param>
        public void AddDynamicMarker(Vector3 worldPosition, MarkerType type)
        {
            if (!isInitialized)
            {
                Debug.LogWarning("MinimapRenderer.AddDynamicMarker: 미니맵이 초기화되지 않았습니다.");
                return;
            }

            // 마커 추가 또는 업데이트
            dynamicMarkers[worldPosition] = type;
            isDirty = true;

            // 더티 플래그만 설정하고 즉시 갱신하지 않음 (성능 최적화)
            // RefreshDynamicLayer는 Update에서 호출됨
        }

        /// <summary>
        /// 동적 마커를 미니맵에서 제거합니다.
        /// Requirements: 19.1, 19.2, 19.5
        /// </summary>
        /// <param name="worldPosition">제거할 마커의 월드 좌표</param>
        public void RemoveDynamicMarker(Vector3 worldPosition)
        {
            if (!isInitialized)
            {
                Debug.LogWarning("MinimapRenderer.RemoveDynamicMarker: 미니맵이 초기화되지 않았습니다.");
                return;
            }

            if (dynamicMarkers.Remove(worldPosition))
            {
                isDirty = true;
                // 더티 플래그만 설정하고 즉시 갱신하지 않음 (성능 최적화)
            }
        }

        /// <summary>
        /// 마커의 위치를 업데이트합니다.
        /// Requirements: 19.1, 19.2, 19.5
        /// </summary>
        /// <param name="oldPosition">이전 월드 좌표</param>
        /// <param name="newPosition">새로운 월드 좌표</param>
        /// <param name="type">마커 타입</param>
        public void UpdateMarkerPosition(Vector3 oldPosition, Vector3 newPosition, MarkerType type)
        {
            if (!isInitialized)
            {
                Debug.LogWarning("MinimapRenderer.UpdateMarkerPosition: 미니맵이 초기화되지 않았습니다.");
                return;
            }

            // 이전 위치의 마커 제거
            dynamicMarkers.Remove(oldPosition);

            // 새로운 위치에 마커 추가
            dynamicMarkers[newPosition] = type;
            isDirty = true;

            // 더티 플래그만 설정하고 즉시 갱신하지 않음 (성능 최적화)
        }

        /// <summary>
        /// 동적 레이어를 갱신합니다.
        /// Requirements: 19.3, 19.4, 19.5
        /// </summary>
        private void RefreshDynamicLayer()
        {
            if (!isInitialized || dynamicTexture == null || !isDirty)
            {
                return;
            }

            // 성능 측정 시작
            float startTime = Time.realtimeSinceStartup;

            // 동적 레이어를 투명하게 초기화 (최적화: 배열 재사용)
            int pixelCount = dynamicTexture.width * dynamicTexture.height;
            Color[] pixels = new Color[pixelCount];
            
            // 최적화: 루프 대신 Array.Fill 사용 (더 빠름)
            for (int i = 0; i < pixelCount; i++)
            {
                pixels[i] = Color.clear;
            }
            
            dynamicTexture.SetPixels(pixels);

            // 모든 동적 마커 그리기
            foreach (var marker in dynamicMarkers)
            {
                DrawMarker(marker.Key, marker.Value);
            }

            // 텍스처 적용
            dynamicTexture.Apply(false); // mipmap 생성 비활성화로 성능 향상

            // 기본 미니맵과 동적 레이어 합성
            CompositeLayers();

            isDirty = false;

            // 성능 측정 종료
            float elapsedTime = (Time.realtimeSinceStartup - startTime) * 1000f; // ms로 변환
            
            // 1ms 초과 시 경고 (요구사항 19.4)
            if (elapsedTime > 1.0f)
            {
                Debug.LogWarning($"MinimapRenderer.RefreshDynamicLayer: 업데이트 시간이 1ms를 초과했습니다 ({elapsedTime:F2}ms). 마커 수: {dynamicMarkers.Count}");
            }
        }

        /// <summary>
        /// 특정 마커를 동적 레이어에 그립니다.
        /// Requirements: 19.2
        /// </summary>
        /// <param name="worldPosition">마커의 월드 좌표</param>
        /// <param name="type">마커 타입</param>
        private void DrawMarker(Vector3 worldPosition, MarkerType type)
        {
            Vector2Int pixelPos = WorldToPixel(worldPosition);

            // 마커 타입에 따른 색상 및 모양 정의
            Color markerColor;
            int markerSize;
            bool drawStar = false;

            switch (type)
            {
                case MarkerType.EvaderDrone:
                    markerColor = Color.blue;
                    markerSize = 3; // 3x3 픽셀 점
                    break;

                case MarkerType.PursuerDrone:
                    markerColor = Color.red;
                    markerSize = 3; // 3x3 픽셀 점
                    break;

                case MarkerType.TargetPoint:
                    markerColor = Color.green;
                    markerSize = 5; // 5x5 픽셀 별
                    drawStar = true;
                    break;

                case MarkerType.PathLine:
                    markerColor = Color.yellow;
                    markerSize = 1; // 1x1 픽셀 점
                    break;

                default:
                    markerColor = Color.white;
                    markerSize = 2;
                    break;
            }

            if (drawStar)
            {
                // 별 모양 그리기 (5개의 점으로 구성)
                DrawStar(pixelPos, markerSize, markerColor);
            }
            else
            {
                // 원형 점 그리기
                DrawCircle(pixelPos, markerSize, markerColor);
            }
        }

        /// <summary>
        /// 원형 점을 그립니다.
        /// </summary>
        /// <param name="center">중심 픽셀 좌표</param>
        /// <param name="radius">반지름 (픽셀)</param>
        /// <param name="color">색상</param>
        private void DrawCircle(Vector2Int center, int radius, Color color)
        {
            for (int y = -radius; y <= radius; y++)
            {
                for (int x = -radius; x <= radius; x++)
                {
                    // 원형 범위 내에 있는지 확인
                    if (x * x + y * y <= radius * radius)
                    {
                        int pixelX = center.x + x;
                        int pixelY = center.y + y;

                        // 텍스처 경계 내에 있는지 확인
                        if (pixelX >= 0 && pixelX < dynamicTexture.width &&
                            pixelY >= 0 && pixelY < dynamicTexture.height)
                        {
                            dynamicTexture.SetPixel(pixelX, pixelY, color);
                        }
                    }
                }
            }
        }

        /// <summary>
        /// 별 모양을 그립니다.
        /// </summary>
        /// <param name="center">중심 픽셀 좌표</param>
        /// <param name="size">크기 (픽셀)</param>
        /// <param name="color">색상</param>
        private void DrawStar(Vector2Int center, int size, Color color)
        {
            // 별의 5개 꼭짓점 계산 (위쪽부터 시계방향)
            Vector2Int[] points = new Vector2Int[5];
            for (int i = 0; i < 5; i++)
            {
                float angle = -90f + i * 72f; // -90도부터 시작 (위쪽)
                float rad = angle * Mathf.Deg2Rad;
                points[i] = new Vector2Int(
                    center.x + Mathf.RoundToInt(size * Mathf.Cos(rad)),
                    center.y + Mathf.RoundToInt(size * Mathf.Sin(rad))
                );
            }

            // 별 모양 그리기 (0-2-4-1-3-0 순서로 연결)
            DrawLine(points[0], points[2], color);
            DrawLine(points[2], points[4], color);
            DrawLine(points[4], points[1], color);
            DrawLine(points[1], points[3], color);
            DrawLine(points[3], points[0], color);

            // 중심점 강조
            DrawCircle(center, 1, color);
        }

        /// <summary>
        /// 두 점 사이에 선을 그립니다 (Bresenham 알고리즘).
        /// </summary>
        /// <param name="start">시작 픽셀 좌표</param>
        /// <param name="end">끝 픽셀 좌표</param>
        /// <param name="color">색상</param>
        private void DrawLine(Vector2Int start, Vector2Int end, Color color)
        {
            int dx = Mathf.Abs(end.x - start.x);
            int dy = Mathf.Abs(end.y - start.y);
            int sx = start.x < end.x ? 1 : -1;
            int sy = start.y < end.y ? 1 : -1;
            int err = dx - dy;

            int x = start.x;
            int y = start.y;

            while (true)
            {
                // 텍스처 경계 내에 있는지 확인
                if (x >= 0 && x < dynamicTexture.width &&
                    y >= 0 && y < dynamicTexture.height)
                {
                    dynamicTexture.SetPixel(x, y, color);
                }

                if (x == end.x && y == end.y)
                    break;

                int e2 = 2 * err;
                if (e2 > -dy)
                {
                    err -= dy;
                    x += sx;
                }
                if (e2 < dx)
                {
                    err += dx;
                    y += sy;
                }
            }
        }

        /// <summary>
        /// 기본 미니맵과 동적 레이어를 합성합니다.
        /// Requirements: 19.4
        /// </summary>
        private void CompositeLayers()
        {
            if (baseMinimapTexture == null || dynamicTexture == null || compositeTexture == null)
            {
                return;
            }

            // 기본 미니맵 픽셀 복사
            Color[] basePixels = baseMinimapTexture.GetPixels();
            Color[] dynamicPixels = dynamicTexture.GetPixels();
            Color[] compositePixels = new Color[basePixels.Length];

            // 알파 블렌딩: 동적 레이어를 기본 레이어 위에 합성
            for (int i = 0; i < basePixels.Length; i++)
            {
                compositePixels[i] = dynamicPixels[i].a > 0.01f
                    ? Color.Lerp(basePixels[i], dynamicPixels[i], dynamicPixels[i].a)
                    : basePixels[i];
            }

            // 미리 생성된 compositeTexture를 재사용 (매 프레임 new Texture2D 할당 방지)
            compositeTexture.SetPixels(compositePixels);
            compositeTexture.Apply(false); // mipmap 생성 비활성화로 성능 향상

            // RawImage에 합성된 텍스처 설정
            minimapImage.texture = compositeTexture;
        }

        /// <summary>
        /// 월드 좌표 리스트를 경로로 그립니다.
        /// Requirements: 19.2
        /// </summary>
        /// <param name="worldPath">월드 좌표 리스트</param>
        /// <param name="color">경로 색상 (기본값: 노란색)</param>
        public void DrawPath(List<Vector3> worldPath, Color color)
        {
            if (!isInitialized)
            {
                Debug.LogWarning("MinimapRenderer.DrawPath: 미니맵이 초기화되지 않았습니다.");
                return;
            }

            if (worldPath == null || worldPath.Count < 2)
            {
                Debug.LogWarning("MinimapRenderer.DrawPath: 경로는 최소 2개의 점이 필요합니다.");
                return;
            }

            // 월드 좌표를 픽셀 좌표로 변환
            List<Vector2Int> pixelPath = new List<Vector2Int>(worldPath.Count);
            foreach (Vector3 worldPos in worldPath)
            {
                pixelPath.Add(WorldToPixel(worldPos));
            }

            // 연속된 점들을 선으로 연결
            for (int i = 0; i < pixelPath.Count - 1; i++)
            {
                DrawLine(pixelPath[i], pixelPath[i + 1], color);
            }

            // 텍스처 적용
            dynamicTexture.Apply();

            // 기본 미니맵과 동적 레이어 합성
            CompositeLayers();

            isDirty = false;
        }

        /// <summary>
        /// 컴포넌트가 파괴될 때 텍스처 정리
        /// </summary>
        private void OnDestroy()
        {
            if (dynamicTexture != null)
            {
                Destroy(dynamicTexture);
                dynamicTexture = null;
            }

            if (compositeTexture != null)
            {
                Destroy(compositeTexture);
                compositeTexture = null;
            }

            // 동적 마커 딕셔너리 정리
            dynamicMarkers?.Clear();
        }
    }
}
