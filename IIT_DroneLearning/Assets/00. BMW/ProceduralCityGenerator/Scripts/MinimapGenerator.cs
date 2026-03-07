using UnityEngine;

namespace ProceduralCityGenerator
{
    /// <summary>
    /// 등고선 스타일 미니맵을 생성하는 클래스
    /// 건물 높이를 시각적으로 표현하여 AI 에이전트가 환경을 이해할 수 있도록 지원
    /// </summary>
    public class MinimapGenerator
    {
        private readonly int resolution;
        private readonly Bounds cityBounds;
        private readonly float pixelsPerMeter;

        /// <summary>
        /// MinimapGenerator 생성자
        /// </summary>
        /// <param name="resolution">미니맵 해상도 (픽셀 단위)</param>
        /// <param name="bounds">도시 경계 (Bounds)</param>
        public MinimapGenerator(int resolution, Bounds bounds)
        {
            this.resolution = resolution;
            this.cityBounds = bounds;

            // pixelsPerMeter 계산: 해상도를 도시의 최대 차원으로 나눔
            float maxDimension = Mathf.Max(bounds.size.x, bounds.size.z);
            this.pixelsPerMeter = resolution / maxDimension;
        }

        /// <summary>
        /// 미니맵 해상도를 반환
        /// </summary>
        public int Resolution => resolution;

        /// <summary>
        /// 도시 경계를 반환
        /// </summary>
        public Bounds CityBounds => cityBounds;

        /// <summary>
        /// 픽셀당 미터 비율을 반환
        /// </summary>
        public float PixelsPerMeter => pixelsPerMeter;

        /// <summary>
        /// 건물 배열과 도시 그래프를 기반으로 등고선 스타일 미니맵을 생성
        /// </summary>
        /// <param name="buildings">건물 배열</param>
        /// <param name="graph">도시 그래프 (선택적)</param>
        /// <param name="strategicLocations">전략적 위치 리스트 (선택적)</param>
        /// <returns>생성된 미니맵 Texture2D</returns>
        public Texture2D GenerateMinimap(Building[] buildings, CityGraph graph = null, System.Collections.Generic.List<StrategicLocation> strategicLocations = null)
        {
            // Texture2D 생성 (지정된 해상도)
            Texture2D minimap = new Texture2D(resolution, resolution, TextureFormat.RGB24, false);

            // 배경을 밝은 색으로 초기화 (개방 공간)
            Color backgroundColor = new Color(0.9f, 0.9f, 0.9f); // 밝은 회색
            Color[] pixels = new Color[resolution * resolution];
            for (int i = 0; i < pixels.Length; i++)
            {
                pixels[i] = backgroundColor;
            }
            minimap.SetPixels(pixels);

            // 건물이 없으면 빈 미니맵 반환
            if (buildings == null || buildings.Length == 0)
            {
                minimap.Apply();
                return minimap;
            }

            // 건물 높이 범위 계산
            float minHeight = float.MaxValue;
            float maxHeight = float.MinValue;
            foreach (var building in buildings)
            {
                if (building.height < minHeight) minHeight = building.height;
                if (building.height > maxHeight) maxHeight = building.height;
            }

            // 건물을 미니맵에 그리기
            foreach (var building in buildings)
            {
                DrawBuilding(minimap, building, minHeight, maxHeight);
            }

            // 전략적 위치 마커 그리기
            if (strategicLocations != null && strategicLocations.Count > 0)
            {
                DrawStrategicMarkers(minimap, strategicLocations);
            }

            // 텍스처 적용
            minimap.Apply();
            return minimap;
        }

        /// <summary>
        /// 건물을 미니맵에 그리는 내부 메서드
        /// </summary>
        /// <param name="minimap">미니맵 텍스처</param>
        /// <param name="building">건물 정보</param>
        /// <param name="minHeight">최소 건물 높이</param>
        /// <param name="maxHeight">최대 건물 높이</param>
        private void DrawBuilding(Texture2D minimap, Building building, float minHeight, float maxHeight)
        {
            // 건물의 월드 좌표를 픽셀 좌표로 변환
            Vector2Int centerPixel = WorldToMinimapPixel(building.position);

            // 건물 크기를 픽셀 단위로 변환
            int pixelWidth = Mathf.Max(1, Mathf.RoundToInt(building.size.x * pixelsPerMeter));
            int pixelDepth = Mathf.Max(1, Mathf.RoundToInt(building.size.z * pixelsPerMeter));

            // 건물 높이에 따른 등고선 스타일 색상 계산
            Color buildingColor = GetHeightColor(building.height, minHeight, maxHeight);

            // 건물 영역의 픽셀 범위 계산
            int startX = centerPixel.x - pixelWidth / 2;
            int endX = centerPixel.x + pixelWidth / 2;
            int startY = centerPixel.y - pixelDepth / 2;
            int endY = centerPixel.y + pixelDepth / 2;

            // 픽셀 그리기
            for (int x = startX; x <= endX; x++)
            {
                for (int y = startY; y <= endY; y++)
                {
                    // 경계 체크
                    if (x >= 0 && x < resolution && y >= 0 && y < resolution)
                    {
                        minimap.SetPixel(x, y, buildingColor);
                    }
                }
            }
        }

        /// <summary>
        /// 건물 높이에 따른 등고선 스타일 색상을 계산
        /// </summary>
        /// <param name="height">건물 높이</param>
        /// <param name="minHeight">최소 건물 높이</param>
        /// <param name="maxHeight">최대 건물 높이</param>
        /// <returns>등고선 스타일 색상</returns>
        private Color GetHeightColor(float height, float minHeight, float maxHeight)
        {
            // 높이를 0~1 범위로 정규화
            float normalizedHeight = 0f;
            if (maxHeight > minHeight)
            {
                normalizedHeight = (height - minHeight) / (maxHeight - minHeight);
            }

            // 등고선 스타일: 높을수록 어두운 색
            // 낮은 건물: 연한 회색 (0.7)
            // 중간 건물: 중간 회색 (0.4)
            // 높은 건물: 진한 회색/검은색 (0.1)
            float grayValue = Mathf.Lerp(0.7f, 0.1f, normalizedHeight);
            return new Color(grayValue, grayValue, grayValue);
        }

        /// <summary>
        /// 월드 좌표를 미니맵 픽셀 좌표로 변환
        /// </summary>
        /// <param name="worldPosition">월드 좌표</param>
        /// <returns>미니맵 픽셀 좌표</returns>
        private Vector2Int WorldToMinimapPixel(Vector3 worldPosition)
        {
            // 도시 경계의 최소 지점을 기준으로 상대 위치 계산
            Vector3 relativePosition = worldPosition - cityBounds.min;

            // 픽셀 좌표로 변환
            int pixelX = Mathf.RoundToInt(relativePosition.x * pixelsPerMeter);
            int pixelY = Mathf.RoundToInt(relativePosition.z * pixelsPerMeter);

            // 경계 내로 제한
            pixelX = Mathf.Clamp(pixelX, 0, resolution - 1);
            pixelY = Mathf.Clamp(pixelY, 0, resolution - 1);

            return new Vector2Int(pixelX, pixelY);
        }

        /// <summary>
        /// 격자 구조를 선으로 미니맵에 표시합니다.
        /// Requirements: 16.2
        /// </summary>
        /// <param name="minimap">미니맵 텍스처</param>
        /// <param name="gridSpacing">격자 간격 (미터 단위)</param>
        public void DrawGridLines(Texture2D minimap, float gridSpacing)
        {
            if (minimap == null || gridSpacing <= 0)
            {
                return;
            }

            // 격자선 색상 (연한 회색, 반투명)
            Color gridLineColor = new Color(0.6f, 0.6f, 0.6f, 0.5f);

            // 격자 간격을 픽셀 단위로 변환
            int pixelSpacing = Mathf.Max(1, Mathf.RoundToInt(gridSpacing * pixelsPerMeter));

            // 수직 격자선 그리기 (X축)
            for (int x = 0; x < resolution; x += pixelSpacing)
            {
                for (int y = 0; y < resolution; y++)
                {
                    if (x >= 0 && x < resolution && y >= 0 && y < resolution)
                    {
                        // 기존 픽셀 색상과 블렌딩
                        Color existingColor = minimap.GetPixel(x, y);
                        Color blendedColor = Color.Lerp(existingColor, gridLineColor, 0.3f);
                        minimap.SetPixel(x, y, blendedColor);
                    }
                }
            }

            // 수평 격자선 그리기 (Z축)
            for (int y = 0; y < resolution; y += pixelSpacing)
            {
                for (int x = 0; x < resolution; x++)
                {
                    if (x >= 0 && x < resolution && y >= 0 && y < resolution)
                    {
                        // 기존 픽셀 색상과 블렌딩
                        Color existingColor = minimap.GetPixel(x, y);
                        Color blendedColor = Color.Lerp(existingColor, gridLineColor, 0.3f);
                        minimap.SetPixel(x, y, blendedColor);
                    }
                }
            }
        }

        /// <summary>
        /// 전략적 위치 마커를 미니맵에 그립니다.
        /// Requirements: 17.5
        /// </summary>
        /// <param name="minimap">미니맵 텍스처</param>
        /// <param name="strategicLocations">전략적 위치 리스트</param>
        private void DrawStrategicMarkers(Texture2D minimap, System.Collections.Generic.List<StrategicLocation> strategicLocations)
        {
            foreach (var location in strategicLocations)
            {
                // 전략적 위치 타입에 따른 색상 가져오기
                Color markerColor = GetStrategyTypeColor(location.locationType);

                // 월드 좌표를 픽셀 좌표로 변환
                Vector2Int pixelPos = WorldToMinimapPixel(location.position);

                // 마커를 작은 점으로 그리기 (3x3 픽셀)
                int markerSize = 2; // 반경
                for (int dx = -markerSize; dx <= markerSize; dx++)
                {
                    for (int dy = -markerSize; dy <= markerSize; dy++)
                    {
                        int x = pixelPos.x + dx;
                        int y = pixelPos.y + dy;

                        // 경계 체크
                        if (x >= 0 && x < resolution && y >= 0 && y < resolution)
                        {
                            // 원형 마커를 위한 거리 체크
                            float distance = Mathf.Sqrt(dx * dx + dy * dy);
                            if (distance <= markerSize)
                            {
                                minimap.SetPixel(x, y, markerColor);
                            }
                        }
                    }
                }
            }
        }

        /// <summary>
        /// 전략적 위치 타입에 따른 색상을 반환합니다.
        /// Requirements: 17.5
        /// - 은폐 지점: 파란색
        /// - 교차로: 노란색
        /// - 막다른 골목: 빨간색
        /// - 개방 구역: 주황색
        /// - 우회 경로: 초록색
        /// </summary>
        /// <param name="strategyType">전략적 위치 타입</param>
        /// <returns>해당 타입의 색상</returns>
        private Color GetStrategyTypeColor(StrategyType strategyType)
        {
            switch (strategyType)
            {
                case StrategyType.CoverPoint:
                    return Color.blue;        // 은폐 지점: 파란색
                case StrategyType.Intersection:
                    return Color.yellow;      // 교차로: 노란색
                case StrategyType.DeadEnd:
                    return Color.red;         // 막다른 골목: 빨간색
                case StrategyType.OpenArea:
                    return new Color(1f, 0.5f, 0f); // 개방 구역: 주황색
                case StrategyType.DetourPath:
                    return Color.green;       // 우회 경로: 초록색
                default:
                    return Color.white;       // 기본값: 흰색
            }
        }

        /// <summary>
        /// 미니맵을 PNG 파일로 저장합니다.
        /// Requirements: 16.6, 16.7
        /// </summary>
        /// <param name="minimap">저장할 미니맵 텍스처</param>
        /// <param name="seedValue">파일명에 포함할 시드 값 (선택적, -1이면 타임스탬프 사용)</param>
        public void SaveMinimapToPNG(Texture2D minimap, int seedValue = -1)
        {
            if (minimap == null)
            {
                Debug.LogError("SaveMinimapToPNG: 미니맵 텍스처가 null입니다.");
                return;
            }

            // Assets/CityMaps 디렉토리 경로
            string directoryPath = "Assets/CityMaps";

            // 디렉토리가 존재하지 않으면 생성
            if (!System.IO.Directory.Exists(directoryPath))
            {
                System.IO.Directory.CreateDirectory(directoryPath);
                Debug.Log($"SaveMinimapToPNG: 디렉토리 생성됨 - {directoryPath}");
            }

            // 파일명 생성 (타임스탬프 또는 시드 값 포함)
            string filename;
            if (seedValue >= 0)
            {
                // 시드 값을 파일명에 포함
                filename = $"Minimap_Seed{seedValue}_{resolution}x{resolution}.png";
            }
            else
            {
                // 타임스탬프를 파일명에 포함
                string timestamp = System.DateTime.Now.ToString("yyyyMMdd_HHmmss");
                filename = $"Minimap_{timestamp}_{resolution}x{resolution}.png";
            }

            // 전체 파일 경로
            string filePath = System.IO.Path.Combine(directoryPath, filename);

            // 텍스처를 PNG로 인코딩
            byte[] pngData = ImageConversion.EncodeToPNG(minimap);

            if (pngData == null || pngData.Length == 0)
            {
                Debug.LogError("SaveMinimapToPNG: PNG 인코딩 실패");
                return;
            }

            // 파일로 저장
            System.IO.File.WriteAllBytes(filePath, pngData);

            Debug.Log($"SaveMinimapToPNG: 미니맵 저장 완료 - {filePath}");

#if UNITY_EDITOR
            // Unity Editor에서 에셋 데이터베이스 새로고침
            UnityEditor.AssetDatabase.Refresh();
#endif
        }
    }
}
