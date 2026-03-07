using UnityEngine;

namespace ProceduralCityGenerator
{
    /// <summary>
    /// SaveMinimapToPNG 메서드 사용 예제
    /// Requirements: 16.6, 16.7
    /// 
    /// 이 클래스는 MinimapGenerator의 SaveMinimapToPNG 메서드를 사용하는 방법을 보여줍니다.
    /// </summary>
    public class SaveMinimapUsageExample : MonoBehaviour
    {
        /// <summary>
        /// 미니맵을 생성하고 PNG 파일로 저장하는 예제
        /// </summary>
        public void ExampleUsage()
        {
            // 1. 도시 경계 설정
            Bounds cityBounds = new Bounds(Vector3.zero, new Vector3(100, 0, 100));

            // 2. MinimapGenerator 생성 (해상도 512x512)
            MinimapGenerator generator = new MinimapGenerator(512, cityBounds);

            // 3. 건물 배열 준비 (실제로는 CityGenerator에서 생성된 건물 사용)
            GridCell cell1 = new GridCell { x = 2, z = 2, hasBuilding = true, buildingHeight = 20, worldPosition = new Vector3(10, 0, 10) };
            GridCell cell2 = new GridCell { x = 6, z = 6, hasBuilding = true, buildingHeight = 35, worldPosition = new Vector3(30, 0, 30) };
            
            Building[] buildings = new Building[]
            {
                new Building(null, new Vector3(10, 0, 10), new Vector3(5, 20, 5), 20, cell1),
                new Building(null, new Vector3(30, 0, 30), new Vector3(8, 35, 8), 35, cell2)
            };

            // 4. 미니맵 생성
            Texture2D minimap = generator.GenerateMinimap(buildings);

            // 5. 미니맵을 PNG 파일로 저장
            // 옵션 1: 시드 값을 파일명에 포함
            int seedValue = 12345;
            generator.SaveMinimapToPNG(minimap, seedValue);
            // 결과: Assets/CityMaps/Minimap_Seed12345_512x512.png

            // 옵션 2: 타임스탬프를 파일명에 포함 (seedValue를 -1로 설정)
            generator.SaveMinimapToPNG(minimap, -1);
            // 결과: Assets/CityMaps/Minimap_20240115_143022_512x512.png (예시)

            // 6. 메모리 정리
            Destroy(minimap);
        }

        /// <summary>
        /// CityGenerator와 통합하여 사용하는 예제
        /// </summary>
        public void IntegrationExample()
        {
            // CityGenerator에서 도시를 생성한 후 미니맵 저장
            CityGenerator cityGenerator = GetComponent<CityGenerator>();
            
            if (cityGenerator != null)
            {
                // 도시 생성
                CityGenerationResult result = cityGenerator.GenerateCity();

                if (result.success)
                {
                    // 미니맵 생성기 초기화
                    // 주의: 실제 구현에서는 CityGenerator 내부에서 처리해야 함
                    // 여기서는 예제를 위해 간단히 작성
                    
                    Debug.Log($"도시 생성 완료. 건물 수: {result.buildingCount}");
                    Debug.Log("미니맵은 Assets/CityMaps 디렉토리에 저장됩니다.");
                }
            }
        }

        /// <summary>
        /// 전략적 위치 마커를 포함한 미니맵 저장 예제
        /// </summary>
        public void ExampleWithStrategicMarkers()
        {
            // 1. 도시 경계 및 생성기 설정
            Bounds cityBounds = new Bounds(Vector3.zero, new Vector3(100, 0, 100));
            MinimapGenerator generator = new MinimapGenerator(1024, cityBounds);

            // 2. 건물 배열 준비
            Building[] buildings = new Building[0]; // 실제로는 생성된 건물 사용

            // 3. 전략적 위치 리스트 준비
            System.Collections.Generic.List<StrategicLocation> strategicLocations = 
                new System.Collections.Generic.List<StrategicLocation>
            {
                new StrategicLocation
                {
                    position = new Vector3(20, 0, 20),
                    locationType = StrategyType.CoverPoint,
                    dangerScore = 0.3f,
                    visibilityScore = 0.2f
                },
                new StrategicLocation
                {
                    position = new Vector3(50, 0, 50),
                    locationType = StrategyType.Intersection,
                    dangerScore = 0.7f,
                    visibilityScore = 0.8f
                }
            };

            // 4. 전략적 위치를 포함한 미니맵 생성
            Texture2D minimap = generator.GenerateMinimap(buildings, null, strategicLocations);

            // 5. 미니맵 저장
            generator.SaveMinimapToPNG(minimap, 99999);

            // 6. 메모리 정리
            Destroy(minimap);
        }
    }
}
