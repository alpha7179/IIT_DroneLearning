using UnityEngine;
using UnityEditor;

namespace ProceduralCityGenerator.Editor
{
    /// <summary>
    /// CityGenerator의 파라미터 검증 기능을 테스트하는 에디터 스크립트
    /// </summary>
    public static class ValidationTests
    {
        [MenuItem("City Generator/Test Parameter Validation")]
        public static void TestParameterValidation()
        {
            Debug.Log("=== Starting Parameter Validation Tests ===");

            // 테스트용 GameObject 생성
            GameObject testObj = new GameObject("TestCityGenerator");
            CityGenerator generator = testObj.AddComponent<CityGenerator>();

            // Test 1: 범위를 벗어난 값 테스트
            Debug.Log("\n--- Test 1: Out of Range Values ---");
            generator.unitDistance = 150f; // 범위: 0.1 ~ 100
            generator.minWidth = -5; // 범위: 1 ~ 100
            generator.buildingDensity = 1.5f; // 범위: 0 ~ 1
            
            var result1 = TestValidation(generator);
            Debug.Log($"Test 1 Result: {result1.warnings.Count} warnings, {result1.errors.Count} errors");

            // Test 2: 최소값이 최대값보다 큰 경우
            Debug.Log("\n--- Test 2: Min > Max Values ---");
            generator.minBuildingHeight = 50f;
            generator.maxBuildingHeight = 10f;
            generator.minWidth = 30;
            generator.maxWidth = 10;
            generator.minDepth = 40;
            generator.maxDepth = 20;

            var result2 = TestValidation(generator);
            Debug.Log($"Test 2 Result: {result2.warnings.Count} warnings, {result2.errors.Count} errors");

            // Test 3: 정상 값 테스트
            Debug.Log("\n--- Test 3: Valid Values ---");
            generator.unitDistance = 1.0f;
            generator.minWidth = 10;
            generator.maxWidth = 20;
            generator.minDepth = 10;
            generator.maxDepth = 20;
            generator.minBuildingHeight = 5f;
            generator.maxBuildingHeight = 20f;
            generator.buildingDensity = 0.7f;

            var result3 = TestValidation(generator);
            Debug.Log($"Test 3 Result: {result3.warnings.Count} warnings, {result3.errors.Count} errors");

            // 정리
            Object.DestroyImmediate(testObj);
            Debug.Log("\n=== Parameter Validation Tests Complete ===");
        }

        [MenuItem("City Generator/Test Grid Layout Creation")]
        public static void TestGridLayoutCreation()
        {
            Debug.Log("=== Starting Grid Layout Creation Tests ===");

            // 테스트용 GameObject 생성
            GameObject testObj = new GameObject("TestCityGenerator");
            CityGenerator generator = testObj.AddComponent<CityGenerator>();

            // Test 1: 기본 파라미터로 격자 생성
            Debug.Log("\n--- Test 1: Default Parameters ---");
            generator.unitDistance = 1.0f;
            generator.minWidth = 5;
            generator.maxWidth = 10;
            generator.minDepth = 5;
            generator.maxDepth = 10;
            generator.buildingWidth = 1.0f;
            generator.buildingDepth = 1.0f;
            generator.buildingSpacing = 0.5f;
            generator.randomSeed = 12345;

            TestGridLayout(generator);

            // Test 2: 다른 시드로 격자 생성 (다른 크기가 나와야 함)
            Debug.Log("\n--- Test 2: Different Seed ---");
            generator.randomSeed = 67890;
            TestGridLayout(generator);

            // Test 3: 고정 크기 (min == max)
            Debug.Log("\n--- Test 3: Fixed Size ---");
            generator.minWidth = 15;
            generator.maxWidth = 15;
            generator.minDepth = 15;
            generator.maxDepth = 15;
            generator.randomSeed = 11111;
            TestGridLayout(generator);

            // Test 4: 큰 단위 거리
            Debug.Log("\n--- Test 4: Large Unit Distance ---");
            generator.unitDistance = 5.0f;
            generator.minWidth = 3;
            generator.maxWidth = 5;
            generator.minDepth = 3;
            generator.maxDepth = 5;
            generator.buildingWidth = 2.0f;
            generator.buildingDepth = 2.0f;
            generator.buildingSpacing = 1.0f;
            generator.randomSeed = 22222;
            TestGridLayout(generator);

            // 정리
            Object.DestroyImmediate(testObj);
            Debug.Log("\n=== Grid Layout Creation Tests Complete ===");
        }

        [MenuItem("City Generator/Test Building Placement")]
        public static void TestBuildingPlacement()
        {
            Debug.Log("=== Starting Building Placement Tests ===");

            // 테스트용 GameObject 생성
            GameObject testObj = new GameObject("TestCityGenerator");
            CityGenerator generator = testObj.AddComponent<CityGenerator>();

            // 기본 머티리얼 생성 (테스트용)
            Material testMaterial = new Material(Shader.Find("Standard"));
            generator.defaultBuildingMaterial = testMaterial;

            // Test 1: 100% 밀도로 건물 배치
            Debug.Log("\n--- Test 1: 100% Density ---");
            generator.unitDistance = 1.0f;
            generator.minWidth = 5;
            generator.maxWidth = 5;
            generator.minDepth = 5;
            generator.maxDepth = 5;
            generator.buildingWidth = 1.0f;
            generator.buildingDepth = 1.0f;
            generator.buildingSpacing = 0.5f;
            generator.minBuildingHeight = 5.0f;
            generator.maxBuildingHeight = 20.0f;
            generator.buildingDensity = 1.0f;
            generator.randomSeed = 12345;

            TestBuildingPlacementInternal(generator);

            // Test 2: 50% 밀도로 건물 배치
            Debug.Log("\n--- Test 2: 50% Density ---");
            generator.buildingDensity = 0.5f;
            generator.randomSeed = 67890;
            TestBuildingPlacementInternal(generator);

            // Test 3: 0% 밀도로 건물 배치 (건물이 생성되지 않아야 함)
            Debug.Log("\n--- Test 3: 0% Density ---");
            generator.buildingDensity = 0.0f;
            generator.randomSeed = 11111;
            TestBuildingPlacementInternal(generator);

            // Test 4: 다양한 건물 높이 범위
            Debug.Log("\n--- Test 4: Various Building Heights ---");
            generator.buildingDensity = 0.7f;
            generator.minBuildingHeight = 10.0f;
            generator.maxBuildingHeight = 50.0f;
            generator.randomSeed = 22222;
            TestBuildingPlacementInternal(generator);

            // 정리
            Object.DestroyImmediate(testMaterial);
            Object.DestroyImmediate(testObj);
            Debug.Log("\n=== Building Placement Tests Complete ===");
        }

        private static ValidationResult TestValidation(CityGenerator generator)
        {
            // Reflection을 사용하여 private 메서드 호출
            var method = typeof(CityGenerator).GetMethod("ValidateParameters", 
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            
            if (method == null)
            {
                Debug.LogError("ValidateParameters 메서드를 찾을 수 없습니다!");
                return new ValidationResult(false);
            }

            return (ValidationResult)method.Invoke(generator, null);
        }

        private static void TestGridLayout(CityGenerator generator)
        {
            // Reflection을 사용하여 private 메서드 호출
            var initSeedMethod = typeof(CityGenerator).GetMethod("InitializeRandomSeed",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            
            var createGridMethod = typeof(CityGenerator).GetMethod("CreateGridLayout",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

            if (initSeedMethod == null || createGridMethod == null)
            {
                Debug.LogError("필요한 메서드를 찾을 수 없습니다!");
                return;
            }

            // 랜덤 시드 초기화
            initSeedMethod.Invoke(generator, null);

            // 격자 레이아웃 생성
            createGridMethod.Invoke(generator, null);

            // 결과 확인을 위해 private 필드 읽기
            var gridField = typeof(CityGenerator).GetField("grid",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var widthField = typeof(CityGenerator).GetField("actualCityWidth",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var depthField = typeof(CityGenerator).GetField("actualCityDepth",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

            if (gridField == null || widthField == null || depthField == null)
            {
                Debug.LogError("필요한 필드를 찾을 수 없습니다!");
                return;
            }

            var grid = (GridCell[,])gridField.GetValue(generator);
            int actualWidth = (int)widthField.GetValue(generator);
            int actualDepth = (int)depthField.GetValue(generator);

            Debug.Log($"생성된 격자 크기: {actualWidth} x {actualDepth}");
            Debug.Log($"격자 범위: 가로 [{generator.minWidth}, {generator.maxWidth}], 세로 [{generator.minDepth}, {generator.maxDepth}]");

            // 격자 크기가 범위 내에 있는지 확인
            bool widthInRange = actualWidth >= generator.minWidth && actualWidth <= generator.maxWidth;
            bool depthInRange = actualDepth >= generator.minDepth && actualDepth <= generator.maxDepth;

            Debug.Log($"가로 크기 범위 확인: {widthInRange}");
            Debug.Log($"세로 크기 범위 확인: {depthInRange}");

            // 몇 개의 셀 좌표 확인
            if (grid != null && actualWidth > 0 && actualDepth > 0)
            {
                Debug.Log($"셀 [0,0] 월드 좌표: {grid[0, 0].worldPosition}");
                
                if (actualWidth > 1 && actualDepth > 1)
                {
                    Debug.Log($"셀 [1,1] 월드 좌표: {grid[1, 1].worldPosition}");
                    
                    // 좌표 계산 공식 검증
                    float expectedX = 1 * (generator.buildingWidth + generator.buildingSpacing) * generator.unitDistance;
                    float expectedZ = 1 * (generator.buildingDepth + generator.buildingSpacing) * generator.unitDistance;
                    Vector3 expected = new Vector3(expectedX, 0f, expectedZ);
                    
                    bool coordinatesMatch = Vector3.Distance(grid[1, 1].worldPosition, expected) < 0.001f;
                    Debug.Log($"좌표 계산 공식 검증: {coordinatesMatch} (예상: {expected}, 실제: {grid[1, 1].worldPosition})");
                }
            }
        }

        private static void TestBuildingPlacementInternal(CityGenerator generator)
        {
            // Reflection을 사용하여 private 메서드 호출
            var initSeedMethod = typeof(CityGenerator).GetMethod("InitializeRandomSeed",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            
            var createGridMethod = typeof(CityGenerator).GetMethod("CreateGridLayout",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

            var placeBuildingsMethod = typeof(CityGenerator).GetMethod("PlaceBuildingsOnGrid",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

            if (initSeedMethod == null || createGridMethod == null || placeBuildingsMethod == null)
            {
                Debug.LogError("필요한 메서드를 찾을 수 없습니다!");
                return;
            }

            // 랜덤 시드 초기화
            initSeedMethod.Invoke(generator, null);

            // 격자 레이아웃 생성
            createGridMethod.Invoke(generator, null);

            // 건물 배치
            placeBuildingsMethod.Invoke(generator, null);

            // 결과 확인을 위해 private 필드 읽기
            var buildingsField = typeof(CityGenerator).GetField("buildings",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var gridField = typeof(CityGenerator).GetField("grid",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var widthField = typeof(CityGenerator).GetField("actualCityWidth",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var depthField = typeof(CityGenerator).GetField("actualCityDepth",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

            if (buildingsField == null || gridField == null || widthField == null || depthField == null)
            {
                Debug.LogError("필요한 필드를 찾을 수 없습니다!");
                return;
            }

            var buildings = (System.Collections.Generic.List<Building>)buildingsField.GetValue(generator);
            var grid = (GridCell[,])gridField.GetValue(generator);
            int actualWidth = (int)widthField.GetValue(generator);
            int actualDepth = (int)depthField.GetValue(generator);

            int totalCells = actualWidth * actualDepth;
            int buildingCount = buildings != null ? buildings.Count : 0;
            float actualDensity = totalCells > 0 ? (float)buildingCount / totalCells : 0f;

            Debug.Log($"격자 크기: {actualWidth} x {actualDepth} (총 {totalCells}개 셀)");
            Debug.Log($"생성된 건물 수: {buildingCount}");
            Debug.Log($"실제 밀도: {actualDensity:P2} (목표: {generator.buildingDensity:P2})");

            // 건물 밀도 검증 (확률적이므로 ±20% 허용)
            float densityDifference = Mathf.Abs(actualDensity - generator.buildingDensity);
            bool densityInRange = densityDifference <= 0.2f || generator.buildingDensity == 0f || generator.buildingDensity == 1f;
            Debug.Log($"밀도 범위 확인: {densityInRange} (차이: {densityDifference:P2})");

            // 건물 높이 범위 검증
            if (buildings != null && buildings.Count > 0)
            {
                float minHeight = float.MaxValue;
                float maxHeight = float.MinValue;

                foreach (var building in buildings)
                {
                    if (building.height < minHeight) minHeight = building.height;
                    if (building.height > maxHeight) maxHeight = building.height;
                }

                Debug.Log($"건물 높이 범위: [{minHeight:F2}, {maxHeight:F2}] (목표: [{generator.minBuildingHeight:F2}, {generator.maxBuildingHeight:F2}])");

                bool heightInRange = minHeight >= generator.minBuildingHeight && maxHeight <= generator.maxBuildingHeight;
                Debug.Log($"높이 범위 확인: {heightInRange}");

                // 첫 번째 건물 정보 출력
                var firstBuilding = buildings[0];
                Debug.Log($"첫 번째 건물 정보:");
                Debug.Log($"  - 이름: {firstBuilding.gameObject.name}");
                Debug.Log($"  - 위치: {firstBuilding.position}");
                Debug.Log($"  - 크기: {firstBuilding.size}");
                Debug.Log($"  - 높이: {firstBuilding.height}");
                Debug.Log($"  - 격자 좌표: ({firstBuilding.gridCell.x}, {firstBuilding.gridCell.z})");
            }

            // 격자 셀의 hasBuilding 플래그 검증
            int cellsWithBuilding = 0;
            for (int x = 0; x < actualWidth; x++)
            {
                for (int z = 0; z < actualDepth; z++)
                {
                    if (grid[x, z].hasBuilding)
                    {
                        cellsWithBuilding++;
                    }
                }
            }

            Debug.Log($"hasBuilding 플래그가 설정된 셀 수: {cellsWithBuilding}");
            bool flagsMatch = cellsWithBuilding == buildingCount;
            Debug.Log($"플래그와 건물 수 일치: {flagsMatch}");
        }

        [MenuItem("City Generator/Test Clear City")]
        public static void TestClearCity()
        {
            Debug.Log("=== Starting Clear City Tests ===");

            // 테스트용 GameObject 생성
            GameObject testObj = new GameObject("TestCityGenerator");
            CityGenerator generator = testObj.AddComponent<CityGenerator>();

            // 기본 머티리얼 생성 (테스트용)
            Material testMaterial = new Material(Shader.Find("Standard"));
            generator.defaultBuildingMaterial = testMaterial;

            // 기본 파라미터 설정
            generator.unitDistance = 1.0f;
            generator.minWidth = 5;
            generator.maxWidth = 5;
            generator.minDepth = 5;
            generator.maxDepth = 5;
            generator.buildingWidth = 1.0f;
            generator.buildingDepth = 1.0f;
            generator.buildingSpacing = 0.5f;
            generator.minBuildingHeight = 5.0f;
            generator.maxBuildingHeight = 20.0f;
            generator.buildingDensity = 0.7f;
            generator.randomSeed = 12345;

            // Test 1: 도시 생성 후 제거
            Debug.Log("\n--- Test 1: Generate and Clear City ---");
            var result = generator.GenerateCity();
            Debug.Log($"도시 생성 완료: 건물 {result.buildingCount}개, 노드 {result.nodeCount}개");

            // 생성된 도시 루트 확인
            GameObject cityRoot = GameObject.Find("City");
            bool cityRootExists = cityRoot != null;
            Debug.Log($"도시 루트 GameObject 존재: {cityRootExists}");

            if (cityRootExists)
            {
                int childCount = cityRoot.transform.childCount;
                Debug.Log($"도시 루트의 자식 수: {childCount}");
            }

            // 도시 제거
            generator.ClearCity();
            Debug.Log("도시 제거 완료");

            // 제거 후 확인
            cityRoot = GameObject.Find("City");
            bool cityRootCleared = cityRoot == null;
            Debug.Log($"도시 루트 GameObject 제거됨: {cityRootCleared}");

            // Test 2: 여러 번 생성 및 제거
            Debug.Log("\n--- Test 2: Multiple Generate and Clear Cycles ---");
            for (int i = 0; i < 3; i++)
            {
                Debug.Log($"\n사이클 {i + 1}:");
                
                // 도시 생성
                result = generator.GenerateCity();
                Debug.Log($"  생성: 건물 {result.buildingCount}개");
                
                cityRoot = GameObject.Find("City");
                Debug.Log($"  도시 루트 존재: {cityRoot != null}");
                
                // 도시 제거
                generator.ClearCity();
                Debug.Log($"  제거 완료");
                
                cityRoot = GameObject.Find("City");
                Debug.Log($"  도시 루트 제거됨: {cityRoot == null}");
            }

            // Test 3: 빈 상태에서 ClearCity 호출 (오류가 발생하지 않아야 함)
            Debug.Log("\n--- Test 3: Clear Empty City ---");
            generator.ClearCity();
            Debug.Log("빈 상태에서 ClearCity 호출 완료 (오류 없음)");

            // 정리
            Object.DestroyImmediate(testMaterial);
            Object.DestroyImmediate(testObj);
            Debug.Log("\n=== Clear City Tests Complete ===");
        }
    }
}
