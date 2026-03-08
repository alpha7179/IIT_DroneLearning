using UnityEngine;
using UnityEditor;
using System.Linq;

namespace ProceduralCityGenerator.Editor
{
    /// <summary>
    /// 건물 이름 지정 기능을 검증하는 테스트
    /// 요구사항 13.4: 각 건물 GameObject의 이름을 격자 좌표로 지정
    /// </summary>
    public static class BuildingNamingTest
    {
        [MenuItem("City Generator/Test Building Naming")]
        public static void TestBuildingNaming()
        {
            Debug.Log("=== Starting Building Naming Test (Requirement 13.4) ===");

            // 테스트용 GameObject 생성
            GameObject testObj = new GameObject("TestCityGenerator");
            CityGenerator generator = testObj.AddComponent<CityGenerator>();

            // 기본 머티리얼 생성
            Material testMaterial = new Material(Shader.Find("Standard"));
            generator.defaultBuildingMaterial = testMaterial;

            // 작은 도시로 테스트 (3x3 격자, 100% 밀도)
            generator.unitDistance = 1.0f;
            generator.minWidth = 3;
            generator.maxWidth = 3;
            generator.minDepth = 3;
            generator.maxDepth = 3;
            generator.buildingWidth = 1.0f;
            generator.buildingDepth = 1.0f;
            generator.buildingSpacing = 0.5f;
            generator.minBuildingHeight = 5.0f;
            generator.maxBuildingHeight = 10.0f;
            generator.buildingDensity = 1.0f; // 100% 밀도로 모든 셀에 건물 생성
            generator.randomSeed = 12345;

            Debug.Log("\n--- Test Configuration ---");
            Debug.Log($"Grid Size: {generator.minWidth}x{generator.minDepth}");
            Debug.Log($"Building Density: {generator.buildingDensity:P0}");
            Debug.Log($"Expected Buildings: {generator.minWidth * generator.minDepth}");

            // 도시 생성
            var result = generator.GenerateCity();

            Debug.Log("\n--- Generation Result ---");
            Debug.Log($"Success: {result.success}");
            Debug.Log($"Buildings Created: {result.buildingCount}");

            // "City" GameObject 찾기
            GameObject cityRoot = GameObject.Find("City");
            
            if (cityRoot == null)
            {
                Debug.LogError("FAILED: 'City' GameObject not found!");
                Cleanup(testMaterial, testObj);
                return;
            }

            Debug.Log($"\n--- Building Naming Verification ---");
            Debug.Log($"City Root: {cityRoot.name}");
            Debug.Log($"Child Count: {cityRoot.transform.childCount}");

            // 모든 건물 이름 검증
            int correctlyNamed = 0;
            int totalBuildings = cityRoot.transform.childCount;

            for (int i = 0; i < totalBuildings; i++)
            {
                GameObject building = cityRoot.transform.GetChild(i).gameObject;
                string buildingName = building.name;

                // 이름 형식 검증: "Building_X_Z"
                if (buildingName.StartsWith("Building_"))
                {
                    string[] parts = buildingName.Split('_');
                    if (parts.Length == 3)
                    {
                        if (int.TryParse(parts[1], out int x) && int.TryParse(parts[2], out int z))
                        {
                            // 좌표가 격자 범위 내에 있는지 확인
                            if (x >= 0 && x < generator.maxWidth && z >= 0 && z < generator.maxDepth)
                            {
                                correctlyNamed++;
                                if (i < 5) // 처음 5개만 출력
                                {
                                    Debug.Log($"  ✓ Building {i}: {buildingName} (Grid: {x}, {z})");
                                }
                            }
                            else
                            {
                                Debug.LogWarning($"  ✗ Building {i}: {buildingName} - Coordinates out of range!");
                            }
                        }
                        else
                        {
                            Debug.LogWarning($"  ✗ Building {i}: {buildingName} - Invalid coordinate format!");
                        }
                    }
                    else
                    {
                        Debug.LogWarning($"  ✗ Building {i}: {buildingName} - Invalid name format!");
                    }
                }
                else
                {
                    Debug.LogWarning($"  ✗ Building {i}: {buildingName} - Does not start with 'Building_'!");
                }
            }

            if (totalBuildings > 5)
            {
                Debug.Log($"  ... and {totalBuildings - 5} more buildings");
            }

            // 결과 요약
            Debug.Log("\n--- Test Summary ---");
            Debug.Log($"Total Buildings: {totalBuildings}");
            Debug.Log($"Correctly Named: {correctlyNamed}");
            Debug.Log($"Success Rate: {(float)correctlyNamed / totalBuildings:P0}");

            bool testPassed = correctlyNamed == totalBuildings && totalBuildings > 0;
            
            if (testPassed)
            {
                Debug.Log("\n<color=green>✓ TEST PASSED: All buildings are correctly named with grid coordinates!</color>");
                Debug.Log("Requirement 13.4 is satisfied: 각 건물 GameObject의 이름을 격자 좌표로 지정");
            }
            else
            {
                Debug.LogError("\n<color=red>✗ TEST FAILED: Some buildings are not correctly named!</color>");
            }

            // 정리
            Cleanup(testMaterial, testObj);
            
            Debug.Log("\n=== Building Naming Test Complete ===");
        }

        private static void Cleanup(Material material, GameObject testObj)
        {
            // 생성된 도시 제거
            GameObject cityRoot = GameObject.Find("City");
            if (cityRoot != null)
            {
                Object.DestroyImmediate(cityRoot);
            }

            // 테스트 오브젝트 제거
            if (material != null)
            {
                Object.DestroyImmediate(material);
            }
            if (testObj != null)
            {
                Object.DestroyImmediate(testObj);
            }
        }
    }
}
