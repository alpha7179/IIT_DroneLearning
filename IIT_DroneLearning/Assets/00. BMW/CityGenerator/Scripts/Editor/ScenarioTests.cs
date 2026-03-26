using NUnit.Framework;
using UnityEngine;
using UnityEditor;
using System.IO;

namespace CityGenerator.Tests
{
    /// <summary>
    /// Task 15.2: 기본 시나리오 테스트
    /// 다양한 파라미터 조합으로 도시 생성, 프리셋 저장/로드, 미니맵 생성을 테스트합니다.
    /// **Validates: Requirements 전체**
    /// </summary>
    [TestFixture]
    public class ScenarioTests
    {
        private GameObject testGameObject;
        private CityGenerator cityGenerator;
        private const string TestPresetPath = "Assets/CityPresets/TestScenario";

        [SetUp]
        public void Setup()
        {
            // 테스트용 GameObject 생성
            testGameObject = new GameObject("TestCityGenerator");
            cityGenerator = testGameObject.AddComponent<CityGenerator>();

            // 기본 머티리얼 설정
            cityGenerator.defaultBuildingMaterial = new Material(Shader.Find("Standard"));

            // CityPresets 디렉토리 확인 및 생성
            if (!Directory.Exists("Assets/CityPresets"))
            {
                Directory.CreateDirectory("Assets/CityPresets");
                AssetDatabase.Refresh();
            }
        }

        [TearDown]
        public void Teardown()
        {
            // 테스트 후 정리
            if (cityGenerator != null)
            {
                cityGenerator.ClearCity();
            }

            if (testGameObject != null)
            {
                Object.DestroyImmediate(testGameObject);
            }

            // 이전 테스트에서 남은 stray "City" 오브젝트 정리
            GameObject cityObj = GameObject.Find("City");
            while (cityObj != null)
            {
                Object.DestroyImmediate(cityObj);
                cityObj = GameObject.Find("City");
            }

            // 테스트 프리셋 파일 삭제
            CleanupTestPresets();
        }

        private void CleanupTestPresets()
        {
            string[] testPresets = new string[]
            {
                "TestScenario_SmallDenseCity.asset",
                "TestScenario_LargeSparseCity.asset",
                "TestScenario_TallBuildings.asset",
                "TestScenario_LowBuildings.asset",
                "TestScenario_NoSpacing.asset",
                "TestScenario_WideSpacing.asset",
                "TestScenario_MixedParameters.asset",
                "TestScenario_ExtremeParameters.asset",
                "TestScenario_MinimalCity.asset",
                "TestScenario_MaximalCity.asset"
            };

            foreach (string presetName in testPresets)
            {
                string path = $"Assets/CityPresets/{presetName}";
                if (AssetDatabase.LoadAssetAtPath<CityParameters>(path) != null)
                {
                    AssetDatabase.DeleteAsset(path);
                }
            }
            AssetDatabase.Refresh();
        }

        // ===== 시나리오 1: 작고 밀집된 도시 =====
        [Test]
        public void Scenario_SmallDenseCity_GeneratesSuccessfully()
        {
            // Arrange: 작고 밀집된 도시 파라미터
            cityGenerator.unitDistance = 1.0f;
            cityGenerator.minWidth = 5;
            cityGenerator.maxWidth = 8;
            cityGenerator.minDepth = 5;
            cityGenerator.maxDepth = 8;
            cityGenerator.buildingWidth = 1.0f;
            cityGenerator.buildingDepth = 1.0f;
            cityGenerator.minBuildingHeight = 5.0f;
            cityGenerator.maxBuildingHeight = 15.0f;
            cityGenerator.buildingSpacing = 0.5f;
            cityGenerator.buildingDensity = 0.9f; // 높은 밀도
            cityGenerator.randomSeed = 1001;

            // Act: 도시 생성
            CityGenerationResult result = cityGenerator.GenerateCity();

            // Assert: 생성 성공 및 특성 확인
            Assert.IsTrue(result.success, "작고 밀집된 도시 생성 실패");
            Assert.Greater(result.buildingCount, 20, "충분한 건물이 생성되지 않음");
            Assert.Greater(result.nodeCount, 0, "그래프 노드가 생성되지 않음");
            Assert.Less(result.generationTime, 5.0f, "생성 시간이 5초를 초과");

            Debug.Log($"[시나리오 1] 작고 밀집된 도시: 건물 {result.buildingCount}개, " +
                     $"노드 {result.nodeCount}개, 시간 {result.generationTime:F2}초");
        }

        // ===== 시나리오 2: 크고 희소한 도시 =====
        [Test]
        public void Scenario_LargeSparseCity_GeneratesSuccessfully()
        {
            // Arrange: 크고 희소한 도시 파라미터
            cityGenerator.unitDistance = 2.0f;
            cityGenerator.minWidth = 20;
            cityGenerator.maxWidth = 25;
            cityGenerator.minDepth = 20;
            cityGenerator.maxDepth = 25;
            cityGenerator.buildingWidth = 1.5f;
            cityGenerator.buildingDepth = 1.5f;
            cityGenerator.minBuildingHeight = 10.0f;
            cityGenerator.maxBuildingHeight = 30.0f;
            cityGenerator.buildingSpacing = 3.0f;
            cityGenerator.buildingDensity = 0.3f; // 낮은 밀도
            cityGenerator.randomSeed = 1002;

            // Act: 도시 생성
            CityGenerationResult result = cityGenerator.GenerateCity();

            // Assert: 생성 성공 및 특성 확인
            Assert.IsTrue(result.success, "크고 희소한 도시 생성 실패");
            Assert.Greater(result.buildingCount, 50, "충분한 건물이 생성되지 않음");
            Assert.Greater(result.nodeCount, 0, "그래프 노드가 생성되지 않음");
            Assert.Less(result.generationTime, 5.0f, "생성 시간이 5초를 초과");

            Debug.Log($"[시나리오 2] 크고 희소한 도시: 건물 {result.buildingCount}개, " +
                     $"노드 {result.nodeCount}개, 시간 {result.generationTime:F2}초");
        }

        // ===== 시나리오 3: 고층 건물 도시 =====
        [Test]
        public void Scenario_TallBuildingsCity_GeneratesSuccessfully()
        {
            // Arrange: 고층 건물 도시 파라미터
            cityGenerator.unitDistance = 1.5f;
            cityGenerator.minWidth = 10;
            cityGenerator.maxWidth = 15;
            cityGenerator.minDepth = 10;
            cityGenerator.maxDepth = 15;
            cityGenerator.buildingWidth = 1.0f;
            cityGenerator.buildingDepth = 1.0f;
            cityGenerator.minBuildingHeight = 30.0f; // 높은 최소 높이
            cityGenerator.maxBuildingHeight = 100.0f; // 매우 높은 최대 높이
            cityGenerator.buildingSpacing = 1.5f;
            cityGenerator.buildingDensity = 0.6f;
            cityGenerator.randomSeed = 1003;

            // Act: 도시 생성
            CityGenerationResult result = cityGenerator.GenerateCity();

            // Assert: 생성 성공 및 특성 확인
            Assert.IsTrue(result.success, "고층 건물 도시 생성 실패");
            Assert.Greater(result.buildingCount, 30, "충분한 건물이 생성되지 않음");
            Assert.Greater(result.nodeCount, 0, "그래프 노드가 생성되지 않음");

            Debug.Log($"[시나리오 3] 고층 건물 도시: 건물 {result.buildingCount}개, " +
                     $"노드 {result.nodeCount}개, 시간 {result.generationTime:F2}초");
        }

        // ===== 시나리오 4: 저층 건물 도시 =====
        [Test]
        public void Scenario_LowBuildingsCity_GeneratesSuccessfully()
        {
            // Arrange: 저층 건물 도시 파라미터
            cityGenerator.unitDistance = 1.0f;
            cityGenerator.minWidth = 12;
            cityGenerator.maxWidth = 18;
            cityGenerator.minDepth = 12;
            cityGenerator.maxDepth = 18;
            cityGenerator.buildingWidth = 1.2f;
            cityGenerator.buildingDepth = 1.2f;
            cityGenerator.minBuildingHeight = 3.0f; // 낮은 최소 높이
            cityGenerator.maxBuildingHeight = 8.0f; // 낮은 최대 높이
            cityGenerator.buildingSpacing = 1.0f;
            cityGenerator.buildingDensity = 0.7f;
            cityGenerator.randomSeed = 1004;

            // Act: 도시 생성
            CityGenerationResult result = cityGenerator.GenerateCity();

            // Assert: 생성 성공 및 특성 확인
            Assert.IsTrue(result.success, "저층 건물 도시 생성 실패");
            Assert.Greater(result.buildingCount, 50, "충분한 건물이 생성되지 않음");
            Assert.Greater(result.nodeCount, 0, "그래프 노드가 생성되지 않음");

            Debug.Log($"[시나리오 4] 저층 건물 도시: 건물 {result.buildingCount}개, " +
                     $"노드 {result.nodeCount}개, 시간 {result.generationTime:F2}초");
        }

        // ===== 시나리오 5: 간격 없는 도시 =====
        [Test]
        public void Scenario_NoSpacingCity_GeneratesSuccessfully()
        {
            // Arrange: 간격 없는 도시 파라미터
            cityGenerator.unitDistance = 1.0f;
            cityGenerator.minWidth = 10;
            cityGenerator.maxWidth = 10;
            cityGenerator.minDepth = 10;
            cityGenerator.maxDepth = 10;
            cityGenerator.buildingWidth = 1.0f;
            cityGenerator.buildingDepth = 1.0f;
            cityGenerator.minBuildingHeight = 5.0f;
            cityGenerator.maxBuildingHeight = 20.0f;
            cityGenerator.buildingSpacing = 0.0f; // 간격 없음
            cityGenerator.buildingDensity = 1.0f; // 최대 밀도
            cityGenerator.randomSeed = 1005;

            // Act: 도시 생성
            CityGenerationResult result = cityGenerator.GenerateCity();

            // Assert: 생성 성공 및 특성 확인
            Assert.IsTrue(result.success, "간격 없는 도시 생성 실패");
            Assert.AreEqual(100, result.buildingCount, "모든 격자 셀에 건물이 생성되어야 함");
            Assert.Greater(result.nodeCount, 0, "그래프 노드가 생성되지 않음");

            Debug.Log($"[시나리오 5] 간격 없는 도시: 건물 {result.buildingCount}개, " +
                     $"노드 {result.nodeCount}개, 시간 {result.generationTime:F2}초");
        }

        // ===== 시나리오 6: 넓은 간격 도시 =====
        [Test]
        public void Scenario_WideSpacingCity_GeneratesSuccessfully()
        {
            // Arrange: 넓은 간격 도시 파라미터
            cityGenerator.unitDistance = 1.0f;
            cityGenerator.minWidth = 8;
            cityGenerator.maxWidth = 12;
            cityGenerator.minDepth = 8;
            cityGenerator.maxDepth = 12;
            cityGenerator.buildingWidth = 0.8f;
            cityGenerator.buildingDepth = 0.8f;
            cityGenerator.minBuildingHeight = 10.0f;
            cityGenerator.maxBuildingHeight = 25.0f;
            cityGenerator.buildingSpacing = 5.0f; // 넓은 간격
            cityGenerator.buildingDensity = 0.5f;
            cityGenerator.randomSeed = 1006;

            // Act: 도시 생성
            CityGenerationResult result = cityGenerator.GenerateCity();

            // Assert: 생성 성공 및 특성 확인
            Assert.IsTrue(result.success, "넓은 간격 도시 생성 실패");
            Assert.Greater(result.buildingCount, 20, "충분한 건물이 생성되지 않음");
            Assert.Greater(result.nodeCount, 0, "그래프 노드가 생성되지 않음");

            Debug.Log($"[시나리오 6] 넓은 간격 도시: 건물 {result.buildingCount}개, " +
                     $"노드 {result.nodeCount}개, 시간 {result.generationTime:F2}초");
        }

        // ===== 시나리오 7: 혼합 파라미터 도시 =====
        [Test]
        public void Scenario_MixedParametersCity_GeneratesSuccessfully()
        {
            // Arrange: 다양한 파라미터 조합
            cityGenerator.unitDistance = 1.8f;
            cityGenerator.minWidth = 15;
            cityGenerator.maxWidth = 20;
            cityGenerator.minDepth = 10;
            cityGenerator.maxDepth = 25;
            cityGenerator.buildingWidth = 1.3f;
            cityGenerator.buildingDepth = 0.9f;
            cityGenerator.minBuildingHeight = 8.0f;
            cityGenerator.maxBuildingHeight = 50.0f;
            cityGenerator.buildingSpacing = 2.2f;
            cityGenerator.buildingDensity = 0.65f;
            cityGenerator.randomSeed = 1007;

            // Act: 도시 생성
            CityGenerationResult result = cityGenerator.GenerateCity();

            // Assert: 생성 성공 및 특성 확인
            Assert.IsTrue(result.success, "혼합 파라미터 도시 생성 실패");
            Assert.Greater(result.buildingCount, 50, "충분한 건물이 생성되지 않음");
            Assert.Greater(result.nodeCount, 0, "그래프 노드가 생성되지 않음");

            Debug.Log($"[시나리오 7] 혼합 파라미터 도시: 건물 {result.buildingCount}개, " +
                     $"노드 {result.nodeCount}개, 시간 {result.generationTime:F2}초");
        }

        // ===== 시나리오 8: 극단적 파라미터 도시 =====
        [Test]
        public void Scenario_ExtremeParametersCity_GeneratesSuccessfully()
        {
            // Arrange: 극단적 파라미터 조합
            cityGenerator.unitDistance = 5.0f; // 큰 단위 거리
            cityGenerator.minWidth = 3;
            cityGenerator.maxWidth = 5;
            cityGenerator.minDepth = 3;
            cityGenerator.maxDepth = 5;
            cityGenerator.buildingWidth = 0.5f; // 작은 건물
            cityGenerator.buildingDepth = 0.5f;
            cityGenerator.minBuildingHeight = 1.0f;
            cityGenerator.maxBuildingHeight = 200.0f; // 매우 높은 건물
            cityGenerator.buildingSpacing = 10.0f; // 매우 넓은 간격
            cityGenerator.buildingDensity = 0.2f; // 매우 낮은 밀도
            cityGenerator.randomSeed = 1008;

            // Act: 도시 생성
            CityGenerationResult result = cityGenerator.GenerateCity();

            // Assert: 생성 성공 및 특성 확인
            Assert.IsTrue(result.success, "극단적 파라미터 도시 생성 실패");
            Assert.Greater(result.buildingCount, 0, "건물이 생성되지 않음");
            Assert.Greater(result.nodeCount, 0, "그래프 노드가 생성되지 않음");

            Debug.Log($"[시나리오 8] 극단적 파라미터 도시: 건물 {result.buildingCount}개, " +
                     $"노드 {result.nodeCount}개, 시간 {result.generationTime:F2}초");
        }

        // ===== 시나리오 9: 최소 크기 도시 =====
        [Test]
        public void Scenario_MinimalCity_GeneratesSuccessfully()
        {
            // Arrange: 최소 크기 도시 파라미터
            cityGenerator.unitDistance = 1.0f;
            cityGenerator.minWidth = 1;
            cityGenerator.maxWidth = 1;
            cityGenerator.minDepth = 1;
            cityGenerator.maxDepth = 1;
            cityGenerator.buildingWidth = 1.0f;
            cityGenerator.buildingDepth = 1.0f;
            cityGenerator.minBuildingHeight = 5.0f;
            cityGenerator.maxBuildingHeight = 10.0f;
            cityGenerator.buildingSpacing = 0.0f;
            cityGenerator.buildingDensity = 1.0f;
            cityGenerator.randomSeed = 1009;

            // Act: 도시 생성
            CityGenerationResult result = cityGenerator.GenerateCity();

            // Assert: 생성 성공 및 특성 확인
            Assert.IsTrue(result.success, "최소 크기 도시 생성 실패");
            Assert.AreEqual(1, result.buildingCount, "1개의 건물이 생성되어야 함");
            Assert.Greater(result.nodeCount, 0, "그래프 노드가 생성되지 않음");

            Debug.Log($"[시나리오 9] 최소 크기 도시: 건물 {result.buildingCount}개, " +
                     $"노드 {result.nodeCount}개, 시간 {result.generationTime:F2}초");
        }

        // ===== 시나리오 10: 최대 크기 도시 (성능 테스트) =====
        [Test]
        public void Scenario_MaximalCity_GeneratesWithinTimeLimit()
        {
            // Arrange: 최대 크기 도시 파라미터
            cityGenerator.unitDistance = 1.0f;
            cityGenerator.minWidth = 40;
            cityGenerator.maxWidth = 40;
            cityGenerator.minDepth = 40;
            cityGenerator.maxDepth = 40;
            cityGenerator.buildingWidth = 1.0f;
            cityGenerator.buildingDepth = 1.0f;
            cityGenerator.minBuildingHeight = 5.0f;
            cityGenerator.maxBuildingHeight = 30.0f;
            cityGenerator.buildingSpacing = 1.0f;
            cityGenerator.buildingDensity = 0.8f;
            cityGenerator.randomSeed = 1010;

            // Act: 도시 생성
            CityGenerationResult result = cityGenerator.GenerateCity();

            // Assert: 생성 성공 및 성능 요구사항 확인
            Assert.IsTrue(result.success, "최대 크기 도시 생성 실패");
            Assert.Greater(result.buildingCount, 1000, "1000개 이상의 건물이 생성되어야 함");
            Assert.Less(result.generationTime, 5.0f, "생성 시간이 5초를 초과");

            Debug.Log($"[시나리오 10] 최대 크기 도시: 건물 {result.buildingCount}개, " +
                     $"노드 {result.nodeCount}개, 시간 {result.generationTime:F2}초");
        }

        // ===== 프리셋 저장 및 로드 테스트 =====
        [Test]
        public void Scenario_SaveAndLoadPreset_WorksCorrectly()
        {
            // Arrange: 특정 파라미터 설정
            cityGenerator.unitDistance = 2.5f;
            cityGenerator.minWidth = 15;
            cityGenerator.maxWidth = 20;
            cityGenerator.minDepth = 12;
            cityGenerator.maxDepth = 18;
            cityGenerator.buildingWidth = 1.3f;
            cityGenerator.buildingDepth = 1.1f;
            cityGenerator.minBuildingHeight = 10.0f;
            cityGenerator.maxBuildingHeight = 40.0f;
            cityGenerator.buildingSpacing = 2.0f;
            cityGenerator.buildingDensity = 0.75f;
            cityGenerator.randomSeed = 2001;
            cityGenerator.minimapResolution = MinimapResolution.Resolution1024;

            // 원본 파라미터 저장
            float originalUnitDistance = cityGenerator.unitDistance;
            int originalMinWidth = cityGenerator.minWidth;
            float originalBuildingDensity = cityGenerator.buildingDensity;
            int originalRandomSeed = cityGenerator.randomSeed;

            // Act: 프리셋 저장
            string presetName = "TestScenario_SaveLoad";
            cityGenerator.SavePreset(presetName);

            // 파라미터 변경
            cityGenerator.unitDistance = 1.0f;
            cityGenerator.minWidth = 5;
            cityGenerator.buildingDensity = 0.3f;
            cityGenerator.randomSeed = 9999;

            // 프리셋 로드
            string assetPath = $"Assets/CityPresets/{presetName}.asset";
            CityParameters preset = AssetDatabase.LoadAssetAtPath<CityParameters>(assetPath);
            Assert.IsNotNull(preset, "프리셋이 저장되지 않음");

            cityGenerator.LoadPreset(preset);

            // Assert: 파라미터가 복원되었는지 확인
            Assert.AreEqual(originalUnitDistance, cityGenerator.unitDistance, 0.001f, "unitDistance가 복원되지 않음");
            Assert.AreEqual(originalMinWidth, cityGenerator.minWidth, "minWidth가 복원되지 않음");
            Assert.AreEqual(originalBuildingDensity, cityGenerator.buildingDensity, 0.001f, "buildingDensity가 복원되지 않음");
            Assert.AreEqual(originalRandomSeed, cityGenerator.randomSeed, "randomSeed가 복원되지 않음");

            Debug.Log($"[프리셋 테스트] 저장 및 로드 성공: {presetName}");

            // Cleanup
            if (AssetDatabase.LoadAssetAtPath<CityParameters>(assetPath) != null)
            {
                AssetDatabase.DeleteAsset(assetPath);
            }
        }

        // ===== 프리셋 저장 후 도시 생성 재현성 테스트 =====
        [Test]
        public void Scenario_PresetReproducibility_GeneratesSameCity()
        {
            // Arrange: 프리셋 저장
            cityGenerator.unitDistance = 1.5f;
            cityGenerator.minWidth = 10;
            cityGenerator.maxWidth = 10;
            cityGenerator.minDepth = 10;
            cityGenerator.maxDepth = 10;
            cityGenerator.buildingDensity = 0.7f;
            cityGenerator.randomSeed = 3001; // 고정 시드

            string presetName = "TestScenario_Reproducibility";
            cityGenerator.SavePreset(presetName);

            // Act: 첫 번째 도시 생성
            CityGenerationResult result1 = cityGenerator.GenerateCity();
            int buildingCount1 = result1.buildingCount;
            int nodeCount1 = result1.nodeCount;

            // 도시 제거
            cityGenerator.ClearCity();

            // 프리셋 로드 및 두 번째 도시 생성
            string assetPath = $"Assets/CityPresets/{presetName}.asset";
            CityParameters preset = AssetDatabase.LoadAssetAtPath<CityParameters>(assetPath);
            cityGenerator.LoadPreset(preset);

            CityGenerationResult result2 = cityGenerator.GenerateCity();
            int buildingCount2 = result2.buildingCount;
            int nodeCount2 = result2.nodeCount;

            // Assert: 동일한 도시가 생성되었는지 확인
            Assert.AreEqual(buildingCount1, buildingCount2, "건물 수가 일치하지 않음");
            Assert.AreEqual(nodeCount1, nodeCount2, "노드 수가 일치하지 않음");

            Debug.Log($"[재현성 테스트] 프리셋으로 동일한 도시 생성 성공: 건물 {buildingCount1}개");

            // Cleanup
            if (AssetDatabase.LoadAssetAtPath<CityParameters>(assetPath) != null)
            {
                AssetDatabase.DeleteAsset(assetPath);
            }
        }

        // ===== 여러 프리셋 저장 및 로드 테스트 =====
        [Test]
        public void Scenario_MultiplePresets_SaveAndLoadCorrectly()
        {
            // Arrange & Act: 여러 프리셋 생성
            string[] presetNames = new string[]
            {
                "TestScenario_SmallDenseCity",
                "TestScenario_LargeSparseCity",
                "TestScenario_TallBuildings"
            };

            float[] densities = new float[] { 0.9f, 0.3f, 0.6f };
            int[] seeds = new int[] { 4001, 4002, 4003 };

            for (int i = 0; i < presetNames.Length; i++)
            {
                cityGenerator.buildingDensity = densities[i];
                cityGenerator.randomSeed = seeds[i];
                cityGenerator.SavePreset(presetNames[i]);
            }

            // Assert: 모든 프리셋이 저장되었는지 확인
            for (int i = 0; i < presetNames.Length; i++)
            {
                string assetPath = $"Assets/CityPresets/{presetNames[i]}.asset";
                CityParameters preset = AssetDatabase.LoadAssetAtPath<CityParameters>(assetPath);
                Assert.IsNotNull(preset, $"프리셋 {presetNames[i]}이 저장되지 않음");

                // 로드하여 값 확인
                cityGenerator.LoadPreset(preset);
                Assert.AreEqual(densities[i], cityGenerator.buildingDensity, 0.001f, 
                    $"프리셋 {presetNames[i]}의 buildingDensity가 일치하지 않음");
                Assert.AreEqual(seeds[i], cityGenerator.randomSeed, 
                    $"프리셋 {presetNames[i]}의 randomSeed가 일치하지 않음");
            }

            Debug.Log($"[다중 프리셋 테스트] {presetNames.Length}개의 프리셋 저장 및 로드 성공");
        }

        // ===== 미니맵 해상도 테스트 =====
        [Test]
        public void Scenario_MinimapResolution256_GeneratesSuccessfully()
        {
            // Arrange: 256x256 해상도 미니맵
            cityGenerator.unitDistance = 1.0f;
            cityGenerator.minWidth = 10;
            cityGenerator.maxWidth = 10;
            cityGenerator.minDepth = 10;
            cityGenerator.maxDepth = 10;
            cityGenerator.buildingDensity = 0.7f;
            cityGenerator.randomSeed = 5001;
            cityGenerator.minimapResolution = MinimapResolution.Resolution256;

            // Act: 도시 생성
            CityGenerationResult result = cityGenerator.GenerateCity();

            // Assert: 생성 성공
            Assert.IsTrue(result.success, "256 해상도 미니맵 도시 생성 실패");
            Assert.Greater(result.buildingCount, 0, "건물이 생성되지 않음");

            Debug.Log($"[미니맵 256] 도시 생성 성공: 건물 {result.buildingCount}개");
        }

        [Test]
        public void Scenario_MinimapResolution512_GeneratesSuccessfully()
        {
            // Arrange: 512x512 해상도 미니맵
            cityGenerator.unitDistance = 1.0f;
            cityGenerator.minWidth = 15;
            cityGenerator.maxWidth = 15;
            cityGenerator.minDepth = 15;
            cityGenerator.maxDepth = 15;
            cityGenerator.buildingDensity = 0.7f;
            cityGenerator.randomSeed = 5002;
            cityGenerator.minimapResolution = MinimapResolution.Resolution512;

            // Act: 도시 생성
            CityGenerationResult result = cityGenerator.GenerateCity();

            // Assert: 생성 성공
            Assert.IsTrue(result.success, "512 해상도 미니맵 도시 생성 실패");
            Assert.Greater(result.buildingCount, 0, "건물이 생성되지 않음");

            Debug.Log($"[미니맵 512] 도시 생성 성공: 건물 {result.buildingCount}개");
        }

        [Test]
        public void Scenario_MinimapResolution1024_GeneratesSuccessfully()
        {
            // Arrange: 1024x1024 해상도 미니맵
            cityGenerator.unitDistance = 1.0f;
            cityGenerator.minWidth = 20;
            cityGenerator.maxWidth = 20;
            cityGenerator.minDepth = 20;
            cityGenerator.maxDepth = 20;
            cityGenerator.buildingDensity = 0.7f;
            cityGenerator.randomSeed = 5003;
            cityGenerator.minimapResolution = MinimapResolution.Resolution1024;

            // Act: 도시 생성
            CityGenerationResult result = cityGenerator.GenerateCity();

            // Assert: 생성 성공
            Assert.IsTrue(result.success, "1024 해상도 미니맵 도시 생성 실패");
            Assert.Greater(result.buildingCount, 0, "건물이 생성되지 않음");

            Debug.Log($"[미니맵 1024] 도시 생성 성공: 건물 {result.buildingCount}개");
        }

        // ===== 랜덤 시드 재현성 테스트 =====
        [Test]
        public void Scenario_FixedSeed_GeneratesIdenticalCities()
        {
            // Arrange: 고정 시드로 두 번 생성
            cityGenerator.unitDistance = 1.0f;
            cityGenerator.minWidth = 12;
            cityGenerator.maxWidth = 12;
            cityGenerator.minDepth = 12;
            cityGenerator.maxDepth = 12;
            cityGenerator.buildingDensity = 0.7f;
            cityGenerator.randomSeed = 6001; // 고정 시드

            // Act: 첫 번째 생성
            CityGenerationResult result1 = cityGenerator.GenerateCity();
            int buildingCount1 = result1.buildingCount;
            int nodeCount1 = result1.nodeCount;

            // 도시 제거
            cityGenerator.ClearCity();

            // 두 번째 생성 (동일한 시드)
            CityGenerationResult result2 = cityGenerator.GenerateCity();
            int buildingCount2 = result2.buildingCount;
            int nodeCount2 = result2.nodeCount;

            // Assert: 동일한 결과
            Assert.AreEqual(buildingCount1, buildingCount2, "건물 수가 일치하지 않음");
            Assert.AreEqual(nodeCount1, nodeCount2, "노드 수가 일치하지 않음");

            Debug.Log($"[시드 재현성] 동일한 도시 생성 성공: 건물 {buildingCount1}개");
        }

        [Test]
        public void Scenario_RandomSeed_GeneratesDifferentCities()
        {
            // Arrange: 랜덤 시드로 두 번 생성
            cityGenerator.unitDistance = 1.0f;
            cityGenerator.minWidth = 12;
            cityGenerator.maxWidth = 12;
            cityGenerator.minDepth = 12;
            cityGenerator.maxDepth = 12;
            cityGenerator.buildingDensity = 0.7f;
            cityGenerator.randomSeed = -1; // 랜덤 시드

            // Act: 첫 번째 생성
            CityGenerationResult result1 = cityGenerator.GenerateCity();
            int seed1 = result1.usedRandomSeed;

            // 도시 제거
            cityGenerator.ClearCity();

            // 두 번째 생성 (랜덤 시드)
            CityGenerationResult result2 = cityGenerator.GenerateCity();
            int seed2 = result2.usedRandomSeed;

            // Assert: 다른 시드가 사용됨
            Assert.AreNotEqual(seed1, seed2, "랜덤 시드가 동일함");

            Debug.Log($"[랜덤 시드] 다른 시드로 도시 생성: 시드1={seed1}, 시드2={seed2}");
        }

        // ===== 도시 제거 및 재생성 테스트 =====
        [Test]
        public void Scenario_ClearAndRegenerate_WorksCorrectly()
        {
            // Arrange & Act: 도시 생성
            cityGenerator.unitDistance = 1.0f;
            cityGenerator.minWidth = 10;
            cityGenerator.maxWidth = 10;
            cityGenerator.minDepth = 10;
            cityGenerator.maxDepth = 10;
            cityGenerator.buildingDensity = 0.7f;
            cityGenerator.randomSeed = 7001;

            CityGenerationResult result1 = cityGenerator.GenerateCity();
            Assert.IsTrue(result1.success, "첫 번째 도시 생성 실패");

            // 도시 제거
            cityGenerator.ClearCity();

            // 씬에서 도시가 제거되었는지 확인
            GameObject cityRoot = GameObject.Find("City");
            Assert.IsNull(cityRoot, "도시가 제거되지 않음");

            // 다시 생성
            CityGenerationResult result2 = cityGenerator.GenerateCity();
            Assert.IsTrue(result2.success, "두 번째 도시 생성 실패");

            // 새로운 도시가 생성되었는지 확인
            cityRoot = GameObject.Find("City");
            Assert.IsNotNull(cityRoot, "새로운 도시가 생성되지 않음");

            Debug.Log($"[제거 및 재생성] 성공: 첫 번째 {result1.buildingCount}개, 두 번째 {result2.buildingCount}개");
        }

        // ===== 연속 생성 테스트 =====
        [Test]
        public void Scenario_ConsecutiveGenerations_WorkCorrectly()
        {
            // Arrange: 여러 번 연속으로 도시 생성
            int[] seeds = new int[] { 8001, 8002, 8003, 8004, 8005 };
            int[] buildingCounts = new int[seeds.Length];

            // Act: 연속 생성
            for (int i = 0; i < seeds.Length; i++)
            {
                cityGenerator.randomSeed = seeds[i];
                CityGenerationResult result = cityGenerator.GenerateCity();
                
                Assert.IsTrue(result.success, $"생성 {i + 1} 실패");
                buildingCounts[i] = result.buildingCount;
            }

            // Assert: 모든 생성이 성공했는지 확인
            for (int i = 0; i < buildingCounts.Length; i++)
            {
                Assert.Greater(buildingCounts[i], 0, $"생성 {i + 1}에서 건물이 생성되지 않음");
            }

            Debug.Log($"[연속 생성] {seeds.Length}번 연속 생성 성공");
        }

        // ===== 파라미터 변경 후 재생성 테스트 =====
        [Test]
        public void Scenario_ParameterChange_AffectsGeneration()
        {
            // Arrange: 낮은 밀도로 생성
            cityGenerator.unitDistance = 1.0f;
            cityGenerator.minWidth = 10;
            cityGenerator.maxWidth = 10;
            cityGenerator.minDepth = 10;
            cityGenerator.maxDepth = 10;
            cityGenerator.buildingDensity = 0.3f; // 낮은 밀도
            cityGenerator.randomSeed = 9001;

            // Act: 첫 번째 생성
            CityGenerationResult result1 = cityGenerator.GenerateCity();
            int lowDensityCount = result1.buildingCount;

            // 밀도를 높여서 재생성
            cityGenerator.buildingDensity = 0.9f; // 높은 밀도
            CityGenerationResult result2 = cityGenerator.GenerateCity();
            int highDensityCount = result2.buildingCount;

            // Assert: 높은 밀도에서 더 많은 건물이 생성됨
            Assert.Greater(highDensityCount, lowDensityCount, 
                "높은 밀도에서 더 많은 건물이 생성되어야 함");

            Debug.Log($"[파라미터 변경] 낮은 밀도: {lowDensityCount}개, 높은 밀도: {highDensityCount}개");
        }

        // ===== 그래프 생성 검증 테스트 =====
        [Test]
        public void Scenario_GraphGeneration_CreatesValidGraph()
        {
            // Arrange: 도시 생성
            cityGenerator.unitDistance = 1.0f;
            cityGenerator.minWidth = 10;
            cityGenerator.maxWidth = 10;
            cityGenerator.minDepth = 10;
            cityGenerator.maxDepth = 10;
            cityGenerator.buildingDensity = 0.7f;
            cityGenerator.randomSeed = 10001;

            // Act: 도시 생성
            CityGenerationResult result = cityGenerator.GenerateCity();

            // Assert: 그래프가 올바르게 생성되었는지 확인
            Assert.IsTrue(result.success, "도시 생성 실패");
            Assert.IsNotNull(result.graph, "그래프가 null");
            Assert.Greater(result.nodeCount, 0, "노드가 생성되지 않음");
            Assert.Greater(result.edgeCount, 0, "엣지가 생성되지 않음");

            // 그래프에서 노드를 가져올 수 있는지 확인
            var allNodes = result.graph.GetAllNodes();
            Assert.IsNotNull(allNodes, "GetAllNodes()가 null 반환");
            Assert.AreEqual(result.nodeCount, allNodes.Count, "노드 수가 일치하지 않음");

            Debug.Log($"[그래프 생성] 노드 {result.nodeCount}개, 엣지 {result.edgeCount}개");
        }

        // ===== 전략적 위치 분석 검증 테스트 =====
        [Test]
        public void Scenario_StrategicLocationAnalysis_CreatesMarkers()
        {
            // Arrange: 도시 생성
            cityGenerator.unitDistance = 1.0f;
            cityGenerator.minWidth = 15;
            cityGenerator.maxWidth = 15;
            cityGenerator.minDepth = 15;
            cityGenerator.maxDepth = 15;
            cityGenerator.buildingDensity = 0.7f;
            cityGenerator.randomSeed = 11001;

            // Act: 도시 생성
            CityGenerationResult result = cityGenerator.GenerateCity();

            // Assert: 전략적 위치 마커가 생성되었는지 확인
            Assert.IsTrue(result.success, "도시 생성 실패");
            Assert.IsNotNull(result.graph, "그래프가 null");

            var allNodes = result.graph.GetAllNodes();
            int nodesWithMarkers = 0;
            foreach (var node in allNodes)
            {
                if (node.strategicMarkers != null && node.strategicMarkers.Count > 0)
                {
                    nodesWithMarkers++;
                }
            }

            Assert.Greater(nodesWithMarkers, 0, "전략적 마커가 있는 노드가 없음");

            Debug.Log($"[전략적 위치] 전체 노드 {allNodes.Count}개 중 {nodesWithMarkers}개에 마커 존재");
        }

        // ===== 건물 계층 구조 검증 테스트 =====
        [Test]
        public void Scenario_BuildingHierarchy_IsCorrect()
        {
            // Arrange: 도시 생성
            cityGenerator.unitDistance = 1.0f;
            cityGenerator.minWidth = 8;
            cityGenerator.maxWidth = 8;
            cityGenerator.minDepth = 8;
            cityGenerator.maxDepth = 8;
            cityGenerator.buildingDensity = 0.7f;
            cityGenerator.randomSeed = 12001;

            // Act: 도시 생성
            CityGenerationResult result = cityGenerator.GenerateCity();

            // Assert: 계층 구조 확인
            Assert.IsTrue(result.success, "도시 생성 실패");

            GameObject cityRoot = GameObject.Find("City");
            Assert.IsNotNull(cityRoot, "City GameObject가 없음");
            Assert.Greater(cityRoot.transform.childCount, 0, "건물이 자식으로 추가되지 않음");

            // 첫 번째 건물 확인
            Transform firstBuilding = cityRoot.transform.GetChild(0);
            Assert.IsNotNull(firstBuilding, "첫 번째 건물이 null");
            Assert.IsTrue(firstBuilding.name.StartsWith("Building_"), "건물 이름 형식이 올바르지 않음");

            // 건물 컴포넌트 확인
            Assert.IsNotNull(firstBuilding.GetComponent<MeshRenderer>(), "MeshRenderer가 없음");
            Assert.IsNotNull(firstBuilding.GetComponent<MeshFilter>(), "MeshFilter가 없음");
            Assert.IsNotNull(firstBuilding.GetComponent<BoxCollider>(), "BoxCollider가 없음");

            Debug.Log($"[계층 구조] City 아래 {cityRoot.transform.childCount}개 건물 존재");
        }

        // ===== 종합 시나리오 테스트 =====
        [Test]
        public void Scenario_CompleteWorkflow_WorksEndToEnd()
        {
            // Arrange: 파라미터 설정
            cityGenerator.unitDistance = 1.5f;
            cityGenerator.minWidth = 12;
            cityGenerator.maxWidth = 15;
            cityGenerator.minDepth = 12;
            cityGenerator.maxDepth = 15;
            cityGenerator.buildingWidth = 1.2f;
            cityGenerator.buildingDepth = 1.2f;
            cityGenerator.minBuildingHeight = 8.0f;
            cityGenerator.maxBuildingHeight = 30.0f;
            cityGenerator.buildingSpacing = 1.5f;
            cityGenerator.buildingDensity = 0.7f;
            cityGenerator.randomSeed = 13001;
            cityGenerator.minimapResolution = MinimapResolution.Resolution512;

            // Act 1: 도시 생성
            CityGenerationResult result1 = cityGenerator.GenerateCity();
            Assert.IsTrue(result1.success, "도시 생성 실패");

            // Act 2: 프리셋 저장
            string presetName = "TestScenario_CompleteWorkflow";
            cityGenerator.SavePreset(presetName);

            // Act 3: 도시 제거
            cityGenerator.ClearCity();
            GameObject cityRoot = GameObject.Find("City");
            Assert.IsNull(cityRoot, "도시가 제거되지 않음");

            // Act 4: 프리셋 로드
            string assetPath = $"Assets/CityPresets/{presetName}.asset";
            CityParameters preset = AssetDatabase.LoadAssetAtPath<CityParameters>(assetPath);
            Assert.IsNotNull(preset, "프리셋이 저장되지 않음");
            cityGenerator.LoadPreset(preset);

            // Act 5: 도시 재생성
            CityGenerationResult result2 = cityGenerator.GenerateCity();
            Assert.IsTrue(result2.success, "도시 재생성 실패");

            // Assert: 동일한 도시가 생성됨
            Assert.AreEqual(result1.buildingCount, result2.buildingCount, "건물 수가 일치하지 않음");
            Assert.AreEqual(result1.nodeCount, result2.nodeCount, "노드 수가 일치하지 않음");

            Debug.Log($"[종합 시나리오] 전체 워크플로우 성공: 건물 {result2.buildingCount}개");

            // Cleanup
            if (AssetDatabase.LoadAssetAtPath<CityParameters>(assetPath) != null)
            {
                AssetDatabase.DeleteAsset(assetPath);
            }
        }
    }
}
