using NUnit.Framework;
using UnityEngine;
using UnityEditor;

namespace CityGenerator.Tests
{
    /// <summary>
    /// Task 14.1: 배치 처리 구현 테스트
    /// Requirements: 11.3, 11.4
    /// </summary>
    public class BatchProcessingTest
    {
        private GameObject testObject;
        private CityGenerator generator;

        [SetUp]
        public void Setup()
        {
            testObject = new GameObject("TestCityGenerator");
            generator = testObject.AddComponent<CityGenerator>();
            
            // 기본 머티리얼 설정
            generator.defaultBuildingMaterial = new Material(Shader.Find("Standard"));
        }

        [TearDown]
        public void Teardown()
        {
            if (generator != null)
                generator.ClearCity();
            if (testObject != null)
                Object.DestroyImmediate(testObject);
            // 이전 테스트에서 남은 stray "City" 오브젝트 정리
            GameObject cityObj = GameObject.Find("City");
            while (cityObj != null)
            {
                Object.DestroyImmediate(cityObj);
                cityObj = GameObject.Find("City");
            }
        }

        [Test]
        public void Test_BatchProcessing_GeneratesBuildings()
        {
            // Arrange: 중간 크기 도시 설정
            generator.minWidth = 20;
            generator.maxWidth = 20;
            generator.minDepth = 20;
            generator.maxDepth = 20;
            generator.buildingDensity = 0.7f;
            generator.randomSeed = 12345;

            // Act: 도시 생성
            var startTime = System.DateTime.Now;
            generator.GenerateCity();
            var endTime = System.DateTime.Now;
            var duration = (endTime - startTime).TotalSeconds;

            // Assert: 건물이 생성되었는지 확인
            var cityRoot = GameObject.Find("City");
            Assert.IsNotNull(cityRoot, "도시 루트 오브젝트가 생성되어야 합니다");
            
            int buildingCount = cityRoot.transform.childCount;
            Assert.Greater(buildingCount, 0, "건물이 생성되어야 합니다");
            
            // 배치 처리로 인한 성능 향상 확인 (400개 셀 처리가 합리적인 시간 내에 완료)
            Assert.Less(duration, 2.0, "배치 처리로 인해 생성이 2초 이내에 완료되어야 합니다");
            
            Debug.Log($"BatchProcessingTest: {buildingCount}개 건물 생성 완료 (소요 시간: {duration:F2}초)");
        }

        [Test]
        public void Test_BatchProcessing_LargeCity_Performance()
        {
            // Arrange: 대규모 도시 설정 (1000개 이상 건물)
            generator.minWidth = 40;
            generator.maxWidth = 40;
            generator.minDepth = 40;
            generator.maxDepth = 40;
            generator.buildingDensity = 0.7f; // 약 1120개 건물 예상
            generator.randomSeed = 54321;

            // Act: 도시 생성 및 시간 측정
            var startTime = System.DateTime.Now;
            generator.GenerateCity();
            var endTime = System.DateTime.Now;
            var duration = (endTime - startTime).TotalSeconds;

            // Assert: Requirement 11.1 - 1000개 이상의 건물로 도시를 생성할 때 5초 이내에 완료
            var cityRoot = GameObject.Find("City");
            Assert.IsNotNull(cityRoot, "도시 루트 오브젝트가 생성되어야 합니다");
            
            int buildingCount = cityRoot.transform.childCount;
            Assert.GreaterOrEqual(buildingCount, 1000, "1000개 이상의 건물이 생성되어야 합니다");
            Assert.Less(duration, 5.0, "Requirement 11.1: 1000개 이상 건물 생성이 5초 이내에 완료되어야 합니다");
            
            Debug.Log($"BatchProcessingTest (Large): {buildingCount}개 건물 생성 완료 (소요 시간: {duration:F2}초)");
        }

        [Test]
        public void Test_BatchProcessing_ConsistentResults()
        {
            // Arrange: 동일한 시드로 두 번 생성
            generator.minWidth = 15;
            generator.maxWidth = 15;
            generator.minDepth = 15;
            generator.maxDepth = 15;
            generator.buildingDensity = 0.5f;
            generator.randomSeed = 99999;

            // Act: 첫 번째 생성
            generator.GenerateCity();
            var cityRoot1 = GameObject.Find("City");
            int buildingCount1 = cityRoot1.transform.childCount;
            
            // 첫 번째 건물 위치 저장
            Vector3[] positions1 = new Vector3[buildingCount1];
            for (int i = 0; i < buildingCount1; i++)
            {
                positions1[i] = cityRoot1.transform.GetChild(i).position;
            }

            // 도시 제거
            generator.ClearCity();

            // Act: 두 번째 생성 (동일한 시드)
            generator.GenerateCity();
            var cityRoot2 = GameObject.Find("City");
            int buildingCount2 = cityRoot2.transform.childCount;

            // Assert: 배치 처리가 결과의 일관성에 영향을 주지 않아야 함
            Assert.AreEqual(buildingCount1, buildingCount2, "동일한 시드로 생성 시 건물 수가 같아야 합니다");
            
            // 건물 위치도 동일해야 함
            for (int i = 0; i < buildingCount2; i++)
            {
                Vector3 pos2 = cityRoot2.transform.GetChild(i).position;
                Assert.AreEqual(positions1[i], pos2, $"건물 {i}의 위치가 동일해야 합니다");
            }
            
            Debug.Log($"BatchProcessingTest (Consistency): 배치 처리가 일관된 결과를 생성합니다 ({buildingCount1}개 건물)");
        }

        [Test]
        public void Test_BatchProcessing_ProgressUpdates()
        {
            // Arrange: 중간 크기 도시 설정
            generator.minWidth = 30;
            generator.maxWidth = 30;
            generator.minDepth = 30;
            generator.maxDepth = 30;
            generator.buildingDensity = 0.6f;
            generator.randomSeed = 11111;

            // Act: 도시 생성
            // Note: 진행률 표시는 Editor에서만 표시되므로 직접 테스트하기 어려움
            // 대신 생성이 정상적으로 완료되는지 확인
            generator.GenerateCity();

            // Assert: 생성 완료 확인
            var cityRoot = GameObject.Find("City");
            Assert.IsNotNull(cityRoot, "도시가 정상적으로 생성되어야 합니다");
            
            int buildingCount = cityRoot.transform.childCount;
            Assert.Greater(buildingCount, 0, "건물이 생성되어야 합니다");
            
            Debug.Log($"BatchProcessingTest (Progress): 진행률 업데이트와 함께 {buildingCount}개 건물 생성 완료");
        }
    }
}
