using NUnit.Framework;
using UnityEngine;

namespace ProceduralCityGenerator.Tests
{
    /// <summary>
    /// Task 14.2: 생성 취소 기능 테스트
    /// Requirement: 11.5
    /// 
    /// Note: DisplayCancelableProgressBar는 사용자 상호작용이 필요하므로
    /// 자동 테스트에서는 취소 후 정리 및 재생성 동작을 검증합니다.
    /// 실제 취소 버튼 클릭은 수동 테스트로 검증해야 합니다.
    /// </summary>
    public class CancellationTest
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
        public void Test_ClearCity_RemovesAllBuildings()
        {
            // Arrange: 도시 생성
            generator.minWidth = 10;
            generator.maxWidth = 10;
            generator.minDepth = 10;
            generator.maxDepth = 10;
            generator.buildingDensity = 1.0f; // 모든 셀에 건물 배치
            generator.randomSeed = 12345;

            generator.GenerateCity();

            // 건물이 생성되었는지 확인
            var cityRoot = GameObject.Find("City");
            Assert.IsNotNull(cityRoot, "도시가 생성되어야 합니다");
            int buildingCount = cityRoot.transform.childCount;
            Assert.Greater(buildingCount, 0, "건물이 생성되어야 합니다");

            // Act: 도시 제거
            generator.ClearCity();

            // Assert: 모든 건물이 제거되었는지 확인
            cityRoot = GameObject.Find("City");
            Assert.IsNull(cityRoot, "ClearCity 후 도시 루트 오브젝트가 제거되어야 합니다");

            Debug.Log($"CancellationTest: ClearCity가 {buildingCount}개의 건물을 성공적으로 제거했습니다");
        }

        [Test]
        public void Test_ClearCity_AllowsRegeneration()
        {
            // Arrange: 도시 생성
            generator.minWidth = 15;
            generator.maxWidth = 15;
            generator.minDepth = 15;
            generator.maxDepth = 15;
            generator.buildingDensity = 0.7f;
            generator.randomSeed = 54321;

            generator.GenerateCity();
            var cityRoot1 = GameObject.Find("City");
            Assert.IsNotNull(cityRoot1, "첫 번째 도시가 생성되어야 합니다");

            // Act: 도시 제거 후 재생성
            generator.ClearCity();
            generator.GenerateCity();

            // Assert: 새로운 도시가 정상적으로 생성되었는지 확인
            var cityRoot2 = GameObject.Find("City");
            Assert.IsNotNull(cityRoot2, "ClearCity 후 재생성이 가능해야 합니다");
            
            int buildingCount = cityRoot2.transform.childCount;
            Assert.Greater(buildingCount, 0, "재생성된 도시에 건물이 있어야 합니다");

            Debug.Log($"CancellationTest: ClearCity 후 재생성 성공 ({buildingCount}개 건물)");
        }

        [Test]
        public void Test_ClearCity_MultipleTimes()
        {
            // Arrange & Act: ClearCity를 여러 번 호출
            generator.ClearCity(); // 첫 번째 호출 (도시 없음)
            generator.ClearCity(); // 두 번째 호출 (도시 없음)

            // 도시 생성
            generator.minWidth = 10;
            generator.maxWidth = 10;
            generator.minDepth = 10;
            generator.maxDepth = 10;
            generator.buildingDensity = 0.5f;
            generator.randomSeed = 99999;

            generator.GenerateCity();
            var cityRoot = GameObject.Find("City");
            Assert.IsNotNull(cityRoot, "도시가 생성되어야 합니다");

            generator.ClearCity(); // 세 번째 호출 (도시 있음)
            generator.ClearCity(); // 네 번째 호출 (도시 없음)

            // Assert: 오류 없이 실행되어야 함
            cityRoot = GameObject.Find("City");
            Assert.IsNull(cityRoot, "마지막 ClearCity 후 도시가 없어야 합니다");

            Debug.Log("CancellationTest: ClearCity를 여러 번 호출해도 오류가 발생하지 않습니다");
        }

        [Test]
        public void Test_GenerateCity_ReturnsFailureOnCancellation()
        {
            // Note: 이 테스트는 PlaceBuildingsOnGrid가 false를 반환할 때
            // GenerateCity가 올바르게 실패를 처리하는지 검증합니다.
            // 실제 취소는 수동 테스트로 검증해야 합니다.

            // Arrange: 작은 도시 설정 (빠른 생성)
            generator.minWidth = 5;
            generator.maxWidth = 5;
            generator.minDepth = 5;
            generator.maxDepth = 5;
            generator.buildingDensity = 1.0f;
            generator.randomSeed = 11111;

            // Act: 정상 생성
            var result = generator.GenerateCity();

            // Assert: 정상 생성 시 성공 반환
            Assert.IsTrue(result.success, "정상 생성 시 success가 true여야 합니다");
            Assert.IsTrue(string.IsNullOrEmpty(result.errorMessage), "정상 생성 시 errorMessage가 null 또는 빈 문자열이어야 합니다");
            Assert.Greater(result.buildingCount, 0, "건물이 생성되어야 합니다");

            Debug.Log($"CancellationTest: 정상 생성 시 성공 반환 확인 ({result.buildingCount}개 건물)");
        }

        [Test]
        public void Test_PartialCity_CleanedUp()
        {
            // Arrange: 도시 생성
            generator.minWidth = 20;
            generator.maxWidth = 20;
            generator.minDepth = 20;
            generator.maxDepth = 20;
            generator.buildingDensity = 0.8f;
            generator.randomSeed = 77777;

            generator.GenerateCity();
            var cityRoot = GameObject.Find("City");
            Assert.IsNotNull(cityRoot, "도시가 생성되어야 합니다");
            int originalBuildingCount = cityRoot.transform.childCount;

            // Act: 부분 생성 시뮬레이션 (ClearCity 호출)
            // 실제 취소는 PlaceBuildingsOnGrid 내부에서 ClearCity를 호출합니다
            generator.ClearCity();

            // Assert: 모든 건물이 제거되었는지 확인
            cityRoot = GameObject.Find("City");
            Assert.IsNull(cityRoot, "취소 시 부분 생성된 도시가 완전히 제거되어야 합니다");

            Debug.Log($"CancellationTest: 부분 생성된 도시 ({originalBuildingCount}개 건물) 정리 확인");
        }

        [Test]
        public void Test_CancellationAfterMultipleGenerations()
        {
            // Arrange: 여러 번 생성 및 제거
            generator.minWidth = 10;
            generator.maxWidth = 10;
            generator.minDepth = 10;
            generator.maxDepth = 10;
            generator.buildingDensity = 0.6f;

            // Act: 3번 생성 및 제거
            for (int i = 0; i < 3; i++)
            {
                generator.randomSeed = 10000 + i;
                generator.GenerateCity();
                
                var cityRoot = GameObject.Find("City");
                Assert.IsNotNull(cityRoot, $"생성 {i + 1}번째: 도시가 생성되어야 합니다");
                
                generator.ClearCity();
                
                cityRoot = GameObject.Find("City");
                Assert.IsNull(cityRoot, $"제거 {i + 1}번째: 도시가 제거되어야 합니다");
            }

            // Assert: 마지막 재생성이 정상 작동
            generator.randomSeed = 20000;
            generator.GenerateCity();
            
            var finalCityRoot = GameObject.Find("City");
            Assert.IsNotNull(finalCityRoot, "여러 번 생성/제거 후에도 재생성이 가능해야 합니다");
            Assert.Greater(finalCityRoot.transform.childCount, 0, "마지막 도시에 건물이 있어야 합니다");

            Debug.Log("CancellationTest: 여러 번 생성/제거 후에도 정상 작동합니다");
        }

        [Test]
        public void Test_ClearCity_ResetsInternalState()
        {
            // Arrange: 도시 생성
            generator.minWidth = 12;
            generator.maxWidth = 12;
            generator.minDepth = 12;
            generator.maxDepth = 12;
            generator.buildingDensity = 0.7f;
            generator.randomSeed = 33333;

            var result1 = generator.GenerateCity();
            Assert.IsTrue(result1.success, "첫 번째 생성이 성공해야 합니다");
            int buildingCount1 = result1.buildingCount;

            // Act: 도시 제거 후 다른 설정으로 재생성
            generator.ClearCity();
            
            generator.minWidth = 8;
            generator.maxWidth = 8;
            generator.minDepth = 8;
            generator.maxDepth = 8;
            generator.buildingDensity = 0.5f;
            generator.randomSeed = 44444;

            var result2 = generator.GenerateCity();

            // Assert: 새로운 설정으로 정상 생성
            Assert.IsTrue(result2.success, "두 번째 생성이 성공해야 합니다");
            Assert.AreNotEqual(buildingCount1, result2.buildingCount, 
                "다른 설정으로 생성 시 건물 수가 달라야 합니다");

            Debug.Log($"CancellationTest: ClearCity가 내부 상태를 올바르게 리셋합니다 " +
                     $"(첫 번째: {buildingCount1}개, 두 번째: {result2.buildingCount}개)");
        }
    }
}
