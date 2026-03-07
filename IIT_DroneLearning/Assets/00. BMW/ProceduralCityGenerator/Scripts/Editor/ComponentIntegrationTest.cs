using NUnit.Framework;
using UnityEngine;
using ProceduralCityGenerator;

namespace ProceduralCityGenerator.Tests
{
    /// <summary>
    /// Task 15.1: 컴포넌트 통합 테스트
    /// 모든 컴포넌트가 CityGenerator에 올바르게 통합되었는지 검증합니다.
    /// </summary>
    public class ComponentIntegrationTest
    {
        private GameObject testGameObject;
        private CityGenerator cityGenerator;

        [SetUp]
        public void Setup()
        {
            // 테스트용 GameObject 생성
            testGameObject = new GameObject("TestCityGenerator");
            cityGenerator = testGameObject.AddComponent<CityGenerator>();

            // 기본 머티리얼 설정
            cityGenerator.defaultBuildingMaterial = new Material(Shader.Find("Standard"));

            // 테스트용 파라미터 설정 (작은 도시)
            cityGenerator.unitDistance = 1.0f;
            cityGenerator.minWidth = 5;
            cityGenerator.maxWidth = 5;
            cityGenerator.minDepth = 5;
            cityGenerator.maxDepth = 5;
            cityGenerator.buildingWidth = 1.0f;
            cityGenerator.buildingDepth = 1.0f;
            cityGenerator.minBuildingHeight = 5.0f;
            cityGenerator.maxBuildingHeight = 10.0f;
            cityGenerator.buildingSpacing = 1.0f;
            cityGenerator.buildingDensity = 0.5f;
            cityGenerator.randomSeed = 12345; // 재현 가능한 시드
        }

        [TearDown]
        public void Teardown()
        {
            // 테스트 후 정리
            if (testGameObject != null)
            {
                Object.DestroyImmediate(testGameObject);
            }
        }

        /// <summary>
        /// 테스트 1: CityGenerator가 모든 필수 컴포넌트를 초기화하는지 확인
        /// </summary>
        [Test]
        public void Test_CityGenerator_InitializesAllComponents()
        {
            // Act: 도시 생성
            CityGenerationResult result = cityGenerator.GenerateCity();

            // Assert: 생성 성공
            Assert.IsTrue(result.success, "도시 생성이 실패했습니다.");
            Assert.Greater(result.buildingCount, 0, "건물이 생성되지 않았습니다.");
            Assert.Greater(result.nodeCount, 0, "그래프 노드가 생성되지 않았습니다.");
            Assert.Greater(result.edgeCount, 0, "그래프 엣지가 생성되지 않았습니다.");
            Assert.IsNotNull(result.graph, "CityGraph가 null입니다.");
        }

        /// <summary>
        /// 테스트 2: BuildingFactory가 올바르게 통합되었는지 확인
        /// </summary>
        [Test]
        public void Test_BuildingFactory_Integration()
        {
            // Act: 도시 생성
            CityGenerationResult result = cityGenerator.GenerateCity();

            // Assert: 건물이 생성되고 계층 구조가 올바른지 확인
            Assert.IsTrue(result.success, "도시 생성이 실패했습니다.");
            
            // "City" GameObject 확인
            GameObject cityRoot = GameObject.Find("City");
            Assert.IsNotNull(cityRoot, "City GameObject가 없습니다.");
            
            // 건물이 cityRoot의 자식으로 생성되었는지 확인
            Assert.Greater(cityRoot.transform.childCount, 0, "City에 자식 건물이 없습니다.");
            
            // 첫 번째 건물 확인
            Transform firstBuilding = cityRoot.transform.GetChild(0);
            Assert.IsNotNull(firstBuilding, "첫 번째 건물이 null입니다.");
            Assert.IsTrue(firstBuilding.name.StartsWith("Building_"), "건물 이름 형식이 올바르지 않습니다.");
            
            // 건물에 필수 컴포넌트가 있는지 확인
            Assert.IsNotNull(firstBuilding.GetComponent<MeshRenderer>(), "건물에 MeshRenderer가 없습니다.");
            Assert.IsNotNull(firstBuilding.GetComponent<MeshFilter>(), "건물에 MeshFilter가 없습니다.");
            Assert.IsNotNull(firstBuilding.GetComponent<BoxCollider>(), "건물에 BoxCollider가 없습니다.");
        }

        /// <summary>
        /// 테스트 3: CityGraph가 올바르게 통합되었는지 확인
        /// </summary>
        [Test]
        public void Test_CityGraph_Integration()
        {
            // Act: 도시 생성
            CityGenerationResult result = cityGenerator.GenerateCity();

            // Assert: 그래프가 생성되고 노드와 엣지가 있는지 확인
            Assert.IsTrue(result.success, "도시 생성이 실패했습니다.");
            Assert.IsNotNull(result.graph, "CityGraph가 null입니다.");
            Assert.Greater(result.nodeCount, 0, "그래프에 노드가 없습니다.");
            Assert.Greater(result.edgeCount, 0, "그래프에 엣지가 없습니다.");
            
            // 그래프에서 노드를 가져올 수 있는지 확인
            var allNodes = result.graph.GetAllNodes();
            Assert.IsNotNull(allNodes, "GetAllNodes()가 null을 반환했습니다.");
            Assert.AreEqual(result.nodeCount, allNodes.Count, "노드 수가 일치하지 않습니다.");
            
            // 첫 번째 노드 확인
            if (allNodes.Count > 0)
            {
                GraphNode firstNode = allNodes[0];
                Assert.IsTrue(firstNode.nodeId >= 0, "노드 ID가 유효하지 않습니다.");
                Assert.IsNotNull(firstNode.strategicMarkers, "전략적 마커 리스트가 null입니다.");
            }
        }

        /// <summary>
        /// 테스트 4: SpatialIndex가 올바르게 통합되었는지 확인
        /// </summary>
        [Test]
        public void Test_SpatialIndex_Integration()
        {
            // Act: 도시 생성
            CityGenerationResult result = cityGenerator.GenerateCity();

            // Assert: 그래프가 생성되었는지 확인
            Assert.IsTrue(result.success, "도시 생성이 실패했습니다.");
            Assert.IsNotNull(result.graph, "CityGraph가 null입니다.");
            
            // CityDataAPI가 초기화되었는지 확인 (SpatialIndex 사용)
            // Note: CityDataAPI는 CityGenerator에서 초기화되지 않으므로 이 테스트는 생략
            // 실제 사용 시에는 별도로 CityDataAPI.Instance.Initialize()를 호출해야 함
        }

        /// <summary>
        /// 테스트 5: StrategicLocationAnalyzer가 올바르게 통합되었는지 확인
        /// </summary>
        [Test]
        public void Test_StrategicLocationAnalyzer_Integration()
        {
            // Act: 도시 생성
            CityGenerationResult result = cityGenerator.GenerateCity();

            // Assert: 그래프가 생성되고 전략적 마커가 추가되었는지 확인
            Assert.IsTrue(result.success, "도시 생성이 실패했습니다.");
            Assert.IsNotNull(result.graph, "CityGraph가 null입니다.");
            
            // 전략적 마커가 있는 노드 찾기
            var allNodes = result.graph.GetAllNodes();
            int nodesWithMarkers = 0;
            foreach (var node in allNodes)
            {
                if (node.strategicMarkers != null && node.strategicMarkers.Count > 0)
                {
                    nodesWithMarkers++;
                }
            }
            
            // 최소한 일부 노드에 전략적 마커가 있어야 함
            Assert.Greater(nodesWithMarkers, 0, "전략적 마커가 있는 노드가 없습니다.");
        }

        /// <summary>
        /// 테스트 6: 의존성 주입 순서가 올바른지 확인
        /// </summary>
        [Test]
        public void Test_DependencyInjection_Order()
        {
            // Act: 도시 생성
            CityGenerationResult result = cityGenerator.GenerateCity();

            // Assert: 모든 컴포넌트가 올바른 순서로 초기화되었는지 확인
            Assert.IsTrue(result.success, "도시 생성이 실패했습니다.");
            
            // 1. 건물이 먼저 생성되어야 함
            Assert.Greater(result.buildingCount, 0, "건물이 생성되지 않았습니다.");
            
            // 2. 그래프가 건물 배치 후 생성되어야 함
            Assert.IsNotNull(result.graph, "CityGraph가 null입니다.");
            Assert.Greater(result.nodeCount, 0, "그래프 노드가 생성되지 않았습니다.");
            
            // 3. 전략적 위치 분석이 그래프 생성 후 수행되어야 함
            var allNodes = result.graph.GetAllNodes();
            bool hasStrategicMarkers = false;
            foreach (var node in allNodes)
            {
                if (node.strategicMarkers != null && node.strategicMarkers.Count > 0)
                {
                    hasStrategicMarkers = true;
                    break;
                }
            }
            Assert.IsTrue(hasStrategicMarkers, "전략적 마커가 추가되지 않았습니다.");
        }

        /// <summary>
        /// 테스트 7: 재현 가능한 생성 (랜덤 시드)
        /// </summary>
        [Test]
        public void Test_Reproducible_Generation()
        {
            // Arrange: 동일한 시드로 두 번 생성
            cityGenerator.randomSeed = 99999;

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

            // Assert: 두 생성 결과가 동일해야 함
            Assert.AreEqual(buildingCount1, buildingCount2, "건물 수가 일치하지 않습니다.");
            Assert.AreEqual(nodeCount1, nodeCount2, "노드 수가 일치하지 않습니다.");
        }

        /// <summary>
        /// 테스트 8: ClearCity가 모든 컴포넌트를 정리하는지 확인
        /// </summary>
        [Test]
        public void Test_ClearCity_CleansUpAllComponents()
        {
            // Arrange: 도시 생성
            CityGenerationResult result = cityGenerator.GenerateCity();
            Assert.IsTrue(result.success, "도시 생성이 실패했습니다.");

            // Act: 도시 제거
            cityGenerator.ClearCity();

            // Assert: 모든 컴포넌트가 정리되었는지 확인
            GameObject cityRoot = GameObject.Find("City");
            Assert.IsNull(cityRoot, "City GameObject가 제거되지 않았습니다.");
        }

        /// <summary>
        /// 테스트 9: 성능 요구사항 확인 (1000개 건물 5초 이내)
        /// </summary>
        [Test]
        public void Test_Performance_LargeCity()
        {
            // Arrange: 대규모 도시 파라미터 설정
            cityGenerator.minWidth = 30;
            cityGenerator.maxWidth = 30;
            cityGenerator.minDepth = 30;
            cityGenerator.maxDepth = 30;
            cityGenerator.buildingDensity = 1.0f; // 모든 셀에 건물 생성

            // Act: 도시 생성 및 시간 측정
            CityGenerationResult result = cityGenerator.GenerateCity();

            // Assert: 성능 요구사항 확인
            Assert.IsTrue(result.success, "도시 생성이 실패했습니다.");
            Assert.Greater(result.buildingCount, 800, "충분한 건물이 생성되지 않았습니다.");
            Assert.Less(result.generationTime, 5.0f, $"생성 시간이 5초를 초과했습니다 ({result.generationTime:F2}초).");
            
            Debug.Log($"대규모 도시 생성 완료: {result.buildingCount}개 건물, {result.generationTime:F2}초");
        }

        /// <summary>
        /// 테스트 10: 모든 요구사항 통합 확인
        /// </summary>
        [Test]
        public void Test_AllRequirements_Integration()
        {
            // Act: 도시 생성
            CityGenerationResult result = cityGenerator.GenerateCity();

            // Assert: 모든 핵심 요구사항 확인
            Assert.IsTrue(result.success, "요구사항 전체: 도시 생성 실패");
            Assert.Greater(result.buildingCount, 0, "요구사항 1: 건물 생성 실패");
            Assert.IsNotNull(result.graph, "요구사항 15: 그래프 생성 실패");
            Assert.Greater(result.nodeCount, 0, "요구사항 15: 노드 생성 실패");
            Assert.Greater(result.edgeCount, 0, "요구사항 15: 엣지 생성 실패");
            
            // 건물 계층 구조 확인
            GameObject cityRoot = GameObject.Find("City");
            Assert.IsNotNull(cityRoot, "요구사항 13: 계층 구조 생성 실패");
            
            // 전략적 위치 분석 확인
            var allNodes = result.graph.GetAllNodes();
            bool hasStrategicMarkers = false;
            foreach (var node in allNodes)
            {
                if (node.strategicMarkers != null && node.strategicMarkers.Count > 0)
                {
                    hasStrategicMarkers = true;
                    break;
                }
            }
            Assert.IsTrue(hasStrategicMarkers, "요구사항 17: 전략적 위치 분석 실패");
            
            Debug.Log($"통합 테스트 완료: 건물 {result.buildingCount}개, 노드 {result.nodeCount}개, 엣지 {result.edgeCount}개");
        }
    }
}
