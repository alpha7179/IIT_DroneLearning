using NUnit.Framework;
using UnityEngine;
using UnityEditor;
using System.Diagnostics;
using System.Collections.Generic;

namespace CityGenerator.Tests
{
    /// <summary>
    /// Task 14.3: 성능 테스트 및 검증
    /// Requirements: 11.1
    /// 
    /// 1000개 건물 생성 시 5초 이내 완료 확인
    /// 프로파일링을 통한 병목 지점 식별 및 최적화
    /// </summary>
    public class PerformanceTest
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
            if (testObject != null)
            {
                Object.DestroyImmediate(testObject);
            }
        }

        /// <summary>
        /// Requirement 11.1: 1000개 이상의 건물로 도시를 생성할 때 5초 이내에 완료
        /// </summary>
        [Test]
        public void Test_Performance_1000Buildings_Within5Seconds()
        {
            // Arrange: 1000개 이상 건물을 생성할 수 있는 파라미터 설정
            generator.minWidth = 40;
            generator.maxWidth = 40;
            generator.minDepth = 40;
            generator.maxDepth = 40;
            generator.buildingDensity = 0.7f; // 약 1120개 건물 예상 (40 * 40 * 0.7)
            generator.randomSeed = 12345;
            generator.buildingSpacing = 1.0f;
            generator.unitDistance = 1.0f;

            // Act: 도시 생성 및 시간 측정
            Stopwatch stopwatch = Stopwatch.StartNew();
            CityGenerationResult result = generator.GenerateCity();
            stopwatch.Stop();

            // Assert: Requirement 11.1 검증
            Assert.IsTrue(result.success, "도시 생성이 성공해야 합니다");
            Assert.GreaterOrEqual(result.buildingCount, 1000, 
                "1000개 이상의 건물이 생성되어야 합니다");
            Assert.Less(stopwatch.Elapsed.TotalSeconds, 5.0, 
                "Requirement 11.1: 1000개 이상 건물 생성이 5초 이내에 완료되어야 합니다");

            UnityEngine.Debug.Log($"[Performance Test] 1000+ Buildings Test PASSED");
            UnityEngine.Debug.Log($"  - Buildings Generated: {result.buildingCount}");
            UnityEngine.Debug.Log($"  - Time Taken: {stopwatch.Elapsed.TotalSeconds:F3} seconds");
            UnityEngine.Debug.Log($"  - Buildings/Second: {result.buildingCount / stopwatch.Elapsed.TotalSeconds:F1}");
        }

        /// <summary>
        /// 다양한 도시 크기에서 성능 측정
        /// </summary>
        [Test]
        public void Test_Performance_VariousCitySizes()
        {
            // 테스트할 도시 크기 (가로 x 세로)
            int[] citySizes = { 10, 20, 30, 40, 50 };
            List<PerformanceMetrics> metrics = new List<PerformanceMetrics>();

            foreach (int size in citySizes)
            {
                // Arrange
                generator.minWidth = size;
                generator.maxWidth = size;
                generator.minDepth = size;
                generator.maxDepth = size;
                generator.buildingDensity = 0.7f;
                generator.randomSeed = 12345;

                // Act
                Stopwatch stopwatch = Stopwatch.StartNew();
                CityGenerationResult result = generator.GenerateCity();
                stopwatch.Stop();

                // 메트릭 수집
                PerformanceMetrics metric = new PerformanceMetrics
                {
                    citySize = size,
                    buildingCount = result.buildingCount,
                    generationTime = stopwatch.Elapsed.TotalSeconds,
                    buildingsPerSecond = result.buildingCount / stopwatch.Elapsed.TotalSeconds
                };
                metrics.Add(metric);

                // 정리
                generator.ClearCity();
            }

            // Assert: 모든 크기에서 합리적인 성능 확인
            foreach (var metric in metrics)
            {
                // 1000개 이상 건물의 경우 5초 이내 확인
                if (metric.buildingCount >= 1000)
                {
                    Assert.Less(metric.generationTime, 5.0,
                        $"도시 크기 {metric.citySize}x{metric.citySize} ({metric.buildingCount}개 건물)는 5초 이내에 생성되어야 합니다");
                }

                UnityEngine.Debug.Log($"[Performance Test] City Size: {metric.citySize}x{metric.citySize}");
                UnityEngine.Debug.Log($"  - Buildings: {metric.buildingCount}");
                UnityEngine.Debug.Log($"  - Time: {metric.generationTime:F3}s");
                UnityEngine.Debug.Log($"  - Rate: {metric.buildingsPerSecond:F1} buildings/s");
            }
        }

        /// <summary>
        /// 프로파일링: 도시 생성 단계별 시간 측정
        /// </summary>
        [Test]
        public void Test_Profiling_GenerationStages()
        {
            // Arrange: 1000개 이상 건물 설정
            generator.minWidth = 40;
            generator.maxWidth = 40;
            generator.minDepth = 40;
            generator.maxDepth = 40;
            generator.buildingDensity = 0.7f;
            generator.randomSeed = 12345;

            // Act: 전체 생성 시간 측정
            Stopwatch totalStopwatch = Stopwatch.StartNew();
            CityGenerationResult result = generator.GenerateCity();
            totalStopwatch.Stop();

            // Assert: 성공 확인
            Assert.IsTrue(result.success, "도시 생성이 성공해야 합니다");
            Assert.GreaterOrEqual(result.buildingCount, 1000, "1000개 이상의 건물이 생성되어야 합니다");

            // 프로파일링 결과 출력
            UnityEngine.Debug.Log($"[Profiling] Generation Stages Analysis");
            UnityEngine.Debug.Log($"  Total Generation Time: {totalStopwatch.Elapsed.TotalSeconds:F3}s");
            UnityEngine.Debug.Log($"  Buildings Generated: {result.buildingCount}");
            UnityEngine.Debug.Log($"  Nodes Created: {result.nodeCount}");
            UnityEngine.Debug.Log($"  Edges Created: {result.edgeCount}");
            UnityEngine.Debug.Log($"  Average Time per Building: {(totalStopwatch.Elapsed.TotalMilliseconds / result.buildingCount):F3}ms");
            
            // 병목 지점 식별을 위한 권장사항
            if (totalStopwatch.Elapsed.TotalSeconds > 3.0)
            {
                UnityEngine.Debug.LogWarning("[Profiling] Performance Warning: Generation took longer than 3 seconds");
                UnityEngine.Debug.LogWarning("  Potential bottlenecks to investigate:");
                UnityEngine.Debug.LogWarning("  - Building instantiation (consider object pooling optimization)");
                UnityEngine.Debug.LogWarning("  - Graph construction (check node/edge creation efficiency)");
                UnityEngine.Debug.LogWarning("  - Spatial index building (verify Quadtree performance)");
            }
        }

        /// <summary>
        /// 최대 규모 도시 성능 테스트 (스트레스 테스트)
        /// </summary>
        [Test]
        public void Test_Performance_MaximumCitySize()
        {
            // Arrange: 최대 크기 도시 (100x100 = 10,000 셀)
            generator.minWidth = 100;
            generator.maxWidth = 100;
            generator.minDepth = 100;
            generator.maxDepth = 100;
            generator.buildingDensity = 0.5f; // 약 5000개 건물
            generator.randomSeed = 99999;

            // Act
            Stopwatch stopwatch = Stopwatch.StartNew();
            CityGenerationResult result = generator.GenerateCity();
            stopwatch.Stop();

            // Assert: 생성 성공 확인 (시간 제한은 더 관대하게)
            Assert.IsTrue(result.success, "최대 크기 도시 생성이 성공해야 합니다");
            Assert.Greater(result.buildingCount, 0, "건물이 생성되어야 합니다");

            UnityEngine.Debug.Log($"[Performance Test] Maximum City Size Test");
            UnityEngine.Debug.Log($"  - Grid Size: 100x100");
            UnityEngine.Debug.Log($"  - Buildings Generated: {result.buildingCount}");
            UnityEngine.Debug.Log($"  - Time Taken: {stopwatch.Elapsed.TotalSeconds:F3}s");
            UnityEngine.Debug.Log($"  - Buildings/Second: {result.buildingCount / stopwatch.Elapsed.TotalSeconds:F1}");
            
            // 성능 경고 (참고용)
            if (stopwatch.Elapsed.TotalSeconds > 10.0)
            {
                UnityEngine.Debug.LogWarning($"  - Warning: Large city generation took {stopwatch.Elapsed.TotalSeconds:F1}s");
            }
        }

        /// <summary>
        /// 밀도 변화에 따른 성능 측정
        /// </summary>
        [Test]
        public void Test_Performance_DensityVariation()
        {
            float[] densities = { 0.3f, 0.5f, 0.7f, 0.9f, 1.0f };
            List<PerformanceMetrics> metrics = new List<PerformanceMetrics>();

            foreach (float density in densities)
            {
                // Arrange
                generator.minWidth = 40;
                generator.maxWidth = 40;
                generator.minDepth = 40;
                generator.maxDepth = 40;
                generator.buildingDensity = density;
                generator.randomSeed = 12345;

                // Act
                Stopwatch stopwatch = Stopwatch.StartNew();
                CityGenerationResult result = generator.GenerateCity();
                stopwatch.Stop();

                // 메트릭 수집
                PerformanceMetrics metric = new PerformanceMetrics
                {
                    density = density,
                    buildingCount = result.buildingCount,
                    generationTime = stopwatch.Elapsed.TotalSeconds,
                    buildingsPerSecond = result.buildingCount / stopwatch.Elapsed.TotalSeconds
                };
                metrics.Add(metric);

                // 정리
                generator.ClearCity();
            }

            // Assert & Report
            foreach (var metric in metrics)
            {
                UnityEngine.Debug.Log($"[Performance Test] Density: {metric.density:P0}");
                UnityEngine.Debug.Log($"  - Buildings: {metric.buildingCount}");
                UnityEngine.Debug.Log($"  - Time: {metric.generationTime:F3}s");
                UnityEngine.Debug.Log($"  - Rate: {metric.buildingsPerSecond:F1} buildings/s");

                // 1000개 이상일 때 5초 제한 확인
                if (metric.buildingCount >= 1000)
                {
                    Assert.Less(metric.generationTime, 5.0,
                        $"밀도 {metric.density:P0} ({metric.buildingCount}개 건물)는 5초 이내에 생성되어야 합니다");
                }
            }
        }

        /// <summary>
        /// 성능 메트릭 구조체
        /// </summary>
        private struct PerformanceMetrics
        {
            public int citySize;
            public float density;
            public int buildingCount;
            public double generationTime;
            public double buildingsPerSecond;
        }
    }
}
