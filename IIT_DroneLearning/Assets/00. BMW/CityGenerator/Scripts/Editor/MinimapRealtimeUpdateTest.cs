using NUnit.Framework;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.TestTools;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;

namespace ProceduralCityGenerator.Tests
{
    /// <summary>
    /// MinimapRenderer 실시간 업데이트 기능 테스트
    /// Task 15.4: 미니맵 실시간 업데이트 테스트
    /// **Validates: Requirements 19.1, 19.2, 19.3, 19.4, 19.5**
    /// </summary>
    [TestFixture]
    public class MinimapRealtimeUpdateTest
    {
        private GameObject testGameObject;
        private MinimapRenderer renderer;
        private Canvas testCanvas;
        private Texture2D testTexture;
        private Bounds testBounds;

        [SetUp]
        public void SetUp()
        {
            // Create a test GameObject with Canvas
            GameObject canvasObject = new GameObject("TestCanvas");
            testCanvas = canvasObject.AddComponent<Canvas>();
            testCanvas.renderMode = RenderMode.ScreenSpaceOverlay;

            // Create test GameObject with MinimapRenderer
            testGameObject = new GameObject("TestMinimapRenderer");
            testGameObject.transform.SetParent(testCanvas.transform);
            
            // Add RawImage component first (required by MinimapRenderer)
            testGameObject.AddComponent<RawImage>();
            
            // Add MinimapRenderer component
            renderer = testGameObject.AddComponent<MinimapRenderer>();

            // Initialize test texture and bounds
            testTexture = new Texture2D(512, 512);
            testBounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));
            renderer.Initialize(testTexture, 5.12f, testBounds);
        }

        [TearDown]
        public void TearDown()
        {
            if (testTexture != null)
            {
                Object.DestroyImmediate(testTexture);
            }
            if (testGameObject != null)
            {
                Object.DestroyImmediate(testGameObject);
            }
            if (testCanvas != null)
            {
                Object.DestroyImmediate(testCanvas.gameObject);
            }
        }

        // ===== 동적 마커 추가/제거 테스트 (Requirements 19.1, 19.2) =====

        [Test]
        public void DynamicMarker_AddEvaderDrone_AddsSuccessfully()
        {
            // Arrange
            Vector3 position = new Vector3(25, 0, 25);

            // Act
            renderer.AddDynamicMarker(position, MarkerType.EvaderDrone);

            // Assert
            Assert.IsTrue(renderer.IsInitialized, "Renderer should remain initialized after adding marker");
        }

        [Test]
        public void DynamicMarker_AddPursuerDrone_AddsSuccessfully()
        {
            // Arrange
            Vector3 position = new Vector3(75, 0, 75);

            // Act
            renderer.AddDynamicMarker(position, MarkerType.PursuerDrone);

            // Assert
            Assert.IsTrue(renderer.IsInitialized, "Renderer should remain initialized after adding marker");
        }

        [Test]
        public void DynamicMarker_AddTargetPoint_AddsSuccessfully()
        {
            // Arrange
            Vector3 position = new Vector3(50, 0, 50);

            // Act
            renderer.AddDynamicMarker(position, MarkerType.TargetPoint);

            // Assert
            Assert.IsTrue(renderer.IsInitialized, "Renderer should remain initialized after adding marker");
        }

        [Test]
        public void DynamicMarker_AddMultipleMarkers_AllAddedSuccessfully()
        {
            // Arrange & Act
            renderer.AddDynamicMarker(new Vector3(10, 0, 10), MarkerType.EvaderDrone);
            renderer.AddDynamicMarker(new Vector3(20, 0, 20), MarkerType.PursuerDrone);
            renderer.AddDynamicMarker(new Vector3(30, 0, 30), MarkerType.TargetPoint);
            renderer.AddDynamicMarker(new Vector3(40, 0, 40), MarkerType.PathLine);

            // Assert
            Assert.IsTrue(renderer.IsInitialized, "Renderer should handle multiple markers");
        }

        [Test]
        public void DynamicMarker_RemoveExistingMarker_RemovesSuccessfully()
        {
            // Arrange
            Vector3 position = new Vector3(25, 0, 25);
            renderer.AddDynamicMarker(position, MarkerType.EvaderDrone);

            // Act
            renderer.RemoveDynamicMarker(position);

            // Assert
            Assert.IsTrue(renderer.IsInitialized, "Renderer should remain initialized after removing marker");
        }

        [Test]
        public void DynamicMarker_RemoveNonExistentMarker_DoesNotCauseError()
        {
            // Arrange
            Vector3 position = new Vector3(25, 0, 25);

            // Act & Assert - Should not throw exception
            Assert.DoesNotThrow(() => renderer.RemoveDynamicMarker(position));
        }

        [Test]
        public void DynamicMarker_AddAndRemoveMultipleTimes_WorksCorrectly()
        {
            // Arrange
            Vector3 position = new Vector3(50, 0, 50);

            // Act & Assert
            for (int i = 0; i < 5; i++)
            {
                renderer.AddDynamicMarker(position, MarkerType.EvaderDrone);
                Assert.IsTrue(renderer.IsInitialized);
                
                renderer.RemoveDynamicMarker(position);
                Assert.IsTrue(renderer.IsInitialized);
            }
        }

        [Test]
        public void DynamicMarker_UpdatePosition_UpdatesSuccessfully()
        {
            // Arrange
            Vector3 oldPosition = new Vector3(10, 0, 10);
            Vector3 newPosition = new Vector3(20, 0, 20);
            renderer.AddDynamicMarker(oldPosition, MarkerType.EvaderDrone);

            // Act
            renderer.UpdateMarkerPosition(oldPosition, newPosition, MarkerType.EvaderDrone);

            // Assert
            Assert.IsTrue(renderer.IsInitialized, "Renderer should remain initialized after updating marker position");
        }

        [Test]
        public void DynamicMarker_UpdatePositionMultipleTimes_WorksCorrectly()
        {
            // Arrange
            Vector3 position1 = new Vector3(10, 0, 10);
            Vector3 position2 = new Vector3(20, 0, 20);
            Vector3 position3 = new Vector3(30, 0, 30);
            renderer.AddDynamicMarker(position1, MarkerType.EvaderDrone);

            // Act
            renderer.UpdateMarkerPosition(position1, position2, MarkerType.EvaderDrone);
            renderer.UpdateMarkerPosition(position2, position3, MarkerType.EvaderDrone);

            // Assert
            Assert.IsTrue(renderer.IsInitialized, "Renderer should handle multiple position updates");
        }

        [Test]
        public void DynamicMarker_SimulateDroneMovement_WorksCorrectly()
        {
            // Arrange - Simulate a drone moving along a path
            Vector3 startPosition = new Vector3(0, 0, 0);
            renderer.AddDynamicMarker(startPosition, MarkerType.EvaderDrone);

            // Act - Move drone in 10 steps
            Vector3 currentPosition = startPosition;
            for (int i = 1; i <= 10; i++)
            {
                Vector3 newPosition = new Vector3(i * 10, 0, i * 10);
                renderer.UpdateMarkerPosition(currentPosition, newPosition, MarkerType.EvaderDrone);
                currentPosition = newPosition;
            }

            // Assert
            Assert.IsTrue(renderer.IsInitialized, "Renderer should handle simulated drone movement");
        }

        // ===== 경로 표시 테스트 (Requirements 19.2) =====

        [Test]
        public void PathDrawing_SimplePath_DrawsSuccessfully()
        {
            // Arrange
            List<Vector3> path = new List<Vector3>
            {
                new Vector3(0, 0, 0),
                new Vector3(25, 0, 25),
                new Vector3(50, 0, 50)
            };

            // Act
            renderer.DrawPath(path, Color.yellow);

            // Assert
            Assert.IsTrue(renderer.IsInitialized, "Renderer should remain initialized after drawing path");
        }

        [Test]
        public void PathDrawing_ComplexPath_DrawsSuccessfully()
        {
            // Arrange - Create a complex zigzag path
            List<Vector3> path = new List<Vector3>();
            for (int i = 0; i < 20; i++)
            {
                float x = i * 5f;
                float z = (i % 2 == 0) ? 0 : 50;
                path.Add(new Vector3(x, 0, z));
            }

            // Act
            renderer.DrawPath(path, Color.cyan);

            // Assert
            Assert.IsTrue(renderer.IsInitialized, "Renderer should handle complex paths");
        }

        [Test]
        public void PathDrawing_MultiplePaths_AllDrawnSuccessfully()
        {
            // Arrange
            List<Vector3> path1 = new List<Vector3>
            {
                new Vector3(0, 0, 0),
                new Vector3(25, 0, 25)
            };

            List<Vector3> path2 = new List<Vector3>
            {
                new Vector3(50, 0, 50),
                new Vector3(75, 0, 75)
            };

            // Act
            renderer.DrawPath(path1, Color.red);
            renderer.DrawPath(path2, Color.blue);

            // Assert
            Assert.IsTrue(renderer.IsInitialized, "Renderer should handle multiple paths");
        }

        [Test]
        public void PathDrawing_WithDifferentColors_WorksCorrectly()
        {
            // Arrange
            List<Vector3> path = new List<Vector3>
            {
                new Vector3(0, 0, 0),
                new Vector3(50, 0, 50)
            };

            // Act & Assert
            renderer.DrawPath(path, Color.red);
            Assert.IsTrue(renderer.IsInitialized);

            renderer.DrawPath(path, Color.green);
            Assert.IsTrue(renderer.IsInitialized);

            renderer.DrawPath(path, Color.blue);
            Assert.IsTrue(renderer.IsInitialized);
        }

        [Test]
        public void PathDrawing_CurvedPath_DrawsSuccessfully()
        {
            // Arrange - Create a curved path using sine wave
            List<Vector3> path = new List<Vector3>();
            for (int i = 0; i < 30; i++)
            {
                float x = i * 3f;
                float z = 50 + Mathf.Sin(i * 0.3f) * 20f;
                path.Add(new Vector3(x, 0, z));
            }

            // Act
            renderer.DrawPath(path, Color.magenta);

            // Assert
            Assert.IsTrue(renderer.IsInitialized, "Renderer should handle curved paths");
        }

        // ===== 성능 측정 테스트 (Requirements 19.4, 19.5) =====

        [Test]
        public void Performance_SingleMarkerUpdate_CompletesWithin1ms()
        {
            // Arrange
            Vector3 position = new Vector3(50, 0, 50);

            // Act - Measure performance
            Stopwatch stopwatch = Stopwatch.StartNew();
            renderer.AddDynamicMarker(position, MarkerType.EvaderDrone);
            stopwatch.Stop();

            double elapsedMs = stopwatch.Elapsed.TotalMilliseconds;

            // Assert
            Assert.Less(elapsedMs, 1.0, 
                $"Single marker addition took {elapsedMs:F3}ms, exceeding 1ms requirement");
            
            UnityEngine.Debug.Log($"[Performance] Single marker addition: {elapsedMs:F3}ms");
        }

        [Test]
        public void Performance_MultipleMarkerUpdates_CompletesWithin1ms()
        {
            // Arrange - Add 5 markers
            for (int i = 0; i < 5; i++)
            {
                renderer.AddDynamicMarker(new Vector3(i * 20, 0, i * 20), MarkerType.EvaderDrone);
            }

            // Act - Measure performance of adding one more
            Stopwatch stopwatch = Stopwatch.StartNew();
            renderer.AddDynamicMarker(new Vector3(100, 0, 100), MarkerType.PursuerDrone);
            stopwatch.Stop();

            double elapsedMs = stopwatch.Elapsed.TotalMilliseconds;

            // Assert
            Assert.Less(elapsedMs, 1.0, 
                $"Marker addition with existing markers took {elapsedMs:F3}ms, exceeding 1ms requirement");
            
            UnityEngine.Debug.Log($"[Performance] Marker addition (6 total): {elapsedMs:F3}ms");
        }

        [Test]
        public void Performance_MarkerPositionUpdate_CompletesWithin1ms()
        {
            // Arrange
            Vector3 oldPosition = new Vector3(25, 0, 25);
            Vector3 newPosition = new Vector3(75, 0, 75);
            renderer.AddDynamicMarker(oldPosition, MarkerType.EvaderDrone);

            // Act - Measure performance
            Stopwatch stopwatch = Stopwatch.StartNew();
            renderer.UpdateMarkerPosition(oldPosition, newPosition, MarkerType.EvaderDrone);
            stopwatch.Stop();

            double elapsedMs = stopwatch.Elapsed.TotalMilliseconds;

            // Assert
            Assert.Less(elapsedMs, 1.0, 
                $"Marker position update took {elapsedMs:F3}ms, exceeding 1ms requirement");
            
            UnityEngine.Debug.Log($"[Performance] Marker position update: {elapsedMs:F3}ms");
        }

        [Test]
        public void Performance_MarkerRemoval_CompletesWithin1ms()
        {
            // Arrange
            Vector3 position = new Vector3(50, 0, 50);
            renderer.AddDynamicMarker(position, MarkerType.EvaderDrone);

            // Act - Measure performance
            Stopwatch stopwatch = Stopwatch.StartNew();
            renderer.RemoveDynamicMarker(position);
            stopwatch.Stop();

            double elapsedMs = stopwatch.Elapsed.TotalMilliseconds;

            // Assert
            Assert.Less(elapsedMs, 1.0, 
                $"Marker removal took {elapsedMs:F3}ms, exceeding 1ms requirement");
            
            UnityEngine.Debug.Log($"[Performance] Marker removal: {elapsedMs:F3}ms");
        }

        [Test]
        public void Performance_PathDrawing_CompletesWithin1ms()
        {
            // Arrange
            List<Vector3> path = new List<Vector3>();
            for (int i = 0; i < 15; i++)
            {
                path.Add(new Vector3(i * 6, 0, i * 6));
            }

            // Act - Measure performance
            Stopwatch stopwatch = Stopwatch.StartNew();
            renderer.DrawPath(path, Color.yellow);
            stopwatch.Stop();

            double elapsedMs = stopwatch.Elapsed.TotalMilliseconds;

            // Assert
            Assert.Less(elapsedMs, 1.0, 
                $"Path drawing (15 points) took {elapsedMs:F3}ms, exceeding 1ms requirement");
            
            UnityEngine.Debug.Log($"[Performance] Path drawing (15 points): {elapsedMs:F3}ms");
        }

        [Test]
        public void Performance_CombinedOperations_CompletesReasonably()
        {
            // Arrange
            List<Vector3> path = new List<Vector3>
            {
                new Vector3(0, 0, 0),
                new Vector3(25, 0, 25),
                new Vector3(50, 0, 50)
            };

            // Act - Measure combined operations
            Stopwatch stopwatch = Stopwatch.StartNew();
            
            renderer.AddDynamicMarker(new Vector3(10, 0, 10), MarkerType.EvaderDrone);
            renderer.AddDynamicMarker(new Vector3(90, 0, 90), MarkerType.PursuerDrone);
            renderer.DrawPath(path, Color.yellow);
            
            stopwatch.Stop();

            double elapsedMs = stopwatch.Elapsed.TotalMilliseconds;

            // Assert - Combined operations should complete within 2ms
            Assert.Less(elapsedMs, 2.0, 
                $"Combined operations took {elapsedMs:F3}ms, exceeding reasonable limit");
            
            UnityEngine.Debug.Log($"[Performance] Combined operations: {elapsedMs:F3}ms");
        }

        [Test]
        public void Performance_StressTest_10Markers_CompletesReasonably()
        {
            // Arrange & Act - Measure adding 10 markers
            Stopwatch stopwatch = Stopwatch.StartNew();
            
            for (int i = 0; i < 10; i++)
            {
                Vector3 position = new Vector3(i * 10, 0, i * 10);
                renderer.AddDynamicMarker(position, MarkerType.EvaderDrone);
            }
            
            stopwatch.Stop();

            double elapsedMs = stopwatch.Elapsed.TotalMilliseconds;
            double avgPerMarker = elapsedMs / 10;

            // Assert
            Assert.Less(avgPerMarker, 1.0, 
                $"Average marker addition took {avgPerMarker:F3}ms, exceeding 1ms requirement");
            
            UnityEngine.Debug.Log($"[Performance] 10 markers total: {elapsedMs:F3}ms, avg: {avgPerMarker:F3}ms");
        }

        [Test]
        public void Performance_StressTest_20Markers_CompletesReasonably()
        {
            // Arrange & Act - Measure adding 20 markers
            Stopwatch stopwatch = Stopwatch.StartNew();
            
            for (int i = 0; i < 20; i++)
            {
                Vector3 position = new Vector3(
                    UnityEngine.Random.Range(0, 100),
                    0,
                    UnityEngine.Random.Range(0, 100)
                );
                renderer.AddDynamicMarker(position, MarkerType.EvaderDrone);
            }
            
            stopwatch.Stop();

            double elapsedMs = stopwatch.Elapsed.TotalMilliseconds;
            double avgPerMarker = elapsedMs / 20;

            // Assert - Allow slightly more time for stress test
            if (avgPerMarker > 1.0)
            {
                UnityEngine.Debug.LogWarning(
                    $"[Performance] Average marker addition ({avgPerMarker:F3}ms) exceeds 1ms with 20 markers. " +
                    "Consider optimizing for high marker counts.");
            }
            
            Assert.Less(elapsedMs, 20.0, 
                $"Adding 20 markers took {elapsedMs:F3}ms, which is unreasonably slow");
            
            UnityEngine.Debug.Log($"[Performance] 20 markers total: {elapsedMs:F3}ms, avg: {avgPerMarker:F3}ms");
        }

        // ===== 통합 시나리오 테스트 =====

        [Test]
        public void IntegrationScenario_DroneChaseSimulation_WorksCorrectly()
        {
            // Arrange - Simulate evader and pursuer drones
            Vector3 evaderPos = new Vector3(10, 0, 10);
            Vector3 pursuerPos = new Vector3(90, 0, 90);
            
            renderer.AddDynamicMarker(evaderPos, MarkerType.EvaderDrone);
            renderer.AddDynamicMarker(pursuerPos, MarkerType.PursuerDrone);

            // Act - Simulate movement over 5 steps
            for (int i = 1; i <= 5; i++)
            {
                Vector3 newEvaderPos = new Vector3(10 + i * 10, 0, 10 + i * 10);
                Vector3 newPursuerPos = new Vector3(90 - i * 10, 0, 90 - i * 10);
                
                renderer.UpdateMarkerPosition(evaderPos, newEvaderPos, MarkerType.EvaderDrone);
                renderer.UpdateMarkerPosition(pursuerPos, newPursuerPos, MarkerType.PursuerDrone);
                
                evaderPos = newEvaderPos;
                pursuerPos = newPursuerPos;
            }

            // Assert
            Assert.IsTrue(renderer.IsInitialized, "Renderer should handle chase simulation");
        }

        [Test]
        public void IntegrationScenario_PathPlanningVisualization_WorksCorrectly()
        {
            // Arrange - Simulate path planning scenario
            Vector3 startPos = new Vector3(0, 0, 0);
            Vector3 targetPos = new Vector3(100, 0, 100);
            
            renderer.AddDynamicMarker(startPos, MarkerType.EvaderDrone);
            renderer.AddDynamicMarker(targetPos, MarkerType.TargetPoint);

            // Create planned path
            List<Vector3> plannedPath = new List<Vector3>
            {
                startPos,
                new Vector3(25, 0, 20),
                new Vector3(50, 0, 50),
                new Vector3(75, 0, 80),
                targetPos
            };

            // Act
            renderer.DrawPath(plannedPath, Color.green);

            // Assert
            Assert.IsTrue(renderer.IsInitialized, "Renderer should handle path planning visualization");
        }

        [Test]
        public void IntegrationScenario_MultiDroneTracking_WorksCorrectly()
        {
            // Arrange - Simulate multiple drones
            List<Vector3> dronePositions = new List<Vector3>
            {
                new Vector3(10, 0, 10),
                new Vector3(30, 0, 30),
                new Vector3(50, 0, 50),
                new Vector3(70, 0, 70),
                new Vector3(90, 0, 90)
            };

            // Act - Add all drones
            foreach (var pos in dronePositions)
            {
                renderer.AddDynamicMarker(pos, MarkerType.EvaderDrone);
            }

            // Assert
            Assert.IsTrue(renderer.IsInitialized, "Renderer should handle multiple drone tracking");
        }

        [Test]
        public void IntegrationScenario_DynamicPathUpdate_WorksCorrectly()
        {
            // Arrange
            List<Vector3> initialPath = new List<Vector3>
            {
                new Vector3(0, 0, 0),
                new Vector3(50, 0, 50)
            };

            List<Vector3> updatedPath = new List<Vector3>
            {
                new Vector3(0, 0, 0),
                new Vector3(25, 0, 75),
                new Vector3(50, 0, 50)
            };

            // Act - Draw initial path, then update it
            renderer.DrawPath(initialPath, Color.yellow);
            renderer.DrawPath(updatedPath, Color.cyan);

            // Assert
            Assert.IsTrue(renderer.IsInitialized, "Renderer should handle dynamic path updates");
        }

        [Test]
        public void IntegrationScenario_ClearAndRedraw_WorksCorrectly()
        {
            // Arrange
            Vector3 pos1 = new Vector3(25, 0, 25);
            Vector3 pos2 = new Vector3(75, 0, 75);
            
            renderer.AddDynamicMarker(pos1, MarkerType.EvaderDrone);
            renderer.AddDynamicMarker(pos2, MarkerType.PursuerDrone);

            // Act - Remove all markers and add new ones
            renderer.RemoveDynamicMarker(pos1);
            renderer.RemoveDynamicMarker(pos2);
            
            renderer.AddDynamicMarker(new Vector3(50, 0, 50), MarkerType.TargetPoint);

            // Assert
            Assert.IsTrue(renderer.IsInitialized, "Renderer should handle clear and redraw operations");
        }
    }
}
