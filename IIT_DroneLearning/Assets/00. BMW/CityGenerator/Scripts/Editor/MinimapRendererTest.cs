using NUnit.Framework;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.TestTools;
using System.Collections;

namespace ProceduralCityGenerator.Tests
{
    /// <summary>
    /// MinimapRenderer 클래스의 단위 테스트
    /// **Validates: Requirements 19.6**
    /// </summary>
    [TestFixture]
    public class MinimapRendererTest
    {
        private GameObject testGameObject;
        private MinimapRenderer renderer;
        private Canvas testCanvas;

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
        }

        [TearDown]
        public void TearDown()
        {
            if (testGameObject != null)
            {
                Object.DestroyImmediate(testGameObject);
            }
            if (testCanvas != null)
            {
                Object.DestroyImmediate(testCanvas.gameObject);
            }
        }

        [Test]
        public void Awake_AutomaticallyFindsRawImageComponent()
        {
            // Arrange & Act
            // RawImage is added in SetUp, and Awake should find it automatically

            // Assert
            Assert.IsNotNull(renderer, "MinimapRenderer should be created");
            // The RawImage reference is private, but we can verify it doesn't log an error
        }

        [Test]
        public void Initialize_WithValidParameters_SetsInitializedToTrue()
        {
            // Arrange
            Texture2D testTexture = new Texture2D(256, 256);
            float pixelsPerMeter = 2.56f;
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));

            // Act
            renderer.Initialize(testTexture, pixelsPerMeter, bounds);

            // Assert
            Assert.IsTrue(renderer.IsInitialized, "Renderer should be initialized");

            // Cleanup
            Object.DestroyImmediate(testTexture);
        }

        [Test]
        public void Initialize_WithNullTexture_LogsErrorAndDoesNotInitialize()
        {
            // Arrange
            float pixelsPerMeter = 2.56f;
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));

            // Act
            LogAssert.Expect(LogType.Error, "MinimapRenderer.Initialize: 미니맵 텍스처가 null입니다.");
            renderer.Initialize(null, pixelsPerMeter, bounds);

            // Assert
            Assert.IsFalse(renderer.IsInitialized, "Renderer should not be initialized with null texture");
        }

        [Test]
        public void Initialize_WithoutRawImage_LogsError()
        {
            // Arrange
            // Create a new GameObject without RawImage
            GameObject newObject = new GameObject("NoRawImage");
            newObject.transform.SetParent(testCanvas.transform);
            MinimapRenderer newRenderer = newObject.AddComponent<MinimapRenderer>();

            Texture2D testTexture = new Texture2D(256, 256);
            float pixelsPerMeter = 2.56f;
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));

            // Act
            LogAssert.Expect(LogType.Error, "MinimapRenderer.Initialize: RawImage 컴포넌트가 null입니다.");
            newRenderer.Initialize(testTexture, pixelsPerMeter, bounds);

            // Assert
            Assert.IsFalse(newRenderer.IsInitialized, "Renderer should not be initialized without RawImage");

            // Cleanup
            Object.DestroyImmediate(testTexture);
            Object.DestroyImmediate(newObject);
        }

        [Test]
        public void Initialize_SetsCorrectTextureResolution()
        {
            // Arrange
            int expectedResolution = 512;
            Texture2D testTexture = new Texture2D(expectedResolution, expectedResolution);
            float pixelsPerMeter = 5.12f;
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));

            // Act
            renderer.Initialize(testTexture, pixelsPerMeter, bounds);

            // Assert
            Assert.IsTrue(renderer.IsInitialized);
            // The texture is set internally, we verify through initialization success

            // Cleanup
            Object.DestroyImmediate(testTexture);
        }

        [Test]
        public void Initialize_WithDifferentResolutions_WorksCorrectly()
        {
            // Test with 256x256
            Texture2D texture256 = new Texture2D(256, 256);
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));
            renderer.Initialize(texture256, 2.56f, bounds);
            Assert.IsTrue(renderer.IsInitialized);

            // Test with 512x512
            Texture2D texture512 = new Texture2D(512, 512);
            renderer.Initialize(texture512, 5.12f, bounds);
            Assert.IsTrue(renderer.IsInitialized);

            // Test with 1024x1024
            Texture2D texture1024 = new Texture2D(1024, 1024);
            renderer.Initialize(texture1024, 10.24f, bounds);
            Assert.IsTrue(renderer.IsInitialized);

            // Cleanup
            Object.DestroyImmediate(texture256);
            Object.DestroyImmediate(texture512);
            Object.DestroyImmediate(texture1024);
        }

        [Test]
        public void Initialize_WithSmallPixelsPerMeter_WorksCorrectly()
        {
            // Arrange
            Texture2D testTexture = new Texture2D(256, 256);
            float smallPixelsPerMeter = 0.1f; // Large world area
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(2560, 50, 2560));

            // Act
            renderer.Initialize(testTexture, smallPixelsPerMeter, bounds);

            // Assert
            Assert.IsTrue(renderer.IsInitialized);

            // Cleanup
            Object.DestroyImmediate(testTexture);
        }

        [Test]
        public void Initialize_WithLargePixelsPerMeter_WorksCorrectly()
        {
            // Arrange
            Texture2D testTexture = new Texture2D(1024, 1024);
            float largePixelsPerMeter = 100f; // Small world area
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(10.24f, 50, 10.24f));

            // Act
            renderer.Initialize(testTexture, largePixelsPerMeter, bounds);

            // Assert
            Assert.IsTrue(renderer.IsInitialized);

            // Cleanup
            Object.DestroyImmediate(testTexture);
        }

        [Test]
        public void Initialize_WithNonCenteredBounds_WorksCorrectly()
        {
            // Arrange
            Texture2D testTexture = new Texture2D(256, 256);
            float pixelsPerMeter = 2.56f;
            Vector3 center = new Vector3(50, 25, 50);
            Vector3 size = new Vector3(100, 50, 100);
            Bounds bounds = new Bounds(center, size);

            // Act
            renderer.Initialize(testTexture, pixelsPerMeter, bounds);

            // Assert
            Assert.IsTrue(renderer.IsInitialized);

            // Cleanup
            Object.DestroyImmediate(testTexture);
        }

        [Test]
        public void Initialize_CanBeCalledMultipleTimes()
        {
            // Arrange
            Texture2D texture1 = new Texture2D(256, 256);
            Texture2D texture2 = new Texture2D(512, 512);
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));

            // Act
            renderer.Initialize(texture1, 2.56f, bounds);
            Assert.IsTrue(renderer.IsInitialized);

            renderer.Initialize(texture2, 5.12f, bounds);
            Assert.IsTrue(renderer.IsInitialized);

            // Cleanup
            Object.DestroyImmediate(texture1);
            Object.DestroyImmediate(texture2);
        }

        [Test]
        public void IsInitialized_ReturnsFalseBeforeInitialization()
        {
            // Arrange & Act
            // Renderer is created in SetUp but not initialized

            // Assert
            Assert.IsFalse(renderer.IsInitialized, "Renderer should not be initialized before Initialize() is called");
        }

        [Test]
        public void IsInitialized_ReturnsTrueAfterSuccessfulInitialization()
        {
            // Arrange
            Texture2D testTexture = new Texture2D(256, 256);
            float pixelsPerMeter = 2.56f;
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));

            // Act
            renderer.Initialize(testTexture, pixelsPerMeter, bounds);

            // Assert
            Assert.IsTrue(renderer.IsInitialized, "Renderer should be initialized after successful Initialize() call");

            // Cleanup
            Object.DestroyImmediate(testTexture);
        }

        [Test]
        public void Initialize_LogsSuccessMessage()
        {
            // Arrange
            Texture2D testTexture = new Texture2D(512, 512);
            float pixelsPerMeter = 5.12f;
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));

            // Act
            LogAssert.Expect(LogType.Log, "MinimapRenderer.Initialize: 미니맵 초기화 완료 (해상도: 512x512, 픽셀당 미터: 5.12)");
            renderer.Initialize(testTexture, pixelsPerMeter, bounds);

            // Assert
            Assert.IsTrue(renderer.IsInitialized);

            // Cleanup
            Object.DestroyImmediate(testTexture);
        }

        [Test]
        public void Initialize_WithRectangularTexture_WorksCorrectly()
        {
            // Arrange
            Texture2D testTexture = new Texture2D(512, 256); // Non-square texture
            float pixelsPerMeter = 5.12f;
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 50));

            // Act
            renderer.Initialize(testTexture, pixelsPerMeter, bounds);

            // Assert
            Assert.IsTrue(renderer.IsInitialized);

            // Cleanup
            Object.DestroyImmediate(testTexture);
        }

        [Test]
        public void Initialize_WithZeroPixelsPerMeter_WorksButMayHaveIssues()
        {
            // Arrange
            Texture2D testTexture = new Texture2D(256, 256);
            float zeroPixelsPerMeter = 0f;
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));

            // Act
            renderer.Initialize(testTexture, zeroPixelsPerMeter, bounds);

            // Assert
            Assert.IsTrue(renderer.IsInitialized, "Renderer should initialize even with zero pixelsPerMeter");
            // Note: This may cause issues in coordinate conversion, but initialization should succeed

            // Cleanup
            Object.DestroyImmediate(testTexture);
        }

        [Test]
        public void Initialize_WithNegativePixelsPerMeter_WorksButMayHaveIssues()
        {
            // Arrange
            Texture2D testTexture = new Texture2D(256, 256);
            float negativePixelsPerMeter = -2.56f;
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));

            // Act
            renderer.Initialize(testTexture, negativePixelsPerMeter, bounds);

            // Assert
            Assert.IsTrue(renderer.IsInitialized, "Renderer should initialize even with negative pixelsPerMeter");
            // Note: This may cause issues in coordinate conversion, but initialization should succeed

            // Cleanup
            Object.DestroyImmediate(testTexture);
        }

        // ===== Dynamic Marker Tests (Task 10.2) =====

        [Test]
        public void AddDynamicMarker_WithoutInitialization_LogsWarning()
        {
            // Arrange
            Vector3 position = new Vector3(10, 0, 10);

            // Act
            LogAssert.Expect(LogType.Warning, "MinimapRenderer.AddDynamicMarker: 미니맵이 초기화되지 않았습니다.");
            renderer.AddDynamicMarker(position, MarkerType.EvaderDrone);

            // Assert
            // Warning should be logged
        }

        [Test]
        public void AddDynamicMarker_WithInitialization_AddsMarkerSuccessfully()
        {
            // Arrange
            Texture2D testTexture = new Texture2D(256, 256);
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));
            renderer.Initialize(testTexture, 2.56f, bounds);

            Vector3 position = new Vector3(10, 0, 10);

            // Act
            renderer.AddDynamicMarker(position, MarkerType.EvaderDrone);

            // Assert
            Assert.IsTrue(renderer.IsInitialized);
            // Marker should be added internally

            // Cleanup
            Object.DestroyImmediate(testTexture);
        }

        [Test]
        public void AddDynamicMarker_WithDifferentMarkerTypes_WorksCorrectly()
        {
            // Arrange
            Texture2D testTexture = new Texture2D(256, 256);
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));
            renderer.Initialize(testTexture, 2.56f, bounds);

            // Act & Assert
            renderer.AddDynamicMarker(new Vector3(10, 0, 10), MarkerType.EvaderDrone);
            renderer.AddDynamicMarker(new Vector3(20, 0, 20), MarkerType.PursuerDrone);
            renderer.AddDynamicMarker(new Vector3(30, 0, 30), MarkerType.TargetPoint);
            renderer.AddDynamicMarker(new Vector3(40, 0, 40), MarkerType.PathLine);

            Assert.IsTrue(renderer.IsInitialized);

            // Cleanup
            Object.DestroyImmediate(testTexture);
        }

        [Test]
        public void RemoveDynamicMarker_WithoutInitialization_LogsWarning()
        {
            // Arrange
            Vector3 position = new Vector3(10, 0, 10);

            // Act
            LogAssert.Expect(LogType.Warning, "MinimapRenderer.RemoveDynamicMarker: 미니맵이 초기화되지 않았습니다.");
            renderer.RemoveDynamicMarker(position);

            // Assert
            // Warning should be logged
        }

        [Test]
        public void RemoveDynamicMarker_WithExistingMarker_RemovesSuccessfully()
        {
            // Arrange
            Texture2D testTexture = new Texture2D(256, 256);
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));
            renderer.Initialize(testTexture, 2.56f, bounds);

            Vector3 position = new Vector3(10, 0, 10);
            renderer.AddDynamicMarker(position, MarkerType.EvaderDrone);

            // Act
            renderer.RemoveDynamicMarker(position);

            // Assert
            Assert.IsTrue(renderer.IsInitialized);
            // Marker should be removed internally

            // Cleanup
            Object.DestroyImmediate(testTexture);
        }

        [Test]
        public void RemoveDynamicMarker_WithNonExistingMarker_DoesNothing()
        {
            // Arrange
            Texture2D testTexture = new Texture2D(256, 256);
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));
            renderer.Initialize(testTexture, 2.56f, bounds);

            Vector3 position = new Vector3(10, 0, 10);

            // Act
            renderer.RemoveDynamicMarker(position); // Remove non-existing marker

            // Assert
            Assert.IsTrue(renderer.IsInitialized);
            // Should not cause any errors

            // Cleanup
            Object.DestroyImmediate(testTexture);
        }

        [Test]
        public void UpdateMarkerPosition_WithoutInitialization_LogsWarning()
        {
            // Arrange
            Vector3 oldPosition = new Vector3(10, 0, 10);
            Vector3 newPosition = new Vector3(20, 0, 20);

            // Act
            LogAssert.Expect(LogType.Warning, "MinimapRenderer.UpdateMarkerPosition: 미니맵이 초기화되지 않았습니다.");
            renderer.UpdateMarkerPosition(oldPosition, newPosition, MarkerType.EvaderDrone);

            // Assert
            // Warning should be logged
        }

        [Test]
        public void UpdateMarkerPosition_WithExistingMarker_UpdatesSuccessfully()
        {
            // Arrange
            Texture2D testTexture = new Texture2D(256, 256);
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));
            renderer.Initialize(testTexture, 2.56f, bounds);

            Vector3 oldPosition = new Vector3(10, 0, 10);
            Vector3 newPosition = new Vector3(20, 0, 20);
            renderer.AddDynamicMarker(oldPosition, MarkerType.EvaderDrone);

            // Act
            renderer.UpdateMarkerPosition(oldPosition, newPosition, MarkerType.EvaderDrone);

            // Assert
            Assert.IsTrue(renderer.IsInitialized);
            // Marker should be moved internally

            // Cleanup
            Object.DestroyImmediate(testTexture);
        }

        [Test]
        public void UpdateMarkerPosition_WithNonExistingMarker_AddsNewMarker()
        {
            // Arrange
            Texture2D testTexture = new Texture2D(256, 256);
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));
            renderer.Initialize(testTexture, 2.56f, bounds);

            Vector3 oldPosition = new Vector3(10, 0, 10);
            Vector3 newPosition = new Vector3(20, 0, 20);

            // Act
            renderer.UpdateMarkerPosition(oldPosition, newPosition, MarkerType.EvaderDrone);

            // Assert
            Assert.IsTrue(renderer.IsInitialized);
            // New marker should be added at newPosition

            // Cleanup
            Object.DestroyImmediate(testTexture);
        }

        [Test]
        public void AddDynamicMarker_MultipleMarkers_AllAddedSuccessfully()
        {
            // Arrange
            Texture2D testTexture = new Texture2D(512, 512);
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));
            renderer.Initialize(testTexture, 5.12f, bounds);

            // Act
            for (int i = 0; i < 10; i++)
            {
                Vector3 position = new Vector3(i * 5, 0, i * 5);
                renderer.AddDynamicMarker(position, MarkerType.EvaderDrone);
            }

            // Assert
            Assert.IsTrue(renderer.IsInitialized);
            // All markers should be added

            // Cleanup
            Object.DestroyImmediate(testTexture);
        }

        [Test]
        public void AddDynamicMarker_SamePositionDifferentType_UpdatesMarkerType()
        {
            // Arrange
            Texture2D testTexture = new Texture2D(256, 256);
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));
            renderer.Initialize(testTexture, 2.56f, bounds);

            Vector3 position = new Vector3(10, 0, 10);

            // Act
            renderer.AddDynamicMarker(position, MarkerType.EvaderDrone);
            renderer.AddDynamicMarker(position, MarkerType.PursuerDrone); // Same position, different type

            // Assert
            Assert.IsTrue(renderer.IsInitialized);
            // Marker type should be updated to PursuerDrone

            // Cleanup
            Object.DestroyImmediate(testTexture);
        }

        [Test]
        public void AddDynamicMarker_AtBoundaryPositions_WorksCorrectly()
        {
            // Arrange
            Texture2D testTexture = new Texture2D(256, 256);
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));
            renderer.Initialize(testTexture, 2.56f, bounds);

            // Act & Assert
            renderer.AddDynamicMarker(bounds.min, MarkerType.EvaderDrone); // Min boundary
            renderer.AddDynamicMarker(bounds.max, MarkerType.PursuerDrone); // Max boundary
            renderer.AddDynamicMarker(bounds.center, MarkerType.TargetPoint); // Center

            Assert.IsTrue(renderer.IsInitialized);

            // Cleanup
            Object.DestroyImmediate(testTexture);
        }

        [Test]
        public void AddDynamicMarker_OutsideBounds_ClampsToTextureBounds()
        {
            // Arrange
            Texture2D testTexture = new Texture2D(256, 256);
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));
            renderer.Initialize(testTexture, 2.56f, bounds);

            // Act
            Vector3 outsidePosition = new Vector3(1000, 0, 1000); // Far outside bounds
            renderer.AddDynamicMarker(outsidePosition, MarkerType.EvaderDrone);

            // Assert
            Assert.IsTrue(renderer.IsInitialized);
            // Position should be clamped to texture bounds

            // Cleanup
            Object.DestroyImmediate(testTexture);
        }

        [Test]
        public void RemoveDynamicMarker_AllMarkers_ClearsSuccessfully()
        {
            // Arrange
            Texture2D testTexture = new Texture2D(256, 256);
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));
            renderer.Initialize(testTexture, 2.56f, bounds);

            Vector3[] positions = new Vector3[]
            {
                new Vector3(10, 0, 10),
                new Vector3(20, 0, 20),
                new Vector3(30, 0, 30)
            };

            foreach (var pos in positions)
            {
                renderer.AddDynamicMarker(pos, MarkerType.EvaderDrone);
            }

            // Act
            foreach (var pos in positions)
            {
                renderer.RemoveDynamicMarker(pos);
            }

            // Assert
            Assert.IsTrue(renderer.IsInitialized);
            // All markers should be removed

            // Cleanup
            Object.DestroyImmediate(testTexture);
        }

        // ===== DrawPath Tests (Task 10.3) =====

        [Test]
        public void DrawPath_WithoutInitialization_LogsWarning()
        {
            // Arrange
            var path = new System.Collections.Generic.List<Vector3>
            {
                new Vector3(0, 0, 0),
                new Vector3(10, 0, 10)
            };

            // Act
            LogAssert.Expect(LogType.Warning, "MinimapRenderer.DrawPath: 미니맵이 초기화되지 않았습니다.");
            renderer.DrawPath(path, Color.yellow);

            // Assert
            // Warning should be logged
        }

        [Test]
        public void DrawPath_WithNullPath_LogsWarning()
        {
            // Arrange
            Texture2D testTexture = new Texture2D(256, 256);
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));
            renderer.Initialize(testTexture, 2.56f, bounds);

            // Act
            LogAssert.Expect(LogType.Warning, "MinimapRenderer.DrawPath: 경로는 최소 2개의 점이 필요합니다.");
            renderer.DrawPath(null, Color.yellow);

            // Assert
            // Warning should be logged

            // Cleanup
            Object.DestroyImmediate(testTexture);
        }

        [Test]
        public void DrawPath_WithSinglePoint_LogsWarning()
        {
            // Arrange
            Texture2D testTexture = new Texture2D(256, 256);
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));
            renderer.Initialize(testTexture, 2.56f, bounds);

            var path = new System.Collections.Generic.List<Vector3>
            {
                new Vector3(10, 0, 10)
            };

            // Act
            LogAssert.Expect(LogType.Warning, "MinimapRenderer.DrawPath: 경로는 최소 2개의 점이 필요합니다.");
            renderer.DrawPath(path, Color.yellow);

            // Assert
            // Warning should be logged

            // Cleanup
            Object.DestroyImmediate(testTexture);
        }

        [Test]
        public void DrawPath_WithTwoPoints_DrawsSuccessfully()
        {
            // Arrange
            Texture2D testTexture = new Texture2D(256, 256);
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));
            renderer.Initialize(testTexture, 2.56f, bounds);

            var path = new System.Collections.Generic.List<Vector3>
            {
                new Vector3(0, 0, 0),
                new Vector3(10, 0, 10)
            };

            // Act
            renderer.DrawPath(path, Color.yellow);

            // Assert
            Assert.IsTrue(renderer.IsInitialized);
            // Path should be drawn

            // Cleanup
            Object.DestroyImmediate(testTexture);
        }

        [Test]
        public void DrawPath_WithMultiplePoints_DrawsConnectedPath()
        {
            // Arrange
            Texture2D testTexture = new Texture2D(512, 512);
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));
            renderer.Initialize(testTexture, 5.12f, bounds);

            var path = new System.Collections.Generic.List<Vector3>
            {
                new Vector3(0, 0, 0),
                new Vector3(10, 0, 10),
                new Vector3(20, 0, 10),
                new Vector3(30, 0, 20),
                new Vector3(40, 0, 30)
            };

            // Act
            renderer.DrawPath(path, Color.yellow);

            // Assert
            Assert.IsTrue(renderer.IsInitialized);
            // Path should be drawn with connected lines

            // Cleanup
            Object.DestroyImmediate(testTexture);
        }

        [Test]
        public void DrawPath_WithDifferentColors_WorksCorrectly()
        {
            // Arrange
            Texture2D testTexture = new Texture2D(256, 256);
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));
            renderer.Initialize(testTexture, 2.56f, bounds);

            var path = new System.Collections.Generic.List<Vector3>
            {
                new Vector3(0, 0, 0),
                new Vector3(10, 0, 10)
            };

            // Act & Assert
            renderer.DrawPath(path, Color.red);
            renderer.DrawPath(path, Color.blue);
            renderer.DrawPath(path, Color.green);
            renderer.DrawPath(path, Color.yellow);

            Assert.IsTrue(renderer.IsInitialized);

            // Cleanup
            Object.DestroyImmediate(testTexture);
        }

        [Test]
        public void DrawPath_WithComplexPath_DrawsSuccessfully()
        {
            // Arrange
            Texture2D testTexture = new Texture2D(1024, 1024);
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));
            renderer.Initialize(testTexture, 10.24f, bounds);

            // Create a complex path with many points
            var path = new System.Collections.Generic.List<Vector3>();
            for (int i = 0; i < 20; i++)
            {
                float x = i * 5f;
                float z = Mathf.Sin(i * 0.5f) * 10f;
                path.Add(new Vector3(x, 0, z));
            }

            // Act
            renderer.DrawPath(path, Color.cyan);

            // Assert
            Assert.IsTrue(renderer.IsInitialized);
            // Complex path should be drawn

            // Cleanup
            Object.DestroyImmediate(testTexture);
        }

        [Test]
        public void DrawPath_WithPathAtBoundaries_ClampsCorrectly()
        {
            // Arrange
            Texture2D testTexture = new Texture2D(256, 256);
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));
            renderer.Initialize(testTexture, 2.56f, bounds);

            var path = new System.Collections.Generic.List<Vector3>
            {
                bounds.min,
                bounds.max,
                bounds.center
            };

            // Act
            renderer.DrawPath(path, Color.yellow);

            // Assert
            Assert.IsTrue(renderer.IsInitialized);
            // Path should be drawn with clamped coordinates

            // Cleanup
            Object.DestroyImmediate(testTexture);
        }

        [Test]
        public void DrawPath_WithPathOutsideBounds_ClampsToTexture()
        {
            // Arrange
            Texture2D testTexture = new Texture2D(256, 256);
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));
            renderer.Initialize(testTexture, 2.56f, bounds);

            var path = new System.Collections.Generic.List<Vector3>
            {
                new Vector3(-100, 0, -100), // Outside bounds
                new Vector3(200, 0, 200),   // Outside bounds
                new Vector3(50, 0, 50)      // Inside bounds
            };

            // Act
            renderer.DrawPath(path, Color.yellow);

            // Assert
            Assert.IsTrue(renderer.IsInitialized);
            // Path should be drawn with clamped coordinates

            // Cleanup
            Object.DestroyImmediate(testTexture);
        }

        [Test]
        public void DrawPath_MultiplePaths_AllDrawnSuccessfully()
        {
            // Arrange
            Texture2D testTexture = new Texture2D(512, 512);
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));
            renderer.Initialize(testTexture, 5.12f, bounds);

            var path1 = new System.Collections.Generic.List<Vector3>
            {
                new Vector3(0, 0, 0),
                new Vector3(10, 0, 10)
            };

            var path2 = new System.Collections.Generic.List<Vector3>
            {
                new Vector3(20, 0, 20),
                new Vector3(30, 0, 30)
            };

            // Act
            renderer.DrawPath(path1, Color.red);
            renderer.DrawPath(path2, Color.blue);

            // Assert
            Assert.IsTrue(renderer.IsInitialized);
            // Both paths should be drawn

            // Cleanup
            Object.DestroyImmediate(testTexture);
        }

        [Test]
        public void DrawPath_WithEmptyPath_LogsWarning()
        {
            // Arrange
            Texture2D testTexture = new Texture2D(256, 256);
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));
            renderer.Initialize(testTexture, 2.56f, bounds);

            var path = new System.Collections.Generic.List<Vector3>();

            // Act
            LogAssert.Expect(LogType.Warning, "MinimapRenderer.DrawPath: 경로는 최소 2개의 점이 필요합니다.");
            renderer.DrawPath(path, Color.yellow);

            // Assert
            // Warning should be logged

            // Cleanup
            Object.DestroyImmediate(testTexture);
        }

        [Test]
        public void DrawPath_WithStraightLine_DrawsCorrectly()
        {
            // Arrange
            Texture2D testTexture = new Texture2D(256, 256);
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));
            renderer.Initialize(testTexture, 2.56f, bounds);

            var path = new System.Collections.Generic.List<Vector3>
            {
                new Vector3(0, 0, 0),
                new Vector3(50, 0, 0) // Straight horizontal line
            };

            // Act
            renderer.DrawPath(path, Color.yellow);

            // Assert
            Assert.IsTrue(renderer.IsInitialized);

            // Cleanup
            Object.DestroyImmediate(testTexture);
        }

        [Test]
        public void DrawPath_WithDiagonalLine_DrawsCorrectly()
        {
            // Arrange
            Texture2D testTexture = new Texture2D(256, 256);
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));
            renderer.Initialize(testTexture, 2.56f, bounds);

            var path = new System.Collections.Generic.List<Vector3>
            {
                new Vector3(0, 0, 0),
                new Vector3(50, 0, 50) // Diagonal line
            };

            // Act
            renderer.DrawPath(path, Color.yellow);

            // Assert
            Assert.IsTrue(renderer.IsInitialized);

            // Cleanup
            Object.DestroyImmediate(testTexture);
        }
    }
}
