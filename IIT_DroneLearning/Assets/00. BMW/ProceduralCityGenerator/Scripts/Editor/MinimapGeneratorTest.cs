using NUnit.Framework;
using UnityEngine;

namespace ProceduralCityGenerator.Tests
{
    /// <summary>
    /// MinimapGenerator 클래스의 단위 테스트
    /// **Validates: Requirements 16.1, 16.8**
    /// </summary>
    [TestFixture]
    public class MinimapGeneratorTest
    {
        [Test]
        public void Constructor_SetsResolutionCorrectly()
        {
            // Arrange
            int expectedResolution = 512;
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));

            // Act
            MinimapGenerator generator = new MinimapGenerator(expectedResolution, bounds);

            // Assert
            Assert.AreEqual(expectedResolution, generator.Resolution);
        }

        [Test]
        public void Constructor_SetsBoundsCorrectly()
        {
            // Arrange
            int resolution = 512;
            Bounds expectedBounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));

            // Act
            MinimapGenerator generator = new MinimapGenerator(resolution, expectedBounds);

            // Assert
            Assert.AreEqual(expectedBounds, generator.CityBounds);
        }

        [Test]
        public void Constructor_CalculatesPixelsPerMeterCorrectly_WhenWidthIsLarger()
        {
            // Arrange
            int resolution = 512;
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 80)); // Width (X) is larger

            // Act
            MinimapGenerator generator = new MinimapGenerator(resolution, bounds);

            // Assert
            float expectedPixelsPerMeter = 512f / 100f; // resolution / max dimension (100)
            Assert.AreEqual(expectedPixelsPerMeter, generator.PixelsPerMeter, 0.001f);
        }

        [Test]
        public void Constructor_CalculatesPixelsPerMeterCorrectly_WhenDepthIsLarger()
        {
            // Arrange
            int resolution = 1024;
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(80, 50, 120)); // Depth (Z) is larger

            // Act
            MinimapGenerator generator = new MinimapGenerator(resolution, bounds);

            // Assert
            float expectedPixelsPerMeter = 1024f / 120f; // resolution / max dimension (120)
            Assert.AreEqual(expectedPixelsPerMeter, generator.PixelsPerMeter, 0.001f);
        }

        [Test]
        public void Constructor_CalculatesPixelsPerMeterCorrectly_WhenDimensionsAreEqual()
        {
            // Arrange
            int resolution = 256;
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(50, 30, 50)); // Width and Depth are equal

            // Act
            MinimapGenerator generator = new MinimapGenerator(resolution, bounds);

            // Assert
            float expectedPixelsPerMeter = 256f / 50f; // resolution / max dimension (50)
            Assert.AreEqual(expectedPixelsPerMeter, generator.PixelsPerMeter, 0.001f);
        }

        [Test]
        public void Constructor_HandlesSmallBounds()
        {
            // Arrange
            int resolution = 512;
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(10, 5, 10));

            // Act
            MinimapGenerator generator = new MinimapGenerator(resolution, bounds);

            // Assert
            float expectedPixelsPerMeter = 512f / 10f; // High pixels per meter for small area
            Assert.AreEqual(expectedPixelsPerMeter, generator.PixelsPerMeter, 0.001f);
        }

        [Test]
        public void Constructor_HandlesLargeBounds()
        {
            // Arrange
            int resolution = 1024;
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(1000, 500, 1000));

            // Act
            MinimapGenerator generator = new MinimapGenerator(resolution, bounds);

            // Assert
            float expectedPixelsPerMeter = 1024f / 1000f; // Low pixels per meter for large area
            Assert.AreEqual(expectedPixelsPerMeter, generator.PixelsPerMeter, 0.001f);
        }

        [Test]
        public void Constructor_HandlesNonCenteredBounds()
        {
            // Arrange
            int resolution = 512;
            Vector3 center = new Vector3(50, 25, 50);
            Vector3 size = new Vector3(100, 50, 100);
            Bounds bounds = new Bounds(center, size);

            // Act
            MinimapGenerator generator = new MinimapGenerator(resolution, bounds);

            // Assert
            Assert.AreEqual(bounds, generator.CityBounds);
            float expectedPixelsPerMeter = 512f / 100f;
            Assert.AreEqual(expectedPixelsPerMeter, generator.PixelsPerMeter, 0.001f);
        }

        [Test]
        public void Constructor_HandlesRectangularBounds()
        {
            // Arrange
            int resolution = 512;
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(200, 50, 100)); // 2:1 ratio

            // Act
            MinimapGenerator generator = new MinimapGenerator(resolution, bounds);

            // Assert
            float expectedPixelsPerMeter = 512f / 200f; // Uses larger dimension
            Assert.AreEqual(expectedPixelsPerMeter, generator.PixelsPerMeter, 0.001f);
        }

        [Test]
        public void GenerateMinimap_CreatesTextureWithCorrectResolution()
        {
            // Arrange
            int resolution = 256;
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));
            MinimapGenerator generator = new MinimapGenerator(resolution, bounds);
            Building[] buildings = new Building[0];

            // Act
            Texture2D minimap = generator.GenerateMinimap(buildings);

            // Assert
            Assert.IsNotNull(minimap);
            Assert.AreEqual(resolution, minimap.width);
            Assert.AreEqual(resolution, minimap.height);
        }

        [Test]
        public void GenerateMinimap_HandlesNullBuildingsArray()
        {
            // Arrange
            int resolution = 256;
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));
            MinimapGenerator generator = new MinimapGenerator(resolution, bounds);

            // Act
            Texture2D minimap = generator.GenerateMinimap(null);

            // Assert
            Assert.IsNotNull(minimap);
            Assert.AreEqual(resolution, minimap.width);
            Assert.AreEqual(resolution, minimap.height);
        }

        [Test]
        public void GenerateMinimap_HandlesEmptyBuildingsArray()
        {
            // Arrange
            int resolution = 256;
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));
            MinimapGenerator generator = new MinimapGenerator(resolution, bounds);
            Building[] buildings = new Building[0];

            // Act
            Texture2D minimap = generator.GenerateMinimap(buildings);

            // Assert
            Assert.IsNotNull(minimap);
            // Empty minimap should have bright background
            Color centerPixel = minimap.GetPixel(resolution / 2, resolution / 2);
            Assert.Greater(centerPixel.r, 0.8f); // Bright background
        }

        [Test]
        public void GenerateMinimap_DrawsBuildingsWithCorrectColors()
        {
            // Arrange
            int resolution = 256;
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));
            MinimapGenerator generator = new MinimapGenerator(resolution, bounds);

            // Create test buildings with different heights
            GameObject go1 = new GameObject("TestBuilding1");
            GameObject go2 = new GameObject("TestBuilding2");
            GameObject go3 = new GameObject("TestBuilding3");

            GridCell cell1 = new GridCell { x = 0, z = 0, hasBuilding = true, buildingHeight = 10f, worldPosition = new Vector3(25, 0, 25) };
            GridCell cell2 = new GridCell { x = 1, z = 1, hasBuilding = true, buildingHeight = 50f, worldPosition = new Vector3(50, 0, 50) };
            GridCell cell3 = new GridCell { x = 2, z = 2, hasBuilding = true, buildingHeight = 100f, worldPosition = new Vector3(75, 0, 75) };

            Building building1 = new Building(go1, new Vector3(25, 0, 25), new Vector3(10, 10, 10), 10f, cell1);
            Building building2 = new Building(go2, new Vector3(50, 0, 50), new Vector3(10, 50, 10), 50f, cell2);
            Building building3 = new Building(go3, new Vector3(75, 0, 75), new Vector3(10, 100, 10), 100f, cell3);

            Building[] buildings = new Building[] { building1, building2, building3 };

            // Act
            Texture2D minimap = generator.GenerateMinimap(buildings);

            // Assert
            Assert.IsNotNull(minimap);

            // Get pixel colors at building positions
            Vector2Int pixel1 = WorldToPixel(generator, building1.position);
            Vector2Int pixel2 = WorldToPixel(generator, building2.position);
            Vector2Int pixel3 = WorldToPixel(generator, building3.position);

            Color color1 = minimap.GetPixel(pixel1.x, pixel1.y);
            Color color2 = minimap.GetPixel(pixel2.x, pixel2.y);
            Color color3 = minimap.GetPixel(pixel3.x, pixel3.y);

            // Lower buildings should be lighter (higher gray value)
            // Higher buildings should be darker (lower gray value)
            Assert.Greater(color1.r, color2.r, "Low building should be lighter than medium building");
            Assert.Greater(color2.r, color3.r, "Medium building should be lighter than high building");

            // Cleanup
            GameObject.DestroyImmediate(go1);
            GameObject.DestroyImmediate(go2);
            GameObject.DestroyImmediate(go3);
        }

        [Test]
        public void GenerateMinimap_HandlesBuiltingsAtBoundaryEdges()
        {
            // Arrange
            int resolution = 256;
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));
            MinimapGenerator generator = new MinimapGenerator(resolution, bounds);

            GameObject go = new GameObject("EdgeBuilding");
            GridCell cell = new GridCell { x = 0, z = 0, hasBuilding = true, buildingHeight = 50f, worldPosition = new Vector3(-50, 0, -50) };
            Building building = new Building(go, new Vector3(-50, 0, -50), new Vector3(10, 50, 10), 50f, cell);
            Building[] buildings = new Building[] { building };

            // Act
            Texture2D minimap = generator.GenerateMinimap(buildings);

            // Assert
            Assert.IsNotNull(minimap);
            // Should not throw exception even with edge buildings

            // Cleanup
            GameObject.DestroyImmediate(go);
        }

        [Test]
        public void DrawGridLines_DrawsGridOnMinimap()
        {
            // Arrange
            int resolution = 256;
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));
            MinimapGenerator generator = new MinimapGenerator(resolution, bounds);
            Building[] buildings = new Building[0];
            Texture2D minimap = generator.GenerateMinimap(buildings);
            
            // Get initial pixel color
            Color initialColor = minimap.GetPixel(0, 0);

            // Act
            generator.DrawGridLines(minimap, 10f); // 10 meter grid spacing
            minimap.Apply();

            // Assert
            // Grid lines should be drawn at regular intervals
            Color gridPixelColor = minimap.GetPixel(0, 0);
            
            // The color should be different after drawing grid lines (blended)
            Assert.AreNotEqual(initialColor, gridPixelColor, "Grid lines should modify pixel colors");
        }

        [Test]
        public void DrawGridLines_HandlesNullTexture()
        {
            // Arrange
            int resolution = 256;
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));
            MinimapGenerator generator = new MinimapGenerator(resolution, bounds);

            // Act & Assert
            // Should not throw exception
            Assert.DoesNotThrow(() => generator.DrawGridLines(null, 10f));
        }

        [Test]
        public void DrawGridLines_HandlesZeroGridSpacing()
        {
            // Arrange
            int resolution = 256;
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));
            MinimapGenerator generator = new MinimapGenerator(resolution, bounds);
            Building[] buildings = new Building[0];
            Texture2D minimap = generator.GenerateMinimap(buildings);

            // Act & Assert
            // Should not throw exception with zero spacing
            Assert.DoesNotThrow(() => generator.DrawGridLines(minimap, 0f));
        }

        [Test]
        public void DrawGridLines_HandlesNegativeGridSpacing()
        {
            // Arrange
            int resolution = 256;
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));
            MinimapGenerator generator = new MinimapGenerator(resolution, bounds);
            Building[] buildings = new Building[0];
            Texture2D minimap = generator.GenerateMinimap(buildings);

            // Act & Assert
            // Should not throw exception with negative spacing
            Assert.DoesNotThrow(() => generator.DrawGridLines(minimap, -5f));
        }

        // Helper method to convert world position to pixel (mimics internal logic)
        private Vector2Int WorldToPixel(MinimapGenerator generator, Vector3 worldPosition)
        {
            Vector3 relativePosition = worldPosition - generator.CityBounds.min;
            int pixelX = Mathf.RoundToInt(relativePosition.x * generator.PixelsPerMeter);
            int pixelY = Mathf.RoundToInt(relativePosition.z * generator.PixelsPerMeter);
            pixelX = Mathf.Clamp(pixelX, 0, generator.Resolution - 1);
            pixelY = Mathf.Clamp(pixelY, 0, generator.Resolution - 1);
            return new Vector2Int(pixelX, pixelY);
        }
    }
}
