using NUnit.Framework;
using System.Collections.Generic;
using UnityEngine;

namespace CityGenerator.Tests
{
    /// <summary>
    /// 전략적 위치 마커 표시 기능의 단위 테스트
    /// **Validates: Requirements 17.5**
    /// </summary>
    [TestFixture]
    public class StrategicMarkersTest
    {
        [Test]
        public void GenerateMinimap_DrawsStrategicMarkers_WhenLocationsProvided()
        {
            // Arrange
            int resolution = 256;
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));
            MinimapGenerator generator = new MinimapGenerator(resolution, bounds);

            // Create test strategic locations
            List<StrategicLocation> strategicLocations = new List<StrategicLocation>
            {
                new StrategicLocation
                {
                    position = new Vector3(25, 0, 25),
                    locationType = StrategyType.CoverPoint,
                    dangerScore = 0.3f,
                    visibilityScore = 0.2f,
                    connectedNodes = new List<int> { 1 }
                },
                new StrategicLocation
                {
                    position = new Vector3(50, 0, 50),
                    locationType = StrategyType.Intersection,
                    dangerScore = 0.7f,
                    visibilityScore = 0.8f,
                    connectedNodes = new List<int> { 2 }
                }
            };

            Building[] buildings = new Building[0];

            // Act
            Texture2D minimap = generator.GenerateMinimap(buildings, null, strategicLocations);

            // Assert
            Assert.IsNotNull(minimap);

            // Check that markers are drawn at the correct positions
            Vector2Int pixel1 = WorldToPixel(generator, strategicLocations[0].position);
            Vector2Int pixel2 = WorldToPixel(generator, strategicLocations[1].position);

            Color color1 = minimap.GetPixel(pixel1.x, pixel1.y);
            Color color2 = minimap.GetPixel(pixel2.x, pixel2.y);

            // CoverPoint should be blue
            Assert.AreEqual(Color.blue, color1, "CoverPoint marker should be blue");

            // Intersection should be yellow
            Assert.AreEqual(Color.yellow, color2, "Intersection marker should be yellow");
        }

        [Test]
        public void GenerateMinimap_HandlesNullStrategicLocations()
        {
            // Arrange
            int resolution = 256;
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));
            MinimapGenerator generator = new MinimapGenerator(resolution, bounds);
            Building[] buildings = new Building[0];

            // Act
            Texture2D minimap = generator.GenerateMinimap(buildings, null, null);

            // Assert
            Assert.IsNotNull(minimap);
            Assert.AreEqual(resolution, minimap.width);
            Assert.AreEqual(resolution, minimap.height);
        }

        [Test]
        public void GenerateMinimap_HandlesEmptyStrategicLocations()
        {
            // Arrange
            int resolution = 256;
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));
            MinimapGenerator generator = new MinimapGenerator(resolution, bounds);
            Building[] buildings = new Building[0];
            List<StrategicLocation> strategicLocations = new List<StrategicLocation>();

            // Act
            Texture2D minimap = generator.GenerateMinimap(buildings, null, strategicLocations);

            // Assert
            Assert.IsNotNull(minimap);
            Assert.AreEqual(resolution, minimap.width);
            Assert.AreEqual(resolution, minimap.height);
        }

        [Test]
        public void GenerateMinimap_DrawsAllStrategyTypes_WithCorrectColors()
        {
            // Arrange
            int resolution = 512;
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));
            MinimapGenerator generator = new MinimapGenerator(resolution, bounds);

            // Create strategic locations for all types
            List<StrategicLocation> strategicLocations = new List<StrategicLocation>
            {
                new StrategicLocation
                {
                    position = new Vector3(10, 0, 10),
                    locationType = StrategyType.CoverPoint,
                    connectedNodes = new List<int> { 1 }
                },
                new StrategicLocation
                {
                    position = new Vector3(30, 0, 30),
                    locationType = StrategyType.Intersection,
                    connectedNodes = new List<int> { 2 }
                },
                new StrategicLocation
                {
                    position = new Vector3(-10, 0, -10),
                    locationType = StrategyType.DeadEnd,
                    connectedNodes = new List<int> { 3 }
                },
                new StrategicLocation
                {
                    position = new Vector3(-30, 0, -30),
                    locationType = StrategyType.OpenArea,
                    connectedNodes = new List<int> { 4 }
                },
                new StrategicLocation
                {
                    position = new Vector3(0, 0, 0),
                    locationType = StrategyType.DetourPath,
                    connectedNodes = new List<int> { 5 }
                }
            };

            Building[] buildings = new Building[0];

            // Act
            Texture2D minimap = generator.GenerateMinimap(buildings, null, strategicLocations);

            // Assert
            Assert.IsNotNull(minimap);

            // Verify each marker color
            Vector2Int pixel1 = WorldToPixel(generator, strategicLocations[0].position);
            Vector2Int pixel2 = WorldToPixel(generator, strategicLocations[1].position);
            Vector2Int pixel3 = WorldToPixel(generator, strategicLocations[2].position);
            Vector2Int pixel4 = WorldToPixel(generator, strategicLocations[3].position);
            Vector2Int pixel5 = WorldToPixel(generator, strategicLocations[4].position);

            Color color1 = minimap.GetPixel(pixel1.x, pixel1.y);
            Color color2 = minimap.GetPixel(pixel2.x, pixel2.y);
            Color color3 = minimap.GetPixel(pixel3.x, pixel3.y);
            Color color4 = minimap.GetPixel(pixel4.x, pixel4.y);
            Color color5 = minimap.GetPixel(pixel5.x, pixel5.y);

            // Verify colors according to requirements
            Assert.AreEqual(Color.blue, color1, "CoverPoint should be blue");
            Assert.AreEqual(Color.yellow, color2, "Intersection should be yellow");
            Assert.AreEqual(Color.red, color3, "DeadEnd should be red");
            // RGBA32 텍스처 저장 시 0.5f → 128/255 ≈ 0.502로 양자화되므로 근사값 비교
            Assert.AreEqual(1f, color4.r, 0.01f, "OpenArea red channel should be ~1");
            Assert.AreEqual(0.5f, color4.g, 0.01f, "OpenArea green channel should be ~0.5");
            Assert.AreEqual(0f, color4.b, 0.01f, "OpenArea blue channel should be ~0");
            Assert.AreEqual(Color.green, color5, "DetourPath should be green");
        }

        [Test]
        public void GenerateMinimap_DrawsMarkersOverBuildings()
        {
            // Arrange
            int resolution = 256;
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 50, 100));
            MinimapGenerator generator = new MinimapGenerator(resolution, bounds);

            // Create a building
            GameObject go = new GameObject("TestBuilding");
            GridCell cell = new GridCell { x = 0, z = 0, hasBuilding = true, buildingHeight = 50f, worldPosition = new Vector3(50, 0, 50) };
            Building building = new Building(go, new Vector3(50, 0, 50), new Vector3(10, 50, 10), 50f, cell);
            Building[] buildings = new Building[] { building };

            // Create a strategic location at the same position
            List<StrategicLocation> strategicLocations = new List<StrategicLocation>
            {
                new StrategicLocation
                {
                    position = new Vector3(50, 0, 50),
                    locationType = StrategyType.CoverPoint,
                    connectedNodes = new List<int> { 1 }
                }
            };

            // Act
            Texture2D minimap = generator.GenerateMinimap(buildings, null, strategicLocations);

            // Assert
            Assert.IsNotNull(minimap);

            // The marker should be drawn over the building (blue color should be visible)
            Vector2Int pixel = WorldToPixel(generator, strategicLocations[0].position);
            Color color = minimap.GetPixel(pixel.x, pixel.y);

            // Should be blue (marker color), not gray (building color)
            Assert.AreEqual(Color.blue, color, "Marker should be drawn over building");

            // Cleanup
            GameObject.DestroyImmediate(go);
        }

        // Helper method to convert world position to pixel
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
