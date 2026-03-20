using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using System.IO;

namespace CityGenerator.Tests
{
    /// <summary>
    /// SaveMinimapToPNG 메서드 테스트
    /// Requirements: 16.6, 16.7
    /// </summary>
    [TestFixture]
    public class SaveMinimapTest
    {
        // 실제 저장 경로: MinimapGenerator.GetProceduralCityFolder() 기준
        private string TestDirectoryPath =>
            Path.Combine(Application.dataPath, "00. BMW", "CityGenerator", "CityMaps");

        [TearDown]
        public void TearDown()
        {
            // 테스트 후 생성된 파일 정리
            string dir = TestDirectoryPath;
            if (Directory.Exists(dir))
            {
                string[] files = Directory.GetFiles(dir, "Minimap_*.png");
                foreach (string file in files)
                {
                    File.Delete(file);
                }
            }
        }

        [Test]
        public void SaveMinimapToPNG_CreatesDirectoryIfNotExists()
        {
            // Arrange
            string dir = TestDirectoryPath;
            if (Directory.Exists(dir))
            {
                Directory.Delete(dir, true);
            }

            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 0, 100));
            MinimapGenerator generator = new MinimapGenerator(256, bounds);
            Texture2D minimap = new Texture2D(256, 256);

            // Act
            generator.SaveMinimapToPNG(minimap, 12345);

            // Assert
            Assert.IsTrue(Directory.Exists(dir), "디렉토리가 생성되어야 합니다.");

            // Cleanup
            Object.DestroyImmediate(minimap);
        }

        [Test]
        public void SaveMinimapToPNG_SavesFileWithSeedValue()
        {
            // Arrange
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 0, 100));
            MinimapGenerator generator = new MinimapGenerator(512, bounds);
            Texture2D minimap = new Texture2D(512, 512);
            int seedValue = 98765;

            // Act
            generator.SaveMinimapToPNG(minimap, seedValue);

            // Assert
            string expectedFilename = $"Minimap_Seed{seedValue}_512x512.png";
            string expectedPath = Path.Combine(TestDirectoryPath, expectedFilename);
            Assert.IsTrue(File.Exists(expectedPath), $"파일이 생성되어야 합니다: {expectedPath}");

            // Cleanup
            Object.DestroyImmediate(minimap);
        }

        [Test]
        public void SaveMinimapToPNG_SavesFileWithTimestamp_WhenSeedIsNegative()
        {
            // Arrange
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 0, 100));
            MinimapGenerator generator = new MinimapGenerator(256, bounds);
            Texture2D minimap = new Texture2D(256, 256);

            // Act
            generator.SaveMinimapToPNG(minimap, -1);

            // Assert
            string dir = TestDirectoryPath;
            string[] files = Directory.GetFiles(dir, "Minimap_*_256x256.png");
            Assert.IsTrue(files.Length > 0, "타임스탬프를 포함한 파일이 생성되어야 합니다.");

            // Cleanup
            Object.DestroyImmediate(minimap);
        }

        [Test]
        public void SaveMinimapToPNG_HandlesNullTexture()
        {
            // Arrange
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 0, 100));
            MinimapGenerator generator = new MinimapGenerator(256, bounds);

            // Act & Assert
            // null 텍스처 전달 시 내부에서 LogError가 발생하므로 예상 처리
            LogAssert.Expect(LogType.Error, new System.Text.RegularExpressions.Regex(".*null.*"));
            Assert.DoesNotThrow(() => generator.SaveMinimapToPNG(null, 12345));
        }

        [Test]
        public void SaveMinimapToPNG_CreatesValidPNGFile()
        {
            // Arrange
            Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 0, 100));
            MinimapGenerator generator = new MinimapGenerator(256, bounds);

            // 간단한 테스트 텍스처 생성
            Texture2D minimap = new Texture2D(256, 256);
            Color[] pixels = new Color[256 * 256];
            for (int i = 0; i < pixels.Length; i++)
            {
                pixels[i] = Color.white;
            }
            minimap.SetPixels(pixels);
            minimap.Apply();

            int seedValue = 54321;

            // Act
            generator.SaveMinimapToPNG(minimap, seedValue);

            // Assert
            string expectedFilename = $"Minimap_Seed{seedValue}_256x256.png";
            string expectedPath = Path.Combine(TestDirectoryPath, expectedFilename);

            Assert.IsTrue(File.Exists(expectedPath), "파일이 생성되어야 합니다.");

            // 파일 크기 확인 (PNG 헤더가 있는지 확인)
            FileInfo fileInfo = new FileInfo(expectedPath);
            Assert.Greater(fileInfo.Length, 0, "파일 크기가 0보다 커야 합니다.");

            // PNG 파일 헤더 확인
            byte[] fileBytes = File.ReadAllBytes(expectedPath);
            Assert.AreEqual(0x89, fileBytes[0], "PNG 매직 넘버 첫 바이트");
            Assert.AreEqual(0x50, fileBytes[1], "PNG 매직 넘버 두 번째 바이트 (P)");
            Assert.AreEqual(0x4E, fileBytes[2], "PNG 매직 넘버 세 번째 바이트 (N)");
            Assert.AreEqual(0x47, fileBytes[3], "PNG 매직 넘버 네 번째 바이트 (G)");

            // Cleanup
            Object.DestroyImmediate(minimap);
        }
    }
}
