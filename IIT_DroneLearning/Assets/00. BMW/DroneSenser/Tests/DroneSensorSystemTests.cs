using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace BMW.DroneSensor
{
    /// <summary>
    /// Unit tests for DroneSensorSystem
    /// </summary>
    public class DroneSensorSystemTests
    {
        private GameObject _testGameObject;
        private DroneSensorSystem _sensorSystem;

        [SetUp]
        public void SetUp()
        {
            // Create a test GameObject with DroneSensorSystem component
            _testGameObject = new GameObject("TestDrone");
            _sensorSystem = _testGameObject.AddComponent<DroneSensorSystem>();
        }

        [TearDown]
        public void TearDown()
        {
            // Clean up test objects
            if (_testGameObject != null)
            {
                Object.DestroyImmediate(_testGameObject);
            }
        }

        #region SetMaxDetectionRange Tests

        /// <summary>
        /// Test: SetMaxDetectionRange with valid positive value
        /// Requirement: 2.3
        /// </summary>
        [Test]
        public void SetMaxDetectionRange_WithValidValue_SetsCorrectly()
        {
            // Arrange
            float expectedRange = 100f;

            // Act
            _sensorSystem.SetMaxDetectionRange(expectedRange);

            // Assert
            Assert.AreEqual(expectedRange, _sensorSystem.MaxDetectionRange, 0.001f,
                "MaxDetectionRange should be set to the provided value");
        }

        /// <summary>
        /// Test: SetMaxDetectionRange with zero value
        /// Requirement: 2.3 - Input validation
        /// </summary>
        [Test]
        public void SetMaxDetectionRange_WithZeroValue_ResetsToDefault()
        {
            // Arrange
            float invalidRange = 0f;
            float expectedDefault = 50f;

            // Act
            _sensorSystem.SetMaxDetectionRange(invalidRange);

            // Assert
            Assert.AreEqual(expectedDefault, _sensorSystem.MaxDetectionRange, 0.001f,
                "MaxDetectionRange should be reset to default 50m when zero is provided");
        }

        /// <summary>
        /// Test: SetMaxDetectionRange with negative value
        /// Requirement: 2.3 - Input validation
        /// </summary>
        [Test]
        public void SetMaxDetectionRange_WithNegativeValue_ResetsToDefault()
        {
            // Arrange
            float invalidRange = -10f;
            float expectedDefault = 50f;

            // Act
            _sensorSystem.SetMaxDetectionRange(invalidRange);

            // Assert
            Assert.AreEqual(expectedDefault, _sensorSystem.MaxDetectionRange, 0.001f,
                "MaxDetectionRange should be reset to default 50m when negative value is provided");
        }

        /// <summary>
        /// Test: SetMaxDetectionRange multiple times
        /// Requirement: 2.3 - Runtime setting
        /// </summary>
        [Test]
        public void SetMaxDetectionRange_MultipleTimes_UpdatesCorrectly()
        {
            // Arrange & Act & Assert
            _sensorSystem.SetMaxDetectionRange(30f);
            Assert.AreEqual(30f, _sensorSystem.MaxDetectionRange, 0.001f);

            _sensorSystem.SetMaxDetectionRange(75f);
            Assert.AreEqual(75f, _sensorSystem.MaxDetectionRange, 0.001f);

            _sensorSystem.SetMaxDetectionRange(120f);
            Assert.AreEqual(120f, _sensorSystem.MaxDetectionRange, 0.001f);
        }

        /// <summary>
        /// Test: SetMaxDetectionRange with very small positive value
        /// Requirement: 2.3
        /// </summary>
        [Test]
        public void SetMaxDetectionRange_WithSmallPositiveValue_SetsCorrectly()
        {
            // Arrange
            float smallRange = 0.1f;

            // Act
            _sensorSystem.SetMaxDetectionRange(smallRange);

            // Assert
            Assert.AreEqual(smallRange, _sensorSystem.MaxDetectionRange, 0.001f,
                "MaxDetectionRange should accept small positive values");
        }

        /// <summary>
        /// Test: SetMaxDetectionRange with very large value
        /// Requirement: 2.3
        /// </summary>
        [Test]
        public void SetMaxDetectionRange_WithLargeValue_SetsCorrectly()
        {
            // Arrange
            float largeRange = 10000f;

            // Act
            _sensorSystem.SetMaxDetectionRange(largeRange);

            // Assert
            Assert.AreEqual(largeRange, _sensorSystem.MaxDetectionRange, 0.001f,
                "MaxDetectionRange should accept large positive values");
        }

        #endregion

        #region GetForwardRayDirection Tests

        /// <summary>
        /// Test: GetForwardRayDirection returns Middle layer N direction in world space
        /// Requirement: 7.2
        /// </summary>
        [Test]
        public void GetForwardRayDirection_ReturnsMiddleLayerNorthDirection()
        {
            // Arrange - Default rotation (no rotation)
            _testGameObject.transform.rotation = Quaternion.identity;

            // Act
            Vector3 forwardDirection = _sensorSystem.GetForwardRayDirection();

            // Assert - Should point forward (0, 0, 1) in world space for Middle layer N direction
            Assert.AreEqual(0f, forwardDirection.x, 0.001f, "Forward ray X component should be 0");
            Assert.AreEqual(0f, forwardDirection.y, 0.001f, "Forward ray Y component should be 0 (horizontal)");
            Assert.AreEqual(1f, forwardDirection.z, 0.001f, "Forward ray Z component should be 1 (forward)");
        }

        /// <summary>
        /// Test: GetForwardRayDirection transforms with drone rotation
        /// Requirement: 7.2 - World space transformation
        /// </summary>
        [Test]
        public void GetForwardRayDirection_TransformsWithDroneRotation()
        {
            // Arrange - Rotate drone 90 degrees around Y axis (yaw right)
            _testGameObject.transform.rotation = Quaternion.Euler(0f, 90f, 0f);

            // Act
            Vector3 forwardDirection = _sensorSystem.GetForwardRayDirection();

            // Assert - Should now point right (1, 0, 0) in world space
            Assert.AreEqual(1f, forwardDirection.x, 0.001f, "Forward ray should point right after 90° yaw");
            Assert.AreEqual(0f, forwardDirection.y, 0.001f, "Forward ray Y component should remain 0");
            Assert.AreEqual(0f, forwardDirection.z, 0.001f, "Forward ray Z component should be 0 after rotation");
        }

        /// <summary>
        /// Test: GetForwardRayDirection is normalized
        /// Requirement: 7.2 - Direction vector should be unit length
        /// </summary>
        [Test]
        public void GetForwardRayDirection_IsNormalized()
        {
            // Arrange - Random rotation
            _testGameObject.transform.rotation = Quaternion.Euler(45f, 30f, 15f);

            // Act
            Vector3 forwardDirection = _sensorSystem.GetForwardRayDirection();

            // Assert - Should be unit length (magnitude = 1)
            float magnitude = forwardDirection.magnitude;
            Assert.AreEqual(1f, magnitude, 0.001f, "Forward ray direction should be normalized (unit length)");
        }

        /// <summary>
        /// Test: GetForwardRayDirection before initialization returns default forward
        /// Requirement: 7.2 - Error handling
        /// </summary>
        [Test]
        public void GetForwardRayDirection_BeforeInitialization_ReturnsDefaultForward()
        {
            // Arrange - Create new sensor without initialization
            GameObject newObj = new GameObject("UninitializedDrone");
            DroneSensorSystem uninitializedSensor = newObj.AddComponent<DroneSensorSystem>();

            // Expect error log
            LogAssert.Expect(LogType.Error, System.Text.RegularExpressions.Regex.Escape("[DroneSensorSystem] 센서가 아직 초기화되지 않았습니다."));

            // Act
            Vector3 forwardDirection = uninitializedSensor.GetForwardRayDirection();

            // Assert - Should return Vector3.forward as fallback
            Assert.AreEqual(Vector3.forward, forwardDirection, "Should return default forward vector before initialization");

            // Cleanup
            Object.DestroyImmediate(newObj);
        }

        #endregion

        #region SetRayEnabled Tests

        /// <summary>
        /// Test: SetRayEnabled with valid index enables ray
        /// Requirement: 2.4
        /// </summary>
        [Test]
        public void SetRayEnabled_WithValidIndex_EnablesRay()
        {
            // Arrange
            int rayIndex = 10;

            // Act
            _sensorSystem.SetRayEnabled(rayIndex, true);

            // Assert - Ray should be enabled (we can't directly check IsEnabled, but we can verify no warning)
            // This test verifies the method accepts valid indices without errors
            LogAssert.NoUnexpectedReceived();
        }

        /// <summary>
        /// Test: SetRayEnabled with valid index disables ray
        /// Requirement: 2.4
        /// </summary>
        [Test]
        public void SetRayEnabled_WithValidIndex_DisablesRay()
        {
            // Arrange
            int rayIndex = 15;

            // Act
            _sensorSystem.SetRayEnabled(rayIndex, false);

            // Assert - Ray should be disabled (we can't directly check IsEnabled, but we can verify no warning)
            LogAssert.NoUnexpectedReceived();
        }

        /// <summary>
        /// Test: SetRayEnabled with index 0 (Top ray)
        /// Requirement: 2.4
        /// </summary>
        [Test]
        public void SetRayEnabled_WithIndexZero_WorksCorrectly()
        {
            // Arrange
            int rayIndex = 0;

            // Act
            _sensorSystem.SetRayEnabled(rayIndex, false);

            // Assert
            LogAssert.NoUnexpectedReceived();
        }

        /// <summary>
        /// Test: SetRayEnabled with index 25 (Bottom ray)
        /// Requirement: 2.4
        /// </summary>
        [Test]
        public void SetRayEnabled_WithIndexTwentyFive_WorksCorrectly()
        {
            // Arrange
            int rayIndex = 25;

            // Act
            _sensorSystem.SetRayEnabled(rayIndex, true);

            // Assert
            LogAssert.NoUnexpectedReceived();
        }

        /// <summary>
        /// Test: SetRayEnabled with negative index logs warning
        /// Requirement: 2.4 - Index range validation
        /// </summary>
        [Test]
        public void SetRayEnabled_WithNegativeIndex_LogsWarning()
        {
            // Arrange
            int invalidIndex = -1;

            // Expect warning log
            LogAssert.Expect(LogType.Warning, System.Text.RegularExpressions.Regex.Escape("[DroneSensorSystem] 잘못된 레이 인덱스: -1. 유효 범위는 0~25입니다."));

            // Act
            _sensorSystem.SetRayEnabled(invalidIndex, true);
        }

        /// <summary>
        /// Test: SetRayEnabled with index 26 logs warning
        /// Requirement: 2.4 - Index range validation
        /// </summary>
        [Test]
        public void SetRayEnabled_WithIndexTwentySix_LogsWarning()
        {
            // Arrange
            int invalidIndex = 26;

            // Expect warning log
            LogAssert.Expect(LogType.Warning, System.Text.RegularExpressions.Regex.Escape("[DroneSensorSystem] 잘못된 레이 인덱스: 26. 유효 범위는 0~25입니다."));

            // Act
            _sensorSystem.SetRayEnabled(invalidIndex, false);
        }

        /// <summary>
        /// Test: SetRayEnabled with very large index logs warning
        /// Requirement: 2.4 - Index range validation
        /// </summary>
        [Test]
        public void SetRayEnabled_WithLargeIndex_LogsWarning()
        {
            // Arrange
            int invalidIndex = 100;

            // Expect warning log
            LogAssert.Expect(LogType.Warning, System.Text.RegularExpressions.Regex.Escape("[DroneSensorSystem] 잘못된 레이 인덱스: 100. 유효 범위는 0~25입니다."));

            // Act
            _sensorSystem.SetRayEnabled(invalidIndex, true);
        }

        /// <summary>
        /// Test: SetRayEnabled multiple times on same ray
        /// Requirement: 2.4
        /// </summary>
        [Test]
        public void SetRayEnabled_MultipleTimes_WorksCorrectly()
        {
            // Arrange
            int rayIndex = 12;

            // Act & Assert - Should work without errors
            _sensorSystem.SetRayEnabled(rayIndex, false);
            LogAssert.NoUnexpectedReceived();

            _sensorSystem.SetRayEnabled(rayIndex, true);
            LogAssert.NoUnexpectedReceived();

            _sensorSystem.SetRayEnabled(rayIndex, false);
            LogAssert.NoUnexpectedReceived();
        }

        /// <summary>
        /// Test: SetRayEnabled on all rays (boundary test)
        /// Requirement: 2.4
        /// </summary>
        [Test]
        public void SetRayEnabled_OnAllRays_WorksCorrectly()
        {
            // Act & Assert - All indices 0-25 should be valid
            for (int i = 0; i < 26; i++)
            {
                _sensorSystem.SetRayEnabled(i, false);
                LogAssert.NoUnexpectedReceived();
            }

            for (int i = 0; i < 26; i++)
            {
                _sensorSystem.SetRayEnabled(i, true);
                LogAssert.NoUnexpectedReceived();
            }
        }

        #endregion

        #region SetDebugVisualization Tests

        /// <summary>
        /// Test: SetDebugVisualization enables debug rays
        /// Requirement: 2.5
        /// </summary>
        [Test]
        public void SetDebugVisualization_WithTrue_EnablesDebugRays()
        {
            // Arrange
            _sensorSystem.ShowDebugRays = false;

            // Act
            _sensorSystem.SetDebugVisualization(true);

            // Assert
            Assert.IsTrue(_sensorSystem.ShowDebugRays,
                "ShowDebugRays should be enabled when SetDebugVisualization(true) is called");
        }

        /// <summary>
        /// Test: SetDebugVisualization disables debug rays
        /// Requirement: 2.5
        /// </summary>
        [Test]
        public void SetDebugVisualization_WithFalse_DisablesDebugRays()
        {
            // Arrange
            _sensorSystem.ShowDebugRays = true;

            // Act
            _sensorSystem.SetDebugVisualization(false);

            // Assert
            Assert.IsFalse(_sensorSystem.ShowDebugRays,
                "ShowDebugRays should be disabled when SetDebugVisualization(false) is called");
        }

        /// <summary>
        /// Test: SetDebugVisualization multiple times
        /// Requirement: 2.5 - Runtime toggling
        /// </summary>
        [Test]
        public void SetDebugVisualization_MultipleTimes_TogglesCorrectly()
        {
            // Act & Assert
            _sensorSystem.SetDebugVisualization(true);
            Assert.IsTrue(_sensorSystem.ShowDebugRays, "Should be enabled after first call");

            _sensorSystem.SetDebugVisualization(false);
            Assert.IsFalse(_sensorSystem.ShowDebugRays, "Should be disabled after second call");

            _sensorSystem.SetDebugVisualization(true);
            Assert.IsTrue(_sensorSystem.ShowDebugRays, "Should be enabled after third call");
        }

        /// <summary>
        /// Test: SetDebugVisualization with same value multiple times
        /// Requirement: 2.5 - Idempotent operation
        /// </summary>
        [Test]
        public void SetDebugVisualization_WithSameValue_RemainsConsistent()
        {
            // Act & Assert - Setting to true multiple times
            _sensorSystem.SetDebugVisualization(true);
            Assert.IsTrue(_sensorSystem.ShowDebugRays);

            _sensorSystem.SetDebugVisualization(true);
            Assert.IsTrue(_sensorSystem.ShowDebugRays, "Should remain true when set to true again");

            // Act & Assert - Setting to false multiple times
            _sensorSystem.SetDebugVisualization(false);
            Assert.IsFalse(_sensorSystem.ShowDebugRays);

            _sensorSystem.SetDebugVisualization(false);
            Assert.IsFalse(_sensorSystem.ShowDebugRays, "Should remain false when set to false again");
        }

        #endregion
    }
}
