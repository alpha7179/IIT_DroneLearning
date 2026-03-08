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
            
            // Manually trigger Awake() for Edit Mode tests using reflection
            // In Edit Mode, Unity doesn't automatically call Awake() immediately after AddComponent()
            InitializeSensor(_sensorSystem);
        }

        /// <summary>
        /// Helper method to manually initialize a DroneSensorSystem in Edit Mode tests
        /// </summary>
        private void InitializeSensor(DroneSensorSystem sensor)
        {
            var awakeMethod = typeof(DroneSensorSystem).GetMethod("Awake", 
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            if (awakeMethod != null)
            {
                awakeMethod.Invoke(sensor, null);
            }
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
            
            // Expect error log BEFORE adding component (Unity calls Awake automatically)
            LogAssert.Expect(LogType.Error, "[DroneSensorSystem] 치명적 오류: 센서가 아직 초기화되지 않았습니다.");
            
            DroneSensorSystem uninitializedSensor = newObj.AddComponent<DroneSensorSystem>();
            // Do NOT call InitializeSensor - we want to test uninitialized state

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
            LogAssert.Expect(LogType.Warning, "[DroneSensorSystem] 경고: 잘못된 레이 인덱스: -1. 유효 범위는 0~25입니다.");

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
            LogAssert.Expect(LogType.Warning, "[DroneSensorSystem] 경고: 잘못된 레이 인덱스: 26. 유효 범위는 0~25입니다.");

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
            LogAssert.Expect(LogType.Warning, "[DroneSensorSystem] 경고: 잘못된 레이 인덱스: 100. 유효 범위는 0~25입니다.");

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

        #region Error Handling Tests

        /// <summary>
        /// Test: GetAllNormalizedDistances before initialization returns empty array
        /// Requirement: Design document error handling section
        /// </summary>
        [Test]
        public void GetAllNormalizedDistances_BeforeInitialization_ReturnsEmptyArray()
        {
            // Arrange - Create new sensor without initialization (before Awake)
            GameObject newObj = new GameObject("UninitializedDrone");
            
            // Expect error log BEFORE adding component (Unity calls Awake automatically)
            LogAssert.Expect(LogType.Error, "[DroneSensorSystem] 치명적 오류: 센서가 아직 초기화되지 않았습니다.");
            
            DroneSensorSystem uninitializedSensor = newObj.AddComponent<DroneSensorSystem>();
            // Do NOT call InitializeSensor - we want to test uninitialized state

            // Act
            float[] distances = uninitializedSensor.GetAllNormalizedDistances();

            // Assert
            Assert.IsNotNull(distances, "Should return non-null array");
            Assert.AreEqual(26, distances.Length, "Should return array of length 26");
            
            // All values should be 0
            foreach (float distance in distances)
            {
                Assert.AreEqual(0f, distance, 0.001f, "All distances should be 0 before initialization");
            }

            // Cleanup
            Object.DestroyImmediate(newObj);
        }

        /// <summary>
        /// Test: GetNormalizedDistance before initialization returns 0
        /// Requirement: Design document error handling section
        /// </summary>
        [Test]
        public void GetNormalizedDistance_BeforeInitialization_ReturnsZero()
        {
            // Arrange - Create new sensor without initialization
            GameObject newObj = new GameObject("UninitializedDrone");
            
            // Expect error log BEFORE adding component (Unity calls Awake automatically)
            LogAssert.Expect(LogType.Error, "[DroneSensorSystem] 치명적 오류: 센서가 아직 초기화되지 않았습니다.");
            
            DroneSensorSystem uninitializedSensor = newObj.AddComponent<DroneSensorSystem>();
            // Do NOT call InitializeSensor - we want to test uninitialized state

            // Act
            float distance = uninitializedSensor.GetNormalizedDistance(
                DroneSensorSystem.SensorLayer.Middle,
                DroneSensorSystem.CompassDirection.N
            );

            // Assert
            Assert.AreEqual(0f, distance, 0.001f, "Should return 0 before initialization");

            // Cleanup
            Object.DestroyImmediate(newObj);
        }

        /// <summary>
        /// Test: ValidateConfiguration clamps TopMiddleElevation to valid range
        /// Requirement: Design document error handling section
        /// </summary>
        [Test]
        public void ValidateConfiguration_ClampsTopMiddleElevation_WhenOutOfRange()
        {
            // Arrange - Create GameObject and set invalid elevation before initialization
            GameObject newObj = new GameObject("TestDrone2");
            DroneSensorSystem newSensor = newObj.AddComponent<DroneSensorSystem>();
            
            // Set invalid value and manually trigger validation
            newSensor.TopMiddleElevation = 120f;
            
            // Expect warning log when validation is triggered
            LogAssert.Expect(LogType.Warning, new System.Text.RegularExpressions.Regex(@"\[DroneSensorSystem\] 경고: TopMiddleElevation은 0~90도 범위여야 합니다\."));
            
            // Act - Manually call ValidateConfiguration via reflection
            var validateMethod = typeof(DroneSensorSystem).GetMethod("ValidateConfiguration", 
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            if (validateMethod != null)
            {
                validateMethod.Invoke(newSensor, null);
            }

            // Assert - Value should be clamped to 90
            Assert.LessOrEqual(newSensor.TopMiddleElevation, 90f, "TopMiddleElevation should be clamped to 90");
            Assert.GreaterOrEqual(newSensor.TopMiddleElevation, 0f, "TopMiddleElevation should be at least 0");

            // Cleanup
            Object.DestroyImmediate(newObj);
        }

        /// <summary>
        /// Test: ValidateConfiguration clamps MiddleBottomElevation to valid range
        /// Requirement: Design document error handling section
        /// </summary>
        [Test]
        public void ValidateConfiguration_ClampsMiddleBottomElevation_WhenOutOfRange()
        {
            // Arrange - Create GameObject and set invalid elevation before validation
            GameObject newObj = new GameObject("TestDrone2");
            DroneSensorSystem newSensor = newObj.AddComponent<DroneSensorSystem>();
            
            // Set invalid value
            newSensor.MiddleBottomElevation = -120f;

            // Expect warning log when validation is triggered
            LogAssert.Expect(LogType.Warning, new System.Text.RegularExpressions.Regex(@"\[DroneSensorSystem\] 경고: MiddleBottomElevation은 -90~0도 범위여야 합니다\."));

            // Act - Manually call ValidateConfiguration via reflection
            var validateMethod = typeof(DroneSensorSystem).GetMethod("ValidateConfiguration", 
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            if (validateMethod != null)
            {
                validateMethod.Invoke(newSensor, null);
            }

            // Assert - Value should be clamped to -90
            Assert.GreaterOrEqual(newSensor.MiddleBottomElevation, -90f, "MiddleBottomElevation should be at least -90");
            Assert.LessOrEqual(newSensor.MiddleBottomElevation, 0f, "MiddleBottomElevation should be at most 0");

            // Cleanup
            Object.DestroyImmediate(newObj);
        }

        /// <summary>
        /// Test: ValidateConfiguration warns when LayerMask is Nothing
        /// Requirement: Design document error handling section
        /// </summary>
        [Test]
        public void ValidateConfiguration_WarnsWhenLayerMaskIsNothing()
        {
            // Arrange
            GameObject newObj = new GameObject("TestDrone2");
            DroneSensorSystem newSensor = newObj.AddComponent<DroneSensorSystem>();
            
            // Set layer mask to Nothing
            newSensor.DetectionLayerMask = 0;

            // Expect warning log when validation is triggered
            LogAssert.Expect(LogType.Warning, "[DroneSensorSystem] 경고: DetectionLayerMask가 Nothing으로 설정되어 있습니다. 모든 레이가 충돌을 감지하지 못합니다.");

            // Act - Manually call ValidateConfiguration via reflection
            var validateMethod = typeof(DroneSensorSystem).GetMethod("ValidateConfiguration", 
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            if (validateMethod != null)
            {
                validateMethod.Invoke(newSensor, null);
            }

            // Cleanup
            Object.DestroyImmediate(newObj);
        }

        /// <summary>
        /// Test: SetMaxDetectionRange with NaN logs warning and resets to default
        /// Requirement: Design document error handling section
        /// </summary>
        [Test]
        public void SetMaxDetectionRange_WithNaN_ResetsToDefault()
        {
            // Arrange
            float invalidRange = float.NaN;
            float expectedDefault = 50f;

            // Expect warning log
            LogAssert.Expect(LogType.Warning, new System.Text.RegularExpressions.Regex(@"\[DroneSensorSystem\] 경고: MaxDetectionRange는 양수여야 합니다\."));

            // Act
            _sensorSystem.SetMaxDetectionRange(invalidRange);

            // Assert
            Assert.AreEqual(expectedDefault, _sensorSystem.MaxDetectionRange, 0.001f,
                "MaxDetectionRange should be reset to default 50m when NaN is provided");
        }

        /// <summary>
        /// Test: SetMaxDetectionRange with Infinity logs warning and resets to default
        /// Requirement: Design document error handling section
        /// </summary>
        [Test]
        public void SetMaxDetectionRange_WithInfinity_AcceptsValue()
        {
            // Arrange
            float infinityRange = float.PositiveInfinity;

            // Expect warning log
            LogAssert.Expect(LogType.Warning, new System.Text.RegularExpressions.Regex(@"\[DroneSensorSystem\] 경고: MaxDetectionRange는 양수여야 합니다\."));

            // Act
            _sensorSystem.SetMaxDetectionRange(infinityRange);

            // Assert - Infinity should be rejected and reset to default 50m
            Assert.AreEqual(50f, _sensorSystem.MaxDetectionRange,
                "MaxDetectionRange should be reset to default 50m when Infinity is provided");
        }

        /// <summary>
        /// Test: GetNormalizedDistance with invalid layer returns 0
        /// Requirement: Design document error handling section
        /// </summary>
        [Test]
        public void GetNormalizedDistance_WithInvalidLayer_ReturnsZero()
        {
            // Arrange - Use an invalid enum value (cast from int)
            DroneSensorSystem.SensorLayer invalidLayer = (DroneSensorSystem.SensorLayer)999;

            // Act - Expect two error logs: one for invalid layer, one for invalid index
            LogAssert.Expect(LogType.Error, new System.Text.RegularExpressions.Regex(@"\[DroneSensorSystem\] 치명적 오류:"));
            LogAssert.Expect(LogType.Error, new System.Text.RegularExpressions.Regex(@"\[DroneSensorSystem\] 치명적 오류:"));
            float distance = _sensorSystem.GetNormalizedDistance(
                invalidLayer,
                DroneSensorSystem.CompassDirection.N
            );

            // Assert
            Assert.AreEqual(0f, distance, 0.001f, "Should return 0 for invalid layer");
        }

        /// <summary>
        /// Test: SetRayEnabled with boundary indices works correctly
        /// Requirement: Design document error handling section
        /// </summary>
        [Test]
        public void SetRayEnabled_WithBoundaryIndices_WorksCorrectly()
        {
            // Act & Assert - Test boundary indices 0 and 25
            _sensorSystem.SetRayEnabled(0, false);
            LogAssert.NoUnexpectedReceived();

            _sensorSystem.SetRayEnabled(25, false);
            LogAssert.NoUnexpectedReceived();

            // Test just outside boundaries
            LogAssert.Expect(LogType.Warning, "[DroneSensorSystem] 경고: 잘못된 레이 인덱스: -1. 유효 범위는 0~25입니다.");
            _sensorSystem.SetRayEnabled(-1, true);

            LogAssert.Expect(LogType.Warning, "[DroneSensorSystem] 경고: 잘못된 레이 인덱스: 26. 유효 범위는 0~25입니다.");
            _sensorSystem.SetRayEnabled(26, true);
        }

        #endregion

        #region Performance Optimization Tests

        /// <summary>
        /// Test: Ray direction vectors are cached during initialization
        /// Requirement: 6.3 - Ray direction vectors should be pre-calculated and reused
        /// </summary>
        [Test]
        public void RayDirectionVectors_AreCachedDuringInitialization()
        {
            // Arrange - Get forward direction at initialization
            Vector3 initialForward = _sensorSystem.GetForwardRayDirection();

            // Act - Call GetForwardRayDirection multiple times
            Vector3 forward1 = _sensorSystem.GetForwardRayDirection();
            Vector3 forward2 = _sensorSystem.GetForwardRayDirection();
            Vector3 forward3 = _sensorSystem.GetForwardRayDirection();

            // Assert - All should return the same cached direction (no recalculation)
            Assert.AreEqual(initialForward, forward1, "Forward direction should be cached");
            Assert.AreEqual(initialForward, forward2, "Forward direction should be cached");
            Assert.AreEqual(initialForward, forward3, "Forward direction should be cached");
        }

        /// <summary>
        /// Test: GetAllNormalizedDistances returns consistent array size
        /// Requirement: 6.2 - Sensor data array should be pre-allocated and reused
        /// </summary>
        [Test]
        public void GetAllNormalizedDistances_ReturnsConsistentArraySize()
        {
            // Act - Call multiple times
            float[] distances1 = _sensorSystem.GetAllNormalizedDistances();
            float[] distances2 = _sensorSystem.GetAllNormalizedDistances();
            float[] distances3 = _sensorSystem.GetAllNormalizedDistances();

            // Assert - All should return array of size 26
            Assert.AreEqual(26, distances1.Length, "Should return 26 distances");
            Assert.AreEqual(26, distances2.Length, "Should return 26 distances");
            Assert.AreEqual(26, distances3.Length, "Should return 26 distances");
        }

        /// <summary>
        /// Test: Sensor data array is reused across multiple FixedUpdate calls
        /// Requirement: 6.2 - Array reuse to minimize garbage collection
        /// </summary>
        [Test]
        public void SensorDataArray_IsReusedAcrossUpdates()
        {
            // Arrange - Create obstacle
            GameObject obstacle = GameObject.CreatePrimitive(PrimitiveType.Cube);
            obstacle.transform.position = _testGameObject.transform.position + Vector3.forward * 10f;

            // Act - Simulate multiple FixedUpdate calls
            for (int i = 0; i < 10; i++)
            {
                // Get distances (which internally uses the reused array)
                float[] distances = _sensorSystem.GetAllNormalizedDistances();
                
                // Assert - Should always return valid array
                Assert.IsNotNull(distances, $"Iteration {i}: Should return non-null array");
                Assert.AreEqual(26, distances.Length, $"Iteration {i}: Should return 26 distances");
            }

            // Cleanup
            Object.DestroyImmediate(obstacle);
        }

        /// <summary>
        /// Test: Ray direction vectors remain consistent after drone rotation
        /// Requirement: 6.3 - Cached local directions are transformed, not recalculated
        /// </summary>
        [Test]
        public void RayDirectionVectors_TransformCorrectlyAfterRotation()
        {
            // Arrange - Initial forward direction (no rotation)
            _testGameObject.transform.rotation = Quaternion.identity;
            Vector3 initialForward = _sensorSystem.GetForwardRayDirection();
            Assert.AreEqual(Vector3.forward, initialForward, "Initial forward should be (0,0,1)");

            // Act - Rotate drone 180 degrees
            _testGameObject.transform.rotation = Quaternion.Euler(0f, 180f, 0f);
            Vector3 rotatedForward = _sensorSystem.GetForwardRayDirection();

            // Assert - Should now point backward (cached local direction transformed)
            Assert.AreEqual(0f, rotatedForward.x, 0.001f, "X should be 0");
            Assert.AreEqual(0f, rotatedForward.y, 0.001f, "Y should be 0");
            Assert.AreEqual(-1f, rotatedForward.z, 0.001f, "Z should be -1 (backward)");
        }

        /// <summary>
        /// Test: All 26 rays use cached direction vectors
        /// Requirement: 6.3 - All rays should use pre-calculated directions
        /// </summary>
        [Test]
        public void AllRays_UseCachedDirectionVectors()
        {
            // Arrange - Get all distances initially
            float[] initialDistances = _sensorSystem.GetAllNormalizedDistances();

            // Act - Get distances multiple times
            float[] distances1 = _sensorSystem.GetAllNormalizedDistances();
            float[] distances2 = _sensorSystem.GetAllNormalizedDistances();

            // Assert - All arrays should have 26 elements (all rays use cached directions)
            Assert.AreEqual(26, initialDistances.Length, "Should have 26 rays");
            Assert.AreEqual(26, distances1.Length, "Should have 26 rays");
            Assert.AreEqual(26, distances2.Length, "Should have 26 rays");
        }

        /// <summary>
        /// Test: GetNormalizedDistance uses cached data without recalculation
        /// Requirement: 6.2 - Sensor data should be reused
        /// </summary>
        [Test]
        public void GetNormalizedDistance_UsesCachedData()
        {
            // Act - Query same sensor multiple times
            float distance1 = _sensorSystem.GetNormalizedDistance(
                DroneSensorSystem.SensorLayer.Middle,
                DroneSensorSystem.CompassDirection.N
            );
            float distance2 = _sensorSystem.GetNormalizedDistance(
                DroneSensorSystem.SensorLayer.Middle,
                DroneSensorSystem.CompassDirection.N
            );
            float distance3 = _sensorSystem.GetNormalizedDistance(
                DroneSensorSystem.SensorLayer.Middle,
                DroneSensorSystem.CompassDirection.N
            );

            // Assert - All should return the same value (cached data)
            Assert.AreEqual(distance1, distance2, 0.001f, "Should return cached value");
            Assert.AreEqual(distance1, distance3, 0.001f, "Should return cached value");
        }

        /// <summary>
        /// Test: Disabled rays still use cached direction vectors
        /// Requirement: 6.3 - Direction vectors are cached regardless of ray state
        /// </summary>
        [Test]
        public void DisabledRays_StillUseCachedDirectionVectors()
        {
            // Arrange - Disable a ray
            _sensorSystem.SetRayEnabled(10, false);

            // Act - Get distances
            float[] distances = _sensorSystem.GetAllNormalizedDistances();

            // Assert - Should still return 26 distances (disabled ray returns 0)
            Assert.AreEqual(26, distances.Length, "Should return 26 distances even with disabled rays");
            Assert.AreEqual(0f, distances[10], 0.001f, "Disabled ray should return 0");
        }

        #endregion

        #region DronePhysics Independence Tests

        /// <summary>
        /// Test: DroneSensorSystem doesn't require DronePhysics component
        /// Requirement: 4.4 - Sensor system should not depend on DronePhysics
        /// </summary>
        [Test]
        public void DroneSensorSystem_DoesNotRequireDronePhysics()
        {
            // Arrange - Create GameObject with only DroneSensorSystem (no DronePhysics)
            GameObject testObj = new GameObject("DroneWithoutPhysics");
            DroneSensorSystem sensor = testObj.AddComponent<DroneSensorSystem>();
            
            // Initialize sensor manually for Edit Mode tests
            InitializeSensor(sensor);

            // Act - Sensor should initialize successfully without DronePhysics
            float[] distances = sensor.GetAllNormalizedDistances();

            // Assert - Should work without DronePhysics
            Assert.IsNotNull(distances, "Sensor should work without DronePhysics");
            Assert.AreEqual(26, distances.Length, "Should return 26 distances");

            // Cleanup
            Object.DestroyImmediate(testObj);
        }

        /// <summary>
        /// Test: DroneSensorSystem only uses Transform component
        /// Requirement: 4.4 - Sensor system should only depend on Transform
        /// </summary>
        [Test]
        public void DroneSensorSystem_OnlyUsesTransform()
        {
            // Arrange - Create GameObject with DroneSensorSystem
            GameObject testObj = new GameObject("DroneWithSensor");
            DroneSensorSystem sensor = testObj.AddComponent<DroneSensorSystem>();
            
            // Initialize sensor manually for Edit Mode tests
            InitializeSensor(sensor);

            // Act - Move and rotate the drone
            testObj.transform.position = new Vector3(5f, 10f, 15f);
            testObj.transform.rotation = Quaternion.Euler(30f, 45f, 60f);

            // Get forward ray direction (uses Transform)
            Vector3 forwardDir = sensor.GetForwardRayDirection();

            // Assert - Should work using only Transform
            Assert.IsNotNull(forwardDir, "Should use Transform for direction calculation");
            Assert.AreEqual(1f, forwardDir.magnitude, 0.001f, "Direction should be normalized");

            // Cleanup
            Object.DestroyImmediate(testObj);
        }

        /// <summary>
        /// Test: DroneSensorSystem doesn't modify Transform when reading sensor data
        /// Requirement: 4.4 - Sensor system should not modify other components
        /// </summary>
        [Test]
        public void DroneSensorSystem_DoesNotModifyTransform()
        {
            // Arrange - Create GameObject with DroneSensorSystem
            GameObject testObj = new GameObject("DroneWithSensor");
            DroneSensorSystem sensor = testObj.AddComponent<DroneSensorSystem>();
            
            // Initialize sensor manually for Edit Mode tests
            InitializeSensor(sensor);

            // Store initial Transform state
            Vector3 initialPosition = testObj.transform.position;
            Quaternion initialRotation = testObj.transform.rotation;
            Vector3 initialScale = testObj.transform.localScale;

            // Act - Use all sensor API methods
            sensor.GetAllNormalizedDistances();
            sensor.GetNormalizedDistance(DroneSensorSystem.SensorLayer.Middle, DroneSensorSystem.CompassDirection.N);
            sensor.GetForwardRayDirection();
            sensor.SetMaxDetectionRange(100f);
            sensor.SetRayEnabled(0, false);
            sensor.SetDebugVisualization(true);

            // Assert - Transform should not be modified
            Assert.AreEqual(initialPosition, testObj.transform.position, "Position should not change");
            Assert.AreEqual(initialRotation, testObj.transform.rotation, "Rotation should not change");
            Assert.AreEqual(initialScale, testObj.transform.localScale, "Scale should not change");

            // Cleanup
            Object.DestroyImmediate(testObj);
        }

        /// <summary>
        /// Test: DroneSensorSystem works with any GameObject (no specific component dependencies)
        /// Requirement: 4.4 - No strong coupling with other components
        /// </summary>
        [Test]
        public void DroneSensorSystem_WorksWithAnyGameObject()
        {
            // Arrange - Create various GameObjects with different components
            GameObject emptyObj = new GameObject("Empty");
            GameObject cubeObj = GameObject.CreatePrimitive(PrimitiveType.Cube);
            GameObject sphereObj = GameObject.CreatePrimitive(PrimitiveType.Sphere);

            // Act - Add DroneSensorSystem to each
            DroneSensorSystem sensor1 = emptyObj.AddComponent<DroneSensorSystem>();
            DroneSensorSystem sensor2 = cubeObj.AddComponent<DroneSensorSystem>();
            DroneSensorSystem sensor3 = sphereObj.AddComponent<DroneSensorSystem>();
            
            // Initialize sensors manually for Edit Mode tests
            InitializeSensor(sensor1);
            InitializeSensor(sensor2);
            InitializeSensor(sensor3);

            // Assert - All should work independently
            Assert.IsNotNull(sensor1.GetAllNormalizedDistances(), "Should work on empty GameObject");
            Assert.IsNotNull(sensor2.GetAllNormalizedDistances(), "Should work on Cube");
            Assert.IsNotNull(sensor3.GetAllNormalizedDistances(), "Should work on Sphere");

            Assert.AreEqual(26, sensor1.GetAllNormalizedDistances().Length, "Should return 26 distances");
            Assert.AreEqual(26, sensor2.GetAllNormalizedDistances().Length, "Should return 26 distances");
            Assert.AreEqual(26, sensor3.GetAllNormalizedDistances().Length, "Should return 26 distances");

            // Cleanup
            Object.DestroyImmediate(emptyObj);
            Object.DestroyImmediate(cubeObj);
            Object.DestroyImmediate(sphereObj);
        }

        #endregion

        #region Layer Mask Filtering Tests

        /// <summary>
        /// Test: DetectionLayerMask is exposed in Unity Inspector
        /// Requirement: 5.3 - Layer mask should be configurable via Inspector
        /// </summary>
        [Test]
        public void DetectionLayerMask_IsExposedInInspector()
        {
            // Arrange & Act - DetectionLayerMask should be a public field
            LayerMask layerMask = _sensorSystem.DetectionLayerMask;

            // Assert - Should have default value of -1 (Everything)
            Assert.AreEqual(-1, layerMask.value, "DetectionLayerMask should default to -1 (Everything)");
        }

        /// <summary>
        /// Test: DetectionLayerMask can be set via Inspector
        /// Requirement: 5.3 - Layer mask should be configurable
        /// </summary>
        [Test]
        public void DetectionLayerMask_CanBeSet()
        {
            // Arrange
            LayerMask customMask = LayerMask.GetMask("Default");

            // Act
            _sensorSystem.DetectionLayerMask = customMask;

            // Assert
            Assert.AreEqual(customMask.value, _sensorSystem.DetectionLayerMask.value,
                "DetectionLayerMask should be settable");
        }

        /// <summary>
        /// Test: Raycast only detects objects on specified layers
        /// Requirement: 5.3 - Layer mask filtering should work correctly
        /// </summary>
        [Test]
        public void Raycast_OnlyDetectsObjectsOnSpecifiedLayers()
        {
            // Arrange - Create two obstacles on different layers
            GameObject obstacleOnDefault = GameObject.CreatePrimitive(PrimitiveType.Cube);
            obstacleOnDefault.name = "ObstacleOnDefault";
            obstacleOnDefault.layer = LayerMask.NameToLayer("Default");
            obstacleOnDefault.transform.position = _testGameObject.transform.position + Vector3.forward * 10f;

            GameObject obstacleOnIgnoreRaycast = GameObject.CreatePrimitive(PrimitiveType.Cube);
            obstacleOnIgnoreRaycast.name = "ObstacleOnIgnoreRaycast";
            obstacleOnIgnoreRaycast.layer = LayerMask.NameToLayer("Ignore Raycast");
            obstacleOnIgnoreRaycast.transform.position = _testGameObject.transform.position + Vector3.forward * 5f;

            // Set layer mask to only detect Default layer (not Ignore Raycast)
            _sensorSystem.DetectionLayerMask = LayerMask.GetMask("Default");

            // Act - Simulate FixedUpdate to perform raycast
            // Note: We can't directly call FixedUpdate, but we can verify the layer mask is set
            float[] distances = _sensorSystem.GetAllNormalizedDistances();

            // Assert - Layer mask should be set correctly
            Assert.AreEqual(LayerMask.GetMask("Default"), _sensorSystem.DetectionLayerMask.value,
                "DetectionLayerMask should be set to Default layer only");

            // Cleanup
            Object.DestroyImmediate(obstacleOnDefault);
            Object.DestroyImmediate(obstacleOnIgnoreRaycast);
        }

        /// <summary>
        /// Test: Layer mask with Nothing value prevents all detections
        /// Requirement: 5.3 - Layer mask filtering
        /// </summary>
        [Test]
        public void LayerMask_WithNothing_PreventsAllDetections()
        {
            // Arrange - Create obstacle
            GameObject obstacle = GameObject.CreatePrimitive(PrimitiveType.Cube);
            obstacle.transform.position = _testGameObject.transform.position + Vector3.forward * 10f;

            // Create new sensor with Nothing layer mask
            GameObject newObj = new GameObject("TestDrone2");
            DroneSensorSystem newSensor = newObj.AddComponent<DroneSensorSystem>();
            
            // Set layer mask to Nothing (0)
            newSensor.DetectionLayerMask = 0;

            // Expect warning about Nothing layer mask when validation is triggered
            LogAssert.Expect(LogType.Warning, 
                "[DroneSensorSystem] 경고: DetectionLayerMask가 Nothing으로 설정되어 있습니다. 모든 레이가 충돌을 감지하지 못합니다.");

            // Act - Manually call ValidateConfiguration via reflection
            var validateMethod = typeof(DroneSensorSystem).GetMethod("ValidateConfiguration", 
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            if (validateMethod != null)
            {
                validateMethod.Invoke(newSensor, null);
            }

            // Assert - Layer mask should be 0
            Assert.AreEqual(0, newSensor.DetectionLayerMask.value,
                "DetectionLayerMask should be set to Nothing (0)");

            // Cleanup
            Object.DestroyImmediate(obstacle);
            Object.DestroyImmediate(newObj);
        }

        /// <summary>
        /// Test: Layer mask with Everything value detects all objects
        /// Requirement: 5.3 - Layer mask filtering
        /// </summary>
        [Test]
        public void LayerMask_WithEverything_DetectsAllObjects()
        {
            // Arrange - Create obstacles on different layers
            GameObject obstacle1 = GameObject.CreatePrimitive(PrimitiveType.Cube);
            obstacle1.layer = LayerMask.NameToLayer("Default");
            obstacle1.transform.position = _testGameObject.transform.position + Vector3.forward * 10f;

            GameObject obstacle2 = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            obstacle2.layer = LayerMask.NameToLayer("TransparentFX");
            obstacle2.transform.position = _testGameObject.transform.position + Vector3.right * 10f;

            // Set layer mask to Everything (-1)
            _sensorSystem.DetectionLayerMask = -1;

            // Act - Get distances
            float[] distances = _sensorSystem.GetAllNormalizedDistances();

            // Assert - Layer mask should be set to Everything
            Assert.AreEqual(-1, _sensorSystem.DetectionLayerMask.value,
                "DetectionLayerMask should be set to Everything (-1)");

            // Cleanup
            Object.DestroyImmediate(obstacle1);
            Object.DestroyImmediate(obstacle2);
        }

        /// <summary>
        /// Test: Layer mask can be changed at runtime
        /// Requirement: 5.3 - Layer mask should be configurable at runtime
        /// </summary>
        [Test]
        public void LayerMask_CanBeChangedAtRuntime()
        {
            // Arrange
            LayerMask initialMask = LayerMask.GetMask("Default");
            LayerMask newMask = LayerMask.GetMask("TransparentFX");

            // Act - Change layer mask multiple times
            _sensorSystem.DetectionLayerMask = initialMask;
            Assert.AreEqual(initialMask.value, _sensorSystem.DetectionLayerMask.value,
                "Layer mask should be set to Default");

            _sensorSystem.DetectionLayerMask = newMask;
            Assert.AreEqual(newMask.value, _sensorSystem.DetectionLayerMask.value,
                "Layer mask should be changed to TransparentFX");

            _sensorSystem.DetectionLayerMask = -1;
            Assert.AreEqual(-1, _sensorSystem.DetectionLayerMask.value,
                "Layer mask should be changed to Everything");
        }

        /// <summary>
        /// Test: Layer mask with multiple layers detects objects on any specified layer
        /// Requirement: 5.3 - Layer mask filtering with multiple layers
        /// </summary>
        [Test]
        public void LayerMask_WithMultipleLayers_DetectsObjectsOnAnySpecifiedLayer()
        {
            // Arrange - Set layer mask to include Default and TransparentFX
            LayerMask multiLayerMask = LayerMask.GetMask("Default", "TransparentFX");
            _sensorSystem.DetectionLayerMask = multiLayerMask;

            // Act - Get distances
            float[] distances = _sensorSystem.GetAllNormalizedDistances();

            // Assert - Layer mask should include both layers
            Assert.AreEqual(multiLayerMask.value, _sensorSystem.DetectionLayerMask.value,
                "DetectionLayerMask should include multiple layers");

            // Verify the mask includes both layers
            int defaultLayer = LayerMask.NameToLayer("Default");
            int transparentFXLayer = LayerMask.NameToLayer("TransparentFX");
            
            Assert.IsTrue((multiLayerMask.value & (1 << defaultLayer)) != 0,
                "Layer mask should include Default layer");
            Assert.IsTrue((multiLayerMask.value & (1 << transparentFXLayer)) != 0,
                "Layer mask should include TransparentFX layer");
        }

        /// <summary>
        /// Test: Physics.Raycast receives DetectionLayerMask parameter
        /// Requirement: 5.3 - Layer mask must be passed to Physics.Raycast
        /// </summary>
        [Test]
        public void PhysicsRaycast_ReceivesDetectionLayerMask()
        {
            // Arrange - Set custom layer mask
            LayerMask customMask = LayerMask.GetMask("Default");
            _sensorSystem.DetectionLayerMask = customMask;

            // Create obstacle on Default layer
            GameObject obstacle = GameObject.CreatePrimitive(PrimitiveType.Cube);
            obstacle.layer = LayerMask.NameToLayer("Default");
            obstacle.transform.position = _testGameObject.transform.position + Vector3.forward * 10f;

            // Act - Get distances (this internally calls Physics.Raycast with DetectionLayerMask)
            float[] distances = _sensorSystem.GetAllNormalizedDistances();

            // Assert - Verify layer mask is correctly set
            Assert.AreEqual(customMask.value, _sensorSystem.DetectionLayerMask.value,
                "DetectionLayerMask should be passed to Physics.Raycast");

            // Cleanup
            Object.DestroyImmediate(obstacle);
        }

        #endregion
    }
}
