using System;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using UnityEngine;
using DroneCamera;
using DroneVisualPipeline;

/// <summary>
/// PursuerAgent — 카메라 유도 기반 추적 드론 에이전트.
///
/// 핵심 원칙:
///   - Evader는 Goal을 향해 비행한다.
///   - Pursuer는 드론 카메라 기준 관측으로 Evader를 화면 안에 유지하며 추적한다.
///   - 복수 Pursuer는 하나의 Evader를 공유하며, capture/goal/timeout 시 함께 에피소드를 종료한다.
/// </summary>
public class PursuerAgent : DroneAgent
{
    private const int SelfVectorObservationSize = 7;
    private const int TargetTrackingObservationSize = 11;
    private const int GroundTruthHintObservationSize = 4;
    private const int RayObservationSize = 26;

    private enum RoundEndReason
    {
        Captured,
        Timeout,
    }

    private static event Action<RoundEndReason, PursuerAgent> RoundEnded;
    private static bool s_isBroadcastingRoundEnd;

    [Header("Pursuer — Goal Zone")]
    [SerializeField] private Goal _goalZone;

    [Header("Pursuer — Episode Settings")]
    [SerializeField] private float _maxEpisodeSeconds = 25f;
    [SerializeField] private float _catchDistance = 2.0f;
    [SerializeField] private int _catchConfirmSteps = 3;

    [Header("Pursuer — Camera Tracking")]
    [SerializeField] private string _targetTag = "Evader";
    [SerializeField] private float _viewportVisibleMargin = 0.05f;
    [SerializeField] private float _lostTargetMemorySeconds = 0.75f;
    [SerializeField] private float _spawnFacingJitterDeg = 4f;

    [Header("Pursuer — Color Target Tracking")]
    [SerializeField] private bool _useColorTargetTracking = true;
    [SerializeField] private Color _targetColor = new Color(1f, 0.06f, 0f, 1f);
    [SerializeField] private float _colorMatchTolerance = 0.32f;
    [SerializeField] private float _minColorSaturation = 0.25f;
    [SerializeField] private float _minColorBrightness = 0.08f;
    [SerializeField] private int _colorSampleStride = 1;
    [SerializeField] private int _minMatchedColorSamples = 1;
    [SerializeField] private float _minMatchedPixelFraction = 0.0001f;
    [SerializeField] private float _colorReferencePixelFraction = 0.02f;
    [SerializeField] private float _colorReferenceDistance = 8f;
    [SerializeField] private float _colorMinEstimatedDistance = 1.5f;

    [Header("Pursuer — Training Ground Truth Hint")]
    [SerializeField] private bool _useTrainingGroundTruthHint = true;
    [SerializeField] private float _groundTruthHintInitialProbability = 0.25f;
    [SerializeField] private float _groundTruthHintFinalProbability = 0f;
    [SerializeField] private float _groundTruthHintDecaySteps = 1000000f;
    [SerializeField] private bool _groundTruthHintOnlyWhenTargetHidden = true;

    [Header("Pursuer — Lucas-Kanade Optical Flow")]
    [SerializeField] private bool _useLucasKanadeFlow = true;
    [SerializeField] private int _lkWindowRadius = 2;
    [SerializeField] private int _lkFeatureStride = 2;
    [SerializeField] private int _lkMaxFeatures = 32;
    [SerializeField] private float _lkMinGradient = 0.015f;
    [SerializeField] private float _lkMinDeterminant = 1e-5f;
    [SerializeField] private float _lkMaxPixelDisplacement = 8f;
    [SerializeField] private int _lkMinTrackedFeatures = 2;

    [Header("Pursuer — Spawn Assist")]
    [SerializeField] private bool _spawnNearTargetAtEpisodeStart = true;
    [SerializeField] private float _nearTargetSpawnMinDistance = 3f;
    [SerializeField] private float _nearTargetSpawnMaxDistance = 6f;
    [SerializeField] private float _nearTargetSpawnVerticalJitter = 0.5f;
    [SerializeField] private float _nearTargetSpawnClearanceRadius = 1.25f;
    [SerializeField] private int _nearTargetSpawnAttempts = 36;
    [SerializeField] private int _spawnSyncFixedSteps = 2;
    [SerializeField] private LayerMask _visibilityMask = ~0;

    [Header("Pursuer — Observation Normalization")]
    [SerializeField] private float _maxDistance = 50f;
    [SerializeField] private float _maxObsSpeed = 10f;
    [SerializeField] private float _maxViewportSpeed = 4f;

    [Header("Pursuer — Observation Sources")]
    [SerializeField] private bool _includeTargetTrackingObservations = true;
    [SerializeField] private bool _includeRayObservations = true;

    [Header("Pursuer — Diagnostics")]
    [SerializeField] private bool _reportPerceptionDiagnostics = true;
    [SerializeField] private int _perceptionDiagnosticsIntervalSteps = 10;

    private PursuerReward _rewardCalculator;
    private EpisodeLogger _episodeLogger;
    private DroneCameraSystem _cameraSystem;
    private DroneVisionSystem _visionSystem;
    private Camera _trackingCamera;

    private float _episodeTimer;
    private int _episodeSteps;
    private bool _episodeClosed;
    private Vector3 _spawnOrigin;

    private bool _isTargetVisible;
    private bool _hasPerceptionSample;
    private bool _hasLastKnownTarget;
    private float _timeSinceTargetSeen;
    private Vector3 _lastKnownTargetWorldPos;
    private Vector3 _lastKnownTargetCameraLocalPos;
    private Vector3 _lastTargetLocalPos;
    private Vector3 _targetLocalVelocity;
    private Vector2 _viewportOffset;
    private Vector2 _viewportVelocity;
    private float _targetColorBlobFraction;
    private Texture2D _colorReadbackTexture;
    private float[] _previousLumaFrame;
    private float[] _currentLumaFrame;
    private int _lumaFrameWidth;
    private int _lumaFrameHeight;
    private bool _hasPreviousLumaFrame;
    private bool _previousLumaFrameHadTarget;
    private float _lastLucasKanadeConfidence;
    private bool _lastGroundTruthHintActive;
    private float _lastGroundTruthHintProbability;
    private bool _pendingEpisodeEnd;
    private int _catchContactSteps;
    private int _perceptionDiagnosticsCounter;
    private int _spawnSyncStepsRemaining;
    private bool _awaitingTargetSpawnPlacement;
    private Vector3 _pendingBaseSpawnPosition;

    public override void Initialize()
    {
        Role = global::DroneRole.Pursuer;

        _rewardCalculator = GetComponent<PursuerReward>();
        if (_rewardCalculator == null)
            _rewardCalculator = gameObject.AddComponent<PursuerReward>();

        _episodeLogger = GetComponent<EpisodeLogger>();
        _cameraSystem = GetComponent<DroneCameraSystem>();
        _visionSystem = GetComponent<DroneVisionSystem>();
        _spawnOrigin = transform.position;

        ResolveGoalZone();
        ResolveTargetTransform();
        ResolveTrackingCamera();
        SyncBehaviorParameters();
    }

    protected override void OnEnable()
    {
        base.OnEnable();
        RoundEnded += HandleRoundEnded;
    }

    protected override void OnDisable()
    {
        RoundEnded -= HandleRoundEnded;

        if (_goalZone != null)
            _goalZone.OnArrival -= OnGoalArrived;

        ReleaseColorReadbackTexture();

        base.OnDisable();
    }

    private void OnValidate()
    {
        _maxEpisodeSeconds = Mathf.Max(1f, _maxEpisodeSeconds);
        _catchDistance = Mathf.Max(0.1f, _catchDistance);
        _catchConfirmSteps = Mathf.Max(1, _catchConfirmSteps);
        _viewportVisibleMargin = Mathf.Clamp(_viewportVisibleMargin, 0f, 0.5f);
        _lostTargetMemorySeconds = Mathf.Max(0f, _lostTargetMemorySeconds);
        _spawnFacingJitterDeg = Mathf.Clamp(_spawnFacingJitterDeg, 0f, 180f);
        _colorMatchTolerance = Mathf.Max(0.001f, _colorMatchTolerance);
        _minColorSaturation = Mathf.Clamp01(_minColorSaturation);
        _minColorBrightness = Mathf.Clamp01(_minColorBrightness);
        _colorSampleStride = Mathf.Max(1, _colorSampleStride);
        _minMatchedColorSamples = Mathf.Max(1, _minMatchedColorSamples);
        _minMatchedPixelFraction = Mathf.Clamp01(_minMatchedPixelFraction);
        _colorReferencePixelFraction = Mathf.Clamp(_colorReferencePixelFraction, 0.0001f, 1f);
        _colorReferenceDistance = Mathf.Max(0.1f, _colorReferenceDistance);
        _colorMinEstimatedDistance = Mathf.Max(0.1f, _colorMinEstimatedDistance);
        _groundTruthHintInitialProbability = Mathf.Clamp01(_groundTruthHintInitialProbability);
        _groundTruthHintFinalProbability = Mathf.Clamp01(_groundTruthHintFinalProbability);
        _groundTruthHintDecaySteps = Mathf.Max(0f, _groundTruthHintDecaySteps);
        _lkWindowRadius = Mathf.Max(1, _lkWindowRadius);
        _lkFeatureStride = Mathf.Max(1, _lkFeatureStride);
        _lkMaxFeatures = Mathf.Max(1, _lkMaxFeatures);
        _lkMinGradient = Mathf.Max(0f, _lkMinGradient);
        _lkMinDeterminant = Mathf.Max(1e-8f, _lkMinDeterminant);
        _lkMaxPixelDisplacement = Mathf.Max(0.1f, _lkMaxPixelDisplacement);
        _lkMinTrackedFeatures = Mathf.Max(1, _lkMinTrackedFeatures);
        _nearTargetSpawnMinDistance = Mathf.Max(0.5f, _nearTargetSpawnMinDistance);
        _nearTargetSpawnMaxDistance = Mathf.Max(_nearTargetSpawnMinDistance, _nearTargetSpawnMaxDistance);
        _nearTargetSpawnVerticalJitter = Mathf.Max(0f, _nearTargetSpawnVerticalJitter);
        _nearTargetSpawnClearanceRadius = Mathf.Max(0.1f, _nearTargetSpawnClearanceRadius);
        _nearTargetSpawnAttempts = Mathf.Max(1, _nearTargetSpawnAttempts);
        _spawnSyncFixedSteps = Mathf.Max(0, _spawnSyncFixedSteps);
        _maxDistance = Mathf.Max(1f, _maxDistance);
        _maxObsSpeed = Mathf.Max(0.1f, _maxObsSpeed);
        _maxViewportSpeed = Mathf.Max(0.1f, _maxViewportSpeed);
        _perceptionDiagnosticsIntervalSteps = Mathf.Max(1, _perceptionDiagnosticsIntervalSteps);
        SyncBehaviorParameters();
    }

    public override void OnEpisodeBegin()
    {
        EnsureRuntimeReferences();

        // _episodeClosed가 false인 채로 호출되면 ML-Agents MaxStep에 의한 강제 종료
        if (!_episodeClosed)
            Debug.Log($"[Pursuer 재스폰] ML-Agents MaxStep 초과로 강제 종료 | name={name} | step={_episodeSteps}");

        _episodeTimer = 0f;
        _episodeSteps = 0;
        _episodeClosed = false;
        _pendingEpisodeEnd = false;
        _catchContactSteps = 0;
        _perceptionDiagnosticsCounter = 0;
        _spawnSyncStepsRemaining = 0;
        _awaitingTargetSpawnPlacement = false;
        ResetPerceptionState();
        _rewardCalculator?.ResetEpisodeState();

        ResolveGoalZone();
        ResolveTargetTransform();
        ResolveTrackingCamera();
        PrepareEpisodeSpawn();
    }

    private void FixedUpdate()
    {
        if (_pendingEpisodeEnd)
        {
            _pendingEpisodeEnd = false;
            EndEpisode();
            return;
        }

        TryProcessDeferredSpawnSync();
        UpdateTargetPerception(Time.fixedDeltaTime);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        if (!IsDroneReady())
        {
            for (int i = 0; i < GetExpectedVectorObservationSize(); i++)
                sensor.AddObservation(0f);
            return;
        }

        Vector3 localVel = transform.InverseTransformDirection(_dronePhysics.GetVelocity());
        Vector3 localAngVel = _dronePhysics.GetLocalAngularVelocity();
        sensor.AddObservation(localVel / _maxObsSpeed);                // 3
        sensor.AddObservation(localAngVel / _maxObsSpeed);            // 3
        sensor.AddObservation(transform.position.y / _maxDistance);   // 1

        bool hasTrackedTarget = false;
        if (_includeTargetTrackingObservations)
        {
            hasTrackedTarget = TryGetTrackedTargetLocalPosition(out Vector3 targetLocalPos);
            Vector3 targetDir = targetLocalPos.sqrMagnitude > 1e-6f
                ? targetLocalPos.normalized
                : Vector3.zero;

            Vector3 observedTargetVelocity = _isTargetVisible ? _targetLocalVelocity : Vector3.zero;
            Vector2 observedViewportOffset = hasTrackedTarget ? _viewportOffset : Vector2.zero;
            float observedViewportSpeed = _isTargetVisible
                ? Mathf.Clamp01(_viewportVelocity.magnitude / _maxViewportSpeed)
                : 0f;

            sensor.AddObservation(targetDir);                             // 3
            sensor.AddObservation(targetLocalPos.magnitude / _maxDistance); // 1
            sensor.AddObservation(observedTargetVelocity / _maxObsSpeed); // 3
            sensor.AddObservation(observedViewportOffset);                // 2
            sensor.AddObservation(_isTargetVisible ? 1f : 0f);           // 1
            sensor.AddObservation(observedViewportSpeed);                 // 1
        }

        AddGroundTruthHintObservation(sensor);

        if (_includeRayObservations && _sensorSystem != null)
        {
            foreach (float d in _sensorSystem.GetAllNormalizedDistances())
                sensor.AddObservation(d);                             // 26
        }
        else if (_includeRayObservations)
        {
            for (int i = 0; i < 26; i++)
                sensor.AddObservation(0f);
        }

        if (!hasTrackedTarget)
        {
            // 관측 차원은 이미 채웠지만, target 관련 상태는 다음 스텝에서 0 유지.
            _targetLocalVelocity = Vector3.zero;
        }
    }

    private void AddGroundTruthHintObservation(VectorSensor sensor)
    {
        _lastGroundTruthHintActive = false;
        _lastGroundTruthHintProbability = GetCurrentGroundTruthHintProbability();

        Vector3 normalizedHintLocalPos = Vector3.zero;
        if (ShouldExposeGroundTruthHint(_lastGroundTruthHintProbability))
        {
            Transform referenceFrame = _trackingCamera != null ? _trackingCamera.transform : transform;
            Vector3 hintLocalPos = referenceFrame.InverseTransformPoint(TargetTransform.position) / _maxDistance;
            normalizedHintLocalPos = new Vector3(
                Mathf.Clamp(hintLocalPos.x, -1f, 1f),
                Mathf.Clamp(hintLocalPos.y, -1f, 1f),
                Mathf.Clamp(hintLocalPos.z, -1f, 1f));
            _lastGroundTruthHintActive = true;
        }

        sensor.AddObservation(_lastGroundTruthHintActive ? 1f : 0f); // 1
        sensor.AddObservation(normalizedHintLocalPos);                // 3
    }

    private bool ShouldExposeGroundTruthHint(float probability)
    {
        if (!_useTrainingGroundTruthHint || probability <= 0f || TargetTransform == null)
            return false;

        if (!IsTrainerControlledBehavior())
            return false;

        if (_groundTruthHintOnlyWhenTargetHidden && _isTargetVisible)
            return false;

        return UnityEngine.Random.value < probability;
    }

    private float GetCurrentGroundTruthHintProbability()
    {
        if (!_useTrainingGroundTruthHint)
            return 0f;

        if (_groundTruthHintDecaySteps <= 0f)
            return _groundTruthHintFinalProbability;

        int environmentStep = Academy.Instance != null
            ? Academy.Instance.TotalStepCount
            : 0;
        float progress = Mathf.Clamp01(environmentStep / _groundTruthHintDecaySteps);
        return Mathf.Lerp(
            _groundTruthHintInitialProbability,
            _groundTruthHintFinalProbability,
            progress);
    }

    private bool IsTrainerControlledBehavior()
    {
        BehaviorParameters behaviorParameters = GetComponent<BehaviorParameters>();
        return behaviorParameters != null &&
               behaviorParameters.BehaviorType == BehaviorType.Default;
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        if (!IsDroneReady() || _episodeClosed || _awaitingTargetSpawnPlacement)
            return;

        _episodeTimer += Time.fixedDeltaTime;
        _episodeSteps++;

        float thrust = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
        float roll = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);
        float pitch = Mathf.Clamp(actions.ContinuousActions[2], -1f, 1f);
        float yaw = Mathf.Clamp(actions.ContinuousActions[3], -1f, 1f);

        _dronePhysics.SetCommand(thrust, roll, pitch, yaw);

        bool hasTarget = TargetTransform != null;
        Vector3 targetPos = hasTarget ? TargetTransform.position : Vector3.zero;
        Rigidbody targetRb = hasTarget ? TargetTransform.GetComponent<Rigidbody>() : null;
        Vector3 targetVel = targetRb != null ? targetRb.linearVelocity : Vector3.zero;

        AddReward(_rewardCalculator.ComputeStepReward(
            hasTarget: hasTarget,
            agentPos: transform.position,
            targetPos: targetPos,
            agentVel: _dronePhysics.GetVelocity(),
            targetVel: targetVel,
            isTargetVisible: _isTargetVisible,
            viewportOffset: _viewportOffset,
            visualTargetArea: _targetColorBlobFraction));

        ReportPerceptionDiagnostics();
        CheckTerminationConditions();
    }

    public void SetCrash()
    {
        if (_episodeClosed)
            return;

        _episodeClosed = true;
        AddReward(_rewardCalculator.CrashPenalty);
        _episodeLogger?.LogEpisode(EpisodeLogger.TermType.Crash, _episodeSteps);
        Debug.Log($"[Pursuer 재스폰] 충돌 | name={name} | step={_episodeSteps}");
        _pendingEpisodeEnd = true;
    }

    private void CheckTerminationConditions()
    {
        if (TargetTransform != null &&
            Vector3.Distance(transform.position, TargetTransform.position) < _catchDistance)
        {
            _catchContactSteps++;
            if (_catchContactSteps >= _catchConfirmSteps)
            {
                NotifyEvaderCaptured();
                BroadcastRoundEnd(RoundEndReason.Captured);
            }
            return;
        }

        _catchContactSteps = 0;

        if (_episodeTimer >= _maxEpisodeSeconds)
            BroadcastRoundEnd(RoundEndReason.Timeout);
    }

    private void HandleRoundEnded(RoundEndReason reason, PursuerAgent source)
    {
        if (_episodeClosed)
            return;

        EnsureRuntimeReferences();
        _episodeClosed = true;

        switch (reason)
        {
            case RoundEndReason.Captured:
                AddReward(_rewardCalculator.CaptureReward);
                _episodeLogger?.LogEpisode(EpisodeLogger.TermType.Captured, _episodeSteps);
                Debug.Log($"[Pursuer 재스폰] 포획 성공 (source={source?.name}) | name={name} | step={_episodeSteps}");
                break;

            case RoundEndReason.Timeout:
                AddReward(_rewardCalculator.TimeoutPenalty);
                _episodeLogger?.LogEpisode(EpisodeLogger.TermType.Timeout, _episodeSteps);
                Debug.Log($"[Pursuer 재스폰] 타임아웃 | name={name} | step={_episodeSteps}");
                NotifyEvaderEndEpisode(); // Pursuer 타임아웃 시 Evader 에피소드도 함께 종료
                break;
        }

        _pendingEpisodeEnd = true;
    }

    private void BroadcastRoundEnd(RoundEndReason reason)
    {
        if (s_isBroadcastingRoundEnd)
            return;

        s_isBroadcastingRoundEnd = true;
        try
        {
            RoundEnded?.Invoke(reason, this);
        }
        finally
        {
            s_isBroadcastingRoundEnd = false;
        }
    }

    private void NotifyEvaderCaptured()
    {
        if (TargetTransform == null)
            return;

        var evaderAgent = TargetTransform.GetComponent<EvaderAgent>();
        if (evaderAgent != null)
        {
            evaderAgent.SetCaptured();
            return;
        }

        var genericAgent = TargetTransform.GetComponent<Agent>();
        genericAgent?.EndEpisode();
    }

    /// <summary>
    /// Pursuer 타임아웃/충돌 시 Evader 에피소드도 함께 종료한다.
    /// EvaderAgent.TerminateEpisode()는 private이므로 Agent.EndEpisode() 직접 호출.
    /// ML-Agents EndEpisode()는 내부적으로 멱등 처리되므로 복수 Pursuer 환경에서 중복 호출 안전.
    /// </summary>
    private void NotifyEvaderEndEpisode()
    {
        if (TargetTransform == null)
            return;

        var agent = TargetTransform.GetComponent<Agent>();
        agent?.EndEpisode();
    }

    /// <summary>
    /// Evader 충돌 시 외부(EvaderAgent)에서 호출. 보상·로그 없이 에피소드만 안전하게 종료한다.
    /// _pendingEpisodeEnd 패턴을 사용하여 Physics 콜백 중 EndEpisode() 직접 호출을 피한다.
    /// </summary>
    public void ForceEndEpisode()
    {
        if (_episodeClosed)
            return;

        _episodeClosed    = true;
        _pendingEpisodeEnd = true;
        Debug.Log($"[Pursuer 재스폰] Evader 충돌로 강제 종료 | name={name} | step={_episodeSteps}");
    }

    private void ResolveGoalZone()
    {
        Goal nextGoal = _goalZone != null ? _goalZone : Goal.Current;
        if (_goalZone == nextGoal)
            return;

        if (_goalZone != null)
            _goalZone.OnArrival -= OnGoalArrived;

        _goalZone = nextGoal;

        if (_goalZone != null)
            _goalZone.OnArrival += OnGoalArrived;
    }

    private void OnGoalArrived(Goal zone) { }

    public void HandleEvaderGoalRespawn()
    {
        EnsureRuntimeReferences();

        AddReward(_rewardCalculator != null ? _rewardCalculator.EvaderGoalPenalty : -1f);
        _episodeLogger?.LogEpisode(EpisodeLogger.TermType.Goal, _episodeSteps);

        _episodeTimer = 0f;
        _episodeSteps = 0;
        _episodeClosed = false;
        _pendingEpisodeEnd = false;
        _catchContactSteps = 0;
        _perceptionDiagnosticsCounter = 0;
        _spawnSyncStepsRemaining = 0;
        _awaitingTargetSpawnPlacement = false;
        ResetPerceptionState();
        _rewardCalculator?.ResetEpisodeState();

        ResolveGoalZone();
        ResolveTargetTransform();
        ResolveTrackingCamera();
        PrepareEpisodeSpawn();
    }

    private void EnsureRuntimeReferences()
    {
        if (_rewardCalculator == null)
            _rewardCalculator = GetComponent<PursuerReward>();

        if (_episodeLogger == null)
            _episodeLogger = GetComponent<EpisodeLogger>();
    }

    private void PrepareEpisodeSpawn()
    {
        if (EpisodeSpawnCoordinator.Instance != null && EpisodeSpawnCoordinator.Instance.IsComputed)
        {
            _pendingBaseSpawnPosition = EpisodeSpawnCoordinator.Instance.GetSpawnPosition(gameObject);
        }
        else if (SpawnCenter.Current != null)
        {
            var range = SpawnCenter.Current.GetPursuerSpawnRange();
            _pendingBaseSpawnPosition = SpawnCenter.Current.GetRandomPosition(range);
        }
        else
        {
            _pendingBaseSpawnPosition = _spawnOrigin;
        }

        ResetPhysicsState();
        _awaitingTargetSpawnPlacement = true;
        _spawnSyncStepsRemaining = Mathf.Max(1, _spawnSyncFixedSteps);
    }

    private int GetExpectedVectorObservationSize()
    {
        int observationSize = SelfVectorObservationSize;
        if (_includeTargetTrackingObservations)
            observationSize += TargetTrackingObservationSize;
        observationSize += GroundTruthHintObservationSize;
        if (_includeRayObservations)
            observationSize += RayObservationSize;
        return observationSize;
    }

    private void SyncBehaviorParameters()
    {
        BehaviorParameters behaviorParameters = GetComponent<BehaviorParameters>();
        if (behaviorParameters == null)
            return;

        behaviorParameters.BrainParameters.VectorObservationSize = GetExpectedVectorObservationSize();
    }

    private void ResolveTargetTransform()
    {
        if (TargetTransform != null && TargetTransform.gameObject.activeInHierarchy)
            return;

        GameObject targetObject = GameObject.FindGameObjectWithTag(_targetTag);
        if (targetObject != null && targetObject.transform != transform)
            TargetTransform = targetObject.transform;
    }

    private void ResolveTrackingCamera()
    {
        if (_cameraSystem == null)
            _cameraSystem = GetComponent<DroneCameraSystem>();

        if (_visionSystem == null)
            _visionSystem = GetComponent<DroneVisionSystem>();

        if (_visionSystem != null && _visionSystem.IsInitialized && _visionSystem.ObservationCamera != null)
            _trackingCamera = _visionSystem.ObservationCamera;
        else if (_cameraSystem != null)
            _trackingCamera = _cameraSystem.Camera;
        else
            _trackingCamera = null;
    }

    private void UpdateTargetPerception(float deltaTime)
    {
        ResolveTargetTransform();
        ResolveTrackingCamera();

        if (_useColorTargetTracking)
        {
            TryUpdateColorTargetPerception(deltaTime);
            return;
        }

        if (TargetTransform == null || _trackingCamera == null)
        {
            MarkTargetNotVisible(deltaTime);
            return;
        }

        Vector3 worldTargetPos = TargetTransform.position;
        Vector3 targetLocalPos = _trackingCamera.transform.InverseTransformPoint(worldTargetPos);
        Vector3 viewportPoint = _trackingCamera.WorldToViewportPoint(worldTargetPos);
        Vector2 nextViewportOffset = new Vector2(
            (viewportPoint.x - 0.5f) * 2f,
            (viewportPoint.y - 0.5f) * 2f);

        bool isInViewport =
            viewportPoint.z > 0f &&
            viewportPoint.x >= -_viewportVisibleMargin &&
            viewportPoint.x <= 1f + _viewportVisibleMargin &&
            viewportPoint.y >= -_viewportVisibleMargin &&
            viewportPoint.y <= 1f + _viewportVisibleMargin;

        bool hasLineOfSight = HasLineOfSightToTarget(worldTargetPos);
        _isTargetVisible = isInViewport && hasLineOfSight;

        if (_isTargetVisible)
        {
            if (_hasPerceptionSample)
            {
                float invDelta = 1f / Mathf.Max(deltaTime, 1e-5f);
                _targetLocalVelocity = (targetLocalPos - _lastTargetLocalPos) * invDelta;
                _viewportVelocity = (nextViewportOffset - _viewportOffset) * invDelta;
            }
            else
            {
                _targetLocalVelocity = Vector3.zero;
                _viewportVelocity = Vector2.zero;
            }

            _lastTargetLocalPos = targetLocalPos;
            _viewportOffset = nextViewportOffset;
            _hasPerceptionSample = true;
            _lastKnownTargetWorldPos = worldTargetPos;
            _lastKnownTargetCameraLocalPos = targetLocalPos;
            _hasLastKnownTarget = true;
            _timeSinceTargetSeen = 0f;
        }
        else
        {
            MarkTargetNotVisible(deltaTime);
        }
    }

    private bool TryUpdateColorTargetPerception(float deltaTime)
    {
        if (_visionSystem == null || !_visionSystem.IsInitialized || _visionSystem.RenderTexture == null)
        {
            MarkTargetNotVisible(deltaTime);
            _previousLumaFrameHadTarget = false;
            return false;
        }

        Camera colorCamera = _visionSystem.ObservationCamera != null
            ? _visionSystem.ObservationCamera
            : _trackingCamera;
        RenderTexture source = _visionSystem.RenderTexture;
        if (colorCamera == null || source == null)
        {
            MarkTargetNotVisible(deltaTime);
            _previousLumaFrameHadTarget = false;
            return false;
        }

        EnsureColorReadbackTexture(source.width, source.height);

        if (colorCamera.targetTexture == source)
            colorCamera.Render();

        RenderTexture previousActive = RenderTexture.active;
        try
        {
            RenderTexture.active = source;
            _colorReadbackTexture.ReadPixels(
                new Rect(0f, 0f, source.width, source.height),
                0,
                0,
                false);
            _colorReadbackTexture.Apply(false);
        }
        finally
        {
            RenderTexture.active = previousActive;
        }

        bool detected = TryFindTargetColorCentroid(
            _colorReadbackTexture,
            out Vector2 viewportCenter,
            out float matchedPixelFraction);

        EnsureLumaBuffers(source.width, source.height);
        FillCurrentLumaFrame(_colorReadbackTexture);

        if (!detected)
        {
            MarkTargetNotVisible(deltaTime);
            StoreCurrentLumaFrame(false);
            return true;
        }

        Vector2 nextViewportOffset = new Vector2(
            (viewportCenter.x - 0.5f) * 2f,
            (viewportCenter.y - 0.5f) * 2f);

        float estimatedDistance = EstimateColorTargetDistance(matchedPixelFraction);
        Ray centerRay = colorCamera.ViewportPointToRay(new Vector3(viewportCenter.x, viewportCenter.y, 1f));
        Vector3 targetLocalPos =
            colorCamera.transform.InverseTransformDirection(centerRay.direction.normalized) *
            estimatedDistance;

        Vector2 opticalFlowPixels = Vector2.zero;
        bool hasOpticalFlow = _useLucasKanadeFlow &&
            TryComputeLucasKanadeFlow(_colorReadbackTexture, out opticalFlowPixels, out _lastLucasKanadeConfidence);

        if (_hasPerceptionSample)
        {
            float invDelta = 1f / Mathf.Max(deltaTime, 1e-5f);

            if (hasOpticalFlow)
            {
                Vector2 previousViewportCenter = new Vector2(
                    viewportCenter.x - opticalFlowPixels.x / source.width,
                    viewportCenter.y - opticalFlowPixels.y / source.height);
                Ray previousCenterRay = colorCamera.ViewportPointToRay(
                    new Vector3(previousViewportCenter.x, previousViewportCenter.y, 1f));
                Vector3 previousTargetLocalPos =
                    colorCamera.transform.InverseTransformDirection(previousCenterRay.direction.normalized) *
                    estimatedDistance;

                _targetLocalVelocity = (targetLocalPos - previousTargetLocalPos) * invDelta;
                _viewportVelocity = new Vector2(
                    opticalFlowPixels.x / source.width * 2f,
                    opticalFlowPixels.y / source.height * 2f) * invDelta;
            }
            else
            {
                _targetLocalVelocity = (targetLocalPos - _lastTargetLocalPos) * invDelta;
                _viewportVelocity = (nextViewportOffset - _viewportOffset) * invDelta;
            }
        }
        else
        {
            _targetLocalVelocity = Vector3.zero;
            _viewportVelocity = Vector2.zero;
            _lastLucasKanadeConfidence = 0f;
        }

        _isTargetVisible = true;
        _targetColorBlobFraction = matchedPixelFraction;
        _lastTargetLocalPos = targetLocalPos;
        _lastKnownTargetCameraLocalPos = targetLocalPos;
        _viewportOffset = nextViewportOffset;
        _hasPerceptionSample = true;
        _hasLastKnownTarget = true;
        _timeSinceTargetSeen = 0f;
        StoreCurrentLumaFrame(true);
        return true;
    }

    private void EnsureColorReadbackTexture(int width, int height)
    {
        if (_colorReadbackTexture != null &&
            _colorReadbackTexture.width == width &&
            _colorReadbackTexture.height == height)
        {
            return;
        }

        if (_colorReadbackTexture != null)
        {
            if (Application.isPlaying)
                Destroy(_colorReadbackTexture);
            else
                DestroyImmediate(_colorReadbackTexture);
        }

        _colorReadbackTexture = new Texture2D(width, height, TextureFormat.RGBA32, false);
    }

    private void EnsureLumaBuffers(int width, int height)
    {
        if (_currentLumaFrame != null &&
            _previousLumaFrame != null &&
            _lumaFrameWidth == width &&
            _lumaFrameHeight == height)
        {
            return;
        }

        int pixelCount = width * height;
        _currentLumaFrame = new float[pixelCount];
        _previousLumaFrame = new float[pixelCount];
        _lumaFrameWidth = width;
        _lumaFrameHeight = height;
        _hasPreviousLumaFrame = false;
        _previousLumaFrameHadTarget = false;
        _lastLucasKanadeConfidence = 0f;
        _lastGroundTruthHintActive = false;
        _lastGroundTruthHintProbability = 0f;
    }

    private void FillCurrentLumaFrame(Texture2D source)
    {
        EnsureLumaBuffers(source.width, source.height);

        var pixels = source.GetRawTextureData<Color32>();
        for (int i = 0; i < pixels.Length; i++)
        {
            Color32 pixel = pixels[i];
            _currentLumaFrame[i] =
                (0.2126f * pixel.r + 0.7152f * pixel.g + 0.0722f * pixel.b) / 255f;
        }
    }

    private void StoreCurrentLumaFrame()
    {
        if (_currentLumaFrame == null || _previousLumaFrame == null)
            return;

        Array.Copy(_currentLumaFrame, _previousLumaFrame, _currentLumaFrame.Length);
        _hasPreviousLumaFrame = true;
    }

    private void StoreCurrentLumaFrame(bool hadTarget)
    {
        StoreCurrentLumaFrame();
        _previousLumaFrameHadTarget = hadTarget;
    }

    private bool TryComputeLucasKanadeFlow(
        Texture2D source,
        out Vector2 meanPixelFlow,
        out float confidence)
    {
        meanPixelFlow = Vector2.zero;
        confidence = 0f;

        if (!_hasPreviousLumaFrame ||
            !_previousLumaFrameHadTarget ||
            _currentLumaFrame == null ||
            _previousLumaFrame == null)
        {
            return false;
        }

        int width = source.width;
        int height = source.height;
        int border = _lkWindowRadius + 1;
        if (width <= border * 2 || height <= border * 2)
            return false;

        var pixels = source.GetRawTextureData<Color32>();
        GetTargetChromaticity(out float targetR, out float targetG, out float targetB);

        int trackedCount = 0;
        int candidateCount = 0;
        Vector2 weightedFlowSum = Vector2.zero;
        float totalWeight = 0f;

        for (int y = border; y < height - border; y += _lkFeatureStride)
        {
            int rowOffset = y * width;
            for (int x = border; x < width - border; x += _lkFeatureStride)
            {
                Color32 pixel = pixels[rowOffset + x];
                if (!TryGetColorMatchWeight(pixel, targetR, targetG, targetB, out _))
                    continue;

                if (!HasEnoughLumaGradient(x, y, width))
                    continue;

                candidateCount++;
                if (!TrySolveLucasKanadeAtPoint(x, y, width, out Vector2 flow, out float determinant))
                    continue;

                if (Mathf.Abs(flow.x) > _lkMaxPixelDisplacement ||
                    Mathf.Abs(flow.y) > _lkMaxPixelDisplacement)
                {
                    continue;
                }

                trackedCount++;
                float weight = Mathf.Sqrt(determinant);
                weightedFlowSum += flow * weight;
                totalWeight += weight;

                if (trackedCount >= _lkMaxFeatures)
                    break;
            }

            if (trackedCount >= _lkMaxFeatures)
                break;
        }

        if (trackedCount < _lkMinTrackedFeatures || totalWeight <= 0f)
            return false;

        meanPixelFlow = weightedFlowSum / totalWeight;
        confidence = Mathf.Clamp01((float)trackedCount / Mathf.Max(_lkMinTrackedFeatures, candidateCount));
        return true;
    }

    private bool HasEnoughLumaGradient(int x, int y, int width)
    {
        int index = y * width + x;
        float gx = 0.5f * (_currentLumaFrame[index + 1] - _currentLumaFrame[index - 1]);
        float gy = 0.5f * (_currentLumaFrame[index + width] - _currentLumaFrame[index - width]);
        return gx * gx + gy * gy >= _lkMinGradient * _lkMinGradient;
    }

    private bool TrySolveLucasKanadeAtPoint(
        int centerX,
        int centerY,
        int width,
        out Vector2 flow,
        out float determinant)
    {
        flow = Vector2.zero;
        determinant = 0f;

        float sumIx2 = 0f;
        float sumIy2 = 0f;
        float sumIxIy = 0f;
        float sumIxIt = 0f;
        float sumIyIt = 0f;

        for (int dy = -_lkWindowRadius; dy <= _lkWindowRadius; dy++)
        {
            int y = centerY + dy;
            int rowOffset = y * width;
            for (int dx = -_lkWindowRadius; dx <= _lkWindowRadius; dx++)
            {
                int x = centerX + dx;
                int index = rowOffset + x;

                float ix = 0.25f * (
                    _currentLumaFrame[index + 1] - _currentLumaFrame[index - 1] +
                    _previousLumaFrame[index + 1] - _previousLumaFrame[index - 1]);
                float iy = 0.25f * (
                    _currentLumaFrame[index + width] - _currentLumaFrame[index - width] +
                    _previousLumaFrame[index + width] - _previousLumaFrame[index - width]);
                float it = _currentLumaFrame[index] - _previousLumaFrame[index];

                sumIx2 += ix * ix;
                sumIy2 += iy * iy;
                sumIxIy += ix * iy;
                sumIxIt += ix * it;
                sumIyIt += iy * it;
            }
        }

        determinant = sumIx2 * sumIy2 - sumIxIy * sumIxIy;
        if (determinant < _lkMinDeterminant)
            return false;

        float bx = -sumIxIt;
        float by = -sumIyIt;
        float invDet = 1f / determinant;
        flow = new Vector2(
            (bx * sumIy2 - sumIxIy * by) * invDet,
            (sumIx2 * by - sumIxIy * bx) * invDet);

        return !float.IsNaN(flow.x) &&
               !float.IsNaN(flow.y) &&
               !float.IsInfinity(flow.x) &&
               !float.IsInfinity(flow.y);
    }

    private bool TryFindTargetColorCentroid(
        Texture2D source,
        out Vector2 viewportCenter,
        out float matchedPixelFraction)
    {
        viewportCenter = Vector2.zero;
        matchedPixelFraction = 0f;

        var pixels = source.GetRawTextureData<Color32>();
        int width = source.width;
        int height = source.height;
        int stride = Mathf.Max(1, _colorSampleStride);
        int sampledCount = 0;
        int matchedCount = 0;
        float weightedX = 0f;
        float weightedY = 0f;
        float totalWeight = 0f;

        GetTargetChromaticity(out float targetR, out float targetG, out float targetB);

        for (int y = 0; y < height; y += stride)
        {
            int rowOffset = y * width;
            for (int x = 0; x < width; x += stride)
            {
                sampledCount++;
                Color32 pixel = pixels[rowOffset + x];
                if (!TryGetColorMatchWeight(pixel, targetR, targetG, targetB, out float weight))
                    continue;

                matchedCount++;
                weightedX += (x + 0.5f) * weight;
                weightedY += (y + 0.5f) * weight;
                totalWeight += weight;
            }
        }

        if (sampledCount <= 0 || totalWeight <= 0f)
            return false;

        matchedPixelFraction = (float)matchedCount / sampledCount;
        if (matchedCount < _minMatchedColorSamples ||
            matchedPixelFraction < _minMatchedPixelFraction)
        {
            return false;
        }

        viewportCenter = new Vector2(
            Mathf.Clamp01(weightedX / totalWeight / width),
            Mathf.Clamp01(weightedY / totalWeight / height));
        return true;
    }

    private void GetTargetChromaticity(out float targetR, out float targetG, out float targetB)
    {
        float sum = Mathf.Max(_targetColor.r + _targetColor.g + _targetColor.b, 1e-5f);
        targetR = _targetColor.r / sum;
        targetG = _targetColor.g / sum;
        targetB = _targetColor.b / sum;
    }

    private bool TryGetColorMatchWeight(
        Color32 pixel,
        float targetR,
        float targetG,
        float targetB,
        out float weight)
    {
        weight = 0f;

        float r = pixel.r / 255f;
        float g = pixel.g / 255f;
        float b = pixel.b / 255f;
        float brightness = Mathf.Max(r, Mathf.Max(g, b));
        if (brightness < _minColorBrightness)
            return false;

        float minChannel = Mathf.Min(r, Mathf.Min(g, b));
        float saturation = brightness <= 1e-5f ? 0f : (brightness - minChannel) / brightness;
        if (saturation < _minColorSaturation)
            return false;

        float sum = Mathf.Max(r + g + b, 1e-5f);
        float chromR = r / sum;
        float chromG = g / sum;
        float chromB = b / sum;
        float colorDistance = Mathf.Sqrt(
            Square(chromR - targetR) +
            Square(chromG - targetG) +
            Square(chromB - targetB));

        if (colorDistance > _colorMatchTolerance)
            return false;

        weight = (1f - colorDistance / _colorMatchTolerance) * saturation * brightness;
        return weight > 0f;
    }

    private float EstimateColorTargetDistance(float matchedPixelFraction)
    {
        float safeFraction = Mathf.Max(matchedPixelFraction, 1e-5f);
        float estimatedDistance = _colorReferenceDistance *
            Mathf.Sqrt(_colorReferencePixelFraction / safeFraction);
        return Mathf.Clamp(estimatedDistance, _colorMinEstimatedDistance, _maxDistance);
    }

    private static float Square(float value)
    {
        return value * value;
    }

    private void MarkTargetNotVisible(float deltaTime)
    {
        _isTargetVisible = false;
        _timeSinceTargetSeen += deltaTime;
        _targetLocalVelocity = Vector3.zero;
        _viewportVelocity = Vector2.zero;
        _targetColorBlobFraction = 0f;
        _lastLucasKanadeConfidence = 0f;

        if (!_hasLastKnownTarget || _timeSinceTargetSeen > _lostTargetMemorySeconds)
        {
            _viewportOffset = Vector2.zero;
            _hasPerceptionSample = false;
        }
    }

    private bool HasLineOfSightToTarget(Vector3 worldTargetPos)
    {
        if (_trackingCamera == null || TargetTransform == null)
            return false;

        Vector3 origin = _trackingCamera.transform.position;
        Vector3 toTarget = worldTargetPos - origin;
        float distance = toTarget.magnitude;
        if (distance <= 1e-5f)
            return true;

        if (!Physics.Raycast(
                origin,
                toTarget.normalized,
                out RaycastHit hit,
                distance,
                _visibilityMask,
                QueryTriggerInteraction.Ignore))
        {
            return true;
        }

        Transform hitTransform = hit.transform;
        return hitTransform == TargetTransform || hitTransform.IsChildOf(TargetTransform);
    }

    private bool TryGetTrackedTargetLocalPosition(out Vector3 targetLocalPos)
    {
        if (_trackingCamera == null)
        {
            targetLocalPos = Vector3.zero;
            return false;
        }

        if (_isTargetVisible && _useColorTargetTracking)
        {
            targetLocalPos = _lastTargetLocalPos;
            return true;
        }

        if (_isTargetVisible && TargetTransform != null)
        {
            targetLocalPos = _trackingCamera.transform.InverseTransformPoint(TargetTransform.position);
            return true;
        }

        if (_hasLastKnownTarget && _timeSinceTargetSeen <= _lostTargetMemorySeconds)
        {
            if (_useColorTargetTracking)
            {
                targetLocalPos = _lastKnownTargetCameraLocalPos;
                return true;
            }

            targetLocalPos = _trackingCamera.transform.InverseTransformPoint(_lastKnownTargetWorldPos);
            return true;
        }

        targetLocalPos = Vector3.zero;
        return false;
    }

    private void ResetPerceptionState()
    {
        _isTargetVisible = false;
        _hasPerceptionSample = false;
        _hasLastKnownTarget = false;
        _timeSinceTargetSeen = 0f;
        _lastKnownTargetWorldPos = Vector3.zero;
        _lastKnownTargetCameraLocalPos = Vector3.zero;
        _lastTargetLocalPos = Vector3.zero;
        _targetLocalVelocity = Vector3.zero;
        _viewportOffset = Vector2.zero;
        _viewportVelocity = Vector2.zero;
        _targetColorBlobFraction = 0f;
        _hasPreviousLumaFrame = false;
        _previousLumaFrameHadTarget = false;
        _lastLucasKanadeConfidence = 0f;
    }

    private void ReportPerceptionDiagnostics()
    {
        if (!_reportPerceptionDiagnostics)
            return;

        _perceptionDiagnosticsCounter++;
        if (_perceptionDiagnosticsCounter < _perceptionDiagnosticsIntervalSteps)
            return;
        _perceptionDiagnosticsCounter = 0;

        Academy academy = Academy.Instance;
        if (academy == null)
            return;

        var stats = academy.StatsRecorder;
        float viewportError = _isTargetVisible ? Mathf.Clamp01(_viewportOffset.magnitude) : 1f;
        float normalizedBlobArea = _colorReferencePixelFraction > 0f
            ? Mathf.Clamp01(_targetColorBlobFraction / _colorReferencePixelFraction)
            : 0f;

        stats.Add("Diagnostics/PursuerTargetVisible", _isTargetVisible ? 1f : 0f);
        stats.Add("Diagnostics/PursuerColorBlobArea", _targetColorBlobFraction);
        stats.Add("Diagnostics/PursuerColorBlobAreaNorm", normalizedBlobArea);
        stats.Add("Diagnostics/PursuerViewportError", viewportError);
        stats.Add("Diagnostics/PursuerViewportCentered", 1f - viewportError);
        stats.Add("Diagnostics/PursuerLKConfidence", _lastLucasKanadeConfidence);
        stats.Add("Diagnostics/PursuerViewportSpeed", Mathf.Clamp01(_viewportVelocity.magnitude / _maxViewportSpeed));
        stats.Add("Diagnostics/PursuerHasLastKnownTarget", _hasLastKnownTarget ? 1f : 0f);
        stats.Add("Diagnostics/PursuerTimeSinceTargetSeen", _timeSinceTargetSeen);
        stats.Add("Diagnostics/PursuerGTHintActive", _lastGroundTruthHintActive ? 1f : 0f);
        stats.Add("Diagnostics/PursuerGTHintProbability", _lastGroundTruthHintProbability);
    }

    private void ReleaseColorReadbackTexture()
    {
        if (_colorReadbackTexture == null)
            return;

        if (Application.isPlaying)
            Destroy(_colorReadbackTexture);
        else
            DestroyImmediate(_colorReadbackTexture);

        _colorReadbackTexture = null;
        _previousLumaFrame = null;
        _currentLumaFrame = null;
        _lumaFrameWidth = 0;
        _lumaFrameHeight = 0;
        _hasPreviousLumaFrame = false;
        _previousLumaFrameHadTarget = false;
        _lastLucasKanadeConfidence = 0f;
        _lastGroundTruthHintActive = false;
        _lastGroundTruthHintProbability = 0f;
    }

    private void TryOverrideSpawnNearTarget()
    {
        if (!_spawnNearTargetAtEpisodeStart)
            return;

        ResolveTargetTransform();
        if (TargetTransform == null)
            return;

        Vector3 targetPos = TargetTransform.position;
        Vector3 bestClearCandidate = Vector3.zero;
        bool hasBestClearCandidate = false;
        int bestOverlapCount = int.MaxValue;
        Vector3 bestOverlapCandidate = targetPos;

        for (int i = 0; i < _nearTargetSpawnAttempts; i++)
        {
            Vector3 candidate = SampleNearTargetSpawnCandidate(targetPos, i);
            int overlapCount = GetSpawnCandidateBlockerCount(candidate);
            if (overlapCount < bestOverlapCount)
            {
                bestOverlapCount = overlapCount;
                bestOverlapCandidate = candidate;
            }

            if (overlapCount > 0)
                continue;

            if (!HasSpawnLineOfSightToTarget(candidate, targetPos))
            {
                if (!hasBestClearCandidate)
                {
                    bestClearCandidate = candidate;
                    hasBestClearCandidate = true;
                }
                continue;
            }

            transform.position = candidate;
            return;
        }

        if (hasBestClearCandidate)
        {
            transform.position = bestClearCandidate;
            return;
        }

        transform.position = bestOverlapCandidate;
    }

    private void TryProcessDeferredSpawnSync()
    {
        if (_awaitingTargetSpawnPlacement)
        {
            ResolveTargetTransform();
            if (TargetTransform == null)
                return;

            transform.position = _pendingBaseSpawnPosition;
            TryOverrideSpawnNearTarget();
            AlignSpawnRotationToTarget();
            ResetPhysicsState();
            _awaitingTargetSpawnPlacement = false;
            _spawnSyncStepsRemaining = Mathf.Max(0, _spawnSyncStepsRemaining - 1);
            return;
        }

        if (_spawnSyncStepsRemaining <= 0)
            return;

        ResolveTargetTransform();
        if (TargetTransform == null)
            return;

        TryOverrideSpawnNearTarget();
        AlignSpawnRotationToTarget();
        ResetPhysicsState();
        _spawnSyncStepsRemaining--;
    }

    private Vector3 SampleNearTargetSpawnCandidate(Vector3 targetPos, int sampleIndex)
    {
        float baseAngle = (Mathf.PI * 2f * sampleIndex) / Mathf.Max(1, _nearTargetSpawnAttempts);
        float angleJitter = UnityEngine.Random.Range(-0.25f, 0.25f);
        float angle = baseAngle + angleJitter;
        float distance = UnityEngine.Random.Range(_nearTargetSpawnMinDistance, _nearTargetSpawnMaxDistance);
        Vector3 horizontalOffset = new Vector3(Mathf.Cos(angle), 0f, Mathf.Sin(angle)) * distance;
        float verticalOffset = UnityEngine.Random.Range(-_nearTargetSpawnVerticalJitter, _nearTargetSpawnVerticalJitter);

        Vector3 candidate = targetPos + horizontalOffset;
        candidate.y = Mathf.Max(1f, targetPos.y + verticalOffset);
        return candidate;
    }

    private int GetSpawnCandidateBlockerCount(Vector3 candidate)
    {
        Collider[] hits = Physics.OverlapSphere(
            candidate,
            _nearTargetSpawnClearanceRadius,
            ~0,
            QueryTriggerInteraction.Ignore);

        int blockerCount = 0;
        foreach (Collider hit in hits)
        {
            if (hit == null || hit.isTrigger)
                continue;

            Transform hitTransform = hit.transform;
            if (hitTransform == transform || hitTransform.IsChildOf(transform))
                continue;

            if (TargetTransform != null &&
                (hitTransform == TargetTransform || hitTransform.IsChildOf(TargetTransform)))
            {
                continue;
            }

            if (_goalZone != null &&
                (hitTransform == _goalZone.transform || hitTransform.IsChildOf(_goalZone.transform)))
            {
                continue;
            }

            blockerCount++;
        }

        return blockerCount;
    }

    private bool HasSpawnLineOfSightToTarget(Vector3 spawnPos, Vector3 targetPos)
    {
        if (TargetTransform == null)
            return false;

        Vector3 origin = spawnPos + Vector3.up * 0.15f;
        Vector3 toTarget = targetPos - origin;
        float distance = toTarget.magnitude;
        if (distance <= 1e-5f)
            return true;

        if (!Physics.Raycast(
                origin,
                toTarget.normalized,
                out RaycastHit hit,
                distance,
                _visibilityMask,
                QueryTriggerInteraction.Ignore))
        {
            return true;
        }

        Transform hitTransform = hit.transform;
        return hitTransform == TargetTransform || hitTransform.IsChildOf(TargetTransform);
    }

    private void AlignSpawnRotationToTarget()
    {
        ResolveTargetTransform();

        if (TargetTransform == null)
        {
            transform.rotation = Quaternion.Euler(0f, UnityEngine.Random.Range(0f, 360f), 0f);
            return;
        }

        Vector3 flatToTarget = TargetTransform.position - transform.position;
        flatToTarget.y = 0f;

        if (flatToTarget.sqrMagnitude < 1e-4f)
        {
            transform.rotation = Quaternion.Euler(0f, UnityEngine.Random.Range(0f, 360f), 0f);
            return;
        }

        Quaternion lookRotation = Quaternion.LookRotation(flatToTarget.normalized, Vector3.up);
        float yawJitter = UnityEngine.Random.Range(-_spawnFacingJitterDeg, _spawnFacingJitterDeg);
        transform.rotation = lookRotation * Quaternion.Euler(0f, yawJitter, 0f);
    }
}
