using System;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
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

    [Header("Pursuer — Camera Tracking")]
    [SerializeField] private string _targetTag = "Evader";
    [SerializeField] private float _viewportVisibleMargin = 0.05f;
    [SerializeField] private float _lostTargetMemorySeconds = 0.75f;
    [SerializeField] private float _spawnFacingJitterDeg = 20f;
    [SerializeField] private LayerMask _visibilityMask = ~0;

    [Header("Pursuer — Observation Normalization")]
    [SerializeField] private float _maxDistance = 50f;
    [SerializeField] private float _maxObsSpeed = 10f;
    [SerializeField] private float _maxViewportSpeed = 4f;

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
    private Vector3 _lastTargetLocalPos;
    private Vector3 _targetLocalVelocity;
    private Vector2 _viewportOffset;
    private Vector2 _viewportVelocity;
    private bool _pendingEpisodeEnd;

    public override void Initialize()
    {
        Role = DroneRole.Pursuer;

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

        base.OnDisable();
    }

    private void OnValidate()
    {
        _maxEpisodeSeconds = Mathf.Max(1f, _maxEpisodeSeconds);
        _catchDistance = Mathf.Max(0.1f, _catchDistance);
        _viewportVisibleMargin = Mathf.Clamp(_viewportVisibleMargin, 0f, 0.5f);
        _lostTargetMemorySeconds = Mathf.Max(0f, _lostTargetMemorySeconds);
        _spawnFacingJitterDeg = Mathf.Clamp(_spawnFacingJitterDeg, 0f, 180f);
        _maxDistance = Mathf.Max(1f, _maxDistance);
        _maxObsSpeed = Mathf.Max(0.1f, _maxObsSpeed);
        _maxViewportSpeed = Mathf.Max(0.1f, _maxViewportSpeed);
    }

    public override void OnEpisodeBegin()
    {
        EnsureRuntimeReferences();

        _episodeTimer = 0f;
        _episodeSteps = 0;
        _episodeClosed = false;
        _pendingEpisodeEnd = false;
        ResetPerceptionState();
        _rewardCalculator?.ResetEpisodeState();

        ResolveGoalZone();
        ResolveTargetTransform();
        ResolveTrackingCamera();

        if (SpawnCenter.Current != null)
        {
            var range = SpawnCenter.Current.GetPursuerSpawnRange();
            transform.position = SpawnCenter.Current.GetRandomPosition(range);
        }
        else
        {
            transform.position = _spawnOrigin;
        }

        AlignSpawnRotationToTarget();
        ResetPhysicsState();
    }

    private void FixedUpdate()
    {
        if (_pendingEpisodeEnd)
        {
            _pendingEpisodeEnd = false;
            EndEpisode();
            return;
        }

        UpdateTargetPerception(Time.fixedDeltaTime);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        if (!IsDroneReady())
        {
            for (int i = 0; i < 44; i++)
                sensor.AddObservation(0f);
            return;
        }

        Vector3 localVel = transform.InverseTransformDirection(_dronePhysics.GetVelocity());
        Vector3 localAngVel = _dronePhysics.GetLocalAngularVelocity();
        sensor.AddObservation(localVel / _maxObsSpeed);                // 3
        sensor.AddObservation(localAngVel / _maxObsSpeed);            // 3
        sensor.AddObservation(transform.position.y / _maxDistance);   // 1

        bool hasTrackedTarget = TryGetTrackedTargetLocalPosition(out Vector3 targetLocalPos);
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

        if (_sensorSystem != null)
        {
            foreach (float d in _sensorSystem.GetAllNormalizedDistances())
                sensor.AddObservation(d);                             // 26
        }
        else
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

    public override void OnActionReceived(ActionBuffers actions)
    {
        if (!IsDroneReady() || _episodeClosed)
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
            viewportOffset: _viewportOffset));

        CheckTerminationConditions();
    }

    public void SetCrash()
    {
        if (_episodeClosed)
            return;

        _episodeClosed = true;
        AddReward(_rewardCalculator.CrashPenalty);
        _episodeLogger?.LogEpisode(EpisodeLogger.TermType.Crash, _episodeSteps);
        _pendingEpisodeEnd = true;
    }

    private void CheckTerminationConditions()
    {
        if (TargetTransform != null &&
            Vector3.Distance(transform.position, TargetTransform.position) < _catchDistance)
        {
            NotifyEvaderCaptured();
            BroadcastRoundEnd(RoundEndReason.Captured);
            return;
        }

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
                break;

            case RoundEndReason.Timeout:
                AddReward(_rewardCalculator.TimeoutPenalty);
                _episodeLogger?.LogEpisode(EpisodeLogger.TermType.Timeout, _episodeSteps);
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
        ResetPerceptionState();
        _rewardCalculator?.ResetEpisodeState();

        ResolveGoalZone();
        ResolveTargetTransform();
        ResolveTrackingCamera();

        if (SpawnCenter.Current != null)
        {
            var range = SpawnCenter.Current.GetPursuerSpawnRange();
            transform.position = SpawnCenter.Current.GetRandomPosition(range);
        }
        else
        {
            transform.position = _spawnOrigin;
        }

        AlignSpawnRotationToTarget();
        ResetPhysicsState();
    }

    private void EnsureRuntimeReferences()
    {
        if (_rewardCalculator == null)
            _rewardCalculator = GetComponent<PursuerReward>();

        if (_episodeLogger == null)
            _episodeLogger = GetComponent<EpisodeLogger>();
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

        if (TargetTransform == null || _trackingCamera == null)
        {
            _isTargetVisible = false;
            _timeSinceTargetSeen += deltaTime;
            _targetLocalVelocity = Vector3.zero;
            _viewportOffset = Vector2.zero;
            _viewportVelocity = Vector2.zero;
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
            _hasLastKnownTarget = true;
            _timeSinceTargetSeen = 0f;
        }
        else
        {
            _timeSinceTargetSeen += deltaTime;
            _targetLocalVelocity = Vector3.zero;
            _viewportVelocity = Vector2.zero;

            if (!_hasLastKnownTarget || _timeSinceTargetSeen > _lostTargetMemorySeconds)
            {
                _viewportOffset = Vector2.zero;
                _hasPerceptionSample = false;
            }
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

        if (_isTargetVisible && TargetTransform != null)
        {
            targetLocalPos = _trackingCamera.transform.InverseTransformPoint(TargetTransform.position);
            return true;
        }

        if (_hasLastKnownTarget && _timeSinceTargetSeen <= _lostTargetMemorySeconds)
        {
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
        _lastTargetLocalPos = Vector3.zero;
        _targetLocalVelocity = Vector3.zero;
        _viewportOffset = Vector2.zero;
        _viewportVelocity = Vector2.zero;
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
