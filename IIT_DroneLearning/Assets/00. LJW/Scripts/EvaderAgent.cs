using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using DroneSensor;

/// <summary>
/// EvaderAgent — DroneAgent 기반 회피 드론 에이전트
///
/// DroneAgent에서 그대로 사용:
///   - _dronePhysics, _sensorSystem (Awake에서 자동 설정)
///   - TargetTransform (Inspector 필드, 추격 드론)
///   - QE/WASD/화살표 Heuristic
///   - DroneRole.Evader
///
/// EvaderAgent가 추가/오버라이드:
///   - Stage0-A/B 전환 (_goalOnlyMode, _currentStage)
///   - 반경·고도 기반 랜덤 스폰
///   - 18-obs 관측 (자기 상태 7 + 목표 4 + 추격자/메모리 7)
///   - EvaderReward 위임 보상 + 종료 조건 (목표·포획·타임아웃)
///   - LOS 기반 추격자 메모리 (Stage1+)
///   - 외부 API: SetCrash(), SetPursuerVisibility()
///
/// 목표지점 관련 (위치 랜덤화·도달 판정):
///   - Goal 컴포넌트에 위임 (_goalZone 참조)
///   - Inspector 슬롯 미연결 시 Goal.Current 자동 사용
///   - OnEpisodeBegin: _goalZone.RandomizePosition()
///   - CheckTermination: _goalZone.IsArrived() + _goalZone.NotifyArrival()
///
/// 담당: 이재왕 (work/evader)
/// </summary>
public class EvaderAgent : DroneAgent
{
    private enum Stage1RewardPreset
    {
        LegacyV7,
        Experimental
    }

    private enum CrashKind
    {
        None,
        BoundaryOverflow,
        OuterWall,
        BuildingOrObstacle,
        Other,
    }

    private enum GoalClearanceFailureType
    {
        None = 0,
        InsideObstacle = 1,
        ObstacleProximity = 2,
        EnvelopeCollision = 3,
        BoundaryMargin = 4,
    }

    [Header("Evader — Auto Defaults")]
    [Tooltip("true면 Stage1-A 권장값을 코드에서 자동 적용해 Inspector 수동 세팅을 최소화한다")]
    [SerializeField] private bool _autoApplyStage1Defaults = true;

    // ───────── Inspector 설정 (Evader 전용) ───────────────────────────────
    // GoalTransform   : DroneAgent.GoalTransform  사용 (목표 지점)
    // TargetTransform : DroneAgent.TargetTransform 사용 (추격 드론)

    [Header("Evader — Goal Zone")]
    [Tooltip("Goal GameObject에 부착된 Goal 컴포넌트")]
    [SerializeField] private Goal _goalZone;

    [Header("Evader — Goal Placement Validation")]
    [Tooltip("Goal이 Building/Wall에 너무 가까우면 에피소드 시작 시 재샘플링")]
    [SerializeField] private bool _validateGoalPlacement = true;
    [Tooltip("Goal 중심 주변 이 반경(m) 안에 Building/Wall이 있으면 재샘플링")]
    [SerializeField] private float _goalObstacleClearanceRadius = 12.0f;
    [Tooltip("Goal 배치 재샘플링 최대 시도 횟수")]
    [SerializeField] private int _goalResampleMaxAttempts = 80;
    [Tooltip("Goal이 경계에 너무 가까우면 재샘플링 (boundaryHalfSize - margin)")]
    [SerializeField] private float _goalBoundaryClearanceMargin = 6.0f;

    [Header("Evader — Sensor Hardening")]
    [Tooltip("true면 Goal을 Ignore Raycast 레이어로 강제해 Goal 오인식 회피")]
    [SerializeField] private bool _autoSetGoalIgnoreRaycastLayer = true;
    [Tooltip("true면 센서 DetectionLayerMask에서 Ignore Raycast 레이어를 제외")]
    [SerializeField] private bool _excludeIgnoreRaycastFromSensorMask = true;
    [Tooltip("true면 DetectionLayerMask가 비어 있을 때 안전한 기본 마스크로 자동 복구")]
    [SerializeField] private bool _forceNonEmptyDetectionLayerMask = true;
    [Tooltip("true면 Stage1+에서 Middle-Bottom 레이어를 끄고 하향 회피 과민 반응을 줄임")]
    [SerializeField] private bool _disableMiddleBottomRaysInStage1 = true;
    [Tooltip("true면 Stage1+에서 Bottom 레이어를 끄고 저층 건물 상공 통과를 허용")]
    [SerializeField] private bool _disableBottomRayInStage1 = true;

    [Header("Evader — Reward Hardening")]
    [Tooltip("true면 Stage1+ 시작 시 EvaderReward를 권장 프로파일로 강제 적용")]
    [SerializeField] private bool _applyStage1RewardProfile = true;
    [Tooltip("true면 프리셋(legacy/experimental) 프로파일을 사용한다. false면 _stage1RewardProfile 수동값을 그대로 사용")]
    [SerializeField] private bool _useStage1RewardPreset = true;
    [Tooltip("실험 중 문제가 생기면 LegacyV7로 즉시 롤백")]
    [SerializeField] private Stage1RewardPreset _stage1RewardPreset = Stage1RewardPreset.Experimental;
    [Tooltip("활성 Stage1 보상 프로파일을 런타임 시작 시 1회 로그 출력")]
    [SerializeField] private bool _logActiveStage1RewardProfile = true;
    [SerializeField] private EvaderReward.Stage1RewardProfile _stage1RewardProfile = new EvaderReward.Stage1RewardProfile
    {
        VelAlignCoeff = 0.013f,
        GoalPriorityDist = 7.0f,
        ProximityCoeff = 0.0060f,
        ProximityThreshold = 0.24f,
        NearGoalObstaclePenaltyScale = 0.78f,
        GoalApproachObstaclePenaltyScale = 0.70f,
        TimePenaltyPerStep = -0.0015f,
        SurvivalRewardPerStep = 0.0008f,
        NearGoalShapingMultiplier = 1.8f,
        MiddleBottomRayPenaltyWeight = 0.12f,
        BottomRayPenaltyWeight = 0.0f,
    };
    [SerializeField] private EvaderReward.Stage1RewardProfile _stage1RewardProfileLegacy = new EvaderReward.Stage1RewardProfile
    {
        VelAlignCoeff = 0.012f,
        GoalPriorityDist = 6.0f,
        ProximityCoeff = 0.006f,
        ProximityThreshold = 0.25f,
        NearGoalObstaclePenaltyScale = 0.35f,
        GoalApproachObstaclePenaltyScale = 0.45f,
        TimePenaltyPerStep = -0.001f,
        SurvivalRewardPerStep = 0.0f,
        NearGoalShapingMultiplier = 3.0f,
        MiddleBottomRayPenaltyWeight = 0.15f,
        BottomRayPenaltyWeight = 0.0f,
    };
    [SerializeField] private EvaderReward.Stage1RewardProfile _stage1RewardProfileExperimental = new EvaderReward.Stage1RewardProfile
    {
        VelAlignCoeff = 0.013f,
        GoalPriorityDist = 7.0f,
        ProximityCoeff = 0.0060f,
        ProximityThreshold = 0.24f,
        NearGoalObstaclePenaltyScale = 0.78f,
        GoalApproachObstaclePenaltyScale = 0.70f,
        TimePenaltyPerStep = -0.0015f,
        SurvivalRewardPerStep = 0.0008f,
        NearGoalShapingMultiplier = 1.8f,
        MiddleBottomRayPenaltyWeight = 0.12f,
        BottomRayPenaltyWeight = 0.0f,
    };

    [Header("Evader — Episode Settings")]
    [SerializeField] private float _maxEpisodeSeconds = 40f;
    [SerializeField] private float _catchDistance     = 1.5f;
    [SerializeField] private float _goalArrivalReward = 3.0f;

    [Header("Evader — Action Smoothing")]
    [Tooltip("true면 정책 출력을 저역통과 + 변화량 제한으로 완화해 급가속/급선회를 줄인다")]
    [SerializeField] private bool _useActionSmoothing = true;
    [Tooltip("절대값이 이 값 미만인 행동 입력은 0으로 처리")]
    [SerializeField] private float _actionDeadzone = 0.03f;
    [Tooltip("현재 명령에서 목표 명령으로 보간 비율 (0~1)")]
    [SerializeField] private float _commandLerpFactor = 0.24f;
    [Tooltip("FixedUpdate 1스텝당 명령 변화량 최대치 (0~1)")]
    [SerializeField] private float _maxCommandDeltaPerStep = 0.14f;

    [Header("Evader — Anti-Hover")]
    [Tooltip("true면 goal로 유의미한 진전이 없을 때 추가 페널티를 부여한다")]
    [SerializeField] private bool _enableStagnationPenalty = true;
    [Tooltip("이 거리(m)보다 멀 때만 정체 판정을 적용한다")]
    [SerializeField] private float _stagnationWatchDistance = 18f;
    [Tooltip("1스텝 목표거리 감소량이 이 값 이하이면 정체로 간주한다 (m/step)")]
    [SerializeField] private float _stagnationProgressThreshold = 0.010f;
    [Tooltip("정체 판정 시작 전 유예 스텝 수")]
    [SerializeField] private int _stagnationGraceSteps = 40;
    [Tooltip("정체 상태에서 스텝당 추가 페널티 (음수)")]
    [SerializeField] private float _stagnationPenaltyPerStep = -0.0012f;

    [Header("Evader — Wall Proximity Safety")]
    [Tooltip("true면 전방 벽 초근접 시 forward 입력을 억제하고 회피 조향을 보정한다")]
    [SerializeField] private bool _enableWallProximitySafety = true;
    [Tooltip("전방 Middle 레이 정규화 거리 임계치 (0~1). 이 값 미만이면 벽 초근접으로 본다")]
    [SerializeField] private float _wallProximityThreshold = 0.13f;
    [Tooltip("초근접 시 forward pitch 입력을 강제로 차단한다")]
    [SerializeField] private bool _blockForwardPitchOnWallProximity = true;
    [Tooltip("forward 차단 진입 위험도 임계치 (0~1)")]
    [SerializeField] private float _wallForwardBlockEnterRisk = 0.52f;
    [Tooltip("forward 차단 해제 위험도 임계치 (0~1, enter보다 낮게 유지)")]
    [SerializeField] private float _wallForwardBlockReleaseRisk = 0.24f;
    [Tooltip("이 위험도 이상이면 제한된 역피치 보조를 시작한다 (완전 역추진 금지)")]
    [SerializeField] private float _wallReverseAssistStartRisk = 0.40f;
    [Tooltip("역피치 보조 최대치 (0~0.35 권장). pitch 하한은 -값으로 제한된다")]
    [SerializeField] private float _wallReverseAssistMaxPitch = 0.16f;
    [Tooltip("벽 회피 시 roll 보정 강도 (0~1)")]
    [SerializeField] private float _wallAvoidRollAssist = 0.16f;
    [Tooltip("벽 회피 시 yaw 보정 강도 (0~1)")]
    [SerializeField] private float _wallAvoidYawAssist = 0.12f;
    [Tooltip("좌우 회피 바이어스 평활 비율 (0~1). 낮을수록 진동 완화")]
    [SerializeField] private float _wallAvoidBiasSmoothing = 0.18f;
    [Tooltip("정면 위험 + 좌우 대칭 상황에서 측면 여유 기반 바이어스 배율")]
    [SerializeField] private float _wallSymmetricClearanceBiasScale = 0.10f;

    [Header("Evader — Boundary Constraints")]
    [Tooltip("이 고도(m) 초과 시 맵 이탈로 판정 (벽 높이 20m)")]
    [SerializeField] private float _maxFlightHeight   = 20f;
    [Tooltip("XZ 절대값이 이 거리(m) 초과 시 맵 이탈로 판정 (도시 반폭 ~72m)")]
    [SerializeField] private float _boundaryHalfSize  = 72f;
    [Tooltip("에피소드 시작 시 스폰 좌표를 경계 안쪽으로 클램프하는 여유 거리(m)")]
    [SerializeField] private float _spawnBoundaryInset = 2f;

    [Header("Evader — Boundary Safety")]
    [Tooltip("true면 경계 근접 시 중심 방향으로 조향을 보정한다")]
    [SerializeField] private bool _enableBoundarySafety = true;
    [Tooltip("경계 위험 구간 시작 여유 거리(m). boundaryHalfSize - margin부터 보정 시작")]
    [SerializeField] private float _boundarySafetyMargin = 10f;
    [Tooltip("경계 근접 시 중심 방향 roll 보정 강도 (0~1)")]
    [SerializeField] private float _boundaryInwardRollAssist = 0.15f;
    [Tooltip("경계 근접 시 중심 방향 yaw 보정 강도 (0~1)")]
    [SerializeField] private float _boundaryInwardYawAssist = 0.22f;
    [Tooltip("경계 근접 + 외향 진행 시 forward pitch 감쇠 하한 (0~1)")]
    [SerializeField] private float _boundaryForwardPitchMinScale = 0.20f;
    [Tooltip("경계 위험 구간 스텝당 추가 페널티 (음수 권장)")]
    [SerializeField] private float _boundaryRiskPenaltyPerStep = -0.0015f;
    [Tooltip("외향 속도 위험도 정규화 기준 (m/s)")]
    [SerializeField] private float _boundaryOutwardSpeedReference = 6.0f;
    [Tooltip("이 위험도 이상에서 긴급 inward steering을 활성화")]
    [SerializeField] private float _boundaryEmergencyRiskThreshold = 0.88f;
    [Tooltip("긴급 복귀 시 추가 roll 보정 강도")]
    [SerializeField] private float _boundaryEmergencyRollAssist = 0.28f;
    [Tooltip("긴급 복귀 시 추가 yaw 보정 강도")]
    [SerializeField] private float _boundaryEmergencyYawAssist = 0.32f;

    [Header("Evader — Terminal Penalties")]
    [Tooltip("경계 이탈 crash 시 terminal reward")]
    [SerializeField] private float _terminalPenaltyBoundaryOverflow = -1.25f;
    [Tooltip("외벽 충돌 crash 시 terminal reward")]
    [SerializeField] private float _terminalPenaltyOuterWall = -1.10f;
    [Tooltip("건물/장애물 충돌 crash 시 terminal reward")]
    [SerializeField] private float _terminalPenaltyBuilding = -1.60f;
    [Tooltip("기타 crash 시 terminal reward")]
    [SerializeField] private float _terminalPenaltyOtherCrash = -1.00f;

    [Header("Evader — Collision Detection")]
    [Tooltip("이 태그 오브젝트와 충돌 시 Crash 처리 (비우면 모든 충돌에 반응)")]
    [SerializeField] private string[] _crashTags = { "Building", "Wall" };

    [Header("Evader — Stage Control")]
    [Tooltip("true = Stage0-A (Pursuer 없음, 순수 목표 도달 학습)")]
    [SerializeField] private bool _goalOnlyMode  = true;
    [SerializeField] private int  _currentStage  = 1;

    [Header("Evader — Episode Checkpoints")]
    [Tooltip("에피소드당 가상 체크포인트 수 (스폰→Goal 직선 위 균등 배치)")]
    [SerializeField] private int   _checkpointCount      = 2;
    [Tooltip("체크포인트 도달 판정 반경 (m)")]
    [SerializeField] private float _checkpointRadius     = 4f;
    [Tooltip("체크포인트 도달 시 보상 (순서대로 1개씩)")]
    [SerializeField] private float _checkpointReward     = 0.1f;
    [Tooltip("체크포인트 보상의 유효 상한. 씬 값이 더 커도 이 값으로 캡핑")]
    [SerializeField] private float _checkpointRewardCap  = 0.05f;
    [Tooltip("체크포인트 인덱스가 증가할수록 보상 감쇠 (1번째는 1.0배)")]
    [SerializeField] private float _checkpointRewardDecay = 0.35f;
    [Tooltip("체크포인트 후보 위치의 장애물 검사 반경 (m)")]
    [SerializeField] private float _checkpointClearRadius = 2f;

    [Header("Evader — SpawnCenter 쿼리 결과 (읽기 전용)")]
    [SerializeField] private float _queriedMinY;
    [SerializeField] private float _queriedMaxY;
    [SerializeField] private float _queriedRadius;

    [Header("Evader — Observation Normalization")]
    [SerializeField] private float _maxDistance = 50f;
    [SerializeField] private float _maxObsSpeed = 10f;

    [Header("Evader — Reward Debug")]
    [Tooltip("true면 N step마다 보상 항목 분해 로그를 출력")]
    [SerializeField] private bool _logRewardBreakdown = false;
    [Tooltip("보상 분해 로그 출력 주기(step)")]
    [SerializeField] private int _rewardLogIntervalSteps = 1000;

    // ───────── 내부 상태 ──────────────────────────────────────────────────
    // _rb 는 DroneAgent.ResetPhysicsState() 로 위임 — 직접 선언 불필요
    private EvaderReward  _rewardCalculator;
    private EpisodeLogger _episodeLogger;
    private float         _episodeTimer;
    private int           _episodeSteps;
    private bool          _episodeEnded;  // OnCollisionEnter 중복 호출 방지

    private Vector3 _spawnCenter;  // SpawnCenter 미설정 시 폴백용 초기 위치
    private int     _goalResampleAttemptsLast;
    private CrashKind _lastCrashKind = CrashKind.None;
    private string _lastCrashObjectName = string.Empty;
    private float _lastCrashTerminalPenalty = -1.0f;
    private GoalClearanceFailureType _lastGoalClearanceFailureType = GoalClearanceFailureType.None;

    // 체크포인트 상태
    private Vector3[] _episodeCheckpoints;
    private int       _nextCheckpointIdx;

    // 추격자 추적 (Stage1+ 은폐 메모리)
    private Vector3 _lastKnownPursuerPos;
    private float   _timeSincePursuerDetected;
    private bool    _isPursuerVisible;
    private bool    _hasLoggedStage1RewardProfile;
    private float   _smoothedThrust;
    private float   _smoothedRoll;
    private float   _smoothedPitch;
    private float   _smoothedYaw;
    private float   _lastGoalDistance = float.MaxValue;
    private int     _stagnationStepCount;
    private bool    _wallForwardBlockLatched;
    private float   _wallAvoidBiasState;
    private static readonly Vector3[] GoalClearanceProbeDirections =
    {
        Vector3.forward,
        Vector3.back,
        Vector3.right,
        Vector3.left,
        (Vector3.forward + Vector3.right).normalized,
        (Vector3.forward + Vector3.left).normalized,
        (Vector3.back + Vector3.right).normalized,
        (Vector3.back + Vector3.left).normalized,
        Vector3.up,
        (Vector3.up + Vector3.forward).normalized,
        (Vector3.up + Vector3.back).normalized,
    };
    private static readonly string[] DefaultObstacleTags = { "Building", "Wall" };
    private static readonly string[] OuterWallNameTokens = { "citywalls", "outerwall", "outer_wall", "boundarywall", "boundary_wall" };
    private static readonly string[] BuildingNameTokens = { "building", "tower", "block", "obstacle" };
    private static readonly string[] ObstacleNameTokens = { "building", "wall", "citywalls", "city_wall" };
    private static readonly string[] NonObstacleNameTokens =
    {
        "goal",
        "spawn",
        "checkpoint",
        "evader",
        "pursuer",
        "drone",
        "sensor",
        "camera",
        "floor",
    };

    // ───────── 초기화 ─────────────────────────────────────────────────────
    // _dronePhysics, _sensorSystem 은 DroneAgent.Awake()에서 자동 설정됨
    public override void Initialize()
    {
        Role              = DroneRole.Evader;
        ApplyStage1RecommendedDefaults();
        // _rb는 DroneAgent.Awake()에서 이미 설정됨
        _rewardCalculator = GetComponent<EvaderReward>();
        if (_rewardCalculator == null)
            _rewardCalculator = gameObject.AddComponent<EvaderReward>();

        _episodeLogger = GetComponent<EpisodeLogger>();

        _spawnCenter = transform.position;  // _spawnCenterTransform 미설정 시 폴백

        // SpawnCenter에서 도망자 범위 쿼리
        QuerySpawnCenter();

        // Inspector 슬롯이 비어있으면 씬의 Goal.Current를 자동으로 사용
        if (_goalZone == null)
        {
            _goalZone = Goal.Current;
            if (_goalZone == null)
                Debug.LogWarning("[EvaderAgent] Goal을 찾을 수 없습니다. Inspector에서 수동 연결하거나 씬에 Goal 컴포넌트를 추가하세요.", this);
        }

        if (_goalZone != null)
            _goalZone.OnArrival += OnGoalArrived;

        ApplyGoalAndSensorHardening();
    }

    private void ApplyStage1RecommendedDefaults()
    {
        if (!_autoApplyStage1Defaults)
            return;

        _goalOnlyMode = true;
        _currentStage = 1;

        _maxEpisodeSeconds = 40f;
        _goalArrivalReward = 3.0f;

        _checkpointCount = 2;
        _checkpointRadius = 4f;
        _checkpointReward = 0.10f;
        _checkpointClearRadius = 2f;

        _validateGoalPlacement = true;
        _goalObstacleClearanceRadius = 14.0f;
        _goalResampleMaxAttempts = 120;
        _goalBoundaryClearanceMargin = 6.0f;

        _autoSetGoalIgnoreRaycastLayer = true;
        _excludeIgnoreRaycastFromSensorMask = true;
        _forceNonEmptyDetectionLayerMask = true;
        _disableMiddleBottomRaysInStage1 = true;
        _disableBottomRayInStage1 = true;

        _applyStage1RewardProfile = true;
        _useStage1RewardPreset = true;
        _stage1RewardPreset = Stage1RewardPreset.Experimental;
        _stage1RewardProfileLegacy = new EvaderReward.Stage1RewardProfile
        {
            VelAlignCoeff = 0.012f,
            GoalPriorityDist = 6.0f,
            ProximityCoeff = 0.006f,
            ProximityThreshold = 0.25f,
            NearGoalObstaclePenaltyScale = 0.35f,
            GoalApproachObstaclePenaltyScale = 0.45f,
            TimePenaltyPerStep = -0.001f,
            SurvivalRewardPerStep = 0.0f,
            NearGoalShapingMultiplier = 3.0f,
            MiddleBottomRayPenaltyWeight = 0.15f,
            BottomRayPenaltyWeight = 0.0f,
        };
        _stage1RewardProfileExperimental = new EvaderReward.Stage1RewardProfile
        {
            VelAlignCoeff = 0.013f,
            GoalPriorityDist = 7.0f,
            ProximityCoeff = 0.0060f,
            ProximityThreshold = 0.24f,
            NearGoalObstaclePenaltyScale = 0.78f,
            GoalApproachObstaclePenaltyScale = 0.70f,
            TimePenaltyPerStep = -0.0015f,
            SurvivalRewardPerStep = 0.0008f,
            NearGoalShapingMultiplier = 1.8f,
            MiddleBottomRayPenaltyWeight = 0.12f,
            BottomRayPenaltyWeight = 0.0f,
        };
        _stage1RewardProfile = _stage1RewardProfileExperimental;

        _useActionSmoothing = true;
        _actionDeadzone = 0.03f;
        _commandLerpFactor = 0.24f;
        _maxCommandDeltaPerStep = 0.14f;

        _enableWallProximitySafety = true;
        _wallProximityThreshold = 0.14f;
        _blockForwardPitchOnWallProximity = true;
        _wallForwardBlockEnterRisk = 0.52f;
        _wallForwardBlockReleaseRisk = 0.24f;
        _wallReverseAssistStartRisk = 0.40f;
        _wallReverseAssistMaxPitch = 0.16f;
        _wallAvoidRollAssist = 0.18f;
        _wallAvoidYawAssist = 0.14f;
        _wallAvoidBiasSmoothing = 0.18f;
        _wallSymmetricClearanceBiasScale = 0.10f;

        _enableBoundarySafety = true;
        _boundarySafetyMargin = 12f;
        _boundaryInwardRollAssist = 0.20f;
        _boundaryInwardYawAssist = 0.26f;
        _boundaryForwardPitchMinScale = 0.15f;
        _boundaryRiskPenaltyPerStep = -0.0030f;
        _boundaryOutwardSpeedReference = 6.0f;
        _boundaryEmergencyRiskThreshold = 0.88f;
        _boundaryEmergencyRollAssist = 0.28f;
        _boundaryEmergencyYawAssist = 0.32f;
        _spawnBoundaryInset = 2f;

        _terminalPenaltyBoundaryOverflow = -1.25f;
        _terminalPenaltyOuterWall = -1.10f;
        _terminalPenaltyBuilding = -1.60f;
        _terminalPenaltyOtherCrash = -1.00f;

        _enableStagnationPenalty = true;
        _stagnationWatchDistance = 18f;
        _stagnationProgressThreshold = 0.010f;
        _stagnationGraceSteps = 40;
        _stagnationPenaltyPerStep = -0.0012f;
    }

    private void OnValidate()
    {
        ApplyStage1RecommendedDefaults();

        _maxEpisodeSeconds = Mathf.Max(1f,   _maxEpisodeSeconds);
        _catchDistance     = Mathf.Max(0.1f, _catchDistance);
        _maxDistance       = Mathf.Max(1f,   _maxDistance);
        _maxObsSpeed       = Mathf.Max(0.1f, _maxObsSpeed);
        _goalObstacleClearanceRadius = Mathf.Max(0.5f, _goalObstacleClearanceRadius);
        _goalResampleMaxAttempts     = Mathf.Max(1,    _goalResampleMaxAttempts);
        _goalBoundaryClearanceMargin = Mathf.Clamp(_goalBoundaryClearanceMargin, 0f, Mathf.Max(0f, _boundaryHalfSize - 1f));
        _rewardLogIntervalSteps      = Mathf.Max(1,    _rewardLogIntervalSteps);
        _goalArrivalReward           = Mathf.Max(0.1f, _goalArrivalReward);
        _checkpointRewardCap         = Mathf.Max(0f,   _checkpointRewardCap);
        _checkpointRewardDecay       = Mathf.Clamp01(_checkpointRewardDecay);
        _actionDeadzone              = Mathf.Clamp01(_actionDeadzone);
        _commandLerpFactor           = Mathf.Clamp01(_commandLerpFactor);
        _maxCommandDeltaPerStep      = Mathf.Clamp(_maxCommandDeltaPerStep, 0.01f, 1f);
        _stagnationWatchDistance     = Mathf.Max(1f, _stagnationWatchDistance);
        _stagnationProgressThreshold = Mathf.Max(0f, _stagnationProgressThreshold);
        _stagnationGraceSteps        = Mathf.Max(1, _stagnationGraceSteps);
        _stagnationPenaltyPerStep    = Mathf.Min(0f, _stagnationPenaltyPerStep);
        _wallProximityThreshold      = Mathf.Clamp(_wallProximityThreshold, 0.02f, 1f);
        _wallForwardBlockEnterRisk   = Mathf.Clamp(_wallForwardBlockEnterRisk, 0.05f, 1f);
        _wallForwardBlockReleaseRisk = Mathf.Clamp(_wallForwardBlockReleaseRisk, 0f, _wallForwardBlockEnterRisk);
        _wallReverseAssistStartRisk  = Mathf.Clamp(_wallReverseAssistStartRisk, 0f, 1f);
        _wallReverseAssistMaxPitch   = Mathf.Clamp(_wallReverseAssistMaxPitch, 0.01f, 0.35f);
        _wallAvoidRollAssist         = Mathf.Clamp01(_wallAvoidRollAssist);
        _wallAvoidYawAssist          = Mathf.Clamp01(_wallAvoidYawAssist);
        _wallAvoidBiasSmoothing      = Mathf.Clamp(_wallAvoidBiasSmoothing, 0.01f, 1f);
        _wallSymmetricClearanceBiasScale = Mathf.Clamp(_wallSymmetricClearanceBiasScale, 0f, 0.35f);
        _boundarySafetyMargin        = Mathf.Clamp(_boundarySafetyMargin, 1f, Mathf.Max(2f, _boundaryHalfSize - 1f));
        _boundaryInwardRollAssist    = Mathf.Clamp01(_boundaryInwardRollAssist);
        _boundaryInwardYawAssist     = Mathf.Clamp01(_boundaryInwardYawAssist);
        _boundaryForwardPitchMinScale = Mathf.Clamp01(_boundaryForwardPitchMinScale);
        _boundaryRiskPenaltyPerStep  = Mathf.Min(0f, _boundaryRiskPenaltyPerStep);
        _boundaryOutwardSpeedReference = Mathf.Max(0.1f, _boundaryOutwardSpeedReference);
        _boundaryEmergencyRiskThreshold = Mathf.Clamp(_boundaryEmergencyRiskThreshold, 0.5f, 1f);
        _boundaryEmergencyRollAssist = Mathf.Clamp01(_boundaryEmergencyRollAssist);
        _boundaryEmergencyYawAssist = Mathf.Clamp01(_boundaryEmergencyYawAssist);
        _spawnBoundaryInset = Mathf.Clamp(_spawnBoundaryInset, 0f, Mathf.Max(0f, _boundaryHalfSize - 1f));

        _terminalPenaltyBoundaryOverflow = Mathf.Min(0f, _terminalPenaltyBoundaryOverflow);
        _terminalPenaltyOuterWall = Mathf.Min(0f, _terminalPenaltyOuterWall);
        _terminalPenaltyBuilding = Mathf.Min(0f, _terminalPenaltyBuilding);
        _terminalPenaltyOtherCrash = Mathf.Min(0f, _terminalPenaltyOtherCrash);

        if (_wallForwardBlockReleaseRisk > _wallForwardBlockEnterRisk - 0.02f)
            _wallForwardBlockReleaseRisk = Mathf.Max(0f, _wallForwardBlockEnterRisk - 0.02f);
    }

    // ───────── 에피소드 시작 ──────────────────────────────────────────────
    // EpisodeSpawnCoordinator 우선, 없으면 SpawnCenter 직접 사용 폴백
    public override void OnEpisodeBegin()
    {
        _episodeTimer             = 0f;
        _episodeSteps             = 0;
        _episodeEnded             = false;
        _timeSincePursuerDetected = 0f;
        _isPursuerVisible         = false;
        _lastKnownPursuerPos      = Vector3.zero;
        _nextCheckpointIdx        = 0;
        _smoothedThrust           = 0f;
        _smoothedRoll             = 0f;
        _smoothedPitch            = 0f;
        _smoothedYaw              = 0f;
        _stagnationStepCount      = 0;
        _lastGoalDistance         = float.MaxValue;
        _wallForwardBlockLatched  = false;
        _wallAvoidBiasState       = 0f;
        _lastCrashKind            = CrashKind.None;
        _lastCrashObjectName      = string.Empty;
        _lastCrashTerminalPenalty = -1.0f;
        _lastGoalClearanceFailureType = GoalClearanceFailureType.None;

        // 코디네이터에 전체 스폰 계산 위임 (Evader/Pursuer/Goal 위치 모두 내부에서 처리)
        if (EpisodeSpawnCoordinator.Instance != null)
        {
            EpisodeSpawnCoordinator.Instance.ComputeSpawn();
            transform.position = EpisodeSpawnCoordinator.Instance.GetSpawnPosition(gameObject);
        }
        else
        {
            // 폴백: SpawnCenter 직접 사용
            if (SpawnCenter.Current != null)
            {
                QuerySpawnCenter();
                var range = SpawnCenter.Current.GetEvaderSpawnRange();
                transform.position = SpawnCenter.Current.GetRandomPosition(range);
            }
            else
            {
                transform.position = _spawnCenter;
            }
            // 코디네이터 없을 때만 Goal 수동 랜덤화
            _goalZone?.RandomizePosition();
        }

        transform.position = ClampPositionWithinBoundary(transform.position, _spawnBoundaryInset);

        // Goal이 건물/벽과 과근접이면 재샘플링하여 도달 불가능 배치를 줄인다.
        if (_goalZone != null && _validateGoalPlacement)
            EnsureGoalPlacementClearance();

        ApplyGoalAndSensorHardening();

        if (_goalResampleAttemptsLast > 0)
            Debug.Log($"[EvaderAgent] Goal placement resampled {_goalResampleAttemptsLast} time(s) for obstacle clearance.", this);

        // 에피소드 시작 시 goal 방향으로 yaw 정렬 (goal 없으면 랜덤)
        if (_goalZone != null)
        {
            Vector3 toGoal = _goalZone.GetPosition() - transform.position;
            toGoal.y = 0f;
            transform.rotation = toGoal.sqrMagnitude > 0.01f
                ? Quaternion.LookRotation(toGoal.normalized, Vector3.up)
                : Quaternion.Euler(0f, Random.Range(0f, 360f), 0f);
        }
        else
            transform.rotation = Quaternion.Euler(0f, Random.Range(0f, 360f), 0f);

        // 물리 초기화 — DroneAgent.ResetPhysicsState() 위임 (Rigidbody + DronePhysics 일괄 초기화)
        ResetPhysicsState();

        if (_goalZone != null)
            _lastGoalDistance = Vector3.Distance(transform.position, _goalZone.GetPosition());

        // 체크포인트 생성 (스폰 위치 확정 후)
        if (_goalZone != null && _checkpointCount > 0)
            GenerateCheckpoints(transform.position, _goalZone.GetPosition());
        else
            _episodeCheckpoints = null;
    }

    // ───────── 관측 수집 (VectorObs = 44) ────────────────────────────────
    // 구성: 자기 상태 7 + 목표 4 + 추격자/메모리 7 + DroneSensorSystem 26 = 44
    // 속도 획득: _dronePhysics API 사용 (DroneAgent와 동일 소스)
    public override void CollectObservations(VectorSensor sensor)
    {
        // DroneAgent.IsDroneReady() — null 가드 통일
        if (!IsDroneReady())
        {
            for (int i = 0; i < 44; i++)
                sensor.AddObservation(0f);
            return;
        }

        // 자기 상태 (7) — _dronePhysics API 경유
        Vector3 localVel    = transform.InverseTransformDirection(_dronePhysics.GetVelocity());
        Vector3 localAngVel = _dronePhysics.GetLocalAngularVelocity();
        sensor.AddObservation(localVel    / _maxObsSpeed);             // 3
        sensor.AddObservation(localAngVel / _maxObsSpeed);             // 3
        sensor.AddObservation(transform.position.y / _maxDistance);    // 1

        // 목표 지점 (4) — 월드 좌표계 방향 (v1 warm-start와 동일 좌표계 유지)
        if (_goalZone != null)
        {
            Vector3 toGoal = _goalZone.GetPosition() - transform.position;
            sensor.AddObservation(toGoal.normalized);                  // 3
            sensor.AddObservation(toGoal.magnitude / _maxDistance);    // 1
        }
        else
        {
            sensor.AddObservation(Vector3.zero);                       // 3
            sensor.AddObservation(1f);                                 // 1
        }

        // 추격자 정보 (7) — DroneAgent.TargetTransform 사용, goalOnlyMode 시 전부 0
        bool usePursuerHint = !_goalOnlyMode && _currentStage == 0 && TargetTransform != null;
        if (usePursuerHint)
        {
            Vector3 toPursuer = TargetTransform.position - transform.position;
            Rigidbody pRb     = TargetTransform.GetComponent<Rigidbody>();
            Vector3 relVel    = pRb != null
                ? pRb.linearVelocity - _dronePhysics.GetVelocity()
                : Vector3.zero;
            sensor.AddObservation(toPursuer / _maxDistance);           // 3
            sensor.AddObservation(relVel    / _maxObsSpeed);           // 3
            sensor.AddObservation(_isPursuerVisible ? 1f : 0f);        // 1
        }
        else
        {
            // goalOnlyMode 또는 Stage1+: 메모리 기반 (초기엔 전부 0)
            sensor.AddObservation(_lastKnownPursuerPos / _maxDistance); // 3
            sensor.AddObservation(GetPursuerRelVelNormalized());        // 3
            sensor.AddObservation(_isPursuerVisible ? 1f : 0f);         // 1
        }

        // DroneSensorSystem (26) — 배민우 팀 거리 센서 연동
        if (_sensorSystem != null)
        {
            foreach (float d in _sensorSystem.GetAllNormalizedDistances())
                sensor.AddObservation(d);                               // 26
        }
        else
        {
            for (int i = 0; i < 26; i++)
                sensor.AddObservation(0f);                              // 26 zeros
        }
    }

    // ───────── 행동 처리 ──────────────────────────────────────────────────
    // Heuristic()은 DroneAgent의 QE/WASD 구현을 그대로 사용
    // Action Space: Continuous 4D — thrust [-1,1] (DroneAgent와 동일 범위)
    public override void OnActionReceived(ActionBuffers actions)
    {
        if (_episodeEnded)
            return;

        _episodeTimer += Time.fixedDeltaTime;
        _episodeSteps++;

        float thrustRaw = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
        float rollRaw   = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);
        float pitchRaw  = Mathf.Clamp(actions.ContinuousActions[2], -1f, 1f);
        float yawRaw    = Mathf.Clamp(actions.ContinuousActions[3], -1f, 1f);

        float thrust = PrepareCommand(thrustRaw, ref _smoothedThrust);
        float roll   = PrepareCommand(rollRaw,   ref _smoothedRoll);
        float pitch  = PrepareCommand(pitchRaw,  ref _smoothedPitch);
        float yaw    = PrepareCommand(yawRaw,    ref _smoothedYaw);

        // 26D 레이 거리 배열 전달 (합산 근접 페널티용)
        float[] obstacleDists = _sensorSystem?.GetAllNormalizedDistances();

        ApplyWallProximitySafety(ref roll, ref pitch, ref yaw, obstacleDists);
        ApplyBoundarySafety(ref roll, ref pitch, ref yaw);

        _dronePhysics?.SetCommand(thrust, roll, pitch, yaw);

        EvaderReward.RewardBreakdown breakdown;
        float stepReward = _rewardCalculator.ComputeStepReward(
            agentPos:      transform.position,
            goalPos:       _goalZone       != null ? _goalZone.GetPosition()     : Vector3.zero,
            pursuerPos:    TargetTransform != null ? TargetTransform.position    : Vector3.zero,
            isPursuerVisible: _isPursuerVisible,
            agentVel:      _dronePhysics   != null ? _dronePhysics.GetVelocity() : Vector3.zero,
            obstacleDists: obstacleDists,
            breakdown: out breakdown
        );

        AddReward(stepReward);

        if (_goalZone != null)
        {
            float goalDistance = Vector3.Distance(transform.position, _goalZone.GetPosition());
            ApplyStagnationPenalty(goalDistance);
        }

        if (_logRewardBreakdown && (_episodeSteps % _rewardLogIntervalSteps == 0))
            LogRewardBreakdown(breakdown);

        // 체크포인트 도달 확인 (순서대로, 한 번만)
        CheckCheckpoints();

        CheckTerminationConditions();
    }

    // ───────── 종료 조건 ──────────────────────────────────────────────────
    // 목표 도달은 Goal.OnTriggerEnter → NotifyArrival() → OnGoalArrived() 경로로 처리
    private void CheckTerminationConditions()
    {
        // 1) 포획 (goalOnlyMode=false 시에만)
        if (!_goalOnlyMode && TargetTransform != null &&
            Vector3.Distance(transform.position, TargetTransform.position) < _catchDistance)
        {
            TerminateEpisode(EpisodeLogger.TermType.Captured, -1.0f, true);
            return;
        }

        // 2) 맵 이탈 — 벽 위로 탈출하거나 XZ 경계 초과 시 crash 처리
        Vector3 pos = transform.position;
        if (pos.y > _maxFlightHeight ||
            Mathf.Abs(pos.x) > _boundaryHalfSize ||
            Mathf.Abs(pos.z) > _boundaryHalfSize)
        {
            SetCrash(CrashKind.BoundaryOverflow);
            return;
        }

        // 3) 타임아웃
        if (_episodeTimer >= _maxEpisodeSeconds)
        {
            TerminateEpisode(EpisodeLogger.TermType.Timeout, 0f, false);
        }
    }

    // ───────── 물리 충돌 감지 (자체) ──────────────────────────────────────
    // 이강민 팀 SetCrash() 외부 API와 이중 감지 방지를 위해 _episodeEnded 플래그 사용
    private void OnCollisionEnter(Collision collision)
    {
        if (_episodeEnded) return;

        bool isCrash = _crashTags == null || _crashTags.Length == 0;  // 태그 미지정 → 모든 충돌
        if (!isCrash)
            isCrash = MatchesCrashTag(collision.gameObject);

        if (isCrash)
        {
            CrashKind crashKind = ClassifyCrashObject(collision.gameObject);
            SetCrash(crashKind, collision.gameObject);
        }
    }

    // ───────── Goal 이벤트 ────────────────────────────────────────────
    /// <summary>
    /// Goal.OnTriggerEnter → NotifyArrival() 경로로 호출된다.
    /// 보상 지급, 로그, 에피소드 종료를 처리한다.
    /// </summary>
    private void OnGoalArrived(Goal zone)
    {
        var pursuerAgents = FindObjectsByType<PursuerAgent>(FindObjectsSortMode.None);
        foreach (var pursuerAgent in pursuerAgents)
            pursuerAgent.HandleEvaderGoalRespawn();

        TerminateEpisode(EpisodeLogger.TermType.Goal, _goalArrivalReward, true);
    }

    private void OnDestroy()
    {
        if (_goalZone != null)
            _goalZone.OnArrival -= OnGoalArrived;
    }

    // ───────── 외부 API ───────────────────────────────────────────────────
    /// <summary>World 팀에서 충돌/추락 감지 시 호출 (내부 OnCollisionEnter도 이 메서드로 수렴)</summary>
    public void SetCrash()
    {
        SetCrash(CrashKind.Other, null);
    }

    private void SetCrash(CrashKind crashKind)
    {
        SetCrash(crashKind, null);
    }

    private void SetCrash(CrashKind crashKind, GameObject hitObject)
    {
        _lastCrashKind = crashKind == CrashKind.None ? CrashKind.Other : crashKind;
        _lastCrashObjectName = hitObject != null ? hitObject.name : string.Empty;
        _lastCrashTerminalPenalty = ResolveCrashTerminalPenalty(_lastCrashKind);
        TerminateEpisode(EpisodeLogger.TermType.Crash, _lastCrashTerminalPenalty, true);
    }

    private float ResolveCrashTerminalPenalty(CrashKind crashKind)
    {
        return crashKind switch
        {
            CrashKind.BoundaryOverflow => _terminalPenaltyBoundaryOverflow,
            CrashKind.OuterWall => _terminalPenaltyOuterWall,
            CrashKind.BuildingOrObstacle => _terminalPenaltyBuilding,
            _ => _terminalPenaltyOtherCrash,
        };
    }

    /// <summary>Pursuer가 capture에 성공했을 때 외부에서 호출</summary>
    public void SetCaptured()
    {
        TerminateEpisode(EpisodeLogger.TermType.Captured, -1.0f, true);
    }

    /// <summary>Sensor 팀에서 LOS 결과를 매 프레임 갱신할 때 호출</summary>
    public void SetPursuerVisibility(bool isVisible, Vector3 pursuerWorldPos)
    {
        _isPursuerVisible = isVisible;
        if (isVisible)
        {
            _lastKnownPursuerPos      = pursuerWorldPos;
            _timeSincePursuerDetected = 0f;
        }
        else
        {
            _timeSincePursuerDetected += Time.deltaTime;
        }
    }

    private void TerminateEpisode(EpisodeLogger.TermType termType, float terminalReward, bool applyTerminalReward)
    {
        if (_episodeEnded)
            return;

        _episodeEnded = true;

        if (applyTerminalReward)
            AddReward(terminalReward);

        _episodeLogger?.LogEpisode(termType, _episodeSteps);
        PushTerminationStats(termType);
        EndEpisode();
    }

    private void PushTerminationStats(EpisodeLogger.TermType termType)
    {
        Academy academy = Academy.Instance;
        if (academy == null)
            return;

        var stats = academy.StatsRecorder;
        bool isCrash = termType == EpisodeLogger.TermType.Crash;
        bool isBoundaryCrash = isCrash && _lastCrashKind == CrashKind.BoundaryOverflow;
        bool isOuterWallCrash = isCrash && _lastCrashKind == CrashKind.OuterWall;
        bool isBuildingCrash = isCrash && _lastCrashKind == CrashKind.BuildingOrObstacle;
        bool isOtherCrash = isCrash && _lastCrashKind == CrashKind.Other;

        stats.Add("Diagnostics/TermGoal",     termType == EpisodeLogger.TermType.Goal ? 1f : 0f);
        stats.Add("Diagnostics/TermTimeout",  termType == EpisodeLogger.TermType.Timeout ? 1f : 0f);
        stats.Add("Diagnostics/TermCrash",    termType == EpisodeLogger.TermType.Crash ? 1f : 0f);
        stats.Add("Diagnostics/TermCaptured", termType == EpisodeLogger.TermType.Captured ? 1f : 0f);
        stats.Add("Diagnostics/TermCrashBoundaryOverflow", isBoundaryCrash ? 1f : 0f);
        stats.Add("Diagnostics/TermCrashOuterWall", isOuterWallCrash ? 1f : 0f);
        stats.Add("Diagnostics/TermCrashBuilding", isBuildingCrash ? 1f : 0f);
        stats.Add("Diagnostics/TermCrashOther", isOtherCrash ? 1f : 0f);
        stats.Add("Diagnostics/TermCrashType", isCrash ? (float)_lastCrashKind : 0f);
        stats.Add("Diagnostics/TermCrashPenalty", isCrash ? _lastCrashTerminalPenalty : 0f);
        stats.Add("Diagnostics/TermCrashObjectKnown", isCrash && !string.IsNullOrEmpty(_lastCrashObjectName) ? 1f : 0f);
        stats.Add("Diagnostics/TermEpisodeSteps", _episodeSteps);

        if (_goalZone != null)
            stats.Add("Diagnostics/TermGoalDistance", Vector3.Distance(transform.position, _goalZone.GetPosition()));

        if (isCrash)
        {
            _lastCrashKind = CrashKind.None;
            _lastCrashObjectName = string.Empty;
            _lastCrashTerminalPenalty = -1.0f;
        }
    }

    // ───────── SpawnCenter 쿼리 API ─────────────────────────────────────────

    /// <summary>SpawnCenter에서 도망자 스폰 범위를 쿼리하여 내부 필드에 저장한다.</summary>
    public void QuerySpawnCenter()
    {
        if (SpawnCenter.Current != null)
        {
            var range = SpawnCenter.Current.GetEvaderSpawnRange();
            _queriedMinY   = range.MinY;
            _queriedMaxY   = range.MaxY;
            _queriedRadius = range.Radius;
        }
    }

    /// <summary>쿼리된 스폰 범위 (읽기 전용)</summary>
    public float QueriedMinY   => _queriedMinY;
    public float QueriedMaxY   => _queriedMaxY;
    public float QueriedRadius => _queriedRadius;

    // ───────── Editor 디버그 기즈모 ──────────────────────────────────────
#if UNITY_EDITOR
    private void OnDrawGizmosSelected()
    {
        // SpawnCenter가 있으면 기즈모는 SpawnCenter에서 표시하므로 여기선 생략
        if (SpawnCenter.Current != null) return;

        // SpawnCenter 없을 때 — 초기 위치 마커만 표시
        Vector3 center = Application.isPlaying ? _spawnCenter : transform.position;
        Gizmos.color = new Color(0f, 0.5f, 1f, 0.9f);
        Gizmos.DrawSphere(center, 0.4f);

        // 체크포인트 시각화 (Play 모드에서만)
        if (!Application.isPlaying || _episodeCheckpoints == null) return;
        for (int i = 0; i < _episodeCheckpoints.Length; i++)
        {
            bool reached = i < _nextCheckpointIdx;
            Gizmos.color = reached
                ? new Color(0.3f, 1f, 0.3f, 0.5f)   // 도달: 초록
                : new Color(1f,   0.8f, 0f,  0.7f);  // 미도달: 노랑
            Gizmos.DrawWireSphere(_episodeCheckpoints[i], _checkpointRadius);
        }
    }
#endif

    // ───────── 체크포인트 ────────────────────────────────────────────────
    /// <summary>
    /// 스폰~Goal 직선 위에 체크포인트를 균등 배치한다.
    /// 각 후보 위치가 장애물 내부이면 주변 8방향으로 최대 4회 탐색해 빈 공간을 찾는다.
    /// </summary>
    private void GenerateCheckpoints(Vector3 startPos, Vector3 goalPos)
    {
        _episodeCheckpoints = new Vector3[_checkpointCount];
        _nextCheckpointIdx  = 0;

        for (int i = 0; i < _checkpointCount; i++)
        {
            float   t         = (i + 1f) / (_checkpointCount + 1f);   // 1/3, 2/3
            Vector3 candidate = Vector3.Lerp(startPos, goalPos, t);
            candidate.y       = startPos.y;  // 드론 스폰 고도 유지
            _episodeCheckpoints[i] = FindOpenCheckpoint(candidate);
        }
    }

    /// <summary>
    /// 후보 위치에 장애물이 없으면 그대로 반환.
    /// 있으면 주변 8방향 × 4 거리 단계로 열린 위치를 탐색한다.
    /// </summary>
    private Vector3 FindOpenCheckpoint(Vector3 candidate)
    {
        if (!IsCheckpointBlocked(candidate)) return candidate;

        for (int radius = 1; radius <= 4; radius++)
        {
            for (int a = 0; a < 8; a++)
            {
                float   angle = a * 45f * Mathf.Deg2Rad;
                Vector3 pos   = candidate + new Vector3(
                    Mathf.Sin(angle) * radius * 3f, 0f,
                    Mathf.Cos(angle) * radius * 3f);
                if (!IsCheckpointBlocked(pos)) return pos;
            }
        }
        return candidate;  // 폴백: 장애물 내부여도 원래 위치 사용
    }

    /// <summary>해당 위치가 Building/Wall 태그 오브젝트 내부인지 확인한다.</summary>
    private bool IsCheckpointBlocked(Vector3 pos)
    {
        Collider[] hits = Physics.OverlapSphere(pos, _checkpointClearRadius, ~0, QueryTriggerInteraction.Ignore);
        foreach (var c in hits)
            if (IsObstacleCollider(c))
                return true;
        return false;
    }

    /// <summary>
    /// Goal 중심 주변의 Building/Wall 충돌을 검사해 과근접이면 재배치한다.
    /// EpisodeSpawnCoordinator 사용 여부와 무관하게 Goal.RandomizePosition()을 호출해
    /// SpawnCenter 제약 내에서 유효 Goal을 찾는다.
    /// </summary>
    private void EnsureGoalPlacementClearance()
    {
        _goalResampleAttemptsLast = 0;
        if (_goalZone == null) return;

        bool isClear = false;
        bool usedFallback = false;
        GoalClearanceFailureType failureType = GoalClearanceFailureType.None;

        for (int i = 0; i < _goalResampleMaxAttempts; i++)
        {
            Vector3 goalPos = _goalZone.GetPosition();
            if (!IsGoalTooCloseToObstacle(goalPos, out failureType))
            {
                isClear = true;
                break;
            }

            _goalZone.RandomizePosition();
            _goalResampleAttemptsLast++;
        }

        if (!isClear)
        {
            // dense obstacle 배치에서도 최소한 "건물 내부 goal"만은 피하도록 완화 재시도
            usedFallback = TryForceGoalOutsideObstacle();
            if (usedFallback)
                isClear = !IsGoalTooCloseToObstacle(_goalZone.GetPosition(), out failureType);
        }

        bool insideObstacle = IsGoalInsideObstacle(_goalZone.GetPosition());
        if (!isClear && failureType == GoalClearanceFailureType.None)
            failureType = insideObstacle ? GoalClearanceFailureType.InsideObstacle : GoalClearanceFailureType.ObstacleProximity;

        _lastGoalClearanceFailureType = isClear ? GoalClearanceFailureType.None : failureType;
        ReportGoalPlacementDiagnostics(isClear, usedFallback, insideObstacle);

        if (!isClear && _goalResampleAttemptsLast > 0)
        {
            Debug.LogWarning(
                $"[EvaderAgent] Goal clearance validation failed after {_goalResampleAttemptsLast} attempts. " +
                "Proceeding with current goal position.",
                this);
        }
    }

    /// <summary>
    /// Goal 주변에 Building/Wall 태그가 있으면 true를 반환한다.
    /// </summary>
    private bool IsGoalTooCloseToObstacle(Vector3 goalPos, out GoalClearanceFailureType failureType)
    {
        failureType = GoalClearanceFailureType.None;

        if (IsGoalTooCloseToBoundary(goalPos))
        {
            failureType = GoalClearanceFailureType.BoundaryMargin;
            return true;
        }

        float clearanceSqr = _goalObstacleClearanceRadius * _goalObstacleClearanceRadius;
        Collider[] hits = Physics.OverlapSphere(goalPos, _goalObstacleClearanceRadius, ~0, QueryTriggerInteraction.Ignore);
        foreach (var c in hits)
        {
            if (!IsObstacleCollider(c))
                continue;

            if (IsPointInsideObstacleCollider(goalPos, c))
            {
                failureType = GoalClearanceFailureType.InsideObstacle;
                return true;
            }

            Vector3 closest = c.ClosestPoint(goalPos);
            if ((closest - goalPos).sqrMagnitude < clearanceSqr)
            {
                failureType = GoalClearanceFailureType.ObstacleProximity;
                return true;
            }
        }

        if (HasGoalEnvelopeCollision(goalPos))
        {
            failureType = GoalClearanceFailureType.EnvelopeCollision;
            return true;
        }

        return false;
    }

    private bool IsGoalTooCloseToBoundary(Vector3 goalPos)
    {
        if (_boundaryHalfSize <= 1f || _goalBoundaryClearanceMargin <= 0f)
            return false;

        float margin = Mathf.Clamp(_goalBoundaryClearanceMargin, 0f, Mathf.Max(0.5f, _boundaryHalfSize - 0.5f));
        float safeLimit = _boundaryHalfSize - margin;
        return Mathf.Abs(goalPos.x) > safeLimit || Mathf.Abs(goalPos.z) > safeLimit;
    }

    private bool IsObstacleCollider(Collider collider)
    {
        if (collider == null || collider.isTrigger)
            return false;

        if (_goalZone != null && IsSameOrChildOf(collider.transform, _goalZone.transform))
            return false;

        return IsObstacleObject(collider.gameObject);
    }

    private bool IsPointInsideObstacleCollider(Vector3 point, Collider collider)
    {
        if (collider == null)
            return false;

        Vector3 closest = collider.ClosestPoint(point);
        return (closest - point).sqrMagnitude <= 0.000001f;
    }

    private bool HasGoalEnvelopeCollision(Vector3 goalPos)
    {
        const float probeStartOffset = 0.05f;
        float probeDistance = Mathf.Max(0.8f, _goalObstacleClearanceRadius * 0.45f);
        float probeRadius = Mathf.Max(0.2f, _goalObstacleClearanceRadius * 0.10f);

        foreach (Vector3 dir in GoalClearanceProbeDirections)
        {
            Vector3 origin = goalPos + dir * probeStartOffset;
            if (Physics.Raycast(origin, dir, out RaycastHit hit, probeDistance, ~0, QueryTriggerInteraction.Ignore) &&
                IsObstacleCollider(hit.collider))
            {
                return true;
            }

            Vector3 samplePos = goalPos + dir * probeDistance;
            Collider[] sampleHits = Physics.OverlapSphere(samplePos, probeRadius, ~0, QueryTriggerInteraction.Ignore);
            foreach (var c in sampleHits)
            {
                if (IsObstacleCollider(c))
                    return true;
            }
        }

        return false;
    }

    private bool IsGoalInsideObstacle(Vector3 goalPos)
    {
        Collider[] hits = Physics.OverlapSphere(goalPos, 0.2f, ~0, QueryTriggerInteraction.Ignore);
        foreach (var c in hits)
        {
            if (IsObstacleCollider(c) && IsPointInsideObstacleCollider(goalPos, c))
                return true;
        }

        return false;
    }

    private bool TryForceGoalOutsideObstacle()
    {
        if (_goalZone == null)
            return false;

        for (int i = 0; i < _goalResampleMaxAttempts; i++)
        {
            if (!IsGoalInsideObstacle(_goalZone.GetPosition()))
                return true;

            _goalZone.RandomizePosition();
            _goalResampleAttemptsLast++;
        }

        return !IsGoalInsideObstacle(_goalZone.GetPosition());
    }

    /// <summary>
    /// 다음 체크포인트와의 거리를 확인해 도달 시 보상 지급 (순서 강제).
    /// </summary>
    private void CheckCheckpoints()
    {
        if (_episodeCheckpoints == null) return;
        if (_nextCheckpointIdx >= _episodeCheckpoints.Length) return;

        float dist = Vector3.Distance(transform.position,
                                      _episodeCheckpoints[_nextCheckpointIdx]);
        if (dist < _checkpointRadius)
        {
            float baseReward = Mathf.Min(_checkpointReward, _checkpointRewardCap);
            float decayMul   = Mathf.Pow(_checkpointRewardDecay, _nextCheckpointIdx);
            float reward     = baseReward * decayMul;
            AddReward(reward);
            _nextCheckpointIdx++;
        }
    }

    private bool MatchesCrashTag(GameObject hitObject)
    {
        if (hitObject == null)
            return false;

        Transform cursor = hitObject.transform;
        while (cursor != null)
        {
            if (MatchesAnyTag(cursor.gameObject, _crashTags) || IsObstacleObject(cursor.gameObject))
                return true;

            cursor = cursor.parent;
        }

        return false;
    }

    private CrashKind ClassifyCrashObject(GameObject hitObject)
    {
        if (hitObject == null)
            return CrashKind.Other;

        Transform hitTransform = hitObject.transform;
        if (IsLikelyOuterWall(hitTransform))
            return CrashKind.OuterWall;

        if (IsLikelyBuilding(hitTransform))
            return CrashKind.BuildingOrObstacle;

        return IsObstacleObject(hitObject)
            ? CrashKind.BuildingOrObstacle
            : CrashKind.Other;
    }

    private bool IsLikelyOuterWall(Transform cursor)
    {
        if (TransformContainsToken(cursor, OuterWallNameTokens))
            return true;

        while (cursor != null)
        {
            if (SafeCompareTag(cursor.gameObject, "Wall"))
                return true;

            string layerName = LayerMask.LayerToName(cursor.gameObject.layer);
            if (!string.IsNullOrEmpty(layerName))
            {
                string lowered = layerName.ToLowerInvariant();
                if (lowered.Contains("wall") || lowered.Contains("boundary"))
                    return true;
            }

            cursor = cursor.parent;
        }

        return false;
    }

    private bool IsLikelyBuilding(Transform cursor)
    {
        if (TransformContainsToken(cursor, BuildingNameTokens))
            return true;

        while (cursor != null)
        {
            if (SafeCompareTag(cursor.gameObject, "Building"))
                return true;

            cursor = cursor.parent;
        }

        return false;
    }

    private static bool TransformContainsToken(Transform cursor, string[] tokens)
    {
        while (cursor != null)
        {
            string lowered = cursor.name.ToLowerInvariant();
            foreach (string token in tokens)
            {
                if (lowered.Contains(token))
                    return true;
            }

            cursor = cursor.parent;
        }

        return false;
    }

    // ───────── 헬퍼 ──────────────────────────────────────────────────────
    private void ApplyWallProximitySafety(ref float roll, ref float pitch, ref float yaw, float[] obstacleDists)
    {
        if (!_enableWallProximitySafety || obstacleDists == null || obstacleDists.Length < 17)
            return;

        float threshold = _wallProximityThreshold;
        float frontDist = obstacleDists[9];
        float frontRightDist = obstacleDists[10];
        float frontLeftDist = obstacleDists[16];

        // Top-Middle 레이(1,2,8)를 함께 참조해 상부 모서리/사선 벽 접근도 포착한다.
        float topFrontDist = obstacleDists.Length > 1 ? obstacleDists[1] : 0f;
        float topFrontRightDist = obstacleDists.Length > 2 ? obstacleDists[2] : 0f;
        float topFrontLeftDist = obstacleDists.Length > 8 ? obstacleDists[8] : 0f;

        float frontRisk = Mathf.Max(DistanceToRisk(frontDist, threshold), DistanceToRisk(topFrontDist, threshold));
        float rightRisk = Mathf.Max(DistanceToRisk(frontRightDist, threshold), DistanceToRisk(topFrontRightDist, threshold));
        float leftRisk = Mathf.Max(DistanceToRisk(frontLeftDist, threshold), DistanceToRisk(topFrontLeftDist, threshold));

        if (frontRisk <= 0f && rightRisk <= 0f && leftRisk <= 0f)
        {
            _wallForwardBlockLatched = false;
            _wallAvoidBiasState = Mathf.Lerp(_wallAvoidBiasState, 0f, 0.2f);
            return;
        }

        bool blockedForward = false;
        if (_blockForwardPitchOnWallProximity)
        {
            if (!_wallForwardBlockLatched && frontRisk >= _wallForwardBlockEnterRisk)
                _wallForwardBlockLatched = true;
            else if (_wallForwardBlockLatched && frontRisk <= _wallForwardBlockReleaseRisk)
                _wallForwardBlockLatched = false;

            if (pitch > 0f && frontRisk > 0f)
            {
                // 1) 먼저 전진을 감쇠한다.
                float dampT = Mathf.InverseLerp(0f, Mathf.Max(0.05f, _wallForwardBlockEnterRisk), frontRisk);
                pitch *= Mathf.Lerp(1f, 0.10f, dampT);

                // 2) 고위험에서는 latch 기반으로 완전 전진 차단.
                if (_wallForwardBlockLatched)
                {
                    pitch = 0f;
                    blockedForward = true;
                }
            }
        }

        float reverseAssist = 0f;
        if (frontRisk > _wallReverseAssistStartRisk)
        {
            // 3) 완전 역추진 대신 제한된 역피치만 허용한다.
            reverseAssist = Mathf.InverseLerp(_wallReverseAssistStartRisk, 1f, frontRisk);
            float reversePitchCap = Mathf.Lerp(0f, _wallReverseAssistMaxPitch, reverseAssist);
            pitch = Mathf.Min(pitch, -reversePitchCap);
        }

        // 4) 좌/우 회피 바이어스를 평활해 지그재그(고주파 진동)를 줄인다.
        float avoidBiasTarget = Mathf.Clamp(leftRisk - rightRisk, -1f, 1f);
        if (Mathf.Abs(avoidBiasTarget) < 0.02f && frontRisk > 0.40f)
        {
            float leftClear = frontLeftDist > 0f ? frontLeftDist : threshold;
            float rightClear = frontRightDist > 0f ? frontRightDist : threshold;
            float clearanceDelta = Mathf.Clamp((rightClear - leftClear) / Mathf.Max(0.001f, threshold), -1f, 1f);
            avoidBiasTarget = clearanceDelta * _wallSymmetricClearanceBiasScale;
        }

        _wallAvoidBiasState = Mathf.Lerp(_wallAvoidBiasState, avoidBiasTarget, _wallAvoidBiasSmoothing);
        float avoidBias = Mathf.Clamp(_wallAvoidBiasState, -1f, 1f);

        float riskScale = Mathf.Clamp01(Mathf.Max(frontRisk, Mathf.Max(leftRisk, rightRisk)));
        float assistScale = Mathf.Lerp(0.45f, 1f, riskScale);

        roll = Mathf.Clamp(roll + avoidBias * _wallAvoidRollAssist * assistScale, -1f, 1f);
        yaw = Mathf.Clamp(yaw + avoidBias * _wallAvoidYawAssist * assistScale, -1f, 1f);

        Academy academy = Academy.Instance;
        if (academy != null)
        {
            academy.StatsRecorder.Add("Diagnostics/WallSafetyActivated", 1f);
            if (blockedForward)
                academy.StatsRecorder.Add("Diagnostics/WallSafetyForwardBlocked", 1f);
            academy.StatsRecorder.Add("Diagnostics/WallSafetyFrontRisk", frontRisk);
            academy.StatsRecorder.Add("Diagnostics/WallSafetyAvoidBias", Mathf.Abs(avoidBias));
            if (reverseAssist > 0f)
                academy.StatsRecorder.Add("Diagnostics/WallSafetyReverseAssist", reverseAssist);
        }
    }

    private void ApplyBoundarySafety(ref float roll, ref float pitch, ref float yaw)
    {
        if (!_enableBoundarySafety || _boundaryHalfSize <= 1f)
            return;

        Vector3 pos = transform.position;
        float margin = Mathf.Clamp(_boundarySafetyMargin, 1f, Mathf.Max(2f, _boundaryHalfSize - 0.5f));
        float warningBoundary = Mathf.Max(0.5f, _boundaryHalfSize - margin);

        float xRisk = Mathf.InverseLerp(warningBoundary, _boundaryHalfSize, Mathf.Abs(pos.x));
        float zRisk = Mathf.InverseLerp(warningBoundary, _boundaryHalfSize, Mathf.Abs(pos.z));
        float boundaryRisk = Mathf.Clamp01(Mathf.Max(xRisk, zRisk));

        if (boundaryRisk <= 0f)
            return;

        Vector3 toCenter = new Vector3(-pos.x, 0f, -pos.z);
        if (toCenter.sqrMagnitude < 0.0001f)
            return;

        Vector3 toCenterDir = toCenter.normalized;
        Vector3 localToCenter = transform.InverseTransformDirection(toCenterDir);

        float outwardSpeed = 0f;
        if (_dronePhysics != null)
        {
            Vector3 velocity = _dronePhysics.GetVelocity();
            outwardSpeed = Mathf.Max(0f, Vector3.Dot(velocity, -toCenterDir));
        }

        float outwardRisk = Mathf.Clamp01(outwardSpeed / Mathf.Max(0.1f, _boundaryOutwardSpeedReference));
        float effectiveRisk = Mathf.Clamp01(Mathf.Max(boundaryRisk, Mathf.Lerp(boundaryRisk, 1f, outwardRisk * 0.55f)));

        float assistScale = Mathf.Lerp(0.40f, 1f, effectiveRisk);
        float inwardSide = Mathf.Clamp(localToCenter.x, -1f, 1f);

        roll = Mathf.Clamp(roll + inwardSide * _boundaryInwardRollAssist * assistScale, -1f, 1f);
        yaw = Mathf.Clamp(yaw + inwardSide * _boundaryInwardYawAssist * assistScale, -1f, 1f);

        if (pitch > 0f)
        {
            // 진행 방향이 중심 반대(바깥쪽)일수록 경계 근접에서 전진 입력을 더 강하게 감쇠한다.
            float outwardHeading = Mathf.Max(0f, Vector3.Dot(transform.forward, -toCenterDir));
            if (outwardHeading > 0f)
            {
                float damp = Mathf.Lerp(1f, _boundaryForwardPitchMinScale, effectiveRisk * outwardHeading);
                pitch *= damp;

                if (effectiveRisk >= _boundaryEmergencyRiskThreshold && outwardHeading > 0.20f)
                {
                    float emergencyScale = Mathf.InverseLerp(_boundaryEmergencyRiskThreshold, 1f, effectiveRisk);
                    pitch = Mathf.Min(pitch, 0f);
                    roll = Mathf.Clamp(roll + inwardSide * _boundaryEmergencyRollAssist * emergencyScale, -1f, 1f);
                    yaw = Mathf.Clamp(yaw + inwardSide * _boundaryEmergencyYawAssist * emergencyScale, -1f, 1f);
                }
            }
        }

        float boundaryPenalty = _boundaryRiskPenaltyPerStep * effectiveRisk;
        if (boundaryPenalty < 0f)
            AddReward(boundaryPenalty);

        Academy academy = Academy.Instance;
        if (academy != null)
        {
            academy.StatsRecorder.Add("Diagnostics/BoundarySafetyActivated", 1f);
            academy.StatsRecorder.Add("Diagnostics/BoundaryRisk", boundaryRisk);
            academy.StatsRecorder.Add("Diagnostics/BoundaryEffectiveRisk", effectiveRisk);
            academy.StatsRecorder.Add("Diagnostics/BoundaryOutwardSpeed", outwardSpeed);
            if (boundaryPenalty < 0f)
                academy.StatsRecorder.Add("Diagnostics/BoundaryRiskPenalty", -boundaryPenalty);
        }
    }

    private Vector3 ClampPositionWithinBoundary(Vector3 position, float inset)
    {
        if (_boundaryHalfSize <= 1f)
            return position;

        float safeInset = Mathf.Clamp(inset, 0f, Mathf.Max(0.5f, _boundaryHalfSize - 0.5f));
        float safeLimit = _boundaryHalfSize - safeInset;

        position.x = Mathf.Clamp(position.x, -safeLimit, safeLimit);
        position.z = Mathf.Clamp(position.z, -safeLimit, safeLimit);

        if (_maxFlightHeight > 0.5f)
            position.y = Mathf.Min(position.y, _maxFlightHeight - 0.5f);

        return position;
    }

    private static float DistanceToRisk(float normalizedDistance, float threshold)
    {
        if (normalizedDistance <= 0f || normalizedDistance >= threshold)
            return 0f;

        return 1f - (normalizedDistance / Mathf.Max(0.001f, threshold));
    }

    private float PrepareCommand(float rawCommand, ref float smoothedState)
    {
        float target = Mathf.Clamp(rawCommand, -1f, 1f);
        if (Mathf.Abs(target) < _actionDeadzone)
            target = 0f;

        if (!_useActionSmoothing)
        {
            smoothedState = target;
            return target;
        }

        float blended = Mathf.Lerp(smoothedState, target, _commandLerpFactor);
        float delta = Mathf.Clamp(blended - smoothedState, -_maxCommandDeltaPerStep, _maxCommandDeltaPerStep);
        smoothedState = Mathf.Clamp(smoothedState + delta, -1f, 1f);
        return smoothedState;
    }

    private void ReportGoalPlacementDiagnostics(bool clearanceOk, bool usedFallback, bool insideObstacle)
    {
        Academy academy = Academy.Instance;
        if (academy == null)
            return;

        academy.StatsRecorder.Add("Diagnostics/GoalResampleAttempts", _goalResampleAttemptsLast);
        academy.StatsRecorder.Add("Diagnostics/GoalResampleFallback", usedFallback ? 1f : 0f);
        academy.StatsRecorder.Add("Diagnostics/GoalPlacementInvalid", clearanceOk ? 0f : 1f);
        academy.StatsRecorder.Add("Diagnostics/GoalInsideObstacle", insideObstacle ? 1f : 0f);
        academy.StatsRecorder.Add("Diagnostics/GoalClearanceFailureType", (float)_lastGoalClearanceFailureType);
        academy.StatsRecorder.Add("Diagnostics/GoalClearanceFailInside", _lastGoalClearanceFailureType == GoalClearanceFailureType.InsideObstacle ? 1f : 0f);
        academy.StatsRecorder.Add("Diagnostics/GoalClearanceFailProximity", _lastGoalClearanceFailureType == GoalClearanceFailureType.ObstacleProximity ? 1f : 0f);
        academy.StatsRecorder.Add("Diagnostics/GoalClearanceFailEnvelope", _lastGoalClearanceFailureType == GoalClearanceFailureType.EnvelopeCollision ? 1f : 0f);
        academy.StatsRecorder.Add("Diagnostics/GoalClearanceFailBoundary", _lastGoalClearanceFailureType == GoalClearanceFailureType.BoundaryMargin ? 1f : 0f);
    }

    private bool IsObstacleObject(GameObject candidate)
    {
        if (candidate == null)
            return false;

        if (MatchesAnyTag(candidate, _crashTags) || MatchesAnyTag(candidate, DefaultObstacleTags))
            return true;

        if (IsLikelyNonObstacle(candidate.transform))
            return false;

        if (IsLikelyObstacleByName(candidate.transform))
            return true;

        string layerName = LayerMask.LayerToName(candidate.layer);
        if (!string.IsNullOrEmpty(layerName))
        {
            string lowered = layerName.ToLowerInvariant();
            if (lowered.Contains("building") || lowered.Contains("wall"))
                return true;
        }

        return false;
    }

    private bool MatchesAnyTag(GameObject candidate, string[] tags)
    {
        if (candidate == null || tags == null || tags.Length == 0)
            return false;

        foreach (string tag in tags)
        {
            if (string.IsNullOrWhiteSpace(tag))
                continue;

            if (SafeCompareTag(candidate, tag))
                return true;
        }

        return false;
    }

    private bool IsLikelyObstacleByName(Transform cursor)
    {
        while (cursor != null)
        {
            string lowered = cursor.name.ToLowerInvariant();
            foreach (string token in ObstacleNameTokens)
            {
                if (lowered.Contains(token))
                    return true;
            }

            cursor = cursor.parent;
        }

        return false;
    }

    private bool IsLikelyNonObstacle(Transform cursor)
    {
        while (cursor != null)
        {
            string lowered = cursor.name.ToLowerInvariant();
            foreach (string token in NonObstacleNameTokens)
            {
                if (lowered.Contains(token))
                    return true;
            }

            cursor = cursor.parent;
        }

        return false;
    }

    private static bool SafeCompareTag(GameObject target, string tag)
    {
        if (target == null || string.IsNullOrWhiteSpace(tag))
            return false;

        try
        {
            return target.CompareTag(tag);
        }
        catch (UnityException)
        {
            return false;
        }
    }

    private static bool IsSameOrChildOf(Transform candidate, Transform potentialAncestor)
    {
        if (candidate == null || potentialAncestor == null)
            return false;

        Transform cursor = candidate;
        while (cursor != null)
        {
            if (cursor == potentialAncestor)
                return true;

            cursor = cursor.parent;
        }

        return false;
    }

    private void ApplyStagnationPenalty(float currentGoalDistance)
    {
        if (!_enableStagnationPenalty || _goalZone == null)
            return;

        if (currentGoalDistance <= _stagnationWatchDistance)
        {
            _stagnationStepCount = 0;
            _lastGoalDistance = currentGoalDistance;
            return;
        }

        if (_lastGoalDistance >= float.MaxValue * 0.5f)
        {
            _lastGoalDistance = currentGoalDistance;
            return;
        }

        float goalProgress = _lastGoalDistance - currentGoalDistance;
        if (goalProgress > _stagnationProgressThreshold)
            _stagnationStepCount = 0;
        else
            _stagnationStepCount++;

        if (_stagnationStepCount > _stagnationGraceSteps)
        {
            AddReward(_stagnationPenaltyPerStep);
            Academy academy = Academy.Instance;
            if (academy != null)
                academy.StatsRecorder.Add("Diagnostics/StagnationPenalty", 1f);
        }

        _lastGoalDistance = currentGoalDistance;
    }

    private Vector3 GetPursuerRelVelNormalized()
    {
        if (TargetTransform == null || !IsDroneReady()) return Vector3.zero;
        Rigidbody pRb = TargetTransform.GetComponent<Rigidbody>();
        return pRb != null
            ? (pRb.linearVelocity - _dronePhysics.GetVelocity()) / _maxObsSpeed
            : Vector3.zero;
    }

    private void LogRewardBreakdown(EvaderReward.RewardBreakdown b)
    {
        string dominantPenalty = b.DominantPenalty();
        Debug.Log(
            $"[RewardBreakdown] epStep={_episodeSteps} total={b.Total:+0.000;-0.000} " +
            $"goal={b.GoalShaping:+0.000;-0.000} vel={b.VelocityAlign:+0.000;-0.000} " +
            $"obs={b.ObstaclePenalty:+0.000;-0.000}(sum={b.ObstacleSum:0.00}) " +
            $"height={b.HeightPenalty:+0.000;-0.000} time={b.TimePenalty:+0.000;-0.000} " +
            $"dist={b.GoalDistance:0.00} nearGoal={b.InGoalPriorityZone} penaltyCause={dominantPenalty}",
            this);
    }

    private EvaderReward.Stage1RewardProfile ResolveStage1RewardProfile()
    {
        if (!_useStage1RewardPreset)
            return _stage1RewardProfile;

        return _stage1RewardPreset == Stage1RewardPreset.Experimental
            ? _stage1RewardProfileExperimental
            : _stage1RewardProfileLegacy;
    }

    private void LogActiveStage1RewardProfile(EvaderReward.Stage1RewardProfile profile)
    {
        string profileName = _useStage1RewardPreset
            ? _stage1RewardPreset.ToString()
            : "Manual";

        Debug.Log(
            $"[EvaderAgent] Stage1 reward profile active: {profileName} " +
            $"(velAlign={profile.VelAlignCoeff:0.###}, goalPriority={profile.GoalPriorityDist:0.###}, " +
            $"proximity={profile.ProximityCoeff:0.###}@{profile.ProximityThreshold:0.###}, " +
            $"nearGoalObsScale={profile.NearGoalObstaclePenaltyScale:0.###}, " +
            $"approachObsScale={profile.GoalApproachObstaclePenaltyScale:0.###}, " +
            $"timePenalty={profile.TimePenaltyPerStep:0.####}, " +
            $"survival={profile.SurvivalRewardPerStep:0.####}, " +
            $"nearGoalShaping={profile.NearGoalShapingMultiplier:0.###})",
            this);
    }

    /// <summary>
    /// Goal 오인식과 하향 레이 과민 반응을 줄이기 위한 런타임 하드닝.
    /// </summary>
    private void ApplyGoalAndSensorHardening()
    {
        int ignoreRaycastLayer = LayerMask.NameToLayer("Ignore Raycast");

        if (_autoSetGoalIgnoreRaycastLayer && _goalZone != null && ignoreRaycastLayer >= 0)
        {
            GameObject goalObject = _goalZone.gameObject;
            if (goalObject.layer != ignoreRaycastLayer)
                goalObject.layer = ignoreRaycastLayer;
        }

        if (_sensorSystem == null)
            return;

        if (_excludeIgnoreRaycastFromSensorMask && ignoreRaycastLayer >= 0)
        {
            int maskWithoutIgnoreRaycast = _sensorSystem.DetectionLayerMask.value & ~(1 << ignoreRaycastLayer);
            _sensorSystem.DetectionLayerMask = maskWithoutIgnoreRaycast;
        }

        if (_forceNonEmptyDetectionLayerMask && _sensorSystem.DetectionLayerMask.value == 0)
        {
            int fallbackMask = ~0;
            if (ignoreRaycastLayer >= 0)
                fallbackMask &= ~(1 << ignoreRaycastLayer);

            _sensorSystem.DetectionLayerMask = fallbackMask;
            Debug.LogWarning("[EvaderAgent] DetectionLayerMask was empty and has been reset to a fallback mask.", this);
        }

        if (_currentStage >= 1)
        {
            if (_applyStage1RewardProfile && _rewardCalculator != null)
            {
                EvaderReward.Stage1RewardProfile activeProfile = ResolveStage1RewardProfile();
                _rewardCalculator.ApplyStage1Profile(activeProfile);

                if (_logActiveStage1RewardProfile && !_hasLoggedStage1RewardProfile)
                {
                    LogActiveStage1RewardProfile(activeProfile);
                    _hasLoggedStage1RewardProfile = true;
                }
            }

            if (_disableMiddleBottomRaysInStage1)
                _sensorSystem.SetMiddleBottomMode(DroneSensorSystem.DiagonalLayerMode.Off);

            if (_disableBottomRayInStage1)
                _sensorSystem.SetBottomEnabled(false);
        }
    }
}
