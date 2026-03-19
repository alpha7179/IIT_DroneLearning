using UnityEngine;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

/// <summary>
/// EvaderAgent — DroneAgent 기반 회피 드론 에이전트
///
/// DroneAgent에서 그대로 사용:
///   - _dronePhysics, _sensorSystem (Awake에서 자동 설정)
///   - GoalTransform, TargetTransform (Inspector 필드)
///   - QE/WASD/화살표 Heuristic
///   - DroneRole.Evader
///
/// EvaderAgent가 추가/오버라이드:
///   - Stage0-A/B 전환 (_goalOnlyMode, _currentStage)
///   - 반경·고도 기반 랜덤 스폰 + GoalTransform 위치 랜덤화
///   - 18-obs 관측 (자기 상태 7 + 목표 4 + 추격자/메모리 7)
///   - EvaderReward 위임 보상 + 종료 조건 (목표·포획·타임아웃)
///   - LOS 기반 추격자 메모리 (Stage1+)
///   - 외부 API: SetCrash(), SetPursuerVisibility()
///
/// 담당: 이재왕 (work/evader)
/// </summary>
public class EvaderAgent : DroneAgent
{
    // ───────── Inspector 설정 (Evader 전용) ───────────────────────────────
    // GoalTransform   : DroneAgent.GoalTransform  사용 (목표 지점)
    // TargetTransform : DroneAgent.TargetTransform 사용 (추격 드론)

    [Header("Evader — Episode Settings")]
    [SerializeField] private float _maxEpisodeSeconds = 25f;
    [SerializeField] private float _catchDistance     = 1.5f;
    [SerializeField] private float _goalDistance      = 2.0f;

    [Header("Evader — Stage Control")]
    [Tooltip("true = Stage0-A (Pursuer 없음, 순수 목표 도달 학습)")]
    [SerializeField] private bool _goalOnlyMode  = true;
    [SerializeField] private int  _currentStage  = 0;

    [Header("Evader — Spawn Randomization")]
    [SerializeField] private float _spawnRadius         = 10f;
    [SerializeField] private float _spawnAltitude       = 5f;
    [SerializeField] private float _goalRandomizeRadius = 20f;

    [Header("Evader — Observation Normalization")]
    [SerializeField] private float _maxDistance = 50f;
    [SerializeField] private float _maxObsSpeed = 10f;

    // ───────── 내부 상태 ──────────────────────────────────────────────────
    // _rb 는 DroneAgent.ResetPhysicsState() 로 위임 — 직접 선언 불필요
    private EvaderReward _rewardCalculator;
    private float        _episodeTimer;

    private Vector3 _spawnCenter;
    private Vector3 _goalCenter;

    // 추격자 추적 (Stage1+ 은폐 메모리)
    private Vector3 _lastKnownPursuerPos;
    private float   _timeSincePursuerDetected;
    private bool    _isPursuerVisible;

    // ───────── 초기화 ─────────────────────────────────────────────────────
    // _dronePhysics, _sensorSystem 은 DroneAgent.Awake()에서 자동 설정됨
    public override void Initialize()
    {
        Role              = DroneRole.Evader;
        // _rb는 DroneAgent.Awake()에서 이미 설정됨
        _rewardCalculator = GetComponent<EvaderReward>();
        if (_rewardCalculator == null)
            _rewardCalculator = gameObject.AddComponent<EvaderReward>();

        _spawnCenter = transform.position;
        _goalCenter  = GoalTransform != null
            ? GoalTransform.position
            : transform.position + Vector3.forward * 15f;
    }

    private void OnValidate()
    {
        _maxEpisodeSeconds   = Mathf.Max(1f,   _maxEpisodeSeconds);
        _catchDistance       = Mathf.Max(0.1f, _catchDistance);
        _goalDistance        = Mathf.Max(0.1f, _goalDistance);
        _spawnRadius         = Mathf.Max(0f,   _spawnRadius);
        _spawnAltitude       = Mathf.Max(0f,   _spawnAltitude);
        _goalRandomizeRadius = Mathf.Max(0f,   _goalRandomizeRadius);
        _maxDistance         = Mathf.Max(1f,   _maxDistance);
        _maxObsSpeed         = Mathf.Max(0.1f, _maxObsSpeed);
    }

    // ───────── 에피소드 시작 ──────────────────────────────────────────────
    // DroneAgent.OnEpisodeBegin()의 CityDataAPI 스폰 대신 반경·고도 기반 스폰 사용
    public override void OnEpisodeBegin()
    {
        _episodeTimer             = 0f;
        _timeSincePursuerDetected = 0f;
        _isPursuerVisible         = false;
        _lastKnownPursuerPos      = Vector3.zero;

        // 드론 랜덤 스폰
        Vector2 offset = Random.insideUnitCircle * _spawnRadius;
        transform.position = _spawnCenter + new Vector3(offset.x, _spawnAltitude, offset.y);
        transform.rotation = Quaternion.Euler(0f, Random.Range(0f, 360f), 0f);

        // 물리 초기화 — DroneAgent.ResetPhysicsState() 위임 (Rigidbody + DronePhysics 일괄 초기화)
        ResetPhysicsState();

        // 목표 지점 랜덤 이동 (GoalTransform = DroneAgent 필드)
        if (GoalTransform != null)
        {
            Vector2 goalOffset = Random.insideUnitCircle * _goalRandomizeRadius;
            GoalTransform.position = _goalCenter + new Vector3(goalOffset.x, 0f, goalOffset.y);
        }
    }

    // ───────── 관측 수집 (VectorObs = 18) ────────────────────────────────
    // 구성: 자기 상태 7 + 목표 4 + 추격자/메모리 7 = 18
    // 속도 획득: _dronePhysics API 사용 (DroneAgent와 동일 소스)
    // Ray: RayPerceptionSensor 컴포넌트에서 별도 처리 (Sensor 담당 배민우)
    public override void CollectObservations(VectorSensor sensor)
    {
        // DroneAgent.IsDroneReady() — null 가드 통일
        if (!IsDroneReady())
        {
            for (int i = 0; i < 18; i++)
                sensor.AddObservation(0f);
            return;
        }

        // 자기 상태 (7) — _dronePhysics API 경유
        Vector3 localVel    = transform.InverseTransformDirection(_dronePhysics.GetVelocity());
        Vector3 localAngVel = _dronePhysics.GetLocalAngularVelocity();
        sensor.AddObservation(localVel    / _maxObsSpeed);             // 3
        sensor.AddObservation(localAngVel / _maxObsSpeed);             // 3
        sensor.AddObservation(transform.position.y / _maxDistance);    // 1

        // 목표 지점 (4) — DroneAgent.GoalTransform 사용
        if (GoalTransform != null)
        {
            Vector3 toGoal = GoalTransform.position - transform.position;
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
    }

    // ───────── 행동 처리 ──────────────────────────────────────────────────
    // Heuristic()은 DroneAgent의 QE/WASD 구현을 그대로 사용
    // Action Space: Continuous 4D — thrust [-1,1] (DroneAgent와 동일 범위)
    public override void OnActionReceived(ActionBuffers actions)
    {
        _episodeTimer += Time.fixedDeltaTime;

        float thrust = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
        float roll   = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);
        float pitch  = Mathf.Clamp(actions.ContinuousActions[2], -1f, 1f);
        float yaw    = Mathf.Clamp(actions.ContinuousActions[3], -1f, 1f);

        _dronePhysics?.SetCommand(thrust, roll, pitch, yaw);

        AddReward(_rewardCalculator.ComputeStepReward(
            agentPos:         transform.position,
            goalPos:          GoalTransform    != null ? GoalTransform.position    : Vector3.zero,
            pursuerPos:       TargetTransform  != null ? TargetTransform.position  : Vector3.zero,
            isPursuerVisible: _isPursuerVisible
        ));

        CheckTerminationConditions();
    }

    // ───────── 종료 조건 ──────────────────────────────────────────────────
    private void CheckTerminationConditions()
    {
        // 1) 목표 도달
        if (GoalTransform != null &&
            Vector3.Distance(transform.position, GoalTransform.position) < _goalDistance)
        {
            AddReward(1.0f);
            EndEpisode();
            return;
        }

        // 2) 포획 (goalOnlyMode=false 시에만)
        if (!_goalOnlyMode && TargetTransform != null &&
            Vector3.Distance(transform.position, TargetTransform.position) < _catchDistance)
        {
            AddReward(-1.0f);
            EndEpisode();
            return;
        }

        // 3) 타임아웃 (생존으로 간주)
        if (_episodeTimer >= _maxEpisodeSeconds)
        {
            AddReward(0.2f);
            EndEpisode();
        }
    }

    // ───────── 외부 API ───────────────────────────────────────────────────
    /// <summary>World 팀에서 충돌/추락 감지 시 호출</summary>
    public void SetCrash()
    {
        AddReward(-1.0f);
        EndEpisode();
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

    // ───────── 헬퍼 ──────────────────────────────────────────────────────
    private Vector3 GetPursuerRelVelNormalized()
    {
        if (TargetTransform == null || !IsDroneReady()) return Vector3.zero;
        Rigidbody pRb = TargetTransform.GetComponent<Rigidbody>();
        return pRb != null
            ? (pRb.linearVelocity - _dronePhysics.GetVelocity()) / _maxObsSpeed
            : Vector3.zero;
    }
}