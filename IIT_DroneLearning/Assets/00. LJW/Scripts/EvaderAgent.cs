using UnityEngine;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

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
    // ───────── Inspector 설정 (Evader 전용) ───────────────────────────────
    // GoalTransform   : DroneAgent.GoalTransform  사용 (목표 지점)
    // TargetTransform : DroneAgent.TargetTransform 사용 (추격 드론)

    [Header("Evader — Goal Zone")]
    [Tooltip("Goal GameObject에 부착된 Goal 컴포넌트")]
    [SerializeField] private Goal _goalZone;

    [Header("Evader — Episode Settings")]
    [SerializeField] private float _maxEpisodeSeconds = 25f;
    [SerializeField] private float _catchDistance     = 1.5f;

    [Header("Evader — Stage Control")]
    [Tooltip("true = Stage0-A (Pursuer 없음, 순수 목표 도달 학습)")]
    [SerializeField] private bool _goalOnlyMode  = true;
    [SerializeField] private int  _currentStage  = 0;

    [Header("Evader — SpawnCenter 쿼리 결과 (읽기 전용)")]
    [SerializeField] private float _queriedMinY;
    [SerializeField] private float _queriedMaxY;
    [SerializeField] private float _queriedRadius;

    [Header("Evader — Observation Normalization")]
    [SerializeField] private float _maxDistance = 50f;
    [SerializeField] private float _maxObsSpeed = 10f;

    // ───────── 내부 상태 ──────────────────────────────────────────────────
    // _rb 는 DroneAgent.ResetPhysicsState() 로 위임 — 직접 선언 불필요
    private EvaderReward  _rewardCalculator;
    private EpisodeLogger _episodeLogger;
    private float         _episodeTimer;
    private int           _episodeSteps;

    private Vector3 _spawnCenter;  // SpawnCenter 미설정 시 폴백용 초기 위치

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
    }

    private void OnValidate()
    {
        _maxEpisodeSeconds = Mathf.Max(1f,   _maxEpisodeSeconds);
        _catchDistance     = Mathf.Max(0.1f, _catchDistance);
        _maxDistance       = Mathf.Max(1f,   _maxDistance);
        _maxObsSpeed       = Mathf.Max(0.1f, _maxObsSpeed);
    }

    // ───────── 에피소드 시작 ──────────────────────────────────────────────
    // DroneAgent.OnEpisodeBegin()의 CityDataAPI 스폰 대신 반경·고도 기반 스폰 사용
    public override void OnEpisodeBegin()
    {
        _episodeTimer             = 0f;
        _episodeSteps             = 0;
        _timeSincePursuerDetected = 0f;
        _isPursuerVisible         = false;
        _lastKnownPursuerPos      = Vector3.zero;

        // 드론 랜덤 스폰 — SpawnCenter 우선, 없으면 초기 위치 기반 폴백
        if (SpawnCenter.Current != null)
        {
            QuerySpawnCenter();
            var range = SpawnCenter.Current.GetEvaderSpawnRange();
            transform.position = SpawnCenter.Current.GetRandomPosition(range);
        }
        else
        {
            // SpawnCenter 없음 — 초기 위치에서 스폰
            transform.position = _spawnCenter;
        }
        transform.rotation = Quaternion.Euler(0f, Random.Range(0f, 360f), 0f);

        // 물리 초기화 — DroneAgent.ResetPhysicsState() 위임 (Rigidbody + DronePhysics 일괄 초기화)
        ResetPhysicsState();

        // 목표 지점 랜덤 이동 — Goal에 위임
        _goalZone?.RandomizePosition();
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

        // 목표 지점 (4) — Goal API 사용
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
        _episodeTimer += Time.fixedDeltaTime;
        _episodeSteps++;

        float thrust = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
        float roll   = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);
        float pitch  = Mathf.Clamp(actions.ContinuousActions[2], -1f, 1f);
        float yaw    = Mathf.Clamp(actions.ContinuousActions[3], -1f, 1f);

        _dronePhysics?.SetCommand(thrust, roll, pitch, yaw);

        AddReward(_rewardCalculator.ComputeStepReward(
            agentPos:         transform.position,
            goalPos:          _goalZone        != null ? _goalZone.GetPosition()   : Vector3.zero,
            pursuerPos:       TargetTransform  != null ? TargetTransform.position  : Vector3.zero,
            isPursuerVisible: _isPursuerVisible,
            agentVel:         _dronePhysics    != null ? _dronePhysics.GetVelocity() : Vector3.zero
        ));

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
            AddReward(-1.0f);
            _episodeLogger?.LogEpisode(EpisodeLogger.TermType.Captured, _episodeSteps);
            EndEpisode();
            return;
        }

        // 3) 타임아웃
        if (_episodeTimer >= _maxEpisodeSeconds)
        {
            _episodeLogger?.LogEpisode(EpisodeLogger.TermType.Timeout, _episodeSteps);
            EndEpisode();
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

        AddReward(1.0f);
        _episodeLogger?.LogEpisode(EpisodeLogger.TermType.Goal, _episodeSteps);
        EndEpisode();
    }

    private void OnDestroy()
    {
        if (_goalZone != null)
            _goalZone.OnArrival -= OnGoalArrived;
    }

    // ───────── 외부 API ───────────────────────────────────────────────────
    /// <summary>World 팀에서 충돌/추락 감지 시 호출</summary>
    public void SetCrash()
    {
        AddReward(-1.0f);
        _episodeLogger?.LogEpisode(EpisodeLogger.TermType.Crash, _episodeSteps);
        EndEpisode();
    }

    /// <summary>Pursuer가 capture에 성공했을 때 외부에서 호출</summary>
    public void SetCaptured()
    {
        AddReward(-1.0f);
        _episodeLogger?.LogEpisode(EpisodeLogger.TermType.Captured, _episodeSteps);
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
    }
#endif

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
