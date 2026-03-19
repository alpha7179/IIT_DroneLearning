using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

/// <summary>
/// EvaderAgent — 회피 드론(Evader) ML-Agents 에이전트
///
/// Stage0: DronePhysics와 직접 연결 (DroneController 없이 동작)
/// 제어 흐름: RL 정책 → SetCommand(throttle, roll, pitch, yaw) → DronePhysics PID
/// 담당: 이재왕 (work/evader)
/// </summary>
public class EvaderAgent : Agent
{
    // ───────── Inspector 설정 ─────────────────────────────────────────────
    [Header("References")]
    [SerializeField] private Transform  _goalTransform;
    [SerializeField] private Transform  _pursuerTransform;   // goalOnlyMode=true 시 비움
    [SerializeField] private DronePhysics _dronePhysics;     // Assets/02. Scripts/DronePhysics.cs

    [Header("Episode Settings")]
    [SerializeField] private float _maxEpisodeSeconds = 25f;
    [SerializeField] private float _catchDistance     = 1.5f;
    [SerializeField] private float _goalDistance      = 2.0f;

    [Header("Stage Control")]
    [Tooltip("true = Stage0-A (Pursuer 없음, 순수 목표 도달 학습)")]
    [SerializeField] private bool _goalOnlyMode  = true;
    [SerializeField] private int  _currentStage  = 0;

    [Header("Spawn Randomization")]
    [SerializeField] private float _spawnRadius          = 10f;
    [SerializeField] private float _spawnAltitude        = 5f;
    [SerializeField] private float _goalRandomizeRadius  = 20f;

    [Header("Observation Normalization")]
    [SerializeField] private float _maxDistance = 50f;
    [SerializeField] private float _maxSpeed    = 10f;

    // ───────── 내부 상태 ──────────────────────────────────────────────────
    private Rigidbody    _rb;
    private EvaderReward _rewardCalculator;
    private float        _episodeTimer;

    private Vector3 _spawnCenter;
    private Vector3 _goalCenter;

    // 추격자 추적 (Stage1+ 은폐 메모리)
    private Vector3 _lastKnownPursuerPos;
    private float   _timeSincePursuerDetected;
    private bool    _isPursuerVisible;

    // ───────── 초기화 ─────────────────────────────────────────────────────
    public override void Initialize()
    {
        _rb               = GetComponent<Rigidbody>();
        _rewardCalculator = GetComponent<EvaderReward>();

        if (_rewardCalculator == null)
            _rewardCalculator = gameObject.AddComponent<EvaderReward>();

        _spawnCenter = transform.position;
        _goalCenter  = _goalTransform != null
            ? _goalTransform.position
            : transform.position + Vector3.forward * 15f;
    }

    // ───────── 에피소드 시작 ──────────────────────────────────────────────
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

        // 물리 초기화
        _rb.linearVelocity  = Vector3.zero;
        _rb.angularVelocity = Vector3.zero;
        _dronePhysics?.ResetPhysics();

        // 목표 지점 랜덤 이동
        if (_goalTransform != null)
        {
            Vector2 goalOffset = Random.insideUnitCircle * _goalRandomizeRadius;
            _goalTransform.position = _goalCenter + new Vector3(goalOffset.x, 0f, goalOffset.y);
        }
    }

    // ───────── 관측 수집 (VectorObs = 18) ────────────────────────────────
    /// 구성: 자기 상태 7 + 목표 4 + 추격자/메모리 7 = 18
    /// (Ray는 RayPerceptionSensor 컴포넌트에서 별도 처리)
    public override void CollectObservations(VectorSensor sensor)
    {
        // 자기 상태 (7)
        Vector3 localVel    = transform.InverseTransformDirection(_rb.linearVelocity);
        Vector3 localAngVel = transform.InverseTransformDirection(_rb.angularVelocity);
        sensor.AddObservation(localVel    / _maxSpeed);        // 3
        sensor.AddObservation(localAngVel / _maxSpeed);        // 3
        sensor.AddObservation(transform.position.y / _maxDistance); // 1

        // 목표 지점 (4)
        if (_goalTransform != null)
        {
            Vector3 toGoal = _goalTransform.position - transform.position;
            sensor.AddObservation(toGoal.normalized);              // 3
            sensor.AddObservation(toGoal.magnitude / _maxDistance); // 1
        }
        else
        {
            sensor.AddObservation(Vector3.zero); // 3
            sensor.AddObservation(1f);           // 1
        }

        // 추격자 정보 (7) — goalOnlyMode 시 전부 0
        bool usePursuerHint = !_goalOnlyMode && _currentStage == 0 && _pursuerTransform != null;
        if (usePursuerHint)
        {
            Vector3 toPursuer = _pursuerTransform.position - transform.position;
            Rigidbody pRb     = _pursuerTransform.GetComponent<Rigidbody>();
            Vector3 relVel    = pRb != null ? pRb.linearVelocity - _rb.linearVelocity : Vector3.zero;
            sensor.AddObservation(toPursuer / _maxDistance);    // 3
            sensor.AddObservation(relVel    / _maxSpeed);       // 3
            sensor.AddObservation(_isPursuerVisible ? 1f : 0f); // 1
        }
        else
        {
            // goalOnlyMode 또는 Stage1+: 메모리 기반 (초기엔 전부 0)
            sensor.AddObservation(_lastKnownPursuerPos / _maxDistance); // 3
            sensor.AddObservation(GetPursuerRelVelNormalized());         // 3
            sensor.AddObservation(_isPursuerVisible ? 1f : 0f);         // 1
        }
    }

    // ───────── 행동 처리 ──────────────────────────────────────────────────
    /// Action Space: Continuous 4D
    ///   [0] thrust_cmd    ∈ [0, 1]
    ///   [1] roll_rate_cmd ∈ [-1, 1]
    ///   [2] pitch_rate_cmd ∈ [-1, 1]
    ///   [3] yaw_rate_cmd  ∈ [-1, 1]
    public override void OnActionReceived(ActionBuffers actions)
    {
        _episodeTimer += Time.fixedDeltaTime;

        float thrust = Mathf.Clamp01(actions.ContinuousActions[0]);
        float roll   = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);
        float pitch  = Mathf.Clamp(actions.ContinuousActions[2], -1f, 1f);
        float yaw    = Mathf.Clamp(actions.ContinuousActions[3], -1f, 1f);

        _dronePhysics?.SetCommand(thrust, roll, pitch, yaw);

        AddReward(_rewardCalculator.ComputeStepReward(
            agentPos:         transform.position,
            goalPos:          _goalTransform   != null ? _goalTransform.position   : Vector3.zero,
            pursuerPos:       _pursuerTransform != null ? _pursuerTransform.position : Vector3.zero,
            isPursuerVisible: _isPursuerVisible
        ));

        CheckTerminationConditions();
    }

    // ───────── 종료 조건 ──────────────────────────────────────────────────
    private void CheckTerminationConditions()
    {
        // 1) 목표 도달
        if (_goalTransform != null &&
            Vector3.Distance(transform.position, _goalTransform.position) < _goalDistance)
        {
            AddReward(1.0f);
            EndEpisode();
            return;
        }

        // 2) 포획 (goalOnlyMode=false 시에만)
        if (!_goalOnlyMode && _pursuerTransform != null &&
            Vector3.Distance(transform.position, _pursuerTransform.position) < _catchDistance)
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

    // ───────── 에디터 테스트용 Heuristic ─────────────────────────────────
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var ca = actionsOut.ContinuousActions;
        ca[0] = Input.GetKey(KeyCode.Space) ? 1f : 0.5f; // 기본 호버링
        ca[1] = Input.GetAxis("Horizontal");
        ca[2] = Input.GetAxis("Vertical");
        ca[3] = 0f;
    }

    // ───────── 헬퍼 ──────────────────────────────────────────────────────
    private Vector3 GetPursuerRelVelNormalized()
    {
        if (_pursuerTransform == null) return Vector3.zero;
        Rigidbody pRb = _pursuerTransform.GetComponent<Rigidbody>();
        return pRb != null ? (pRb.linearVelocity - _rb.linearVelocity) / _maxSpeed : Vector3.zero;
    }
}
