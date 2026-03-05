using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

/// <summary>
/// EvaderAgent — 회피 드론(Evader) ML-Agents 에이전트
///
/// 역할: 추격자(Pursuer)에게 잡히지 않으면서 목표 지점(Goal Zone)에 도달한다.
/// 담당: 이재왕 (work/evader)
///
/// 제어 방식: RL 정책 → PID setpoint 출력 → PID 컨트롤러 → Rigidbody Force/Torque
/// (직접 모터/액추에이터 제어 금지 — Sim2Real 원칙)
///
/// 의존성:
///   - EvaderReward   : 보상 계산 분리
///   - DroneController (World 담당 이강민): PID 파이프라인 제공
///   - SensorManager  (Sensor 담당 배민우): Ray/Camera 관측 제공
/// </summary>
public class EvaderAgent : Agent
{
    // ───────── Inspector 설정 ─────────────────────────────────────────────
    [Header("References")]
    [SerializeField] private Transform _goalTransform;       // 목표 지점 Transform
    [SerializeField] private Transform _pursuerTransform;    // 추격자 Transform (Stage0 힌트용)
    [SerializeField] private DroneController _droneController; // World 담당 PID 파이프라인

    [Header("Episode Settings")]
    [SerializeField] private float _maxEpisodeSeconds = 20f;
    [SerializeField] private float _catchDistance = 2.0f;    // 포획 판정 거리 (팀 합의 필요)
    [SerializeField] private float _goalDistance = 2.0f;     // 목표 도달 판정 거리

    [Header("Stage Control")]
    [Tooltip("0: Ray+힌트(Ground-truth Pursuer pos), 1: 장애물+힌트 제거, 2: Self-Play, 3: Domain Random")]
    [SerializeField] private int _currentStage = 0;

    [Header("Observation Normalization")]
    [SerializeField] private float _maxDistance = 50f;       // 위치 정규화 기준 거리
    [SerializeField] private float _maxSpeed = 10f;          // 속도 정규화 기준

    // ───────── 내부 상태 ──────────────────────────────────────────────────
    private Rigidbody _rb;
    private EvaderReward _rewardCalculator;
    private float _episodeTimer;

    // 구조적 기억 (Stage1+ 은폐 대응)
    private Vector3 _lastKnownPursuerPos;
    private float _timeSincePursuerDetected;
    private bool _isPursuerVisible;

    // ───────── Unity Lifecycle ────────────────────────────────────────────
    public override void Initialize()
    {
        _rb = GetComponent<Rigidbody>();
        _rewardCalculator = GetComponent<EvaderReward>();

        if (_rewardCalculator == null)
            _rewardCalculator = gameObject.AddComponent<EvaderReward>();
    }

    public override void OnEpisodeBegin()
    {
        _episodeTimer = 0f;
        _timeSincePursuerDetected = 0f;
        _isPursuerVisible = false;
        _lastKnownPursuerPos = Vector3.zero;

        // NOTE: 실제 위치/속도 초기화는 World(이강민) 담당 DroneController.Reset()에서 처리
        // _droneController?.ResetDrone();
    }

    // ───────── 관측 수집 ──────────────────────────────────────────────────
    /// <summary>
    /// Observation Space 구성:
    ///   Stage0 기준 총 약 24차원 (팀 합의 후 확정)
    ///
    ///   [자기 상태 - 7]
    ///     로컬 속도(3), 로컬 각속도(3), 고도(1)
    ///
    ///   [목표 지점 - 4]
    ///     상대 방향(3, 정규화), 거리(1, 정규화)
    ///
    ///   [추격자 힌트 - Stage0 전용 - 7]
    ///     상대 위치(3, 정규화), 상대 속도(3, 정규화), 가시 여부(1)
    ///
    ///   [은폐 기억 - Stage1+ - 5]
    ///     마지막 추격자 위치(3, 정규화), 경과 시간(1), 가시 여부(1)
    ///
    ///   Ray는 RayPerceptionSensor 컴포넌트로 별도 처리 (Sensor 담당 배민우)
    /// </summary>
    public override void CollectObservations(VectorSensor sensor)
    {
        // ── 자기 상태 ──────────────────────────────────────
        Vector3 localVelocity = transform.InverseTransformDirection(_rb.linearVelocity);
        Vector3 localAngularVelocity = transform.InverseTransformDirection(_rb.angularVelocity);
        float altitude = transform.position.y;

        sensor.AddObservation(localVelocity / _maxSpeed);          // 3
        sensor.AddObservation(localAngularVelocity / _maxSpeed);   // 3
        sensor.AddObservation(altitude / _maxDistance);            // 1

        // ── 목표 지점 ──────────────────────────────────────
        if (_goalTransform != null)
        {
            Vector3 toGoal = _goalTransform.position - transform.position;
            float goalDist = toGoal.magnitude;
            sensor.AddObservation(toGoal.normalized);              // 3
            sensor.AddObservation(goalDist / _maxDistance);        // 1
        }
        else
        {
            sensor.AddObservation(Vector3.zero);                   // 3
            sensor.AddObservation(1f);                             // 1
        }

        // ── 추격자 정보 ────────────────────────────────────
        if (_currentStage == 0 && _pursuerTransform != null)
        {
            // Stage0: Ground-truth 힌트 허용
            Vector3 toPursuer = _pursuerTransform.position - transform.position;
            Rigidbody pursuerRb = _pursuerTransform.GetComponent<Rigidbody>();
            Vector3 pursuerRelVel = pursuerRb != null
                ? pursuerRb.linearVelocity - _rb.linearVelocity
                : Vector3.zero;

            sensor.AddObservation(toPursuer / _maxDistance);      // 3
            sensor.AddObservation(pursuerRelVel / _maxSpeed);     // 3
            sensor.AddObservation(_isPursuerVisible ? 1f : 0f);   // 1
        }
        else
        {
            // Stage1+: 구조적 기억만 허용
            sensor.AddObservation(_lastKnownPursuerPos / _maxDistance);  // 3
            sensor.AddObservation(pursuerRelVelNormalized());             // 3
            sensor.AddObservation(_isPursuerVisible ? 1f : 0f);          // 1
        }
    }

    // ───────── 행동 처리 ──────────────────────────────────────────────────
    /// <summary>
    /// Action Space: Continuous 4D (PID setpoints)
    ///   actions[0]: thrust_cmd    ∈ [0, 1]   → 추력 명령
    ///   actions[1]: roll_rate_cmd ∈ [-1, 1]  → 롤 각속도 명령
    ///   actions[2]: pitch_rate_cmd ∈ [-1, 1] → 피치 각속도 명령
    ///   actions[3]: yaw_rate_cmd  ∈ [-1, 1]  → 요 각속도 명령
    ///
    ///   적용 주기: FixedUpdate (50Hz, dt=0.02 — 팀 합의 필요)
    /// </summary>
    public override void OnActionReceived(ActionBuffers actions)
    {
        _episodeTimer += Time.fixedDeltaTime;

        float thrustCmd     = Mathf.Clamp01(actions.ContinuousActions[0]);
        float rollRateCmd   = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);
        float pitchRateCmd  = Mathf.Clamp(actions.ContinuousActions[2], -1f, 1f);
        float yawRateCmd    = Mathf.Clamp(actions.ContinuousActions[3], -1f, 1f);

        // World 담당 PID 컨트롤러에 setpoint 전달
        // _droneController?.SetPIDSetpoints(thrustCmd, rollRateCmd, pitchRateCmd, yawRateCmd);

        // ── 보상 계산 ────────────────────────────────────────
        float reward = _rewardCalculator.ComputeStepReward(
            agentPos: transform.position,
            goalPos: _goalTransform != null ? _goalTransform.position : Vector3.zero,
            pursuerPos: _pursuerTransform != null ? _pursuerTransform.position : Vector3.zero,
            isPursuerVisible: _isPursuerVisible
        );
        AddReward(reward);

        // ── 종료 조건 점검 ────────────────────────────────────
        CheckTerminationConditions();
    }

    // ───────── 종료 조건 ──────────────────────────────────────────────────
    private void CheckTerminationConditions()
    {
        // 1) 목표 도달
        if (_goalTransform != null)
        {
            float distToGoal = Vector3.Distance(transform.position, _goalTransform.position);
            if (distToGoal < _goalDistance)
            {
                AddReward(1.0f);  // 목표 도달 보너스
                EndEpisode();
                return;
            }
        }

        // 2) 포획 (Pursuer에게 잡힘)
        if (_pursuerTransform != null)
        {
            float distToPursuer = Vector3.Distance(transform.position, _pursuerTransform.position);
            if (distToPursuer < _catchDistance)
            {
                AddReward(-1.0f); // 포획 페널티
                EndEpisode();
                return;
            }
        }

        // 3) 타임아웃 (생존으로 간주 — 부분 보상)
        if (_episodeTimer >= _maxEpisodeSeconds)
        {
            AddReward(0.2f);  // 생존 부분 보상
            EndEpisode();
            return;
        }

        // 4) Crash 및 고도 이탈은 DroneController(World)에서 OnCollision 콜백으로 처리
        //    → SetCrash() 메서드를 통해 외부에서 종료 트리거
    }

    /// <summary>World 담당이 충돌/추락 감지 시 호출하는 외부 종료 트리거</summary>
    public void SetCrash()
    {
        AddReward(-1.0f);
        EndEpisode();
    }

    // ───────── 시각화 (Heuristic / Editor 전용) ───────────────────────────
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // WASD / 화살표 키로 수동 조작 테스트
        var continuousActions = actionsOut.ContinuousActions;
        continuousActions[0] = Input.GetKey(KeyCode.Space) ? 1f : 0.5f;  // 기본 호버링
        continuousActions[1] = Input.GetAxis("Horizontal");
        continuousActions[2] = Input.GetAxis("Vertical");
        continuousActions[3] = 0f;
    }

    // ───────── 헬퍼 ──────────────────────────────────────────────────────
    private Vector3 pursuerRelVelNormalized()
    {
        if (_pursuerTransform == null) return Vector3.zero;
        Rigidbody pursuerRb = _pursuerTransform.GetComponent<Rigidbody>();
        if (pursuerRb == null) return Vector3.zero;
        return (pursuerRb.linearVelocity - _rb.linearVelocity) / _maxSpeed;
    }

    /// <summary>Sensor(배민우) 담당에서 LOS 결과를 매 프레임 갱신할 때 호출</summary>
    public void SetPursuerVisibility(bool isVisible, Vector3 pursuerWorldPos)
    {
        _isPursuerVisible = isVisible;
        if (isVisible)
        {
            _lastKnownPursuerPos = pursuerWorldPos;
            _timeSincePursuerDetected = 0f;
        }
        else
        {
            _timeSincePursuerDetected += Time.deltaTime;
        }
    }
}
