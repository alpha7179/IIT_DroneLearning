using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using BMW.DroneSensor;

/// <summary>
/// 드론 에이전트 기본 클래스 (Tracker / Evader 공통)
///
/// ────────────────────────────────────────
/// 액션 공간: Continuous 4개
///   action[0] = thrust  ∈ [-1, 1]  고도 목표 변화
///   action[1] = roll    ∈ [-1, 1]  Roll 목표 각도
///   action[2] = pitch   ∈ [-1, 1]  Pitch 목표 각도
///   action[3] = yaw     ∈ [-1, 1]  Yaw 직접 토크
///
/// ────────────────────────────────────────
/// 상태 공간: 44차원
///   [0-2]   자신의 위치 (로컬)          (3)
///   [3-5]   자신의 선속도               (3)
///   [6-8]   자신의 로컬 각속도          (3)
///   [9-11]  자신의 회전 (정규화 Euler)  (3)
///   [12-14] 타겟까지 상대 위치          (3)
///   [15-17] 타겟과의 상대 속도          (3)
///   [18-43] Ray 센서 거리값 (26개)      (26)
///   합계: 44
///
/// ────────────────────────────────────────
/// Heuristic 키 매핑:
///   E / Q : Thrust (상승 / 하강)
///   A / D : Roll   (좌 / 우 기울기)
///   W / S : Pitch  (전진 / 후진 기울기)
///   J / L : Yaw    (좌 / 우 회전)
/// </summary>
public class DroneAgent : Agent
{
    // ──────────────────────────────────────────
    // Inspector
    // ──────────────────────────────────────────
    [Header("참조 오브젝트")]
    public Transform TargetTransform;   // Tracker → Evader, Evader → Tracker
    public Transform GoalTransform;     // Evader 전용 목표 지점

    [Header("에피소드 설정")]
    public float SpawnRangeX = 8f;
    public float SpawnRangeZ = 8f;
    public float SpawnHeight = 8f;      // 충분히 높게 → Altitude PID Ki 누적 시간 확보

    [Header("보상 파라미터")]
    public float CatchDistance = 2f;    // Tracker 성공 판정 거리
    public float StepPenalty   = -0.001f;

    // ──────────────────────────────────────────
    // 컴포넌트 참조
    // ──────────────────────────────────────────
    private DronePhysics      _dronePhysics;
    private DroneSensorSystem _sensorSystem;

    // ──────────────────────────────────────────
    // 초기화
    // ──────────────────────────────────────────
    protected override void Awake()
    {
        base.Awake();
        _dronePhysics = GetComponent<DronePhysics>();
        _sensorSystem = GetComponent<DroneSensorSystem>();
    }

    // ──────────────────────────────────────────
    // 에피소드 시작
    // ──────────────────────────────────────────
    public override void OnEpisodeBegin()
    {
        _dronePhysics.ResetPhysics();

        transform.localPosition = new Vector3(
            Random.Range(-SpawnRangeX, SpawnRangeX),
            SpawnHeight,
            Random.Range(-SpawnRangeZ, SpawnRangeZ)
        );
    }

    // ──────────────────────────────────────────
    // 상태 공간 수집 (44차원)
    // ──────────────────────────────────────────
    public override void CollectObservations(VectorSensor sensor)
    {
        // [0-2] 자신의 로컬 위치 (3)
        sensor.AddObservation(transform.localPosition);

        // [3-5] 자신의 선속도 (3)
        sensor.AddObservation(_dronePhysics.GetVelocity());

        // [6-8] 자신의 로컬 각속도 (3)
        sensor.AddObservation(_dronePhysics.GetLocalAngularVelocity());

        // [9-11] 자신의 회전 정규화 (3)
        Vector3 euler = _dronePhysics.GetRotationEuler();
        sensor.AddObservation(WrapAngle(euler.x) / 180f);
        sensor.AddObservation(WrapAngle(euler.y) / 180f);
        sensor.AddObservation(WrapAngle(euler.z) / 180f);

        // [12-14] 타겟 상대 위치 (3)
        if (TargetTransform != null)
            sensor.AddObservation(TargetTransform.localPosition - transform.localPosition);
        else
            sensor.AddObservation(Vector3.zero);

        // [15-17] 타겟 상대 속도 (3)
        if (TargetTransform != null)
        {
            Rigidbody targetRb = TargetTransform.GetComponent<Rigidbody>();
            if (targetRb != null)
                sensor.AddObservation(targetRb.linearVelocity - _dronePhysics.GetVelocity());
            else
                sensor.AddObservation(Vector3.zero);
        }
        else
        {
            sensor.AddObservation(Vector3.zero);
        }

        // [18-43] Ray 센서 거리값 (26)
        if (_sensorSystem != null)
        {
            foreach (float d in _sensorSystem.GetAllNormalizedDistances())
                sensor.AddObservation(d);
        }
        else
        {
            for (int i = 0; i < 26; i++)
                sensor.AddObservation(0f);
        }
    }

    // ──────────────────────────────────────────
    // 액션 수신
    // ──────────────────────────────────────────
    public override void OnActionReceived(ActionBuffers actions)
    {
        float thrust = actions.ContinuousActions[0];
        float roll   = actions.ContinuousActions[1];
        float pitch  = actions.ContinuousActions[2];
        float yaw    = actions.ContinuousActions[3];

        // DronePhysics에 고수준 명령 전달
        _dronePhysics.SetCommand(thrust, roll, pitch, yaw);

        // 스텝 패널티 (생존 보상 역할)
        AddReward(StepPenalty);
    }

    // ──────────────────────────────────────────
    // Heuristic (수동 테스트)
    //
    // E / Q : 상승 / 하강  (Thrust)
    // A / D : 좌 / 우 기울기 (Roll)
    // W / S : 전진 / 후진 기울기 (Pitch)
    // J / L : 좌 / 우 회전 (Yaw)
    // ──────────────────────────────────────────
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var ca = actionsOut.ContinuousActions;

        const float str = 0.5f;

        // Thrust: E(상승) / Q(하강)
        ca[0] = Input.GetKey(KeyCode.E) ?  str :
                Input.GetKey(KeyCode.Q) ? -str : 0f;

        // Roll: D(우 기울기) / A(좌 기울기)
        ca[1] = Input.GetKey(KeyCode.D) ?  str :
                Input.GetKey(KeyCode.A) ? -str : 0f;

        // Pitch: W(전진) / S(후진)
        ca[2] = Input.GetKey(KeyCode.W) ?  str :
                Input.GetKey(KeyCode.S) ? -str : 0f;

        // Yaw: L(우 회전) / J(좌 회전)
        ca[3] = Input.GetKey(KeyCode.L) ?  str :
                Input.GetKey(KeyCode.J) ? -str : 0f;
    }

    // ──────────────────────────────────────────
    // 유틸
    // ──────────────────────────────────────────
    private float WrapAngle(float angle)
    {
        if (angle > 180f) angle -= 360f;
        return angle;
    }
}
