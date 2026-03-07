using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

/// <summary>
/// 드론 에이전트 기본 클래스
/// Tracker/Evader 공통 구조
/// 액션 공간: Discrete 13가지 (이동 7 + 회전 6)
/// 상태 공간: 12차원
/// </summary>
public class DroneAgent : Agent
{
    [Header("컴포넌트")]
    private DronePhysics _dronePhysics;

    [Header("참조 오브젝트")]
    public Transform TargetTransform;
    public Transform GoalTransform;

    [Header("에피소드 설정")]
    public float SpawnRangeX = 8f;
    public float SpawnRangeZ = 8f;
    public float SpawnHeight = 3f;

    [Header("보상 파라미터")]
    public float CatchDistance = 2f;
    public float StepPenalty = -0.001f;

    protected override void Awake()
    {
        base.Awake(); // 부모 Agent.Awake() 먼저 실행
        _dronePhysics = GetComponent<DronePhysics>();
    }
    /// <summary>
    /// 에피소드 시작 시 초기화
    /// </summary>
    public override void OnEpisodeBegin()
    {
        _dronePhysics.ResetPhysics();

        transform.localPosition = new Vector3(
            Random.Range(-SpawnRangeX, SpawnRangeX),
            SpawnHeight,
            Random.Range(-SpawnRangeZ, SpawnRangeZ)
        );
    }

    /// <summary>
    /// 상태 공간 정의 (총 12차원)
    /// </summary>
    public override void CollectObservations(VectorSensor sensor)
    {
        // 자신의 위치 (3)
        sensor.AddObservation(transform.localPosition);

        // 자신의 속도 (3)
        sensor.AddObservation(_dronePhysics.GetVelocity());

        // 자신의 회전 (Pitch, Yaw, Roll) 정규화 (3)
        Vector3 rot = _dronePhysics.GetRotation();
        sensor.AddObservation(rot.x / 180f); // Pitch
        sensor.AddObservation(rot.y / 180f); // Yaw
        sensor.AddObservation(rot.z / 180f); // Roll

        // 타겟 상대 위치 (3)
        if (TargetTransform != null)
            sensor.AddObservation(TargetTransform.localPosition - transform.localPosition);
        else
            sensor.AddObservation(Vector3.zero);
    }

    /// <summary>
    /// 액션 수신 및 처리
    /// 0: 정지
    /// 1: +X (우)      2: -X (좌)
    /// 3: +Z (전진)    4: -Z (후진)
    /// 5: +Y (상승)    6: -Y (하강)
    /// 7: Yaw+         8: Yaw-
    /// 9: Pitch+       10: Pitch-
    /// 11: Roll+       12: Roll-
    /// </summary>
    public override void OnActionReceived(ActionBuffers actions)
    {
        int action = actions.DiscreteActions[0];

        float x = 0f, y = 0f, z = 0f;
        float yaw = 0f, pitch = 0f, roll = 0f;

        switch (action)
        {
            case 0:  break;
            case 1:  x = +1f;     break;
            case 2:  x = -1f;     break;
            case 3:  z = +1f;     break;
            case 4:  z = -1f;     break;
            case 5:  y = +1f;     break;
            case 6:  y = -1f;     break;
            case 7:  yaw = +1f;   break;
            case 8:  yaw = -1f;   break;
            case 9:  pitch = +1f; break;
            case 10: pitch = -1f; break;
            case 11: roll = +1f;  break;
            case 12: roll = -1f;  break;
        }

        _dronePhysics.ApplyMovement(x, y, z);

        if (yaw != 0f)   _dronePhysics.ApplyYaw(yaw);
        if (pitch != 0f) _dronePhysics.ApplyPitch(pitch);
        if (roll != 0f)  _dronePhysics.ApplyRoll(roll);

        AddReward(StepPenalty);
    }

    /// <summary>
    /// 키보드 수동 조작 (테스트용)
    /// W/S: 전후, A/D: 좌우, E/Q: 상하
    /// J/L: Yaw, I/K: Pitch, U/O: Roll
    /// </summary>
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discrete = actionsOut.DiscreteActions;

        if (Input.GetKey(KeyCode.D))      discrete[0] = 1;
        else if (Input.GetKey(KeyCode.A)) discrete[0] = 2;
        else if (Input.GetKey(KeyCode.W)) discrete[0] = 3;
        else if (Input.GetKey(KeyCode.S)) discrete[0] = 4;
        else if (Input.GetKey(KeyCode.E)) discrete[0] = 5;
        else if (Input.GetKey(KeyCode.Q)) discrete[0] = 6;
        else if (Input.GetKey(KeyCode.L)) discrete[0] = 7;
        else if (Input.GetKey(KeyCode.J)) discrete[0] = 8;
        else if (Input.GetKey(KeyCode.I)) discrete[0] = 9;
        else if (Input.GetKey(KeyCode.K)) discrete[0] = 10;
        else if (Input.GetKey(KeyCode.O)) discrete[0] = 11;
        else if (Input.GetKey(KeyCode.U)) discrete[0] = 12;
        else                              discrete[0] = 0;
    }
}