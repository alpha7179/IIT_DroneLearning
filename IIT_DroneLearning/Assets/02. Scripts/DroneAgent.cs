using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using BMW.DroneSensor;
using ProceduralCityGenerator;

/// <summary>드론 역할 — Inspector에서 선택</summary>
public enum DroneRole
{
    Pursuer,  // 추격 드론 — GetPursuerSpawnPosition()
    Evader,   // 도망 드론 — GetEvaderSpawnPosition()
}

/// <summary>
/// RL Agent wrapper for drone physics.
/// Action: [thrust, roll, pitch, yaw] in [-1, 1].
/// Heuristic maps QE / WASD / LeftRight Arrow for manual verification.
/// </summary>
public class DroneAgent : Agent
{
    [Header("References")]
    public Transform TargetTransform;
    public Transform GoalTransform;

    [Header("Role")]
    [Tooltip("Pursuer: 추격 드론 스폰 위치 사용\nEvader: 도망 드론 스폰 위치 사용")]
    public DroneRole Role = DroneRole.Pursuer;

    [Header("Episode Spawn")]
    public float SpawnRangeX = 8f;
    public float SpawnRangeZ = 8f;
    public float SpawnHeight = 8f;
    public float SpawnYawMaxDeg = 180f;
    public float StepPenalty = -0.001f;

    [Header("Heuristic")]
    public float ManualThrottle = 0.5f;
    public float ManualAttitude = 0.5f;
    public float ManualYawRate = 0.4f;

    private DronePhysics _dronePhysics;
    private DroneSensorSystem _sensorSystem;

    protected override void Awake()
    {
        base.Awake();
        _dronePhysics = GetComponent<DronePhysics>();
        _sensorSystem = GetComponent<DroneSensorSystem>();
    }

    public override void OnEpisodeBegin()
    {
        if (_dronePhysics == null)
            return;

        _dronePhysics.ResetPhysics();

        // CityGenerator 연동: Role에 따라 스폰 위치 자동 선택
        if (CityDataAPI.Instance != null && CityDataAPI.Instance.HasSpawnConfiguration())
        {
            Vector3 spawnPos = Role == DroneRole.Evader
                ? CityDataAPI.Instance.GetEvaderSpawnPosition()
                : CityDataAPI.Instance.GetPursuerSpawnPosition();

            transform.SetPositionAndRotation(
                spawnPos,
                Quaternion.Euler(0f, Random.Range(-SpawnYawMaxDeg, SpawnYawMaxDeg), 0f)
            );
            return;
        }

        // 폴백: CityDataAPI 없을 때 랜덤 스폰
        transform.SetPositionAndRotation(
            new Vector3(
                Random.Range(-SpawnRangeX, SpawnRangeX),
                SpawnHeight,
                Random.Range(-SpawnRangeZ, SpawnRangeZ)
            ),
            Quaternion.Euler(0f, Random.Range(-SpawnYawMaxDeg, SpawnYawMaxDeg), 0f)
        );
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        if (_dronePhysics == null)
        {
            for (int i = 0; i < 44; i++)
                sensor.AddObservation(0f);
            return;
        }

        // 44 observations (compatibility): pos(3), velocity(3), ang vel(3), euler(3),
        // target-relative pos(3), target-relative vel(3), ray distance(26)
        sensor.AddObservation(transform.localPosition);
        sensor.AddObservation(_dronePhysics.GetVelocity());
        sensor.AddObservation(_dronePhysics.GetLocalAngularVelocity());

        Vector3 euler = _dronePhysics.GetRotationEuler();
        sensor.AddObservation(NormalizeSignedAngle(euler.x));
        sensor.AddObservation(NormalizeSignedAngle(euler.y));
        sensor.AddObservation(NormalizeSignedAngle(euler.z));

        if (TargetTransform != null)
            sensor.AddObservation(TargetTransform.localPosition - transform.localPosition);
        else
            sensor.AddObservation(Vector3.zero);

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

    public override void OnActionReceived(ActionBuffers actions)
    {
        if (_dronePhysics == null)
            return;

        float thrust = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
        float roll = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);
        float pitch = Mathf.Clamp(actions.ContinuousActions[2], -1f, 1f);
        float yaw = Mathf.Clamp(actions.ContinuousActions[3], -1f, 1f);

        _dronePhysics.SetCommand(thrust, roll, pitch, yaw);
        AddReward(StepPenalty);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var ca = actionsOut.ContinuousActions;

        // QE: 상승/하강 (고도 setpoint 조절)
        float thrust = 0f;
        if (Input.GetKey(KeyCode.E)) thrust += ManualThrottle;
        if (Input.GetKey(KeyCode.Q)) thrust -= ManualThrottle;

        // WASD: 월드 기준 동/서/남/북 이동 입력
        Vector3 worldMove = Vector3.zero;
        if (Input.GetKey(KeyCode.W)) worldMove += Vector3.forward;   // North(+Z)
        if (Input.GetKey(KeyCode.S)) worldMove += Vector3.back;      // South(-Z)
        if (Input.GetKey(KeyCode.D)) worldMove += Vector3.right;      // East(+X)
        if (Input.GetKey(KeyCode.A)) worldMove += Vector3.left;       // West(-X)

        if (worldMove.sqrMagnitude > 1f)
            worldMove.Normalize();

        // 드론 헤딩 기준으로 변환해 로컬 롤/피치 입력으로 매핑
        Vector3 localMove = Quaternion.Euler(0f, -transform.eulerAngles.y, 0f) * worldMove;

        float roll = 0f;
        float pitch = 0f;
        roll = Mathf.Clamp(localMove.x * ManualAttitude, -1f, 1f);
        pitch = Mathf.Clamp(localMove.z * ManualAttitude, -1f, 1f);

        // 좌우 화살표: yaw 회전
        float yaw = 0f;
        if (Input.GetKey(KeyCode.RightArrow)) yaw += ManualYawRate;
        if (Input.GetKey(KeyCode.LeftArrow)) yaw -= ManualYawRate;

        ca[0] = Mathf.Clamp(thrust, -1f, 1f);
        ca[1] = Mathf.Clamp(roll, -1f, 1f);
        ca[2] = Mathf.Clamp(pitch, -1f, 1f);
        ca[3] = Mathf.Clamp(yaw, -1f, 1f);
    }

    private static float NormalizeSignedAngle(float angle)
    {
        float v = angle;
        while (v > 180f) v -= 360f;
        while (v < -180f) v += 360f;
        return v / 180f;
    }
}
