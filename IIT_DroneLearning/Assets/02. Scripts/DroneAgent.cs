using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using DroneSensor;
using CityGenerator;

/// <summary>?쒕줎 ??븷 ??Inspector?먯꽌 ?좏깮</summary>
public enum DroneRole
{
    Pursuer,  // 異붽꺽 ?쒕줎 ??GetPursuerSpawnPosition()
    Evader,   // ?꾨쭩 ?쒕줎 ??GetEvaderSpawnPosition()
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
    [Tooltip("Pursuer: 異붽꺽 ?쒕줎 ?ㅽ룿 ?꾩튂 ?ъ슜\nEvader: ?꾨쭩 ?쒕줎 ?ㅽ룿 ?꾩튂 ?ъ슜")]
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

    protected DronePhysics _dronePhysics;
    protected DroneSensorSystem _sensorSystem;
    protected Rigidbody _rb;

    protected override void Awake()
    {
        base.Awake();

        // ?쒕툕?대옒??EvaderAgent ??媛 媛숈? 媛앹껜???덉쑝硫?DroneAgent(踰좎씠????鍮꾪솢?깊솕
        if (GetType() == typeof(DroneAgent))
        {
            var agents = GetComponents<DroneAgent>();
            foreach (var agent in agents)
            {
                if (agent != this && agent.GetType() != typeof(DroneAgent))
                {
                    Debug.LogWarning(
                        $"[DroneAgent] '{agent.GetType().Name}'??媛숈? 媛앹껜???덉뒿?덈떎. " +
                        $"DroneAgent(踰좎씠??瑜?鍮꾪솢?깊솕?⑸땲??", this);
                    enabled = false;
                    return;
                }
            }
        }

        _rb           = GetComponent<Rigidbody>();
        _dronePhysics = GetComponent<DronePhysics>();
        _sensorSystem = GetComponent<DroneSensorSystem>();
    }

    /// <summary>DronePhysics 而댄룷?뚰듃媛 以鍮꾨릺?덈뒗吏 ?뺤씤 ???쒕툕?대옒??null 媛?쒖슜</summary>
    protected bool IsDroneReady() => _dronePhysics != null;

    /// <summary>Rigidbody ?띾룄 珥덇린??+ DronePhysics ?대? ?곹깭 珥덇린??/summary>
    protected void ResetPhysicsState()
    {
        if (_rb != null)
        {
            _rb.linearVelocity  = Vector3.zero;
            _rb.angularVelocity = Vector3.zero;
        }
        _dronePhysics?.ResetPhysics();
    }

    private void OnValidate()
    {
        SpawnRangeX    = Mathf.Max(0f, SpawnRangeX);
        SpawnRangeZ    = Mathf.Max(0f, SpawnRangeZ);
        SpawnHeight    = Mathf.Max(0f, SpawnHeight);
        SpawnYawMaxDeg = Mathf.Clamp(SpawnYawMaxDeg, 0f, 180f);
        StepPenalty    = Mathf.Min(0f, StepPenalty);
        ManualThrottle = Mathf.Max(0f, ManualThrottle);
        ManualAttitude = Mathf.Max(0f, ManualAttitude);
        ManualYawRate  = Mathf.Max(0f, ManualYawRate);
    }

    public override void OnEpisodeBegin()
    {
        if (_dronePhysics == null)
            return;

        _dronePhysics.ResetPhysics();

        // CityGenerator ?곕룞: Role???곕씪 ?ㅽ룿 ?꾩튂 ?먮룞 ?좏깮
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

        // ?대갚: CityDataAPI ?놁쓣 ???쒕뜡 ?ㅽ룿
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

        float thrust = 0f;
        if (Input.GetKey(KeyCode.E)) thrust += ManualThrottle;
        if (Input.GetKey(KeyCode.Q)) thrust -= ManualThrottle;

        Transform reference = transform;
        var droneCameraSystem = GetComponent<DroneCamera.DroneCameraSystem>();
        if (droneCameraSystem != null && droneCameraSystem.Camera != null)
            reference = droneCameraSystem.Camera.transform;

        Vector3 referenceForward = Vector3.ProjectOnPlane(reference.forward, Vector3.up);
        Vector3 referenceRight = Vector3.ProjectOnPlane(reference.right, Vector3.up);

        if (referenceForward.sqrMagnitude < 1e-6f) referenceForward = transform.forward;
        if (referenceRight.sqrMagnitude < 1e-6f) referenceRight = transform.right;

        referenceForward.Normalize();
        referenceRight.Normalize();

        Vector3 worldMove = Vector3.zero;
        if (Input.GetKey(KeyCode.W)) worldMove += referenceForward;
        if (Input.GetKey(KeyCode.S)) worldMove -= referenceForward;
        if (Input.GetKey(KeyCode.D)) worldMove += referenceRight;
        if (Input.GetKey(KeyCode.A)) worldMove -= referenceRight;

        if (worldMove.sqrMagnitude > 1f)
            worldMove.Normalize();

        Vector3 localMove = transform.InverseTransformDirection(worldMove);

        float roll = Mathf.Clamp(localMove.x * ManualAttitude, -1f, 1f);
        float pitch = Mathf.Clamp(localMove.z * ManualAttitude, -1f, 1f);

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
