using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using BMW.DroneSensor;

/// <summary>
/// DroneAgent
///
/// 紐⑤뱶 2媛?吏??
///
/// 1) SingleDroneTestMode = true
///    - ?寃??놁씠 ?쒕줎 1?留?鍮꾪뻾 ?뚯뒪??
///    - Heuristic 議곗쥌 / 臾쇰━ ?덉젙???뺤씤??
///
/// 2) SingleDroneTestMode = false
///    - Target 異붿쟻??RL ?먯씠?꾪듃
///    - Tracker baseline
///
/// ?≪뀡(Continuous 4)
///   [0] thrustDelta : [-1, 1]
///   [1] rollCmd     : [-1, 1]
///   [2] pitchCmd    : [-1, 1]
///   [3] yawCmd      : [-1, 1]
///
/// 愿痢?
/// - ?꾩옱 援ы쁽??SingleDroneTestMode ?щ?? ?곴??놁씠 ??긽 47李⑥썝
/// - target???놁쑝硫?target 愿??6李⑥썝? zero padding
///
/// 援ъ꽦:
///   湲곕낯 15
///   target 6 (?놁쑝硫?zero)
///   ray 26
///   珥앺빀 47
///
/// 以묒슂:
/// - Behavior Parameters??Vector Observation Size??47濡?留욎텛?몄슂.
/// </summary>
[RequireComponent(typeof(DronePhysics))]
public class DroneAgent : Agent
{
    [Header("Mode")]
    [Tooltip("true硫??寃??놁씠 ?⑤룆 ?쒕줎 鍮꾪뻾 ?뚯뒪??紐⑤뱶")]
    public bool SingleDroneTestMode = true;

    [Tooltip("true硫??뚯뒪??紐⑤뱶?먯꽌 reward/termination???먯뒯?섍쾶 ?곸슜")]
    public bool RelaxTerminationInTestMode = true;

    [Header("References")]
    public DronePhysics DronePhysics;
    public Transform Target;
    public Transform AgentSpawnPoint;
    public Transform TargetSpawnPoint;
    public DroneSensorSystem RaySensor;

    [Header("Episode")]
    public float EpisodeTimeLimit = 30f;
    public float ArenaRadius = 25f;
    public float MinSafeAltitude = 0.15f;
    public float MaxSafeAltitude = 25f;

    [Header("Task")]
    public float CaptureDistance = 1.5f;
    public LayerMask OcclusionMask = ~0;

    [Header("Reward")]
    public float StepPenalty = -0.0005f;
    public float DistanceRewardScale = 0.03f;
    public float VisibleReward = 0.001f;
    public float CaptureReward = 2.0f;
    public float CollisionPenalty = -1.0f;
    public float OutOfBoundsPenalty = -1.0f;
    public float TiltPenaltyScale = -0.0005f;
    public float AngularRatePenaltyScale = -0.0003f;

    [Header("Random Reset")]
    public bool RandomizeSpawn = true;
    public float SpawnRadius = 6f;
    public float TargetSpawnRadius = 6f;
    public float SpawnHeight = 2.0f;
    public float TargetSpawnHeight = 2.0f;

    [Header("Test Mode Debug")]
    [Tooltip("?뚯뒪??紐⑤뱶?먯꽌 異⑸룎 ???먰뵾?뚮뱶 醫낅즺 ?щ?")]
    public bool EndEpisodeOnCollisionInTestMode = false;

    [Tooltip("?뚯뒪??紐⑤뱶?먯꽌 ??꾨━諛?醫낅즺 ?щ?")]
    public bool EndEpisodeOnTimeLimitInTestMode = false;

    [Tooltip("?뚯뒪??紐⑤뱶?먯꽌 ?믪씠/?곸뿭 ?댄깉 醫낅즺 ?щ?")]
    public bool EndEpisodeOnBoundsInTestMode = true;

    [Header("Direct Keyboard")]
    [Tooltip("SingleDroneTestMode?먯꽌 ML-Agents ?숈옉怨?臾닿??섍쾶 ???낅젰??吏곸젒 ?곸슜")]
    public bool EnableDirectKeyboardControl = true;

    private Rigidbody _targetRb;
    private float _episodeTimer;
    private float _prevDistance;

    public override void Initialize()
    {
        if (DronePhysics == null)
            DronePhysics = GetComponent<DronePhysics>();

        if (RaySensor == null)
            RaySensor = GetComponentInChildren<DroneSensorSystem>();

        if (Target != null)
            _targetRb = Target.GetComponent<Rigidbody>();
    }

    private void FixedUpdate()
    {
        if (!EnableDirectKeyboardControl || DronePhysics == null)
            return;

        float thrustDelta;
        float rollCmd;
        float pitchCmd;
        float yawCmd;

        ReadKeyboardCommand(out thrustDelta, out rollCmd, out pitchCmd, out yawCmd);
        DronePhysics.SetCommand(thrustDelta, rollCmd, pitchCmd, yawCmd);
    }

    public override void OnEpisodeBegin()
    {
        _episodeTimer = 0f;

        ResetAgentPose();

        if (!SingleDroneTestMode && Target != null)
        {
            ResetTargetPose();
            Vector3 toTarget = Target.position - transform.position;
            _prevDistance = toTarget.magnitude;
        }
        else
        {
            _prevDistance = 0f;
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        Vector3 localVel = DronePhysics.GetLocalVelocity();
        Vector3 localOmega = DronePhysics.GetLocalAngularVelocity();

        sensor.AddObservation(localVel); // 3
        sensor.AddObservation(localOmega / 5f); // 3
        sensor.AddObservation(transform.forward); // 3
        sensor.AddObservation(transform.up); // 3
        sensor.AddObservation(Mathf.Clamp(DronePhysics.GetCurrentRollDeg() / 45f, -1f, 1f)); // 1
        sensor.AddObservation(Mathf.Clamp(DronePhysics.GetCurrentPitchDeg() / 45f, -1f, 1f)); // 1
        sensor.AddObservation(Mathf.Clamp01(DronePhysics.GetCurrentCollectiveThrust())); // 1

        if (!SingleDroneTestMode && Target != null)
        {
            Vector3 relPosWorld = Target.position - transform.position;
            Vector3 relPosLocal = transform.InverseTransformDirection(relPosWorld);
            sensor.AddObservation(relPosLocal / 20f); // 3

            Vector3 targetVel = Vector3.zero;
            if (_targetRb != null)
                targetVel = _targetRb.linearVelocity;

            Vector3 relVel = transform.InverseTransformDirection(targetVel - DronePhysics.GetVelocity());
            sensor.AddObservation(relVel / 10f); // 3
        }
        else
        {
            sensor.AddObservation(Vector3.zero); // 3
            sensor.AddObservation(Vector3.zero); // 3
        }

        if (RaySensor != null)
        {
            float[] distances = RaySensor.GetAllNormalizedDistances();
            if (distances != null && distances.Length == 26)
            {
                for (int i = 0; i < 26; i++)
                    sensor.AddObservation(distances[i]);
            }
            else
            {
                for (int i = 0; i < 26; i++)
                    sensor.AddObservation(1f);
            }
        }
        else
        {
            for (int i = 0; i < 26; i++)
                sensor.AddObservation(1f);
        }

        // 珥앺빀 = 15 + 6 + 26 = 47
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        if (EnableDirectKeyboardControl)
        {
            float kbThrustDelta;
            float kbRollCmd;
            float kbPitchCmd;
            float kbYawCmd;
            ReadKeyboardCommand(out kbThrustDelta, out kbRollCmd, out kbPitchCmd, out kbYawCmd);
            DronePhysics.SetCommand(kbThrustDelta, kbRollCmd, kbPitchCmd, kbYawCmd);
            ApplyRewardsAndTermination();
            return;
        }
        float actionThrustDelta = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
        float actionRollCmd = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);
        float actionPitchCmd = Mathf.Clamp(actions.ContinuousActions[2], -1f, 1f);
        float actionYawCmd = 0f;
        if (actions.ContinuousActions.Length > 3)
            actionYawCmd = Mathf.Clamp(actions.ContinuousActions[3], -1f, 1f);

        DronePhysics.SetCommand(actionThrustDelta, actionRollCmd, actionPitchCmd, actionYawCmd);

        ApplyRewardsAndTermination();
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var a = actionsOut.ContinuousActions;
        float thrustDelta;
        float rollCmd;
        float pitchCmd;
        float yawCmd;

        ReadKeyboardCommand(out thrustDelta, out rollCmd, out pitchCmd, out yawCmd);

        a[0] = thrustDelta;
        a[1] = rollCmd;
        a[2] = pitchCmd;
        a[3] = yawCmd;
    }

    private void ReadKeyboardCommand(
        out float thrustDelta,
        out float rollCmd,
        out float pitchCmd,
        out float yawCmd)
    {
        thrustDelta = 0f;
        rollCmd = 0f;
        pitchCmd = 0f;
        yawCmd = 0f;

        if (Input.GetKey(KeyCode.Q))
            thrustDelta = 1f;
        else if (Input.GetKey(KeyCode.E))
            thrustDelta = -1f;

        float rawRoll = 0f;
        float rawPitch = 0f;
        float rawYaw = 0f;

        if (Input.GetKey(KeyCode.A))
            rawRoll -= 1f;
        if (Input.GetKey(KeyCode.D))
            rawRoll += 1f;

        if (Input.GetKey(KeyCode.W))
            rawPitch += 1f;
        if (Input.GetKey(KeyCode.S))
            rawPitch -= 1f;

        if (Input.GetKey(KeyCode.LeftArrow))
            rawYaw += 1f;
        if (Input.GetKey(KeyCode.RightArrow))
            rawYaw -= 1f;

        Vector2 horiz = new Vector2(rawRoll, rawPitch);
        if (horiz.sqrMagnitude > 1f)
            horiz.Normalize();

        rollCmd = horiz.x;
        pitchCmd = horiz.y;
        yawCmd = rawYaw;
    }

    private void ApplyRewardsAndTermination()
    {
        _episodeTimer += Time.fixedDeltaTime;

        // =====================================================
        // Test Mode
        // =====================================================
        if (SingleDroneTestMode)
        {
            // 臾쇰━ ?뚯뒪???④퀎?먯꽌??reward蹂대떎 醫낅즺 議곌굔????以묒슂
            // RL ?숈뒿???꾨땲誘濡??꾩슂 理쒖냼?쒕쭔 ?곸슜
            if (!RelaxTerminationInTestMode)
            {
                AddReward(StepPenalty);

                float tilt = Mathf.Abs(DronePhysics.GetCurrentRollDeg()) + Mathf.Abs(DronePhysics.GetCurrentPitchDeg());
                AddReward(tilt * TiltPenaltyScale);

                Vector3 omega = DronePhysics.GetLocalAngularVelocity();
                AddReward((Mathf.Abs(omega.x) + Mathf.Abs(omega.y) + Mathf.Abs(omega.z)) * AngularRatePenaltyScale);
            }

            if (EndEpisodeOnBoundsInTestMode)
            {
                float y = transform.position.y;
                if (y < MinSafeAltitude || y > MaxSafeAltitude)
                {
                    AddReward(OutOfBoundsPenalty);
                    EndEpisode();
                    return;
                }

                Vector2 p = new Vector2(transform.position.x, transform.position.z);
                if (p.magnitude > ArenaRadius)
                {
                    AddReward(OutOfBoundsPenalty);
                    EndEpisode();
                    return;
                }
            }

            if (EndEpisodeOnTimeLimitInTestMode && _episodeTimer >= EpisodeTimeLimit)
            {
                EndEpisode();
                return;
            }

            return;
        }

        // =====================================================
        // RL Mode
        // =====================================================
        AddReward(StepPenalty);

        float totalTilt = Mathf.Abs(DronePhysics.GetCurrentRollDeg()) + Mathf.Abs(DronePhysics.GetCurrentPitchDeg());
        AddReward(totalTilt * TiltPenaltyScale);

        Vector3 localOmega = DronePhysics.GetLocalAngularVelocity();
        AddReward((Mathf.Abs(localOmega.x) + Mathf.Abs(localOmega.y) + Mathf.Abs(localOmega.z)) * AngularRatePenaltyScale);

        if (Target != null)
        {
            Vector3 toTarget = Target.position - transform.position;
            float distance = toTarget.magnitude;

            float delta = _prevDistance - distance;
            AddReward(delta * DistanceRewardScale);
            _prevDistance = distance;

            if (HasLineOfSightToTarget())
                AddReward(VisibleReward);

            if (distance <= CaptureDistance)
            {
                AddReward(CaptureReward);
                EndEpisode();
                return;
            }
        }

        float agentY = transform.position.y;
        if (agentY < MinSafeAltitude || agentY > MaxSafeAltitude)
        {
            AddReward(OutOfBoundsPenalty);
            EndEpisode();
            return;
        }

        Vector2 agentXZ = new Vector2(transform.position.x, transform.position.z);
        if (agentXZ.magnitude > ArenaRadius)
        {
            AddReward(OutOfBoundsPenalty);
            EndEpisode();
            return;
        }

        if (_episodeTimer >= EpisodeTimeLimit)
        {
            EndEpisode();
        }
    }

    private bool HasLineOfSightToTarget()
    {
        if (Target == null)
            return false;

        Vector3 origin = transform.position;
        Vector3 targetPos = Target.position;
        Vector3 dir = targetPos - origin;
        float dist = dir.magnitude;

        if (dist <= 1e-4f)
            return true;

        if (Physics.Raycast(origin, dir.normalized, out RaycastHit hit, dist, OcclusionMask))
        {
            return hit.transform == Target || hit.transform.IsChildOf(Target);
        }

        return true;
    }

    private void ResetAgentPose()
    {
        Vector3 pos;
        Quaternion rot;

        if (RandomizeSpawn)
        {
            Vector2 r = Random.insideUnitCircle * SpawnRadius;
            pos = new Vector3(r.x, SpawnHeight, r.y);

            float yaw = Random.Range(-180f, 180f);
            rot = Quaternion.Euler(0f, yaw, 0f);
        }
        else if (AgentSpawnPoint != null)
        {
            pos = AgentSpawnPoint.position;
            rot = AgentSpawnPoint.rotation;
        }
        else
        {
            pos = new Vector3(0f, SpawnHeight, 0f);
            rot = Quaternion.identity;
        }

        DronePhysics.ResetPhysicsState(pos, rot);
    }

    private void ResetTargetPose()
    {
        if (Target == null)
            return;

        Vector3 pos;
        Quaternion rot;

        if (RandomizeSpawn)
        {
            Vector2 r = Random.insideUnitCircle * TargetSpawnRadius;
            pos = new Vector3(r.x, TargetSpawnHeight, r.y);

            int safety = 0;
            while (Vector3.Distance(pos, transform.position) < 4f && safety < 30)
            {
                r = Random.insideUnitCircle * TargetSpawnRadius;
                pos = new Vector3(r.x, TargetSpawnHeight, r.y);
                safety++;
            }

            float yaw = Random.Range(-180f, 180f);
            rot = Quaternion.Euler(0f, yaw, 0f);
        }
        else if (TargetSpawnPoint != null)
        {
            pos = TargetSpawnPoint.position;
            rot = TargetSpawnPoint.rotation;
        }
        else
        {
            pos = new Vector3(5f, TargetSpawnHeight, 5f);
            rot = Quaternion.identity;
        }

        Target.position = pos;
        Target.rotation = rot;

        Rigidbody rb = Target.GetComponent<Rigidbody>();
        if (rb != null)
        {
            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
        }

        _targetRb = rb;
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (SingleDroneTestMode)
        {
            if (EndEpisodeOnCollisionInTestMode)
            {
                AddReward(CollisionPenalty);
                EndEpisode();
            }
            return;
        }

        if (Target != null)
        {
            if (collision.transform == Target || collision.transform.IsChildOf(Target))
                return;
        }

        AddReward(CollisionPenalty);
        EndEpisode();
    }
}
