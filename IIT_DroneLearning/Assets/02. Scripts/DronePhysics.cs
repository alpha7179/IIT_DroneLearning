using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
public class DronePhysics : MonoBehaviour
{
    [Header("Drone Body")]
    public float Mass = 1f;
    public float LinearDrag = 1.8f;
    public float AngularDrag = 6f;
    public bool UseGravity = true;
    public bool ForceCenterOfMassZero = true;

    [Header("PID Gains (drone-simualator style)")]
    [Tooltip("Pitch angle controller")]
    public float PitchKp = 0.2f;

    [Tooltip("Pitch integral gain")]
    public float PitchKi = 0.01f;

    [Tooltip("Roll angle controller")]
    public float RollKp = 0.2f;

    [Tooltip("Roll integral gain")]
    public float RollKi = 0.01f;

    [Tooltip("Yaw angle controller")]
    public float YawKp = 0.2f;

    [Tooltip("Yaw integral gain")]
    public float YawKi = 0.01f;

    [Header("Command Response")]
    [Tooltip("기울기 명령 반응 속도(도/sec)")]
    public float AttitudeCommandDegPerSec = 35f;

    [Tooltip("정지 입력에서 목표각 복귀 속도")]
    public float AttitudeReturnDegPerSec = 8f;

    [Tooltip("최대 목표 롤/피치 각도(도)")]
    public float MaxCommandAngle = 25f;

    [Tooltip("상승/하강 반응 계수")]
    public float VerticalCommandScale = 0.6f;

    [Tooltip("중력보상 토크/힘 계산 기반")]
    public float MotorArmScale = 0.5f;

    [Tooltip("모터 출력 상한")]
    public float MaxMotorForce = 12f;

    private Rigidbody _rb;
    private float _thrustCommand;
    private float _rollCommand;
    private float _pitchCommand;
    private float _yawCommand;

    private float _desiredPitch;
    private float _desiredRoll;
    private float _desiredYaw;

    private float _integralPitch;
    private float _integralRoll;
    private float _integralYaw;

    private float _collectiveThrottle;
    private float _hoverMotorThrottle;
    private float _maxCollectiveThrottle;

    private Vector3[] _motorLocalPos = new Vector3[4];

    private void Awake()
    {
        _rb = GetComponent<Rigidbody>();

        _rb.mass = Mass;
        _rb.useGravity = UseGravity;
        _rb.linearDamping = LinearDrag;
        _rb.angularDamping = AngularDrag;
        _rb.freezeRotation = false;
        _rb.maxAngularVelocity = 20f;

        if (ForceCenterOfMassZero)
            _rb.centerOfMass = Vector3.zero;

        BuildMotorLayoutFromCollider();
        RecalculateHoverThrottle();
        ResetControllers();
    }

    private void BuildMotorLayoutFromCollider()
    {
        float armX = MotorArmScale;
        float armZ = MotorArmScale;

        var col = GetComponent<Collider>();
        if (col != null)
        {
            Vector3 ext = col.bounds.extents;
            armX = Mathf.Max(ext.x, 0.2f);
            armZ = Mathf.Max(ext.z, 0.2f);
        }

        _motorLocalPos[0] = new Vector3(-armX, 0f, +armZ);
        _motorLocalPos[1] = new Vector3(+armX, 0f, +armZ);
        _motorLocalPos[2] = new Vector3(-armX, 0f, -armZ);
        _motorLocalPos[3] = new Vector3(+armX, 0f, -armZ);
    }

    private void RecalculateHoverThrottle()
    {
        if (_rb == null)
            return;

        // remote 기준 식: mass * 9.8 / 4
        _hoverMotorThrottle = (_rb.mass * Mathf.Abs(Physics.gravity.y)) / 4f;
        _maxCollectiveThrottle = Mathf.Max(_hoverMotorThrottle * 3f, MaxMotorForce);
    }

    private void ResetControllers()
    {
        _thrustCommand = 0f;
        _rollCommand = 0f;
        _pitchCommand = 0f;
        _yawCommand = 0f;

        _desiredPitch = 0f;
        _desiredRoll = 0f;
        _desiredYaw = 0f;

        _integralPitch = 0f;
        _integralRoll = 0f;
        _integralYaw = 0f;

        _collectiveThrottle = _hoverMotorThrottle;
    }

    private void FixedUpdate()
    {
        ApplyControl(Time.fixedDeltaTime);
    }

    public void SetCommand(float thrustDelta, float rollCmd, float pitchCmd, float yawCmd)
    {
        _thrustCommand = Mathf.Clamp(thrustDelta, -1f, 1f);
        _rollCommand = Mathf.Clamp(rollCmd, -1f, 1f);
        _pitchCommand = Mathf.Clamp(pitchCmd, -1f, 1f);
        _yawCommand = Mathf.Clamp(yawCmd, -1f, 1f);
    }

    private void ApplyControl(float dt)
    {
        Vector3 localOmega = transform.InverseTransformDirection(_rb.angularVelocity);
        float currentRoll = GetCurrentRollDeg();
        float currentPitch = GetCurrentPitchDeg();
        float currentYaw = NormalizeAngle180(transform.localEulerAngles.y);

        // simultaneous command input: both 방향 입력이 들어오면 덧셈되어 목표각이 함께 갱신됨
        float commandPitch = -_pitchCommand * AttitudeCommandDegPerSec * dt;
        float commandRoll = _rollCommand * AttitudeCommandDegPerSec * dt;
        float commandYaw = _yawCommand * AttitudeCommandDegPerSec * dt;

        _desiredPitch = Mathf.Clamp(_desiredPitch + commandPitch, -MaxCommandAngle, MaxCommandAngle);
        _desiredRoll = Mathf.Clamp(_desiredRoll + commandRoll, -MaxCommandAngle, MaxCommandAngle);

        if (Mathf.Abs(_yawCommand) > 0.01f)
            _desiredYaw = Mathf.Clamp(_desiredYaw + commandYaw, -180f, 180f);
        else
            _desiredYaw = Mathf.Lerp(_desiredYaw, 0f, AttitudeReturnDegPerSec * dt / 180f);

        if (Mathf.Abs(_rollCommand) < 0.001f)
            _desiredRoll = Mathf.Lerp(_desiredRoll, 0f, AttitudeReturnDegPerSec * dt);

        if (Mathf.Abs(_pitchCommand) < 0.001f)
            _desiredPitch = Mathf.Lerp(_desiredPitch, 0f, AttitudeReturnDegPerSec * dt);

        // keep thrust stable while level hovering
        _collectiveThrottle = _hoverMotorThrottle + _thrustCommand * VerticalCommandScale;
        if (_collectiveThrottle < 0f)
            _collectiveThrottle = 0f;

        float upProjection = Mathf.Max(Vector3.Dot(transform.up, Vector3.up), 0.2f);

        float pitchError = _desiredPitch - currentPitch - localOmega.x * Mathf.Rad2Deg * dt;
        float rollError = _desiredRoll - currentRoll - localOmega.z * Mathf.Rad2Deg * dt;
        float yawError = _desiredYaw - currentYaw - localOmega.y * Mathf.Rad2Deg * dt;

        _integralPitch += PitchKi * pitchError * dt;
        _integralRoll += RollKi * rollError * dt;
        _integralYaw += YawKi * yawError * dt;

        _integralPitch = Mathf.Clamp(_integralPitch, -3f, 3f);
        _integralRoll = Mathf.Clamp(_integralRoll, -3f, 3f);
        _integralYaw = Mathf.Clamp(_integralYaw, -3f, 3f);

        float dPitch = PitchKp * pitchError + _integralPitch;
        float dRoll = RollKp * rollError + _integralRoll;
        float dYaw = YawKp * yawError + _integralYaw;

        float m1 = _collectiveThrottle - dPitch - dRoll + dYaw;
        float m2 = _collectiveThrottle - dPitch + dRoll - dYaw;
        float m3 = _collectiveThrottle + dPitch - dRoll - dYaw;
        float m4 = _collectiveThrottle + dPitch + dRoll + dYaw;

        float motorMax = _maxCollectiveThrottle / 4f;
        float minMotor = 0f;
        m1 = Mathf.Clamp(m1 / upProjection, minMotor, motorMax);
        m2 = Mathf.Clamp(m2 / upProjection, minMotor, motorMax);
        m3 = Mathf.Clamp(m3 / upProjection, minMotor, motorMax);
        m4 = Mathf.Clamp(m4 / upProjection, minMotor, motorMax);

        Vector3 f1 = m1 * Vector3.up;
        Vector3 f2 = m2 * Vector3.up;
        Vector3 f3 = m3 * Vector3.up;
        Vector3 f4 = m4 * Vector3.up;

        Vector3 totalForce = f1 + f2 + f3 + f4;
        _rb.AddRelativeForce(totalForce, ForceMode.Force);

        Vector3 torque =
            Vector3.Cross(_motorLocalPos[0], f1) +
            Vector3.Cross(_motorLocalPos[1], f2) +
            Vector3.Cross(_motorLocalPos[2], f3) +
            Vector3.Cross(_motorLocalPos[3], f4);

        // yaw 방향 분리 항목은 기존 검증 스크립트의 모터 토크 항목을 사용
        torque += m1 * Vector3.up + m2 * Vector3.down + m3 * Vector3.down + m4 * Vector3.up;
        _rb.AddRelativeTorque(torque, ForceMode.Force);
    }

    public void ResetPhysicsState(Vector3 position, Quaternion rotation)
    {
        _rb.position = position;
        _rb.rotation = rotation;
        _rb.linearVelocity = Vector3.zero;
        _rb.angularVelocity = Vector3.zero;

        _thrustCommand = 0f;
        _rollCommand = 0f;
        _pitchCommand = 0f;
        _yawCommand = 0f;

        _desiredPitch = 0f;
        _desiredRoll = 0f;
        _desiredYaw = 0f;
        _integralPitch = 0f;
        _integralRoll = 0f;
        _integralYaw = 0f;

        _collectiveThrottle = _hoverMotorThrottle;
    }

    public void ResetPhysics()
    {
        ResetPhysicsState(transform.position, Quaternion.identity);
    }

    public Vector3 GetVelocity()
    {
        return _rb.linearVelocity;
    }

    public Vector3 GetLocalVelocity()
    {
        return transform.InverseTransformDirection(_rb.linearVelocity);
    }

    public Vector3 GetLocalAngularVelocity()
    {
        return transform.InverseTransformDirection(_rb.angularVelocity);
    }

    public Rigidbody GetRigidbody()
    {
        return _rb;
    }

    public float GetCurrentCollectiveThrust()
    {
        return Mathf.InverseLerp(0f, _maxCollectiveThrottle, _collectiveThrottle);
    }

    public float GetCurrentRollDeg()
    {
        return NormalizeAngle180(transform.localEulerAngles.z);
    }

    public float GetCurrentPitchDeg()
    {
        return NormalizeAngle180(transform.localEulerAngles.x);
    }

    private static float NormalizeAngle180(float angle)
    {
        return Mathf.Repeat(angle + 180f, 360f) - 180f;
    }
}
