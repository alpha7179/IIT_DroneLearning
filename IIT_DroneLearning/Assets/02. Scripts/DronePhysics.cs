using UnityEngine;

/// <summary>
/// PID based quadcopter controller.
/// - Altitude: setpoint tracking with PID -> thrust
/// - Roll/Pitch: angle target tracking with PID -> relative torque
/// - Yaw: angle target tracking with PID -> relative torque
/// </summary>
[RequireComponent(typeof(Rigidbody))]
public class DronePhysics : MonoBehaviour
{
    [Header("Rigidbody")]
    public float Mass = 1.0f;
    public float LinearDrag = 1.0f;
    public float AngularDrag = 4.0f;
    public float MaxAngularVelocityDeg = 720f;

[Header("Motor / Body")]
public float MaxMotorThrust = 6.0f;
public float ArmLength = 0.25f;
public bool AutoFitMotorLayoutFromCollider = true;
public float MotorHeightOffset = -0.02f;

    [Header("Altitude")]
    public float MinAltitude = 0.5f;
    public float MaxAltitude = 50f;
    public float MaxAltitudeStep = 2.0f;
    public float AltDeadband = 0.03f;
    public float AltVelocityDamping = 1.2f;

    [Header("Limits")]
    public float MaxHorizontalSpeed = 4f;
    public float MaxVerticalSpeed = 2.5f;

[Header("Attitude limits")]
public float MaxRollAngle = 20f;
public float MaxPitchAngle = 20f;
public float MaxYawRateDeg = 45f;
public float MaxRollTorque = 2.2f;
public float MaxPitchTorque = 2.2f;
public float MaxYawTorque = 1.4f;
public float AttitudeDeadbandDeg = 0.6f;
public float YawDeadbandDeg = 0.8f;
public float RollRateDamping = 0.12f;
public float PitchRateDamping = 0.12f;
public float YawRateDamping = 0.15f;

    [Header("PID - Altitude")]
    public float AltKp = 5.5f;
    public float AltKi = 0.25f;
    public float AltKd = 2.4f;

    [Header("PID - Roll")]
    public float RollKp = 1.8f;
    public float RollKi = 0.02f;
    public float RollKd = 0.25f;

    [Header("PID - Pitch")]
    public float PitchKp = 1.8f;
    public float PitchKi = 0.02f;
    public float PitchKd = 0.25f;

[Header("PID - Yaw")]
public float YawKp = 1.2f;
public float YawKi = 0.01f;
public float YawKd = 0.10f;

    private Rigidbody _rb;
    private PIDController _altPID;
    private PIDController _rollPID;
    private PIDController _pitchPID;
    private PIDController _yawPID;

    private float _targetAltitude;
    private float _targetRoll;
    private float _targetPitch;
    private float _targetYawDeg;
    private float _yawRateInput;
    private float _throttleInput;
    private float _yawTorqueCmd;

    private float[] _motorThrust = new float[4];
    private Vector3[] _motorPositions = new Vector3[4];

    private void Awake()
    {
        _rb = GetComponent<Rigidbody>();
        _rb.mass = Mass;
        _rb.useGravity = true;
        _rb.linearDamping = LinearDrag;
        _rb.angularDamping = AngularDrag;
        _rb.maxAngularVelocity = Mathf.Max(1f, MaxAngularVelocityDeg) * Mathf.Deg2Rad;

        ConfigureInertiaTensor();
        ConfigureMotorLayout();

        _altPID = new PIDController(AltKp, AltKi, AltKd);
        _rollPID = new PIDController(RollKp, RollKi, RollKd);
        _pitchPID = new PIDController(PitchKp, PitchKi, PitchKd);
        _yawPID = new PIDController(YawKp, YawKi, YawKd);

        _targetAltitude = transform.position.y;
        _targetRoll = 0f;
        _targetPitch = 0f;
        _targetYawDeg = transform.eulerAngles.y;
        _yawRateInput = 0f;
        _throttleInput = 0f;
    }

    private void FixedUpdate()
    {
        float dt = Time.fixedDeltaTime;

        // integrate altitude target for smooth climb/descend input
        _targetAltitude = Mathf.Clamp(
            _targetAltitude + _throttleInput * MaxAltitudeStep * dt,
            MinAltitude,
            MaxAltitude
        );

        UpdateControl(dt);
        ApplyForces();
        ClampAltitude();
        ClampVelocity();
    }

    /// <summary>
    /// throttle: [-1,1], roll/pitch: [-1,1], yaw: [-1,1]
    /// </summary>
    public void SetCommand(float throttle, float roll, float pitch, float yaw)
    {
        _throttleInput = Mathf.Clamp(throttle, -1f, 1f);

        _targetRoll = Mathf.Clamp(roll * MaxRollAngle, -MaxRollAngle, MaxRollAngle);
        _targetPitch = Mathf.Clamp(pitch * MaxPitchAngle, -MaxPitchAngle, MaxPitchAngle);
        _yawRateInput = Mathf.Clamp(yaw, -1f, 1f) * MaxYawRateDeg;
    }

    public Vector3 GetVelocity() => _rb.linearVelocity;
    public Vector3 GetLocalAngularVelocity() => transform.InverseTransformDirection(_rb.angularVelocity);
    public Vector3 GetRotationEuler() => transform.eulerAngles;

    public void ResetPhysics()
    {
        _rb.linearVelocity = Vector3.zero;
        _rb.angularVelocity = Vector3.zero;
        _rb.rotation = Quaternion.identity;
        transform.position = new Vector3(transform.position.x, Mathf.Clamp(transform.position.y, MinAltitude, MaxAltitude), transform.position.z);

        _targetAltitude = transform.position.y;
        _targetRoll = 0f;
        _targetPitch = 0f;
        _targetYawDeg = transform.eulerAngles.y;
        _yawRateInput = 0f;
        _throttleInput = 0f;

        _altPID.Reset();
        _rollPID.Reset();
        _pitchPID.Reset();
        _yawPID.Reset();

        for (int i = 0; i < 4; i++)
            _motorThrust[i] = 0.25f;
    }

    private void UpdateControl(float dt)
    {
        float currentAlt = transform.position.y;
        float rollError = WrapSignedAngle(_targetRoll - transform.eulerAngles.z);
        float pitchError = WrapSignedAngle(_targetPitch - transform.eulerAngles.x);
        float altError = _targetAltitude - currentAlt;
        Vector3 localAngVel = GetLocalAngularVelocity();
        float altVel = _rb.linearVelocity.y;

        if (Mathf.Abs(rollError) < AttitudeDeadbandDeg)
            rollError = 0f;
        if (Mathf.Abs(pitchError) < AttitudeDeadbandDeg)
            pitchError = 0f;

        float yawRateDeg = localAngVel.y * Mathf.Rad2Deg;
        _targetYawDeg = Mathf.Repeat(_targetYawDeg + _yawRateInput * dt, 360f);
        float yawError = WrapSignedAngle(_targetYawDeg - transform.eulerAngles.y);
        if (Mathf.Abs(yawError) < YawDeadbandDeg)
            yawError = 0f;

        if (Mathf.Abs(altError) < AltDeadband)
            altError = 0f;

        float altAccCmd = _altPID.Update(altError, dt) - altVel * AltVelocityDamping;
        float rollTorqueCmd = _rollPID.Update(rollError, dt) - (localAngVel.z * Mathf.Rad2Deg * RollRateDamping);
        float pitchTorqueCmd = _pitchPID.Update(pitchError, dt) - (localAngVel.x * Mathf.Rad2Deg * PitchRateDamping);
        float yawTorqueCmd = _yawPID.Update(yawError, dt) - (yawRateDeg * YawRateDamping);

        float g = Mathf.Abs(Physics.gravity.y);
        float liftCmd = _rb.mass * (g + altAccCmd);
        float baseNorm = Mathf.Clamp(liftCmd / (4f * MaxMotorThrust), 0f, 1f);

        float rollMix = Mathf.Clamp(rollTorqueCmd / Mathf.Max(1e-4f, MaxRollTorque), -1f, 1f);
        float pitchMix = Mathf.Clamp(pitchTorqueCmd / Mathf.Max(1e-4f, MaxPitchTorque), -1f, 1f);
        // Roll/Pitch mix for horizontal movement, yaw는 별도 토크로 제어
        // Front, Right, Back, Left (line-segment center layout)
        _motorThrust[0] = Mathf.Clamp01(baseNorm - pitchMix);
        _motorThrust[1] = Mathf.Clamp01(baseNorm + rollMix);
        _motorThrust[2] = Mathf.Clamp01(baseNorm + pitchMix);
        _motorThrust[3] = Mathf.Clamp01(baseNorm - rollMix);

        _yawTorqueCmd = Mathf.Clamp(yawTorqueCmd, -MaxYawTorque, MaxYawTorque);
    }

    private void ApplyForces()
    {
        for (int i = 0; i < 4; i++)
        {
            float thrust = _motorThrust[i] * MaxMotorThrust;
            Vector3 worldPos = transform.TransformPoint(_motorPositions[i]);
            Vector3 force = transform.up * thrust;
            _rb.AddForceAtPosition(force, worldPos, ForceMode.Force);
        }

        if (!Mathf.Approximately(_yawTorqueCmd, 0f))
        {
            _rb.AddRelativeTorque(Vector3.up * _yawTorqueCmd, ForceMode.Force);
        }
    }

    private void ConfigureMotorLayout()
    {
        if (AutoFitMotorLayoutFromCollider)
        {
            Collider col = GetComponent<Collider>();
            if (col is BoxCollider box)
            {
                Vector3 size = Vector3.Scale(box.size, transform.localScale);
                float halfX = Mathf.Max(0.05f, size.x * 0.5f);
                float halfZ = Mathf.Max(0.05f, size.z * 0.5f);
                float y = box.center.y + MotorHeightOffset;

                _motorPositions = new Vector3[]
                {
                    new Vector3(0f, y, halfZ),      // Front
                    new Vector3(halfX, y, 0f),      // Right
                    new Vector3(0f, y, -halfZ),     // Back
                    new Vector3(-halfX, y, 0f)      // Left
                };
                return;
            }
        }

        float a = ArmLength;
        _motorPositions = new Vector3[]
        {
            new Vector3(0f, MotorHeightOffset, a),     // Front
            new Vector3(a, MotorHeightOffset, 0f),      // Right
            new Vector3(0f, MotorHeightOffset, -a),     // Back
            new Vector3(-a, MotorHeightOffset, 0f)      // Left
        };
    }

    private void ConfigureInertiaTensor()
    {
        // rough box approximation if collider exists
        Collider col = GetComponent<Collider>();
        if (col is BoxCollider box)
        {
            Vector3 size = Vector3.Scale(box.size, transform.localScale);
            float x = Mathf.Max(size.x, 0.001f);
            float y = Mathf.Max(size.y, 0.001f);
            float z = Mathf.Max(size.z, 0.001f);
            _rb.inertiaTensor = new Vector3(
                _rb.mass * (y * y + z * z) / 12f,
                _rb.mass * (x * x + z * z) / 12f,
                _rb.mass * (x * x + y * y) / 12f
            );
            _rb.inertiaTensorRotation = Quaternion.identity;
        }
    }

    private void ClampAltitude()
    {
        Vector3 pos = transform.position;
        Vector3 vel = _rb.linearVelocity;

        if (pos.y < MinAltitude)
        {
            pos.y = MinAltitude;
            vel.y = Mathf.Max(0f, vel.y);
            _targetAltitude = Mathf.Max(_targetAltitude, MinAltitude);
        }
        else if (pos.y > MaxAltitude)
        {
            pos.y = MaxAltitude;
            vel.y = Mathf.Min(0f, vel.y);
            _targetAltitude = Mathf.Min(_targetAltitude, MaxAltitude);
        }

        transform.position = pos;
        _rb.linearVelocity = vel;
    }

    private void ClampVelocity()
    {
        Vector3 vel = _rb.linearVelocity;
        Vector3 horizontal = new Vector3(vel.x, 0f, vel.z);

        if (horizontal.sqrMagnitude > MaxHorizontalSpeed * MaxHorizontalSpeed)
        {
            horizontal = horizontal.normalized * MaxHorizontalSpeed;
            vel.x = horizontal.x;
            vel.z = horizontal.z;
        }

        vel.y = Mathf.Clamp(vel.y, -MaxVerticalSpeed, MaxVerticalSpeed);
        _rb.linearVelocity = vel;
    }

    private static float WrapSignedAngle(float angle)
    {
        angle = Mathf.Repeat(angle + 180f, 360f) - 180f;
        return angle;
    }
}

[System.Serializable]
public class PIDController
{
    private readonly float _kp;
    private readonly float _ki;
    private readonly float _kd;
    private float _integral;
    private float _prevError;
    private bool _firstUpdate = true;

    private const float IntegralClamp = 10f;

    public PIDController(float kp, float ki, float kd)
    {
        _kp = kp;
        _ki = ki;
        _kd = kd;
    }

    public float Update(float error, float dt)
    {
        if (_firstUpdate)
        {
            _prevError = error;
            _firstUpdate = false;
        }

        _integral = Mathf.Clamp(_integral + error * dt, -IntegralClamp, IntegralClamp);
        float derivative = (error - _prevError) / Mathf.Max(dt, 1e-6f);
        _prevError = error;

        return _kp * error + _ki * _integral + _kd * derivative;
    }

    public void Reset()
    {
        _integral = 0f;
        _prevError = 0f;
        _firstUpdate = true;
    }
}
