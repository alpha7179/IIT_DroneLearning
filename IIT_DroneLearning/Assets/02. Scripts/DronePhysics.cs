using UnityEngine;

/// <summary>
/// 쿼드콥터 4모터 + PID 기반 드론 물리 제어기
///
/// 설계 근거:
///   simeonradivoev/Quadcopter-Controller 및
///   Habrador Unity PID 드론 구현을 참조하여
///   검증된 구조로 통일
///
/// 핵심 원칙:
///   1. Unity 중력 ON (useGravity = true)
///      Altitude PID의 Ki 적분항이 중력을 자연스럽게 보상
///   2. 4모터 AddForceAtPosition → Roll/Pitch 토크 물리적 자동 발생
///      (τ = r × F, Unity PhysX가 자동 계산)
///   3. Yaw Ki = 0 (모든 레퍼런스 공통 → 적분 폭주 방지)
///   4. Yaw PID 제거 → AngularDrag 감쇠 + RL 직접 토크
///
/// 모터 배치 (탑뷰):
///   M1(앞왼) ---- M2(앞오)
///      |               |
///   M3(뒤왼) ---- M4(뒤오)
///
/// RL 액션 (DroneAgent에서 SetCommand 호출):
///   action[0] = thrust  ∈ [-1,1]  고도 목표 변화
///   action[1] = roll    ∈ [-1,1]  Roll 목표 각도
///   action[2] = pitch   ∈ [-1,1]  Pitch 목표 각도
///   action[3] = yaw     ∈ [-1,1]  Yaw 직접 토크
/// </summary>
[RequireComponent(typeof(Rigidbody))]
public class DronePhysics : MonoBehaviour
{
    // ──────────────────────────────────────────
    // Inspector 파라미터
    // ──────────────────────────────────────────

    [Header("Rigidbody 설정")]
    public float Mass        = 1.0f;
    public float LinearDrag  = 1.0f;   // 레퍼런스 권장값: 1.0
    public float AngularDrag = 5.0f;   // 높게 유지 → Yaw 진동 억제

    [Header("모터 설정")]
    public float MaxMotorThrust = 6.0f;
    // 4모터 합산 최대 24N > 중력 9.81N 이어야 hover 가능
    public float ArmLength      = 0.25f;  // 모터~중심 거리 (m)

    [Header("고도 제한")]
    public float MinAltitude = 0.5f;
    public float MaxAltitude = 50.0f;

    [Header("속도 제한")]
    public float MaxHorizontalSpeed = 8.0f;
    public float MaxVerticalSpeed   = 5.0f;

    [Header("목표값 범위")]
    public float MaxAltitudeStep = 2.0f;   // 1회 명령당 고도 목표 변화량 (m)
    public float MaxRollAngle    = 30.0f;  // 최대 Roll 목표 각도 (도)
    public float MaxPitchAngle   = 30.0f;  // 최대 Pitch 목표 각도 (도)
    public float YawTorqueScale  = 3.0f;   // Yaw 직접 토크 스케일

    [Header("PID - Altitude")]
    [Tooltip("Ki 적분항이 중력(9.81N)을 자동 보상 → hoverNorm 불필요")]
    public float AltKp = 3.0f;
    public float AltKi = 0.5f;
    public float AltKd = 2.0f;

    [Header("PID - Roll")]
    public float RollKp = 2.0f;
    public float RollKi = 0.0f;  // 모든 레퍼런스 공통: Ki=0 (진동 방지)
    public float RollKd = 0.5f;

    [Header("PID - Pitch")]
    public float PitchKp = 2.0f;
    public float PitchKi = 0.0f; // 모든 레퍼런스 공통: Ki=0 (진동 방지)
    public float PitchKd = 0.5f;

    // Yaw PID 없음 → AngularDrag + RL 직접 토크로 처리

    // ──────────────────────────────────────────
    // 내부 상태
    // ──────────────────────────────────────────
    private Rigidbody     _rb;
    private PIDController _altPID;
    private PIDController _rollPID;
    private PIDController _pitchPID;

    private float _targetAltitude;
    private float _targetRoll  = 0f;
    private float _targetPitch = 0f;
    private float _yawTorque   = 0f;

    private float[]   _motorThrust    = new float[4];
    private Vector3[] _motorPositions;

    // ──────────────────────────────────────────
    // 초기화
    // ──────────────────────────────────────────
    private void Awake()
    {
        _rb                = GetComponent<Rigidbody>();
        _rb.mass           = Mass;
        _rb.linearDamping  = LinearDrag;
        _rb.angularDamping = AngularDrag;
        _rb.useGravity     = true;   // Unity 중력 ON
        _rb.freezeRotation = false;

        // 관성 텐서 명시 설정
        // Collider 형태(큐브 등)에 의존하지 않고 실제 쿼드콥터 비율로 고정
        // Ixx ≈ Iyy (Roll/Pitch 대칭) << Izz (Yaw)
        _rb.inertiaTensor         = new Vector3(0.02f, 0.04f, 0.02f);
        _rb.inertiaTensorRotation = Quaternion.identity;

        // 모터 위치 (로컬 좌표)
        float a = ArmLength;
        _motorPositions = new Vector3[]
        {
            new Vector3(-a, 0f,  a),  // M1: 앞왼
            new Vector3( a, 0f,  a),  // M2: 앞오
            new Vector3(-a, 0f, -a),  // M3: 뒤왼
            new Vector3( a, 0f, -a),  // M4: 뒤오
        };

        _altPID   = new PIDController(AltKp,   AltKi,   AltKd);
        _rollPID  = new PIDController(RollKp,  RollKi,  RollKd);
        _pitchPID = new PIDController(PitchKp, PitchKi, PitchKd);

        _targetAltitude = transform.position.y;
    }

    // ──────────────────────────────────────────
    // 물리 업데이트
    // ──────────────────────────────────────────
    private void FixedUpdate()
    {
        float dt = Time.fixedDeltaTime;
        ComputePID(dt);
        ApplyMotorForces();
        ClampAltitude();
        ClampVelocity();
    }

    // ──────────────────────────────────────────
    // RL 인터페이스
    // ──────────────────────────────────────────

    /// <summary>
    /// 고수준 명령 (DroneAgent의 OnActionReceived에서 호출)
    /// thrust : 고도 목표를 높이거나 낮춤       [-1, 1]
    /// roll   : 목표 Roll 각도 설정             [-1, 1]
    /// pitch  : 목표 Pitch 각도 설정            [-1, 1]
    /// yaw    : Yaw 직접 토크 (PID 없음)        [-1, 1]
    /// </summary>
    public void SetCommand(float thrust, float roll, float pitch, float yaw)
    {
        _targetAltitude = Mathf.Clamp(
            _targetAltitude + thrust * MaxAltitudeStep,
            MinAltitude,
            MaxAltitude
        );

        _targetRoll  = roll  * MaxRollAngle;
        _targetPitch = pitch * MaxPitchAngle;
        _yawTorque   = yaw   * YawTorqueScale;
    }

    // ──────────────────────────────────────────
    // 상태 조회 (DroneAgent의 CollectObservations에서 사용)
    // ──────────────────────────────────────────
    public Vector3 GetVelocity()             => _rb.linearVelocity;
    public Vector3 GetLocalAngularVelocity() => transform.InverseTransformDirection(_rb.angularVelocity);
    public Vector3 GetRotationEuler()        => transform.eulerAngles;

    public void ResetPhysics()
    {
        _rb.linearVelocity  = Vector3.zero;
        _rb.angularVelocity = Vector3.zero;
        transform.rotation  = Quaternion.identity;

        _targetAltitude = transform.position.y;
        _targetRoll     = 0f;
        _targetPitch    = 0f;
        _yawTorque      = 0f;

        _altPID.Reset();
        _rollPID.Reset();
        _pitchPID.Reset();

        for (int i = 0; i < 4; i++)
            _motorThrust[i] = 0f;
    }

    // ──────────────────────────────────────────
    // 내부 물리 계산
    // ──────────────────────────────────────────

    /// <summary>
    /// PID 계산 → 모터 믹싱
    ///
    /// [Altitude PID]
    ///   오차 = 목표 고도 - 현재 고도
    ///   Ki 적분항이 시간이 지남에 따라 중력(mg = 9.81N)을 자동 보상
    ///   → 별도 hoverNorm 계산 불필요
    ///   → 논문 수식: u_t = Kp*e_z + Ki*∫e_z dt + Kd*ė_z
    ///
    /// [Roll/Pitch PID]
    ///   오차 = 목표 각도 - 현재 각도
    ///   → 모터 불균형 r, p 결정
    ///   → AddForceAtPosition이 τ = r × F 자동 계산
    ///   → Ki = 0 (모든 레퍼런스 공통: 각도 진동 방지)
    /// </summary>
    private void ComputePID(float dt)
    {
        float currentAlt   = transform.position.y;
        float currentRoll  = WrapAngle(transform.eulerAngles.z);
        float currentPitch = WrapAngle(transform.eulerAngles.x);

        float altError   = _targetAltitude - currentAlt;
        float rollError  = _targetRoll     - currentRoll;
        float pitchError = _targetPitch    - currentPitch;

        float altOut   = _altPID.Update(altError,   dt);
        float rollOut  = _rollPID.Update(rollError,  dt);
        float pitchOut = _pitchPID.Update(pitchError, dt);

        // 클램프: t ∈ [0,1], r/p ∈ [-0.3, 0.3]
        float t = Mathf.Clamp(altOut,   0f,    1.0f);
        float r = Mathf.Clamp(rollOut, -0.3f,  0.3f);
        float p = Mathf.Clamp(pitchOut,-0.3f,  0.3f);

        // 모터 믹싱
        //           Thrust  Roll  Pitch
        _motorThrust[0] = Mathf.Clamp01(t - r + p);  // M1 앞왼
        _motorThrust[1] = Mathf.Clamp01(t + r + p);  // M2 앞오
        _motorThrust[2] = Mathf.Clamp01(t - r - p);  // M3 뒤왼
        _motorThrust[3] = Mathf.Clamp01(t + r - p);  // M4 뒤오
    }

    /// <summary>
    /// 각 모터 위치에서 Force 적용
    ///
    /// AddForceAtPosition(F, r_world):
    ///   Unity PhysX → τ = (r - r_cm) × F 자동 계산
    ///   → Roll/Pitch 토크 물리적으로 발생
    ///   → 별도 토크 계산 불필요
    ///
    /// Yaw:
    ///   직접 AddTorque (PID 없음)
    ///   AngularDrag가 감쇠 담당
    /// </summary>
    private void ApplyMotorForces()
    {
        for (int i = 0; i < 4; i++)
        {
            float   thrustN  = _motorThrust[i] * MaxMotorThrust;
            Vector3 worldPos = transform.TransformPoint(_motorPositions[i]);
            Vector3 forceDir = transform.up * thrustN;

            _rb.AddForceAtPosition(forceDir, worldPos, ForceMode.Force);
        }

        // Yaw 직접 토크
        _rb.AddTorque(transform.up * _yawTorque, ForceMode.Force);
    }

    private void ClampAltitude()
    {
        Vector3 pos = transform.position;

        if (pos.y < MinAltitude)
        {
            pos.y              = MinAltitude;
            transform.position = pos;
            Vector3 vel        = _rb.linearVelocity;
            if (vel.y < 0f) { vel.y = 0f; _rb.linearVelocity = vel; }
        }
        else if (pos.y > MaxAltitude)
        {
            pos.y              = MaxAltitude;
            transform.position = pos;
            Vector3 vel        = _rb.linearVelocity;
            if (vel.y > 0f) { vel.y = 0f; _rb.linearVelocity = vel; }
        }
    }

    private void ClampVelocity()
    {
        Vector3 vel        = _rb.linearVelocity;
        Vector3 horizontal = new Vector3(vel.x, 0f, vel.z);

        if (horizontal.magnitude > MaxHorizontalSpeed)
        {
            horizontal = horizontal.normalized * MaxHorizontalSpeed;
            vel.x      = horizontal.x;
            vel.z      = horizontal.z;
        }

        vel.y              = Mathf.Clamp(vel.y, -MaxVerticalSpeed, MaxVerticalSpeed);
        _rb.linearVelocity = vel;
    }

    private float WrapAngle(float angle)
    {
        if (angle > 180f) angle -= 360f;
        return angle;
    }
}

// ──────────────────────────────────────────────────────────────
// 범용 PID 컨트롤러
//
// 이산화 공식 (FixedUpdate 기준):
//   u[k] = Kp * e[k]
//         + Ki * Σ(e[j] * Δt)   (적분, Anti-windup 클램프)
//         + Kd * (e[k]-e[k-1]) / Δt  (미분)
// ──────────────────────────────────────────────────────────────
[System.Serializable]
public class PIDController
{
    private float _kp, _ki, _kd;
    private float _integral    = 0f;
    private float _prevError   = 0f;
    private bool  _firstUpdate = true;

    private const float IntegralClamp = 10.0f;

    public PIDController(float kp, float ki, float kd)
    {
        _kp = kp; _ki = ki; _kd = kd;
    }

    public float Update(float error, float dt)
    {
        if (_firstUpdate)
        {
            _prevError   = error;
            _firstUpdate = false;
        }

        _integral = Mathf.Clamp(
            _integral + error * dt,
            -IntegralClamp, IntegralClamp
        );

        float derivative = (error - _prevError) / Mathf.Max(dt, 1e-5f);
        _prevError = error;

        return _kp * error + _ki * _integral + _kd * derivative;
    }

    public void Reset()
    {
        _integral    = 0f;
        _prevError   = 0f;
        _firstUpdate = true;
    }
}
