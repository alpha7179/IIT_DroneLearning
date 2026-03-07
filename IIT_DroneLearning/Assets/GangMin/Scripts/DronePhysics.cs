using UnityEngine;

/// <summary>
/// 드론의 6-DOF 물리 기반 이동/회전 컴포넌트
/// 이동 3축 (X, Y, Z) + 회전 3축 (Yaw, Pitch, Roll)
/// </summary>
public class DronePhysics : MonoBehaviour
{
    [Header("비행 파라미터")]
    public float ThrustForce = 10f;
    public float MaxSpeed = 8f;
    public float MaxAltitude = 30f;
    public float MinAltitude = 1f;
    public float DragCoefficient = 0.95f;

    [Header("회전 파라미터")]
    public float YawSpeed = 90f;    // 초당 Yaw 회전 각도
    public float PitchSpeed = 45f;  // 초당 Pitch 회전 각도
    public float RollSpeed = 45f;   // 초당 Roll 회전 각도

    [Header("회전 제한")]
    public float MaxPitchAngle = 45f;  // Pitch 최대 각도
    public float MaxRollAngle = 45f;   // Roll 최대 각도

    private Rigidbody _rb;

    void Awake()
    {
        _rb = GetComponent<Rigidbody>();
        _rb.useGravity = false;
        _rb.linearDamping = 1f;
        _rb.angularDamping = 5f;
        _rb.freezeRotation = true; // 물리 회전 고정, 수동 제어
    }

    /// <summary>
    /// 이동 처리 (드론 로컬 좌표 기준)
    /// </summary>
    public void ApplyMovement(float x, float y, float z)
    {
        Vector3 localMove = new Vector3(x, y, z);
        Vector3 worldMove = transform.TransformDirection(localMove) * ThrustForce;

        _rb.AddForce(worldMove, ForceMode.Force);

        // 속도 제한
        if (_rb.linearVelocity.magnitude > MaxSpeed)
            _rb.linearVelocity = _rb.linearVelocity.normalized * MaxSpeed;

        // 공기저항
        _rb.linearVelocity *= DragCoefficient;

        // 고도 제한
        Vector3 pos = transform.position;
        pos.y = Mathf.Clamp(pos.y, MinAltitude, MaxAltitude);
        transform.position = pos;
    }

    /// <summary>
    /// Yaw 회전 (좌우 회전, 제한 없음)
    /// </summary>
    public void ApplyYaw(float direction)
    {
        float yawAmount = direction * YawSpeed * Time.fixedDeltaTime;
        transform.Rotate(0f, yawAmount, 0f, Space.World);
    }

    /// <summary>
    /// Pitch 회전 (앞뒤 기울기, 각도 제한 있음)
    /// </summary>
    public void ApplyPitch(float direction)
    {
        float currentPitch = NormalizeAngle(transform.eulerAngles.x);

        // 최대 각도 초과 시 회전 방지
        if (direction > 0 && currentPitch >= MaxPitchAngle) return;
        if (direction < 0 && currentPitch <= -MaxPitchAngle) return;

        float pitchAmount = direction * PitchSpeed * Time.fixedDeltaTime;
        transform.Rotate(pitchAmount, 0f, 0f, Space.Self);
    }

    /// <summary>
    /// Roll 회전 (좌우 기울기, 각도 제한 있음)
    /// </summary>
    public void ApplyRoll(float direction)
    {
        float currentRoll = NormalizeAngle(transform.eulerAngles.z);

        // 최대 각도 초과 시 회전 방지
        if (direction > 0 && currentRoll >= MaxRollAngle) return;
        if (direction < 0 && currentRoll <= -MaxRollAngle) return;

        float rollAmount = direction * RollSpeed * Time.fixedDeltaTime;
        transform.Rotate(0f, 0f, rollAmount, Space.Self);
    }

    /// <summary>
    /// 현재 속도 반환
    /// </summary>
    public Vector3 GetVelocity() => _rb.linearVelocity;

    /// <summary>
    /// 현재 회전 각도 반환 (Yaw, Pitch, Roll)
    /// </summary>
    public Vector3 GetRotation()
    {
        return new Vector3(
            NormalizeAngle(transform.eulerAngles.x), // Pitch
            NormalizeAngle(transform.eulerAngles.y), // Yaw
            NormalizeAngle(transform.eulerAngles.z)  // Roll
        );
    }

    /// <summary>
    /// 물리 상태 초기화 (에피소드 리셋 시 호출)
    /// </summary>
    public void ResetPhysics()
    {
        _rb.linearVelocity = Vector3.zero;
        _rb.angularVelocity = Vector3.zero;
        transform.rotation = Quaternion.identity; // 회전 초기화
    }

    /// <summary>
    /// Unity Euler 각도 (0~360)를 -180~180으로 정규화
    /// </summary>
    private float NormalizeAngle(float angle)
    {
        if (angle > 180f) angle -= 360f;
        return angle;
    }
}