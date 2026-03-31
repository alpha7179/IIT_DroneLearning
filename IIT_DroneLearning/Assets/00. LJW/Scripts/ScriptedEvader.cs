using UnityEngine;

/// <summary>
/// ScriptedEvader — 규칙 기반 회피 드론 (Scripted Baseline)
///
/// 목적:
///   - Pursuer 팀이 Stage0/1 학습 상대로 사용할 수 있는 scripted 에이전트
///   - RL 학습 없이 단순 규칙으로 목표 지점을 향해 이동하며 장애물을 회피
///
/// 전략:
///   1. Goal-seeking : 목표 지점 방향으로 비례 속도 제어
///   2. Obstacle avoidance : 8방향 SphereCast로 장애물 감지 후 조향
///   3. Altitude hold  : 목표 고도 유지
///   4. (Optional) Pursuer escape : Pursuer가 가까우면 반대 방향 가중치 추가
///
/// 담당: 이재왕 (work/evader)
/// </summary>
[RequireComponent(typeof(DronePhysics))]
public class ScriptedEvader : MonoBehaviour
{
    // ─── Inspector 설정 ──────────────────────────────────────────────────────
    [Header("References")]
    [SerializeField] private Transform _goalTransform;
    [SerializeField] private Transform _pursuerTransform;   // null이면 Pursuer 회피 비활성

    [Header("Goal Seeking")]
    [SerializeField] private float _goalSpeed      = 0.6f;  // 목표 방향 추력 [0,1]
    [SerializeField] private float _targetAltitude = 5.0f;  // 유지할 비행 고도 (m)

    [Header("Obstacle Avoidance")]
    [SerializeField] private float _avoidRadius     = 1.5f; // SphereCast 반경
    [SerializeField] private float _avoidDistance   = 6.0f; // 감지 거리
    [SerializeField] private float _avoidStrength   = 1.2f; // 회피 가중치

    [Header("Pursuer Escape")]
    [SerializeField] private float _escapeDistance  = 10f;  // 이 거리 이내면 도망
    [SerializeField] private float _escapeStrength  = 0.8f; // 도망 가중치

    // ─── 내부 ────────────────────────────────────────────────────────────────
    private DronePhysics _dronePhysics;

    // 8방향 수평 회피 레이 방향
    private static readonly Vector3[] AvoidDirs = {
        Vector3.forward, Vector3.back, Vector3.left, Vector3.right,
        (Vector3.forward + Vector3.right).normalized,
        (Vector3.forward + Vector3.left).normalized,
        (Vector3.back    + Vector3.right).normalized,
        (Vector3.back    + Vector3.left).normalized,
    };

    void Awake()
    {
        _dronePhysics = GetComponent<DronePhysics>();
    }

    void FixedUpdate()
    {
        if (_goalTransform == null) return;

        Vector3 desiredDir = ComputeDesiredDirection();
        ApplyDesiredDirection(desiredDir);
    }

    // ─── 방향 계산 ───────────────────────────────────────────────────────────
    private Vector3 ComputeDesiredDirection()
    {
        Vector3 pos = transform.position;

        // 1) Goal-seeking
        Vector3 toGoal = _goalTransform.position - pos;
        Vector3 desired = toGoal.normalized;

        // 2) Obstacle avoidance (수평면 기준)
        Vector3 avoidance = Vector3.zero;
        foreach (Vector3 dir in AvoidDirs)
        {
            Vector3 worldDir = transform.TransformDirection(dir);
            if (Physics.SphereCast(pos, _avoidRadius, worldDir, out RaycastHit hit, _avoidDistance))
            {
                // 장애물에 가까울수록 강하게 반발
                float weight = 1f - (hit.distance / _avoidDistance);
                avoidance -= worldDir * weight;
            }
        }
        desired += avoidance * _avoidStrength;

        // 3) Pursuer escape
        if (_pursuerTransform != null)
        {
            Vector3 toPursuer = _pursuerTransform.position - pos;
            if (toPursuer.magnitude < _escapeDistance)
            {
                float escapeWeight = 1f - (toPursuer.magnitude / _escapeDistance);
                desired -= toPursuer.normalized * escapeWeight * _escapeStrength;
            }
        }

        return desired.normalized;
    }

    // ─── 움직임 적용 ─────────────────────────────────────────────────────────
    private void ApplyDesiredDirection(Vector3 desiredDir)
    {
        // 수평 이동 (로컬 좌표 → roll/pitch)
        Vector3 localDir = transform.InverseTransformDirection(desiredDir);
        float roll  = localDir.x * _goalSpeed;
        float pitch = localDir.z * _goalSpeed;

        // Yaw — 목표 방향으로 부드럽게 회전
        float yawError = Mathf.DeltaAngle(
            transform.eulerAngles.y,
            Mathf.Atan2(desiredDir.x, desiredDir.z) * Mathf.Rad2Deg
        );
        float yaw = Mathf.Clamp(yawError / 90f, -1f, 1f);

        // 고도 유지 → throttle
        float altError  = _targetAltitude - transform.position.y;
        float throttle  = Mathf.Clamp(altError / 3f, -1f, 1f);

        _dronePhysics.SetCommand(throttle, roll, pitch, yaw);
    }

    // ─── 공개 API (에피소드 리셋 시 외부에서 호출) ───────────────────────────
    public void ResetAgent(Vector3 spawnPos)
    {
        transform.position = spawnPos;
        _dronePhysics.ResetPhysics();
    }

    // ─── Editor 디버그 기즈모 ─────────────────────────────────────────────────
#if UNITY_EDITOR
    private void OnDrawGizmosSelected()
    {
        Gizmos.color = Color.cyan;
        Gizmos.DrawWireSphere(transform.position, _avoidRadius);

        if (_goalTransform != null)
        {
            Gizmos.color = Color.green;
            Gizmos.DrawLine(transform.position, _goalTransform.position);
        }

        if (_pursuerTransform != null)
        {
            Gizmos.color = Color.red;
            Gizmos.DrawWireSphere(transform.position, _escapeDistance);
        }
    }
#endif
}
