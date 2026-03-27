using UnityEngine;

/// <summary>
/// EvaderReward — 회피 드론 보상 함수 계산기
///
/// EvaderAgent에서 보상 로직을 분리하여 튜닝 편의성을 높인다.
/// Inspector에서 가중치를 실시간으로 조정할 수 있다.
///
/// 보상 구조:
///   R_total = w_goal * R_goal + w_survival * R_survival
///           + w_occlusion * R_occlusion + w_collision * R_collision
///           + w_time * R_time
///
/// 담당: 이재왕 (work/evader)
/// </summary>
public class EvaderReward : MonoBehaviour
{
    // ───────── Inspector 가중치 설정 ──────────────────────────────────────
    [Header("Goal Shaping")]
    [Tooltip("목표 지점 접근에 따른 potential-based shaping 계수")]
    [SerializeField] private float _goalShapingCoeff = 0.3f;

    [Header("Survival")]
    [Tooltip("매 스텝 생존 보너스 (Stage0에서는 0 권장 — timePenalty와 상쇄됨)")]
    [SerializeField] private float _survivalRewardPerStep = 0.0f;

    [Header("Occlusion (Stage1+)")]
    [Tooltip("Pursuer LOS 차단 성공 시 보너스 (0이면 비활성)")]
    [SerializeField] private float _occlusionBonus = 0.0f;   // Stage0에서는 0

    [Header("Path Guidance")]
    [Tooltip("목표 방향으로 Yaw 정렬 시 보너스 (Stage0 최단 직선 경로 유도). Stage1+에서는 0 권장)")]
    [SerializeField] private float _yawAlignCoeff = 0.001f;

    [Tooltip("목표 방향 실제 이동 속도 보상 (호버링 방지). dot(vel, toGoal) / maxObsSpeed × coeff")]
    [SerializeField] private float _velAlignCoeff = 0.01f;

    [Header("Observation Normalization")]
    [SerializeField] private float _maxObsSpeed = 10f;

    [Header("Time Penalty")]
    [Tooltip("스텝당 시간 페널티 (음수 유지)")]
    [SerializeField] private float _timePenaltyPerStep = -0.001f;

    // ───────── 내부 상태 ──────────────────────────────────────────────────
    private float _prevGoalDistance = float.MaxValue;
    private bool _prevPursuerVisible = false;
    private Transform _agentTransform;

    // ───────── 에피소드 초기화 ────────────────────────────────────────────
    private void Awake()
    {
        _agentTransform = transform;
    }

    private void OnEnable()
    {
        _prevGoalDistance = float.MaxValue;
        _prevPursuerVisible = false;
    }

    // ───────── 보상 계산 (EvaderAgent에서 매 스텝 호출) ──────────────────
    /// <summary>
    /// 스텝 보상을 계산하여 반환한다.
    /// Episode-end 보상(capture, goal, crash)은 EvaderAgent에서 직접 AddReward.
    /// </summary>
    public float ComputeStepReward(
        Vector3 agentPos,
        Vector3 goalPos,
        Vector3 pursuerPos,
        bool    isPursuerVisible,
        Vector3 agentVel)
    {
        float reward = 0f;

        // ── 목표 지점 접근 shaping ────────────────────────────────────────
        float currGoalDist = Vector3.Distance(agentPos, goalPos);
        if (_prevGoalDistance < float.MaxValue)
        {
            float distDelta = _prevGoalDistance - currGoalDist;
            reward += _goalShapingCoeff * distDelta;
        }
        _prevGoalDistance = currGoalDist;

        // ── 생존 보너스 ──────────────────────────────────────────────────
        reward += _survivalRewardPerStep;

        // ── LOS 차단 보너스 (Stage1+) ─────────────────────────────────────
        if (_occlusionBonus > 0f)
        {
            // 이번 스텝에 LOS가 차단되었을 때만 보너스 (지속 보너스는 숨기만 하는 문제 유발)
            bool justHidden = _prevPursuerVisible && !isPursuerVisible;
            if (justHidden)
                reward += _occlusionBonus;
        }
        _prevPursuerVisible = isPursuerVisible;

        // ── Yaw 정렬 보상 (목표 방향으로 드론 정면이 향할수록 +보상) ────────
        if (_yawAlignCoeff > 0f && currGoalDist > 0.5f)
        {
            Vector3 fwd = _agentTransform.forward; fwd.y = 0f;
            Vector3 dir = goalPos - agentPos; dir.y = 0f;
            if (fwd.sqrMagnitude > 0f && dir.sqrMagnitude > 0f)
            {
                float alignment = Vector3.Dot(fwd.normalized, dir.normalized);
                reward += _yawAlignCoeff * alignment;
            }
        }

        // ── 속도-목표 정렬 보상 (호버링 방지, 최단 경로 유도) ────────────────
        if (_velAlignCoeff > 0f && currGoalDist > 0.5f)
        {
            Vector3 toGoalDir = (goalPos - agentPos).normalized;
            float velTowardGoal = Vector3.Dot(agentVel, toGoalDir) / _maxObsSpeed;
            reward += _velAlignCoeff * velTowardGoal;
        }

        // ── 시간 페널티 ──────────────────────────────────────────────────
        reward += _timePenaltyPerStep;

        return reward;
    }

    // ───────── 디버그 헬퍼 ───────────────────────────────────────────────
    /// <summary>Inspector에서 현재 가중치 설정 확인용</summary>
    public string GetRewardSummary()
    {
        return $"[Reward] goalShaping={_goalShapingCoeff} survival={_survivalRewardPerStep} " +
               $"occlusion={_occlusionBonus} timePenalty={_timePenaltyPerStep}";
    }
}
