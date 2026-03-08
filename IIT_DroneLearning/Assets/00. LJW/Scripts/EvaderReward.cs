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
    [SerializeField] private float _goalShapingCoeff = 0.02f;

    [Header("Survival")]
    [Tooltip("매 스텝 생존 보너스 (+)")]
    [SerializeField] private float _survivalRewardPerStep = 0.001f;

    [Header("Occlusion (Stage1+)")]
    [Tooltip("Pursuer LOS 차단 성공 시 보너스 (0이면 비활성)")]
    [SerializeField] private float _occlusionBonus = 0.0f;   // Stage0에서는 0

    [Header("Time Penalty")]
    [Tooltip("스텝당 시간 페널티 (음수 유지)")]
    [SerializeField] private float _timePenaltyPerStep = -0.001f;

    // ───────── 내부 상태 ──────────────────────────────────────────────────
    private float _prevGoalDistance = float.MaxValue;
    private bool _prevPursuerVisible = false;

    // ───────── 에피소드 초기화 ────────────────────────────────────────────
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
        bool isPursuerVisible)
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
