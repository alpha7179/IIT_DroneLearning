using UnityEngine;

/// <summary>
/// PursuerReward — 카메라 유도 기반 추적 드론 보상 함수 계산기.
///
/// PursuerAgent에서 step reward를 분리해 Inspector 튜닝을 쉽게 만든다.
/// 주요 shaping은 다음 네 가지다.
///   1. Evader와의 거리 감소
///   2. Evader를 향한 상대 closing speed
///   3. 카메라 내 시야 유지
///   4. 화면 중심 정렬
/// </summary>
public class PursuerReward : MonoBehaviour
{
    [Header("Distance Shaping")]
    [Tooltip("Evader와의 거리 감소량에 곱해지는 shaping 계수")]
    [SerializeField] private float _distanceShapingCoeff = 0.12f;

    [Header("Close Pursuit")]
    [Tooltip("Evader에 가까울수록 추가되는 bounded reward 계수")]
    [SerializeField] private float _proximityRewardCoeff = 0.025f;

    [Tooltip("근접 reward가 포화되는 기준 거리")]
    [SerializeField] private float _proximityRewardDistance = 10f;

    [Header("Catch Focus")]
    [Tooltip("catch 직전 근접 구간에서 추가로 강조되는 reward 계수")]
    [SerializeField] private float _nearCatchRewardCoeff = 0.03f;

    [Tooltip("catch reward를 강조할 근접 반경")]
    [SerializeField] private float _nearCatchDistance = 4f;

    [Header("Approach Guidance")]
    [Tooltip("Evader를 향한 상대 closing speed 보상 계수")]
    [SerializeField] private float _closingSpeedCoeff = 0.01f;

    [Header("Camera Tracking")]
    [Tooltip("Evader가 카메라 시야 안에 있을 때 받는 스텝 보상")]
    [SerializeField] private float _visibilityRewardPerStep = 0.002f;

    [Tooltip("화면 중심에 Evader를 둘수록 증가하는 보상 계수")]
    [SerializeField] private float _viewportCenterCoeff = 0.008f;

    [Tooltip("한 번 보던 Evader를 놓친 순간에만 부여되는 penalty")]
    [SerializeField] private float _lostTargetPenalty = -0.03f;

    [Header("Observation Normalization")]
    [SerializeField] private float _maxObsSpeed = 10f;

    [Header("Time Penalty")]
    [Tooltip("스텝당 시간 패널티 (음수 유지)")]
    [SerializeField] private float _timePenaltyPerStep = -0.0005f;

    [Header("Reward Stabilization")]
    [Tooltip("한 스텝에서 반영할 거리 변화량 상한")]
    [SerializeField] private float _maxDistanceDeltaPerStep = 0.5f;

    [Tooltip("step reward 총합의 절대값 상한")]
    [SerializeField] private float _maxStepRewardMagnitude = 0.06f;

    [Header("Terminal Rewards")]
    [SerializeField] private float _captureReward = 2.0f;
    [SerializeField] private float _evaderGoalPenalty = -0.35f;
    [SerializeField] private float _timeoutPenalty = -0.08f;
    [SerializeField] private float _crashPenalty = -1.0f;

    private float _prevTargetDistance = float.MaxValue;
    private bool _wasTargetVisible;

    private void OnEnable()
    {
        ResetEpisodeState();
    }

    private void OnValidate()
    {
        _maxObsSpeed = Mathf.Max(0.1f, _maxObsSpeed);
        _proximityRewardDistance = Mathf.Max(0.1f, _proximityRewardDistance);
        _nearCatchDistance = Mathf.Max(0.1f, _nearCatchDistance);
        _maxDistanceDeltaPerStep = Mathf.Max(0.01f, _maxDistanceDeltaPerStep);
        _maxStepRewardMagnitude = Mathf.Max(0.001f, _maxStepRewardMagnitude);
    }

    public void ResetEpisodeState()
    {
        _prevTargetDistance = float.MaxValue;
        _wasTargetVisible = false;
    }

    public float ComputeStepReward(
        bool hasTarget,
        Vector3 agentPos,
        Vector3 targetPos,
        Vector3 agentVel,
        Vector3 targetVel,
        bool isTargetVisible,
        Vector2 viewportOffset)
    {
        float reward = _timePenaltyPerStep;

        if (!hasTarget)
        {
            ResetEpisodeState();
            return ClampStepReward(reward);
        }

        float currTargetDist = Vector3.Distance(agentPos, targetPos);
        if (_prevTargetDistance < float.MaxValue)
        {
            float distDelta = Mathf.Clamp(
                _prevTargetDistance - currTargetDist,
                -_maxDistanceDeltaPerStep,
                _maxDistanceDeltaPerStep);
            reward += _distanceShapingCoeff * distDelta;
        }
        _prevTargetDistance = currTargetDist;

        float proximity = 1f - Mathf.Clamp01(currTargetDist / _proximityRewardDistance);
        reward += _proximityRewardCoeff * proximity;

        float nearCatch = 1f - Mathf.Clamp01(currTargetDist / _nearCatchDistance);
        reward += _nearCatchRewardCoeff * nearCatch * nearCatch;

        if (_closingSpeedCoeff > 0f && currTargetDist > 0.1f)
        {
            Vector3 toTargetDir = (targetPos - agentPos).normalized;
            float closingSpeed = Mathf.Clamp(
                Vector3.Dot(agentVel - targetVel, toTargetDir) / _maxObsSpeed,
                -1f,
                1f);
            reward += _closingSpeedCoeff * closingSpeed;
        }

        if (isTargetVisible)
        {
            reward += _visibilityRewardPerStep;

            float centeredness = 1f - Mathf.Clamp01(viewportOffset.magnitude);
            reward += _viewportCenterCoeff * centeredness;
        }
        else if (_wasTargetVisible)
        {
            reward += _lostTargetPenalty;
        }

        _wasTargetVisible = isTargetVisible;

        return ClampStepReward(reward);
    }

    private float ClampStepReward(float reward)
    {
        return Mathf.Clamp(reward, -_maxStepRewardMagnitude, _maxStepRewardMagnitude);
    }

    public float CaptureReward => _captureReward;
    public float EvaderGoalPenalty => _evaderGoalPenalty;
    public float TimeoutPenalty => _timeoutPenalty;
    public float CrashPenalty => _crashPenalty;
}
