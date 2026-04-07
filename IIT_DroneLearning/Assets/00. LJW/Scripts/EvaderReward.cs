using UnityEngine;

/// <summary>
/// EvaderReward — 회피 드론 보상 함수 계산기
///
/// 보상 구조 (Stage1-A):
///   R_step = R_goal_shaping + R_vel_obstacle_aware + R_obstacle + R_height + R_time
///   Terminal: +1.0 (goal), -1.0 (crash/capture), 0 (timeout)
///   Checkpoint: +checkpointReward 별도 (EvaderAgent에서 직접 AddReward)
///
///   R_goal_shaping        = goalShapingCoeff × (prevDist - currDist)
///   R_vel_obstacle_aware  = velAlignCoeff × dot(localVel, targetDir)
///                           targetDir: 장애물 없을 때 → goal방향,
///                                       장애물 감지 시 → 척력+goal 블렌드 (회피 우선)
///   R_obstacle            = -proximityCoeff × Σ(1 - d_i/threshold)  [26 rays]
///   R_height              = heightPenaltyCoeff × max(0, y - warningY)
///   R_time                = timePenaltyPerStep  (-0.003: 호버링 명확히 불리하게)
///
/// 담당: 이재왕 (work/evader)
/// </summary>
public class EvaderReward : MonoBehaviour
{
    [System.Serializable]
    public struct Stage1RewardProfile
    {
        public float VelAlignCoeff;
        public float GoalPriorityDist;
        public float ProximityCoeff;
        public float ProximityThreshold;
        public float NearGoalObstaclePenaltyScale;
        public float MiddleBottomRayPenaltyWeight;
        public float BottomRayPenaltyWeight;
    }

    public struct RewardBreakdown
    {
        public float Total;
        public float GoalShaping;
        public float Survival;
        public float Occlusion;
        public float YawAlign;
        public float VelocityAlign;
        public float ObstaclePenalty;
        public float HeightPenalty;
        public float TimePenalty;
        public float ObstacleSum;
        public float GoalDistance;
        public bool  InGoalPriorityZone;

        public string DominantPenalty()
        {
            float obstacleAbs = Mathf.Abs(ObstaclePenalty);
            float heightAbs   = Mathf.Abs(HeightPenalty);
            float timeAbs     = Mathf.Abs(TimePenalty);

            if (obstacleAbs >= heightAbs && obstacleAbs >= timeAbs)
                return "obstacle";
            if (heightAbs >= obstacleAbs && heightAbs >= timeAbs)
                return "height";
            return "time";
        }
    }

    // ───────── Inspector 가중치 설정 ──────────────────────────────────────
    [Header("Goal Shaping")]
    [Tooltip("목표 지점 접근에 따른 potential-based shaping 계수")]
    [SerializeField] private float _goalShapingCoeff = 0.3f;

    [Header("Survival")]
    [Tooltip("매 스텝 생존 보너스 (0 권장 — timePenalty와 상쇄됨)")]
    [SerializeField] private float _survivalRewardPerStep = 0.0f;

    [Header("Occlusion (Stage1+)")]
    [Tooltip("Pursuer LOS 차단 성공 시 보너스 (0이면 비활성)")]
    [SerializeField] private float _occlusionBonus = 0.0f;

    [Header("Path Guidance — Obstacle-Aware Velocity")]
    [Tooltip("장애물 없을 때 goal 방향 / 장애물 감지 시 회피방향 합산 속도 보상.\n" +
             "Middle 레이(9-16) 척력 벡터와 goal 방향을 장애물 강도에 따라 블렌드.")]
    [SerializeField] private float _velAlignCoeff = 0.005f;
    [Tooltip("이 거리(m) 이내에서는 척력 블렌드를 무시하고 순수 goal 방향 보상만 적용 (goal 근처 뒤로 물러남 방지)")]
    [SerializeField] private float _goalPriorityDist = 6f;

    [Tooltip("Yaw 정렬 보상 (Stage1+에서는 0 권장)")]
    [SerializeField] private float _yawAlignCoeff = 0.0f;

    [Header("Observation Normalization")]
    [SerializeField] private float _maxObsSpeed = 10f;

    [Header("Obstacle Proximity Penalty")]
    [Tooltip("이 정규화 거리 미만 레이 감지 시 페널티 합산 시작 (0~1)")]
    [SerializeField] private float _proximityThreshold = 0.15f;
    [Tooltip("레이 1개당 최대 페널티 기여값. 26개 합산되므로 작게 유지")]
    [SerializeField] private float _proximityCoeff = 0.002f;
    [Tooltip("goalPriorityDist 이내에서 장애물 페널티 최소 스케일 (0~1). goal 근처 급선회 완화")]
    [SerializeField] private float _nearGoalObstaclePenaltyScale = 0.35f;
    [Tooltip("Top 레이(0) 장애물 페널티 가중치")]
    [SerializeField] private float _topRayPenaltyWeight = 0.1f;
    [Tooltip("Top-Middle 레이(1~8) 장애물 페널티 가중치")]
    [SerializeField] private float _topMiddleRayPenaltyWeight = 0.35f;
    [Tooltip("Middle 레이(9~16) 장애물 페널티 가중치")]
    [SerializeField] private float _middleRayPenaltyWeight = 1.0f;
    [Tooltip("Middle-Bottom 레이(17~24) 장애물 페널티 가중치")]
    [SerializeField] private float _middleBottomRayPenaltyWeight = 0.15f;
    [Tooltip("Bottom 레이(25) 장애물 페널티 가중치")]
    [SerializeField] private float _bottomRayPenaltyWeight = 0.0f;

    [Header("Height Soft Penalty")]
    [Tooltip("이 고도(m) 초과 시 페널티 시작 (벽 높이 20m 기준)")]
    [SerializeField] private float _heightWarningY = 17f;
    [Tooltip("고도 경고 구간에서 초과 1m당 페널티 (음수 유지)")]
    [SerializeField] private float _heightPenaltyCoeff = -0.02f;

    [Header("Time Penalty")]
    [Tooltip("스텝당 시간 페널티 (음수 유지). 큰 맵에서는 지나친 rushing을 막지 않도록 약하게 유지")]
    [SerializeField] private float _timePenaltyPerStep = -0.001f;

    // ───────── Middle 레이어 방향 매핑 ────────────────────────────────────
    // DroneSensorSystem 인덱스 9~16: Middle 레이어 (수평 8방위, 드론 로컬 좌표계)
    // 방위각: N=0°(+Z), NE=45°, E=90°(+X), SE=135°, S=180°(-Z), SW=225°, W=270°(-X), NW=315°
    private static readonly Vector3[] MiddleRayDirs =
    {
        new Vector3( 0f,      0f,  1f     ),  // 9:  N
        new Vector3( 0.707f,  0f,  0.707f ),  // 10: NE
        new Vector3( 1f,      0f,  0f     ),  // 11: E
        new Vector3( 0.707f,  0f, -0.707f ),  // 12: SE
        new Vector3( 0f,      0f, -1f     ),  // 13: S
        new Vector3(-0.707f,  0f, -0.707f ),  // 14: SW
        new Vector3(-1f,      0f,  0f     ),  // 15: W
        new Vector3(-0.707f,  0f,  0.707f ),  // 16: NW
    };

    // ───────── 내부 상태 ──────────────────────────────────────────────────
    private float     _prevGoalDistance  = float.MaxValue;
    private bool      _prevPursuerVisible = false;
    private Transform _agentTransform;

    public void ApplyStage1Profile(Stage1RewardProfile profile)
    {
        _velAlignCoeff = profile.VelAlignCoeff;
        _goalPriorityDist = profile.GoalPriorityDist;
        _proximityCoeff = profile.ProximityCoeff;
        _proximityThreshold = profile.ProximityThreshold;
        _nearGoalObstaclePenaltyScale = profile.NearGoalObstaclePenaltyScale;
        _middleBottomRayPenaltyWeight = profile.MiddleBottomRayPenaltyWeight;
        _bottomRayPenaltyWeight = profile.BottomRayPenaltyWeight;
    }

    // ───────── 에피소드 초기화 ────────────────────────────────────────────
    private void Awake() => _agentTransform = transform;

    private void OnEnable()
    {
        _prevGoalDistance   = float.MaxValue;
        _prevPursuerVisible = false;
    }

    // ───────── 보상 계산 ──────────────────────────────────────────────────
    /// <summary>
    /// 스텝 보상 계산. Episode-end 보상(goal/crash/capture)과 체크포인트 보상은 EvaderAgent에서 직접 AddReward.
    /// </summary>
    public float ComputeStepReward(
        Vector3 agentPos,
        Vector3 goalPos,
        Vector3 pursuerPos,
        bool    isPursuerVisible,
        Vector3 agentVel,
        float[] obstacleDists = null)
    {
        RewardBreakdown ignored;
        return ComputeStepReward(
            agentPos,
            goalPos,
            pursuerPos,
            isPursuerVisible,
            agentVel,
            obstacleDists,
            out ignored);
    }

    public float ComputeStepReward(
        Vector3 agentPos,
        Vector3 goalPos,
        Vector3 pursuerPos,
        bool    isPursuerVisible,
        Vector3 agentVel,
        float[] obstacleDists,
        out RewardBreakdown breakdown)
    {
        float reward = 0f;
        breakdown = default;

        // ── Goal Shaping (potential-based) ───────────────────────────────
        // Near-goal multiplier: 3m 이내에서 shaping 3x 강화 → orbit → entry 전환 유도
        float currGoalDist   = Vector3.Distance(agentPos, goalPos);
        float shapingMult    = currGoalDist < 3f ? 3f : 1f;
        if (_prevGoalDistance < float.MaxValue)
        {
            float goalShaping = _goalShapingCoeff * shapingMult * (_prevGoalDistance - currGoalDist);
            reward += goalShaping;
            breakdown.GoalShaping = goalShaping;
        }
        _prevGoalDistance = currGoalDist;
        breakdown.GoalDistance = currGoalDist;

        // ── 생존 보너스 ──────────────────────────────────────────────────
        reward += _survivalRewardPerStep;
        breakdown.Survival = _survivalRewardPerStep;

        // ── LOS 차단 보너스 (Stage1+) ─────────────────────────────────────
        if (_occlusionBonus > 0f)
        {
            bool justHidden = _prevPursuerVisible && !isPursuerVisible;
            if (justHidden)
            {
                reward += _occlusionBonus;
                breakdown.Occlusion = _occlusionBonus;
            }
        }
        _prevPursuerVisible = isPursuerVisible;

        // ── Yaw 정렬 보상 ────────────────────────────────────────────────
        if (_yawAlignCoeff > 0f && currGoalDist > 0.1f)
        {
            Vector3 fwd = _agentTransform.forward; fwd.y = 0f;
            Vector3 dir = goalPos - agentPos;      dir.y = 0f;
            if (fwd.sqrMagnitude > 0f && dir.sqrMagnitude > 0f)
            {
                float yawAlign = _yawAlignCoeff * Vector3.Dot(fwd.normalized, dir.normalized);
                reward += yawAlign;
                breakdown.YawAlign = yawAlign;
            }
        }

        // ── Obstacle-Aware Velocity Reward (핵심) ────────────────────────
        // goal 근접 구간(< _goalPriorityDist): 척력 무시, 순수 goal 방향 보상
        // 원거리: 장애물 감지 시 척력 방향으로 블렌드
        if (_velAlignCoeff > 0f && currGoalDist > 0.1f)
        {
            Vector3 toGoalWorld = (goalPos - agentPos).normalized;

            bool inGoalPriorityZone = currGoalDist < _goalPriorityDist;
            breakdown.InGoalPriorityZone = inGoalPriorityZone;

            if (!inGoalPriorityZone && obstacleDists != null)
            {
                // 척력 벡터 계산 (로컬 좌표계)
                Vector3 repulsionLocal = ComputeRepulsionDir(obstacleDists);

                float rawStrength      = Mathf.Clamp01(repulsionLocal.magnitude / 4f);
                float obstacleStrength = rawStrength * 0.25f;  // 0.55 -> 0.25: 척력이 goal 방향을 압도 방지

                Vector3 targetDir;
                if (obstacleStrength > 0.02f)
                {
                    Vector3 toGoalLocal = _agentTransform.InverseTransformDirection(toGoalWorld);
                    toGoalLocal.y = 0f;
                    Vector3 blended = Vector3.Slerp(
                        toGoalLocal.normalized,
                        repulsionLocal.normalized,
                        obstacleStrength);
                    targetDir = _agentTransform.TransformDirection(blended).normalized;
                }
                else
                {
                    targetDir = toGoalWorld;
                }

                float velAlign = Vector3.Dot(agentVel, targetDir) / _maxObsSpeed;
                float velAlignReward = _velAlignCoeff * velAlign;
                reward += velAlignReward;
                breakdown.VelocityAlign = velAlignReward;
            }
            else
            {
                // goal 근접 구간 or 센서 없음: 순수 goal 방향 속도 보상
                float velAlign = Vector3.Dot(agentVel, toGoalWorld) / _maxObsSpeed;

                // goal 근접 구간에서는 후진을 벌점으로 키우기보다 "전진 보상"만 남겨
                // 골 직전 급선회/후진 루프를 줄인다.
                if (inGoalPriorityZone)
                    velAlign = Mathf.Max(0f, velAlign);

                float velAlignReward = _velAlignCoeff * velAlign;
                reward += velAlignReward;
                breakdown.VelocityAlign = velAlignReward;
            }
        }

        // ── 장애물 근접 페널티 (26개 레이 가중 합산) ────────────────────
        // Goal이 Ignore Raycast 레이어에 있으므로 센서에 감지되지 않음.
        // goal 근접 구간에서도 항상 활성: 실제 건물/벽만 페널티 발생.
        if (obstacleDists != null)
        {
            float obstacleSum = 0f;
            int rayCount = Mathf.Min(obstacleDists.Length, 26);
            for (int i = 0; i < rayCount; i++)
            {
                float d = obstacleDists[i];
                if (d <= 0f || d >= _proximityThreshold)
                    continue;

                float rayWeight = GetObstacleRayPenaltyWeight(i);
                if (rayWeight <= 0f)
                    continue;

                obstacleSum += (1f - d / _proximityThreshold) * rayWeight;
            }

            // goal 근접 구간에서는 페널티를 감쇠해 "골존 바로 앞 이탈"을 줄인다.
            float nearGoalScale = 1f;
            if (_goalPriorityDist > 0.01f)
            {
                float t = Mathf.Clamp01(currGoalDist / _goalPriorityDist);
                nearGoalScale = Mathf.Lerp(_nearGoalObstaclePenaltyScale, 1f, t);
            }

            float obstaclePenalty = -_proximityCoeff * obstacleSum * nearGoalScale;
            reward += obstaclePenalty;
            breakdown.ObstacleSum = obstacleSum;
            breakdown.ObstaclePenalty = obstaclePenalty;
        }

        // ── 고도 소프트 페널티 ────────────────────────────────────────────
        if (agentPos.y > _heightWarningY)
        {
            float heightPenalty = _heightPenaltyCoeff * (agentPos.y - _heightWarningY);
            reward += heightPenalty;
            breakdown.HeightPenalty = heightPenalty;
        }

        // ── 시간 페널티 ──────────────────────────────────────────────────
        reward += _timePenaltyPerStep;
        breakdown.TimePenalty = _timePenaltyPerStep;
        breakdown.Total = reward;

        return reward;
    }

    /// <summary>
    /// 26레이 인덱스 기준 장애물 페널티 가중치를 반환한다.
    /// </summary>
    private float GetObstacleRayPenaltyWeight(int rayIndex)
    {
        if (rayIndex == 0)
            return _topRayPenaltyWeight;
        if (rayIndex >= 1 && rayIndex <= 8)
            return _topMiddleRayPenaltyWeight;
        if (rayIndex >= 9 && rayIndex <= 16)
            return _middleRayPenaltyWeight;
        if (rayIndex >= 17 && rayIndex <= 24)
            return _middleBottomRayPenaltyWeight;
        if (rayIndex == 25)
            return _bottomRayPenaltyWeight;
        return 0f;
    }

    // ───────── 방향성 척력 벡터 계산 ─────────────────────────────────────
    /// <summary>
    /// Middle 레이어 인덱스 9~16에서 로컬 좌표계 척력 벡터를 계산한다.
    /// 장애물이 있는 방향의 반대로 합산. 반환값은 드론 로컬 좌표계.
    /// </summary>
    private Vector3 ComputeRepulsionDir(float[] dists)
    {
        if (dists == null || dists.Length < 17) return Vector3.zero;

        Vector3 repulsion = Vector3.zero;
        for (int i = 0; i < 8; i++)
        {
            float d = dists[9 + i];
            if (d > 0f && d < _proximityThreshold)
            {
                float weight = 1f - d / _proximityThreshold;  // 가까울수록 강함
                repulsion -= MiddleRayDirs[i] * weight;        // 장애물 방향 반대
            }
        }
        return repulsion;
    }

    // ───────── 디버그 헬퍼 ───────────────────────────────────────────────
    public string GetRewardSummary()
    {
        return $"[Reward] goalShaping={_goalShapingCoeff} velAlign={_velAlignCoeff} " +
               $"proximity={_proximityCoeff}@{_proximityThreshold} time={_timePenaltyPerStep}";
    }
}
