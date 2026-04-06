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

    [Tooltip("Yaw 정렬 보상 (Stage1+에서는 0 권장)")]
    [SerializeField] private float _yawAlignCoeff = 0.0f;

    [Header("Observation Normalization")]
    [SerializeField] private float _maxObsSpeed = 10f;

    [Header("Obstacle Proximity Penalty")]
    [Tooltip("이 정규화 거리 미만 레이 감지 시 페널티 합산 시작 (0~1)")]
    [SerializeField] private float _proximityThreshold = 0.15f;
    [Tooltip("레이 1개당 최대 페널티 기여값. 26개 합산되므로 작게 유지")]
    [SerializeField] private float _proximityCoeff = 0.002f;

    [Header("Height Soft Penalty")]
    [Tooltip("이 고도(m) 초과 시 페널티 시작 (벽 높이 20m 기준)")]
    [SerializeField] private float _heightWarningY = 17f;
    [Tooltip("고도 경고 구간에서 초과 1m당 페널티 (음수 유지)")]
    [SerializeField] private float _heightPenaltyCoeff = -0.02f;

    [Header("Time Penalty")]
    [Tooltip("스텝당 시간 페널티 (음수 유지). -0.003: 350스텝 타임아웃 = -1.05 → 충돌(-1.3)보다 나쁘지 않게")]
    [SerializeField] private float _timePenaltyPerStep = -0.003f;

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
        float reward = 0f;

        // ── Goal Shaping (potential-based) ───────────────────────────────
        float currGoalDist = Vector3.Distance(agentPos, goalPos);
        if (_prevGoalDistance < float.MaxValue)
            reward += _goalShapingCoeff * (_prevGoalDistance - currGoalDist);
        _prevGoalDistance = currGoalDist;

        // ── 생존 보너스 ──────────────────────────────────────────────────
        reward += _survivalRewardPerStep;

        // ── LOS 차단 보너스 (Stage1+) ─────────────────────────────────────
        if (_occlusionBonus > 0f)
        {
            bool justHidden = _prevPursuerVisible && !isPursuerVisible;
            if (justHidden) reward += _occlusionBonus;
        }
        _prevPursuerVisible = isPursuerVisible;

        // ── Yaw 정렬 보상 ────────────────────────────────────────────────
        if (_yawAlignCoeff > 0f && currGoalDist > 0.1f)
        {
            Vector3 fwd = _agentTransform.forward; fwd.y = 0f;
            Vector3 dir = goalPos - agentPos;      dir.y = 0f;
            if (fwd.sqrMagnitude > 0f && dir.sqrMagnitude > 0f)
                reward += _yawAlignCoeff * Vector3.Dot(fwd.normalized, dir.normalized);
        }

        // ── Obstacle-Aware Velocity Reward (핵심) ────────────────────────
        // 장애물 없음 → goal 방향 속도 보상
        // 장애물 감지 → 척력 방향으로 블렌드: "goal 방향으로 가되, 벽이 있으면 옆으로 꺾어"
        if (_velAlignCoeff > 0f && currGoalDist > 0.1f)
        {
            Vector3 toGoalWorld = (goalPos - agentPos).normalized;

            if (obstacleDists != null)
            {
                // 척력 벡터 계산 (로컬 좌표계)
                Vector3 repulsionLocal = ComputeRepulsionDir(obstacleDists);

                // 장애물 강도: /4f 로 정규화해 2개 레이 = 0.5 블렌드 (기존 /2f는 2개만으로 100% 척력)
                // 최대 0.55 캡: goal 방향 성분 최소 45% 보장 (너무 멀리 후퇴 방지)
                float rawStrength      = Mathf.Clamp01(repulsionLocal.magnitude / 4f);
                float obstacleStrength = rawStrength * 0.55f;

                Vector3 targetDir;
                if (obstacleStrength > 0.02f)
                {
                    // 세계 좌표계 goal 방향을 로컬로 변환해 척력과 같은 공간에서 블렌드
                    Vector3 toGoalLocal = _agentTransform.InverseTransformDirection(toGoalWorld);
                    toGoalLocal.y = 0f;
                    Vector3 blended = Vector3.Slerp(
                        toGoalLocal.normalized,
                        repulsionLocal.normalized,
                        obstacleStrength);
                    // 블렌드된 로컬 방향을 다시 세계 좌표로
                    targetDir = _agentTransform.TransformDirection(blended).normalized;
                }
                else
                {
                    targetDir = toGoalWorld;
                }

                float velAlign = Vector3.Dot(agentVel, targetDir) / _maxObsSpeed;
                reward += _velAlignCoeff * velAlign;
            }
            else
            {
                // obstacleDists 없을 때 폴백: 단순 goal 방향 속도
                float velAlign = Vector3.Dot(agentVel, toGoalWorld) / _maxObsSpeed;
                reward += _velAlignCoeff * velAlign;
            }
        }

        // ── 장애물 근접 페널티 (26개 레이 합산) ─────────────────────────
        if (obstacleDists != null)
        {
            float obstacleSum = 0f;
            foreach (float d in obstacleDists)
                if (d > 0f && d < _proximityThreshold)
                    obstacleSum += 1f - d / _proximityThreshold;
            reward -= _proximityCoeff * obstacleSum;
        }

        // ── 고도 소프트 페널티 ────────────────────────────────────────────
        if (agentPos.y > _heightWarningY)
            reward += _heightPenaltyCoeff * (agentPos.y - _heightWarningY);

        // ── 시간 페널티 ──────────────────────────────────────────────────
        reward += _timePenaltyPerStep;

        return reward;
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
