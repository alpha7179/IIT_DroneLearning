using UnityEngine;
using UnityEditor;

/// <summary>
/// EvaderAgent 커스텀 인스펙터
///
/// 섹션 구분:
///   [DroneAgent — 수정 가능]  : EvaderAgent가 실제 사용하는 DroneAgent 필드 (편집 가능)
///   [DroneAgent — 읽기 전용] : EvaderAgent가 오버라이드하여 실제로 사용되지 않는 DroneAgent 필드 (확인 전용)
///   [EvaderAgent]             : EvaderAgent 전용 필드 (편집 가능)
/// </summary>
[CustomEditor(typeof(EvaderAgent))]
public class EvaderAgentEditor : Editor
{
    // 섹션 접힘 상태
    private bool _baseFoldout     = true;
    private bool _baseROFoldout   = true;
    private bool _evaderFoldout   = true;

    // 배경 색상
    private static readonly Color ColorEditable  = new Color(0.18f, 0.22f, 0.28f, 1f);
    private static readonly Color ColorReadOnly  = new Color(0.22f, 0.18f, 0.18f, 1f);
    private static readonly Color ColorEvader    = new Color(0.18f, 0.24f, 0.20f, 1f);

    public override void OnInspectorGUI()
    {
        serializedObject.Update();
        EvaderAgent agent = (EvaderAgent)target;

        // ─── DroneAgent — 수정 가능 (EvaderAgent가 실제 사용) ──────────────
        DrawSection("DroneAgent — 수정 가능", ref _baseFoldout, ColorEditable, () =>
        {
            DrawProp("_goalZone",       "Goal Zone (Goal 컴포넌트)");
            DrawProp("TargetTransform", "Target Transform (추격자)");
            EditorGUILayout.Space(2);
            DrawProp("ManualThrottle");
            DrawProp("ManualAttitude");
            DrawProp("ManualYawRate");
        });

        EditorGUILayout.Space(4);

        // ─── DroneAgent — 읽기 전용 (EvaderAgent가 오버라이드하여 미사용) ──
        DrawSection("DroneAgent — 읽기 전용 (오버라이드됨)", ref _baseROFoldout, ColorReadOnly, () =>
        {
            GUI.enabled = false;
            EditorGUILayout.EnumPopup(new GUIContent("Role", "Initialize()에서 Evader로 고정됨"), agent.Role);
            EditorGUILayout.ObjectField(new GUIContent("Goal Transform", "Goal 컴포넌트로 대체됨. EvaderAgent 미사용."), agent.GoalTransform, typeof(Transform), true);
            EditorGUILayout.FloatField(new GUIContent("Spawn Range X", "SpawnCenter로 대체됨"), agent.SpawnRangeX);
            EditorGUILayout.FloatField(new GUIContent("Spawn Range Z", "SpawnCenter로 대체됨"), agent.SpawnRangeZ);
            EditorGUILayout.FloatField(new GUIContent("Spawn Height",  "SpawnCenter로 대체됨"), agent.SpawnHeight);
            EditorGUILayout.FloatField(new GUIContent("Spawn Yaw Max", "EvaderAgent: 0~360 전방향 고정"), agent.SpawnYawMaxDeg);
            EditorGUILayout.FloatField(new GUIContent("Step Penalty",  "EvaderAgent: EvaderReward로 대체됨"), agent.StepPenalty);
            GUI.enabled = true;
        });

        EditorGUILayout.Space(4);

        // ─── EvaderAgent 전용 ────────────────────────────────────────────────
        DrawSection("EvaderAgent", ref _evaderFoldout, ColorEvader, () =>
        {
            DrawProp("_autoApplyStage1Defaults");
            EditorGUILayout.Space(2);

            EditorGUILayout.LabelField("Goal / Placement", EditorStyles.miniBoldLabel);
            DrawProp("_goalZone");
            DrawProp("_validateGoalPlacement");
            DrawProp("_goalObstacleClearanceRadius");
            DrawProp("_goalResampleMaxAttempts");
            EditorGUILayout.Space(2);

            EditorGUILayout.LabelField("Sensor Hardening", EditorStyles.miniBoldLabel);
            DrawProp("_autoSetGoalIgnoreRaycastLayer");
            DrawProp("_excludeIgnoreRaycastFromSensorMask");
            DrawProp("_forceNonEmptyDetectionLayerMask");
            DrawProp("_disableMiddleBottomRaysInStage1");
            DrawProp("_disableBottomRayInStage1");
            EditorGUILayout.Space(2);

            EditorGUILayout.LabelField("Reward Hardening", EditorStyles.miniBoldLabel);
            DrawProp("_applyStage1RewardProfile");
            DrawProp("_useStage1RewardPreset");
            DrawProp("_stage1RewardPreset");
            DrawProp("_stage1RewardProfileExperimental", "Experimental Profile");
            DrawProp("_stage1RewardProfileLegacy", "LegacyV7 Profile");
            DrawProp("_stage1RewardProfile", "Manual Profile");
            DrawProp("_logActiveStage1RewardProfile");
            EditorGUILayout.Space(2);

            EditorGUILayout.LabelField("Episode Settings", EditorStyles.miniBoldLabel);
            DrawProp("_maxEpisodeSeconds");
            DrawProp("_catchDistance");
            DrawProp("_goalArrivalReward");
            EditorGUILayout.Space(2);

            EditorGUILayout.LabelField("Checkpoint Settings", EditorStyles.miniBoldLabel);
            DrawProp("_checkpointCount");
            DrawProp("_checkpointRadius");
            DrawProp("_checkpointReward");
            DrawProp("_checkpointRewardCap");
            DrawProp("_checkpointRewardDecay");
            DrawProp("_checkpointClearRadius");
            EditorGUILayout.Space(2);

            EditorGUILayout.LabelField("Action Smoothing", EditorStyles.miniBoldLabel);
            DrawProp("_useActionSmoothing");
            DrawProp("_actionDeadzone");
            DrawProp("_commandLerpFactor");
            DrawProp("_maxCommandDeltaPerStep");
            EditorGUILayout.Space(2);

            EditorGUILayout.LabelField("Wall Proximity Safety", EditorStyles.miniBoldLabel);
            DrawProp("_enableWallProximitySafety");
            DrawProp("_wallProximityThreshold");
            DrawProp("_blockForwardPitchOnWallProximity");
            DrawProp("_wallAvoidRollAssist");
            DrawProp("_wallAvoidYawAssist");
            EditorGUILayout.Space(2);

            EditorGUILayout.LabelField("Anti-Hover", EditorStyles.miniBoldLabel);
            DrawProp("_enableStagnationPenalty");
            DrawProp("_stagnationWatchDistance");
            DrawProp("_stagnationProgressThreshold");
            DrawProp("_stagnationGraceSteps");
            DrawProp("_stagnationPenaltyPerStep");
            EditorGUILayout.Space(2);

            EditorGUILayout.LabelField("Boundary / Collision", EditorStyles.miniBoldLabel);
            DrawProp("_maxFlightHeight");
            DrawProp("_boundaryHalfSize");
            DrawProp("_spawnBoundaryInset");
            DrawProp("_enableBoundarySafety");
            DrawProp("_boundarySafetyMargin");
            DrawProp("_boundaryInwardRollAssist");
            DrawProp("_boundaryInwardYawAssist");
            DrawProp("_boundaryForwardPitchMinScale");
            DrawProp("_boundaryRiskPenaltyPerStep");
            DrawProp("_boundaryOutwardSpeedReference");
            DrawProp("_boundaryEmergencyRiskThreshold");
            DrawProp("_boundaryEmergencyRollAssist");
            DrawProp("_boundaryEmergencyYawAssist");
            DrawProp("_boundaryEmergencyPitchAssist");
            DrawProp("_boundarySoftClampInset");
            DrawProp("_boundarySoftClampTolerance");
            DrawProp("_boundarySoftClampPenalty");
            DrawProp("_heightSafetyMargin");
            DrawProp("_heightDescendThrottle");
            DrawProp("_terminalPenaltyBoundaryOverflow");
            DrawProp("_crashTags");
            EditorGUILayout.Space(2);

            EditorGUILayout.LabelField("Stage Control", EditorStyles.miniBoldLabel);
            DrawProp("_goalOnlyMode");
            DrawProp("_currentStage");
            EditorGUILayout.Space(2);
            EditorGUILayout.LabelField("SpawnCenter 쿼리 결과 (읽기 전용)", EditorStyles.miniBoldLabel);
            GUI.enabled = false;
            DrawProp("_queriedMinY",   "Queried Min Y");
            DrawProp("_queriedMaxY",   "Queried Max Y");
            DrawProp("_queriedRadius", "Queried Radius");
            GUI.enabled = true;
            EditorGUILayout.Space(2);
            EditorGUILayout.LabelField("Observation Normalization", EditorStyles.miniBoldLabel);
            DrawProp("_maxDistance");
            DrawProp("_maxObsSpeed");
            EditorGUILayout.Space(2);

            EditorGUILayout.LabelField("Reward Debug", EditorStyles.miniBoldLabel);
            DrawProp("_logRewardBreakdown");
            DrawProp("_rewardLogIntervalSteps");
        });

        serializedObject.ApplyModifiedProperties();
    }

    // ── 헬퍼 ─────────────────────────────────────────────────────────────────

    private void DrawSection(string title, ref bool foldout, Color bgColor, System.Action drawContent)
    {
        var rect = EditorGUILayout.BeginVertical();
        EditorGUI.DrawRect(rect, bgColor);
        foldout = EditorGUILayout.Foldout(foldout, title, true, EditorStyles.foldoutHeader);
        if (foldout)
        {
            EditorGUI.indentLevel++;
            drawContent();
            EditorGUI.indentLevel--;
        }
        EditorGUILayout.EndVertical();
    }

    private void DrawProp(string propName, string label = null)
    {
        var prop = serializedObject.FindProperty(propName);
        if (prop == null) return;
        if (label != null)
            EditorGUILayout.PropertyField(prop, new GUIContent(label));
        else
            EditorGUILayout.PropertyField(prop);
    }
}
