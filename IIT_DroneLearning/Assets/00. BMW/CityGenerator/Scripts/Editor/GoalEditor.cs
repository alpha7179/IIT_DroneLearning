using UnityEngine;
using UnityEditor;

/// <summary>
/// Goal 커스텀 인스펙터
/// SpawnCenter 쿼리 결과 필드를 읽기 전용으로 표시한다.
/// </summary>
[CustomEditor(typeof(Goal))]
public class GoalEditor : Editor
{
    public override void OnInspectorGUI()
    {
        serializedObject.Update();
        bool hasSpawnCenter = SpawnCenter.Current != null;

        // ── Center & Randomization ───────────────────────────────────────
        if (hasSpawnCenter)
        {
            EditorGUILayout.HelpBox("SpawnCenter가 활성화되어 있습니다. 범위는 SpawnCenter에서 관리됩니다.", MessageType.Info);
            GUI.enabled = false;
        }

        DrawProp("_centerTransform");
        DrawProp("_randomizeRadius");

        if (hasSpawnCenter)
            GUI.enabled = true;

        // ── Building Overlap Shrink ──────────────────────────────────────
        DrawProp("_enableShrinkOnOverlap");
        DrawProp("_minShrinkRadius");
        DrawProp("_shrinkStep");

        // ── SpawnCenter 쿼리 결과 (읽기 전용) ───────────────────────────
        GUI.enabled = false;
        DrawProp("_queriedMinY");
        DrawProp("_queriedMaxY");
        bool isRect = SpawnCenter.Current != null &&
                      SpawnCenter.Current.ShapeMode == SpawnCenter.RangeMode.Rectangle;
        if (isRect)
        {
            DrawProp("_queriedWidth");
            DrawProp("_queriedDepth");
        }
        else
        {
            DrawProp("_queriedRadius");
        }
        GUI.enabled = true;

        serializedObject.ApplyModifiedProperties();
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
