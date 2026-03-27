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

        // SpawnCenter가 있으면 Center/Randomization은 SpawnCenter에서 관리되므로 읽기 전용
        if (hasSpawnCenter)
        {
            EditorGUILayout.HelpBox("SpawnCenter가 활성화되어 있습니다. 범위는 SpawnCenter에서 관리됩니다.", MessageType.Info);
            GUI.enabled = false;
        }

        EditorGUILayout.LabelField("Goal — Center", EditorStyles.boldLabel);
        DrawProp("_centerTransform", "Center Transform");

        EditorGUILayout.Space(2);
        EditorGUILayout.LabelField("Goal — Randomization", EditorStyles.boldLabel);
        DrawProp("_randomizeRadius", "Randomize Radius");

        if (hasSpawnCenter)
            GUI.enabled = true;

        EditorGUILayout.Space(4);
        EditorGUILayout.LabelField("SpawnCenter 쿼리 결과 (읽기 전용)", EditorStyles.boldLabel);
        GUI.enabled = false;
        DrawProp("_queriedMinY", "Queried Min Y");
        DrawProp("_queriedMaxY", "Queried Max Y");
        bool isRect = SpawnCenter.Current != null &&
                      SpawnCenter.Current.ShapeMode == SpawnCenter.RangeMode.Rectangle;
        if (isRect)
        {
            DrawProp("_queriedWidth", "Queried Width  (X 반폭)");
            DrawProp("_queriedDepth", "Queried Depth  (Z 반폭)");
        }
        else
        {
            DrawProp("_queriedRadius", "Queried Radius");
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
