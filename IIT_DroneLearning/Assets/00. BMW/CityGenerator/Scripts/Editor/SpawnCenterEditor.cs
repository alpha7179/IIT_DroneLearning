using UnityEngine;
using UnityEditor;

/// <summary>
/// SpawnCenter 커스텀 인스펙터
/// SyncMode에 따라 Synchronized면 공용 범위만, Desynchronized면 개별 범위를 표시한다.
/// RangeMode에 따라 Radius 또는 Width/Depth 필드를 표시한다.
/// </summary>
[CustomEditor(typeof(SpawnCenter))]
public class SpawnCenterEditor : Editor
{
    public override void OnInspectorGUI()
    {
        serializedObject.Update();

        var modeProp  = serializedObject.FindProperty("_syncMode");
        var shapeProp = serializedObject.FindProperty("_rangeMode");

        EditorGUILayout.PropertyField(modeProp,  new GUIContent("Sync Mode"));
        EditorGUILayout.PropertyField(shapeProp, new GUIContent("Range Mode"));

        EditorGUILayout.Space(4);

        var syncMode  = (SpawnCenter.SyncMode) modeProp.enumValueIndex;
        var rangeMode = (SpawnCenter.RangeMode)shapeProp.enumValueIndex;

        if (syncMode == SpawnCenter.SyncMode.Synchronized)
        {
            EditorGUILayout.LabelField("공용 범위 (골존·추적자·도망자 공유)", EditorStyles.boldLabel);
            DrawSpawnRange(serializedObject.FindProperty("_sharedRange"), rangeMode);
        }
        else
        {
            EditorGUILayout.LabelField("골존 범위", EditorStyles.boldLabel);
            DrawSpawnRange(serializedObject.FindProperty("_goalRange"), rangeMode);

            EditorGUILayout.Space(4);
            EditorGUILayout.LabelField("추적자 범위", EditorStyles.boldLabel);
            DrawSpawnRange(serializedObject.FindProperty("_pursuerRange"), rangeMode);

            EditorGUILayout.Space(4);
            EditorGUILayout.LabelField("도망자 범위", EditorStyles.boldLabel);
            DrawSpawnRange(serializedObject.FindProperty("_evaderRange"), rangeMode);
        }

        serializedObject.ApplyModifiedProperties();
    }

    private void DrawSpawnRange(SerializedProperty rangeProp, SpawnCenter.RangeMode rangeMode)
    {
        if (rangeProp == null) return;
        EditorGUI.indentLevel++;

        EditorGUILayout.PropertyField(rangeProp.FindPropertyRelative("MinY"), new GUIContent("Min Y"));
        EditorGUILayout.PropertyField(rangeProp.FindPropertyRelative("MaxY"), new GUIContent("Max Y"));

        if (rangeMode == SpawnCenter.RangeMode.Radius)
        {
            EditorGUILayout.PropertyField(rangeProp.FindPropertyRelative("Radius"), new GUIContent("Radius"));
        }
        else
        {
            EditorGUILayout.PropertyField(rangeProp.FindPropertyRelative("Width"), new GUIContent("Width  (X 반폭)"));
            EditorGUILayout.PropertyField(rangeProp.FindPropertyRelative("Depth"), new GUIContent("Depth  (Z 반폭)"));
        }

        EditorGUI.indentLevel--;
    }
}
