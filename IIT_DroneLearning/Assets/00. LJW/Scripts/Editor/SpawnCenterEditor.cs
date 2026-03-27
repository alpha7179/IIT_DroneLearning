using UnityEngine;
using UnityEditor;

/// <summary>
/// SpawnCenter 커스텀 인스펙터
/// SyncMode에 따라 Synchronized면 공용 범위만, Desynchronized면 개별 범위를 표시한다.
/// </summary>
[CustomEditor(typeof(SpawnCenter))]
public class SpawnCenterEditor : Editor
{
    public override void OnInspectorGUI()
    {
        serializedObject.Update();

        var modeProp = serializedObject.FindProperty("_syncMode");
        EditorGUILayout.PropertyField(modeProp, new GUIContent("Sync Mode"));

        EditorGUILayout.Space(4);

        var mode = (SpawnCenter.SyncMode)modeProp.enumValueIndex;

        if (mode == SpawnCenter.SyncMode.Synchronized)
        {
            EditorGUILayout.LabelField("공용 범위 (골존·추적자·도망자 공유)", EditorStyles.boldLabel);
            DrawSpawnRange(serializedObject.FindProperty("_sharedRange"));
        }
        else
        {
            EditorGUILayout.LabelField("골존 범위", EditorStyles.boldLabel);
            DrawSpawnRange(serializedObject.FindProperty("_goalRange"));

            EditorGUILayout.Space(4);
            EditorGUILayout.LabelField("추적자 범위", EditorStyles.boldLabel);
            DrawSpawnRange(serializedObject.FindProperty("_pursuerRange"));

            EditorGUILayout.Space(4);
            EditorGUILayout.LabelField("도망자 범위", EditorStyles.boldLabel);
            DrawSpawnRange(serializedObject.FindProperty("_evaderRange"));
        }

        serializedObject.ApplyModifiedProperties();
    }

    private void DrawSpawnRange(SerializedProperty rangeProp)
    {
        if (rangeProp == null) return;
        EditorGUI.indentLevel++;
        EditorGUILayout.PropertyField(rangeProp.FindPropertyRelative("MinY"), new GUIContent("Min Y"));
        EditorGUILayout.PropertyField(rangeProp.FindPropertyRelative("MaxY"), new GUIContent("Max Y"));
        EditorGUILayout.PropertyField(rangeProp.FindPropertyRelative("Radius"), new GUIContent("Radius"));
        EditorGUI.indentLevel--;
    }
}
