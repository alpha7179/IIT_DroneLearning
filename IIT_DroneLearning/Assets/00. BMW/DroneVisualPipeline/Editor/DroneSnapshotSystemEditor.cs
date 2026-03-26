using System.IO;
using UnityEngine;
using UnityEditor;
using DroneVisualPipeline;

[CustomEditor(typeof(DroneSnapshotSystem))]
public class DroneSnapshotSystemEditor : UnityEditor.Editor
{
    private DroneSnapshotSystem _snap;

    private void OnEnable()
    {
        _snap = target as DroneSnapshotSystem;
        EditorApplication.playModeStateChanged += OnPlayModeStateChanged;
        MigrateOldBasePath();
    }

    /// <summary>
    /// 구 basePath("Assets/..." 형식) → 신 basePath("CapturedData", 프로젝트 루트 기준)로 자동 마이그레이션.
    /// Assets 내부 경로는 Unity 파일 감시자가 .meta를 생성하여 에디터 리로드를 유발하므로 외부로 이동.
    /// </summary>
    private void MigrateOldBasePath()
    {
        if (_snap == null) return;
        if (!_snap.basePath.StartsWith("Assets/")) return;

        string oldPath = _snap.basePath;
        _snap.basePath = "CapturedData";
        EditorUtility.SetDirty(_snap);
        Debug.Log($"[DroneSnapshotSystem] basePath 마이그레이션: '{oldPath}' → 'CapturedData'");
    }

    private void OnDisable()
    {
        EditorApplication.playModeStateChanged -= OnPlayModeStateChanged;
    }

    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        if (_snap == null) return;

        EditorGUILayout.Space(8);

        var lineRect = EditorGUILayout.GetControlRect(false, 1);
        EditorGUI.DrawRect(lineRect, new Color(0.5f, 0.5f, 0.5f, 0.5f));

        EditorGUILayout.Space(4);
        EditorGUILayout.LabelField("데이터 관리", EditorStyles.boldLabel);

        string fullPath = BuildFullPath();
        bool exists     = !string.IsNullOrEmpty(fullPath) && Directory.Exists(fullPath);

        if (exists)
        {
            EditorGUILayout.HelpBox(fullPath, MessageType.None);
        }

        GUI.backgroundColor = exists ? new Color(1f, 0.35f, 0.35f) : new Color(0.6f, 0.6f, 0.6f);
        using (new EditorGUI.DisabledScope(!exists))
        {
            if (GUILayout.Button(exists ? "캡처 데이터 삭제" : "캡처 데이터 없음", GUILayout.Height(28)))
            {
                if (EditorUtility.DisplayDialog(
                        "캡처 데이터 삭제",
                        $"아래 폴더의 모든 데이터를 삭제합니다.\n이 작업은 되돌릴 수 없습니다.\n\n{fullPath}",
                        "삭제", "취소"))
                {
                    DoDelete(fullPath);
                }
            }
        }
        GUI.backgroundColor = Color.white;
    }

    private void OnPlayModeStateChanged(PlayModeStateChange state)
    {
        if (state != PlayModeStateChange.EnteredEditMode) return;
        if (_snap == null || !_snap.deleteOnPlayModeExit) return;

        string fullPath = BuildFullPath();
        if (string.IsNullOrEmpty(fullPath)) return;

        DoDelete(fullPath);
        Debug.Log("[DroneSnapshotSystem] 플레이 모드 종료 → 캡처 데이터 자동 삭제 완료.");
    }

    private void DoDelete(string fullPath)
    {
        if (!Directory.Exists(fullPath))
        {
            Debug.LogWarning($"[DroneSnapshotSystem] 삭제할 폴더가 없습니다: {fullPath}");
            return;
        }

        try
        {
            Directory.Delete(fullPath, recursive: true);

            string meta = fullPath.TrimEnd('/') + ".meta";
            if (File.Exists(meta)) File.Delete(meta);

            AssetDatabase.Refresh();
            Debug.Log($"[DroneSnapshotSystem] 삭제 완료: {fullPath}");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"[DroneSnapshotSystem] 삭제 실패: {e.Message}");
        }
    }

    private string BuildFullPath()
    {
        if (_snap == null || string.IsNullOrWhiteSpace(_snap.basePath)) return string.Empty;

        // Application.dataPath = ".../IIT_DroneLearning/Assets"
        // → 프로젝트 루트 = ".../IIT_DroneLearning/"
        string projectRoot = Application.dataPath;
        if (projectRoot.EndsWith("/Assets") || projectRoot.EndsWith("\\Assets"))
            projectRoot = projectRoot.Substring(0, projectRoot.Length - "Assets".Length);

        return Path.Combine(projectRoot, _snap.basePath).Replace('\\', '/');
    }
}
