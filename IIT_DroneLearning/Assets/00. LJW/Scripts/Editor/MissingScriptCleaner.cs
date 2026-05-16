using UnityEngine;
using UnityEditor;

/// <summary>
/// 씬 내 모든 GameObject에서 Missing Script(참조 끊긴 컴포넌트)를 제거한다.
///
/// 사용법:
///   Unity 메뉴 → Tools → LJW → Remove Missing Scripts in Scene
/// </summary>
public static class MissingScriptCleaner
{
    [MenuItem("Tools/LJW/Remove Missing Scripts in Scene")]
    public static void RemoveMissingScriptsInScene()
    {
        int totalRemoved = 0;
        int objectsAffected = 0;

        GameObject[] allObjects = Resources.FindObjectsOfTypeAll<GameObject>();

        foreach (GameObject go in allObjects)
        {
            // 씬에 속한 오브젝트만 (에셋·프리팹 제외)
            if (!go.scene.IsValid()) continue;

            int before = go.GetComponents<Component>().Length;
            int removed = GameObjectUtility.RemoveMonoBehavioursWithMissingScript(go);

            if (removed > 0)
            {
                Debug.Log($"[MissingScriptCleaner] '{go.name}' — {removed}개 제거됨");
                totalRemoved += removed;
                objectsAffected++;
            }
        }

        if (totalRemoved == 0)
        {
            Debug.Log("[MissingScriptCleaner] Missing script 없음. 씬이 이미 깨끗합니다.");
        }
        else
        {
            Debug.Log($"[MissingScriptCleaner] 완료 — {objectsAffected}개 오브젝트에서 총 {totalRemoved}개 제거. 씬을 저장하세요.");
            EditorUtility.DisplayDialog(
                "Missing Script Cleaner",
                $"총 {totalRemoved}개의 Missing Script를 제거했습니다.\n({objectsAffected}개 오브젝트 영향)\n\n씬을 저장하세요 (Ctrl+S).",
                "확인"
            );
        }

        // 씬 dirty 표시 (저장 유도)
        if (totalRemoved > 0)
            UnityEditor.SceneManagement.EditorSceneManager.MarkAllScenesDirty();
    }
}
