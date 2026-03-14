using UnityEditor;
using UnityEngine;

namespace ProceduralCityGenerator
{
    /// <summary>
    /// CityBatchGenerator 컴포넌트를 위한 Custom Inspector
    /// </summary>
    [CustomEditor(typeof(CityBatchGenerator))]
    public class CityBatchGeneratorEditor : UnityEditor.Editor
    {
        private string GetCityGeneratorRootPath()
        {
            var script = MonoScript.FromMonoBehaviour((CityBatchGenerator)target);
            string scriptPath = AssetDatabase.GetAssetPath(script);
            string scriptsDir = System.IO.Path.GetDirectoryName(scriptPath);
            string rootDir    = System.IO.Path.GetDirectoryName(scriptsDir);
            return rootDir.Replace('\\', '/');
        }

        public override void OnInspectorGUI()
        {
            var batch = (CityBatchGenerator)target;

            EditorGUI.BeginChangeCheck();

            bool isSameMode = batch.seedMode == BatchSeedMode.AllSame;

            // === Batch Layout 섹션 ===
            EditorGUILayout.Space(5);
            EditorGUILayout.LabelField("Batch Layout", EditorStyles.boldLabel);
            EditorGUILayout.Space(3);

            batch.columns = EditorGUILayout.IntSlider(
                new GUIContent("Columns (n)", "가로 열 수."),
                batch.columns, 1, 20);

            batch.rows = EditorGUILayout.IntSlider(
                new GUIContent("Rows (m)", "세로 행 수."),
                batch.rows, 1, 20);

            if (isSameMode)
            {
                EditorGUILayout.HelpBox(
                    $"AllSame 모드: Seed {batch.baseSeed}로 {batch.columns}×{batch.rows} = {batch.columns * batch.rows}개 도시 생성. CSV/JSON/미니맵은 첫 번째 도시에서만 저장됩니다.",
                    MessageType.Info);
            }
            else
            {
                EditorGUILayout.HelpBox(
                    $"총 {batch.columns * batch.rows}개 도시가 생성됩니다.",
                    MessageType.Info);
            }

            // === Spacing 섹션 ===
            EditorGUILayout.Space(10);
            EditorGUILayout.LabelField("Spacing", EditorStyles.boldLabel);
            EditorGUILayout.Space(3);

            batch.spacingX = EditorGUILayout.FloatField(
                new GUIContent("Spacing X", "도시 간 가로 여백 (월드 단위)."),
                batch.spacingX);

            batch.spacingZ = EditorGUILayout.FloatField(
                new GUIContent("Spacing Z", "도시 간 세로 여백 (월드 단위)."),
                batch.spacingZ);

            // 음수 방지
            if (batch.spacingX < 0f) batch.spacingX = 0f;
            if (batch.spacingZ < 0f) batch.spacingZ = 0f;

            // === Seed Settings 섹션 ===
            EditorGUILayout.Space(10);
            EditorGUILayout.LabelField("Seed Settings", EditorStyles.boldLabel);
            EditorGUILayout.Space(3);

            batch.seedMode = (BatchSeedMode)EditorGUILayout.EnumPopup(
                new GUIContent("Seed Mode",
                    "AllRandom: 각 도시 고유 씨드\n" +
                    "AllSame: 동일 씨드 1개만 생성\n" +
                    "Sequential: baseSeed부터 순차 증가"),
                batch.seedMode);

            bool showBaseSeed = batch.seedMode != BatchSeedMode.AllRandom;

            using (new EditorGUI.DisabledScope(!showBaseSeed))
            {
                string label   = batch.seedMode == BatchSeedMode.AllSame ? "Seed" : "Base Seed";
                string tooltip = batch.seedMode == BatchSeedMode.AllSame
                    ? "생성에 사용할 씨드 값"
                    : "시작 씨드 값 (-1이면 시간 기반 랜덤 시작)";

                batch.baseSeed = EditorGUILayout.IntField(
                    new GUIContent(label, tooltip),
                    batch.baseSeed);
            }

            // === City Template 섹션 ===
            EditorGUILayout.Space(10);
            EditorGUILayout.LabelField("City Template", EditorStyles.boldLabel);
            EditorGUILayout.Space(3);

            batch.cityTemplateObject = (GameObject)EditorGUILayout.ObjectField(
                new GUIContent("Template Object",
                    "CityGenerator가 붙은 GameObject를 드래그하세요. 설정 시 아래 필드보다 우선 적용됩니다."),
                batch.cityTemplateObject, typeof(GameObject), true);

            // cityTemplateObject에 CityGenerator가 없으면 경고
            if (batch.cityTemplateObject != null &&
                batch.cityTemplateObject.GetComponent<CityGenerator>() == null)
            {
                EditorGUILayout.HelpBox(
                    $"'{batch.cityTemplateObject.name}'에 CityGenerator 컴포넌트가 없습니다.",
                    MessageType.Error);
            }

            batch.cityTemplate = (CityGenerator)EditorGUILayout.ObjectField(
                new GUIContent("Template Component",
                    "CityGenerator 컴포넌트를 직접 지정합니다. Template Object가 비어 있을 때만 사용됩니다."),
                batch.cityTemplate, typeof(CityGenerator), true);

            // 유효한 소스가 하나도 없을 때 경고
            bool hasValidTemplate =
                (batch.cityTemplateObject != null &&
                 batch.cityTemplateObject.GetComponent<CityGenerator>() != null) ||
                batch.cityTemplate != null;

            if (!hasValidTemplate)
            {
                EditorGUILayout.HelpBox(
                    "Template Object 또는 Template Component 중 하나를 설정해야 생성이 가능합니다.",
                    MessageType.Warning);
            }
            else if (batch.cityTemplateObject != null &&
                     batch.cityTemplateObject.GetComponent<CityGenerator>() != null)
            {
                EditorGUILayout.HelpBox(
                    $"소스: '{batch.cityTemplateObject.name}'의 CityGenerator (Template Object 우선)",
                    MessageType.Info);
            }

            // 변경 사항이 있으면 저장
            if (EditorGUI.EndChangeCheck())
            {
                EditorUtility.SetDirty(target);
            }

            // === 배치 생성 제어 ===
            EditorGUILayout.Space(10);
            EditorGUILayout.LabelField("배치 생성 제어", EditorStyles.boldLabel);

            using (new EditorGUI.DisabledScope(!hasValidTemplate))
            {
                if (GUILayout.Button("Generate Batch", GUILayout.Height(30)))
                {
                    Undo.RegisterFullObjectHierarchyUndo(batch.gameObject, "Generate Batch");
                    batch.GenerateBatch();
                    EditorUtility.SetDirty(batch.gameObject);
                }
            }

            EditorGUILayout.Space(5);

            if (GUILayout.Button("Clear Batch", GUILayout.Height(30)))
            {
                Undo.RegisterFullObjectHierarchyUndo(batch.gameObject, "Clear Batch");
                batch.ClearBatch();
                EditorUtility.SetDirty(batch.gameObject);
            }

            // === 데이터 관리 ===
            EditorGUILayout.Space(10);
            EditorGUILayout.LabelField("데이터 관리", EditorStyles.boldLabel);
            EditorGUILayout.Space(3);

            string rootPath = GetCityGeneratorRootPath();

            EditorGUILayout.BeginHorizontal();

            GUI.backgroundColor = new Color(1f, 0.6f, 0.3f); // 주황
            if (GUILayout.Button("CityData 비우기", GUILayout.Height(28)))
            {
                string folderPath = rootPath + "/CityData";
                if (EditorUtility.DisplayDialog(
                    "내보내기 파일 삭제",
                    $"폴더 안의 모든 파일을 삭제합니다.\n{folderPath}\n\n이 작업은 되돌릴 수 없습니다.",
                    "삭제", "취소"))
                {
                    DeleteFolder(folderPath);
                }
            }

            GUI.backgroundColor = new Color(1f, 0.6f, 0.3f); // 주황
            if (GUILayout.Button("CityMaps 비우기", GUILayout.Height(28)))
            {
                string folderPath = rootPath + "/CityMaps";
                if (EditorUtility.DisplayDialog(
                    "내보내기 파일 삭제",
                    $"폴더 안의 모든 파일을 삭제합니다.\n{folderPath}\n\n이 작업은 되돌릴 수 없습니다.",
                    "삭제", "취소"))
                {
                    DeleteFolder(folderPath);
                }
            }

            GUI.backgroundColor = Color.white;
            EditorGUILayout.EndHorizontal();
        }

        /// <summary>
        /// Assets 상대 경로 폴더 안의 모든 에셋을 삭제합니다 (하위 폴더는 유지).
        /// </summary>
        private static void DeleteFolder(string folder)
        {
            if (!AssetDatabase.IsValidFolder(folder))
            {
                Debug.Log($"CityBatchGeneratorEditor: '{folder}' 폴더가 없습니다. 건너뜁니다.");
                return;
            }

            int deletedCount = 0;
            string[] guids = AssetDatabase.FindAssets("", new[] { folder });
            foreach (string guid in guids)
            {
                string path = AssetDatabase.GUIDToAssetPath(guid);
                if (AssetDatabase.IsValidFolder(path)) continue;

                if (AssetDatabase.DeleteAsset(path))
                    deletedCount++;
            }

            AssetDatabase.Refresh();
            Debug.Log($"CityBatchGeneratorEditor: '{folder}' — {deletedCount}개 파일 삭제 완료.");
        }
    }
}
