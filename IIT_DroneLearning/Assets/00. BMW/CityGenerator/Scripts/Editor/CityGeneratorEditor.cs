using UnityEditor;
using UnityEngine;
using System.IO;

namespace ProceduralCityGenerator
{
    /// <summary>
    /// CityGenerator 컴포넌트를 위한 Custom Inspector
    /// Requirements: 6.1, 13.5, 14.1, 14.3, 2.1, 3.1-3.4, 4.1, 5.1-5.2, 8.1-8.2, 9.1, 10.1, 16.3
    /// </summary>
    [CustomEditor(typeof(CityGenerator))]
    public class CityGeneratorEditor : UnityEditor.Editor
    {
        /// <summary>
        /// 이 스크립트가 위치한 CityGenerator 루트 폴더의 Assets 상대 경로를 반환합니다.
        /// 예: "Assets/00. BMW/CityGenerator"
        /// </summary>
        private string GetCityGeneratorRootPath()
        {
            var script = MonoScript.FromMonoBehaviour((CityGenerator)target);
            string scriptPath = AssetDatabase.GetAssetPath(script);
            // scriptPath: "Assets/00. BMW/CityGenerator/Scripts/CityGenerator.cs"
            string scriptsDir = Path.GetDirectoryName(scriptPath);   // .../Scripts
            string rootDir    = Path.GetDirectoryName(scriptsDir);   // .../CityGenerator
            return rootDir.Replace('\\', '/');
        }

        /// <summary>
        /// Assets 상대 경로 폴더 안의 모든 파일을 삭제합니다 (하위 폴더는 유지).
        /// </summary>
        private void ClearFolderContents(string assetFolderPath)
        {
            // Assets/... → 프로젝트 루트 기준 절대 경로
            string projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
            string fullPath    = Path.GetFullPath(Path.Combine(projectRoot, assetFolderPath));

            if (!Directory.Exists(fullPath))
            {
                Debug.LogWarning($"[CityGenerator] 폴더가 없습니다: {assetFolderPath}");
                return;
            }

            string[] files = Directory.GetFiles(fullPath, "*", SearchOption.TopDirectoryOnly);
            int count = 0;
            foreach (string file in files)
            {
                File.Delete(file);
                count++;
            }

            AssetDatabase.Refresh();
            Debug.Log($"[CityGenerator] {assetFolderPath} — {count}개 파일 삭제 완료.");
        }
        /// <summary>
        /// Inspector GUI를 그립니다.
        /// </summary>
        public override void OnInspectorGUI()
        {
            // 대상 CityGenerator 가져오기
            CityGenerator cityGenerator = (CityGenerator)target;

            // 변경 사항 추적 시작
            EditorGUI.BeginChangeCheck();

            // === Grid Settings 섹션 ===
            EditorGUILayout.Space(5);
            EditorGUILayout.LabelField("Grid Settings", EditorStyles.boldLabel);
            EditorGUILayout.Space(3);

            // Requirement 2.1: 단위_거리 파라미터
            cityGenerator.unitDistance = EditorGUILayout.Slider(
                new GUIContent("Unit Distance", "격자 시스템의 1 단위가 나타내는 실제 거리 (미터). 범위: 0.1 ~ 100.0"),
                cityGenerator.unitDistance,
                0.1f,
                100f);

            EditorGUILayout.Space(3);

            // Requirements 3.1-3.4: 도시 크기 제어
            cityGenerator.minWidth = EditorGUILayout.IntSlider(
                new GUIContent("Min Width", "도시 최소 가로 길이 (격자 단위). 범위: 1 ~ 100"),
                cityGenerator.minWidth,
                1,
                100);

            cityGenerator.maxWidth = EditorGUILayout.IntSlider(
                new GUIContent("Max Width", "도시 최대 가로 길이 (격자 단위). 범위: 1 ~ 100"),
                cityGenerator.maxWidth,
                1,
                100);

            cityGenerator.minDepth = EditorGUILayout.IntSlider(
                new GUIContent("Min Depth", "도시 최소 세로 길이 (격자 단위). 범위: 1 ~ 100"),
                cityGenerator.minDepth,
                1,
                100);

            cityGenerator.maxDepth = EditorGUILayout.IntSlider(
                new GUIContent("Max Depth", "도시 최대 세로 길이 (격자 단위). 범위: 1 ~ 100"),
                cityGenerator.maxDepth,
                1,
                100);

            // === Building Settings 섹션 ===
            EditorGUILayout.Space(10);
            EditorGUILayout.LabelField("Building Settings", EditorStyles.boldLabel);
            EditorGUILayout.Space(3);

            // Requirements 8.1-8.2: 건물 크기 설정
            cityGenerator.buildingWidth = EditorGUILayout.Slider(
                new GUIContent("Building Width", "건물 가로 크기 (단위_거리). 범위: 0.5 ~ 50.0"),
                cityGenerator.buildingWidth,
                0.5f,
                50f);

            cityGenerator.buildingDepth = EditorGUILayout.Slider(
                new GUIContent("Building Depth", "건물 세로 크기 (단위_거리). 범위: 0.5 ~ 50.0"),
                cityGenerator.buildingDepth,
                0.5f,
                50f);

            EditorGUILayout.Space(3);

            // Requirements 5.1-5.2: 건물 높이 제어
            cityGenerator.minBuildingHeight = EditorGUILayout.Slider(
                new GUIContent("Min Building Height", "건물 최소 높이 (단위_거리). 범위: 1.0 ~ 500.0"),
                cityGenerator.minBuildingHeight,
                1f,
                500f);

            cityGenerator.maxBuildingHeight = EditorGUILayout.Slider(
                new GUIContent("Max Building Height", "건물 최대 높이 (단위_거리). 범위: 1.0 ~ 500.0"),
                cityGenerator.maxBuildingHeight,
                1f,
                500f);

            EditorGUILayout.Space(3);

            // Requirement 4.1: 건물 간격 설정
            cityGenerator.buildingSpacing = EditorGUILayout.Slider(
                new GUIContent("Building Spacing", "건물 사이의 간격 (단위_거리). 범위: 0.0 ~ 50.0. 0.0일 때 간격 없이 인접 배치"),
                cityGenerator.buildingSpacing,
                0f,
                50f);

            // Requirement 10.1: 건물 밀도 제어
            cityGenerator.buildingDensity = EditorGUILayout.Slider(
                new GUIContent("Building Density", "격자 셀에 건물이 배치될 확률. 범위: 0.0 ~ 1.0. 1.0일 때 모든 셀에 건물 배치"),
                cityGenerator.buildingDensity,
                0f,
                1f);

            // === Generation Settings 섹션 ===
            EditorGUILayout.Space(10);
            EditorGUILayout.LabelField("Generation Settings", EditorStyles.boldLabel);
            EditorGUILayout.Space(3);

            // Requirement 9.1: 랜덤 시드 제어
            cityGenerator.randomSeed = EditorGUILayout.IntField(
                new GUIContent("Random Seed", "재현 가능한 도시 생성을 위한 시드값. -1: 시간 기반 랜덤 시드 사용"),
                cityGenerator.randomSeed);

            // 기본 건물 머티리얼
            cityGenerator.defaultBuildingMaterial = (Material)EditorGUILayout.ObjectField(
                new GUIContent("Default Building Material", "건물에 적용될 기본 머티리얼"),
                cityGenerator.defaultBuildingMaterial,
                typeof(Material),
                false);

            // === Minimap Settings 섹션 ===
            EditorGUILayout.Space(10);
            EditorGUILayout.LabelField("Minimap Settings", EditorStyles.boldLabel);
            EditorGUILayout.Space(3);

            // Requirement 16.3: 미니맵 해상도
            cityGenerator.minimapResolution = (MinimapResolution)EditorGUILayout.EnumPopup(
                new GUIContent("Minimap Resolution", "생성될 미니맵의 해상도. 지원: 256x256, 512x512, 1024x1024"),
                cityGenerator.minimapResolution);

            // === Wall Settings 섹션 ===
            EditorGUILayout.Space(10);
            EditorGUILayout.LabelField("Wall Settings", EditorStyles.boldLabel);
            EditorGUILayout.Space(3);

            cityGenerator.spawnWalls = EditorGUILayout.Toggle(
                new GUIContent("Spawn Walls", "도시 경계에 4면 벽을 생성합니다. (기본: 비활성)"),
                cityGenerator.spawnWalls);

            if (cityGenerator.spawnWalls)
            {
                EditorGUI.indentLevel++;
                cityGenerator.wallHeight = EditorGUILayout.Slider(
                    new GUIContent("Wall Height", "벽의 높이 (단위_거리). 범위: 1 ~ 200"),
                    cityGenerator.wallHeight, 1f, 200f);

                cityGenerator.wallThickness = EditorGUILayout.Slider(
                    new GUIContent("Wall Thickness", "벽의 두께 (단위_거리). 범위: 0.1 ~ 20"),
                    cityGenerator.wallThickness, 0.1f, 20f);

                cityGenerator.wallMaterial = (Material)EditorGUILayout.ObjectField(
                    new GUIContent("Wall Material", "벽에 사용할 머티리얼 (미설정 시 기본 흰색)"),
                    cityGenerator.wallMaterial, typeof(Material), false);
                EditorGUI.indentLevel--;
            }

            // === Floor Settings 섹션 ===
            EditorGUILayout.Space(10);
            EditorGUILayout.LabelField("Floor Settings", EditorStyles.boldLabel);
            EditorGUILayout.Space(3);

            cityGenerator.spawnFloor = EditorGUILayout.Toggle(
                new GUIContent("Spawn Floor", "도시 바닥(Plane)을 생성합니다. 도시 크기에 자동으로 맞춰집니다. (기본: 비활성)"),
                cityGenerator.spawnFloor);

            if (cityGenerator.spawnFloor)
            {
                EditorGUI.indentLevel++;
                cityGenerator.floorMaterial = (Material)EditorGUILayout.ObjectField(
                    new GUIContent("Floor Material", "바닥에 사용할 머티리얼 (미설정 시 기본 흰색)"),
                    cityGenerator.floorMaterial, typeof(Material), false);
                EditorGUI.indentLevel--;
            }

            // === Layout Mode 섹션 ===
            EditorGUILayout.Space(10);
            EditorGUILayout.LabelField("Layout Mode", EditorStyles.boldLabel);
            EditorGUILayout.Space(3);

            // 3개 토글 버튼으로 레이아웃 모드 선택
            EditorGUILayout.BeginHorizontal();
            DrawLayoutModeButton(cityGenerator, CityLayoutMode.PureGrid,    "Pure Grid",    "완전 격자 — 균일한 블록 간격");
            DrawLayoutModeButton(cityGenerator, CityLayoutMode.Hybrid,      "Hybrid",       "격자 + 랜덤 오프셋 — 불규칙 블록·크기");
            DrawLayoutModeButton(cityGenerator, CityLayoutMode.PureRandom,  "Pure Random",  "유기적 도로망 — 가장 현실적인 도시 모양");
            EditorGUILayout.EndHorizontal();

            // Hybrid / PureRandom 전용 파라미터 (PureGrid일 때 숨김)
            if (cityGenerator.layoutMode != CityLayoutMode.PureGrid)
            {
                EditorGUILayout.Space(5);

                cityGenerator.randomOffsetStrength = EditorGUILayout.Slider(
                    new GUIContent("Offset Strength",
                        "건물 위치 랜덤 오프셋 세기.\n0 = 격자 위치 고정 / 1 = 최대 불규칙"),
                    cityGenerator.randomOffsetStrength, 0f, 1f);

                EditorGUILayout.Space(3);

                cityGenerator.minBlockSize = EditorGUILayout.IntSlider(
                    new GUIContent("Min Block Size",
                        "블록(도로 간격) 최소 크기 (격자 단위). 작을수록 골목이 촘촘해집니다."),
                    cityGenerator.minBlockSize, 2, 8);

                cityGenerator.maxBlockSize = EditorGUILayout.IntSlider(
                    new GUIContent("Max Block Size",
                        "블록(도로 간격) 최대 크기 (격자 단위). 클수록 큰 블록이 생성됩니다."),
                    cityGenerator.maxBlockSize, 3, 12);

                // min > max 경고
                if (cityGenerator.minBlockSize > cityGenerator.maxBlockSize)
                {
                    EditorGUILayout.HelpBox(
                        "Min Block Size가 Max Block Size보다 큽니다. 값을 조정하세요.",
                        MessageType.Warning);
                }
            }

            // === Spawn Configuration 섹션 ===
            EditorGUILayout.Space(10);
            EditorGUILayout.LabelField("Spawn Configuration", EditorStyles.boldLabel);
            EditorGUILayout.Space(3);

            cityGenerator.autoGenerateSpawns = EditorGUILayout.Toggle(
                new GUIContent("Auto Generate Spawns", "도시 생성 시 Evader 스폰, Pursuer 스폰, 타겟 포인트를 자동으로 배치합니다."),
                cityGenerator.autoGenerateSpawns);

            if (cityGenerator.autoGenerateSpawns)
            {
                EditorGUI.indentLevel++;
                cityGenerator.minSpawnSeparation = EditorGUILayout.Slider(
                    new GUIContent("Min Spawn Separation", "스폰 포인트 간 최소 거리 (월드 단위). 값이 너무 크면 자동으로 완화됩니다."),
                    cityGenerator.minSpawnSeparation, 1f, 200f);

                cityGenerator.minSpawnHeight = EditorGUILayout.Slider(
                    new GUIContent("Min Spawn Height", "드론 스폰/타겟 포인트의 최저 비행 고도 (지면 기준, DronePhysics.MinAltitude=0.5 / DroneAgent.SpawnHeight=8 참고)."),
                    cityGenerator.minSpawnHeight, 0.5f, 50f);

                cityGenerator.maxSpawnHeight = EditorGUILayout.Slider(
                    new GUIContent("Max Spawn Height", "드론 스폰/타겟 포인트의 최고 비행 고도 (지면 기준, DronePhysics.MaxAltitude=50 참고)."),
                    cityGenerator.maxSpawnHeight, 0.5f, 50f);

                // Min이 Max를 초과하지 않도록 보정
                if (cityGenerator.minSpawnHeight > cityGenerator.maxSpawnHeight)
                    cityGenerator.maxSpawnHeight = cityGenerator.minSpawnHeight;

                EditorGUI.indentLevel--;

                // 생성 결과 표시 (ReadOnly)
                SpawnConfiguration sc = cityGenerator.GetSpawnConfiguration();
                if (sc.isValid)
                {
                    EditorGUILayout.Space(5);
                    EditorGUILayout.LabelField("— 생성 결과 (ReadOnly) —", EditorStyles.miniLabel);
                    EditorGUI.BeginDisabledGroup(true);
                    EditorGUILayout.Vector3Field(new GUIContent("Evader Spawn", "도망 드론 스폰 위치 (하늘색)"), sc.evaderSpawnPosition);
                    EditorGUILayout.Vector3Field(new GUIContent("Pursuer Spawn", "추적 드론 스폰 위치 (빨간색)"), sc.pursuerSpawnPosition);
                    EditorGUILayout.Vector3Field(new GUIContent("Target Point", "타겟 포인트 위치 (초록색)"), sc.targetPosition);
                    EditorGUILayout.FloatField(new GUIContent("Min Sep Achieved", "실제 달성된 최소 분리 거리 (m)"), sc.achievedMinSeparation);
                    EditorGUI.EndDisabledGroup();
                }
            }

            // 변경 사항이 있으면 저장
            if (EditorGUI.EndChangeCheck())
            {
                EditorUtility.SetDirty(cityGenerator);
            }

            // 공간 추가
            EditorGUILayout.Space(10);

            // 버튼 섹션
            EditorGUILayout.LabelField("도시 생성 제어", EditorStyles.boldLabel);

            // Requirement 6.1: 생성 버튼 제공
            if (GUILayout.Button("도시 생성 (Generate City)", GUILayout.Height(30)))
            {
                cityGenerator.GenerateCity();
            }

            EditorGUILayout.Space(5);

            // Requirement 13.5: 초기화 버튼 제공
            if (GUILayout.Button("도시 초기화 (Clear City)", GUILayout.Height(30)))
            {
                if (EditorUtility.DisplayDialog(
                    "도시 초기화 확인",
                    "생성된 모든 건물을 제거하시겠습니까?",
                    "예",
                    "아니오"))
                {
                    cityGenerator.ClearCity();
                }
            }

            EditorGUILayout.Space(10);

            // 프리셋 관리 섹션
            EditorGUILayout.LabelField("프리셋 관리", EditorStyles.boldLabel);

            // Requirement 14.1: 프리셋 저장 버튼 제공
            if (GUILayout.Button("프리셋 저장 (Save Preset)", GUILayout.Height(30)))
            {
                string presetName = EditorUtility.SaveFilePanel(
                    "프리셋 저장",
                    "Assets/CityPresets",
                    "CityPreset",
                    "asset");

                if (!string.IsNullOrEmpty(presetName))
                {
                    // 절대 경로를 상대 경로로 변환
                    if (presetName.StartsWith(Application.dataPath))
                    {
                        presetName = "Assets" + presetName.Substring(Application.dataPath.Length);
                    }

                    // 확장자 제거하고 파일명만 추출
                    string fileName = System.IO.Path.GetFileNameWithoutExtension(presetName);
                    cityGenerator.SavePreset(fileName);
                }
            }

            EditorGUILayout.Space(5);

            // Requirement 14.3: 프리셋 로드 버튼 제공
            if (GUILayout.Button("프리셋 로드 (Load Preset)", GUILayout.Height(30)))
            {
                string presetPath = EditorUtility.OpenFilePanel(
                    "프리셋 로드",
                    "Assets/CityPresets",
                    "asset");

                if (!string.IsNullOrEmpty(presetPath))
                {
                    // 절대 경로를 상대 경로로 변환
                    if (presetPath.StartsWith(Application.dataPath))
                    {
                        presetPath = "Assets" + presetPath.Substring(Application.dataPath.Length);
                    }

                    // ScriptableObject 로드
                    CityParameters preset = AssetDatabase.LoadAssetAtPath<CityParameters>(presetPath);
                    
                    if (preset != null)
                    {
                        cityGenerator.LoadPreset(preset);
                        // Inspector 업데이트
                        EditorUtility.SetDirty(cityGenerator);
                    }
                    else
                    {
                        EditorUtility.DisplayDialog(
                            "프리셋 로드 실패",
                            "프리셋 파일을 로드할 수 없습니다.",
                            "확인");
                    }
                }
            }

            // === 데이터 관리 섹션 ===
            EditorGUILayout.Space(10);
            EditorGUILayout.LabelField("데이터 관리", EditorStyles.boldLabel);
            EditorGUILayout.Space(3);

            string rootPath = GetCityGeneratorRootPath();

            EditorGUILayout.BeginHorizontal();

            // CityData 비우기
            GUI.backgroundColor = new Color(1f, 0.6f, 0.3f); // 주황
            if (GUILayout.Button("CityData 비우기", GUILayout.Height(28)))
            {
                string folderPath = rootPath + "/CityData";
                if (EditorUtility.DisplayDialog(
                    "CityData 비우기",
                    $"폴더 안의 모든 파일을 삭제합니다.\n{folderPath}\n\n계속하시겠습니까?",
                    "삭제", "취소"))
                {
                    ClearFolderContents(folderPath);
                }
            }

            // CityMaps 비우기
            GUI.backgroundColor = new Color(1f, 0.6f, 0.3f); // 주황
            if (GUILayout.Button("CityMaps 비우기", GUILayout.Height(28)))
            {
                string folderPath = rootPath + "/CityMaps";
                if (EditorUtility.DisplayDialog(
                    "CityMaps 비우기",
                    $"폴더 안의 모든 파일을 삭제합니다.\n{folderPath}\n\n계속하시겠습니까?",
                    "삭제", "취소"))
                {
                    ClearFolderContents(folderPath);
                }
            }

            GUI.backgroundColor = Color.white;
            EditorGUILayout.EndHorizontal();

            // 경로 표시 (읽기전용 참고용)
            EditorGUILayout.Space(3);
            EditorGUI.BeginDisabledGroup(true);
            EditorGUILayout.TextField("루트 경로", rootPath);
            EditorGUI.EndDisabledGroup();
        }

        /// <summary>
        /// 레이아웃 모드 선택용 토글 버튼을 그립니다.
        /// 현재 선택된 모드는 강조(진한 배경)로 표시됩니다.
        /// </summary>
        private void DrawLayoutModeButton(
            CityGenerator gen, CityLayoutMode mode, string label, string tooltip)
        {
            bool selected = gen.layoutMode == mode;

            // 선택된 버튼은 밝은 하늘색 배경으로 강조
            GUI.backgroundColor = selected
                ? new Color(0.4f, 0.8f, 1.0f)   // 선택 = 하늘색
                : new Color(0.85f, 0.85f, 0.85f); // 미선택 = 회색

            GUIStyle style = new GUIStyle(GUI.skin.button)
            {
                fontStyle = selected ? FontStyle.Bold : FontStyle.Normal,
                fixedHeight = 28
            };

            if (GUILayout.Button(new GUIContent(label, tooltip), style))
                gen.layoutMode = mode;

            GUI.backgroundColor = Color.white; // 색상 초기화
        }
    }
}
