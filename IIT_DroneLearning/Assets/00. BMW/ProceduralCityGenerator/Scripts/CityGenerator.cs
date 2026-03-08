using System.Collections.Generic;
using UnityEngine;

namespace ProceduralCityGenerator
{
    /// <summary>
    /// 프로시저럴 도시 생성의 중심 컴포넌트
    /// Unity GameObject에 부착되어 Inspector를 통해 제어됩니다.
    /// </summary>
    public class CityGenerator : MonoBehaviour
    {
        #region Parameters

        [Header("Grid Settings")]
        [Tooltip("격자 시스템의 1 단위가 나타내는 실제 거리 (미터)")]
        [Range(0.1f, 100f)]
        public float unitDistance = 1.0f;

        [Tooltip("도시 최소 가로 길이 (격자 단위)")]
        [Range(1, 100)]
        public int minWidth = 10;

        [Tooltip("도시 최대 가로 길이 (격자 단위)")]
        [Range(1, 100)]
        public int maxWidth = 20;

        [Tooltip("도시 최소 세로 길이 (격자 단위)")]
        [Range(1, 100)]
        public int minDepth = 10;

        [Tooltip("도시 최대 세로 길이 (격자 단위)")]
        [Range(1, 100)]
        public int maxDepth = 20;

        [Header("Building Settings")]
        [Tooltip("건물 가로 크기 (단위_거리)")]
        [Range(0.5f, 50f)]
        public float buildingWidth = 1.0f;

        [Tooltip("건물 세로 크기 (단위_거리)")]
        [Range(0.5f, 50f)]
        public float buildingDepth = 1.0f;

        [Tooltip("건물 최소 높이 (단위_거리)")]
        [Range(1f, 500f)]
        public float minBuildingHeight = 5.0f;

        [Tooltip("건물 최대 높이 (단위_거리)")]
        [Range(1f, 500f)]
        public float maxBuildingHeight = 20.0f;

        [Tooltip("건물 사이의 간격 (단위_거리)")]
        [Range(0f, 50f)]
        public float buildingSpacing = 1.0f;

        [Tooltip("격자 셀에 건물이 배치될 확률 (0.0 ~ 1.0)")]
        [Range(0f, 1f)]
        public float buildingDensity = 0.7f;

        [Header("Generation Settings")]
        [Tooltip("재현 가능한 도시 생성을 위한 시드값 (-1: 시간 기반 랜덤)")]
        public int randomSeed = -1;

        [Tooltip("기본 건물 머티리얼")]
        public Material defaultBuildingMaterial;

        [Header("Minimap Settings")]
        [Tooltip("생성될 미니맵의 해상도")]
        public MinimapResolution minimapResolution = MinimapResolution.Resolution512;

        #endregion

        #region Internal State

        private GameObject cityRoot;
        private CityGraph cityGraph;
        private SpatialIndex spatialIndex;
        private BuildingFactory buildingFactory;
        private MinimapGenerator minimapGenerator;
        private List<Building> buildings;
        private List<StrategicLocation> strategicLocations;
        private GridCell[,] grid;
        private int actualCityWidth;
        private int actualCityDepth;
        private int usedRandomSeed;

        #endregion

        #region Unity Lifecycle Methods

        /// <summary>
        /// Awake는 스크립트 인스턴스가 로드될 때 호출됩니다.
        /// </summary>
        private void Awake()
        {
            // 내부 상태 초기화
            buildings = new List<Building>();
            
            // 기본 머티리얼이 없으면 경고
            if (defaultBuildingMaterial == null)
            {
                Debug.LogWarning("CityGenerator.Awake: defaultBuildingMaterial이 설정되지 않았습니다. 기본 머티리얼을 사용합니다.");
            }
        }

        /// <summary>
        /// Start는 첫 프레임 업데이트 전에 호출됩니다.
        /// </summary>
        private void Start()
        {
            // 필요한 경우 여기서 추가 초기화를 수행할 수 있습니다.
            Debug.Log("CityGenerator.Start: 초기화 완료");
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// 현재 파라미터를 기반으로 새로운 도시를 생성합니다.
        /// Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
        /// </summary>
        /// <returns>도시 생성 결과</returns>
        public CityGenerationResult GenerateCity()
        {
            // 생성 시간 측정 시작
            System.Diagnostics.Stopwatch stopwatch = System.Diagnostics.Stopwatch.StartNew();

            CityGenerationResult result = new CityGenerationResult();

            try
            {
                // Requirement 6.3: 새로운 도시를 생성할 때, 이전에 생성된 모든 건물을 제거
                ClearCity();

                // Requirement 12.6: 도시 생성을 시작하기 전에 모든 파라미터를 검증
                ValidationResult validationResult = ValidateParameters();
                
                if (!validationResult.isValid)
                {
                    // Requirement 6.5: 도시 생성이 실패하면, Unity 콘솔에 오류 메시지를 기록
                    result.errorMessage = "파라미터 검증 실패로 도시 생성을 중단합니다.";
                    Debug.LogError($"CityGenerator.GenerateCity: {result.errorMessage}");
                    return result;
                }

                if (validationResult.warnings.Count > 0)
                {
                    Debug.LogWarning($"CityGenerator.GenerateCity: {validationResult.warnings.Count}개의 경고가 있지만 도시 생성을 계속합니다.");
                }

                // Requirement 6.2: 생성 버튼이 클릭되면, 현재 파라미터를 기반으로 새로운 도시를 생성
                Debug.Log("CityGenerator.GenerateCity: 도시 생성 시작");

                // 랜덤 시드 초기화
                InitializeRandomSeed();
                result.usedRandomSeed = usedRandomSeed;

                // 격자 생성
                CreateGridLayout();

                // 건물 배치
                bool placementSuccess = PlaceBuildingsOnGrid();
                
                // 취소되었으면 생성 중단
                if (!placementSuccess)
                {
                    result.success = false;
                    result.errorMessage = "사용자가 도시 생성을 취소했습니다.";
                    stopwatch.Stop();
                    result.generationTime = (float)stopwatch.Elapsed.TotalSeconds;
                    return result;
                }
                
                result.buildingCount = buildings.Count;

                // 그래프 구축
                BuildCityGraph();
                result.nodeCount = cityGraph.NodeCount;
                result.edgeCount = cityGraph.EdgeCount;
                result.graph = cityGraph;

                // 미니맵 생성 (탑뷰 이미지 자동 생성 및 PNG 저장)
                result.minimap = GenerateMinimap();

                // 그래프 데이터 내보내기 (JSON + CSV) — 도망자 드론 오프라인 지형 분석용
                ExportGraphData();

                // 생성 성공
                result.success = true;

                // 생성 시간 측정 종료
                stopwatch.Stop();
                result.generationTime = (float)stopwatch.Elapsed.TotalSeconds;

                // Requirement 6.4: 도시 생성이 완료되면, 생성된 건물 수를 Unity 콘솔에 기록
                Debug.Log($"CityGenerator.GenerateCity: 도시 생성 완료! " +
                         $"건물: {result.buildingCount}개, " +
                         $"노드: {result.nodeCount}개, " +
                         $"엣지: {result.edgeCount}개, " +
                         $"생성 시간: {result.generationTime:F2}초, " +
                         $"사용된 시드: {result.usedRandomSeed}");
            }
            catch (System.Exception ex)
            {
                // Requirement 6.5: 도시 생성이 실패하면, Unity 콘솔에 오류 메시지를 기록
                result.success = false;
                result.errorMessage = ex.Message;
                stopwatch.Stop();
                result.generationTime = (float)stopwatch.Elapsed.TotalSeconds;
                
                Debug.LogError($"CityGenerator.GenerateCity: 도시 생성 중 오류 발생 - {ex.Message}\n{ex.StackTrace}");
            }

            return result;
        }

        /// <summary>
        /// 생성된 모든 건물을 제거합니다.
        /// Requirements: 6.3, 13.1, 13.2, 13.5
        /// </summary>
        public void ClearCity()
        {
            Debug.Log("CityGenerator.ClearCity: 도시 제거 시작");

            // Requirement 13.2: 새로운 도시를 생성할 때, 이전 "생성된_도시" GameObject를 파괴
            if (cityRoot != null)
            {
                // Editor 모드에서는 DestroyImmediate 사용
                if (Application.isPlaying)
                {
                    Destroy(cityRoot);
                }
                else
                {
                    DestroyImmediate(cityRoot);
                }
                
                cityRoot = null;
                Debug.Log("CityGenerator.ClearCity: 도시 루트 GameObject 제거 완료");
            }

            // BuildingFactory 풀 초기화
            if (buildingFactory != null)
            {
                buildingFactory.ClearPool();
                buildingFactory = null;
                Debug.Log("CityGenerator.ClearCity: BuildingFactory 풀 초기화 완료");
            }

            // 건물 리스트 초기화
            if (buildings != null)
            {
                buildings.Clear();
            }

            // 그래프 초기화
            if (cityGraph != null)
            {
                cityGraph = null;
            }

            // 공간 인덱스 초기화
            if (spatialIndex != null)
            {
                spatialIndex.Clear();
                spatialIndex = null;
            }

            // 격자 초기화
            grid = null;

            Debug.Log("CityGenerator.ClearCity: 도시 제거 완료");
        }

        /// <summary>
        /// 생성된 도시 그래프를 반환합니다.
        /// </summary>
        public CityGraph GetCityGraph() => cityGraph;

        /// <summary>
        /// 분석된 전략적 위치 리스트를 반환합니다.
        /// </summary>
        public List<StrategicLocation> GetStrategicLocations() => strategicLocations;

        /// <summary>
        /// 현재 도시를 탑뷰(위에서 내려다본) 미니맵 이미지로 생성하고 PNG로 저장합니다.
        /// </summary>
        /// <returns>생성된 미니맵 Texture2D, 실패 시 null</returns>
        private Texture2D GenerateMinimap()
        {
            if (buildings == null || buildings.Count == 0)
            {
                Debug.LogWarning("CityGenerator.GenerateMinimap: 건물 데이터가 없어 미니맵을 생성할 수 없습니다.");
                return null;
            }

            // 도시 경계(Bounds) 계산 — 모든 건물을 포함하는 최소 직육면체
            Bounds cityBounds = new Bounds(buildings[0].position, Vector3.zero);
            foreach (Building b in buildings)
            {
                cityBounds.Encapsulate(new Bounds(b.position, b.size));
            }

            // MinimapGenerator 생성 (Inspector에서 설정한 해상도 사용)
            int resolutionInt = (int)minimapResolution;
            minimapGenerator = new MinimapGenerator(resolutionInt, cityBounds);

            // 탑뷰 이미지 생성 (건물 높이별 등고선 색상 + 전략적 위치 마커)
            Texture2D minimap = minimapGenerator.GenerateMinimap(
                buildings.ToArray(),
                cityGraph,
                strategicLocations
            );

            // Assets/CityMaps/ 폴더에 PNG 자동 저장
            minimapGenerator.SaveMinimapToPNG(minimap, usedRandomSeed);

            Debug.Log($"CityGenerator.GenerateMinimap: {resolutionInt}x{resolutionInt} 미니맵 생성 완료 (Seed: {usedRandomSeed})");
            return minimap;
        }

        /// <summary>
        /// 도시 그래프 데이터를 JSON 및 CSV 파일로 내보냅니다.
        /// Assets/CityData/ 폴더에 저장되며, 도망자 드론의 오프라인 지형 분석에 활용됩니다.
        /// </summary>
        private void ExportGraphData()
        {
            if (cityGraph == null)
            {
                Debug.LogWarning("CityGenerator.ExportGraphData: cityGraph가 null입니다.");
                return;
            }

            // 도시 경계 계산 (GenerateMinimap과 동일한 방식)
            Bounds cityBounds = new Bounds(Vector3.zero, Vector3.zero);
            if (buildings != null && buildings.Count > 0)
            {
                cityBounds = new Bounds(buildings[0].position, Vector3.zero);
                foreach (Building b in buildings)
                    cityBounds.Encapsulate(new Bounds(b.position, b.size));
            }

            CityGraphExporter.ExportAll(cityGraph, strategicLocations, usedRandomSeed, cityBounds);
        }

        /// <summary>
        /// 현재 파라미터를 프리셋으로 저장합니다.
        /// Requirements: 14.1, 14.2, 14.5
        /// </summary>
        /// <param name="presetName">프리셋 이름</param>
        public void SavePreset(string presetName)
        {
#if UNITY_EDITOR
            if (string.IsNullOrEmpty(presetName))
            {
                Debug.LogError("CityGenerator.SavePreset: 프리셋 이름이 비어있습니다.");
                return;
            }

            // Requirement 14.5: Assets/CityPresets 디렉토리에 저장
            string directoryPath = "Assets/CityPresets";
            
            // 디렉토리가 없으면 생성
            if (!UnityEditor.AssetDatabase.IsValidFolder(directoryPath))
            {
                // Assets 폴더가 존재하는지 확인
                if (!UnityEditor.AssetDatabase.IsValidFolder("Assets"))
                {
                    Debug.LogError("CityGenerator.SavePreset: Assets 폴더를 찾을 수 없습니다.");
                    return;
                }
                
                // CityPresets 폴더 생성
                UnityEditor.AssetDatabase.CreateFolder("Assets", "CityPresets");
                Debug.Log($"CityGenerator.SavePreset: {directoryPath} 디렉토리 생성 완료");
            }

            // Requirement 14.2: 현재 파라미터를 ScriptableObject 에셋으로 직렬화
            // CityParameters ScriptableObject 인스턴스 생성
            CityParameters preset = ScriptableObject.CreateInstance<CityParameters>();
            
            // 현재 파라미터를 ScriptableObject에 복사
            preset.CopyFrom(this);
            
            // 파일 경로 생성 (중복 방지를 위해 타임스탬프 추가 가능)
            string assetPath = $"{directoryPath}/{presetName}.asset";
            
            // 이미 존재하는 파일인지 확인
            if (UnityEditor.AssetDatabase.LoadAssetAtPath<CityParameters>(assetPath) != null)
            {
                Debug.LogWarning($"CityGenerator.SavePreset: {assetPath} 파일이 이미 존재합니다. 덮어쓰기합니다.");
            }
            
            // AssetDatabase를 사용하여 에셋 생성
            UnityEditor.AssetDatabase.CreateAsset(preset, assetPath);
            
            // AssetDatabase 새로고침
            UnityEditor.AssetDatabase.SaveAssets();
            UnityEditor.AssetDatabase.Refresh();
            
            Debug.Log($"CityGenerator.SavePreset: 프리셋 저장 완료 - {assetPath}");
#else
            Debug.LogWarning("CityGenerator.SavePreset: 이 기능은 Unity Editor에서만 사용할 수 있습니다.");
#endif
        }

        /// <summary>
        /// 프리셋에서 파라미터를 로드합니다.
        /// </summary>
        /// <param name="preset">로드할 CityParameters</param>
        public void LoadPreset(CityParameters preset)
        {
            if (preset == null)
            {
                Debug.LogError("CityGenerator.LoadPreset: preset is null");
                return;
            }

            preset.ApplyTo(this);
            Debug.Log($"CityGenerator.LoadPreset: {preset.name} 로드 완료");
        }

        #endregion

        #region Private Methods

        /// <summary>
        /// 모든 파라미터를 검증하고 필요한 경우 수정합니다.
        /// Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6
        /// </summary>
        /// <returns>검증 결과</returns>
        private ValidationResult ValidateParameters()
        {
            ValidationResult result = new ValidationResult(true);

            // 1. 범위 제한 (Clamp) 로직 - Requirement 12.1
            ClampParameter(ref unitDistance, 0.1f, 100f, "unitDistance", ref result);
            ClampParameter(ref minWidth, 1, 100, "minWidth", ref result);
            ClampParameter(ref maxWidth, 1, 100, "maxWidth", ref result);
            ClampParameter(ref minDepth, 1, 100, "minDepth", ref result);
            ClampParameter(ref maxDepth, 1, 100, "maxDepth", ref result);
            ClampParameter(ref buildingWidth, 0.5f, 50f, "buildingWidth", ref result);
            ClampParameter(ref buildingDepth, 0.5f, 50f, "buildingDepth", ref result);
            ClampParameter(ref minBuildingHeight, 1f, 500f, "minBuildingHeight", ref result);
            ClampParameter(ref maxBuildingHeight, 1f, 500f, "maxBuildingHeight", ref result);
            ClampParameter(ref buildingSpacing, 0f, 50f, "buildingSpacing", ref result);
            ClampParameter(ref buildingDensity, 0f, 1f, "buildingDensity", ref result);

            // 2. 최소/최대 값 교환 로직 - Requirements 12.3, 12.4, 12.5
            SwapIfNeeded(ref minBuildingHeight, ref maxBuildingHeight, "건물 높이", ref result);
            SwapIfNeeded(ref minWidth, ref maxWidth, "가로 길이", ref result);
            SwapIfNeeded(ref minDepth, ref maxDepth, "세로 길이", ref result);

            // 3. 경고 및 오류 메시지 로깅 - Requirement 12.2
            foreach (var warning in result.warnings)
            {
                Debug.LogWarning($"CityGenerator.ValidateParameters: {warning}");
            }

            foreach (var error in result.errors)
            {
                Debug.LogError($"CityGenerator.ValidateParameters: {error}");
            }

            return result;
        }

        /// <summary>
        /// float 파라미터를 지정된 범위로 제한합니다.
        /// </summary>
        private void ClampParameter(ref float value, float min, float max, string paramName, ref ValidationResult result)
        {
            float original = value;
            value = Mathf.Clamp(value, min, max);

            if (!Mathf.Approximately(original, value))
            {
                result.clampedValues[paramName] = (original, value);
                result.warnings.Add($"{paramName} 값이 {original}에서 {value}로 제한되었습니다. (유효 범위: {min} ~ {max})");
            }
        }

        /// <summary>
        /// int 파라미터를 지정된 범위로 제한합니다.
        /// </summary>
        private void ClampParameter(ref int value, int min, int max, string paramName, ref ValidationResult result)
        {
            int original = value;
            value = Mathf.Clamp(value, min, max);

            if (original != value)
            {
                result.clampedValues[paramName] = (original, value);
                result.warnings.Add($"{paramName} 값이 {original}에서 {value}로 제한되었습니다. (유효 범위: {min} ~ {max})");
            }
        }

        /// <summary>
        /// 최소값이 최대값보다 크면 두 값을 교환합니다.
        /// </summary>
        private void SwapIfNeeded(ref float min, ref float max, string paramDescription, ref ValidationResult result)
        {
            if (min > max)
            {
                float temp = min;
                min = max;
                max = temp;
                result.warnings.Add($"{paramDescription}의 최소값({max})이 최대값({min})보다 커서 값을 교환했습니다.");
            }
        }

        /// <summary>
        /// 최소값이 최대값보다 크면 두 값을 교환합니다.
        /// </summary>
        private void SwapIfNeeded(ref int min, ref int max, string paramDescription, ref ValidationResult result)
        {
            if (min > max)
            {
                int temp = min;
                min = max;
                max = temp;
                result.warnings.Add($"{paramDescription}의 최소값({max})이 최대값({min})보다 커서 값을 교환했습니다.");
            }
        }

        /// <summary>
        /// 랜덤 시드를 초기화합니다.
        /// Requirements: 9.1, 9.2, 9.3, 9.5
        /// </summary>
        private void InitializeRandomSeed()
        {
            // Requirement 9.3: randomSeed가 -1일 때 시간 기반 시드 생성
            if (randomSeed == -1)
            {
                usedRandomSeed = System.Environment.TickCount;
            }
            else
            {
                // Requirement 9.2: 특정 시드 값 사용 (재현 가능한 생성)
                usedRandomSeed = randomSeed;
            }

            // Unity.Random.InitState 호출
            Random.InitState(usedRandomSeed);

            // Requirement 9.5: 사용된 시드 값 로깅
            Debug.Log($"CityGenerator.InitializeRandomSeed: 랜덤 시드 초기화 완료. 사용된 시드: {usedRandomSeed}");
        }

        /// <summary>
        /// 격자 레이아웃을 생성합니다.
        /// Requirements: 3.5, 3.6, 7.1, 7.2, 7.3
        /// </summary>
        private void CreateGridLayout()
        {
            // Requirement 3.5: 최소_가로길이와 최대_가로길이 사이의 가로 크기로 격자를 생성
            actualCityWidth = Random.Range(minWidth, maxWidth + 1);
            
            // Requirement 3.6: 최소_세로길이와 최대_세로길이 사이의 세로 크기로 격자를 생성
            actualCityDepth = Random.Range(minDepth, maxDepth + 1);

            Debug.Log($"CityGenerator.CreateGridLayout: 도시 크기 결정 완료. 가로: {actualCityWidth}, 세로: {actualCityDepth}");

            // Requirement 7.1: 2D 격자 구조 생성
            grid = new GridCell[actualCityWidth, actualCityDepth];

            // 각 셀의 월드 좌표 계산 (CityGenerator 오브젝트 위치를 도시 중심으로 정렬)
            float cellW = (buildingWidth + buildingSpacing) * unitDistance;
            float cellD = (buildingDepth + buildingSpacing) * unitDistance;
            float originX = transform.position.x - (actualCityWidth  * cellW) / 2f;
            float originZ = transform.position.z - (actualCityDepth * cellD) / 2f;

            for (int x = 0; x < actualCityWidth; x++)
            {
                for (int z = 0; z < actualCityDepth; z++)
                {
                    // Requirement 7.2: 위치를 (격자_x * (건물_가로 + 건물_간격) * 단위_거리, 0, 격자_z * (건물_세로 + 건물_간격) * 단위_거리)로 계산
                    float worldX = originX + x * cellW;
                    float worldZ = originZ + z * cellD;

                    // GridCell 초기화
                    grid[x, z] = new GridCell
                    {
                        x = x,
                        z = z,
                        hasBuilding = false,
                        buildingHeight = 0f,
                        worldPosition = new Vector3(worldX, 0f, worldZ) // Requirement 7.3: 격자 원점에 정렬
                    };
                }
            }

            Debug.Log($"CityGenerator.CreateGridLayout: 격자 레이아웃 생성 완료. 총 {actualCityWidth * actualCityDepth}개의 셀 생성");
        }

        /// <summary>
        /// 격자에 건물을 배치합니다.
        /// Requirements: 1.3, 4.2, 4.3, 5.3, 7.2, 8.3, 8.4, 9.4, 10.3, 10.4, 10.5, 11.4, 11.5
        /// </summary>
        /// <returns>생성이 취소되었으면 false, 완료되었으면 true</returns>
        private bool PlaceBuildingsOnGrid()
        {
            // BuildingFactory 초기화
            if (buildingFactory == null)
            {
                // cityRoot가 없으면 생성
                if (cityRoot == null)
                {
                    cityRoot = new GameObject("City");
                }

                buildingFactory = new BuildingFactory(defaultBuildingMaterial, cityRoot.transform);
            }

            // 건물 리스트 초기화 (Edit Mode에서는 Awake가 호출되지 않으므로 null 가드 필요)
            if (buildings == null)
                buildings = new List<Building>();
            buildings.Clear();

            int buildingsPlaced = 0;
            int totalCells = actualCityWidth * actualCityDepth;
            int processedCells = 0;

            // Requirement 11.3: 건물 생성 작업을 배치 처리
            // 배치 크기 결정 (성능과 응답성의 균형)
            int batchSize = Mathf.Max(50, totalCells / 20); // 최소 50개, 또는 전체의 5%

#if UNITY_EDITOR
            // Requirement 11.4: Unity Editor에 진행률 표시줄을 표시
            // Requirement 11.5: 진행 중인 도시 생성을 취소할 수 있어야 함
            bool showProgressBar = !Application.isPlaying;
#endif

            int cellsInCurrentBatch = 0;

            // 격자의 모든 셀을 순회
            for (int x = 0; x < actualCityWidth; x++)
            {
                for (int z = 0; z < actualCityDepth; z++)
                {
                    processedCells++;
                    cellsInCurrentBatch++;

                    // Requirement 10.3: 건물_밀도와 동일한 확률로 각 격자 셀에 건물을 배치
                    float randomValue = Random.value;
                    
                    // Requirement 10.5: 건물_밀도가 0.0일 때 건물을 생성하지 않음
                    if (randomValue <= buildingDensity)
                    {
                        // Requirement 9.4: 건물 높이를 랜덤으로 결정 (minBuildingHeight~maxBuildingHeight)
                        // Requirement 5.3: 건물을 생성할 때, 최소_건물높이와 최대_건물높이 사이의 높이를 할당
                        float buildingHeight = Random.Range(minBuildingHeight, maxBuildingHeight);

                        // Requirement 7.2: 건물 위치 계산 공식 적용
                        Vector3 position = grid[x, z].worldPosition;
                        
                        // 건물의 중심이 위치에 오도록 Y 좌표 조정
                        position.y = (buildingHeight * unitDistance) / 2f;

                        // Requirement 8.3, 8.4: 건물 크기 설정
                        Vector3 scale = new Vector3(
                            buildingWidth * unitDistance,
                            buildingHeight * unitDistance,
                            buildingDepth * unitDistance
                        );

                        // 건물 이름 설정 (격자 좌표 기반)
                        string buildingName = $"Building_{x}_{z}";

                        // Requirement 1.3: BuildingFactory를 통한 건물 생성
                        GameObject buildingGO = buildingFactory.CreateBuilding(position, scale, buildingName);

                        // Building 객체 생성 및 리스트에 추가
                        GridCell cell = grid[x, z];
                        cell.hasBuilding = true;
                        cell.buildingHeight = buildingHeight;
                        grid[x, z] = cell;

                        Building building = new Building(buildingGO, position, scale, buildingHeight, cell);
                        buildings.Add(building);

                        buildingsPlaced++;
                    }

                    // Requirement 11.3, 11.4: 각 배치 후 진행률 표시줄 업데이트
                    if (cellsInCurrentBatch >= batchSize || processedCells >= totalCells)
                    {
#if UNITY_EDITOR
                        // 진행률 표시 및 취소 확인
                        if (showProgressBar)
                        {
                            float progress = (float)processedCells / totalCells;
                            
                            // DisplayCancelableProgressBar를 사용하여 취소 버튼 지원
                            bool cancelled = UnityEditor.EditorUtility.DisplayCancelableProgressBar(
                                "도시 생성 중",
                                $"건물 배치 중... ({buildingsPlaced}개 생성됨, {processedCells}/{totalCells} 셀 처리됨)",
                                progress
                            );

                            // 취소 버튼이 클릭되면 생성 중단
                            if (cancelled)
                            {
                                UnityEditor.EditorUtility.ClearProgressBar();
                                Debug.LogWarning($"CityGenerator.PlaceBuildingsOnGrid: 사용자가 도시 생성을 취소했습니다. {buildingsPlaced}개의 건물이 생성되었습니다.");
                                
                                // 부분 생성된 도시 정리
                                ClearCity();
                                return false;
                            }
                        }
#endif
                        // 배치 카운터 리셋
                        cellsInCurrentBatch = 0;
                    }
                }
            }

#if UNITY_EDITOR
            // 진행률 바 제거
            if (showProgressBar)
            {
                UnityEditor.EditorUtility.ClearProgressBar();
            }
#endif

            Debug.Log($"CityGenerator.PlaceBuildingsOnGrid: 건물 배치 완료. 총 {buildingsPlaced}개의 건물 생성 (밀도: {buildingDensity:P0})");
            return true;
        }

        /// <summary>
        /// 도시 그래프를 구축합니다.
        /// Requirements: 15.1, 15.4, 15.5, 17.1, 17.4
        /// </summary>
        private void BuildCityGraph()
        {
            // Requirement 15.1: 도시_생성기는 도시 생성 시 그래프 자료구조를 함께 생성
            Debug.Log("CityGenerator.BuildCityGraph: 도시 그래프 구축 시작");

            // CityGraph 초기화
            if (cityGraph == null)
            {
                cityGraph = new CityGraph();
            }

            // Requirement 15.4, 15.5: 격자 기반으로 그래프 구축
            cityGraph.BuildFromGrid(grid, unitDistance);
            Debug.Log($"CityGenerator.BuildCityGraph: 그래프 구축 완료. 노드: {cityGraph.NodeCount}, 엣지: {cityGraph.EdgeCount}");

            // Requirement 17.1: 전략적 위치 분석
            // 생성 지점은 도시 중심으로 가정
            Vector3 spawnPoint = new Vector3(
                (actualCityWidth * (buildingWidth + buildingSpacing) * unitDistance) / 2f,
                0f,
                (actualCityDepth * (buildingDepth + buildingSpacing) * unitDistance) / 2f
            );

            strategicLocations = StrategicLocationAnalyzer.AnalyzeLocations(
                cityGraph,
                buildings.ToArray(),
                spawnPoint
            );

            // Requirement 17.4: 전략적 위치를 그래프 노드에 태그 추가
            foreach (StrategicLocation location in strategicLocations)
            {
                if (location.connectedNodes != null && location.connectedNodes.Count > 0)
                {
                    foreach (int nodeId in location.connectedNodes)
                    {
                        if (cityGraph.HasNode(nodeId))
                        {
                            GraphNode node = cityGraph.GetNode(nodeId);
                            
                            // 전략적 마커가 이미 포함되어 있지 않으면 추가
                            if (!node.strategicMarkers.Contains(location.locationType))
                            {
                                node.strategicMarkers.Add(location.locationType);
                            }
                            
                            cityGraph.UpdateNode(nodeId, node);
                        }
                    }
                }
            }

            Debug.Log($"CityGenerator.BuildCityGraph: 전략적 위치 태그 추가 완료. 총 {strategicLocations.Count}개의 전략적 위치");

            // Requirement 18.4: SpatialIndex 구축
            // 도시 경계 계산
            float cityWidth = actualCityWidth * (buildingWidth + buildingSpacing) * unitDistance;
            float cityDepth = actualCityDepth * (buildingDepth + buildingSpacing) * unitDistance;
            
            Bounds cityBounds = new Bounds(
                new Vector3(cityWidth / 2f, 0f, cityDepth / 2f),
                new Vector3(cityWidth, 100f, cityDepth) // Y축은 충분히 크게 설정
            );

            spatialIndex = new SpatialIndex(cityBounds);

            // 모든 노드를 SpatialIndex에 삽입
            List<GraphNode> allNodes = cityGraph.GetAllNodes();
            foreach (GraphNode node in allNodes)
            {
                spatialIndex.Insert(node.nodeId, node.position);
            }

            Debug.Log($"CityGenerator.BuildCityGraph: SpatialIndex 구축 완료. 총 {allNodes.Count}개의 노드 인덱싱");

            // Requirement 18.2: CityDataAPI 초기화 (싱글톤 패턴으로 접근)
            if (CityDataAPI.Instance != null)
            {
                CityDataAPI.Instance.Initialize(cityGraph, spatialIndex, strategicLocations);
                Debug.Log("CityGenerator.BuildCityGraph: CityDataAPI 초기화 완료");
            }
            else
            {
                Debug.LogWarning("CityGenerator.BuildCityGraph: CityDataAPI 인스턴스를 찾을 수 없습니다. 런타임 쿼리 API를 사용하려면 씬에 CityDataAPI 컴포넌트를 추가하세요.");
            }
        }

        #endregion
    }
}
