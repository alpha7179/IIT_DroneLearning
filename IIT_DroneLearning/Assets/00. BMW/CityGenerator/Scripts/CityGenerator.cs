using System.Collections.Generic;
using UnityEngine;

namespace CityGenerator
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
        public float unitDistance = 2.0f;

        [Tooltip("도시 최소 가로 길이 (격자 단위)")]
        [Range(1, 100)]
        public int minWidth = 20;

        [Tooltip("도시 최대 가로 길이 (격자 단위)")]
        [Range(1, 100)]
        public int maxWidth = 20;

        [Tooltip("도시 최소 세로 길이 (격자 단위)")]
        [Range(1, 100)]
        public int minDepth = 20;

        [Tooltip("도시 최대 세로 길이 (격자 단위)")]
        [Range(1, 100)]
        public int maxDepth = 20;

        [Header("Building Settings")]
        [Tooltip("건물 가로 크기 (단위_거리)")]
        [Range(0.5f, 50f)]
        public float buildingWidth = 2.0f;

        [Tooltip("건물 세로 크기 (단위_거리)")]
        [Range(0.5f, 50f)]
        public float buildingDepth = 2.0f;

        [Tooltip("건물 최소 높이 (단위_거리)")]
        [Range(1f, 500f)]
        public float minBuildingHeight = 5.0f;

        [Tooltip("건물 최대 높이 (단위_거리)")]
        [Range(1f, 500f)]
        public float maxBuildingHeight = 10.0f;

        [Tooltip("건물 사이의 간격 (단위_거리)")]
        [Range(0f, 50f)]
        public float buildingSpacing = 0.5f;

        [Tooltip("격자 셀에 건물이 배치될 확률 (0.0 ~ 1.0)")]
        [Range(0f, 1f)]
        public float buildingDensity = 0.8f;

        [Header("Generation Settings")]
        [Tooltip("재현 가능한 도시 생성을 위한 시드값 (-1: 시간 기반 랜덤)")]
        public int randomSeed = -1;

        [Tooltip("기본 건물 머티리얼")]
        public Material defaultBuildingMaterial;

        [Header("Minimap Settings")]
        [Tooltip("생성될 미니맵의 해상도")]
        public MinimapResolution minimapResolution = MinimapResolution.Resolution512;

        [Header("Wall Settings")]
        [Tooltip("도시 경계에 벽 생성 여부 (기본: 비활성)")]
        public bool spawnWalls = false;

        [Tooltip("벽의 높이 (단위_거리). maxBuildingHeight 대비 비율로 자동 스케일됩니다.")]
        [Range(1f, 200f)]
        public float wallHeight = 20f;

        [Tooltip("벽의 두께 (단위_거리)")]
        [Range(0.1f, 20f)]
        public float wallThickness = 1f;

        [Tooltip("벽에 사용할 머티리얼 (미설정 시 기본 흰색)")]
        public Material wallMaterial;

        [Header("Floor Settings")]
        [Tooltip("도시 바닥(Plane) 생성 여부 (기본: 비활성)")]
        public bool spawnFloor = false;

        [Tooltip("바닥에 사용할 머티리얼 (미설정 시 기본 흰색)")]
        public Material floorMaterial;

        [Header("Layout Mode")]
        [Tooltip("도시 레이아웃 방식 선택 (PureGrid=완전격자, Hybrid=격자+오프셋, PureRandom=유기적 도로망)")]
        public CityLayoutMode layoutMode = CityLayoutMode.PureGrid;

        [Tooltip("Hybrid/PureRandom: 건물 위치 랜덤 오프셋 세기 (0=격자 고정, 1=최대 불규칙)")]
        [Range(0f, 1f)]
        public float randomOffsetStrength = 0.6f;

        [Tooltip("Hybrid/PureRandom: 블록 최소 크기 (격자 단위)")]
        [Range(2, 8)]
        public int minBlockSize = 5;

        [Tooltip("Hybrid/PureRandom: 블록 최대 크기 (격자 단위)")]
        [Range(3, 12)]
        public int maxBlockSize = 5;

        [Header("Spawn Configuration")]
        [Tooltip("도시 생성 시 스폰/타겟 포인트를 자동으로 배치합니다.")]
        public bool autoGenerateSpawns = false;

        [Tooltip("스폰 포인트 간 최소 거리 (월드 단위)")]
        [Range(1f, 200f)]
        public float minSpawnSeparation = 10f;

        [Tooltip("드론 스폰/타겟 포인트의 최저 비행 고도 (지면 기준, 월드 단위). DronePhysics.MinAltitude = 0.5f / DroneAgent.SpawnHeight = 8f 기준.")]
        [Range(0.5f, 50f)]
        public float minSpawnHeight = 8f;

        [Tooltip("드론 스폰/타겟 포인트의 최고 비행 고도 (지면 기준, 월드 단위). DronePhysics.MaxAltitude = 50f 기준.")]
        [Range(0.5f, 50f)]
        public float maxSpawnHeight = 50f;

        [Header("Export Settings")]
        [Tooltip("true이면 미니맵 PNG 및 그래프 CSV/JSON 파일 저장을 건너뜁니다.\nCityBatchGenerator가 AllSame 모드로 중복 파일 저장을 막을 때 사용합니다.")]
        public bool suppressFileExport = false;

        #endregion

        #region Internal State

        private GameObject cityGroupRoot;  // 건물·벽·바닥을 묶는 최상위 그룹
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
        private bool[,] roadMask;   // true = 도로 셀 (건물 배치 불가)
        private GameObject wallsRoot;
        private GameObject floorObject;
        private SpawnConfiguration spawnConfiguration;
        private GameObject spawnSystemRoot;

        // EpisodeSpawnCoordinator 재생성 시 Inspector 설정 보존용 캐시
        private bool  _cachedEnablePursuerBoundarySpawn = false;
        private float _cachedPursuerBoundaryRadius      = 30f;

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
            // Play 모드 진입 시 내부 참조(cityGraph 등)가 직렬화되지 않아 null이 됨.
            // 씬에 도시 건물이 존재하면 CityMetadata를 재구축하여 등록한다.
            // cityGroupRoot는 private 비직렬화 필드이므로, 자식 오브젝트로 복원 시도
            if (cityGroupRoot == null)
            {
                // transform 하위에서 CityGroup_ 이름의 자식을 찾아 복원
                foreach (Transform child in transform)
                {
                    if (child.name.StartsWith("CityGroup_"))
                    {
                        cityGroupRoot = child.gameObject;
                        break;
                    }
                }
            }

            // 도시 그룹이 존재하지만 cityGraph가 없으면 도시를 재생성
            if (cityGroupRoot != null && cityGraph == null)
            {
                Debug.Log("CityGenerator.Start: 기존 도시가 감지되었으나 내부 참조가 없습니다. 도시를 재생성합니다.");
                ClearCity();
                GenerateCity();
            }
            else if (cityGroupRoot != null && cityGraph != null)
            {
                // 내부 참조가 모두 유효하면 CityMetadata만 재등록
                CityMetadata metadata = BuildCityMetadata();
                if (CityDataAPI.Instance != null)
                {
                    CityDataAPI.Instance.SetCityMetadata(metadata);
                    Debug.Log("CityGenerator.Start: CityMetadata를 CityDataAPI에 재등록했습니다.");
                }
                if (SpawnCenter.Current != null && SpawnCenter.Current.AutoSyncFromCity)
                {
                    SpawnCenter.Current.SyncFromCityMetadata();
                }
            }

            Debug.Log("CityGenerator.Start: 초기화 완료");
        }

        /// <summary>
        /// Inspector 값이 변경될 때 호출됩니다 (Editor 전용).
        /// spawnWalls / spawnFloor 토글 시 도시가 이미 생성된 경우 즉시 반영합니다.
        /// </summary>
        private void OnValidate()
        {
            // 도시가 아직 생성되지 않았으면 무시
            if (cityGroupRoot == null) return;

            ApplyWallsState();
            ApplyFloorState();
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
                
                // 연결성 보장: Hybrid/PureRandom에서 막힌 길 해소
                EnsureRoadConnectivity();
                result.buildingCount = buildings.Count;

                // 그래프 구축
                BuildCityGraph();
                result.nodeCount = cityGraph.NodeCount;
                result.edgeCount = cityGraph.EdgeCount;
                result.graph = cityGraph;

                // Requirement 2.1, 2.2: CityMetadata 생성 및 CityDataAPI 등록
                CityMetadata metadata = BuildCityMetadata();
                if (CityDataAPI.Instance != null)
                    CityDataAPI.Instance.SetCityMetadata(metadata);

                // Requirement 2.3, 2.4: autoGenerateSpawns 조건부 스폰 구성 생성
                if (autoGenerateSpawns)
                {
                    GenerateSpawnConfiguration();
                    if (CityDataAPI.Instance != null)
                    {
                        CityDataAPI.Instance.SetSpawnConfiguration(spawnConfiguration);
                        Debug.Log("CityGenerator.GenerateCity: SpawnConfiguration CityDataAPI에 등록 완료");
                    }
                }

                // 벽 / 바닥 생성 (OnValidate와 동일한 경로로 상태 적용)
                ApplyWallsState();
                ApplyFloorState();

                // 미니맵 생성 (탑뷰 이미지 자동 생성 및 PNG 저장)
                // suppressFileExport == true 이면 PNG 파일 저장 없이 텍스처만 생성
                // Requirement 2.10: autoGenerateSpawns가 true인 경우에만 고정 스폰 마커 포함
                result.minimap = GenerateMinimap();

                // 그래프 데이터 내보내기 (JSON + CSV) — 도망자 드론 오프라인 지형 분석용
                // suppressFileExport == true 이면 파일 저장을 건너뜁니다
                if (!suppressFileExport)
                    ExportGraphData();

                // Requirement 2.6, 8.2: SpawnSystem 자동 생성
                CreateSpawnSystem();

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
        /// Requirements: 6.3, 8.8, 13.1, 13.2, 13.5
        /// </summary>
        public void ClearCity()
        {
            Debug.Log("CityGenerator.ClearCity: 도시 제거 시작");

            // cityGroupRoot 하나를 제거하면 City_Buildings·CityWalls·CityFloor 전체 삭제
            // ■ DestroyImmediate 사용 이유:
            //   spawnSystemRoot가 cityGroupRoot의 자식이므로, Destroy()로 예약 파괴하면
            //   자식의 싱글톤(SpawnCenter.Current 등)이 프레임 끝까지 살아있어
            //   CreateSpawnSystem()에서 "이미 존재" 판정 → 재생성 안 함 → 프레임 끝 파괴 버그.
            if (cityGroupRoot != null)
            {
                // cityGroupRoot 파괴 시 자식인 spawnSystemRoot(EpisodeSpawnCoordinator 포함)도 함께 파괴됨.
                // 재생성 후 Inspector 설정이 리셋되지 않도록 파괴 전에 설정값을 캐싱한다.
                if (EpisodeSpawnCoordinator.Instance != null)
                {
                    _cachedEnablePursuerBoundarySpawn = EpisodeSpawnCoordinator.Instance.EnablePursuerBoundarySpawn;
                    _cachedPursuerBoundaryRadius      = EpisodeSpawnCoordinator.Instance.PursuerBoundaryRadius;
                }

                DestroyImmediate(cityGroupRoot);
                cityGroupRoot = null;
                Debug.Log("CityGenerator.ClearCity: CityGroup GameObject 제거 완료");
            }
            // 그룹 없이 개별 생성된 경우 안전망
            if (cityRoot != null)
            {
                DestroyImmediate(cityRoot);
            }
            if (wallsRoot != null)
            {
                DestroyImmediate(wallsRoot);
            }
            if (floorObject != null)
            {
                DestroyImmediate(floorObject);
            }
            cityRoot = null;
            wallsRoot = null;
            floorObject = null;

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

            // 격자 및 도로 마스크 초기화
            grid = null;
            roadMask = null;

            // 스폰 구성 초기화
            spawnConfiguration = new SpawnConfiguration();

            // SpawnSystem 제거 (Requirement 8.8)
            // cityGroupRoot 제거 시 자식으로 함께 파괴되지만, 참조를 명시적으로 정리
            // ■ DestroyImmediate 사용 이유:
            //   Destroy()는 프레임 끝까지 오브젝트가 살아있어 싱글톤(SpawnCenter.Current,
            //   EpisodeSpawnCoordinator.Instance)이 null이 아닌 상태로 남는다.
            //   이후 CreateSpawnSystem()에서 "이미 존재" 판정 → 새로 생성 안 함 →
            //   프레임 끝에 실제 파괴 → SpawnCenter 객체가 사라지는 버그 발생.
            //   DestroyImmediate로 즉시 파괴하여 싱글톤 참조를 확실히 정리한다.
            if (spawnSystemRoot != null)
            {
                DestroyImmediate(spawnSystemRoot);
                spawnSystemRoot = null;
            }

            // CityDataAPI 메타데이터 초기화
            if (CityDataAPI.Instance != null)
                CityDataAPI.Instance.SetCityMetadata(null);

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
        /// 자동 생성된 스폰/타겟 포인트 구성을 반환합니다.
        /// </summary>
        public SpawnConfiguration GetSpawnConfiguration() => spawnConfiguration;

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

            // 도시 경계(Bounds) 계산 — 실제 격자 전체 면적 기준 (건물 유무에 관계없이 도로 포함)
            // 격자 originX/Z에 cellW/2 보정이 적용되어 시각적 도시 중심 = transform.position
            float cellW = (buildingWidth + buildingSpacing) * unitDistance;
            float cellD = (buildingDepth + buildingSpacing) * unitDistance;
            Bounds cityBounds = new Bounds(
                new Vector3(transform.position.x, 0f, transform.position.z),
                new Vector3(actualCityWidth * cellW, 1f, actualCityDepth * cellD)
            );

            // MinimapGenerator 생성 (Inspector에서 설정한 해상도 사용)
            int resolutionInt = (int)minimapResolution;
            minimapGenerator = new MinimapGenerator(resolutionInt, cityBounds);

            // 탑뷰 이미지 생성 (건물 높이별 등고선 색상 + 전략적 위치 마커 + 스폰 마커)
            // Requirement 2.10: autoGenerateSpawns가 true인 경우에만 고정 스폰 마커 포함
            // 동적 스폰 마커는 에피소드 시작 시 MinimapRenderer가 갱신한다.
            SpawnConfiguration? spawnMarkerConfig = (autoGenerateSpawns && spawnConfiguration.isValid)
                ? (SpawnConfiguration?)spawnConfiguration
                : null;
            Texture2D minimap = minimapGenerator.GenerateMinimap(
                buildings.ToArray(),
                cityGraph,
                strategicLocations,
                spawnMarkerConfig
            );

            // Assets/CityMaps/ 폴더에 PNG 자동 저장 (파일명에 레이아웃 모드 포함)
            // suppressFileExport == true 이면 저장 건너뜀
            if (!suppressFileExport)
                minimapGenerator.SaveMinimapToPNG(minimap, usedRandomSeed, layoutMode.ToString());

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

            CityGraphExporter.ExportAll(cityGraph, strategicLocations, usedRandomSeed, cityBounds, layoutMode.ToString());
        }

        /// <summary>
        /// spawnWalls 상태에 맞춰 벽을 생성하거나 제거합니다.
        /// </summary>
        private void ApplyWallsState()
        {
            if (spawnWalls)
            {
                if (wallsRoot == null)
                    SpawnWalls();
            }
            else
            {
                if (wallsRoot != null)
                {
                    if (Application.isPlaying) Destroy(wallsRoot);
                    else DestroyImmediate(wallsRoot);
                    wallsRoot = null;
                }
            }
        }

        /// <summary>
        /// spawnFloor 상태에 맞춰 바닥을 생성하거나 제거합니다.
        /// </summary>
        private void ApplyFloorState()
        {
            if (spawnFloor)
            {
                if (floorObject == null)
                    SpawnFloor();
            }
            else
            {
                if (floorObject != null)
                {
                    if (Application.isPlaying) Destroy(floorObject);
                    else DestroyImmediate(floorObject);
                    floorObject = null;
                }
            }
        }

        /// <summary>
        /// 도시 경계 4면에 벽을 생성합니다. 벽 크기는 도시 크기에 비례하며,
        /// 경계 건물과 절대 겹치지 않도록 최대 건물 반폭·오프셋을 계산해 배치합니다.
        /// </summary>
        private void SpawnWalls()
        {
            // 셀 크기 계산
            float cellW = (buildingWidth + buildingSpacing) * unitDistance;
            float cellD = (buildingDepth + buildingSpacing) * unitDistance;
            float cityWorldWidth = actualCityWidth * cellW;
            float cityWorldDepth = actualCityDepth * cellD;

            // ── 건물 최대 반폭 계산 ──────────────────────────────────────────────
            // 건물은 셀 원점(코너)에 중심이 배치됩니다.
            //   x=0: 중심이 -cityWorldWidth/2 → 서쪽으로 buildingHalfW만큼 돌출
            // Hybrid/PureRandom은 추가로 크기 변동(widthVar) + 위치 오프셋이 있습니다.
            float maxSizeVar = layoutMode == CityLayoutMode.PureRandom ? 1.5f
                             : layoutMode == CityLayoutMode.Hybrid     ? 1.25f
                             : 1.0f;

            float maxBuildingHalfW = buildingWidth  * maxSizeVar * unitDistance / 2f;
            float maxBuildingHalfD = buildingDepth  * maxSizeVar * unitDistance / 2f;

            // 위치 오프셋 최댓값 (PureGrid는 0)
            float maxOffX = layoutMode != CityLayoutMode.PureGrid
                          ? cellW * randomOffsetStrength * 0.45f : 0f;
            float maxOffZ = layoutMode != CityLayoutMode.PureGrid
                          ? cellD * randomOffsetStrength * 0.45f : 0f;

            // 벽 내면이 위치해야 할 최소 거리 (도시 중심 기준)
            // 격자 origin에 cellW/2 보정이 적용되어 바깥 셀 중심이 ±(N-1)*cellW/2에 위치하므로
            // 바깥 건물 최대 돌출 = (N-1)*cellW/2 + maxBuildingHalfW + maxOffX
            //                    = cityWorldWidth/2 - cellW/2 + maxBuildingHalfW + maxOffX
            float halfW = cityWorldWidth / 2f - cellW * 0.5f + maxBuildingHalfW + maxOffX;
            float halfD = cityWorldDepth / 2f - cellD * 0.5f + maxBuildingHalfD + maxOffZ;
            // ────────────────────────────────────────────────────────────────────

            float wh = wallHeight;
            float wt = wallThickness;

            Vector3 center = transform.position;
            wallsRoot = new GameObject("CityWalls");
            wallsRoot.transform.position = center;
            if (cityGroupRoot != null)
                wallsRoot.transform.SetParent(cityGroupRoot.transform, true);

            Material mat = wallMaterial;

            // North (+Z): 내면이 +halfD, 중심이 +halfD + wt/2
            CreateWallPanel(wallsRoot.transform, "Wall_North",
                new Vector3(0f, wh / 2f, halfD + wt / 2f),
                new Vector3(halfW * 2f + wt * 2f, wh, wt),
                mat);

            // South (-Z)
            CreateWallPanel(wallsRoot.transform, "Wall_South",
                new Vector3(0f, wh / 2f, -halfD - wt / 2f),
                new Vector3(halfW * 2f + wt * 2f, wh, wt),
                mat);

            // East (+X): 내면이 +halfW, 중심이 +halfW + wt/2
            CreateWallPanel(wallsRoot.transform, "Wall_East",
                new Vector3(halfW + wt / 2f, wh / 2f, 0f),
                new Vector3(wt, wh, halfD * 2f),
                mat);

            // West (-X)
            CreateWallPanel(wallsRoot.transform, "Wall_West",
                new Vector3(-halfW - wt / 2f, wh / 2f, 0f),
                new Vector3(wt, wh, halfD * 2f),
                mat);

            Debug.Log($"CityGenerator.SpawnWalls: 벽 생성 완료 " +
                      $"(도시: {cityWorldWidth:F1}x{cityWorldDepth:F1}, " +
                      $"벽 범위: {halfW * 2f:F1}x{halfD * 2f:F1}, 높이: {wh:F1})");
        }

        /// <summary>
        /// 단일 벽 패널을 Cube Primitive로 생성합니다.
        /// </summary>
        private void CreateWallPanel(Transform parent, string name, Vector3 localPosition, Vector3 size, Material mat)
        {
            GameObject wall = GameObject.CreatePrimitive(PrimitiveType.Cube);
            wall.name = name;
            wall.tag = "Wall";

            int wallLayer = LayerMask.NameToLayer("Wall");
            if (wallLayer >= 0)
                wall.layer = wallLayer;
            else
                Debug.LogWarning("[CityGenerator] 'Wall' 레이어가 없습니다. Project Settings > Tags and Layers에서 추가하세요.");

            wall.transform.SetParent(parent, false);
            wall.transform.localPosition = localPosition;
            wall.transform.localScale = size;

            if (mat != null)
            {
                var renderer = wall.GetComponent<MeshRenderer>();
                if (renderer != null)
                    renderer.sharedMaterial = mat;
            }
        }

        /// <summary>
        /// 도시 바닥을 Plane Primitive로 생성합니다.
        /// 크기는 SpawnWalls와 동일한 halfW/halfD 기준으로 벽 내부 면적과 일치합니다.
        /// </summary>
        private void SpawnFloor()
        {
            // SpawnWalls와 동일한 halfW / halfD 계산
            float cellW = (buildingWidth + buildingSpacing) * unitDistance;
            float cellD = (buildingDepth + buildingSpacing) * unitDistance;
            float cityWorldWidth = actualCityWidth * cellW;
            float cityWorldDepth = actualCityDepth * cellD;

            float maxSizeVar = layoutMode == CityLayoutMode.PureRandom ? 1.5f
                             : layoutMode == CityLayoutMode.Hybrid     ? 1.25f
                             : 1.0f;

            float maxBuildingHalfW = buildingWidth  * maxSizeVar * unitDistance / 2f;
            float maxBuildingHalfD = buildingDepth  * maxSizeVar * unitDistance / 2f;

            float maxOffX = layoutMode != CityLayoutMode.PureGrid
                          ? cellW * randomOffsetStrength * 0.45f : 0f;
            float maxOffZ = layoutMode != CityLayoutMode.PureGrid
                          ? cellD * randomOffsetStrength * 0.45f : 0f;

            // SpawnWalls와 동일한 보정 적용: 바깥 셀 중심이 ±(N-1)*cellW/2에 위치
            float halfW = cityWorldWidth / 2f - cellW * 0.5f + maxBuildingHalfW + maxOffX;
            float halfD = cityWorldDepth / 2f - cellD * 0.5f + maxBuildingHalfD + maxOffZ;

            // Unity Plane 기본 크기는 10x10 유닛 → 벽 내부 전체 넓이(halfW*2 × halfD*2)로 스케일
            float scaleX = halfW * 2f / 10f;
            float scaleZ = halfD * 2f / 10f;

            floorObject = GameObject.CreatePrimitive(PrimitiveType.Plane);
            floorObject.name = "CityFloor";
            floorObject.transform.position = new Vector3(transform.position.x, 0f, transform.position.z);
            floorObject.transform.localScale = new Vector3(scaleX, 1f, scaleZ);

            int groundLayer = LayerMask.NameToLayer("Ground");
            if (groundLayer >= 0)
                floorObject.layer = groundLayer;
            else
                Debug.LogWarning("[CityGenerator] 'Ground' 레이어가 없습니다. Project Settings > Tags and Layers에서 추가하세요.");

            if (cityGroupRoot != null)
                floorObject.transform.SetParent(cityGroupRoot.transform, true);

            if (floorMaterial != null)
            {
                var renderer = floorObject.GetComponent<MeshRenderer>();
                if (renderer != null)
                    renderer.sharedMaterial = floorMaterial;
            }

            Debug.Log($"CityGenerator.SpawnFloor: 바닥 생성 완료 (크기: {halfW * 2f:F1}x{halfD * 2f:F1}, 스케일: {scaleX:F2}x{scaleZ:F2})");
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

            // 각 셀의 월드 좌표 계산 (CityGenerator 오브젝트 위치를 도시 시각적 중심으로 정렬)
            // 셀 worldPosition은 셀 중심 좌표이므로, 시각적 도시 중심이 transform.position과 일치하려면
            // originX = transform.position.x - N*cellW/2 + cellW/2 (= 좌측 가장자리 기준)
            float cellW = (buildingWidth + buildingSpacing) * unitDistance;
            float cellD = (buildingDepth + buildingSpacing) * unitDistance;
            float originX = transform.position.x - (actualCityWidth  * cellW) / 2f + cellW * 0.5f;
            float originZ = transform.position.z - (actualCityDepth * cellD) / 2f + cellD * 0.5f;

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

            // 레이아웃 모드에 따른 도로 마스크 생성 (모드 불문 항상 초기화)
            roadMask = new bool[actualCityWidth, actualCityDepth];
            if (layoutMode == CityLayoutMode.Hybrid)
                GenerateHybridRoadMask();
            else if (layoutMode == CityLayoutMode.PureRandom)
                GeneratePureRandomRoadMask();
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
                // cityGroupRoot가 없으면 생성 (건물·벽·바닥을 묶는 최상위 그룹)
                if (cityGroupRoot == null)
                {
                    cityGroupRoot = new GameObject($"CityGroup_Seed{usedRandomSeed}");
                    cityGroupRoot.transform.SetParent(transform, false);
                    cityGroupRoot.transform.SetPositionAndRotation(transform.position, Quaternion.identity);
                }

                // cityRoot가 없으면 생성 후 그룹 하위로 배치
                if (cityRoot == null)
                {
                    cityRoot = new GameObject("City_Buildings");
                    cityRoot.transform.SetParent(cityGroupRoot.transform, false);
                }

                buildingFactory = new BuildingFactory(defaultBuildingMaterial, cityRoot.transform);
            }

            // 건물 리스트 초기화 (Edit Mode에서는 Awake가 호출되지 않으므로 null 가드 필요)
            if (buildings == null)
                buildings = new List<Building>();
            buildings.Clear();

            // Hybrid/PureRandom은 도로 마스크가 이미 셀을 소모하므로
            // 건물 가능 셀 내 체감 밀도를 보정해 PureGrid와 비슷한 건물 수를 유지한다.
            float effectiveDensity = buildingDensity;
            if (layoutMode == CityLayoutMode.Hybrid)
                effectiveDensity = Mathf.Min(1f, buildingDensity * 1.4f);
            else if (layoutMode == CityLayoutMode.PureRandom)
                effectiveDensity = Mathf.Min(1f, buildingDensity * 1.7f);

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

                    // 도로 셀은 건물 배치 불가 (Hybrid / PureRandom)
                    if (roadMask[x, z]) continue;

                    // Requirement 10.3: 건물_밀도와 동일한 확률로 각 격자 셀에 건물을 배치
                    float randomValue = Random.value;

                    // Requirement 10.5: 건물_밀도가 0.0일 때 건물을 생성하지 않음
                    if (randomValue <= effectiveDensity)
                    {
                        // Requirement 9.4: 건물 높이를 랜덤으로 결정 (minBuildingHeight~maxBuildingHeight)
                        float buildingHeight = Random.Range(minBuildingHeight, maxBuildingHeight);

                        // 레이아웃 모드별 위치 오프셋 (PureGrid는 격자 위치 그대로)
                        Vector3 position = grid[x, z].worldPosition;
                        if (layoutMode != CityLayoutMode.PureGrid)
                        {
                            float cellW = (buildingWidth + buildingSpacing) * unitDistance;
                            float cellD = (buildingDepth + buildingSpacing) * unitDistance;
                            // 도로와 겹치지 않도록 최대 45% 오프셋으로 제한
                            float maxOX = cellW * randomOffsetStrength * 0.45f;
                            float maxOZ = cellD * randomOffsetStrength * 0.45f;
                            position.x += Random.Range(-maxOX, maxOX);
                            position.z += Random.Range(-maxOZ, maxOZ);
                        }

                        // 건물의 중심이 위치에 오도록 Y 좌표 조정
                        position.y = (buildingHeight * unitDistance) / 2f;

                        // 레이아웃 모드별 건물 가로·세로 크기 변동
                        float widthVar = 1f, depthVar = 1f;
                        if (layoutMode == CityLayoutMode.Hybrid)
                        {
                            widthVar = Random.Range(0.75f, 1.25f);
                            depthVar = Random.Range(0.75f, 1.25f);
                        }
                        else if (layoutMode == CityLayoutMode.PureRandom)
                        {
                            widthVar = Random.Range(0.5f, 1.5f);
                            depthVar = Random.Range(0.5f, 1.5f);
                        }

                        // Requirement 8.3, 8.4: 건물 크기 설정
                        Vector3 scale = new Vector3(
                            buildingWidth * widthVar * unitDistance,
                            buildingHeight * unitDistance,
                            buildingDepth * depthVar * unitDistance
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

            Debug.Log($"CityGenerator.PlaceBuildingsOnGrid: 건물 배치 완료. 총 {buildingsPlaced}개의 건물 생성 (설정 밀도: {buildingDensity:P0}, 적용 밀도: {effectiveDensity:P0})");
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

        // ─────────────────────────────────────────────────────────────────
        // 도로 마스크 생성
        // ─────────────────────────────────────────────────────────────────

        /// <summary>
        /// Hybrid 모드용 도로 마스크를 생성합니다.
        /// 불규칙한 간격의 수평·수직 도로선이 교차하는 격자망을 만듭니다.
        /// </summary>
        private void GenerateHybridRoadMask()
        {
            // 수직 도로 (X축 방향 열) — 불규칙 간격
            int cursor = 0;
            while (cursor < actualCityWidth - 1)
            {
                cursor += Random.Range(minBlockSize, maxBlockSize + 1);
                if (cursor < actualCityWidth)
                    for (int z = 0; z < actualCityDepth; z++)
                        roadMask[cursor, z] = true;
            }

            // 수평 도로 (Z축 방향 행) — 불규칙 간격
            cursor = 0;
            while (cursor < actualCityDepth - 1)
            {
                cursor += Random.Range(minBlockSize, maxBlockSize + 1);
                if (cursor < actualCityDepth)
                    for (int x = 0; x < actualCityWidth; x++)
                        roadMask[x, cursor] = true;
            }

            int roadCellCount = CountRoadCells();
            Debug.Log($"[CityGenerator] Hybrid 도로 마스크 완료: {roadCellCount}개 도로 셀");
        }

        /// <summary>
        /// PureRandom 모드용 도로 마스크를 생성합니다.
        /// 완만하게 굴곡진 주 간선도로 + 40% 확률의 지선도로로 유기적 블록을 만듭니다.
        /// 연속성 보장: 완만히 굴곡진 도로는 L자 연결로 4방향 연결성을 유지합니다.
        /// </summary>
        private void GeneratePureRandomRoadMask()
        {
            // ── 수평 주 간선도로 (2~4개) ────────────────────────────────
            int numH = Random.Range(2, 5);
            for (int i = 1; i <= numH; i++)
            {
                int baseZ = Mathf.RoundToInt((float)i / (numH + 1) * actualCityDepth);
                int jitter = Random.Range(-actualCityDepth / 8, actualCityDepth / 8 + 1);
                int startZ = Mathf.Clamp(baseZ + jitter, 1, actualCityDepth - 2);

                // 완만한 굴곡: 4~8셀마다 ±1 이동 (L자 연결로 4-연결성 유지)
                List<int> wanderX = BuildWanderPoints(actualCityWidth);
                int currentZ = startZ;
                for (int x = 0; x < actualCityWidth; x++)
                {
                    if (wanderX.Contains(x) && x < actualCityWidth - 1)
                    {
                        int newZ = Mathf.Clamp(currentZ + Random.Range(-1, 2), 1, actualCityDepth - 2);
                        if (newZ != currentZ)
                        {
                            roadMask[x, currentZ] = true; // 이전 행
                            roadMask[x, newZ]     = true; // 새 행 (L자 연결)
                            currentZ = newZ;
                        }
                    }
                    roadMask[x, currentZ] = true;
                }
            }

            // ── 수직 주 간선도로 (2~4개) ────────────────────────────────
            int numV = Random.Range(2, 5);
            for (int i = 1; i <= numV; i++)
            {
                int baseX = Mathf.RoundToInt((float)i / (numV + 1) * actualCityWidth);
                int jitter = Random.Range(-actualCityWidth / 8, actualCityWidth / 8 + 1);
                int startX = Mathf.Clamp(baseX + jitter, 1, actualCityWidth - 2);

                List<int> wanderZ = BuildWanderPoints(actualCityDepth);
                int currentX = startX;
                for (int z = 0; z < actualCityDepth; z++)
                {
                    if (wanderZ.Contains(z) && z < actualCityDepth - 1)
                    {
                        int newX = Mathf.Clamp(currentX + Random.Range(-1, 2), 1, actualCityWidth - 2);
                        if (newX != currentX)
                        {
                            roadMask[currentX, z] = true;
                            roadMask[newX, z]     = true;
                            currentX = newX;
                        }
                    }
                    roadMask[currentX, z] = true;
                }
            }

            // ── 지선도로 (40% 확률) ─────────────────────────────────────
            int sideInterval = Random.Range(minBlockSize, maxBlockSize + 1);
            for (int x = sideInterval; x < actualCityWidth - 1; x += sideInterval + Random.Range(0, 3))
                if (Random.value < 0.4f)
                    for (int z = 0; z < actualCityDepth; z++)
                        roadMask[x, z] = true;

            for (int z = sideInterval; z < actualCityDepth - 1; z += sideInterval + Random.Range(0, 3))
                if (Random.value < 0.4f)
                    for (int x = 0; x < actualCityWidth; x++)
                        roadMask[x, z] = true;

            int roadCellCount = CountRoadCells();
            Debug.Log($"[CityGenerator] PureRandom 도로 마스크 완료: {roadCellCount}개 도로 셀");
        }

        /// <summary>
        /// 도로 굴곡용 랜덤 이동 지점 목록을 생성합니다.
        /// 4~8셀 간격으로 분포하는 인덱스 집합을 반환합니다.
        /// </summary>
        private List<int> BuildWanderPoints(int length)
        {
            var points = new List<int>();
            int p = Random.Range(4, 8);
            while (p < length - 1)
            {
                points.Add(p);
                p += Random.Range(4, 8);
            }
            return points;
        }

        /// <summary>도로 셀 수를 반환합니다.</summary>
        private int CountRoadCells()
        {
            int count = 0;
            for (int x = 0; x < actualCityWidth; x++)
                for (int z = 0; z < actualCityDepth; z++)
                    if (roadMask[x, z]) count++;
            return count;
        }

        // ─────────────────────────────────────────────────────────────────
        // 연결성 보장
        // ─────────────────────────────────────────────────────────────────

        /// <summary>
        /// BFS로 빈 셀(건물 없는 셀)의 연결 성분을 검사하고,
        /// 고립된 영역을 해소할 때까지 경계에 있는 최소 높이 건물을 제거합니다.
        /// PureGrid는 격자 구조가 연결성을 보장하므로 스킵합니다.
        /// </summary>
        private void EnsureRoadConnectivity()
        {
            if (layoutMode == CityLayoutMode.PureGrid) return;

            int maxIterations = Mathf.Min(buildings.Count / 5 + 1, 50);
            int removedCount  = 0;

            for (int iter = 0; iter < maxIterations; iter++)
            {
                // ── Step 1: BFS — 빈 셀 연결 성분 계산 ─────────────────
                // label: -1=건물, 0=미방문 빈셀, >0=성분 ID
                int[,] label = new int[actualCityWidth, actualCityDepth];
                for (int x = 0; x < actualCityWidth; x++)
                    for (int z = 0; z < actualCityDepth; z++)
                        label[x, z] = grid[x, z].hasBuilding ? -1 : 0;

                var componentCells = new Dictionary<int, List<(int x, int z)>>();
                int compId = 0;

                for (int sx = 0; sx < actualCityWidth; sx++)
                {
                    for (int sz = 0; sz < actualCityDepth; sz++)
                    {
                        if (label[sx, sz] != 0) continue;

                        compId++;
                        var cells = new List<(int, int)>();
                        var queue = new Queue<(int, int)>();
                        queue.Enqueue((sx, sz));
                        label[sx, sz] = compId;

                        while (queue.Count > 0)
                        {
                            var (cx, cz) = queue.Dequeue();
                            cells.Add((cx, cz));

                            // 4방향 연결 탐색
                            (int dx, int dz)[] dirs = { (-1,0),(1,0),(0,-1),(0,1) };
                            foreach (var (dx, dz) in dirs)
                            {
                                int nx = cx + dx, nz = cz + dz;
                                if (nx < 0 || nx >= actualCityWidth ||
                                    nz < 0 || nz >= actualCityDepth) continue;
                                if (label[nx, nz] != 0) continue;
                                label[nx, nz] = compId;
                                queue.Enqueue((nx, nz));
                            }
                        }
                        componentCells[compId] = cells;
                    }
                }

                if (componentCells.Count <= 1) break; // 모두 연결됨

                // ── Step 2: 가장 큰 성분(메인 공간) 찾기 ───────────────
                int mainId = -1, maxSize = 0;
                foreach (var kvp in componentCells)
                    if (kvp.Value.Count > maxSize) { maxSize = kvp.Value.Count; mainId = kvp.Key; }

                // ── Step 3: 고립 성분 경계의 최소 높이 건물 탐색 ────────
                (int bx, int bz) best = (-1, -1);
                float minH = float.MaxValue;

                foreach (var kvp in componentCells)
                {
                    if (kvp.Key == mainId) continue;
                    foreach (var (cx, cz) in kvp.Value)
                    {
                        (int dx, int dz)[] dirs = { (-1,0),(1,0),(0,-1),(0,1) };
                        foreach (var (dx, dz) in dirs)
                        {
                            int nx = cx + dx, nz = cz + dz;
                            if (nx < 0 || nx >= actualCityWidth ||
                                nz < 0 || nz >= actualCityDepth) continue;
                            if (label[nx, nz] != -1) continue; // 건물 셀만 대상
                            float h = grid[nx, nz].buildingHeight;
                            if (h < minH) { minH = h; best = (nx, nz); }
                        }
                    }
                }

                if (best.bx == -1) break; // 제거 가능한 건물 없음

                // ── Step 4: 해당 건물 제거 ──────────────────────────────
                RemoveBuildingAtCell(best.bx, best.bz);
                removedCount++;
            }

            if (removedCount > 0)
                Debug.Log($"[CityGenerator] 연결성 보장 완료: {removedCount}개 건물 제거");
        }

        /// <summary>
        /// 지정한 격자 셀의 건물을 제거하고 그리드 상태를 갱신합니다.
        /// </summary>
        private void RemoveBuildingAtCell(int x, int z)
        {
            Building target = null;
            foreach (var b in buildings)
            {
                if (b.gridCell.x == x && b.gridCell.z == z)
                {
                    target = b;
                    break;
                }
            }

            if (target != null)
            {
                if (Application.isPlaying)
                    Object.Destroy(target.gameObject);
                else
                    Object.DestroyImmediate(target.gameObject);
                buildings.Remove(target);
            }

            GridCell cell = grid[x, z];
            cell.hasBuilding  = false;
            cell.buildingHeight = 0f;
            grid[x, z] = cell;
        }

        /// <summary>
        /// 내부 상태에서 CityMetadata 객체를 조립합니다.
        /// Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.1
        /// </summary>
        /// <returns>조립된 CityMetadata 객체</returns>
        private CityMetadata BuildCityMetadata()
        {
            CityMetadata metadata = new CityMetadata();

            // Requirement 1.2: 격자 크기
            metadata.actualCityWidth = actualCityWidth;
            metadata.actualCityDepth = actualCityDepth;

            // Requirement 1.3: 건물 높이 범위
            metadata.minBuildingHeight = minBuildingHeight;
            metadata.maxBuildingHeight = maxBuildingHeight;

            // Requirement 1.4: 건물 목록 참조
            metadata.buildings = buildings;

            // Requirement 1.5: 도로 그래프 참조
            metadata.cityGraph = cityGraph;

            // Requirement 1.6: 전략적 위치 목록 참조
            metadata.strategicLocations = strategicLocations;

            // Requirement 1.8: 사용된 랜덤 시드
            metadata.usedRandomSeed = usedRandomSeed;

            // Requirement 1.9: 레이아웃 모드
            metadata.layoutMode = layoutMode;

            // Requirement 1.1: 도시 경계(Bounds) 계산 — GenerateMinimap()과 동일한 로직
            float cellW = (buildingWidth + buildingSpacing) * unitDistance;
            float cellD = (buildingDepth + buildingSpacing) * unitDistance;
            metadata.cityBounds = new Bounds(
                new Vector3(transform.position.x, 0f, transform.position.z),
                new Vector3(actualCityWidth * cellW, 1f, actualCityDepth * cellD)
            );

            // Requirement 1.7: 유효 스폰 후보 노드 필터링 (캐싱)
            // GenerateSpawnConfiguration()과 동일한 필터링 로직:
            // OpenSpace/Intersection 타입, 엣지 존재, 건물 Bounds 내부 아닌 노드
            metadata.validSpawnCandidates = new List<GraphNode>();
            if (cityGraph != null)
            {
                foreach (GraphNode node in cityGraph.GetAllNodes())
                {
                    if (node.nodeType != NodeType.OpenSpace && node.nodeType != NodeType.Intersection)
                        continue;

                    List<GraphEdge> edges = cityGraph.GetEdges(node.nodeId);
                    if (edges == null || edges.Count == 0)
                        continue;

                    bool insideBuilding = false;
                    if (buildings != null)
                    {
                        foreach (Building b in buildings)
                        {
                            if (b.bounds.Contains(node.position))
                            {
                                insideBuilding = true;
                                break;
                            }
                        }
                    }
                    if (!insideBuilding)
                        metadata.validSpawnCandidates.Add(node);
                }
            }

            Debug.Log($"CityGenerator.BuildCityMetadata: CityMetadata 생성 완료. " +
                      $"유효 스폰 후보: {metadata.validSpawnCandidates.Count}개, " +
                      $"건물: {metadata.buildings?.Count ?? 0}개, " +
                      $"시드: {metadata.usedRandomSeed}");

            return metadata;
        }

        /// <summary>
        /// SpawnSystem GameObject를 생성하고 SpawnCenter + EpisodeSpawnCoordinator를 부착합니다.
        /// Requirements: 2.6, 2.7, 2.8, 2.9
        /// </summary>
        private void CreateSpawnSystem()
        {
            // 씬에 이미 EpisodeSpawnCoordinator가 존재하면 중복 생성 건너뜀
            // (수동 배치된 SpawnCenter/EpisodeSpawnCoordinator 우선)
            if (EpisodeSpawnCoordinator.Instance != null)
            {
                Debug.Log("[CityGenerator] 씬에 EpisodeSpawnCoordinator가 이미 존재합니다. SpawnSystem 자동 생성을 건너뜁니다. " +
                          "기존 코디네이터에 CityMetadata를 동기화합니다.");

                // 기존 SpawnCenter가 있으면 도시 크기 동기화
                if (SpawnCenter.Current != null && SpawnCenter.Current.AutoSyncFromCity)
                {
                    SpawnCenter.Current.SyncFromCityMetadata();
                }
                return;
            }

            // "SpawnSystem" 빈 GameObject 생성
            spawnSystemRoot = new GameObject("SpawnSystem");

            // cityGroupRoot 하위에 배치 (ClearCity() 시 함께 제거)
            spawnSystemRoot.transform.SetParent(cityGroupRoot.transform, false);

            // 도시 중앙(cityGroupRoot 위치)에 배치
            spawnSystemRoot.transform.position = cityGroupRoot.transform.position;

            // SpawnCenter 컴포넌트 부착 + AutoSyncFromCity 활성화
            SpawnCenter spawnCenter = spawnSystemRoot.AddComponent<SpawnCenter>();
            spawnCenter.AutoSyncFromCity = true;

            // 도시 크기를 공용 범위에 즉시 반영 (ComputeSpawn() 호출 전에도 올바른 범위 보장)
            spawnCenter.SyncFromCityMetadata();

            // EpisodeSpawnCoordinator 컴포넌트 부착 + Strategy = CityMetadata
            EpisodeSpawnCoordinator coordinator = spawnSystemRoot.AddComponent<EpisodeSpawnCoordinator>();
            coordinator.Strategy = EpisodeSpawnCoordinator.SpawnStrategy.CityMetadata;

            // ClearCity()에서 캐싱해 둔 Inspector 설정을 복원 (도시 재생성 시 설정 리셋 방지)
            coordinator.EnablePursuerBoundarySpawn = _cachedEnablePursuerBoundarySpawn;
            coordinator.PursuerBoundaryRadius      = _cachedPursuerBoundaryRadius;

            Debug.Log("CityGenerator.CreateSpawnSystem: SpawnSystem 자동 생성 완료 " +
                      $"(위치: {spawnSystemRoot.transform.position}, " +
                      $"AutoSyncFromCity: true, Strategy: CityMetadata)");
        }

        /// <summary>
        /// 도망 드론 스폰, 추적 드론 스폰, 타겟 포인트를 자동으로 결정합니다.
        /// OpenSpace/Intersection 노드 중 서로 minSpawnSeparation 이상 떨어진 3개를 선택합니다.
        /// </summary>
        private void GenerateSpawnConfiguration()
        {
            if (cityGraph == null)
            {
                Debug.LogWarning("CityGenerator.GenerateSpawnConfiguration: cityGraph가 null입니다.");
                return;
            }

            // 전체 유효 후보 수집 (OpenSpace/Intersection, 엣지 있음, 건물 내부 아님)
            List<GraphNode> allCandidates = new List<GraphNode>();
            foreach (GraphNode node in cityGraph.GetAllNodes())
            {
                if (node.nodeType != NodeType.OpenSpace && node.nodeType != NodeType.Intersection)
                    continue;

                // 그래프에서 고립된 노드 제외 (엣지가 없으면 드론이 진입/이탈 불가)
                List<GraphEdge> edges = cityGraph.GetEdges(node.nodeId);
                if (edges == null || edges.Count == 0)
                    continue;

                // 건물 내부인 노드 제외
                bool insideBuilding = false;
                foreach (Building b in buildings)
                {
                    if (b.bounds.Contains(node.position))
                    {
                        insideBuilding = true;
                        break;
                    }
                }
                if (!insideBuilding)
                    allCandidates.Add(node);
            }

            if (allCandidates.Count < 3)
            {
                Debug.LogWarning($"CityGenerator.GenerateSpawnConfiguration: 유효한 후보 노드가 {allCandidates.Count}개뿐입니다. 스폰 포인트 생성을 건너뜁니다.");
                return;
            }

            // ── 최외각 경계 후보 추출 후 랜덤 배치 ──────────────────────────────────
            // 1) 유효 후보 전체의 바운딩 박스(minX/maxX/minZ/maxZ) 계산
            // 2) 각 노드의 "가장 가까운 경계까지의 거리" 산출
            // 3) 거리 오름차순 정렬 → 상위 20%(최소 10개) = 경계 후보 풀
            // 4) 풀을 시드 기반으로 셔플 → 앞 2개를 evader/pursuer로 선택
            float bMinX = float.MaxValue, bMaxX = float.MinValue;
            float bMinZ = float.MaxValue, bMaxZ = float.MinValue;
            foreach (GraphNode n in allCandidates)
            {
                if (n.position.x < bMinX) bMinX = n.position.x;
                if (n.position.x > bMaxX) bMaxX = n.position.x;
                if (n.position.z < bMinZ) bMinZ = n.position.z;
                if (n.position.z > bMaxZ) bMaxZ = n.position.z;
            }

            // 각 노드의 경계까지 최단 거리 (4방향 중 가장 가까운 쪽)
            List<(GraphNode node, float dist)> byEdgeDist = new List<(GraphNode, float)>(allCandidates.Count);
            foreach (GraphNode n in allCandidates)
            {
                float d = Mathf.Min(
                    Mathf.Min(n.position.x - bMinX, bMaxX - n.position.x),
                    Mathf.Min(n.position.z - bMinZ, bMaxZ - n.position.z)
                );
                byEdgeDist.Add((n, d));
            }
            byEdgeDist.Sort((a, b) => a.dist.CompareTo(b.dist));

            // 상위 20%, 최소 10개를 경계 후보 풀로 구성
            int perimCount = Mathf.Clamp(allCandidates.Count / 5, 10, allCandidates.Count);
            List<GraphNode> perimCandidates = new List<GraphNode>(perimCount);
            for (int i = 0; i < perimCount; i++)
                perimCandidates.Add(byEdgeDist[i].node);

            // 재현 가능한 셔플 (usedRandomSeed 기반)
            System.Random rng = new System.Random(usedRandomSeed);
            for (int i = perimCandidates.Count - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                GraphNode tmp = perimCandidates[i];
                perimCandidates[i] = perimCandidates[j];
                perimCandidates[j] = tmp;
            }

            // 셔플된 경계 후보 앞 2개 = evader / pursuer
            GraphNode evaderNode  = perimCandidates[0];
            GraphNode pursuerNode = perimCandidates[1];

            Debug.Log($"CityGenerator.GenerateSpawnConfiguration: 경계 후보 {perimCount}개 중 랜덤 배치 " +
                      $"(경계 dist: Evader={byEdgeDist.Find(x => x.node.nodeId == evaderNode.nodeId).dist:F1}, " +
                      $"Pursuer={byEdgeDist.Find(x => x.node.nodeId == pursuerNode.nodeId).dist:F1})");

            // 타겟 선택용 allCandidates 셔플
            List<GraphNode> shuffled = new List<GraphNode>(allCandidates);
            for (int i = shuffled.Count - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                GraphNode tmp = shuffled[i];
                shuffled[i] = shuffled[j];
                shuffled[j] = tmp;
            }

            // 타겟: evader/pursuer와 minSpawnSeparation 이상 떨어진 노드를 셔플된 목록에서 선택
            // 실패 시 50%씩 완화하여 최대 3회 재시도
            GraphNode targetNode = default;
            bool targetFound = false;
            float sep = minSpawnSeparation;

            for (int attempt = 0; attempt <= 3 && !targetFound; attempt++)
            {
                if (attempt > 0)
                {
                    sep *= 0.5f;
                    Debug.LogWarning($"CityGenerator.GenerateSpawnConfiguration: 타겟 분리 거리를 {sep:F1}로 완화합니다. (시도 {attempt}/3)");
                }

                foreach (GraphNode n in shuffled)
                {
                    if (n.nodeId == evaderNode.nodeId || n.nodeId == pursuerNode.nodeId) continue;
                    float de = Vector3.Distance(n.position, evaderNode.position);
                    float dp = Vector3.Distance(n.position, pursuerNode.position);
                    if (de >= sep && dp >= sep)
                    {
                        targetNode  = n;
                        targetFound = true;
                        break;
                    }
                }
            }

            if (!targetFound)
            {
                // 마지막 폴백: 다른 노드 중 첫 번째
                Debug.LogWarning("CityGenerator.GenerateSpawnConfiguration: 타겟 분리 조건 미충족. 강제 선택합니다.");
                foreach (GraphNode n in shuffled)
                {
                    if (n.nodeId != evaderNode.nodeId && n.nodeId != pursuerNode.nodeId)
                    {
                        targetNode = n;
                        break;
                    }
                }
            }

            float achieved = Mathf.Min(
                Vector3.Distance(evaderNode.position, pursuerNode.position),
                Mathf.Min(
                    Vector3.Distance(evaderNode.position, targetNode.position),
                    Vector3.Distance(pursuerNode.position, targetNode.position)
                )
            );

            float heightRange = Mathf.Max(0f, maxSpawnHeight - minSpawnHeight);
            float evaderY  = evaderNode.elevation  + minSpawnHeight + (float)(rng.NextDouble() * heightRange);
            float pursuerY = pursuerNode.elevation + minSpawnHeight + (float)(rng.NextDouble() * heightRange);
            float targetY  = targetNode.elevation  + minSpawnHeight + (float)(rng.NextDouble() * heightRange);

            spawnConfiguration = new SpawnConfiguration
            {
                evaderSpawnPosition  = new Vector3(evaderNode.position.x,  evaderY,  evaderNode.position.z),
                pursuerSpawnPosition = new Vector3(pursuerNode.position.x, pursuerY, pursuerNode.position.z),
                targetPosition       = new Vector3(targetNode.position.x,  targetY,  targetNode.position.z),
                evaderSpawnNodeId    = evaderNode.nodeId,
                pursuerSpawnNodeId   = pursuerNode.nodeId,
                targetNodeId         = targetNode.nodeId,
                achievedMinSeparation = achieved,
                isValid              = true
            };

            Debug.Log($"CityGenerator.GenerateSpawnConfiguration: 스폰 포인트 생성 완료 " +
                      $"(Evader: {spawnConfiguration.evaderSpawnPosition}, " +
                      $"Pursuer: {spawnConfiguration.pursuerSpawnPosition}, " +
                      $"Target: {spawnConfiguration.targetPosition}, " +
                      $"최소분리거리: {achieved:F1})");
        }

        #endregion
    }
}
