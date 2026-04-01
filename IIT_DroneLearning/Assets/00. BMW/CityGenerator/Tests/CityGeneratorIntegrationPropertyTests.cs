using System.Reflection;
using NUnit.Framework;
using UnityEngine;

namespace CityGenerator
{
    /// <summary>
    /// CityGenerator 통합 속성 기반 테스트.
    ///
    /// 실제 CityGenerator.GenerateCity()를 호출하여 CityMetadata 등록,
    /// autoGenerateSpawns 플래그 동작, SpawnSystem 자동 생성, ClearCity() 정리를 검증한다.
    ///
    /// GenerateCity()는 비용이 크므로 반복 횟수를 10~20회로 제한하고,
    /// 작은 격자(3x3~5x5)를 사용하여 테스트 속도를 유지한다.
    ///
    /// Feature: city-spawn-refactor
    /// </summary>
    public class CityGeneratorIntegrationPropertyTests
    {
        private const int Iterations = 15;

        private System.Random _rng;

        // 씬 오브젝트
        private GameObject _generatorObject;
        private CityGenerator _generator;
        private GameObject _apiGameObject;
        private CityDataAPI _api;

        [SetUp]
        public void SetUp()
        {
            _rng = new System.Random(42);

            // CityDataAPI 싱글톤 생성
            _apiGameObject = new GameObject("TestCityDataAPI");
            _api = _apiGameObject.AddComponent<CityDataAPI>();

            // CityGenerator 생성
            _generatorObject = new GameObject("TestCityGenerator");
            _generator = _generatorObject.AddComponent<CityGenerator>();

            // 빠른 테스트를 위한 기본 설정
            _generator.suppressFileExport = true;
        }

        [TearDown]
        public void TearDown()
        {
            // ClearCity로 생성된 오브젝트 정리
            if (_generator != null)
                _generator.ClearCity();

            if (_generatorObject != null)
                Object.DestroyImmediate(_generatorObject);
            if (_apiGameObject != null)
                Object.DestroyImmediate(_apiGameObject);

            // SpawnSystem이 남아있을 수 있으므로 추가 정리
            var leftover = GameObject.Find("SpawnSystem");
            if (leftover != null)
                Object.DestroyImmediate(leftover);
        }

        // ────────────────────────────────────────────
        // 헬퍼
        // ────────────────────────────────────────────

        private float Rand(float min, float max) =>
            (float)(_rng.NextDouble() * (max - min) + min);

        private int RandInt(int min, int max) =>
            _rng.Next(min, max + 1);

        /// <summary>
        /// CityGenerator에 작은 도시 파라미터를 랜덤으로 설정한다.
        /// 격자 크기 3~5, 건물 밀도 0.3~0.8로 제한하여 빠른 생성을 보장한다.
        /// </summary>
        private void SetSmallCityParameters()
        {
            int size = RandInt(3, 5);
            _generator.minWidth = size;
            _generator.maxWidth = size;
            _generator.minDepth = size;
            _generator.maxDepth = size;

            _generator.unitDistance = Rand(1f, 3f);
            _generator.buildingWidth = Rand(1f, 3f);
            _generator.buildingDepth = Rand(1f, 3f);
            _generator.buildingSpacing = Rand(0.2f, 1f);
            _generator.buildingDensity = Rand(0.3f, 0.8f);

            float minH = Rand(2f, 10f);
            _generator.minBuildingHeight = minH;
            _generator.maxBuildingHeight = Rand(minH, minH + 20f);

            _generator.randomSeed = RandInt(1, 99999);
            _generator.layoutMode = (CityLayoutMode)RandInt(0, 2);

            _generator.spawnWalls = false;
            _generator.spawnFloor = false;
            _generator.suppressFileExport = true;
        }

        /// <summary>
        /// Reflection을 통해 private 필드 값을 읽는다.
        /// </summary>
        private T GetPrivateField<T>(object target, string fieldName)
        {
            var field = target.GetType().GetField(fieldName,
                BindingFlags.NonPublic | BindingFlags.Instance);
            Assert.IsNotNull(field, $"필드 '{fieldName}'을 찾을 수 없습니다.");
            return (T)field.GetValue(target);
        }

        // ────────────────────────────────────────────
        // Feature: city-spawn-refactor, Property 3: GenerateCity() 후 CityMetadata 등록
        // **Validates: Requirements 2.1, 2.2, 8.1**
        //
        // For any 유효한 도시 파라미터로 GenerateCity()를 실행한 후,
        // CityDataAPI.Instance.HasCityMetadata()가 true를 반환하고,
        // CityDataAPI.Instance.GetCityMetadata()가 null이 아닌 유효한
        // CityMetadata 객체를 반환해야 한다.
        // ────────────────────────────────────────────

        [Test, Category("city-spawn-refactor")]
        public void Property3_GenerateCity_RegistersCityMetadata()
        {
            for (int i = 0; i < Iterations; i++)
            {
                // 이전 도시 정리
                _generator.ClearCity();

                // 랜덤 파라미터 설정
                SetSmallCityParameters();

                // GenerateCity 실행
                CityGenerationResult result = _generator.GenerateCity();

                Assert.IsTrue(result.success,
                    $"[P3 iter={i}] GenerateCity()가 성공해야 합니다. 오류: {result.errorMessage}");

                // Req 2.1, 2.2: CityDataAPI에 CityMetadata가 등록되어야 한다
                Assert.IsTrue(CityDataAPI.Instance.HasCityMetadata(),
                    $"[P3 iter={i}] GenerateCity() 후 HasCityMetadata()가 true여야 합니다.");

                CityMetadata metadata = CityDataAPI.Instance.GetCityMetadata();
                Assert.IsNotNull(metadata,
                    $"[P3 iter={i}] GenerateCity() 후 GetCityMetadata()가 null이 아니어야 합니다.");

                // Req 8.1: CityMetadata 필드 유효성 검증
                Assert.Greater(metadata.cityBounds.size.x, 0f,
                    $"[P3 iter={i}] cityBounds.size.x는 0보다 커야 합니다.");
                Assert.Greater(metadata.cityBounds.size.z, 0f,
                    $"[P3 iter={i}] cityBounds.size.z는 0보다 커야 합니다.");
                Assert.Greater(metadata.actualCityWidth, 0,
                    $"[P3 iter={i}] actualCityWidth는 양수여야 합니다.");
                Assert.Greater(metadata.actualCityDepth, 0,
                    $"[P3 iter={i}] actualCityDepth는 양수여야 합니다.");
                Assert.LessOrEqual(metadata.minBuildingHeight, metadata.maxBuildingHeight,
                    $"[P3 iter={i}] minBuildingHeight <= maxBuildingHeight이어야 합니다.");
                Assert.IsNotNull(metadata.buildings,
                    $"[P3 iter={i}] buildings는 null이 아니어야 합니다.");
                Assert.IsNotNull(metadata.cityGraph,
                    $"[P3 iter={i}] cityGraph는 null이 아니어야 합니다.");
                Assert.IsNotNull(metadata.strategicLocations,
                    $"[P3 iter={i}] strategicLocations는 null이 아니어야 합니다.");
                Assert.IsNotNull(metadata.validSpawnCandidates,
                    $"[P3 iter={i}] validSpawnCandidates는 null이 아니어야 합니다.");
            }
        }

        // ────────────────────────────────────────────
        // Feature: city-spawn-refactor, Property 4: autoGenerateSpawns 플래그에 따른 SpawnConfiguration 등록
        // **Validates: Requirements 2.3, 2.4, 7.6**
        //
        // For any 유효한 도시 파라미터에 대해,
        // autoGenerateSpawns가 true이면 GenerateCity() 후
        // CityDataAPI.Instance.HasSpawnConfiguration()이 true를 반환하고,
        // autoGenerateSpawns가 false이면 HasSpawnConfiguration()이 false를 반환해야 한다.
        // ────────────────────────────────────────────

        [Test, Category("city-spawn-refactor")]
        public void Property4_AutoGenerateSpawns_FlagBehavior()
        {
            for (int i = 0; i < Iterations; i++)
            {
                // 이전 도시 정리
                _generator.ClearCity();

                // 랜덤 파라미터 설정
                SetSmallCityParameters();

                // autoGenerateSpawns를 랜덤으로 true/false 설정
                bool autoSpawns = _rng.NextDouble() < 0.5;
                _generator.autoGenerateSpawns = autoSpawns;

                // GenerateCity 실행
                CityGenerationResult result = _generator.GenerateCity();

                Assert.IsTrue(result.success,
                    $"[P4 iter={i}] GenerateCity()가 성공해야 합니다. 오류: {result.errorMessage}");

                if (autoSpawns)
                {
                    // Req 2.3: autoGenerateSpawns == true이면 SpawnConfiguration 등록
                    Assert.IsTrue(CityDataAPI.Instance.HasSpawnConfiguration(),
                        $"[P4 iter={i}] autoGenerateSpawns=true일 때 " +
                        $"HasSpawnConfiguration()이 true여야 합니다.");
                }
                else
                {
                    // Req 2.4: autoGenerateSpawns == false이면 SpawnConfiguration 미등록
                    Assert.IsFalse(CityDataAPI.Instance.HasSpawnConfiguration(),
                        $"[P4 iter={i}] autoGenerateSpawns=false일 때 " +
                        $"HasSpawnConfiguration()이 false여야 합니다.");
                }
            }
        }

        // ────────────────────────────────────────────
        // Feature: city-spawn-refactor, Property 5: SpawnSystem 자동 생성 및 설정
        // **Validates: Requirements 2.6, 2.7, 2.8, 2.9, 8.2, 8.3**
        //
        // For any 유효한 도시 파라미터로 GenerateCity()를 실행한 후,
        // cityGroupRoot 하위에 "SpawnSystem" 이름의 GameObject가 존재하고,
        // 해당 GameObject에 SpawnCenter 컴포넌트와 EpisodeSpawnCoordinator 컴포넌트가
        // 부착되어 있고, SpawnCenter.AutoSyncFromCity가 true이고,
        // EpisodeSpawnCoordinator의 SpawnStrategy가 CityMetadata로 설정되어 있어야 한다.
        // ────────────────────────────────────────────

        [Test, Category("city-spawn-refactor")]
        public void Property5_SpawnSystem_AutoCreationAndConfiguration()
        {
            for (int i = 0; i < Iterations; i++)
            {
                // 이전 도시 정리
                _generator.ClearCity();

                // 랜덤 파라미터 설정
                SetSmallCityParameters();

                // GenerateCity 실행
                CityGenerationResult result = _generator.GenerateCity();

                Assert.IsTrue(result.success,
                    $"[P5 iter={i}] GenerateCity()가 성공해야 합니다. 오류: {result.errorMessage}");

                // Req 2.9: cityGroupRoot 하위에 SpawnSystem이 존재해야 한다
                GameObject cityGroupRoot = GetPrivateField<GameObject>(_generator, "cityGroupRoot");
                Assert.IsNotNull(cityGroupRoot,
                    $"[P5 iter={i}] cityGroupRoot가 null이 아니어야 합니다.");

                // cityGroupRoot 하위에서 "SpawnSystem" 찾기
                Transform spawnSystemTransform = cityGroupRoot.transform.Find("SpawnSystem");
                Assert.IsNotNull(spawnSystemTransform,
                    $"[P5 iter={i}] cityGroupRoot 하위에 'SpawnSystem' GameObject가 존재해야 합니다.");

                GameObject spawnSystem = spawnSystemTransform.gameObject;

                // Req 2.6, 8.2: SpawnCenter 컴포넌트 부착 확인
                SpawnCenter spawnCenter = spawnSystem.GetComponent<SpawnCenter>();
                Assert.IsNotNull(spawnCenter,
                    $"[P5 iter={i}] SpawnSystem에 SpawnCenter 컴포넌트가 부착되어 있어야 합니다.");

                // Req 2.7: SpawnCenter.AutoSyncFromCity == true
                Assert.IsTrue(spawnCenter.AutoSyncFromCity,
                    $"[P5 iter={i}] SpawnCenter.AutoSyncFromCity가 true여야 합니다.");

                // Req 2.8, 8.2: EpisodeSpawnCoordinator 컴포넌트 부착 확인
                EpisodeSpawnCoordinator coordinator = spawnSystem.GetComponent<EpisodeSpawnCoordinator>();
                Assert.IsNotNull(coordinator,
                    $"[P5 iter={i}] SpawnSystem에 EpisodeSpawnCoordinator 컴포넌트가 부착되어 있어야 합니다.");

                // Req 8.3: SpawnStrategy == CityMetadata
                Assert.AreEqual(
                    EpisodeSpawnCoordinator.SpawnStrategy.CityMetadata,
                    coordinator.Strategy,
                    $"[P5 iter={i}] EpisodeSpawnCoordinator.Strategy가 CityMetadata여야 합니다. " +
                    $"실제: {coordinator.Strategy}");
            }
        }

        // ────────────────────────────────────────────
        // Feature: city-spawn-refactor, Property 16: ClearCity() 시 SpawnSystem 제거
        // **Validates: Requirements 8.8**
        //
        // For any GenerateCity() 후 ClearCity()를 호출하면,
        // "SpawnSystem" 이름의 GameObject가 씬에 존재하지 않아야 하고,
        // CityDataAPI.HasCityMetadata()가 false를 반환해야 한다.
        // ────────────────────────────────────────────

        [Test, Category("city-spawn-refactor")]
        public void Property16_ClearCity_RemovesSpawnSystem()
        {
            for (int i = 0; i < Iterations; i++)
            {
                // 랜덤 파라미터 설정
                SetSmallCityParameters();

                // GenerateCity 실행
                CityGenerationResult result = _generator.GenerateCity();

                Assert.IsTrue(result.success,
                    $"[P16 iter={i}] GenerateCity()가 성공해야 합니다. 오류: {result.errorMessage}");

                // 생성 후 SpawnSystem이 존재하는지 확인 (전제 조건)
                GameObject cityGroupRoot = GetPrivateField<GameObject>(_generator, "cityGroupRoot");
                Assert.IsNotNull(cityGroupRoot,
                    $"[P16 iter={i}] GenerateCity() 후 cityGroupRoot가 존재해야 합니다.");

                Transform spawnSystemBefore = cityGroupRoot.transform.Find("SpawnSystem");
                Assert.IsNotNull(spawnSystemBefore,
                    $"[P16 iter={i}] GenerateCity() 후 SpawnSystem이 존재해야 합니다 (전제 조건).");

                // ClearCity 호출
                _generator.ClearCity();

                // Req 8.8: "SpawnSystem" GameObject가 씬에 존재하지 않아야 한다
                GameObject spawnSystemAfter = GameObject.Find("SpawnSystem");
                Assert.IsNull(spawnSystemAfter,
                    $"[P16 iter={i}] ClearCity() 후 'SpawnSystem' GameObject가 씬에 존재하면 안 됩니다.");

                // Req 8.8: CityDataAPI.HasCityMetadata()가 false를 반환해야 한다
                Assert.IsFalse(CityDataAPI.Instance.HasCityMetadata(),
                    $"[P16 iter={i}] ClearCity() 후 HasCityMetadata()가 false여야 합니다.");

                // spawnSystemRoot 내부 필드도 null이어야 한다
                GameObject spawnSystemRoot = GetPrivateField<GameObject>(_generator, "spawnSystemRoot");
                Assert.IsNull(spawnSystemRoot,
                    $"[P16 iter={i}] ClearCity() 후 spawnSystemRoot 필드가 null이어야 합니다.");
            }
        }
    }
}
