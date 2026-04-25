using UnityEngine;

namespace CityGenerator
{
    /// <summary>
    /// 에피소드 카운트 기반 도시 시드 자동 로테이션 컴포넌트.
    ///
    /// 에피소드가 일정 횟수 진행된 후 CityGenerator의 randomSeed를 자동으로 변경하고
    /// GenerateCity()를 호출하여 새로운 도시 레이아웃을 생성한다.
    /// 밀집도·건물 높이·격자 크기 등 Generation_Parameters는 일절 변경하지 않는다.
    ///
    /// 통합 방법:
    ///   EvaderAgent.OnEpisodeBegin() 내부에서
    ///   EpisodeSpawnCoordinator.ComputeSpawn() 직전에
    ///   WorldSeedRotator.Instance?.NotifyEpisodeBegin() 을 호출한다.
    ///
    /// 트리거 모드:
    ///   Interval  — 내부 에피소드 카운터가 RotationInterval에 도달할 때마다 로테이션 (사이클 리셋)
    ///   Boundary  — 누적 총 에피소드 수가 RotationInterval의 배수에 도달할 때 로테이션 (절대 경계값 기준)
    ///
    /// 시드 전략:
    ///   Sequential — baseSeed 부터 로테이션마다 +1 (재현 가능한 학습)
    ///   Random     — System.Environment.TickCount 기반 랜덤 (다양성 극대화)
    /// </summary>
    public class WorldSeedRotator : MonoBehaviour
    {
        // ───────── 싱글톤 ─────────────────────────────────────────────
        public static WorldSeedRotator Instance { get; private set; }

        // ───────── 열거형 ─────────────────────────────────────────────

        /// <summary>시드값 변경 방식.</summary>
        public enum SeedSequenceStrategy
        {
            /// <summary>시작 시드부터 로테이션마다 1씩 증가 — 재현 가능한 학습에 적합</summary>
            Sequential = 0,
            /// <summary>로테이션마다 System.Environment.TickCount 기반 랜덤 시드 생성 — 다양성 극대화</summary>
            Random = 1,
        }

        /// <summary>시드 로테이션 트리거 방식.</summary>
        public enum RotationTriggerMode
        {
            /// <summary>N 에피소드마다 로테이션 발생 후 카운터 리셋 (주기적 순환)</summary>
            Interval = 0,
            /// <summary>누적 총 에피소드 수가 N의 배수에 도달할 때 로테이션 (절대 경계값 기준, 카운터 리셋 없음)</summary>
            Boundary = 1,
        }

        // ───────── Inspector 설정 ─────────────────────────────────────

        [Header("World Seed Rotation — 기본 설정")]
        [Tooltip("시드 로테이션 기능 활성화/비활성화 (false 시 카운트 및 로테이션 모두 중단)")]
        [SerializeField] private bool _enabled = true;

        [Tooltip("CityGenerator 컴포넌트 참조 (Inspector에서 반드시 연결)")]
        [SerializeField] private CityGenerator _cityGenerator;

        [Header("World Seed Rotation — 트리거 설정")]
        [Tooltip(
            "Interval : N 에피소드마다 반복 로테이션, 로테이션 후 사이클 카운터 리셋\n" +
            "Boundary : 누적 총 에피소드 수가 N 의 배수(N, 2N, 3N…)에 도달할 때 로테이션, 카운터 리셋 없음")]
        [SerializeField] private RotationTriggerMode _triggerMode = RotationTriggerMode.Interval;

        [Tooltip("로테이션 간격 (Interval 모드) 또는 경계 단위 (Boundary 모드) — 최솟값 1")]
        [Min(1)]
        [SerializeField] private int _rotationInterval = 10;

        [Header("World Seed Rotation — 시드 전략")]
        [Tooltip("Sequential: 시작 시드부터 매 로테이션마다 +1\nRandom: TickCount 기반 랜덤 시드")]
        [SerializeField] private SeedSequenceStrategy _seedStrategy = SeedSequenceStrategy.Sequential;

        [Tooltip("Sequential 전략의 시작 시드값 (첫 번째 로테이션 시드 = baseSeed + 1)")]
        [SerializeField] private int _baseSeed = 42;

        // ───────── 내부 상태 ──────────────────────────────────────────

        /// <summary>Interval 모드용 사이클 내 카운터 (0 ~ RotationInterval-1)</summary>
        private int _episodeCounter;

        /// <summary>누적 총 에피소드 수 (Boundary 모드 기준 + 공통 통계용)</summary>
        private int _totalEpisodeCount;

        /// <summary>성공한 시드 로테이션 횟수</summary>
        private int _totalRotationCount;

        /// <summary>현재 적용된 시드값</summary>
        private int _currentSeed;

        // ───────── 읽기 전용 프로퍼티 ─────────────────────────────────

        /// <summary>Interval 모드: 현재 사이클 내 에피소드 카운터 (0 ~ RotationInterval-1)</summary>
        public int EpisodeCounter => _episodeCounter;

        /// <summary>누적 총 에피소드 수</summary>
        public int TotalEpisodeCount => _totalEpisodeCount;

        /// <summary>성공한 시드 로테이션 총 횟수</summary>
        public int TotalRotationCount => _totalRotationCount;

        /// <summary>현재 적용 중인 시드값</summary>
        public int CurrentSeed => _currentSeed;

        /// <summary>기능 활성화 상태</summary>
        public bool IsEnabled => _enabled;

        /// <summary>
        /// 다음 LateUpdate에서 로테이션이 예약되어 있는지 여부.
        /// Physics 콜백(OnTriggerEnter 등) 내부에서 DestroyImmediate 를 호출하면
        /// "Destroying GameObjects immediately is not permitted during physics trigger/contact"
        /// 오류가 발생하므로, 로테이션 실행을 LateUpdate 까지 안전하게 지연시킨다.
        /// </summary>
        public bool IsPendingRotation => _pendingRotation;

        // ───────── 내부 상태 (지연 실행) ─────────────────────────────

        /// <summary>LateUpdate에서 실행 대기 중인 로테이션 플래그</summary>
        private bool _pendingRotation;

        // ───────── 생명주기 ───────────────────────────────────────────

        private void Awake()
        {
            if (Instance != null && Instance != this)
            {
                Debug.LogWarning(
                    $"[WorldSeedRotator] 씬에 이미 인스턴스가 존재합니다. '{name}'은 비활성화됩니다.", this);
                enabled = false;
                return;
            }
            Instance = this;

            if (_cityGenerator == null)
                Debug.LogError(
                    "[WorldSeedRotator] CityGenerator 참조가 설정되지 않았습니다. " +
                    "Inspector에서 CityGenerator를 연결하세요.", this);

            _currentSeed        = _baseSeed;
            _episodeCounter     = 0;
            _totalEpisodeCount  = 0;
            _totalRotationCount = 0;
            _pendingRotation    = false;
        }

        private void OnDestroy()
        {
            if (Instance == this) Instance = null;
        }

        private void OnValidate()
        {
            _rotationInterval = Mathf.Max(1, _rotationInterval);
        }

        /// <summary>
        /// 예약된 로테이션을 Physics 콜백이 끝난 안전한 시점(LateUpdate)에 실행한다.
        /// ClearCity() 내부의 DestroyImmediate 가 Physics 콜백 중 호출되면 Unity 오류가 발생하므로
        /// LateUpdate 에서 실행하여 이를 방지한다.
        /// </summary>
        private void LateUpdate()
        {
            if (!_pendingRotation) return;
            if (_cityGenerator == null)
            {
                _pendingRotation = false;
                Debug.LogError(
                    "[WorldSeedRotator] CityGenerator 참조가 null입니다. " +
                    "시드 로테이션을 건너뜁니다. Inspector에서 CityGenerator를 연결하세요.", this);
                return;
            }
            _pendingRotation = false;
            ExecuteRotation();

            // 도시 재생성 후 새 건물 Bounds 기준으로 스폰 위치를 재계산한다.
            // (ComputeSpawn이 이전 도시 기준으로 이미 실행되었기 때문에
            //  골존/드론이 새 건물과 겹치는 문제를 방지)
            if (EpisodeSpawnCoordinator.Instance != null)
                EpisodeSpawnCoordinator.Instance.ComputeSpawn();
        }

        // ───────── 핵심 공개 API ──────────────────────────────────────

        /// <summary>
        /// 에피소드 시작 시 호출. 카운터를 증가시키고 조건 충족 시 로테이션을 예약한다.
        ///
        /// EvaderAgent.OnEpisodeBegin()에서 EpisodeSpawnCoordinator.ComputeSpawn() 직전에 호출한다.
        ///
        /// 실제 도시 재생성(GenerateCity → ClearCity → DestroyImmediate)은 Physics 콜백이 끝난
        /// 안전한 시점인 LateUpdate 에서 실행된다. 이 때문에 로테이션이 발생한 에피소드에서는
        /// ComputeSpawn 이 이전 도시 기준으로 스폰 위치를 계산하고, 그 다음 에피소드부터
        /// 새 도시가 적용된다 (1 에피소드 지연, RL 학습에서 무시 가능).
        /// </summary>
        public void NotifyEpisodeBegin()
        {
            if (!_enabled) return;

            // 누적 카운터 증가 (Interval/Boundary 공통)
            _totalEpisodeCount++;

            bool shouldRotate;

            if (_triggerMode == RotationTriggerMode.Interval)
            {
                _episodeCounter++;
                shouldRotate = _episodeCounter >= _rotationInterval;
                if (shouldRotate)
                    _episodeCounter = 0; // 카운터는 로테이션 성공 여부와 무관하게 즉시 리셋
            }
            else // Boundary
            {
                // 누적 에피소드 수가 rotationInterval 의 배수일 때 로테이션
                // (첫 번째 로테이션: totalEpisodeCount == rotationInterval)
                shouldRotate = _totalEpisodeCount % _rotationInterval == 0;
            }

            if (!shouldRotate) return;

            // Physics 콜백(OnTriggerEnter 등) 내부에서 DestroyImmediate 를 호출하면 Unity 오류 발생.
            // 플래그를 세워 LateUpdate 에서 안전하게 실행한다.
            _pendingRotation = true;
        }

        // ───────── 내부 메서드 ────────────────────────────────────────

        /// <summary>시드 전략에 따라 다음 시드값을 계산한다.</summary>
        private int ComputeNextSeed()
        {
            return _seedStrategy switch
            {
                SeedSequenceStrategy.Sequential => _baseSeed + _totalRotationCount + 1,
                SeedSequenceStrategy.Random     => System.Environment.TickCount,
                _                               => _currentSeed + 1,
            };
        }

        /// <summary>
        /// 실제 시드 로테이션 실행.
        /// CityGenerator.randomSeed 를 변경하고 GenerateCity() 를 호출한다.
        /// 실패 시 이전 시드로 복구를 시도하며, 복구도 실패하면 오류를 기록한다.
        /// </summary>
        private void ExecuteRotation()
        {
            int previousSeed = _currentSeed;
            int newSeed      = ComputeNextSeed();

            // 1. 새 시드 적용 및 도시 재생성
            _cityGenerator.randomSeed = newSeed;
            CityGenerationResult result = _cityGenerator.GenerateCity();

            if (result.success)
            {
                _currentSeed = newSeed;
                _totalRotationCount++;

                string metadataStatus = (CityDataAPI.Instance != null && CityDataAPI.Instance.HasCityMetadata())
                    ? "CityMetadata 등록 확인됨"
                    : "⚠ CityDataAPI 확인 불가";

                Debug.LogWarning(
                    $"  🌍 월드 시드 변경  |  Seed {previousSeed} → {newSeed}\n" +
                    $"  로테이션 #{_totalRotationCount}  |  건물 {result.buildingCount}개  |  {metadataStatus}");
            }
            else
            {
                // 2. 실패 시 이전 시드로 복구 시도
                Debug.LogError(
                    $"[WorldSeedRotator] 시드 로테이션 실패 — " +
                    $"시드 {newSeed} 재생성 실패: {result.errorMessage}. " +
                    $"이전 시드 {previousSeed} 로 복구를 시도합니다.");

                _cityGenerator.randomSeed = previousSeed;
                CityGenerationResult recoveryResult = _cityGenerator.GenerateCity();

                if (!recoveryResult.success)
                {
                    Debug.LogError(
                        $"[WorldSeedRotator] 복구 실패 — " +
                        $"이전 시드 {previousSeed} 로도 재생성 실패: {recoveryResult.errorMessage}");
                }
            }
        }
    }
}
