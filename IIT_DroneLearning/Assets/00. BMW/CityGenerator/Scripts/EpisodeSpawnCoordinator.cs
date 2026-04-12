using UnityEngine;
using System;
using System.Collections.Generic;
using CityGenerator;

/// <summary>
/// EpisodeSpawnCoordinator — 에피소드 중앙집중식 스폰 통제 시스템
///
/// 역할:
///   EvaderAgent.OnEpisodeBegin() → ComputeSpawn() 호출
///   → 씬의 태그("Evader" / "Pursuer")로 드론 수를 자동 파악
///   → 드론 수만큼 스폰 위치를 계산·캐싱 (모든 위치 간 최소 이격 보장)
///   → 각 드론/Goal은 GetSpawnPosition(gameObject) 또는 GetGoalPosition()으로 자신의 위치를 수신
///
/// 태그 기준 (Inspector에서 변경 가능):
///   _evaderTag  : "Evader" (기본)
///   _pursuerTag : "Pursuer" (기본)
///
/// 스폰 전략 (SpawnStrategy):
///   CityMetadata      : CityMetadata 기반 에피소드별 동적 스폰 (기본 권장)
///   SpawnCenterRandom : SpawnCenter 범위 내 랜덤
///   CityDataAPI       : CityDataAPI 스폰 설정 사용 (다수 드론 시 SpawnCenter 폴백)
///   Fallback          : Inspector 범위 내 랜덤
///
/// 최소 이격 거리 (_minSeparation):
///   이미 배치된 모든 위치와 비교하여 _maxRetry번 재시도.
///
/// 주요 API:
///   ComputeSpawn()                 : EvaderAgent.OnEpisodeBegin() 첫 줄에서 호출
///   GetSpawnPosition(gameObject)   : 각 드론의 OnEpisodeBegin()에서 자신의 GameObject를 인수로 호출
///   GetGoalPosition()              : Goal.RandomizePosition()에서 호출
///   IsComputed                     : 계산 완료 여부 확인
///
/// 담당: 배민우 (00. BMW)
/// </summary>
public class EpisodeSpawnCoordinator : MonoBehaviour
{
    // ───────── 스폰 전략 ──────────────────────────────────────────────────
    public enum SpawnStrategy
    {
        [Tooltip("SpawnCenter 범위 내 랜덤 스폰")]
        SpawnCenterRandom = 0,
        [Tooltip("CityDataAPI 고정 스폰 설정 사용. 다수 드론이면 SpawnCenter 폴백.")]
        CityDataAPI = 1,
        [Tooltip("SpawnCenter/CityDataAPI 없을 때 Inspector 설정 범위 내 랜덤")]
        Fallback = 2,
        [Tooltip("CityMetadata 기반 에피소드별 동적 스폰 (기본 권장)")]
        CityMetadata = 3,
    }

    // ───────── 스폰 결과 구조체 ───────────────────────────────────────────
    /// <summary>각 엔티티의 스폰 결과 (위치 + 초기 Yaw)</summary>
    public struct SpawnResult
    {
        public Vector3 Position;
        public float   YawDegrees;
    }

    // ───────── 싱글톤 ─────────────────────────────────────────────────────
    public static EpisodeSpawnCoordinator Instance { get; private set; }

    // ───────── Inspector 설정 ─────────────────────────────────────────────
    [Header("Spawn Coordinator — Tags")]
    [Tooltip("Evader 드론에 지정된 Unity 태그")]
    [SerializeField] private string _evaderTag  = "Evader";
    [Tooltip("Pursuer 드론에 지정된 Unity 태그")]
    [SerializeField] private string _pursuerTag = "Pursuer";

    [Header("Spawn Coordinator — Strategy")]
    [SerializeField] private SpawnStrategy _strategy = SpawnStrategy.SpawnCenterRandom;

    [Header("Spawn Coordinator — Separation")]
    [Tooltip("이미 배치된 모든 위치로부터 유지해야 할 최소 이격 거리 (m)")]
    [SerializeField] private float _minSeparation = 10f;
    [Tooltip("이격 거리 미충족 시 재시도 최대 횟수")]
    [SerializeField] private int   _maxRetry      = 20;

    [Header("Spawn Coordinator — Spawn Height (CityMetadata 전략)")]
    [Tooltip("CityMetadata 전략에서 노드 고도(elevation) 기준 최저 스폰 높이 (m)")]
    [SerializeField] private float _minSpawnHeight = 5f;
    [Tooltip("CityMetadata 전략에서 노드 고도(elevation) 기준 최고 스폰 높이 (m)")]
    [SerializeField] private float _maxSpawnHeight = 25f;

    [Header("Spawn Coordinator — Fallback Range (SpawnCenter 없을 때)")]
    [Tooltip("폴백 XZ 랜덤 스폰 반폭 (m)")]
    [SerializeField] private float _fallbackRange  = 10f;
    [Tooltip("폴백 스폰 고도 (Y, m)")]
    [SerializeField] private float _fallbackHeight = 15f;

    [Header("Spawn Coordinator — Debug (읽기 전용)")]
    [SerializeField] private int  _debugEvaderCount;
    [SerializeField] private int  _debugPursuerCount;

    // ───────── 내부 캐시 ──────────────────────────────────────────────────
    // 드론 GameObject → 해당 드론의 스폰 결과
    private readonly Dictionary<GameObject, SpawnResult> _spawnMap       = new Dictionary<GameObject, SpawnResult>();

    // 역할별 순서 보존 리스트 (인덱스 기반 조회용)
    private readonly List<GameObject> _evaderObjects  = new List<GameObject>();
    private readonly List<GameObject> _pursuerObjects = new List<GameObject>();

    // 이격 거리 검사용 — 이미 확정된 모든 스폰 위치
    private readonly List<Vector3> _occupiedPositions = new List<Vector3>();

    // 건물 Bounds 캐시 — ComputeSpawn() 시작 시 갱신.
    // 건물이 없는 환경에서는 빈 리스트 → IsNotOverlappingBuildings()가 항상 true 반환.
    private readonly List<Bounds> _buildingBounds = new List<Bounds>();

    private SpawnResult _goalResult;
    [SerializeField] private bool _isComputed;

    // ───────── 이벤트 ─────────────────────────────────────────────────────
    /// <summary>ComputeSpawn() 완료 후 발생.</summary>
    public event Action OnSpawnComputed;

    // ───────── 프로퍼티 ───────────────────────────────────────────────────
    public bool IsComputed   => _isComputed;
    public int  EvaderCount  => _evaderObjects.Count;
    public int  PursuerCount => _pursuerObjects.Count;
    public SpawnStrategy Strategy { get => _strategy; set => _strategy = value; }

    // ───────── 초기화 ─────────────────────────────────────────────────────
    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Debug.LogWarning($"[EpisodeSpawnCoordinator] 씬에 이미 인스턴스가 존재합니다. '{name}'은 비활성화됩니다.", this);
            enabled = false;
            return;
        }
        Instance = this;
    }

    private void OnDestroy()
    {
        if (Instance == this) Instance = null;
    }

    private void OnValidate()
    {
        _minSeparation  = Mathf.Max(0f,  _minSeparation);
        _maxRetry       = Mathf.Max(1,   _maxRetry);
        _minSpawnHeight = Mathf.Max(0f,  _minSpawnHeight);
        _maxSpawnHeight = Mathf.Max(_minSpawnHeight, _maxSpawnHeight);
        _fallbackRange  = Mathf.Max(1f,  _fallbackRange);
        _fallbackHeight = Mathf.Max(0f,  _fallbackHeight);
    }

    // ───────── 핵심 API ───────────────────────────────────────────────────

    /// <summary>
    /// 에피소드 시작 시 호출. 태그로 씬의 드론 수를 파악하고 전체 스폰 위치를 계산·캐싱한다.
    /// EvaderAgent.OnEpisodeBegin()의 첫 번째 줄에서 호출한다.
    /// </summary>
    public void ComputeSpawn()
    {
        _isComputed = false;

        // 씬에서 역할별 드론 목록 갱신
        RefreshDroneLists();

        // 공유 상태 초기화
        _spawnMap.Clear();
        _occupiedPositions.Clear();

        // 건물 Bounds 캐싱 — CityDataAPI에서 메타데이터를 읽어 건물 Bounds 목록 갱신.
        // API 없거나 메타데이터 없는 환경(빈 리스트 포함)이면 빈 리스트로 유지.
        _buildingBounds.Clear();
        if (CityDataAPI.Instance != null && CityDataAPI.Instance.HasCityMetadata())
        {
            var meta = CityDataAPI.Instance.GetCityMetadata();
            if (meta.buildings != null)
                foreach (var b in meta.buildings)
                    _buildingBounds.Add(b.bounds);
        }

        // SpawnCenter AutoSyncFromCity 활성화 시 도시 메타데이터 동기화
        if (SpawnCenter.Current != null && SpawnCenter.Current.AutoSyncFromCity)
        {
            SpawnCenter.Current.SyncFromCityMetadata();
        }

        // CityMetadata 전략은 Goal 위치를 내부에서 처리하므로 별도 플래그로 분기
        bool goalHandledByStrategy = false;

        switch (_strategy)
        {
            case SpawnStrategy.CityMetadata:
                goalHandledByStrategy = ComputeCityMetadataWithFallback();
                break;
            case SpawnStrategy.SpawnCenterRandom:
                ComputeFromSpawnCenter();
                break;
            case SpawnStrategy.CityDataAPI:
                ComputeFromCityDataAPI();
                break;
            case SpawnStrategy.Fallback:
                ComputeFallback();
                break;
        }

        // CityMetadata 전략이 Goal을 직접 처리하지 않은 경우에만 기존 Goal 로직 실행
        if (!goalHandledByStrategy)
        {
            _goalResult = ComputeGoalPosition();
            Goal.Current?.ApplySpawnPosition(_goalResult.Position);
        }

        _isComputed        = true;
        _debugEvaderCount  = _evaderObjects.Count;
        _debugPursuerCount = _pursuerObjects.Count;

        OnSpawnComputed?.Invoke();
    }

    // ──── 드론별 위치 조회 ────

    /// <summary>
    /// 개별 드론의 스폰 결과를 반환한다.
    /// 각 드론의 OnEpisodeBegin()에서 자신의 gameObject를 인수로 호출한다.
    /// </summary>
    /// <returns>맵에 등록된 드론이면 true, 미등록이면 false</returns>
    public bool TryGetSpawnResult(GameObject droneObject, out SpawnResult result)
    {
        return _spawnMap.TryGetValue(droneObject, out result);
    }

    /// <summary>
    /// 개별 드론의 스폰 위치를 반환한다. 미등록 시 Vector3.zero 반환.
    /// </summary>
    public Vector3 GetSpawnPosition(GameObject droneObject)
    {
        return _spawnMap.TryGetValue(droneObject, out SpawnResult r) ? r.Position : Vector3.zero;
    }

    /// <summary>
    /// 개별 드론의 초기 Yaw를 반환한다. 미등록 시 0 반환.
    /// </summary>
    public float GetSpawnYaw(GameObject droneObject)
    {
        return _spawnMap.TryGetValue(droneObject, out SpawnResult r) ? r.YawDegrees : 0f;
    }

    // ──── Goal 위치 조회 ────

    /// <summary>이번 에피소드의 Goal 스폰 위치.</summary>
    public Vector3 GetGoalPosition() => _goalResult.Position;

    /// <summary>이번 에피소드의 Goal SpawnResult.</summary>
    public SpawnResult GetGoalResult() => _goalResult;

    // ──── 인덱스 기반 조회 (하위 호환 / 선택적 사용) ────

    /// <summary>n번째 Evader의 스폰 위치. 범위 초과 시 Vector3.zero.</summary>
    public Vector3 GetEvaderPosition(int index = 0)
    {
        if (index < 0 || index >= _evaderObjects.Count) return Vector3.zero;
        return GetSpawnPosition(_evaderObjects[index]);
    }

    /// <summary>n번째 Pursuer의 스폰 위치. 범위 초과 시 Vector3.zero.</summary>
    public Vector3 GetPursuerPosition(int index = 0)
    {
        if (index < 0 || index >= _pursuerObjects.Count) return Vector3.zero;
        return GetSpawnPosition(_pursuerObjects[index]);
    }

    // ───────── 드론 목록 갱신 ────────────────────────────────────────────

    private void RefreshDroneLists()
    {
        _evaderObjects.Clear();
        _pursuerObjects.Clear();

        if (!string.IsNullOrEmpty(_evaderTag))
            _evaderObjects.AddRange(GameObject.FindGameObjectsWithTag(_evaderTag));

        if (!string.IsNullOrEmpty(_pursuerTag))
            _pursuerObjects.AddRange(GameObject.FindGameObjectsWithTag(_pursuerTag));

        if (_evaderObjects.Count == 0)
            Debug.LogWarning($"[EpisodeSpawnCoordinator] 태그 '{_evaderTag}'인 Evader 드론이 없습니다.", this);
        if (_pursuerObjects.Count == 0)
            Debug.LogWarning($"[EpisodeSpawnCoordinator] 태그 '{_pursuerTag}'인 Pursuer 드론이 없습니다.", this);
    }

    // ───────── 전략별 계산 ────────────────────────────────────────────────

    /// <summary>
    /// CityMetadata 전략의 유효성 검사 및 폴백 처리.
    /// 모든 검증을 통과하면 ComputeFromCityMetadata()를 호출하고 true를 반환한다 (Goal 포함 처리).
    /// 폴백 발생 시 false를 반환하여 기존 Goal 로직이 실행되도록 한다.
    /// </summary>
    /// <returns>true: CityMetadata 전략이 Goal을 포함하여 처리함, false: 폴백 발생</returns>
    private bool ComputeCityMetadataWithFallback()
    {
        // 1. CityDataAPI 인스턴스 존재 여부 확인
        if (CityDataAPI.Instance == null)
        {
            Debug.LogWarning("[EpisodeSpawnCoordinator] CityDataAPI 인스턴스가 없습니다. Fallback으로 전환합니다.", this);
            ComputeFallback();
            return false;
        }

        // 2. CityMetadata 유효성 확인
        if (!CityDataAPI.Instance.HasCityMetadata())
        {
            Debug.LogWarning("[EpisodeSpawnCoordinator] CityMetadata가 유효하지 않습니다. SpawnCenterRandom으로 폴백합니다.", this);
            ComputeFromSpawnCenter();
            return false;
        }

        // 3. validSpawnCandidates 확인
        var candidates = CityDataAPI.Instance.GetValidSpawnCandidates();
        int totalDrones = _evaderObjects.Count + _pursuerObjects.Count;
        int requiredPositions = totalDrones + 1; // +1 for Goal

        if (candidates == null || candidates.Count == 0 || candidates.Count < requiredPositions)
        {
            Debug.LogWarning($"[EpisodeSpawnCoordinator] validSpawnCandidates가 부족합니다 ({candidates?.Count ?? 0}개 < {requiredPositions}개 필요). SpawnCenterRandom으로 폴백합니다.", this);
            ComputeFromSpawnCenter();
            return false;
        }

        // 모든 검증 통과 — CityMetadata 전략 실행 (Goal 포함)
        ComputeFromCityMetadata();
        return true;
    }

    private void ComputeFromSpawnCenter()
    {
        if (SpawnCenter.Current == null)
        {
            Debug.LogWarning("[EpisodeSpawnCoordinator] SpawnCenter가 없습니다. Fallback으로 전환합니다.", this);
            ComputeFallback();
            return;
        }

        var sc = SpawnCenter.Current;

        var evaderRange = sc.GetEvaderSpawnRange();
        foreach (var go in _evaderObjects)
            AssignSpawn(go, () => sc.GetRandomPosition(evaderRange), RandomYaw());

        var pursuerRange = sc.GetPursuerSpawnRange();
        foreach (var go in _pursuerObjects)
            AssignSpawn(go, () => sc.GetRandomPosition(pursuerRange), RandomYaw());
    }

    private void ComputeFromCityDataAPI()
    {
        // CityDataAPI는 단일 Evader/Pursuer 위치만 제공 → 다수이면 SpawnCenter 폴백
        bool multiDrone = _evaderObjects.Count > 1 || _pursuerObjects.Count > 1;
        if (multiDrone)
        {
            Debug.LogWarning("[EpisodeSpawnCoordinator] CityDataAPI는 단일 드론만 지원합니다. SpawnCenter로 폴백합니다.", this);
            ComputeFromSpawnCenter();
            return;
        }

        if (CityDataAPI.Instance == null || !CityDataAPI.Instance.HasSpawnConfiguration())
        {
            Debug.LogWarning("[EpisodeSpawnCoordinator] CityDataAPI 스폰 설정 없음. SpawnCenter로 폴백합니다.", this);
            ComputeFromSpawnCenter();
            return;
        }

        if (_evaderObjects.Count == 1)
        {
            var pos    = CityDataAPI.Instance.GetEvaderSpawnPosition();
            _spawnMap[_evaderObjects[0]] = new SpawnResult { Position = pos, YawDegrees = RandomYaw() };
            _occupiedPositions.Add(pos);
        }

        if (_pursuerObjects.Count == 1)
        {
            var pos    = CityDataAPI.Instance.GetPursuerSpawnPosition();
            _spawnMap[_pursuerObjects[0]] = new SpawnResult { Position = pos, YawDegrees = RandomYaw() };
            _occupiedPositions.Add(pos);
        }
    }

    /// <summary>
    /// CityMetadata 기반 에피소드별 동적 스폰 계산.
    /// 유효 스폰 후보 노드에서 경계 후보를 우선 추출하여 Evader/Pursuer를 배치하고,
    /// 나머지 후보에서 Goal을 선택한다. 모든 위치 간 _minSeparation 이격 거리를 보장한다.
    /// </summary>
    private void ComputeFromCityMetadata()
    {
        // 1. CityMetadata 조회
        CityMetadata metadata = CityDataAPI.Instance.GetCityMetadata();
        List<GraphNode> allCandidates = metadata.validSpawnCandidates;

        int totalDrones = _evaderObjects.Count + _pursuerObjects.Count;
        int requiredPositions = totalDrones + 1; // +1 for Goal

        if (allCandidates == null || allCandidates.Count < requiredPositions)
        {
            Debug.LogWarning($"[EpisodeSpawnCoordinator] CityMetadata 유효 후보 노드가 부족합니다 ({allCandidates?.Count ?? 0}개 < {requiredPositions}개 필요). SpawnCenterRandom으로 폴백합니다.", this);
            ComputeFromSpawnCenter();
            return;
        }

        // 2. 최외각 경계 후보 추출 (GenerateSpawnConfiguration 알고리즘 통합)
        //    유효 후보 전체의 바운딩 박스 계산 → 경계까지 거리 산출 → 상위 20%(최소 10개)
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

        // 3. 경계 후보 풀 셔플 (에피소드마다 다른 결과를 위해 UnityEngine.Random 사용)
        ShuffleList(perimCandidates);

        // 4. 경계 후보에서 Evader 위치 선택 (각 드론마다 개별 노드 할당)
        HashSet<int> usedNodeIds = new HashSet<int>();
        float heightRange = Mathf.Max(0f, _maxSpawnHeight - _minSpawnHeight);

        foreach (var go in _evaderObjects)
        {
            GraphNode? selected = SelectCandidateWithSeparation(perimCandidates, usedNodeIds);
            if (!selected.HasValue)
            {
                // 경계 후보 소진 시 전체 후보에서 선택
                selected = SelectCandidateWithSeparation(allCandidates, usedNodeIds);
            }
            if (!selected.HasValue)
            {
                // 이격 조건 완화: 아무 미사용 노드 선택
                selected = SelectAnyUnusedCandidate(allCandidates, usedNodeIds);
            }

            GraphNode node = selected.Value;
            usedNodeIds.Add(node.nodeId);
            float y = node.elevation + _minSpawnHeight + UnityEngine.Random.Range(0f, heightRange);
            Vector3 pos = new Vector3(node.position.x, y, node.position.z);
            _spawnMap[go] = new SpawnResult { Position = pos, YawDegrees = RandomYaw() };
            _occupiedPositions.Add(pos);
        }

        // 5. 경계 후보에서 Pursuer 위치 선택 (Evader와 _minSeparation 이격 보장)
        foreach (var go in _pursuerObjects)
        {
            GraphNode? selected = SelectCandidateWithSeparation(perimCandidates, usedNodeIds);
            if (!selected.HasValue)
            {
                selected = SelectCandidateWithSeparation(allCandidates, usedNodeIds);
            }
            if (!selected.HasValue)
            {
                selected = SelectAnyUnusedCandidate(allCandidates, usedNodeIds);
            }

            GraphNode node = selected.Value;
            usedNodeIds.Add(node.nodeId);
            float y = node.elevation + _minSpawnHeight + UnityEngine.Random.Range(0f, heightRange);
            Vector3 pos = new Vector3(node.position.x, y, node.position.z);
            _spawnMap[go] = new SpawnResult { Position = pos, YawDegrees = RandomYaw() };
            _occupiedPositions.Add(pos);
        }

        // 6. 나머지 후보에서 Goal 위치 선택 (모든 드론과 _minSeparation 이격 보장)
        //    전체 후보를 셔플하여 경계 후보에 편향되지 않도록 함
        List<GraphNode> goalCandidates = new List<GraphNode>(allCandidates);
        ShuffleList(goalCandidates);

        GraphNode? goalNode = SelectCandidateWithSeparation(goalCandidates, usedNodeIds);
        if (!goalNode.HasValue)
        {
            goalNode = SelectAnyUnusedCandidate(goalCandidates, usedNodeIds);
        }
        if (!goalNode.HasValue && goalCandidates.Count > 0)
        {
            // 최종 폴백: 사용된 노드라도 선택
            goalNode = goalCandidates[0];
        }

        if (goalNode.HasValue)
        {
            GraphNode gn = goalNode.Value;
            float cityMaxHeight = GetMaxWorldBuildingHeight(metadata);
            // Goal 실린더는 0 ~ cityMaxHeight 를 Y축으로 꽉 채움
            // Y 중앙값 = cityMaxHeight / 2
            // 건물 데이터 없을 시 기존 랜덤 범위 폴백
            float goalY = cityMaxHeight > 0f
                ? cityMaxHeight * 0.5f
                : gn.elevation + _minSpawnHeight + UnityEngine.Random.Range(0f, heightRange);
            _goalResult = new SpawnResult
            {
                Position = new Vector3(gn.position.x, goalY, gn.position.z),
                YawDegrees = 0f
            };
            // [Fix] 실제 goalY 기준으로 추가하여 드론과의 이격 거리 판정이 올바르게 동작하도록 수정.
            // 이전 코드는 gn.elevation(지면 높이)을 Y로 사용해 높은 고도 드론과의 이격이 실제보다 짧게 평가됨.
            _occupiedPositions.Add(_goalResult.Position);
        }

        // Goal 위치 적용 — 건물 최대 높이로 실린더가 0~maxHeight를 꽉 채우도록 설정
        if (Goal.Current != null)
        {
            float cityMaxH = GetMaxWorldBuildingHeight(metadata);
            if (cityMaxH > 0f)
                Goal.Current.ApplySpawnPositionWithHeightRange(_goalResult.Position, 0f, cityMaxH);
            else
                Goal.Current.ApplySpawnPosition(_goalResult.Position);
        }

        // 디버그: 스폰 위치와 도시 경계 비교
        Debug.Log($"[EpisodeSpawnCoordinator] CityMetadata 스폰 완료 — " +
                  $"도시 경계: center={metadata.cityBounds.center}, size={metadata.cityBounds.size}, " +
                  $"Evader[0]: {(_evaderObjects.Count > 0 ? GetSpawnPosition(_evaderObjects[0]).ToString() : "없음")}, " +
                  $"Pursuer[0]: {(_pursuerObjects.Count > 0 ? GetSpawnPosition(_pursuerObjects[0]).ToString() : "없음")}, " +
                  $"Goal: {_goalResult.Position}, " +
                  $"후보 노드 수: {allCandidates.Count}");
    }

    /// <summary>
    /// 후보 리스트에서 이격 조건 AND 건물 비중첩 조건을 모두 만족하고 미사용인 노드를 선택한다.
    /// CityMetadata 전략 전용: 실제 스폰 Y는 랜덤 범위에서 결정되므로 이격 검사는 XZ 평면 기준으로 수행한다.
    /// (_occupiedPositions에는 실제 비행 고도의 Y가 저장되어 있으므로 node.position.y로 3D 검사하면
    ///  Y 차이가 인위적으로 커져 이격 조건을 잘못 통과할 수 있음)
    /// </summary>
    private GraphNode? SelectCandidateWithSeparation(List<GraphNode> candidates, HashSet<int> usedNodeIds)
    {
        foreach (GraphNode node in candidates)
        {
            if (usedNodeIds.Contains(node.nodeId)) continue;
            if (IsFarEnoughXZ(node.position.x, node.position.z) &&
                IsNotOverlappingBuildings(new Vector3(node.position.x, node.position.y, node.position.z)))
                return node;
        }
        return null;
    }

    /// <summary>후보 XZ 좌표가 이미 배치된 모든 위치에서 _minSeparation 이상 떨어져 있는지 XZ 평면 기준으로 확인.</summary>
    private bool IsFarEnoughXZ(float x, float z)
    {
        foreach (var pos in _occupiedPositions)
        {
            float dx = x - pos.x;
            float dz = z - pos.z;
            if (dx * dx + dz * dz < _minSeparation * _minSeparation)
                return false;
        }
        return true;
    }

    /// <summary>
    /// 이격 및 건물 충돌 조건 완화: 아무 미사용 노드를 선택한다.
    /// SelectCandidateWithSeparation()이 두 번 모두 실패했을 때만 호출되는 최후 폴백.
    /// 이 경로는 건물 중첩 가능성을 의도적으로 허용한다 (모든 후보가 건물 내부인 극단적 상황).
    /// </summary>
    private GraphNode? SelectAnyUnusedCandidate(List<GraphNode> candidates, HashSet<int> usedNodeIds)
    {
        foreach (GraphNode node in candidates)
        {
            if (!usedNodeIds.Contains(node.nodeId))
                return node;
        }
        return null;
    }

    /// <summary>
    /// Fisher-Yates 셔플 (UnityEngine.Random 사용).
    /// </summary>
    private static void ShuffleList(List<GraphNode> list)
    {
        for (int i = list.Count - 1; i > 0; i--)
        {
            int j = UnityEngine.Random.Range(0, i + 1);
            GraphNode tmp = list[i];
            list[i] = list[j];
            list[j] = tmp;
        }
    }

    /// <summary>
    /// CityMetadata의 건물 목록에서 실제 월드 좌표 기준 최대 건물 높이를 반환한다.
    /// building.size.y = buildingHeight * unitDistance (월드 단위).
    /// 건물이 없으면 0을 반환한다.
    /// </summary>
    private static float GetMaxWorldBuildingHeight(CityMetadata metadata)
    {
        if (metadata.buildings == null || metadata.buildings.Count == 0) return 0f;
        float max = 0f;
        foreach (var b in metadata.buildings)
            if (b.size.y > max) max = b.size.y;
        return max;
    }

    private void ComputeFallback()
    {
        foreach (var go in _evaderObjects)
            AssignSpawn(go, RandomFallbackPosition, RandomYaw());

        foreach (var go in _pursuerObjects)
            AssignSpawn(go, RandomFallbackPosition, RandomYaw());
    }

    // ───────── Goal 위치 계산 ─────────────────────────────────────────────

    private SpawnResult ComputeGoalPosition()
    {
        float goalOriginalRadius = (Goal.Current != null) ? Goal.Current.OriginalRadius  : 0f;
        float goalMinRadius      = (Goal.Current != null) ? Goal.Current.MinShrinkRadius : 0f;
        float goalShrinkStep     = (Goal.Current != null) ? Goal.Current.ShrinkStep      : 0.5f;

        if (SpawnCenter.Current != null)
        {
            var sc        = SpawnCenter.Current;
            var goalRange = sc.GetGoalSpawnRange();
            float midY    = (goalRange.MinY + goalRange.MaxY) * 0.5f;

            // 폴백 추적: 이격은 통과했지만 최소 반경에서도 건물 중첩인 최선 후보
            Vector3 bestCandidate    = Vector3.zero;
            float   bestFittedRadius = goalMinRadius;
            float   bestOverlap      = float.MaxValue;
            bool    bestFound        = false;

            for (int i = 0; i < _maxRetry; i++)
            {
                Vector3 rnd = sc.GetRandomPosition(goalRange);
                Vector3 pos = new Vector3(rnd.x, midY, rnd.z);

                if (!IsFarEnoughFromAll(pos)) continue;

                // 이격 통과 → 반경을 줄여가며 건물 겹침 해소 시도
                if (TryFitGoalAtPosition(pos, goalOriginalRadius, goalMinRadius, goalShrinkStep,
                                         out float fittedRadius))
                {
                    // 겹침 없는 반경 확보 성공 → 이 위치 확정
                    Goal.Current?.SetColliderRadius(fittedRadius);
                    _occupiedPositions.Add(pos);
                    return new SpawnResult { Position = pos, YawDegrees = 0f };
                }

                // 최소 반경에서도 겹침 → 폴백 후보로 기록 (overlap 가장 작은 위치 선호)
                float overlap = BuildingOverlapDepth(pos);
                if (overlap < bestOverlap)
                {
                    bestCandidate    = pos;
                    bestFittedRadius = goalMinRadius;
                    bestOverlap      = overlap;
                    bestFound        = true;
                }
            }

            // 재시도 초과: 이격 조건은 통과한 최소 중첩 위치 + 최소 반경으로 폴백
            if (bestFound)
            {
                Debug.LogWarning($"[EpisodeSpawnCoordinator] ComputeGoalPosition() _maxRetry({_maxRetry}) 초과: " +
                                 $"최소 중첩 위치({bestCandidate}) + 최소 반경({bestFittedRadius:F2}m) 사용.", this);
                Goal.Current?.SetColliderRadius(bestFittedRadius);
                _occupiedPositions.Add(bestCandidate);
                return new SpawnResult { Position = bestCandidate, YawDegrees = 0f };
            }

            // 이격 조건조차 만족하는 위치 없음: 마지막 랜덤 위치로 강제 배치
            Vector3 last        = sc.GetRandomPosition(goalRange);
            Vector3 fallbackPos = new Vector3(last.x, midY, last.z);
            Goal.Current?.SetColliderRadius(goalMinRadius);
            _occupiedPositions.Add(fallbackPos);
            return new SpawnResult { Position = fallbackPos, YawDegrees = 0f };
        }

        Vector3 fallback = RandomFallbackPosition();
        _occupiedPositions.Add(fallback);
        return new SpawnResult { Position = fallback, YawDegrees = 0f };
    }

    /// <summary>
    /// 주어진 위치에서 Goal 반경을 originalRadius → minRadius 방향으로 줄여가며 건물 비중첩을 탐색한다.
    /// 겹침 없는 반경을 찾으면 fittedRadius에 저장하고 true를 반환한다.
    /// minRadius에서도 여전히 겹치면 false를 반환한다 (위치 이동 필요).
    /// </summary>
    private bool TryFitGoalAtPosition(Vector3 pos, float originalRadius, float minRadius,
                                       float shrinkStep, out float fittedRadius)
    {
        fittedRadius = originalRadius;
        if (_buildingBounds.Count == 0) return true; // 건물 없는 환경: 어떤 반경이든 허용

        float r = originalRadius;
        while (true)
        {
            if (IsNotOverlappingBuildings(pos, r))
            {
                fittedRadius = r;
                return true;
            }
            if (r <= minRadius)
            {
                fittedRadius = minRadius;
                return false; // 최소 반경에서도 겹침 → 위치 이동 필요
            }
            r = Mathf.Max(minRadius, r - shrinkStep);
        }
    }

    // ───────── 헬퍼 ──────────────────────────────────────────────────────

    /// <summary>
    /// 드론 하나에 스폰 위치를 할당한다.
    /// positionFactory를 최대 _maxRetry번 호출하여 이격 및 건물 비중첩 조건을 만족하는 위치를 찾는다.
    /// _maxRetry 초과 시 시도한 위치 중 건물 중첩이 가장 적은(overlap depth 최소) 위치를 사용하고 경고 로그를 출력한다.
    /// </summary>
    private void AssignSpawn(GameObject go, Func<Vector3> positionFactory, float yaw)
    {
        Vector3 chosen        = positionFactory();
        float   chosenOverlap = BuildingOverlapDepth(chosen);

        for (int i = 0; i < _maxRetry; i++)
        {
            Vector3 candidate = positionFactory();
            if (IsFarEnoughFromAll(candidate) && IsNotOverlappingBuildings(candidate))
            {
                chosen = candidate;
                chosenOverlap = 0f;
                break;
            }

            // _maxRetry 초과 대비: 현재까지 최소 중첩 위치를 추적
            float overlap = BuildingOverlapDepth(candidate);
            if (IsFarEnoughFromAll(candidate) && overlap < chosenOverlap)
            {
                chosen        = candidate;
                chosenOverlap = overlap;
            }
        }

        if (chosenOverlap > 0f)
            Debug.LogWarning($"[EpisodeSpawnCoordinator] AssignSpawn() _maxRetry({_maxRetry}) 초과: " +
                             $"'{go.name}'에 최소 중첩 위치({chosen}) 사용 (overlap={chosenOverlap:F2}m).", this);

        _spawnMap[go] = new SpawnResult { Position = chosen, YawDegrees = yaw };
        _occupiedPositions.Add(chosen);
    }

    /// <summary>후보 위치가 이미 배치된 모든 위치에서 _minSeparation 이상 떨어져 있는지 확인.</summary>
    private bool IsFarEnoughFromAll(Vector3 candidate)
    {
        foreach (var pos in _occupiedPositions)
            if (Vector3.Distance(candidate, pos) < _minSeparation)
                return false;
        return true;
    }

    /// <summary>
    /// 후보 위치가 캐싱된 모든 건물 Bounds와 중첩되지 않는지 확인.
    /// colliderRadius만큼 건물 Bounds를 확장(각 면 +colliderRadius)한 후 Contains() 검사.
    /// _buildingBounds가 비어있으면 항상 true 반환 (건물 없는 환경 보존).
    ///
    /// [주의] Bounds.Expand(amount)는 size를 amount 늘리므로 각 면이 amount/2씩 확장된다.
    ///   각 면을 colliderRadius만큼 확장하려면 Expand(colliderRadius * 2f)를 사용한다.
    /// </summary>
    private bool IsNotOverlappingBuildings(Vector3 candidate, float colliderRadius = 0f)
    {
        if (_buildingBounds.Count == 0) return true;

        float expandAmount = colliderRadius * 2f;
        foreach (var b in _buildingBounds)
        {
            Bounds expanded = b;
            if (expandAmount > 0f) expanded.Expand(expandAmount);
            if (expanded.Contains(candidate)) return false;
        }
        return true;
    }

    /// <summary>
    /// 후보 위치가 건물 Bounds에 얼마나 깊이 들어가 있는지를 반환한다.
    /// 중첩이 없으면 0, 중첩이 있으면 건물 중심까지의 거리 역수(값이 클수록 깊이 중첩)를 반환한다.
    /// _maxRetry 초과 시 최소 중첩 위치 선택에 사용한다.
    /// </summary>
    private float BuildingOverlapDepth(Vector3 candidate)
    {
        float totalDepth = 0f;
        foreach (var b in _buildingBounds)
        {
            if (!b.Contains(candidate)) continue;
            // 중심까지의 거리가 짧을수록 깊이 중첩 → 역수를 더해 큰 값 = 더 깊은 중첩
            float dist = Vector3.Distance(candidate, b.center);
            totalDepth += (dist > 0f) ? (1f / dist) : float.MaxValue * 0.001f;
        }
        return totalDepth;
    }

    private Vector3 RandomFallbackPosition()
        => new Vector3(
            UnityEngine.Random.Range(-_fallbackRange, _fallbackRange),
            _fallbackHeight,
            UnityEngine.Random.Range(-_fallbackRange, _fallbackRange));

    private static float RandomYaw() => UnityEngine.Random.Range(0f, 360f);

    // ───────── Editor 디버그 기즈모 ──────────────────────────────────────
#if UNITY_EDITOR
    private void OnDrawGizmosSelected()
    {
        if (!Application.isPlaying || !_isComputed) return;

        // Evader 스폰 위치 — 파랑
        Gizmos.color = new Color(0.3f, 0.5f, 1f, 0.9f);
        for (int i = 0; i < _evaderObjects.Count; i++)
        {
            if (!_spawnMap.TryGetValue(_evaderObjects[i], out var r)) continue;
            Gizmos.DrawSphere(r.Position, 0.6f);
            UnityEditor.Handles.color = Gizmos.color;
            UnityEditor.Handles.Label(r.Position + Vector3.up * 1.2f, $"Evader[{i}]");
        }

        // Pursuer 스폰 위치 — 빨강
        Gizmos.color = new Color(1f, 0.3f, 0.3f, 0.9f);
        for (int i = 0; i < _pursuerObjects.Count; i++)
        {
            if (!_spawnMap.TryGetValue(_pursuerObjects[i], out var r)) continue;
            Gizmos.DrawSphere(r.Position, 0.6f);
            UnityEditor.Handles.color = Gizmos.color;
            UnityEditor.Handles.Label(r.Position + Vector3.up * 1.2f, $"Pursuer[{i}]");
        }

        // Goal 스폰 위치 — 노랑
        Gizmos.color = new Color(1f, 1f, 0f, 0.9f);
        Gizmos.DrawSphere(_goalResult.Position, 0.6f);
        UnityEditor.Handles.color = Gizmos.color;
        UnityEditor.Handles.Label(_goalResult.Position + Vector3.up * 1.2f, "Goal");

        // 모든 스폰 위치 간 이격 거리 연결선
        Gizmos.color = new Color(1f, 1f, 1f, 0.2f);
        for (int i = 0; i < _occupiedPositions.Count; i++)
            for (int j = i + 1; j < _occupiedPositions.Count; j++)
                Gizmos.DrawLine(_occupiedPositions[i], _occupiedPositions[j]);
    }
#endif
}
