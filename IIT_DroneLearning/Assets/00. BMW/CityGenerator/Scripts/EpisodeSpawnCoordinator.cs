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
            float goalY = gn.elevation + _minSpawnHeight + UnityEngine.Random.Range(0f, heightRange);
            _goalResult = new SpawnResult
            {
                Position = new Vector3(gn.position.x, goalY, gn.position.z),
                YawDegrees = 0f
            };
            _occupiedPositions.Add(_goalResult.Position);
        }

        // Goal 위치 적용
        Goal.Current?.ApplySpawnPosition(_goalResult.Position);

        // 디버그: 스폰 위치와 도시 경계 비교
        Debug.Log($"[EpisodeSpawnCoordinator] CityMetadata 스폰 완료 — " +
                  $"도시 경계: center={metadata.cityBounds.center}, size={metadata.cityBounds.size}, " +
                  $"Evader[0]: {(_evaderObjects.Count > 0 ? GetSpawnPosition(_evaderObjects[0]).ToString() : "없음")}, " +
                  $"Pursuer[0]: {(_pursuerObjects.Count > 0 ? GetSpawnPosition(_pursuerObjects[0]).ToString() : "없음")}, " +
                  $"Goal: {_goalResult.Position}, " +
                  $"후보 노드 수: {allCandidates.Count}");
    }

    /// <summary>
    /// 후보 리스트에서 이격 조건을 만족하고 미사용인 노드를 선택한다.
    /// </summary>
    private GraphNode? SelectCandidateWithSeparation(List<GraphNode> candidates, HashSet<int> usedNodeIds)
    {
        foreach (GraphNode node in candidates)
        {
            if (usedNodeIds.Contains(node.nodeId)) continue;
            Vector3 candidatePos = new Vector3(node.position.x, node.position.y, node.position.z);
            if (IsFarEnoughFromAll(candidatePos))
                return node;
        }
        return null;
    }

    /// <summary>
    /// 이격 조건 무시, 미사용 노드 중 첫 번째를 선택한다.
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
        if (SpawnCenter.Current != null)
        {
            var sc        = SpawnCenter.Current;
            var goalRange = sc.GetGoalSpawnRange();
            float midY    = (goalRange.MinY + goalRange.MaxY) * 0.5f;

            for (int i = 0; i < _maxRetry; i++)
            {
                Vector3 rnd = sc.GetRandomPosition(goalRange);
                Vector3 pos = new Vector3(rnd.x, midY, rnd.z);
                if (IsFarEnoughFromAll(pos))
                {
                    _occupiedPositions.Add(pos);
                    return new SpawnResult { Position = pos, YawDegrees = 0f };
                }
            }

            // 재시도 초과: 마지막 후보 그대로 사용
            Vector3 last = sc.GetRandomPosition(goalRange);
            return new SpawnResult { Position = new Vector3(last.x, midY, last.z), YawDegrees = 0f };
        }

        return new SpawnResult { Position = RandomFallbackPosition(), YawDegrees = 0f };
    }

    // ───────── 헬퍼 ──────────────────────────────────────────────────────

    /// <summary>
    /// 드론 하나에 스폰 위치를 할당한다.
    /// positionFactory를 최대 _maxRetry번 호출하여 이격 조건을 만족하는 위치를 찾는다.
    /// </summary>
    private void AssignSpawn(GameObject go, Func<Vector3> positionFactory, float yaw)
    {
        Vector3 chosen = positionFactory();
        for (int i = 0; i < _maxRetry; i++)
        {
            Vector3 candidate = positionFactory();
            if (IsFarEnoughFromAll(candidate))
            {
                chosen = candidate;
                break;
            }
        }

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
