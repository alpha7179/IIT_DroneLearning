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
///   SpawnCenterRandom : SpawnCenter 범위 내 랜덤 (기본)
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
        [Tooltip("SpawnCenter 범위 내 랜덤 스폰 (기본 권장)")]
        SpawnCenterRandom,
        [Tooltip("CityDataAPI 스폰 설정 사용. 다수 드론이면 SpawnCenter 폴백.")]
        CityDataAPI,
        [Tooltip("SpawnCenter/CityDataAPI 없을 때 Inspector 설정 범위 내 랜덤")]
        Fallback,
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
    [SerializeField] private float _minSeparation = 5f;
    [Tooltip("이격 거리 미충족 시 재시도 최대 횟수")]
    [SerializeField] private int   _maxRetry      = 20;

    [Header("Spawn Coordinator — Fallback Range (SpawnCenter 없을 때)")]
    [Tooltip("폴백 XZ 랜덤 스폰 반폭 (m)")]
    [SerializeField] private float _fallbackRange  = 10f;
    [Tooltip("폴백 스폰 고도 (Y, m)")]
    [SerializeField] private float _fallbackHeight = 5f;

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

        switch (_strategy)
        {
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

        // Goal 위치 계산 후 Goal에 직접 적용
        _goalResult = ComputeGoalPosition();
        Goal.Current?.ApplySpawnPosition(_goalResult.Position);

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
