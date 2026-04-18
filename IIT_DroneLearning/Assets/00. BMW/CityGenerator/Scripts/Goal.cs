using UnityEngine;
using System;
using CityGenerator;

/// <summary>
/// Goal — 목표 지점 관리 컴포넌트
///
/// Goal GameObject에 부착하면:
///   1. 태그가 자동으로 "Goal"로 설정된다 (CLAUDE.md 표준 태그).
///   2. isTrigger Collider가 없으면 CapsuleCollider가 자동 추가된다.
///   3. Goal.Current에 자동 등록되어 EvaderAgent가 Inspector 없이 찾을 수 있다.
///
/// 랜덤화 기준점:
///   _centerTransform이 할당되어 있으면 해당 Transform 위치가 기준점.
///   없으면 Awake 시점의 자기 자신 위치가 기준점.
///
/// 실린더 형태:
///   SpawnCenter가 있으면 minY~maxY 범위를 읽어 Y 중점으로 위치를 잡고
///   CapsuleCollider 높이를 (maxY - minY)에 꽉 차도록 자동 조절한다.
///   동기화 모드면 sharedRange, 비동기 모드면 goalRange 값을 사용한다.
///
/// 도달 판정:
///   "Evader" 태그를 가진 객체가 Trigger Collider에 진입하면 OnArrival 이벤트 발생.
///
/// </summary>
[RequireComponent(typeof(Collider))]
public class Goal : MonoBehaviour
{
    // ───────── 태그 상수 ──────────────────────────────────────────────────
    private const string GoalTag   = "Goal";
    private const string EvaderTag = "Evader";

    // ───────── 자동 등록 (Static) ─────────────────────────────────────────
    /// <summary>현재 씬에서 활성화된 Goal. Enable 시 자동 등록.</summary>
    public static Goal Current { get; private set; }

    // ───────── Inspector 설정 ─────────────────────────────────────────────
    [Header("Goal — Center")]
    [Tooltip("랜덤화 기준점으로 사용할 Transform. 비워두면 SpawnCenter 또는 자기 자신의 초기 위치를 사용.")]
    [SerializeField] private Transform _centerTransform;

    [Header("Goal — Randomization (SpawnCenter 없을 때 폴백)")]
    [Tooltip("에피소드마다 목표가 이동할 수 있는 최대 반경 (m, XZ 평면)")]
    [SerializeField] private float _randomizeRadius = 20f;

    [Header("Goal — Building Overlap Shrink")]
    [Tooltip("활성화 시 건물과 겹칠 경우 CapsuleCollider 반경을 축소하여 충돌 회피")]
    [SerializeField] private bool  _enableShrinkOnOverlap = false;
    [Tooltip("축소 시 허용 최소 반경 (m). 이 값 이하로는 축소하지 않음.")]
    [SerializeField] private float _minShrinkRadius = 0.5f;
    [Tooltip("반경 축소 단계 (m). 한 번에 이 값씩 줄여가며 건물 겹침을 검사.")]
    [SerializeField] private float _shrinkStep = 0.5f;

    [Header("Goal — SpawnCenter 쿼리 결과 (읽기 전용)")]
    [SerializeField] private float _queriedMinY;
    [SerializeField] private float _queriedMaxY;
    [SerializeField] private float _queriedRadius;
    [SerializeField] private float _queriedWidth;
    [SerializeField] private float _queriedDepth;

    // ───────── 이벤트 ─────────────────────────────────────────────────────
    /// <summary>"Evader" 태그 객체가 Trigger에 진입하면 발생.</summary>
    public event Action<Goal> OnArrival;

    // ───────── 내부 상태 ──────────────────────────────────────────────────
    private Vector3         _selfCenter;        // _centerTransform 미설정 시 폴백
    private CapsuleCollider _capsuleCollider;
    private float           _originalRadius;    // Awake 시점의 원래 반경 저장

    // ───────── Editor: 컴포넌트 추가 시 자동 설정 ────────────────────────
#if UNITY_EDITOR
    private void Reset()
    {
        try { gameObject.tag = GoalTag; }
        catch { Debug.LogWarning($"[Goal] '{GoalTag}' 태그가 Tags에 등록되어 있지 않습니다.", this); }

        if (GetComponent<Collider>() == null)
        {
            var col = gameObject.AddComponent<CapsuleCollider>();
            col.isTrigger = true;
            col.radius    = 2.0f;
            col.height    = 10f;
            col.direction = 1; // Y축
        }
        else
            GetComponent<Collider>().isTrigger = true;
    }
#endif

    // ───────── 초기화 ─────────────────────────────────────────────────────
    private void Awake()
    {
        _selfCenter      = transform.position;
        _capsuleCollider = GetComponent<CapsuleCollider>();
        _originalRadius  = (_capsuleCollider != null) ? _capsuleCollider.radius : 0f;

        var col = GetComponent<Collider>();
        if (col != null && !col.isTrigger)
        {
            Debug.LogWarning("[Goal] Collider의 Is Trigger가 꺼져 있습니다. 자동으로 활성화합니다.", this);
            col.isTrigger = true;
        }

        // SpawnCenter에서 골존 범위 쿼리 + 실린더 형태 적용
        QuerySpawnCenter();
    }

    /// <summary>SpawnCenter에서 골존 스폰 범위를 쿼리하여 내부 필드에 저장하고 실린더를 조형한다.</summary>
    public void QuerySpawnCenter()
    {
        if (SpawnCenter.Current != null)
        {
            var range = SpawnCenter.Current.GetGoalSpawnRange();
            _queriedMinY   = range.MinY;
            _queriedMaxY   = range.MaxY;
            _queriedRadius = range.Radius;
            _queriedWidth  = range.Width;
            _queriedDepth  = range.Depth;

            // SpawnCenter 기준점을 centerTransform 대신 사용
            _centerTransform = SpawnCenter.Current.transform;

            UpdateCylinderShape();
        }
    }

    /// <summary>
    /// 쿼리된 minY/maxY 기준으로 오브젝트의 localScale.y를 조절한다.
    /// 월드 높이(worldHeight) = capsuleCollider.height × localScale.y 이므로
    /// localScale.y = targetWorldHeight / capsuleCollider.height 로 역산.
    /// </summary>
    private void UpdateCylinderShape()
    {
        if (_capsuleCollider == null) return;
        if (_capsuleCollider.height <= 0f) return;

        _capsuleCollider.direction = 1; // Y축

        float targetWorldHeight = Mathf.Max(0.1f, _queriedMaxY - _queriedMinY);
        float newScaleY = targetWorldHeight / _capsuleCollider.height;

        Vector3 s = transform.localScale;
        transform.localScale = new Vector3(s.x, newScaleY, s.z);
    }

    /// <summary>쿼리된 스폰 범위 (읽기 전용)</summary>
    public float QueriedMinY   => _queriedMinY;
    public float QueriedMaxY   => _queriedMaxY;
    public float QueriedRadius => _queriedRadius;
    public float QueriedWidth  => _queriedWidth;
    public float QueriedDepth  => _queriedDepth;

    private void OnEnable()
    {
        if (Current != null && Current != this)
            Debug.LogWarning($"[Goal] '{Current.name}'이 이미 등록되어 있습니다. '{name}'으로 교체합니다.", this);
        Current = this;
    }

    private void OnDisable()
    {
        if (Current == this) Current = null;
    }

    private void OnValidate()
    {
        _randomizeRadius = Mathf.Max(0f, _randomizeRadius);
        _minShrinkRadius = Mathf.Max(0.01f, _minShrinkRadius);
        _shrinkStep      = Mathf.Max(0.01f, _shrinkStep);
    }

    // ───────── 충돌 판정 (Trigger) ────────────────────────────────────────
    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag(EvaderTag))
            NotifyArrival();
    }

    // ───────── Public API ─────────────────────────────────────────────────

    /// <summary>랜덤화 기준 중심 위치. _centerTransform 할당 시 해당 위치, 아니면 초기 자기 위치.</summary>
    public Vector3 GetCenter()
    {
        if (_centerTransform != null) return _centerTransform.position;
        return Application.isPlaying ? _selfCenter : transform.position;
    }

    /// <summary>랜덤화 반경 (읽기 전용 API).</summary>
    public float RandomizeRadius => _randomizeRadius;

    /// <summary>현재 CapsuleCollider의 반경 (읽기 전용).</summary>
    public float ColliderRadius  => (_capsuleCollider != null) ? _capsuleCollider.radius : 0f;
    /// <summary>Awake 시점에 기록된 원래 반경. EpisodeSpawnCoordinator가 shrink 계산 시 기준값으로 사용.</summary>
    public float OriginalRadius  => _originalRadius;
    /// <summary>허용 최소 반경. EpisodeSpawnCoordinator가 shrink 하한으로 사용.</summary>
    public float MinShrinkRadius => _minShrinkRadius;
    /// <summary>반경 축소 단계. EpisodeSpawnCoordinator가 shrink 보폭으로 사용.</summary>
    public float ShrinkStep      => _shrinkStep;

    /// <summary>
    /// EpisodeSpawnCoordinator가 shrink-before-move 결과를 직접 적용할 때 호출.
    /// _originalRadius 는 변경하지 않는다.
    /// </summary>
    public void SetColliderRadius(float r)
    {
        if (_capsuleCollider == null) return;
        _capsuleCollider.radius = Mathf.Max(_minShrinkRadius, r);
    }

    /// <summary>
    /// EpisodeSpawnCoordinator가 ComputeSpawn() 내에서 직접 호출한다.
    /// 위치를 적용하고 실린더 높이를 갱신한다.
    /// </summary>
    public void ApplySpawnPosition(Vector3 worldPosition)
    {
        // QuerySpawnCenter() 호출 제거:
        // ApplySpawnPositionWithHeightRange()로 설정된 높이를 유지하기 위해
        // 이 메서드는 XZ 위치만 갱신하고 실린더 높이는 건드리지 않는다.
        transform.position = worldPosition;
        if (_enableShrinkOnOverlap) TryShrinkToAvoidBuildings();
    }

    /// <summary>
    /// CityMetadata 전략에서 호출. 건물 높이 데이터를 직접 반영하여
    /// Goal 실린더가 minY~maxY 범위를 Y축으로 완전히 채우도록 설정한다.
    ///
    /// ■ 동작
    ///   1. Y 중앙값 = (minY + maxY) / 2 로 위치 설정
    ///   2. CapsuleCollider 높이 = maxY - minY 로 스케일 조절
    ///   3. XZ 위치는 worldPosition에서 가져옴
    /// </summary>
    /// <param name="worldPosition">Goal의 XZ 위치 (Y는 무시되고 재계산됨)</param>
    /// <param name="minY">실린더 하단 Y (보통 0)</param>
    /// <param name="maxY">실린더 상단 Y (가장 높은 건물의 월드 높이)</param>
    public void ApplySpawnPositionWithHeightRange(Vector3 worldPosition, float minY, float maxY)
    {
        // 쿼리 결과 필드 갱신 (Inspector 디버그용)
        _queriedMinY = minY;
        _queriedMaxY = maxY;

        // 실린더 높이 조형
        UpdateCylinderShape();

        // Y 중앙값으로 위치 설정
        float centerY = (minY + maxY) * 0.5f;
        transform.position = new Vector3(worldPosition.x, centerY, worldPosition.z);
        if (_enableShrinkOnOverlap) TryShrinkToAvoidBuildings();
    }

    /// <summary>
    /// 현재 위치에서 CapsuleCollider가 건물과 겹치면 반경을 점진적으로 축소한다.
    /// _enableShrinkOnOverlap = false이면 즉시 반환 (기존 동작 보존).
    ///
    /// 매 호출마다 _originalRadius 에서 시작하므로 이전 에피소드의 축소 값이 누적되지 않는다.
    /// 건물 Bounds 정보는 CityDataAPI를 통해 조회한다.
    /// CityDataAPI가 없거나 메타데이터가 없으면 동작하지 않는다.
    /// </summary>
    private void TryShrinkToAvoidBuildings()
    {
        if (!_enableShrinkOnOverlap) return;
        if (_capsuleCollider == null) return;
        if (CityDataAPI.Instance == null || !CityDataAPI.Instance.HasCityMetadata()) return;

        var meta = CityDataAPI.Instance.GetCityMetadata();
        if (meta.buildings == null || meta.buildings.Count == 0) return;

        // 매 에피소드 원래 반경에서 시작 (이전 shrink 누적 방지)
        _capsuleCollider.radius = _originalRadius;

        float r = _originalRadius;
        while (true)
        {
            bool overlaps = false;
            foreach (var b in meta.buildings)
            {
                Bounds expanded = b.bounds;
                expanded.Expand(r * 2f);
                if (expanded.Contains(transform.position)) { overlaps = true; break; }
            }

            if (!overlaps)
            {
                _capsuleCollider.radius = r;
                if (r < _originalRadius)
                    Debug.Log($"[Goal] TryShrinkToAvoidBuildings: 반경 {_originalRadius:F2} → {r:F2}m로 축소.", this);
                return;
            }

            if (r <= _minShrinkRadius)
            {
                _capsuleCollider.radius = _minShrinkRadius;
                Debug.LogWarning($"[Goal] TryShrinkToAvoidBuildings: 최소 반경({_minShrinkRadius}m)에서도 건물과 중첩됨.", this);
                return;
            }

            r = Mathf.Max(_minShrinkRadius, r - _shrinkStep);
        }
    }

    /// <summary>
    /// 에피소드 시작 시 수동 호출용 (EpisodeSpawnCoordinator 없을 때 폴백).
    /// 우선순위:
    ///   1. SpawnCenter 범위 내 랜덤
    ///   2. Fallback: 반경 기반 랜덤
    /// </summary>
    public void RandomizePosition()
    {
        if (SpawnCenter.Current != null)
        {
            // QuerySpawnCenter() 호출 제거:
            // EpisodeSpawnCoordinator가 ApplySpawnPositionWithHeightRange()로 설정한
            // 실린더 높이(_queriedMinY/_queriedMaxY)를 유지한 채 XZ 위치만 재랜덤화한다.
            var     range  = SpawnCenter.Current.GetGoalSpawnRange();
            Vector3 rndPos = SpawnCenter.Current.GetRandomPosition(range);
            float   midY   = (_queriedMinY + _queriedMaxY) * 0.5f;

            transform.position = new Vector3(rndPos.x, midY, rndPos.z);
            return;
        }

        // 폴백: SpawnCenter 없을 때 반경 기반 랜덤
        Vector2 offset = UnityEngine.Random.insideUnitCircle * _randomizeRadius;
        Vector3 c = GetCenter();
        transform.position = c + new Vector3(offset.x, 0f, offset.y);
    }

    /// <summary>OnArrival 이벤트 발생. OnTriggerEnter에서 자동 호출.</summary>
    public void NotifyArrival() => OnArrival?.Invoke(this);

    /// <summary>현재 Goal의 월드 위치 반환.</summary>
    public Vector3 GetPosition() => transform.position;

    // ───────── Editor 디버그 기즈모 ───────────────────────────────────────
#if UNITY_EDITOR
    private void OnDrawGizmosSelected()
    {
        Vector3 center = GetCenter();

        // 중심점 마커 (노란 구)
        Gizmos.color = new Color(1f, 1f, 0f, 0.9f);
        Gizmos.DrawSphere(center, 0.4f);

        // 랜덤화 범위 기즈모 (SpawnCenter 없을 때 폴백 반경)
        if (SpawnCenter.Current == null)
        {
            Gizmos.color = new Color(1f, 1f, 0f, 0.3f);
            DrawCircleXZ(center, _randomizeRadius, 32);
        }

        // 중심점 → 현재 위치 연결선
        if (Application.isPlaying && transform.position != center)
        {
            Gizmos.color = new Color(1f, 1f, 0f, 0.5f);
            Gizmos.DrawLine(center, transform.position);
        }

        // Trigger Collider 범위 — 실린더 (녹색 두 원 + 수직선)
        if (_capsuleCollider != null)
        {
            float r      = _capsuleCollider.radius * transform.lossyScale.x;
            float halfH  = _capsuleCollider.height * transform.lossyScale.y * 0.5f;
            Vector3 pos  = transform.position;
            Vector3 top  = pos + Vector3.up * halfH;
            Vector3 bot  = pos - Vector3.up * halfH;

            Gizmos.color = new Color(0f, 1f, 0f, 0.4f);
            DrawCircleXZ(top, r, 32);
            DrawCircleXZ(bot, r, 32);

            // 4방향 수직 연결선
            for (int i = 0; i < 4; i++)
            {
                float angle = i * 90f * Mathf.Deg2Rad;
                Vector3 edge = new(Mathf.Cos(angle) * r, 0f, Mathf.Sin(angle) * r);
                Gizmos.DrawLine(bot + edge, top + edge);
            }
        }
    }

    private static void DrawCircleXZ(Vector3 center, float radius, int segments)
    {
        float step = 360f / segments;
        Vector3 prev = center + new Vector3(radius, 0f, 0f);
        for (int i = 1; i <= segments; i++)
        {
            float angle = i * step * Mathf.Deg2Rad;
            Vector3 next = center + new Vector3(Mathf.Cos(angle) * radius, 0f, Mathf.Sin(angle) * radius);
            Gizmos.DrawLine(prev, next);
            prev = next;
        }
    }
#endif
}
