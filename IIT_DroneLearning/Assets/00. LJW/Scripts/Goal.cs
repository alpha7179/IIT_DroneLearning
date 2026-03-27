using UnityEngine;
using System;

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

    [Header("Goal — Randomization")]
    [Tooltip("에피소드마다 목표가 이동할 수 있는 최대 반경 (m, XZ 평면)")]
    [SerializeField] private float _randomizeRadius = 20f;

    [Header("Goal — SpawnCenter 쿼리 결과 (읽기 전용)")]
    [SerializeField] private float _queriedMinY;
    [SerializeField] private float _queriedMaxY;
    [SerializeField] private float _queriedRadius;

    // ───────── 이벤트 ─────────────────────────────────────────────────────
    /// <summary>"Evader" 태그 객체가 Trigger에 진입하면 발생.</summary>
    public event Action<Goal> OnArrival;

    // ───────── 내부 상태 ──────────────────────────────────────────────────
    private Vector3         _selfCenter;      // _centerTransform 미설정 시 폴백
    private CapsuleCollider _capsuleCollider;

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

    /// <summary>
    /// 에피소드 시작 시 호출.
    /// SpawnCenter가 있으면 XZ는 쿼리된 Radius 안에서 랜덤, Y는 (minY+maxY)/2 중점으로 고정.
    /// 실린더 높이도 함께 갱신한다.
    /// </summary>
    public void RandomizePosition()
    {
        // SpawnCenter가 있으면 쿼리 갱신 후 실린더 방식으로 배치
        if (SpawnCenter.Current != null)
        {
            QuerySpawnCenter(); // _queriedMinY/MaxY/Radius 갱신 + 높이 조절

            Vector3 center = SpawnCenter.Current.GetCenter();
            Vector2 xzOffset = UnityEngine.Random.insideUnitCircle * _queriedRadius;
            float   midY     = (_queriedMinY + _queriedMaxY) * 0.5f;

            transform.position = center + new Vector3(xzOffset.x, midY, xzOffset.y);
            return;
        }

        // 폴백: SpawnCenter 없을 때 기존 방식
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

        // 랜덤화 반경 (노란 원, 기준점 기준)
        Gizmos.color = new Color(1f, 1f, 0f, 0.3f);
        DrawCircleXZ(center, _randomizeRadius, 32);

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
