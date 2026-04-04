using UnityEngine;
using System;
using CityGenerator;

/// <summary>
/// SpawnCenter — 중점 기반 스폰 범위 관리 컴포넌트
///
/// 빈 GameObject에 부착하여 골존·추적자·도망자의 스폰 범위를 중앙 집중 관리한다.
///
/// SyncMode:
///   Synchronized   : 세 객체가 동일한 범위를 공유. 기즈모 1개.
///   Desynchronized : 골존·추적자·도망자 각각 별도 범위. 기즈모 3개.
///
/// RangeMode:
///   Radius    : XZ 평면을 원형 반경으로 지정.
///   Rectangle : XZ 평면을 너비(Width, X축)·깊이(Depth, Z축)로 지정.
///
/// API:
///   GetGoalSpawnRange()    → SpawnRange (골존용)
///   GetPursuerSpawnRange() → SpawnRange (추적자용)
///   GetEvaderSpawnRange()  → SpawnRange (도망자용)
///   GetRandomPosition()    → RangeMode에 따라 원형 또는 직사각형 내 랜덤 위치
///
/// </summary>
public class SpawnCenter : MonoBehaviour
{
    // ───────── 스폰 범위 데이터 ───────────────────────────────────────────
    [Serializable]
    public struct SpawnRange
    {
        [Tooltip("최소 높이 (Y축)")]
        public float MinY;
        [Tooltip("최대 높이 (Y축)")]
        public float MaxY;
        [Tooltip("XZ 평면 반경 (RangeMode.Radius 시 사용)")]
        public float Radius;
        [Tooltip("X축 반폭 (RangeMode.Rectangle 시 사용)")]
        public float Width;
        [Tooltip("Z축 반폭 (RangeMode.Rectangle 시 사용)")]
        public float Depth;

        /// <summary>Radius 모드용 생성자</summary>
        public SpawnRange(float minY, float maxY, float radius)
        {
            MinY   = minY;
            MaxY   = maxY;
            Radius = Mathf.Max(0f, radius);
            Width  = radius;
            Depth  = radius;
            ValidateMinMax(ref MinY, ref MaxY);
        }

        /// <summary>Rectangle 모드용 생성자</summary>
        public SpawnRange(float minY, float maxY, float width, float depth)
        {
            MinY   = minY;
            MaxY   = maxY;
            Radius = Mathf.Max(width, depth);
            Width  = Mathf.Max(0f, width);
            Depth  = Mathf.Max(0f, depth);
            ValidateMinMax(ref MinY, ref MaxY);
        }

        /// <summary>minY > maxY이면 자동 스왑</summary>
        public static void ValidateMinMax(ref float minY, ref float maxY)
        {
            if (minY > maxY)
            {
                float tmp = minY;
                minY = maxY;
                maxY = tmp;
            }
        }
    }

    // ───────── 동기화 모드 ────────────────────────────────────────────────
    public enum SyncMode
    {
        [Tooltip("골존·추적자·도망자가 동일한 범위를 공유")]
        Synchronized,
        [Tooltip("골존·추적자·도망자 각각 별도 범위")]
        Desynchronized,
    }

    // ───────── 범위 형태 모드 ─────────────────────────────────────────────
    public enum RangeMode
    {
        [Tooltip("XZ 평면을 원형 반경으로 지정")]
        Radius,
        [Tooltip("XZ 평면을 너비(Width)·깊이(Depth)로 지정")]
        Rectangle,
    }

    // ───────── 자동 등록 (Static) ─────────────────────────────────────────
    /// <summary>현재 씬에서 활성화된 SpawnCenter. Enable 시 자동 등록.</summary>
    public static SpawnCenter Current { get; private set; }

    // ───────── Inspector 설정 ─────────────────────────────────────────────
    [Header("Spawn Center — City Sync")]
    [Tooltip("활성화 시 ComputeSpawn() 호출 시 도시 메타데이터로 SpawnRange를 자동 동기화")]
    [SerializeField] private bool _autoSyncFromCity = false;

    [Header("Spawn Center — Mode")]
    [SerializeField] private SyncMode  _syncMode  = SyncMode.Synchronized;
    [SerializeField] private RangeMode _rangeMode = RangeMode.Rectangle;

    [Header("Synchronized — 공용 범위")]
    [SerializeField] private SpawnRange _sharedRange = new SpawnRange(5f, 25f, 50f);

    [Header("Desynchronized — 골존 범위")]
    [SerializeField] private SpawnRange _goalRange = new SpawnRange(0f, 30f, 50f);

    [Header("Desynchronized — 추적자 범위")]
    [SerializeField] private SpawnRange _pursuerRange = new SpawnRange(5f, 25f, 50f);

    [Header("Desynchronized — 도망자 범위")]
    [SerializeField] private SpawnRange _evaderRange = new SpawnRange(5f, 25f, 50f);

    // ───────── 프로퍼티 ───────────────────────────────────────────────────
    public SyncMode  Mode      => _syncMode;
    public RangeMode ShapeMode => _rangeMode;
    public bool AutoSyncFromCity { get => _autoSyncFromCity; set => _autoSyncFromCity = value; }

    // ───────── 초기화 ─────────────────────────────────────────────────────
    private void OnEnable()
    {
        if (Current != null && Current != this)
            Debug.LogWarning($"[SpawnCenter] '{Current.name}'이 이미 등록되어 있습니다. '{name}'으로 교체합니다.", this);
        Current = this;
    }

    private void OnDisable()
    {
        if (Current == this) Current = null;
    }

    private void OnValidate()
    {
        ValidateRange(ref _sharedRange);
        ValidateRange(ref _goalRange);
        ValidateRange(ref _pursuerRange);
        ValidateRange(ref _evaderRange);
    }

    private static void ValidateRange(ref SpawnRange range)
    {
        SpawnRange.ValidateMinMax(ref range.MinY, ref range.MaxY);
        range.Radius = Mathf.Max(0f, range.Radius);
        range.Width  = Mathf.Max(0f, range.Width);
        range.Depth  = Mathf.Max(0f, range.Depth);
    }

    // ───────── Public API ─────────────────────────────────────────────────

    /// <summary>중점 월드 위치</summary>
    public Vector3 GetCenter() => transform.position;

    /// <summary>골존 스폰 범위 반환</summary>
    public SpawnRange GetGoalSpawnRange()
        => _syncMode == SyncMode.Synchronized ? _sharedRange : _goalRange;

    /// <summary>추적자 스폰 범위 반환</summary>
    public SpawnRange GetPursuerSpawnRange()
        => _syncMode == SyncMode.Synchronized ? _sharedRange : _pursuerRange;

    /// <summary>도망자 스폰 범위 반환</summary>
    public SpawnRange GetEvaderSpawnRange()
        => _syncMode == SyncMode.Synchronized ? _sharedRange : _evaderRange;

    /// <summary>
    /// RangeMode에 따라 랜덤 위치를 생성한다.
    ///   Radius    : XZ 원형 영역 내 균등 분포
    ///   Rectangle : XZ 직사각형 영역 내 균등 분포
    /// </summary>
    public Vector3 GetRandomPosition(SpawnRange range)
    {
        Vector3 center = GetCenter();
        float   y      = UnityEngine.Random.Range(range.MinY, range.MaxY);

        if (_rangeMode == RangeMode.Radius)
        {
            Vector2 offset = UnityEngine.Random.insideUnitCircle * range.Radius;
            return center + new Vector3(offset.x, y, offset.y);
        }
        else // Rectangle
        {
            float x = UnityEngine.Random.Range(-range.Width, range.Width);
            float z = UnityEngine.Random.Range(-range.Depth, range.Depth);
            return center + new Vector3(x, y, z);
        }
    }

    // ───────── City Metadata 동기화 ──────────────────────────────────────

    /// <summary>
    /// CityDataAPI에서 CityMetadata를 조회하여 SpawnRange를 자동 설정한다.
    /// 메타데이터가 없으면 기존 SpawnRange를 유지하고 경고를 출력한다.
    /// </summary>
    public void SyncFromCityMetadata()
    {
        if (CityDataAPI.Instance == null || !CityDataAPI.Instance.HasCityMetadata())
        {
            Debug.LogWarning("[SpawnCenter] CityMetadata가 없습니다. 기존 SpawnRange를 유지합니다.", this);
            return;
        }

        CityMetadata metadata = CityDataAPI.Instance.GetCityMetadata();
        Bounds cityBounds = metadata.cityBounds;

        // cityBounds 기반 SpawnRange Width/Depth/Radius 자동 설정
        float width  = cityBounds.extents.x;
        float depth  = cityBounds.extents.z;
        float radius = Mathf.Max(width, depth);

        // 건물 높이 범위 기반 MinY/MaxY 자동 설정
        float minY = metadata.minBuildingHeight;
        float maxY = metadata.maxBuildingHeight;

        // 공용 범위 설정
        _sharedRange.Width  = width;
        _sharedRange.Depth  = depth;
        _sharedRange.Radius = radius;
        _sharedRange.MinY   = minY;
        _sharedRange.MaxY   = maxY;

        // Desynchronized 모드용 개별 범위도 동기화
        _goalRange.Width    = width;
        _goalRange.Depth    = depth;
        _goalRange.Radius   = radius;
        _goalRange.MinY     = minY;
        _goalRange.MaxY     = maxY;

        _pursuerRange.Width  = width;
        _pursuerRange.Depth  = depth;
        _pursuerRange.Radius = radius;
        _pursuerRange.MinY   = minY;
        _pursuerRange.MaxY   = maxY;

        _evaderRange.Width  = width;
        _evaderRange.Depth  = depth;
        _evaderRange.Radius = radius;
        _evaderRange.MinY   = minY;
        _evaderRange.MaxY   = maxY;

        // Transform 위치를 cityBounds.center로 이동
        transform.position = cityBounds.center;
    }

    // ───────── Editor 디버그 기즈모 ───────────────────────────────────────
#if UNITY_EDITOR
    private void OnDrawGizmosSelected()
    {
        Vector3 center = transform.position;

        if (_syncMode == SyncMode.Synchronized)
        {
            DrawRangeGizmo(center, _sharedRange, new Color(1f, 1f, 1f, 0.8f), "Shared");
        }
        else
        {
            DrawRangeGizmo(center, _goalRange,   new Color(1f, 1f, 0f, 0.6f),   "Goal");
            DrawRangeGizmo(center, _pursuerRange, new Color(1f, 0.3f, 0.3f, 0.6f), "Pursuer");
            DrawRangeGizmo(center, _evaderRange,  new Color(0.3f, 0.5f, 1f, 0.6f), "Evader");
        }

        // 중심점 마커
        Gizmos.color = new Color(1f, 1f, 1f, 0.9f);
        Gizmos.DrawSphere(center, 0.5f);
    }

    private void DrawRangeGizmo(Vector3 center, SpawnRange range, Color color, string label)
    {
        Gizmos.color = color;

        Vector3 minCenter = center + Vector3.up * range.MinY;
        Vector3 maxCenter = center + Vector3.up * range.MaxY;

        if (_rangeMode == RangeMode.Radius)
        {
            DrawGizmoCircleXZ(minCenter, range.Radius, 48);
            DrawGizmoCircleXZ(maxCenter, range.Radius, 48);

            // 수직 연결선 (4방향)
            for (int i = 0; i < 4; i++)
            {
                float   angle  = i * 90f * Mathf.Deg2Rad;
                Vector3 offset = new Vector3(Mathf.Cos(angle) * range.Radius, 0f, Mathf.Sin(angle) * range.Radius);
                Gizmos.DrawLine(minCenter + offset, maxCenter + offset);
            }

            UnityEditor.Handles.color = color;
            UnityEditor.Handles.Label(maxCenter + Vector3.up * 0.5f,
                $"{label}\nR={range.Radius:F1} Y=[{range.MinY:F1}, {range.MaxY:F1}]");
        }
        else // Rectangle
        {
            DrawGizmoRectXZ(minCenter, range.Width, range.Depth);
            DrawGizmoRectXZ(maxCenter, range.Width, range.Depth);

            // 수직 연결선 (4모서리)
            Vector3[] corners = GetRectCorners(range.Width, range.Depth);
            foreach (Vector3 c in corners)
                Gizmos.DrawLine(minCenter + c, maxCenter + c);

            UnityEditor.Handles.color = color;
            UnityEditor.Handles.Label(maxCenter + Vector3.up * 0.5f,
                $"{label}\nW={range.Width:F1} D={range.Depth:F1} Y=[{range.MinY:F1}, {range.MaxY:F1}]");
        }
    }

    private static void DrawGizmoCircleXZ(Vector3 center, float radius, int segments)
    {
        float   step = 360f / segments;
        Vector3 prev = center + new Vector3(radius, 0f, 0f);
        for (int i = 1; i <= segments; i++)
        {
            float   angle = i * step * Mathf.Deg2Rad;
            Vector3 next  = center + new Vector3(Mathf.Cos(angle) * radius, 0f, Mathf.Sin(angle) * radius);
            Gizmos.DrawLine(prev, next);
            prev = next;
        }
    }

    private static void DrawGizmoRectXZ(Vector3 center, float halfW, float halfD)
    {
        Vector3 a = center + new Vector3(-halfW, 0f, -halfD);
        Vector3 b = center + new Vector3( halfW, 0f, -halfD);
        Vector3 c = center + new Vector3( halfW, 0f,  halfD);
        Vector3 d = center + new Vector3(-halfW, 0f,  halfD);
        Gizmos.DrawLine(a, b);
        Gizmos.DrawLine(b, c);
        Gizmos.DrawLine(c, d);
        Gizmos.DrawLine(d, a);
    }

    private static Vector3[] GetRectCorners(float halfW, float halfD)
    {
        return new[]
        {
            new Vector3(-halfW, 0f, -halfD),
            new Vector3( halfW, 0f, -halfD),
            new Vector3( halfW, 0f,  halfD),
            new Vector3(-halfW, 0f,  halfD),
        };
    }
#endif
}
