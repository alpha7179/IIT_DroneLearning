using UnityEngine;
using System;

/// <summary>
/// SpawnCenter — 중점 기반 스폰 범위 관리 컴포넌트
///
/// 빈 GameObject에 부착하여 골존·추적자·도망자의 스폰 범위를 중앙 집중 관리한다.
///
/// 모드:
///   Synchronized   : 세 객체가 동일한 범위(minY, maxY, Radius)를 공유. 기즈모 1개.
///   Desynchronized : 골존·추적자·도망자 각각 별도 범위. 기즈모 3개.
///
/// API:
///   GetGoalSpawnRange()    → SpawnRange (골존용)
///   GetPursuerSpawnRange() → SpawnRange (추적자용)
///   GetEvaderSpawnRange()  → SpawnRange (도망자용)
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
        [Tooltip("XZ 평면 반경")]
        public float Radius;

        public SpawnRange(float minY, float maxY, float radius)
        {
            MinY   = minY;
            MaxY   = maxY;
            Radius = Mathf.Max(0f, radius);
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

    // ───────── 자동 등록 (Static) ─────────────────────────────────────────
    /// <summary>현재 씬에서 활성화된 SpawnCenter. Enable 시 자동 등록.</summary>
    public static SpawnCenter Current { get; private set; }

    // ───────── Inspector 설정 ─────────────────────────────────────────────
    [Header("Spawn Center — Mode")]
    [SerializeField] private SyncMode _syncMode = SyncMode.Synchronized;

    [Header("Synchronized — 공용 범위")]
    [SerializeField] private SpawnRange _sharedRange = new SpawnRange(0f, 10f, 15f);

    [Header("Desynchronized — 골존 범위")]
    [SerializeField] private SpawnRange _goalRange = new SpawnRange(0f, 5f, 20f);

    [Header("Desynchronized — 추적자 범위")]
    [SerializeField] private SpawnRange _pursuerRange = new SpawnRange(3f, 10f, 15f);

    [Header("Desynchronized — 도망자 범위")]
    [SerializeField] private SpawnRange _evaderRange = new SpawnRange(3f, 10f, 15f);

    // ───────── 프로퍼티 ───────────────────────────────────────────────────
    public SyncMode Mode => _syncMode;

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

    /// <summary>주어진 SpawnRange 기준으로 랜덤 위치 생성</summary>
    public Vector3 GetRandomPosition(SpawnRange range)
    {
        Vector3 center = GetCenter();
        Vector2 offset = UnityEngine.Random.insideUnitCircle * range.Radius;
        float y = UnityEngine.Random.Range(range.MinY, range.MaxY);
        return center + new Vector3(offset.x, y, offset.y);
    }

    // ───────── Editor 디버그 기즈모 ───────────────────────────────────────
#if UNITY_EDITOR
    private void OnDrawGizmosSelected()
    {
        Vector3 center = transform.position;

        if (_syncMode == SyncMode.Synchronized)
        {
            // 공용 범위 — 흰색 1개
            DrawRangeGizmo(center, _sharedRange, new Color(1f, 1f, 1f, 0.8f), "Shared");
        }
        else
        {
            // 골존 — 노란색
            DrawRangeGizmo(center, _goalRange, new Color(1f, 1f, 0f, 0.6f), "Goal");
            // 추적자 — 빨간색
            DrawRangeGizmo(center, _pursuerRange, new Color(1f, 0.3f, 0.3f, 0.6f), "Pursuer");
            // 도망자 — 파란색
            DrawRangeGizmo(center, _evaderRange, new Color(0.3f, 0.5f, 1f, 0.6f), "Evader");
        }

        // 중심점 마커
        Gizmos.color = new Color(1f, 1f, 1f, 0.9f);
        Gizmos.DrawSphere(center, 0.5f);
    }

    private void DrawRangeGizmo(Vector3 center, SpawnRange range, Color color, string label)
    {
        Gizmos.color = color;

        // minY 높이의 원
        Vector3 minCenter = center + Vector3.up * range.MinY;
        DrawGizmoCircleXZ(minCenter, range.Radius, 48);

        // maxY 높이의 원
        Vector3 maxCenter = center + Vector3.up * range.MaxY;
        DrawGizmoCircleXZ(maxCenter, range.Radius, 48);

        // 수직 연결선 (4방향)
        for (int i = 0; i < 4; i++)
        {
            float angle = i * 90f * Mathf.Deg2Rad;
            Vector3 offset = new Vector3(Mathf.Cos(angle) * range.Radius, 0f, Mathf.Sin(angle) * range.Radius);
            Gizmos.DrawLine(minCenter + offset, maxCenter + offset);
        }

        // 라벨
        UnityEditor.Handles.color = color;
        UnityEditor.Handles.Label(maxCenter + Vector3.up * 0.5f,
            $"{label}\nR={range.Radius:F1} Y=[{range.MinY:F1}, {range.MaxY:F1}]");
    }

    private static void DrawGizmoCircleXZ(Vector3 center, float radius, int segments)
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
