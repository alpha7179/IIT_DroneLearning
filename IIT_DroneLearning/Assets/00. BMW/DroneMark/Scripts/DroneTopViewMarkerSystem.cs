using UnityEngine;
using System.Collections.Generic;

namespace DroneCamera
{
    /// <summary>
    /// 탑뷰 드론 마커 통합 매니저 — 씬의 빈 GameObject 하나에 부착한다.
    ///
    /// ■ 기존 DroneTopViewMarker (드론별 개별 컴포넌트) 대체
    ///   - 모든 드론의 마커 위치를 이 컴포넌트 하나에서 일괄 계산한다.
    ///   - Pursuer 리스트 + Pursuer 마커 프리팹, Evader 리스트 + Evader 마커 프리팹을 분리 관리한다.
    ///   - 드론 목록을 비워두면 Start 시점에 태그로 자동 탐지한다.
    ///
    /// ■ 성능 최적화
    ///   - updateInterval (초): 0이면 매 LateUpdate, 양수면 해당 주기마다만 재계산.
    ///   - moveThreshold (m): XZ 이동량이 threshold 미만이면 해당 드론 마커 갱신 스킵.
    ///
    /// ■ 사용 방법
    ///   1. 빈 GameObject 생성 → DroneTopViewMarkerSystem 추가.
    ///   2. Pursuer/Evader 마커 프리팹 할당.
    ///   3. 드론 목록을 비워두면 태그로 자동 탐지, 직접 지정하려면 Inspector에서 입력.
    ///   4. 기존 드론 GameObject의 DroneTopViewMarker 컴포넌트는 제거한다.
    ///
    /// 담당: 배민우 (00. BMW)
    /// </summary>
    public class DroneTopViewMarkerSystem : MonoBehaviour
    {
        // ───────── 싱글톤 ─────────────────────────────────────────────────
        public static DroneTopViewMarkerSystem Instance { get; private set; }

        // ───────── 자동 탐지 태그 ─────────────────────────────────────────
        [Header("자동 탐지 — Tags (드론 목록이 비어있을 때 사용)")]
        [SerializeField] private string _pursuerTag = "Pursuer";
        [SerializeField] private string _evaderTag  = "Evader";

        // ───────── 마커 프리팹 ────────────────────────────────────────────
        [Header("마커 프리팹")]
        [Tooltip("Pursuer 드론용 탑뷰 마커 프리팹")]
        [SerializeField] private GameObject _pursuerMarkerPrefab;
        [Tooltip("Evader 드론용 탑뷰 마커 프리팹")]
        [SerializeField] private GameObject _evaderMarkerPrefab;

        // ───────── 드론 목록 (수동 지정 / 빈 경우 태그 자동 탐지) ─────────
        [Header("드론 목록 (비워두면 Start 시 태그로 자동 탐지)")]
        [SerializeField] private List<GameObject> _pursuerDrones = new List<GameObject>();
        [SerializeField] private List<GameObject> _evaderDrones  = new List<GameObject>();

        // ───────── 마커 설정 ──────────────────────────────────────────────
        [Header("마커 설정")]
        [Tooltip("마커가 고정될 글로벌 Y 높이. TopView 카메라보다 낮아야 보임.")]
        [SerializeField] private float _markerHeight = 90f;
        [Tooltip("XZ 이동량이 이 값 미만이면 해당 드론의 마커 갱신을 스킵 (0 = 항상 갱신)")]
        [SerializeField] private float _moveThreshold = 0.05f;

        // ───────── 업데이트 주기 ──────────────────────────────────────────
        [Header("업데이트 주기")]
        [Tooltip("마커 재계산 간격 (초). 0이면 매 LateUpdate 갱신, 양수면 해당 주기마다만 갱신.")]
        [SerializeField] private float _updateInterval = 0f;

        // ───────── Debug (읽기 전용) ──────────────────────────────────────
        // Unity Inspector에서만 읽히는 필드 → C# 분석기 경고 억제
#pragma warning disable 0414
        [Header("Debug — 읽기 전용")]
        [SerializeField] private int _debugPursuerCount;
        [SerializeField] private int _debugEvaderCount;
#pragma warning restore 0414

        // ───────── 내부 상태 ──────────────────────────────────────────────
        private struct Entry
        {
            public Transform  drone;
            public GameObject marker;
            public Vector2    lastXZ;
        }

        private readonly List<Entry> _pursuerEntries = new List<Entry>();
        private readonly List<Entry> _evaderEntries  = new List<Entry>();

        private float _sqrThreshold;
        private float _nextUpdateTime;

        // ───────── 초기화 ─────────────────────────────────────────────────
        private void Awake()
        {
            if (Instance != null && Instance != this)
            {
                Debug.LogWarning($"[DroneTopViewMarkerSystem] 중복 인스턴스가 감지됐습니다. '{name}'을 비활성화합니다.", this);
                enabled = false;
                return;
            }
            Instance = this;
        }

        private void Start()
        {
            _sqrThreshold = _moveThreshold * _moveThreshold;

            // [SerializeField] 리스트를 직접 수정하지 않고 로컬 소스로만 사용한다.
            // 수동 지정이 없으면 태그로 탐지한 결과를 임시 리스트로 받는다.
            var pursuerSources = _pursuerDrones.Count > 0 ? _pursuerDrones : FindByTag(_pursuerTag);
            var evaderSources  = _evaderDrones.Count  > 0 ? _evaderDrones  : FindByTag(_evaderTag);

            int markLayer = LayerMask.NameToLayer("Mark");

            foreach (var go in pursuerSources)
                if (go != null)
                    _pursuerEntries.Add(CreateEntry(go, _pursuerMarkerPrefab, markLayer, "Pursuer"));

            foreach (var go in evaderSources)
                if (go != null)
                    _evaderEntries.Add(CreateEntry(go, _evaderMarkerPrefab, markLayer, "Evader"));

            _debugPursuerCount = _pursuerEntries.Count;
            _debugEvaderCount  = _evaderEntries.Count;

            if (_pursuerEntries.Count == 0 && _evaderEntries.Count == 0)
                Debug.LogWarning("[DroneTopViewMarkerSystem] 등록된 드론이 없습니다. 태그 또는 드론 목록을 확인하세요.", this);

            // 에피소드 스폰 완료 이벤트 구독 — 텔레포트 후 마커를 즉시 강제 갱신하기 위함
            if (EpisodeSpawnCoordinator.Instance != null)
                EpisodeSpawnCoordinator.Instance.OnSpawnComputed += ForceUpdateAllMarkers;
            else
                Debug.LogWarning("[DroneTopViewMarkerSystem] EpisodeSpawnCoordinator를 찾을 수 없습니다. 에피소드 리셋 후 마커 즉시 갱신이 비활성화됩니다.", this);
        }

        private void OnDestroy()
        {
            // 이벤트 구독 해제 (OnDestroy 시점에 Instance가 이미 null일 수 있으므로 null 체크)
            if (EpisodeSpawnCoordinator.Instance != null)
                EpisodeSpawnCoordinator.Instance.OnSpawnComputed -= ForceUpdateAllMarkers;

            DestroyMarkers(_pursuerEntries);
            DestroyMarkers(_evaderEntries);
            if (Instance == this) Instance = null;
        }

        private void OnValidate()
        {
            _moveThreshold  = Mathf.Max(0f, _moveThreshold);
            _updateInterval = Mathf.Max(0f, _updateInterval);
        }

        // ───────── LateUpdate ─────────────────────────────────────────────

        private void LateUpdate()
        {
            // 주기 제한: updateInterval > 0이면 해당 시간이 지났을 때만 갱신
            if (_updateInterval > 0f)
            {
                if (Time.time < _nextUpdateTime) return;
                _nextUpdateTime = Time.time + _updateInterval;
            }

            UpdateEntries(_pursuerEntries);
            UpdateEntries(_evaderEntries);
        }

        // ───────── 헬퍼 ──────────────────────────────────────────────────

        /// <summary>
        /// 태그로 씬 내 GameObject를 탐색하여 새 리스트로 반환한다.
        /// [SerializeField] 리스트를 수정하지 않기 위해 로컬 리스트로 반환한다.
        /// </summary>
        private static List<GameObject> FindByTag(string tag)
        {
            if (string.IsNullOrEmpty(tag)) return new List<GameObject>();
            return new List<GameObject>(GameObject.FindGameObjectsWithTag(tag));
        }

        /// <summary>
        /// 에피소드 스폰 완료 시 호출된다.
        /// 모든 Entry의 lastXZ를 무효화하여 다음 LateUpdate에서 마커 위치를 즉시 강제 갱신한다.
        /// </summary>
        private void ForceUpdateAllMarkers()
        {
            ResetLastXZ(_pursuerEntries);
            ResetLastXZ(_evaderEntries);
        }

        private static void ResetLastXZ(List<Entry> entries)
        {
            for (int i = 0; i < entries.Count; i++)
            {
                Entry e = entries[i];
                e.lastXZ   = new Vector2(float.MaxValue, float.MaxValue); // threshold 강제 통과
                entries[i] = e;
            }
        }

        private Entry CreateEntry(GameObject droneGO, GameObject prefab, int markLayer, string role)
        {
            Vector3 pos = droneGO.transform.position;
            GameObject marker = null;

            if (prefab != null)
            {
                marker = Instantiate(
                    prefab,
                    new Vector3(pos.x, _markerHeight, pos.z),
                    Quaternion.identity,
                    null); // 최상위 하이라키

                marker.name = $"TopMark_{role}_{droneGO.name}";

                if (markLayer >= 0)
                    SetLayerRecursive(marker, markLayer);
            }
            else
            {
                Debug.LogWarning($"[DroneTopViewMarkerSystem] {role} 마커 프리팹이 없습니다. ({droneGO.name})", this);
            }

            return new Entry
            {
                drone  = droneGO.transform,
                marker = marker,
                lastXZ = new Vector2(pos.x, pos.z),
            };
        }

        private void UpdateEntries(List<Entry> entries)
        {
            for (int i = 0; i < entries.Count; i++)
            {
                Entry e = entries[i];
                if (e.drone == null || e.marker == null) continue;

                Vector3 pos       = e.drone.position;
                Vector2 currentXZ = new Vector2(pos.x, pos.z);

                if (_sqrThreshold > 0f &&
                    (currentXZ - e.lastXZ).sqrMagnitude < _sqrThreshold)
                    continue;

                e.lastXZ  = currentXZ;
                entries[i] = e; // struct이므로 리스트에 다시 대입

                e.marker.transform.SetPositionAndRotation(
                    new Vector3(pos.x, _markerHeight, pos.z),
                    Quaternion.identity);
            }
        }

        private static void DestroyMarkers(List<Entry> entries)
        {
            foreach (var e in entries)
                if (e.marker != null)
                    Destroy(e.marker);
            entries.Clear();
        }

        private static void SetLayerRecursive(GameObject obj, int layer)
        {
            obj.layer = layer;
            foreach (Transform child in obj.transform)
                SetLayerRecursive(child.gameObject, layer);
        }

        // ───────── Public API (런타임 드론 추가/제거) ─────────────────────

        /// <summary>런타임에 Pursuer 드론을 마커 시스템에 추가한다.</summary>
        public void RegisterPursuer(GameObject droneGO)
        {
            int markLayer = LayerMask.NameToLayer("Mark");
            _pursuerEntries.Add(CreateEntry(droneGO, _pursuerMarkerPrefab, markLayer, "Pursuer"));
            _debugPursuerCount = _pursuerEntries.Count;
        }

        /// <summary>런타임에 Evader 드론을 마커 시스템에 추가한다.</summary>
        public void RegisterEvader(GameObject droneGO)
        {
            int markLayer = LayerMask.NameToLayer("Mark");
            _evaderEntries.Add(CreateEntry(droneGO, _evaderMarkerPrefab, markLayer, "Evader"));
            _debugEvaderCount = _evaderEntries.Count;
        }

        /// <summary>해당 드론의 마커를 제거하고 목록에서 삭제한다.</summary>
        public void Unregister(GameObject droneGO)
        {
            RemoveFrom(_pursuerEntries, droneGO);
            RemoveFrom(_evaderEntries, droneGO);
            _debugPursuerCount = _pursuerEntries.Count;
            _debugEvaderCount  = _evaderEntries.Count;
        }

        private static void RemoveFrom(List<Entry> entries, GameObject droneGO)
        {
            for (int i = entries.Count - 1; i >= 0; i--)
            {
                if (entries[i].drone == null || entries[i].drone.gameObject == droneGO)
                {
                    if (entries[i].marker != null)
                        Destroy(entries[i].marker);
                    entries.RemoveAt(i);
                }
            }
        }
    }
}
