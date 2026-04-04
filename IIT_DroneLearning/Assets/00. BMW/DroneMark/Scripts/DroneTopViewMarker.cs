using UnityEngine;

namespace DroneCamera
{
    /// <summary>
    /// 드론에 부착하면 탑뷰 식별용 마커를 최상위 하이라키에 자동 생성한다.
    ///
    /// ■ 동작 원리
    ///   - Start()      : markerPrefab을 Instantiate → parent = null (최상위 오브젝트)
    ///   - LateUpdate() : 드론 XZ가 moveThreshold 이상 이동했을 때만 마커 위치 갱신
    ///                    Y는 markerHeight 고정, Rotation은 항상 identity (PID 기울기 무영향)
    ///   - OnDestroy()  : 마커 오브젝트 자동 제거
    /// </summary>
    public class DroneTopViewMarker : MonoBehaviour
    {
        [Header("마커 프리팹")]
        [Tooltip("탑뷰에서 드론 위치를 나타낼 프리팹 (Mark.prefab 등)")]
        public GameObject markerPrefab;

        [Header("마커 위치")]
        [Tooltip("마커가 고정될 글로벌 Y 높이. TopView 카메라(기본 Y=100)보다 낮아야 보임.")]
        public float markerHeight = 90f;

        [Header("업데이트 최적화")]
        [Tooltip("XZ 이동량이 이 값 미만이면 마커 위치를 갱신하지 않음 (0 = 매 프레임 갱신)")]
        public float moveThreshold = 0.05f;

        [Header("마커 이름 (선택)")]
        [Tooltip("비워두면 'DroneMarker_<GameObject명>' 으로 자동 설정")]
        public string markerName = "";

        private GameObject _markerInstance;
        private float _sqrThreshold;
        private Vector2 _lastXZ;   // 마지막으로 마커를 갱신했을 때의 드론 XZ

        private void Start()
        {
            if (markerPrefab == null)
            {
                Debug.LogWarning($"[DroneTopViewMarker] markerPrefab이 할당되지 않았습니다. ({gameObject.name})");
                return;
            }

            _sqrThreshold = moveThreshold * moveThreshold;

            Vector3 pos = transform.position;
            _lastXZ = new Vector2(pos.x, pos.z);

            // 최상위 하이라키에 생성 (parent = null)
            _markerInstance = Instantiate(
                markerPrefab,
                new Vector3(pos.x, markerHeight, pos.z),
                Quaternion.identity,
                null);

            _markerInstance.name = string.IsNullOrEmpty(markerName)
                ? $"DroneMarker_{gameObject.name}"
                : markerName;

            // 프리팹 레이어와 무관하게 마커 전체를 "Mark" 레이어로 강제 설정
            int markLayer = LayerMask.NameToLayer("Mark");
            if (markLayer >= 0)
                SetLayerRecursive(_markerInstance, markLayer);
        }

        private void LateUpdate()
        {
            if (_markerInstance == null) return;

            Vector3 pos = transform.position;
            Vector2 currentXZ = new Vector2(pos.x, pos.z);

            // threshold 이상 이동했을 때만 갱신
            if (_sqrThreshold > 0f &&
                (currentXZ - _lastXZ).sqrMagnitude < _sqrThreshold)
                return;

            _lastXZ = currentXZ;
            _markerInstance.transform.SetPositionAndRotation(
                new Vector3(pos.x, markerHeight, pos.z),
                Quaternion.identity);
        }

        private void OnDestroy()
        {
            if (_markerInstance != null)
                Destroy(_markerInstance);
        }

        private static void SetLayerRecursive(GameObject obj, int layer)
        {
            obj.layer = layer;
            foreach (Transform child in obj.transform)
                SetLayerRecursive(child.gameObject, layer);
        }
    }
}
