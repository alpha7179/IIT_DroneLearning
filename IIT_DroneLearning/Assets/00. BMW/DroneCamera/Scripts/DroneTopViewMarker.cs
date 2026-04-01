using UnityEngine;

namespace DroneCamera
{
    /// <summary>
    /// 드론에 부착하면 탑뷰 식별용 마커를 최상위 하이라키에 자동 생성한다.
    ///
    /// ■ 동작 원리
    ///   - Start()  : markerPrefab을 Instantiate → parent = null (최상위 오브젝트)
    ///   - LateUpdate() : 마커의 글로벌 Position = (drone.x, markerHeight, drone.z)
    ///                    Rotation = Quaternion.identity  (PID 기울기 무영향)
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

        [Header("마커 이름 (선택)")]
        [Tooltip("비워두면 'DroneMarker_<GameObject명>' 으로 자동 설정")]
        public string markerName = "";

        private GameObject _markerInstance;

        private void Start()
        {
            if (markerPrefab == null)
            {
                Debug.LogWarning($"[DroneTopViewMarker] markerPrefab이 할당되지 않았습니다. ({gameObject.name})");
                return;
            }

            // 초기 위치: 드론의 XZ, 지정된 글로벌 Y
            Vector3 spawnPos = new Vector3(transform.position.x, markerHeight, transform.position.z);

            // 최상위 하이라키에 생성 (parent = null)
            _markerInstance = Instantiate(markerPrefab, spawnPos, Quaternion.identity, null);

            string instanceName = string.IsNullOrEmpty(markerName)
                ? $"DroneMarker_{gameObject.name}"
                : markerName;
            _markerInstance.name = instanceName;
        }

        private void LateUpdate()
        {
            if (_markerInstance == null) return;

            // X/Z는 드론을 따라가고, Y는 고정 글로벌 값 유지
            // Rotation은 항상 identity → PID 기울기 영향 없음
            _markerInstance.transform.SetPositionAndRotation(
                new Vector3(transform.position.x, markerHeight, transform.position.z),
                Quaternion.identity
            );
        }

        private void OnDestroy()
        {
            if (_markerInstance != null)
                Destroy(_markerInstance);
        }
    }
}
