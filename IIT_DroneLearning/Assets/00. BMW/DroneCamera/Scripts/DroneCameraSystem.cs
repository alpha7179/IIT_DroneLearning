using System;
using System.Reflection;
using UnityEngine;
using BMW.DroneSensor;

namespace BMW.DroneCamera
{
    /// <summary>
    /// 드론 역할 — BMW.DroneCamera 로컬 사본.
    /// Assembly-CSharp(DroneAgent)를 named assembly에서 직접 참조할 수 없으므로 미러링.
    /// </summary>
    public enum DroneRole { Pursuer, Evader }

    /// <summary>
    /// 드론 GameObject에 부착되는 FPV 카메라 컴포넌트.
    /// 드론의 자식 GameObject로 Camera를 생성하고, CameraControlSystem에 등록한다.
    ///
    /// 전방 방향 결정:
    ///   - DroneSensorSystem 존재 시: Middle/N 레이 방향(로컬 +Z)과 자동 정렬됨 (Quaternion.identity)
    ///   - DroneSensorSystem 미존재 시: 드론 Transform 로컬 +Z축 사용 (동일하게 Quaternion.identity)
    /// 카메라는 드론의 자식 Transform이므로 드론 회전 시 자동 추종한다.
    /// </summary>
    public class DroneCameraSystem : MonoBehaviour
    {
        [Header("카메라 설정")]
        [Tooltip("카메라 오프셋 위치 (드론 로컬 좌표)")]
        public Vector3 cameraOffset = new Vector3(0f, 0.1f, 0.2f);

        [Tooltip("카메라 Field of View")]
        public float fieldOfView = 60f;

        [Tooltip("카메라 Near Clip Plane")]
        public float nearClipPlane = 0.1f;

        [Tooltip("카메라 Far Clip Plane")]
        public float farClipPlane = 500f;

        /// <summary>생성된 Camera 컴포넌트 참조 (read-only)</summary>
        public Camera Camera { get; private set; }

        // DroneSensorSystem 선택적 참조
        private DroneSensorSystem _sensorSystem;

        // ────────────────────────────────────────────
        // Unity 생명주기
        // ────────────────────────────────────────────

        private void Awake()
        {
            // DroneSensorSystem 선택적 탐색
            _sensorSystem = GetComponent<DroneSensorSystem>();

            // 자식 GameObject 생성
            var cameraGO = new GameObject("DroneCamera");
            cameraGO.transform.SetParent(transform, false);
            cameraGO.transform.localPosition = cameraOffset;

            // 전방 방향 설정
            // DroneSensorSystem의 Middle/N 로컬 방향 = (0,0,1) = 드론 로컬 +Z
            // localRotation = identity 로 두면 두 경우 모두 자동 만족
            cameraGO.transform.localRotation = Quaternion.identity;

            // Camera 컴포넌트 추가 및 설정
            Camera = cameraGO.AddComponent<Camera>();
            Camera.fieldOfView = fieldOfView;
            Camera.nearClipPlane = nearClipPlane;
            Camera.farClipPlane = farClipPlane;
        }

        private void Start()
        {
            // CameraControlSystem 탐색 및 등록
            var controlSystem = FindFirstObjectByType<CameraControlSystem>();
            if (controlSystem == null)
            {
                Debug.LogWarning("[DroneCameraSystem] CameraControlSystem을 찾을 수 없습니다. 독립 동작합니다.");
                return;
            }
            controlSystem.RegisterDroneCamera(this, ResolveRole());
        }

        private void OnDestroy()
        {
            // CameraControlSystem에서 등록 해제
            var controlSystem = FindFirstObjectByType<CameraControlSystem>();
            controlSystem?.UnregisterDroneCamera(this);
        }

        // ────────────────────────────────────────────
        // 공개 API
        // ────────────────────────────────────────────

        /// <summary>카메라 활성/비활성 전환</summary>
        public void SetEnabled(bool enabled)
        {
            if (Camera != null) Camera.enabled = enabled;
        }

        /// <summary>Viewport Rect 설정</summary>
        public void SetViewportRect(Rect rect)
        {
            if (Camera != null) Camera.rect = rect;
        }

        // ────────────────────────────────────────────
        // 내부 헬퍼
        // ────────────────────────────────────────────

        /// <summary>
        /// DroneAgent.Role을 리플렉션으로 읽어 로컬 DroneRole로 변환.
        /// Assembly-CSharp 참조 불가로 인한 리플렉션 사용.
        /// </summary>
        private DroneRole ResolveRole()
        {
            foreach (var comp in GetComponents<MonoBehaviour>())
            {
                if (comp == null || comp.GetType().Name != "DroneAgent") continue;

                var roleField = comp.GetType().GetField(
                    "Role", BindingFlags.Public | BindingFlags.Instance);
                if (roleField == null) break;

                // DroneRole enum: Pursuer = 0, Evader = 1
                int roleValue = Convert.ToInt32(roleField.GetValue(comp));
                return roleValue == 1 ? DroneRole.Evader : DroneRole.Pursuer;
            }
            return DroneRole.Pursuer; // DroneAgent 미존재 시 기본값
        }
    }
}
