using System;
using System.Reflection;
using UnityEngine;

namespace DroneCamera
{
    /// <summary>
    /// 드론 역할 — BMW.DroneCamera 로컬 사본.
    /// Assembly-CSharp(DroneAgent)를 named assembly에서 직접 참조할 수 없으므로 미러링.
    /// </summary>
    public enum DroneRole { Pursuer, Evader }

    /// <summary>
    /// 드론 GameObject에 부착되는 FPV 카메라 컴포넌트.
    /// 카메라를 두 개 생성한다:
    ///   Camera         (SoloCamera)     — 개별 디스플레이 단독 전체 화면
    ///   MultiViewCamera                 — Display 1 다중뷰 분할 화면
    ///
    /// CameraControlSystem 존재 시:
    ///   SoloCamera  → Pursuer: Display 5 (targetDisplay=4) / Evader: Display 4 (targetDisplay=3)
    ///   MultiViewCamera → Display 1 (targetDisplay=0), 분할 viewport
    ///
    /// CameraControlSystem 미존재 시 (독립 동작):
    ///   SoloCamera only → Evader: Display 2 (targetDisplay=1) / Pursuer: Display 3 (targetDisplay=2)
    ///   MultiViewCamera 비활성화
    /// </summary>
    public class DroneCameraSystem : MonoBehaviour
    {
        [Header("카메라 위치")]
        [Tooltip("카메라 좌우·상하 오프셋 (드론 로컬 X·Y 좌표)")]
        public Vector2 cameraLateralOffset = new Vector2(0f, 0f);

        [Tooltip("드론으로부터 카메라까지의 전방 거리 (드론 로컬 +Z)")]
        [Range(0f, 100f)]
        public float cameraDistance = 0.5f;

        [Header("카메라 설정")]
        [Tooltip("카메라 Field of View")]
        [Range(10f, 170f)]
        public float fieldOfView = 60f;

        [Tooltip("카메라 Near Clip Plane")]
        public float nearClipPlane = 0.1f;

        [Tooltip("카메라 Far Clip Plane")]
        public float farClipPlane = 500f;

        /// <summary>개별 디스플레이용 카메라 (단독 전체 화면)</summary>
        public Camera Camera { get; private set; }

        /// <summary>Display 1 다중뷰 분할용 카메라</summary>
        public Camera MultiViewCamera { get; private set; }

        private Transform _cameraTransform;
        private Transform _multiViewTransform;

        // CCS 참조 캐시 (OnValidate 실시간 알림용)
        private CameraControlSystem _controlSystem;

        // ────────────────────────────────────────────
        // Unity 생명주기
        // ────────────────────────────────────────────

        private void Awake()
        {
            // 개별 디스플레이용 카메라
            var soloGO = new GameObject("DroneCamera_Solo");
            soloGO.transform.SetParent(transform, false);
            soloGO.transform.localRotation = Quaternion.identity;
            _cameraTransform = soloGO.transform;
            Camera = soloGO.AddComponent<Camera>();
            ApplyCameraSettings();

            // 다중뷰(Display 1) 분할용 카메라
            var multiGO = new GameObject("DroneCamera_MultiView");
            multiGO.transform.SetParent(transform, false);
            multiGO.transform.localRotation = Quaternion.identity;
            _multiViewTransform = multiGO.transform;
            MultiViewCamera = multiGO.AddComponent<Camera>();
            MultiViewCamera.targetDisplay = 0;
            MultiViewCamera.enabled = false;
            ApplyCameraSettingsTo(MultiViewCamera);

            ApplyCameraPosition();
        }

        private void LateUpdate()
        {
            ApplyCameraPosition();
        }

        /// <summary>인스펙터 값 변경 시 카메라 파라미터 실시간 반영.</summary>
        private void OnValidate()
        {
            if (_cameraTransform != null)   ApplyCameraPosition();
            if (Camera != null)             ApplyCameraSettings();
            if (MultiViewCamera != null)    ApplyCameraSettingsTo(MultiViewCamera);

            // 플레이 모드에서 CCS 레이아웃 재적용
            if (Application.isPlaying && _controlSystem != null)
            {
                int n = _controlSystem.GetActiveBatchCount();
                _controlSystem.ApplyLayout(n);
            }
        }

        private void Start()
        {
            _controlSystem = FindFirstObjectByType<CameraControlSystem>();

            if (_controlSystem == null)
            {
                Debug.LogWarning("[DroneCameraSystem] CameraControlSystem을 찾을 수 없습니다. 독립 동작합니다.");
                // CCS 없음 → SoloCamera만 사용, 역할 기반 디스플레이 할당
                DroneRole role = ResolveRole();
                Camera.targetDisplay = role == DroneRole.Evader ? 1 : 2;
                if (MultiViewCamera != null) MultiViewCamera.enabled = false;
                return;
            }

            _controlSystem.RegisterDroneCamera(this, ResolveRole());
        }

        private void OnDestroy()
        {
            var cs = _controlSystem != null
                ? _controlSystem
                : FindFirstObjectByType<CameraControlSystem>();
            cs?.UnregisterDroneCamera(this);
        }

        // ────────────────────────────────────────────
        // 공개 API — CameraControlSystem에서 호출
        // ────────────────────────────────────────────

        /// <summary>Solo/MultiView 카메라 모두 활성/비활성 전환</summary>
        public void SetEnabled(bool enabled)
        {
            if (Camera != null)         Camera.enabled = enabled;
            if (MultiViewCamera != null) MultiViewCamera.enabled = enabled;
        }

        /// <summary>MultiView 카메라만 활성/비활성 전환 (분할 레이아웃 제어용)</summary>
        public void SetMultiViewEnabled(bool enabled)
        {
            if (MultiViewCamera != null) MultiViewCamera.enabled = enabled;
        }

        /// <summary>Solo 카메라만 활성/비활성 전환 (개별 디스플레이 제어용)</summary>
        public void SetSoloEnabled(bool enabled)
        {
            if (Camera != null) Camera.enabled = enabled;
        }

        /// <summary>MultiView 카메라의 분할 Viewport Rect 설정 (Display 1 다중뷰용)</summary>
        public void SetViewportRect(Rect rect)
        {
            if (MultiViewCamera != null) MultiViewCamera.rect = rect;
        }

        /// <summary>Solo 카메라의 Viewport Rect 설정 (개별 디스플레이 전체 화면용)</summary>
        public void SetSoloViewportRect(Rect rect)
        {
            if (Camera != null) Camera.rect = rect;
        }

        /// <summary>Solo 카메라의 targetDisplay 설정 (Pursuer=4, Evader=3)</summary>
        public void SetSoloDisplay(int display)
        {
            if (Camera != null) Camera.targetDisplay = display;
        }

        /// <summary>MultiView 카메라의 targetDisplay 설정 (항상 0 = Display 1)</summary>
        public void SetMultiViewDisplay(int display)
        {
            if (MultiViewCamera != null) MultiViewCamera.targetDisplay = display;
        }

        // ────────────────────────────────────────────
        // 내부 헬퍼
        // ────────────────────────────────────────────

        private void ApplyCameraPosition()
        {
            var localPos = new Vector3(
                cameraLateralOffset.x,
                cameraLateralOffset.y,
                cameraDistance);

            if (_cameraTransform != null)    _cameraTransform.localPosition    = localPos;
            if (_multiViewTransform != null) _multiViewTransform.localPosition = localPos;
        }

        private void ApplyCameraSettings()
        {
            if (Camera != null) ApplyCameraSettingsTo(Camera);
        }

        private void ApplyCameraSettingsTo(Camera cam)
        {
            cam.fieldOfView   = fieldOfView;
            cam.nearClipPlane = nearClipPlane;
            cam.farClipPlane  = farClipPlane;
        }

        /// <summary>
        /// DroneAgent.Role을 리플렉션으로 읽어 로컬 DroneRole로 변환.
        /// Assembly-CSharp 참조 불가로 인한 리플렉션 사용.
        /// </summary>
        private DroneRole ResolveRole()
        {
            foreach (var comp in GetComponents<MonoBehaviour>())
            {
                if (comp == null || !IsDroneAgentComponent(comp.GetType())) continue;

                var roleField = FindFieldInHierarchy(
                    comp.GetType(),
                    "Role",
                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                if (roleField == null) break;

                // DroneRole enum: Pursuer = 0, Evader = 1
                int roleValue = Convert.ToInt32(roleField.GetValue(comp));
                return roleValue == 1 ? DroneRole.Evader : DroneRole.Pursuer;
            }
            return DroneRole.Pursuer; // DroneAgent 미존재 시 기본값
        }

        private static bool IsDroneAgentComponent(Type type)
        {
            while (type != null)
            {
                if (type.Name == "DroneAgent") return true;
                type = type.BaseType;
            }
            return false;
        }

        private static FieldInfo FindFieldInHierarchy(Type type, string fieldName, BindingFlags flags)
        {
            while (type != null)
            {
                var field = type.GetField(fieldName, flags | BindingFlags.DeclaredOnly);
                if (field != null) return field;
                type = type.BaseType;
            }
            return null;
        }
    }
}
