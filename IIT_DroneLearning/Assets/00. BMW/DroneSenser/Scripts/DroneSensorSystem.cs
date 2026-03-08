using UnityEngine;

namespace BMW.DroneSensor
{
    /// <summary>
    /// 드론 레이 센서 시스템
    /// 26방향 레이캐스트를 통해 드론 주변 360도 전방위 장애물 감지를 제공합니다.
    /// 5개 레이어(Top, Top-Middle, Middle, Middle-Bottom, Bottom)로 구성됩니다.
    /// </summary>
    public class DroneSensorSystem : MonoBehaviour
{
    #region 열거형 정의
    
    /// <summary>
    /// 센서 레이어 정의
    /// </summary>
    public enum SensorLayer
    {
        Top,            // 수직 위 (1개)
        TopMiddle,      // 위쪽 대각선 (8개)
        Middle,         // 수평 (8개)
        MiddleBottom,   // 아래쪽 대각선 (8개)
        Bottom          // 수직 아래 (1개)
    }

    /// <summary>
    /// 8방위 나침반 방향
    /// </summary>
    public enum CompassDirection
    {
        N,   // 북 (전방, 0°)
        NE,  // 북동 (45°)
        E,   // 동 (우측, 90°)
        SE,  // 남동 (135°)
        S,   // 남 (후방, 180°)
        SW,  // 남서 (225°)
        W,   // 서 (좌측, 270°)
        NW   // 북서 (315°)
    }
    
    #endregion

    #region Unity Inspector 노출 속성
    
    [Header("센서 설정")]
    [Tooltip("최대 감지 거리 (미터)")]
    public float MaxDetectionRange = 50f;
    
    [Tooltip("Top-Middle 레이어 고도각 (도)")]
    [Range(0f, 90f)]
    public float TopMiddleElevation = 45f;
    
    [Tooltip("Middle-Bottom 레이어 고도각 (도)")]
    [Range(-90f, 0f)]
    public float MiddleBottomElevation = -45f;
    
    [Tooltip("감지 대상 레이어")]
    public LayerMask DetectionLayerMask = -1;

    [Header("디버그")]
    [Tooltip("Scene 뷰에서 레이 시각화")]
    public bool ShowDebugRays = true;
    
    [Tooltip("충돌 감지 시 레이 색상")]
    public Color RayHitColor = Color.red;
    
    [Tooltip("충돌 없을 때 레이 색상")]
    public Color RayMissColor = Color.green;
    
    #endregion

    #region 내부 클래스 및 구조체
    
    /// <summary>
    /// 각 레이의 설정 정보를 저장합니다.
    /// </summary>
    private class RayConfiguration
    {
        public Vector3 LocalDirection;      // 드론 로컬 좌표계 방향 벡터
        public SensorLayer Layer;           // 소속 레이어
        public CompassDirection Direction;  // 수평 방향 (해당 시)
        public bool IsEnabled;              // 활성화 상태
        public int Index;                   // 배열 인덱스 (0~25)
    }

    /// <summary>
    /// 레이캐스트 결과를 저장합니다.
    /// </summary>
    private struct SensorData
    {
        public float RawDistance;           // 실제 충돌 거리 (미터)
        public float NormalizedDistance;    // 정규화된 거리 (0.0~1.0)
        public bool HasHit;                 // 충돌 감지 여부
        public Vector3 HitPoint;            // 충돌 지점 (월드 좌표)
    }
    
    #endregion

    #region 비공개 필드
    
    // 레이 설정 배열 (26개)
    private RayConfiguration[] _rayConfigs;
    
    // 센서 데이터 배열 (26개)
    private SensorData[] _sensorData;
    
    // 초기화 완료 플래그
    private bool _isInitialized = false;
    
    #endregion

    #region 공개 API 메서드
    
    /// <summary>
    /// 모든 센서의 정규화된 거리 값을 반환합니다 (26개).
    /// 반환값: 0.0 = 충돌 없음, 1.0 = 최대 범위에서 충돌
    /// </summary>
    public float[] GetAllNormalizedDistances()
    {
        if (!_isInitialized)
        {
            Debug.LogError("[DroneSensorSystem] 센서가 아직 초기화되지 않았습니다.");
            return new float[26];
        }

        float[] distances = new float[26];
        for (int i = 0; i < 26; i++)
        {
            distances[i] = _sensorData[i].NormalizedDistance;
        }
        return distances;
    }

    /// <summary>
    /// 특정 레이어와 방향의 정규화된 거리 값을 반환합니다.
    /// </summary>
    /// <param name="layer">센서 레이어 (Top, TopMiddle, Middle, MiddleBottom, Bottom)</param>
    /// <param name="direction">수평 방향 (N, NE, E, SE, S, SW, W, NW) - 8방향 레이어만 해당</param>
    public float GetNormalizedDistance(SensorLayer layer, CompassDirection direction)
    {
        if (!_isInitialized)
        {
            Debug.LogError("[DroneSensorSystem] 센서가 아직 초기화되지 않았습니다.");
            return 0f;
        }

        int index = GetRayIndex(layer, direction);
        if (index < 0 || index >= 26)
        {
            Debug.LogError($"[DroneSensorSystem] 잘못된 레이 인덱스: {index}");
            return 0f;
        }

        return _sensorData[index].NormalizedDistance;
    }

    /// <summary>
    /// 최대 감지 거리를 런타임에 설정합니다.
    /// </summary>
    public void SetMaxDetectionRange(float range)
    {
        if (range <= 0f)
        {
            Debug.LogWarning($"[DroneSensorSystem] MaxDetectionRange는 양수여야 합니다. 기본값 50m로 설정합니다.");
            MaxDetectionRange = 50f;
            return;
        }
        MaxDetectionRange = range;
    }

    /// <summary>
    /// 특정 레이를 활성화/비활성화합니다.
    /// </summary>
    public void SetRayEnabled(int rayIndex, bool enabled)
    {
        if (rayIndex < 0 || rayIndex >= 26)
        {
            Debug.LogWarning($"[DroneSensorSystem] 잘못된 레이 인덱스: {rayIndex}. 유효 범위는 0~25입니다.");
            return;
        }

        if (_rayConfigs != null && _rayConfigs[rayIndex] != null)
        {
            _rayConfigs[rayIndex].IsEnabled = enabled;
        }
    }

    /// <summary>
    /// 전방 레이 방향을 월드 공간에서 반환합니다 (카메라 정렬용).
    /// </summary>
    public Vector3 GetForwardRayDirection()
    {
        if (!_isInitialized)
        {
            Debug.LogError("[DroneSensorSystem] 센서가 아직 초기화되지 않았습니다.");
            return Vector3.forward;
        }

        // Middle 레이어의 N(북) 방향 레이 (인덱스 9)
        return transform.TransformDirection(_rayConfigs[9].LocalDirection);
    }

    /// <summary>
    /// 디버그 시각화를 토글합니다.
    /// </summary>
    public void SetDebugVisualization(bool enabled)
    {
        ShowDebugRays = enabled;
    }
    
    #endregion

    #region Unity 생명주기 메서드
    
    /// <summary>
    /// 센서 시스템 초기화
    /// </summary>
    private void Awake()
    {
        InitializeRayConfigurations();
        InitializeSensorData();
        ValidateConfiguration();
        _isInitialized = true;
    }

    /// <summary>
    /// 물리 업데이트마다 26개 레이캐스트를 수행합니다.
    /// </summary>
    private void FixedUpdate()
    {
        if (!_isInitialized)
        {
            return;
        }

        // 드론의 모든 콜라이더를 가져와서 자기 자신 무시
        Collider[] droneColliders = GetComponentsInChildren<Collider>();

        // 26개 레이 모두 수행
        for (int i = 0; i < 26; i++)
        {
            RayConfiguration rayConfig = _rayConfigs[i];

            // 비활성화된 레이는 건너뛰기
            if (!rayConfig.IsEnabled)
            {
                _sensorData[i].HasHit = false;
                _sensorData[i].RawDistance = 0f;
                _sensorData[i].NormalizedDistance = 0f;
                _sensorData[i].HitPoint = Vector3.zero;
                continue;
            }

            // 로컬 방향을 월드 방향으로 변환
            Vector3 worldDirection = transform.TransformDirection(rayConfig.LocalDirection);

            // 레이캐스트 수행
            RaycastHit hit;
            bool hasHit = Physics.Raycast(
                transform.position,
                worldDirection,
                out hit,
                MaxDetectionRange,
                DetectionLayerMask
            );

            // 자기 자신의 콜라이더와 충돌한 경우 무시
            if (hasHit && droneColliders != null)
            {
                bool isSelfCollision = false;
                foreach (Collider droneCollider in droneColliders)
                {
                    if (hit.collider == droneCollider)
                    {
                        isSelfCollision = true;
                        break;
                    }
                }

                // 자기 자신과의 충돌이면 충돌 없음으로 처리
                if (isSelfCollision)
                {
                    hasHit = false;
                }
            }

            // 센서 데이터 업데이트
            if (hasHit)
            {
                // 충돌 감지: 거리 저장 및 정규화
                float distance = hit.distance;
                _sensorData[i].HasHit = true;
                _sensorData[i].RawDistance = distance;
                _sensorData[i].NormalizedDistance = distance / MaxDetectionRange;
                _sensorData[i].HitPoint = hit.point;
            }
            else
            {
                // 충돌 없음: 0.0으로 설정
                _sensorData[i].HasHit = false;
                _sensorData[i].RawDistance = 0f;
                _sensorData[i].NormalizedDistance = 0f;
                _sensorData[i].HitPoint = Vector3.zero;
            }

            // 디버그 시각화
            if (ShowDebugRays)
            {
                Color rayColor = hasHit ? RayHitColor : RayMissColor;
                float drawDistance = hasHit ? hit.distance : MaxDetectionRange;
                Debug.DrawRay(transform.position, worldDirection * drawDistance, rayColor);
            }
        }
    }

    
    #endregion

    #region 초기화 메서드
    
    /// <summary>
    /// 26개 레이의 방향 벡터를 사전 계산합니다.
    /// </summary>
    private void InitializeRayConfigurations()
    {
        _rayConfigs = new RayConfiguration[26];
        
        // 인덱스 0: Top (수직 위)
        _rayConfigs[0] = CreateRayConfig(0, SensorLayer.Top, CompassDirection.N, 0f, 90f);
        
        // 인덱스 1-8: Top-Middle (위쪽 대각선)
        for (int i = 0; i < 8; i++)
        {
            CompassDirection dir = (CompassDirection)i;
            float azimuth = i * 45f;
            _rayConfigs[1 + i] = CreateRayConfig(1 + i, SensorLayer.TopMiddle, dir, azimuth, TopMiddleElevation);
        }
        
        // 인덱스 9-16: Middle (수평)
        for (int i = 0; i < 8; i++)
        {
            CompassDirection dir = (CompassDirection)i;
            float azimuth = i * 45f;
            _rayConfigs[9 + i] = CreateRayConfig(9 + i, SensorLayer.Middle, dir, azimuth, 0f);
        }
        
        // 인덱스 17-24: Middle-Bottom (아래쪽 대각선)
        for (int i = 0; i < 8; i++)
        {
            CompassDirection dir = (CompassDirection)i;
            float azimuth = i * 45f;
            _rayConfigs[17 + i] = CreateRayConfig(17 + i, SensorLayer.MiddleBottom, dir, azimuth, MiddleBottomElevation);
        }
        
        // 인덱스 25: Bottom (수직 아래)
        _rayConfigs[25] = CreateRayConfig(25, SensorLayer.Bottom, CompassDirection.N, 0f, -90f);
    }
    
    /// <summary>
    /// 구면 좌표계를 사용하여 레이 설정을 생성합니다.
    /// </summary>
    /// <param name="index">레이 인덱스 (0~25)</param>
    /// <param name="layer">센서 레이어</param>
    /// <param name="direction">나침반 방향</param>
    /// <param name="azimuthDegrees">방위각 (도)</param>
    /// <param name="elevationDegrees">고도각 (도)</param>
    private RayConfiguration CreateRayConfig(int index, SensorLayer layer, CompassDirection direction, 
                                             float azimuthDegrees, float elevationDegrees)
    {
        // 도를 라디안으로 변환
        float azimuthRad = azimuthDegrees * Mathf.Deg2Rad;
        float elevationRad = elevationDegrees * Mathf.Deg2Rad;
        
        // 구면 좌표계 공식:
        // x = cos(elevation) * sin(azimuth)
        // y = sin(elevation)
        // z = cos(elevation) * cos(azimuth)
        Vector3 localDirection = new Vector3(
            Mathf.Cos(elevationRad) * Mathf.Sin(azimuthRad),
            Mathf.Sin(elevationRad),
            Mathf.Cos(elevationRad) * Mathf.Cos(azimuthRad)
        );
        
        return new RayConfiguration
        {
            Index = index,
            Layer = layer,
            Direction = direction,
            LocalDirection = localDirection.normalized,
            IsEnabled = true
        };
    }
    
    /// <summary>
    /// 센서 데이터 배열을 초기화합니다.
    /// </summary>
    private void InitializeSensorData()
    {
        _sensorData = new SensorData[26];
        
        for (int i = 0; i < 26; i++)
        {
            _sensorData[i] = new SensorData
            {
                RawDistance = 0f,
                NormalizedDistance = 0f,
                HasHit = false,
                HitPoint = Vector3.zero
            };
        }
    }
    
    /// <summary>
    /// 설정 매개변수를 검증합니다.
    /// </summary>
    private void ValidateConfiguration()
    {
        // 최대 감지 거리 검증
        if (MaxDetectionRange <= 0f)
        {
            Debug.LogWarning("[DroneSensorSystem] MaxDetectionRange는 양수여야 합니다. 기본값 50m로 설정합니다.");
            MaxDetectionRange = 50f;
        }
        
        // Top-Middle 고도각 검증 (30~60도 권장, 0~90도 허용)
        if (TopMiddleElevation < 0f || TopMiddleElevation > 90f)
        {
            Debug.LogWarning($"[DroneSensorSystem] TopMiddleElevation은 0~90도 범위여야 합니다. 현재 값: {TopMiddleElevation}도. 45도로 클램핑합니다.");
            TopMiddleElevation = Mathf.Clamp(TopMiddleElevation, 0f, 90f);
        }
        
        // Middle-Bottom 고도각 검증 (-30~-60도 권장, -90~0도 허용)
        if (MiddleBottomElevation < -90f || MiddleBottomElevation > 0f)
        {
            Debug.LogWarning($"[DroneSensorSystem] MiddleBottomElevation은 -90~0도 범위여야 합니다. 현재 값: {MiddleBottomElevation}도. -45도로 클램핑합니다.");
            MiddleBottomElevation = Mathf.Clamp(MiddleBottomElevation, -90f, 0f);
        }
        
        // 레이어 마스크 검증
        if (DetectionLayerMask.value == 0)
        {
            Debug.LogWarning("[DroneSensorSystem] DetectionLayerMask가 Nothing으로 설정되어 있습니다. 모든 레이가 충돌을 감지하지 못합니다.");
        }
        
        // Transform 컴포넌트 검증
        if (transform == null)
        {
            Debug.LogError("[DroneSensorSystem] Transform 컴포넌트를 찾을 수 없습니다. 초기화 실패.");
            _isInitialized = false;
        }
    }
    
    #endregion

    #region 내부 헬퍼 메서드
    
    /// <summary>
    /// 레이어와 방향을 배열 인덱스로 매핑합니다.
    /// </summary>
    private int GetRayIndex(SensorLayer layer, CompassDirection direction)
    {
        switch (layer)
        {
            case SensorLayer.Top:
                return 0;
            case SensorLayer.TopMiddle:
                return 1 + (int)direction;
            case SensorLayer.Middle:
                return 9 + (int)direction;
            case SensorLayer.MiddleBottom:
                return 17 + (int)direction;
            case SensorLayer.Bottom:
                return 25;
            default:
                Debug.LogError($"[DroneSensorSystem] 잘못된 센서 레이어: {layer}");
                return -1;
        }
    }
    
    #endregion
}
}
