using UnityEngine;

namespace BMW.DroneSensor
{
    /// <summary>
    /// 드론 레이 센서 시스템
    /// 26방향 레이캐스트를 통해 드론 주변 360도 전방위 장애물 감지를 제공합니다.
    /// 5개 레이어(Top, Top-Middle, Middle, Middle-Bottom, Bottom)로 구성됩니다.
    /// </summary>
    /// <remarks>
    /// <para><b>좌표계 및 변환 가정:</b></para>
    /// 
    /// <para><b>1. 월드 좌표계 (Unity 표준)</b></para>
    /// <list type="bullet">
    /// <item>Unity 왼손 좌표계 사용</item>
    /// <item>+X: 우측 (동쪽)</item>
    /// <item>+Y: 위쪽 (상방)</item>
    /// <item>+Z: 전방 (북쪽)</item>
    /// </list>
    /// 
    /// <para><b>2. 드론 로컬 좌표계</b></para>
    /// <list type="bullet">
    /// <item>+X: 드론의 우측</item>
    /// <item>+Y: 드론의 상방</item>
    /// <item>+Z: 드론의 전방 (기수 방향)</item>
    /// <item>로컬 좌표계는 드론의 회전(Transform.rotation)에 따라 월드 좌표계로 변환됩니다</item>
    /// </list>
    /// 
    /// <para><b>3. 구면 좌표계 (레이 방향 계산)</b></para>
    /// <list type="bullet">
    /// <item>방위각(Azimuth): 수평면에서의 각도 (0° = 북/전방, 시계방향 증가)</item>
    /// <item>고도각(Elevation): 수평면 기준 수직 각도 (-90° = 아래, 0° = 수평, +90° = 위)</item>
    /// <item>변환 공식:
    ///   <code>
    ///   x = cos(elevation) * sin(azimuth)
    ///   y = sin(elevation)
    ///   z = cos(elevation) * cos(azimuth)
    ///   </code>
    /// </item>
    /// </list>
    /// 
    /// <para><b>4. 나침반 방향 매핑</b></para>
    /// <list type="bullet">
    /// <item>N (북): 0° - 드론 전방 (+Z)</item>
    /// <item>NE (북동): 45° - 전방 우측</item>
    /// <item>E (동): 90° - 드론 우측 (+X)</item>
    /// <item>SE (남동): 135° - 후방 우측</item>
    /// <item>S (남): 180° - 드론 후방 (-Z)</item>
    /// <item>SW (남서): 225° - 후방 좌측</item>
    /// <item>W (서): 270° - 드론 좌측 (-X)</item>
    /// <item>NW (북서): 315° - 전방 좌측</item>
    /// </list>
    /// 
    /// <para><b>5. 레이어별 고도각</b></para>
    /// <list type="bullet">
    /// <item>Top: +90° (수직 위)</item>
    /// <item>Top-Middle: +45° (기본값, Inspector에서 조정 가능, 범위: 0~90°)</item>
    /// <item>Middle: 0° (수평)</item>
    /// <item>Middle-Bottom: -45° (기본값, Inspector에서 조정 가능, 범위: -90~0°)</item>
    /// <item>Bottom: -90° (수직 아래)</item>
    /// </list>
    /// 
    /// <para><b>6. 변환 계층 구조</b></para>
    /// <list type="bullet">
    /// <item>레이 방향은 초기화 시 로컬 좌표계에서 사전 계산되어 캐시됩니다</item>
    /// <item>매 FixedUpdate마다 Transform.TransformDirection()을 사용하여 월드 좌표계로 변환됩니다</item>
    /// <item>드론이 회전하면 모든 레이도 함께 회전합니다 (드론 중심 기준)</item>
    /// <item>레이캐스트는 드론의 Transform.position에서 시작됩니다</item>
    /// </list>
    /// 
    /// <para><b>7. 향후 카메라 통합을 위한 가정</b></para>
    /// <list type="bullet">
    /// <item>카메라는 드론의 자식 GameObject로 부착될 것으로 가정합니다</item>
    /// <item>카메라의 전방 방향은 Middle 레이어의 N(북) 방향 레이와 정렬되어야 합니다</item>
    /// <item>GetForwardRayDirection() 메서드를 사용하여 카메라 방향을 설정할 수 있습니다</item>
    /// <item>카메라 센서는 별도의 컴포넌트로 구현되며, DroneSensorSystem과 독립적으로 작동합니다</item>
    /// <item>카메라와 레이 센서는 동일한 Transform 계층 구조를 공유하므로 일관된 좌표 변환이 보장됩니다</item>
    /// </list>
    /// 
    /// <para><b>8. 거리 정규화</b></para>
    /// <list type="bullet">
    /// <item>실제 거리(미터) → 정규화된 거리: distance / MaxDetectionRange</item>
    /// <item>충돌 없음: 0.0 (최대 감지 거리 내에 장애물 없음)</item>
    /// <item>최대 범위 충돌: 1.0 (MaxDetectionRange 거리에서 충돌)</item>
    /// <item>중간 거리 충돌: 0.0~1.0 사이 값</item>
    /// </list>
    /// </remarks>
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

    // 자기 자신 콜라이더 캐시 (FixedUpdate 매 프레임 GC 할당 방지)
    private Collider[] _droneColliders;
    
    #endregion

    #region 공개 API 메서드
    
    /// <summary>
    /// 모든 센서의 정규화된 거리 값을 반환합니다 (26개).
    /// </summary>
    /// <returns>
    /// 26개 요소의 float 배열. 각 값은 0.0~1.0 범위:
    /// - 0.0 = 충돌 없음 (최대 감지 거리 내에 장애물 없음)
    /// - 1.0 = 최대 감지 거리에서 충돌
    /// - 0.0~1.0 = 중간 거리에서 충돌 (실제 거리 / 최대 감지 거리)
    /// </returns>
    /// <remarks>
    /// 배열 인덱스 매핑:
    /// - 인덱스 0: Top (수직 위)
    /// - 인덱스 1-8: Top-Middle (N, NE, E, SE, S, SW, W, NW)
    /// - 인덱스 9-16: Middle (N, NE, E, SE, S, SW, W, NW)
    /// - 인덱스 17-24: Middle-Bottom (N, NE, E, SE, S, SW, W, NW)
    /// - 인덱스 25: Bottom (수직 아래)
    /// 
    /// 사용 예시:
    /// <code>
    /// float[] distances = sensorSystem.GetAllNormalizedDistances();
    /// // ML-Agents VectorSensor에 추가
    /// foreach (float distance in distances)
    /// {
    ///     sensor.AddObservation(distance);
    /// }
    /// </code>
    /// </remarks>
    public float[] GetAllNormalizedDistances()
    {
        if (!_isInitialized)
        {
            Debug.LogError("[DroneSensorSystem] 치명적 오류: 센서가 아직 초기화되지 않았습니다.");
            return new float[26];
        }

        if (_sensorData == null || _sensorData.Length != 26)
        {
            Debug.LogError($"[DroneSensorSystem] 치명적 오류: 센서 데이터 배열이 유효하지 않습니다. 길이: {(_sensorData?.Length ?? 0)}");
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
    /// <param name="layer">
    /// 센서 레이어:
    /// - Top: 수직 위 (1개 레이)
    /// - TopMiddle: 위쪽 대각선 (8개 레이, 기본 45도)
    /// - Middle: 수평 (8개 레이, 0도)
    /// - MiddleBottom: 아래쪽 대각선 (8개 레이, 기본 -45도)
    /// - Bottom: 수직 아래 (1개 레이)
    /// </param>
    /// <param name="direction">
    /// 수평 나침반 방향 (8방위):
    /// - N: 북 (전방, 0도)
    /// - NE: 북동 (45도)
    /// - E: 동 (우측, 90도)
    /// - SE: 남동 (135도)
    /// - S: 남 (후방, 180도)
    /// - SW: 남서 (225도)
    /// - W: 서 (좌측, 270도)
    /// - NW: 북서 (315도)
    /// 
    /// 주의: Top 및 Bottom 레이어는 단일 레이이므로 direction 매개변수가 무시됩니다.
    /// </param>
    /// <returns>
    /// 정규화된 거리 값 (0.0~1.0):
    /// - 0.0 = 충돌 없음
    /// - 1.0 = 최대 감지 거리에서 충돌
    /// </returns>
    /// <remarks>
    /// 이 메서드는 GetAllNormalizedDistances() 배열의 해당 인덱스 값과 동일한 결과를 반환합니다.
    /// 
    /// 사용 예시:
    /// <code>
    /// // 전방 수평 레이의 거리 조회
    /// float forwardDistance = sensorSystem.GetNormalizedDistance(
    ///     SensorLayer.Middle, 
    ///     CompassDirection.N
    /// );
    /// 
    /// // 우측 위쪽 대각선 레이의 거리 조회
    /// float rightTopDistance = sensorSystem.GetNormalizedDistance(
    ///     SensorLayer.TopMiddle, 
    ///     CompassDirection.E
    /// );
    /// </code>
    /// </remarks>
    public float GetNormalizedDistance(SensorLayer layer, CompassDirection direction)
    {
        if (!_isInitialized)
        {
            Debug.LogError("[DroneSensorSystem] 치명적 오류: 센서가 아직 초기화되지 않았습니다.");
            return 0f;
        }

        if (_sensorData == null || _sensorData.Length != 26)
        {
            Debug.LogError($"[DroneSensorSystem] 치명적 오류: 센서 데이터 배열이 유효하지 않습니다. 길이: {(_sensorData?.Length ?? 0)}");
            return 0f;
        }

        int index = GetRayIndex(layer, direction);
        if (index < 0 || index >= 26)
        {
            Debug.LogError($"[DroneSensorSystem] 치명적 오류: 잘못된 레이 인덱스: {index}");
            return 0f;
        }

        return _sensorData[index].NormalizedDistance;
    }

    /// <summary>
    /// 최대 감지 거리를 런타임에 설정합니다.
    /// </summary>
    /// <param name="range">
    /// 새로운 최대 감지 거리 (미터 단위). 양수 값이어야 합니다.
    /// </param>
    /// <remarks>
    /// 0 이하의 값이 전달되면 경고 로그를 출력하고 기본값 50m로 복원됩니다.
    /// 이 값은 레이캐스트의 최대 범위와 거리 정규화에 사용됩니다.
    /// 
    /// 사용 예시:
    /// <code>
    /// // 감지 거리를 100m로 확장
    /// sensorSystem.SetMaxDetectionRange(100f);
    /// 
    /// // 좁은 공간에서 감지 거리를 20m로 축소
    /// sensorSystem.SetMaxDetectionRange(20f);
    /// </code>
    /// </remarks>
    public void SetMaxDetectionRange(float range)
        {
            if (range <= 0f || float.IsNaN(range) || float.IsInfinity(range))
            {
                Debug.LogWarning($"[DroneSensorSystem] 경고: MaxDetectionRange는 양수여야 합니다. 입력값: {range}m. 기본값 50m로 복원합니다.");
                MaxDetectionRange = 50f;
                return;
            }
            MaxDetectionRange = range;
        }


    /// <summary>
    /// 특정 레이를 활성화 또는 비활성화합니다.
    /// </summary>
    /// <param name="rayIndex">
    /// 레이 인덱스 (0~25). 유효하지 않은 인덱스는 경고 로그를 출력하고 무시됩니다.
    /// 인덱스 매핑:
    /// - 0: Top
    /// - 1-8: Top-Middle (N, NE, E, SE, S, SW, W, NW)
    /// - 9-16: Middle (N, NE, E, SE, S, SW, W, NW)
    /// - 17-24: Middle-Bottom (N, NE, E, SE, S, SW, W, NW)
    /// - 25: Bottom
    /// </param>
    /// <param name="enabled">
    /// true = 레이 활성화 (레이캐스트 수행), false = 레이 비활성화 (레이캐스트 건너뛰기)
    /// </param>
    /// <remarks>
    /// 비활성화된 레이는 레이캐스트를 수행하지 않으며, 거리 값은 0.0으로 설정됩니다.
    /// 이 기능은 특정 방향의 센서를 선택적으로 끄거나 성능 최적화에 사용할 수 있습니다.
    /// 
    /// 사용 예시:
    /// <code>
    /// // Top 레이 비활성화
    /// sensorSystem.SetRayEnabled(0, false);
    /// 
    /// // Middle 레이어의 전방(N) 레이만 활성화하고 나머지 비활성화
    /// for (int i = 9; i <= 16; i++)
    /// {
    ///     sensorSystem.SetRayEnabled(i, i == 9); // 인덱스 9만 true
    /// }
    /// </code>
    /// </remarks>
    public void SetRayEnabled(int rayIndex, bool enabled)
    {
        if (rayIndex < 0 || rayIndex >= 26)
        {
            Debug.LogWarning($"[DroneSensorSystem] 경고: 잘못된 레이 인덱스: {rayIndex}. 유효 범위는 0~25입니다.");
            return;
        }

        if (_rayConfigs == null || _rayConfigs.Length != 26)
        {
            Debug.LogError($"[DroneSensorSystem] 치명적 오류: 레이 설정 배열이 유효하지 않습니다. 길이: {(_rayConfigs?.Length ?? 0)}");
            return;
        }

        if (_rayConfigs[rayIndex] != null)
        {
            _rayConfigs[rayIndex].IsEnabled = enabled;
        }
        else
        {
            Debug.LogError($"[DroneSensorSystem] 치명적 오류: 레이 설정이 null입니다. 인덱스: {rayIndex}");
        }
    }

    /// <summary>
    /// 전방 레이의 월드 공간 방향 벡터를 반환합니다 (카메라 정렬용).
    /// </summary>
    /// <returns>
    /// Middle 레이어의 N(북/전방) 방향 레이를 월드 공간으로 변환한 방향 벡터.
    /// 드론의 현재 회전이 적용된 정규화된 벡터입니다.
    /// </returns>
    /// <remarks>
    /// 이 메서드는 향후 카메라 센서 통합 시 카메라를 전방 레이와 정렬하는 데 사용됩니다.
    /// Middle 레이어의 N 방향 레이 (인덱스 9)를 기준으로 합니다.
    /// 
    /// 사용 예시:
    /// <code>
    /// // 카메라를 전방 레이 방향으로 정렬
    /// Vector3 forwardDirection = sensorSystem.GetForwardRayDirection();
    /// camera.transform.rotation = Quaternion.LookRotation(forwardDirection);
    /// </code>
    /// </remarks>
    public Vector3 GetForwardRayDirection()
    {
        if (!_isInitialized)
        {
            Debug.LogError("[DroneSensorSystem] 치명적 오류: 센서가 아직 초기화되지 않았습니다.");
            return Vector3.forward;
        }

        if (_rayConfigs == null || _rayConfigs.Length != 26)
        {
            Debug.LogError($"[DroneSensorSystem] 치명적 오류: 레이 설정 배열이 유효하지 않습니다. 길이: {(_rayConfigs?.Length ?? 0)}");
            return Vector3.forward;
        }

        // Middle 레이어의 N(북) 방향 레이 (인덱스 9)
        return transform.TransformDirection(_rayConfigs[9].LocalDirection);
    }

    /// <summary>
    /// 디버그 시각화를 활성화 또는 비활성화합니다.
    /// </summary>
    /// <param name="enabled">
    /// true = Scene 뷰에서 레이 시각화 활성화, false = 시각화 비활성화
    /// </param>
    /// <remarks>
    /// 디버그 시각화가 활성화되면 Scene 뷰에서 모든 레이가 색상으로 표시됩니다:
    /// - 빨간색: 충돌 감지된 레이
    /// - 녹색: 충돌 없는 레이
    /// 
    /// 시각화는 개발 및 디버깅 중에만 사용하고, 프로덕션 빌드에서는 비활성화하는 것이 좋습니다.
    /// 
    /// 사용 예시:
    /// <code>
    /// // 디버그 시각화 활성화
    /// sensorSystem.SetDebugVisualization(true);
    /// 
    /// // 성능 테스트를 위해 시각화 비활성화
    /// sensorSystem.SetDebugVisualization(false);
    /// </code>
    /// </remarks>
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
        // [Fix] ValidateConfiguration을 먼저 호출해 TopMiddleElevation / MiddleBottomElevation을
        // 올바른 범위로 클램핑한 뒤, 방향 벡터를 계산합니다.
        // 순서가 역전되면 잘못된 각도로 캐시된 레이 방향이 수정되지 않는 버그가 발생합니다.
        ValidateConfiguration();
        InitializeRayConfigurations();
        InitializeSensorData();
        _droneColliders = GetComponentsInChildren<Collider>();
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

        // 26개 레이 모두 수행
        for (int i = 0; i < 26; i++)
        {
            // 배열 인덱스 범위 검증
            if (i < 0 || i >= _rayConfigs.Length || i >= _sensorData.Length)
            {
                Debug.LogError($"[DroneSensorSystem] 치명적 오류: 배열 인덱스 범위 초과. 인덱스: {i}");
                continue;
            }

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

            // 레이캐스트 수행 (예외 처리 포함)
            bool hasHit = false;
            RaycastHit hit = default;
            
            try
            {
                hasHit = Physics.Raycast(
                    transform.position,
                    worldDirection,
                    out hit,
                    MaxDetectionRange,
                    DetectionLayerMask
                );
            }
            catch (System.Exception ex)
            {
                Debug.LogError($"[DroneSensorSystem] 치명적 오류: 레이캐스트 실행 중 예외 발생. 레이 인덱스: {i}, 예외: {ex.Message}");
                // 예외 발생 시 충돌 없음으로 처리
                _sensorData[i].HasHit = false;
                _sensorData[i].RawDistance = 0f;
                _sensorData[i].NormalizedDistance = 0f;
                _sensorData[i].HitPoint = Vector3.zero;
                continue;
            }

            // 자기 자신의 콜라이더와 충돌한 경우 무시
            if (hasHit && _droneColliders != null)
            {
                bool isSelfCollision = false;
                foreach (Collider droneCollider in _droneColliders)
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
            Debug.LogWarning($"[DroneSensorSystem] 경고: MaxDetectionRange는 양수여야 합니다. 현재 값: {MaxDetectionRange}m. 기본값 50m로 복원합니다.");
            MaxDetectionRange = 50f;
        }
        
        // Top-Middle 고도각 검증 (0~90도 허용)
        if (TopMiddleElevation < 0f || TopMiddleElevation > 90f)
        {
            float originalValue = TopMiddleElevation;
            TopMiddleElevation = Mathf.Clamp(TopMiddleElevation, 0f, 90f);
            Debug.LogWarning($"[DroneSensorSystem] 경고: TopMiddleElevation은 0~90도 범위여야 합니다. 입력값: {originalValue}도. {TopMiddleElevation}도로 클램핑합니다.");
        }
        
        // Middle-Bottom 고도각 검증 (-90~0도 허용)
        if (MiddleBottomElevation < -90f || MiddleBottomElevation > 0f)
        {
            float originalValue = MiddleBottomElevation;
            MiddleBottomElevation = Mathf.Clamp(MiddleBottomElevation, -90f, 0f);
            Debug.LogWarning($"[DroneSensorSystem] 경고: MiddleBottomElevation은 -90~0도 범위여야 합니다. 입력값: {originalValue}도. {MiddleBottomElevation}도로 클램핑합니다.");
        }
        
        // 레이어 마스크 검증
        if (DetectionLayerMask.value == 0)
        {
            Debug.LogWarning("[DroneSensorSystem] 경고: DetectionLayerMask가 Nothing으로 설정되어 있습니다. 모든 레이가 충돌을 감지하지 못합니다.");
        }
    }
    
    #endregion

    #region 내부 헬퍼 메서드
    
    /// <summary>
    /// 레이어와 방향을 배열 인덱스로 매핑합니다.
    /// </summary>
    private int GetRayIndex(SensorLayer layer, CompassDirection direction)
    {
        int index;
        
        switch (layer)
        {
            case SensorLayer.Top:
                index = 0;
                break;
            case SensorLayer.TopMiddle:
                index = 1 + (int)direction;
                break;
            case SensorLayer.Middle:
                index = 9 + (int)direction;
                break;
            case SensorLayer.MiddleBottom:
                index = 17 + (int)direction;
                break;
            case SensorLayer.Bottom:
                index = 25;
                break;
            default:
                Debug.LogError($"[DroneSensorSystem] 치명적 오류: 잘못된 센서 레이어: {layer}");
                return -1;
        }
        
        // 인덱스 범위 검증
        if (index < 0 || index >= 26)
        {
            Debug.LogError($"[DroneSensorSystem] 치명적 오류: 계산된 레이 인덱스가 범위를 벗어났습니다. 레이어: {layer}, 방향: {direction}, 인덱스: {index}");
            return -1;
        }
        
        return index;
    }
    
    #endregion
}
}
