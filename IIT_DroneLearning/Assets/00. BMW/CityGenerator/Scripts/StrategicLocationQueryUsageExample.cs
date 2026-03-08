using UnityEngine;
using System.Collections.Generic;

namespace ProceduralCityGenerator
{
    /// <summary>
    /// CityDataAPI의 전략적 위치 쿼리 메서드 사용 예제
    /// GetCoverPoints 및 GetNearestStrategicLocation 메서드 사용법을 보여줍니다.
    /// </summary>
    public class StrategicLocationQueryUsageExample : MonoBehaviour
    {
        [Header("Query Settings")]
        [Tooltip("은폐 지점을 검색할 반경")]
        public float searchRadius = 10f;

        [Tooltip("검색할 전략적 위치 타입")]
        public StrategyType targetStrategyType = StrategyType.CoverPoint;

        [Header("Visualization")]
        [Tooltip("은폐 지점을 시각화할지 여부")]
        public bool visualizeCoverPoints = true;

        [Tooltip("가장 가까운 전략적 위치를 시각화할지 여부")]
        public bool visualizeNearestLocation = true;

        [Tooltip("은폐 지점 표시 색상")]
        public Color coverPointColor = Color.blue;

        [Tooltip("가장 가까운 위치 표시 색상")]
        public Color nearestLocationColor = Color.green;

        private List<StrategicLocation> currentCoverPoints;
        private StrategicLocation currentNearestLocation;

        void Start()
        {
            // CityDataAPI가 초기화되었는지 확인
            if (!CityDataAPI.Instance.IsInitialized())
            {
                Debug.LogWarning("StrategicLocationQueryUsageExample: CityDataAPI is not initialized. " +
                                 "Please generate a city first.");
                return;
            }

            Debug.Log("StrategicLocationQueryUsageExample: CityDataAPI is ready.");
        }

        void Update()
        {
            // CityDataAPI가 초기화되지 않았으면 리턴
            if (!CityDataAPI.Instance.IsInitialized())
            {
                return;
            }

            // 현재 위치 기준으로 쿼리 수행
            Vector3 currentPosition = transform.position;

            // 1. 주변 은폐 지점 검색
            currentCoverPoints = CityDataAPI.Instance.GetCoverPoints(currentPosition, searchRadius);

            // 2. 가장 가까운 전략적 위치 검색
            currentNearestLocation = CityDataAPI.Instance.GetNearestStrategicLocation(
                currentPosition, targetStrategyType);
        }

        void OnDrawGizmos()
        {
            if (!Application.isPlaying || !CityDataAPI.Instance.IsInitialized())
            {
                return;
            }

            Vector3 currentPosition = transform.position;

            // 검색 반경 시각화
            Gizmos.color = new Color(1f, 1f, 0f, 0.2f);
            Gizmos.DrawWireSphere(currentPosition, searchRadius);

            // 은폐 지점 시각화
            if (visualizeCoverPoints && currentCoverPoints != null)
            {
                Gizmos.color = coverPointColor;
                foreach (StrategicLocation coverPoint in currentCoverPoints)
                {
                    Gizmos.DrawSphere(coverPoint.position, 0.5f);
                    Gizmos.DrawLine(currentPosition, coverPoint.position);
                }
            }

            // 가장 가까운 전략적 위치 시각화
            if (visualizeNearestLocation && 
                currentNearestLocation.position != Vector3.zero)
            {
                Gizmos.color = nearestLocationColor;
                Gizmos.DrawSphere(currentNearestLocation.position, 0.7f);
                Gizmos.DrawLine(currentPosition, currentNearestLocation.position);
            }
        }

        /// <summary>
        /// 특정 위치 주변의 은폐 지점을 검색하는 예제 메서드
        /// </summary>
        public void FindCoverPointsExample()
        {
            if (!CityDataAPI.Instance.IsInitialized())
            {
                Debug.LogError("CityDataAPI is not initialized!");
                return;
            }

            Vector3 searchPosition = transform.position;
            float radius = searchRadius;

            // 은폐 지점 검색
            List<StrategicLocation> coverPoints = CityDataAPI.Instance.GetCoverPoints(
                searchPosition, radius);

            Debug.Log($"Found {coverPoints.Count} cover points within {radius} units of {searchPosition}");

            // 각 은폐 지점 정보 출력
            foreach (StrategicLocation coverPoint in coverPoints)
            {
                float distance = Vector3.Distance(searchPosition, coverPoint.position);
                Debug.Log($"  Cover Point: Position={coverPoint.position}, " +
                          $"Distance={distance:F2}, Visibility={coverPoint.visibilityScore:F2}");
            }
        }

        /// <summary>
        /// 가장 가까운 전략적 위치를 검색하는 예제 메서드
        /// </summary>
        public void FindNearestStrategicLocationExample()
        {
            if (!CityDataAPI.Instance.IsInitialized())
            {
                Debug.LogError("CityDataAPI is not initialized!");
                return;
            }

            Vector3 searchPosition = transform.position;
            StrategyType type = targetStrategyType;

            // 가장 가까운 전략적 위치 검색
            StrategicLocation nearestLocation = CityDataAPI.Instance.GetNearestStrategicLocation(
                searchPosition, type);

            // 결과 확인
            if (nearestLocation.position == Vector3.zero && nearestLocation.connectedNodes == null)
            {
                Debug.Log($"No strategic location of type {type} found.");
            }
            else
            {
                float distance = Vector3.Distance(searchPosition, nearestLocation.position);
                Debug.Log($"Nearest {type}: Position={nearestLocation.position}, " +
                          $"Distance={distance:F2}, Visibility={nearestLocation.visibilityScore:F2}");
            }
        }

        /// <summary>
        /// 모든 전략적 위치 타입에 대해 가장 가까운 위치를 검색하는 예제
        /// </summary>
        public void FindAllNearestStrategicLocationsExample()
        {
            if (!CityDataAPI.Instance.IsInitialized())
            {
                Debug.LogError("CityDataAPI is not initialized!");
                return;
            }

            Vector3 searchPosition = transform.position;

            Debug.Log($"Finding nearest strategic locations from {searchPosition}:");

            // 모든 전략 타입에 대해 검색
            StrategyType[] allTypes = new StrategyType[]
            {
                StrategyType.CoverPoint,
                StrategyType.Intersection,
                StrategyType.DeadEnd,
                StrategyType.OpenArea,
                StrategyType.DetourPath
            };

            foreach (StrategyType type in allTypes)
            {
                StrategicLocation location = CityDataAPI.Instance.GetNearestStrategicLocation(
                    searchPosition, type);

                if (location.position != Vector3.zero || location.connectedNodes != null)
                {
                    float distance = Vector3.Distance(searchPosition, location.position);
                    Debug.Log($"  {type}: Distance={distance:F2}, Position={location.position}");
                }
                else
                {
                    Debug.Log($"  {type}: Not found");
                }
            }
        }

        /// <summary>
        /// 은폐 지점을 거리순으로 정렬하는 예제
        /// </summary>
        public void SortCoverPointsByDistanceExample()
        {
            if (!CityDataAPI.Instance.IsInitialized())
            {
                Debug.LogError("CityDataAPI is not initialized!");
                return;
            }

            Vector3 searchPosition = transform.position;
            List<StrategicLocation> coverPoints = CityDataAPI.Instance.GetCoverPoints(
                searchPosition, searchRadius);

            if (coverPoints.Count == 0)
            {
                Debug.Log("No cover points found.");
                return;
            }

            // 거리순으로 정렬
            coverPoints.Sort((a, b) =>
            {
                float distA = Vector3.Distance(searchPosition, a.position);
                float distB = Vector3.Distance(searchPosition, b.position);
                return distA.CompareTo(distB);
            });

            Debug.Log($"Cover points sorted by distance (total: {coverPoints.Count}):");
            for (int i = 0; i < Mathf.Min(5, coverPoints.Count); i++)
            {
                float distance = Vector3.Distance(searchPosition, coverPoints[i].position);
                Debug.Log($"  {i + 1}. Distance={distance:F2}, Position={coverPoints[i].position}");
            }
        }
    }
}
