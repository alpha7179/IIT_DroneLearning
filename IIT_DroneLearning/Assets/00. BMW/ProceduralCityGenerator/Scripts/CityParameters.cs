using UnityEngine;

namespace ProceduralCityGenerator
{
    /// <summary>
    /// 도시 생성 파라미터를 저장하는 ScriptableObject
    /// 프리셋으로 저장하고 로드할 수 있습니다.
    /// </summary>
    [CreateAssetMenu(fileName = "CityPreset", menuName = "City Generator/City Parameters", order = 1)]
    public class CityParameters : ScriptableObject
    {
        [Header("Grid Settings")]
        [Tooltip("격자 시스템의 1 단위가 나타내는 실제 거리 (미터)")]
        [Range(0.1f, 100f)]
        public float unitDistance = 1.0f;

        [Tooltip("도시 최소 가로 길이 (격자 단위)")]
        [Range(1, 100)]
        public int minWidth = 10;

        [Tooltip("도시 최대 가로 길이 (격자 단위)")]
        [Range(1, 100)]
        public int maxWidth = 20;

        [Tooltip("도시 최소 세로 길이 (격자 단위)")]
        [Range(1, 100)]
        public int minDepth = 10;

        [Tooltip("도시 최대 세로 길이 (격자 단위)")]
        [Range(1, 100)]
        public int maxDepth = 20;

        [Header("Building Settings")]
        [Tooltip("건물 가로 크기 (단위_거리)")]
        [Range(0.5f, 50f)]
        public float buildingWidth = 1.0f;

        [Tooltip("건물 세로 크기 (단위_거리)")]
        [Range(0.5f, 50f)]
        public float buildingDepth = 1.0f;

        [Tooltip("건물 최소 높이 (단위_거리)")]
        [Range(1f, 500f)]
        public float minBuildingHeight = 5.0f;

        [Tooltip("건물 최대 높이 (단위_거리)")]
        [Range(1f, 500f)]
        public float maxBuildingHeight = 20.0f;

        [Tooltip("건물 사이의 간격 (단위_거리)")]
        [Range(0f, 50f)]
        public float buildingSpacing = 1.0f;

        [Tooltip("격자 셀에 건물이 배치될 확률 (0.0 ~ 1.0)")]
        [Range(0f, 1f)]
        public float buildingDensity = 0.7f;

        [Header("Generation Settings")]
        [Tooltip("재현 가능한 도시 생성을 위한 시드값 (-1: 시간 기반 랜덤)")]
        public int randomSeed = -1;

        [Header("Minimap Settings")]
        [Tooltip("생성될 미니맵의 해상도")]
        public MinimapResolution minimapResolution = MinimapResolution.Resolution512;

        /// <summary>
        /// CityGenerator로부터 현재 파라미터를 복사합니다.
        /// </summary>
        /// <param name="generator">파라미터를 복사할 CityGenerator</param>
        public void CopyFrom(CityGenerator generator)
        {
            if (generator == null)
            {
                Debug.LogError("CityParameters.CopyFrom: generator is null");
                return;
            }

            unitDistance = generator.unitDistance;
            minWidth = generator.minWidth;
            maxWidth = generator.maxWidth;
            minDepth = generator.minDepth;
            maxDepth = generator.maxDepth;
            buildingWidth = generator.buildingWidth;
            buildingDepth = generator.buildingDepth;
            minBuildingHeight = generator.minBuildingHeight;
            maxBuildingHeight = generator.maxBuildingHeight;
            buildingSpacing = generator.buildingSpacing;
            buildingDensity = generator.buildingDensity;
            randomSeed = generator.randomSeed;
            minimapResolution = generator.minimapResolution;
        }

        /// <summary>
        /// 저장된 파라미터를 CityGenerator에 적용합니다.
        /// </summary>
        /// <param name="generator">파라미터를 적용할 CityGenerator</param>
        public void ApplyTo(CityGenerator generator)
        {
            if (generator == null)
            {
                Debug.LogError("CityParameters.ApplyTo: generator is null");
                return;
            }

            generator.unitDistance = unitDistance;
            generator.minWidth = minWidth;
            generator.maxWidth = maxWidth;
            generator.minDepth = minDepth;
            generator.maxDepth = maxDepth;
            generator.buildingWidth = buildingWidth;
            generator.buildingDepth = buildingDepth;
            generator.minBuildingHeight = minBuildingHeight;
            generator.maxBuildingHeight = maxBuildingHeight;
            generator.buildingSpacing = buildingSpacing;
            generator.buildingDensity = buildingDensity;
            generator.randomSeed = randomSeed;
            generator.minimapResolution = minimapResolution;
        }
    }
}
