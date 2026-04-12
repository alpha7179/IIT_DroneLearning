using System.Collections.Generic;
using UnityEngine;

namespace CityGenerator
{
    /// <summary>
    /// 건물 GameObject의 오브젝트 풀링을 구현하는 팩토리 클래스
    /// Queue 기반 풀을 사용하여 대규모 도시 생성 시 성능을 최적화합니다.
    /// </summary>
    public class BuildingFactory
    {
        private Queue<GameObject> buildingPool;
        private Material defaultMaterial;
        private Transform parentTransform;
        private int totalCreated;

        /// <summary>
        /// BuildingFactory 생성자
        /// </summary>
        /// <param name="material">건물에 적용할 기본 머티리얼</param>
        /// <param name="parent">건물이 배치될 부모 Transform</param>
        /// <param name="initialPoolSize">초기 풀 크기 (기본값: 100)</param>
        public BuildingFactory(Material material, Transform parent, int initialPoolSize = 100)
        {
            buildingPool = new Queue<GameObject>(initialPoolSize);
            defaultMaterial = material;
            parentTransform = parent;
            totalCreated = 0;

            // 초기 풀 생성
            for (int i = 0; i < initialPoolSize; i++)
            {
                GameObject building = CreateNewBuilding();
                building.SetActive(false);
                buildingPool.Enqueue(building);
            }
        }

        /// <summary>
        /// 건물을 생성하거나 풀에서 가져옵니다.
        /// </summary>
        /// <param name="position">건물 위치</param>
        /// <param name="scale">건물 크기</param>
        /// <param name="name">건물 이름</param>
        /// <returns>생성되거나 풀에서 가져온 건물 GameObject</returns>
        public GameObject CreateBuilding(Vector3 position, Vector3 scale, string name)
        {
            GameObject building;

            // 풀에 사용 가능한 건물이 있으면 재사용
            if (buildingPool.Count > 0)
            {
                building = buildingPool.Dequeue();
            }
            else
            {
                // 풀이 비어있으면 새로 생성
                building = CreateNewBuilding();
            }

            // 건물 설정
            building.name = name;
            building.transform.position = position;
            building.transform.localScale = scale;
            building.SetActive(true);

            return building;
        }

        /// <summary>
        /// 건물을 풀로 반환합니다.
        /// </summary>
        /// <param name="building">반환할 건물 GameObject</param>
        public void ReturnBuilding(GameObject building)
        {
            if (building == null)
            {
                Debug.LogWarning("BuildingFactory.ReturnBuilding: building is null");
                return;
            }

            // 건물 비활성화 및 풀에 반환
            building.SetActive(false);
            buildingPool.Enqueue(building);
        }

        /// <summary>
        /// 풀의 모든 건물을 제거하고 초기화합니다.
        /// </summary>
        public void ClearPool()
        {
            // 풀의 모든 GameObject 파괴
            while (buildingPool.Count > 0)
            {
                GameObject building = buildingPool.Dequeue();
                if (building != null)
                {
                    Object.Destroy(building);
                }
            }

            buildingPool.Clear();
            totalCreated = 0;
        }

        /// <summary>
        /// 새로운 건물 GameObject를 생성합니다.
        /// Unity 기본 큐브 프리미티브를 사용하고 머티리얼과 콜라이더를 추가합니다.
        /// </summary>
        /// <returns>새로 생성된 건물 GameObject</returns>
        private GameObject CreateNewBuilding()
        {
            // Unity 기본 큐브 프리미티브 생성 (요구사항 1.1)
            GameObject building = GameObject.CreatePrimitive(PrimitiveType.Cube);
            building.transform.SetParent(parentTransform);

            // 기본 머티리얼 적용 (요구사항 1.2)
            if (defaultMaterial != null)
            {
                Renderer renderer = building.GetComponent<Renderer>();
                if (renderer != null)
                {
                    renderer.material = defaultMaterial;
                }
            }

            // BoxCollider는 CreatePrimitive에 의해 자동으로 추가됨 (요구사항 1.4)
            // 명시적으로 확인하여 없으면 추가
            if (building.GetComponent<BoxCollider>() == null)
            {
                building.AddComponent<BoxCollider>();
            }

            building.tag = "Building";

            totalCreated++;
            return building;
        }

        /// <summary>
        /// 현재 풀에 있는 건물 수를 반환합니다.
        /// </summary>
        public int PoolCount => buildingPool.Count;

        /// <summary>
        /// 지금까지 생성된 총 건물 수를 반환합니다.
        /// </summary>
        public int TotalCreated => totalCreated;
    }
}
