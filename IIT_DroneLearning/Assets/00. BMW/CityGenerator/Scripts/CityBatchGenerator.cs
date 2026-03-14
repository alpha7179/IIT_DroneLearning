using System.Collections.Generic;
using UnityEngine;

namespace ProceduralCityGenerator
{
    /// <summary>씨드 할당 방식을 결정하는 열거형</summary>
    public enum BatchSeedMode
    {
        AllRandom,    // 각 도시마다 시간 기반 난수 씨드 (-1 전달)
        AllSame,      // 어차피 동일하므로 1개만 생성 (columns·rows 무시)
        Sequential    // baseSeed, baseSeed+1, baseSeed+2, ... 순서대로
    }

    /// <summary>
    /// ML-Agents 학습용 도시를 n × m 격자로 대량 생성하는 컴포넌트.
    /// 각 슬롯에 CityGenerator를 독립 실행하며, 씨드 모드에 따라 다양성을 제어합니다.
    /// AllSame 모드는 결과가 동일하므로 1개만 생성합니다.
    /// </summary>
    public class CityBatchGenerator : MonoBehaviour
    {
        #region Parameters

        [Header("Batch Layout")]
        [Tooltip("가로 열 수 (n). AllSame 모드에서는 무시됩니다.")]
        [Range(1, 20)]
        public int columns = 3;

        [Tooltip("세로 행 수 (m). AllSame 모드에서는 무시됩니다.")]
        [Range(1, 20)]
        public int rows = 3;

        [Header("Spacing")]
        [Tooltip("도시 간 가로 여백 (월드 단위). AllSame 모드에서는 무시됩니다.")]
        [Min(0f)]
        public float spacingX = 20f;

        [Tooltip("도시 간 세로 여백 (월드 단위). AllSame 모드에서는 무시됩니다.")]
        [Min(0f)]
        public float spacingZ = 20f;

        [Header("Seed Settings")]
        [Tooltip("씨드 할당 방식 선택")]
        public BatchSeedMode seedMode = BatchSeedMode.AllRandom;

        [Tooltip("AllSame: 모든 도시에 적용할 씨드 / Sequential: 시작 씨드 (-1이면 시간 기반 랜덤 시작)")]
        public int baseSeed = 0;

        [Header("City Template")]
        [Tooltip("파라미터 복사 원본 — GameObject로 지정 (CityGenerator 컴포넌트가 붙어 있어야 합니다). " +
                 "설정 시 아래 cityTemplate보다 우선 적용됩니다.")]
        public GameObject cityTemplateObject;

        [Tooltip("파라미터 복사 원본 — CityGenerator 컴포넌트를 직접 지정합니다. " +
                 "cityTemplateObject가 비어 있을 때만 사용됩니다.")]
        public CityGenerator cityTemplate;

        #endregion

        #region Internal State

        private GameObject batchRoot;

        #endregion

        #region Public API

        /// <summary>
        /// 배치 생성을 실행합니다. 기존 배치가 있으면 먼저 제거합니다.
        /// </summary>
        public void GenerateBatch()
        {
            ClearBatch();

            if (GetResolvedTemplate() == null)
            {
                Debug.LogError("CityBatchGenerator.GenerateBatch: cityTemplateObject 또는 cityTemplate을 설정해야 합니다.");
                return;
            }

            // 슬롯 크기는 모든 모드에서 공통으로 사용
            float cellW, cellD, slotWidth, slotDepth;
            CalculateSlotSize(out cellW, out cellD, out slotWidth, out slotDepth);

            // AllSame: 동일 씨드로 n × m 격자 생성 (파일은 첫 번째 도시에서만 저장)
            if (seedMode == BatchSeedMode.AllSame)
            {
                batchRoot = new GameObject($"CityBatch_{columns}x{rows}_Seed{baseSeed}");
                batchRoot.transform.SetPositionAndRotation(transform.position, Quaternion.identity);

                for (int r = 0; r < rows; r++)
                {
                    for (int c = 0; c < columns; c++)
                    {
                        float posX = transform.position.x + c * (slotWidth + spacingX) + slotWidth * 0.5f;
                        float posZ = transform.position.z + r * (slotDepth + spacingZ) + slotDepth * 0.5f;
                        Vector3 pos = new Vector3(posX, transform.position.y, posZ);

                        // 첫 번째 도시(0, 0)만 파일 저장, 나머지는 suppressFileExport = true
                        bool suppress = !(c == 0 && r == 0);
                        SpawnCityAt(batchRoot.transform, pos, baseSeed, c, r, suppress);
                    }
                }

                Debug.Log($"CityBatchGenerator: AllSame 모드 — Seed {baseSeed} 도시 {columns}×{rows} = {columns * rows}개 생성 완료 (파일은 1개만 저장)");
                return;
            }

            // AllRandom / Sequential: n × m 격자 생성

            batchRoot = new GameObject($"CityBatch_{columns}x{rows}");
            batchRoot.transform.SetPositionAndRotation(transform.position, Quaternion.identity);

            int resolvedBaseSeed = (seedMode == BatchSeedMode.Sequential && baseSeed < 0)
                ? (int)System.DateTime.Now.Ticks
                : baseSeed;

            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < columns; c++)
                {
                    float posX = transform.position.x + c * (slotWidth + spacingX) + slotWidth * 0.5f;
                    float posZ = transform.position.z + r * (slotDepth + spacingZ) + slotDepth * 0.5f;
                    Vector3 pos = new Vector3(posX, transform.position.y, posZ);

                    int seed = seedMode == BatchSeedMode.Sequential
                        ? resolvedBaseSeed + (c + r * columns)
                        : -1; // AllRandom

                    SpawnCityAt(batchRoot.transform, pos, seed, c, r);
                }
            }

            Debug.Log($"CityBatchGenerator: {columns}×{rows} = {columns * rows}개 도시 생성 완료 (모드: {seedMode})");
        }

        /// <summary>
        /// 생성된 배치를 모두 제거합니다.
        /// </summary>
        public void ClearBatch()
        {
            if (batchRoot != null)
            {
                if (Application.isPlaying) Destroy(batchRoot);
                else DestroyImmediate(batchRoot);
                batchRoot = null;
            }

            // 씬에 남아 있는 이전 배치 루트도 정리 (AllRandom/Sequential 및 AllSame 이름 모두 처리)
            foreach (var name in new[]
            {
                $"CityBatch_{columns}x{rows}",
                $"CityBatch_{columns}x{rows}_Seed{baseSeed}"
            })
            {
                var orphan = GameObject.Find(name);
                if (orphan != null && orphan != batchRoot)
                {
                    if (Application.isPlaying) Destroy(orphan);
                    else DestroyImmediate(orphan);
                }
            }
        }

        #endregion

        #region Private Helpers

        /// <summary>
        /// cityTemplateObject(GameObject) → cityTemplate(컴포넌트) 순으로 유효한 CityGenerator를 반환합니다.
        /// cityTemplateObject가 설정되어 있으면 해당 오브젝트의 CityGenerator 컴포넌트를 우선 사용합니다.
        /// </summary>
        private CityGenerator GetResolvedTemplate()
        {
            if (cityTemplateObject != null)
            {
                var gen = cityTemplateObject.GetComponent<CityGenerator>();
                if (gen != null) return gen;

                Debug.LogWarning($"CityBatchGenerator: cityTemplateObject '{cityTemplateObject.name}'에 " +
                                 "CityGenerator 컴포넌트가 없습니다. cityTemplate으로 폴백합니다.");
            }
            return cityTemplate;
        }

        /// <summary>
        /// 슬롯 하나의 월드 크기를 template.maxWidth/maxDepth 기준으로 계산합니다.
        /// </summary>
        private void CalculateSlotSize(out float cellW, out float cellD,
                                       out float slotWidth, out float slotDepth)
        {
            var t  = GetResolvedTemplate();
            cellW      = (t.buildingWidth  + t.buildingSpacing) * t.unitDistance;
            cellD      = (t.buildingDepth  + t.buildingSpacing) * t.unitDistance;
            slotWidth  = t.maxWidth  * cellW;
            slotDepth  = t.maxDepth  * cellD;
        }

        /// <summary>
        /// 지정 위치에 CityGenerator를 부착한 슬롯 GameObject를 생성하고 도시를 생성합니다.
        /// </summary>
        /// <param name="suppressFileExport">true이면 미니맵 PNG 및 그래프 CSV/JSON 파일 저장을 건너뜁니다.</param>
        private void SpawnCityAt(Transform parent, Vector3 worldPos, int seed, int col, int row, bool suppressFileExport = false)
        {
            var slotGO = new GameObject($"City_col{col}_row{row}");
            slotGO.transform.SetParent(parent, false);
            slotGO.transform.position = worldPos;

            var gen = slotGO.AddComponent<CityGenerator>();
            CopyTemplateParameters(GetResolvedTemplate(), gen);
            gen.randomSeed = seed;
            gen.suppressFileExport = suppressFileExport;

            gen.GenerateCity();
        }

        /// <summary>
        /// template의 공개 파라미터를 target으로 전부 복사합니다.
        /// </summary>
        private void CopyTemplateParameters(CityGenerator src, CityGenerator dst)
        {
            // Grid Settings
            dst.unitDistance         = src.unitDistance;
            dst.minWidth             = src.minWidth;
            dst.maxWidth             = src.maxWidth;
            dst.minDepth             = src.minDepth;
            dst.maxDepth             = src.maxDepth;

            // Building Settings
            dst.buildingWidth        = src.buildingWidth;
            dst.buildingDepth        = src.buildingDepth;
            dst.minBuildingHeight    = src.minBuildingHeight;
            dst.maxBuildingHeight    = src.maxBuildingHeight;
            dst.buildingSpacing      = src.buildingSpacing;
            dst.buildingDensity      = src.buildingDensity;

            // Materials
            dst.defaultBuildingMaterial = src.defaultBuildingMaterial;
            dst.wallMaterial            = src.wallMaterial;
            dst.floorMaterial           = src.floorMaterial;

            // Minimap
            dst.minimapResolution    = src.minimapResolution;

            // Wall / Floor
            dst.spawnWalls           = src.spawnWalls;
            dst.wallHeight           = src.wallHeight;
            dst.wallThickness        = src.wallThickness;
            dst.spawnFloor           = src.spawnFloor;

            // Layout Mode
            dst.layoutMode           = src.layoutMode;
            dst.randomOffsetStrength = src.randomOffsetStrength;
            dst.minBlockSize         = src.minBlockSize;
            dst.maxBlockSize         = src.maxBlockSize;

            // Spawn Configuration
            dst.autoGenerateSpawns   = src.autoGenerateSpawns;
            dst.minSpawnSeparation   = src.minSpawnSeparation;
            dst.minSpawnHeight       = src.minSpawnHeight;
            dst.maxSpawnHeight       = src.maxSpawnHeight;
        }

        #endregion
    }
}
