using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;

namespace CityGenerator
{
    /// <summary>
    /// 도시 그래프(노드·엣지·전략 위치)를 JSON 및 CSV 파일로 내보내는 정적 유틸리티 클래스.
    /// 도망자 드론이 도시 지형을 오프라인으로 분석할 수 있도록
    /// ProceduralCityGenerator/CityData/ 폴더에 저장합니다.
    ///
    /// 출력 파일 (동일 seed면 덮어쓰기):
    ///   {ProceduralCityGenerator}/CityData/City_Seed{N}.json          — 전체 그래프 + 전략 위치 + 메타데이터
    ///   {ProceduralCityGenerator}/CityData/City_Seed{N}_nodes.csv     — 노드 속성 테이블
    ///   {ProceduralCityGenerator}/CityData/City_Seed{N}_edges.csv     — 엣지(연결) 테이블
    ///
    /// 경로는 이 스크립트 파일의 위치를 기준으로 동적으로 계산되므로
    /// ProceduralCityGenerator 폴더가 이동해도 자동으로 반영됩니다.
    /// </summary>
    public static class CityGraphExporter
    {
        // ─────────────────────────────────────────────────────────────
        // 동적 경로 헬퍼
        // ─────────────────────────────────────────────────────────────

        /// <summary>
        /// 이 스크립트(CityGraphExporter.cs)의 위치를 기준으로
        /// ProceduralCityGenerator 폴더의 Unity Asset 경로를 반환합니다.
        ///
        /// 파일 구조 가정:
        ///   {ProceduralCityGenerator}/Scripts/CityGraphExporter.cs
        ///           ↑ 2단계 상위 = ProceduralCityGenerator 폴더
        /// </summary>
        private static string GetProceduralCityFolder()
        {
#if UNITY_EDITOR
            // FindAssets는 파일명에 검색어가 포함된 모든 스크립트를 반환하므로
            // 정확히 일치하는 파일만 사용
            string[] guids = UnityEditor.AssetDatabase.FindAssets("CityGraphExporter t:Script");
            foreach (string guid in guids)
            {
                string assetPath = UnityEditor.AssetDatabase.GUIDToAssetPath(guid);
                if (Path.GetFileName(assetPath) == "CityGraphExporter.cs")
                {
                    // "Assets/00. BMW/ProceduralCityGenerator/Scripts/CityGraphExporter.cs"
                    string scriptsFolder = Path.GetDirectoryName(assetPath);
                    // "Assets/00. BMW/ProceduralCityGenerator/Scripts"
                    string baseFolder = Path.GetDirectoryName(scriptsFolder);
                    // "Assets/00. BMW/ProceduralCityGenerator"
                    return baseFolder.Replace('\\', '/');
                }
            }
#endif
            // Fallback: AssetDatabase를 사용할 수 없는 환경(런타임 빌드 등)
            return "Assets";
        }

        // ─────────────────────────────────────────────────────────────
        // 공개 API
        // ─────────────────────────────────────────────────────────────

        /// <summary>
        /// JSON + 노드 CSV + 엣지 CSV 를 한번에 저장합니다.
        /// </summary>
        /// <param name="graph">도시 그래프</param>
        /// <param name="strategicLocations">전략적 위치 목록</param>
        /// <param name="seed">생성에 사용된 랜덤 시드 (파일명 식별자)</param>
        /// <param name="cityBounds">도시 전체 경계 (Bounds)</param>
        /// <returns>저장된 디렉토리 경로</returns>
        public static string ExportAll(
            CityGraph graph,
            List<StrategicLocation> strategicLocations,
            int seed,
            Bounds cityBounds,
            string layoutTag = "")
        {
            if (graph == null)
            {
                Debug.LogError("[CityGraphExporter] graph가 null입니다. 내보내기를 중단합니다.");
                return null;
            }

            // Unity 에셋 상대 경로 → 절대 파일시스템 경로로 변환
            // Application.dataPath = ".../Assets" 이므로 "Assets" 접두사를 제거하고 결합
            string assetRelPath = GetProceduralCityFolder() + "/CityData";
            string outputDir = (Application.dataPath + assetRelPath.Substring("Assets".Length))
                               .Replace('\\', '/');

            Debug.Log($"[CityGraphExporter] CityData 절대 경로 = {outputDir}");

            if (!Directory.Exists(outputDir))
                Directory.CreateDirectory(outputDir);

            // 파일명: City_Seed{N}_{LayoutMode}  (태그 없으면 시드만)
            string tag      = string.IsNullOrEmpty(layoutTag) ? "" : $"_{layoutTag}";
            string baseName = $"City_Seed{seed}{tag}";

            ExportJson(graph, strategicLocations, seed, cityBounds, layoutTag,
                       Path.Combine(outputDir, baseName + ".json"));

            ExportNodesCSV(graph, strategicLocations,
                           Path.Combine(outputDir, baseName + "_nodes.csv"));

            ExportEdgesCSV(graph,
                           Path.Combine(outputDir, baseName + "_edges.csv"));

#if UNITY_EDITOR
            UnityEditor.AssetDatabase.Refresh();
#endif
            Debug.Log($"[CityGraphExporter] 내보내기 완료 → {outputDir}/{baseName}{{.json, _nodes.csv, _edges.csv}}");
            return outputDir;
        }

        // ─────────────────────────────────────────────────────────────
        // JSON
        // ─────────────────────────────────────────────────────────────

        /// <summary>
        /// 그래프 전체(메타데이터 + 노드 + 엣지 + 전략 위치)를 JSON으로 저장합니다.
        /// Unity JsonUtility는 List·배열 중첩 구조를 일부 지원하지 못하므로 StringBuilder로 직접 구성합니다.
        /// </summary>
        private static void ExportJson(
            CityGraph graph,
            List<StrategicLocation> strategicLocations,
            int seed,
            Bounds bounds,
            string layoutTag,
            string filePath)
        {
            var sb = new StringBuilder();
            sb.AppendLine("{");

            // ── 메타데이터 ──────────────────────────────────────────
            sb.AppendLine("  \"metadata\": {");
            sb.AppendLine($"    \"seed\": {seed},");
            sb.AppendLine($"    \"layoutMode\": \"{layoutTag}\",");
            sb.AppendLine($"    \"nodeCount\": {graph.NodeCount},");
            sb.AppendLine($"    \"edgeCount\": {graph.EdgeCount},");
            sb.AppendLine($"    \"strategicLocationCount\": {(strategicLocations?.Count ?? 0)},");
            AppendVec3Line(sb, "boundsMin", bounds.min, comma: true);
            AppendVec3Line(sb, "boundsMax", bounds.max, comma: true);
            sb.AppendLine($"    \"generatedAt\": \"{System.DateTime.Now:yyyy-MM-dd HH:mm:ss}\"");
            sb.AppendLine("  },");

            // ── 노드 ────────────────────────────────────────────────
            var nodes = graph.GetAllNodes();
            sb.AppendLine("  \"nodes\": [");
            for (int i = 0; i < nodes.Count; i++)
            {
                var n = nodes[i];
                bool last = (i == nodes.Count - 1);

                sb.AppendLine("    {");
                sb.AppendLine($"      \"id\": {n.nodeId},");
                sb.AppendLine($"      \"x\": {n.position.x:F3},");
                sb.AppendLine($"      \"y\": {n.position.y:F3},");
                sb.AppendLine($"      \"z\": {n.position.z:F3},");
                sb.AppendLine($"      \"nodeType\": \"{n.nodeType}\",");
                sb.AppendLine($"      \"elevation\": {n.elevation:F3},");
                sb.AppendLine($"      \"isVisibleFromSpawn\": {n.isVisibleFromSpawn.ToString().ToLower()},");

                // strategicMarkers
                var markers = n.strategicMarkers ?? new List<StrategyType>();
                sb.Append("      \"strategicMarkers\": [");
                for (int m = 0; m < markers.Count; m++)
                {
                    sb.Append($"\"{markers[m]}\"");
                    if (m < markers.Count - 1) sb.Append(", ");
                }
                sb.AppendLine("],");

                // surroundingBuildingHeights
                var heights = n.surroundingBuildingHeights ?? new float[0];
                sb.Append("      \"surroundingBuildingHeights\": [");
                for (int h = 0; h < heights.Length; h++)
                {
                    sb.Append($"{heights[h]:F2}");
                    if (h < heights.Length - 1) sb.Append(", ");
                }
                sb.AppendLine("]");

                sb.Append(last ? "    }" : "    },");
                sb.AppendLine();
            }
            sb.AppendLine("  ],");

            // ── 엣지 ────────────────────────────────────────────────
            sb.AppendLine("  \"edges\": [");
            var allEdges = new List<GraphEdge>();
            foreach (var node in nodes)
                allEdges.AddRange(graph.GetEdges(node.nodeId));

            for (int i = 0; i < allEdges.Count; i++)
            {
                var e = allEdges[i];
                bool last = (i == allEdges.Count - 1);
                sb.Append($"    {{ \"from\": {e.startNodeId}, \"to\": {e.endNodeId}, " +
                          $"\"cost\": {e.travelCost:F3}, \"visibility\": {e.visibilityScore:F3}, " +
                          $"\"pathType\": \"{e.pathType}\" }}");
                sb.AppendLine(last ? "" : ",");
            }
            sb.AppendLine("  ],");

            // ── 전략적 위치 ─────────────────────────────────────────
            var locs = strategicLocations ?? new List<StrategicLocation>();
            sb.AppendLine("  \"strategicLocations\": [");
            for (int i = 0; i < locs.Count; i++)
            {
                var loc = locs[i];
                bool last = (i == locs.Count - 1);

                sb.AppendLine("    {");
                sb.AppendLine($"      \"x\": {loc.position.x:F3},");
                sb.AppendLine($"      \"y\": {loc.position.y:F3},");
                sb.AppendLine($"      \"z\": {loc.position.z:F3},");
                sb.AppendLine($"      \"locationType\": \"{loc.locationType}\",");
                sb.AppendLine($"      \"dangerScore\": {loc.dangerScore:F3},");
                sb.AppendLine($"      \"visibilityScore\": {loc.visibilityScore:F3},");

                var connected = loc.connectedNodes ?? new List<int>();
                sb.Append("      \"connectedNodes\": [");
                for (int c = 0; c < connected.Count; c++)
                {
                    sb.Append(connected[c]);
                    if (c < connected.Count - 1) sb.Append(", ");
                }
                sb.AppendLine("]");

                sb.Append(last ? "    }" : "    },");
                sb.AppendLine();
            }
            sb.AppendLine("  ]");

            sb.AppendLine("}");

            File.WriteAllText(filePath, sb.ToString(), Encoding.UTF8);
            Debug.Log($"[CityGraphExporter] JSON 저장: {filePath}");
        }

        // ─────────────────────────────────────────────────────────────
        // 노드 CSV
        // ─────────────────────────────────────────────────────────────

        /// <summary>
        /// 노드 속성을 CSV로 저장합니다.
        /// 각 행 = 하나의 노드. strategicLocations의 dangerScore/visibilityScore를
        /// connectedNodes를 통해 노드에 역매핑합니다.
        /// </summary>
        private static void ExportNodesCSV(
            CityGraph graph,
            List<StrategicLocation> strategicLocations,
            string filePath)
        {
            // nodeId → (dangerScore, visibilityScore, locationType) 역매핑
            var dangerMap = new Dictionary<int, float>();
            var visMap    = new Dictionary<int, float>();
            var typeMap   = new Dictionary<int, string>();

            if (strategicLocations != null)
            {
                foreach (var loc in strategicLocations)
                {
                    if (loc.connectedNodes == null) continue;
                    foreach (int nid in loc.connectedNodes)
                    {
                        // 하나의 노드가 여러 전략 위치에 속할 수 있으므로 첫 번째 값만 기록
                        if (!dangerMap.ContainsKey(nid))
                        {
                            dangerMap[nid] = loc.dangerScore;
                            visMap[nid]    = loc.visibilityScore;
                            typeMap[nid]   = loc.locationType.ToString();
                        }
                    }
                }
            }

            var sb = new StringBuilder();
            sb.AppendLine("nodeId,pos_x,pos_y,pos_z,nodeType,elevation," +
                          "isVisibleFromSpawn,strategicType,dangerScore," +
                          "visibilityScore,avgSurroundingHeight,neighborCount");

            foreach (var node in graph.GetAllNodes())
            {
                float avgHeight = 0f;
                if (node.surroundingBuildingHeights != null &&
                    node.surroundingBuildingHeights.Length > 0)
                {
                    foreach (float h in node.surroundingBuildingHeights)
                        avgHeight += h;
                    avgHeight /= node.surroundingBuildingHeights.Length;
                }

                string stType  = typeMap.TryGetValue(node.nodeId, out string t) ? t : "";
                float  danger  = dangerMap.TryGetValue(node.nodeId, out float d) ? d : 0f;
                float  vis     = visMap.TryGetValue(node.nodeId, out float v)    ? v : 0f;
                int    nbCount = graph.GetEdges(node.nodeId).Count;

                sb.AppendLine(
                    $"{node.nodeId}," +
                    $"{node.position.x:F3},{node.position.y:F3},{node.position.z:F3}," +
                    $"{node.nodeType},{node.elevation:F3}," +
                    $"{node.isVisibleFromSpawn},{stType}," +
                    $"{danger:F3},{vis:F3},{avgHeight:F3},{nbCount}");
            }

            File.WriteAllText(filePath, sb.ToString(), Encoding.UTF8);
            Debug.Log($"[CityGraphExporter] Nodes CSV 저장: {filePath}");
        }

        // ─────────────────────────────────────────────────────────────
        // 엣지 CSV
        // ─────────────────────────────────────────────────────────────

        /// <summary>
        /// 모든 엣지를 CSV로 저장합니다. 각 행 = 하나의 방향성 엣지.
        /// </summary>
        private static void ExportEdgesCSV(CityGraph graph, string filePath)
        {
            var sb = new StringBuilder();
            sb.AppendLine("startNodeId,endNodeId,travelCost,visibilityScore,pathType");

            foreach (var node in graph.GetAllNodes())
            {
                foreach (var edge in graph.GetEdges(node.nodeId))
                {
                    sb.AppendLine(
                        $"{edge.startNodeId},{edge.endNodeId}," +
                        $"{edge.travelCost:F3},{edge.visibilityScore:F3},{edge.pathType}");
                }
            }

            File.WriteAllText(filePath, sb.ToString(), Encoding.UTF8);
            Debug.Log($"[CityGraphExporter] Edges CSV 저장: {filePath}");
        }

        // ─────────────────────────────────────────────────────────────
        // 헬퍼
        // ─────────────────────────────────────────────────────────────

        private static void AppendVec3Line(StringBuilder sb, string key, Vector3 v, bool comma)
        {
            string trail = comma ? "," : "";
            sb.AppendLine(
                $"    \"{key}\": {{ \"x\": {v.x:F3}, \"y\": {v.y:F3}, \"z\": {v.z:F3} }}{trail}");
        }
    }
}
