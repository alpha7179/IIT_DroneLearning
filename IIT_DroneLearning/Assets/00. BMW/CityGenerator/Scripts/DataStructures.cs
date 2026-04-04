using System;
using System.Collections.Generic;
using UnityEngine;

namespace CityGenerator
{
    /// <summary>
    /// 그래프 노드의 타입을 정의하는 열거형
    /// </summary>
    [Serializable]
    public enum NodeType
    {
        OpenSpace,        // 개방 공간
        BuildingCorner,   // 건물 모서리
        AlleyEntrance,    // 골목 입구
        Intersection      // 교차로
    }

    /// <summary>
    /// 경로의 타입을 정의하는 열거형
    /// </summary>
    [Serializable]
    public enum PathType
    {
        Direct,     // 직선
        Detour,     // 우회
        Concealed   // 은폐
    }

    /// <summary>
    /// 전략적 위치의 타입을 정의하는 열거형
    /// </summary>
    [Serializable]
    public enum StrategyType
    {
        CoverPoint,    // 은폐 지점
        Intersection,  // 교차로
        DeadEnd,       // 막다른 골목
        OpenArea,      // 개방 구역
        DetourPath     // 우회 경로
    }

    /// <summary>
    /// 도시 레이아웃 생성 방식을 정의하는 열거형
    /// </summary>
    [Serializable]
    public enum CityLayoutMode
    {
        PureGrid,    // 완전 격자 — 균일한 블록 간격
        Hybrid,      // 격자 + 랜덤 오프셋 — 불규칙 블록, 건물 위치·크기 변화
        PureRandom   // 유기적 도로망 + 블록 내 자유 배치
    }

    /// <summary>
    /// 미니맵에 표시되는 동적 마커의 타입을 정의하는 열거형
    /// </summary>
    [Serializable]
    public enum MarkerType
    {
        EvaderDrone,   // 도망자 드론 (파란색)
        PursuerDrone,  // 추적자 드론 (빨간색)
        TargetPoint,   // 목표 지점 (초록색)
        PathLine       // 경로 표시
    }

    /// <summary>
    /// 미니맵 해상도를 정의하는 열거형
    /// </summary>
    [Serializable]
    public enum MinimapResolution
    {
        Resolution256 = 256,
        Resolution512 = 512,
        Resolution1024 = 1024
    }

    /// <summary>
    /// 격자 셀 정보를 저장하는 구조체
    /// </summary>
    [Serializable]
    public struct GridCell
    {
        public int x;                    // 격자 X 좌표
        public int z;                    // 격자 Z 좌표
        public bool hasBuilding;         // 건물 존재 여부
        public float buildingHeight;     // 건물 높이
        public Vector3 worldPosition;    // 월드 좌표
    }

    /// <summary>
    /// 그래프의 노드를 표현하는 구조체
    /// </summary>
    [Serializable]
    public struct GraphNode
    {
        public int nodeId;                              // 노드 고유 식별자
        public Vector3 position;                        // 위치 좌표
        public NodeType nodeType;                       // 노드 타입
        public float elevation;                         // 고도 정보 (지면 높이)
        public float[] surroundingBuildingHeights;      // 주변 건물 높이 (가림 현상 계산용)
        public List<StrategyType> strategicMarkers;     // 전략적 위치 마커
        public bool isVisibleFromSpawn;                 // 생성 지점에서 보이는지 여부
    }

    /// <summary>
    /// 그래프의 엣지를 표현하는 구조체
    /// </summary>
    [Serializable]
    public struct GraphEdge
    {
        public int startNodeId;          // 시작 노드 ID
        public int endNodeId;            // 끝 노드 ID
        public float travelCost;         // 이동 비용 (거리 기반)
        public float visibilityScore;    // 가시성 정보 (0.0 ~ 1.0)
        public PathType pathType;        // 경로 타입
    }

    /// <summary>
    /// 전략적 위치 정보를 저장하는 구조체
    /// </summary>
    [Serializable]
    public struct StrategicLocation
    {
        public Vector3 position;              // 위치 좌표
        public StrategyType locationType;     // 위치 타입
        public float dangerScore;             // 위험도 점수 (0.0 ~ 1.0)
        public float visibilityScore;         // 가시성 점수 (추적자로부터 보일 확률)
        public List<int> connectedNodes;      // 연결된 노드 ID 리스트
    }

    /// <summary>
    /// 건물 정보를 저장하는 클래스
    /// </summary>
    [Serializable]
    public class Building
    {
        public GameObject gameObject;    // 건물 GameObject 참조
        public Vector3 position;         // 위치
        public Vector3 size;             // 크기
        public float height;             // 높이
        public GridCell gridCell;        // 격자 셀 정보
        public Bounds bounds;            // 경계 박스

        /// <summary>
        /// 건물 생성자
        /// </summary>
        public Building(GameObject go, Vector3 pos, Vector3 sz, float h, GridCell cell)
        {
            gameObject = go;
            position = pos;
            size = sz;
            height = h;
            gridCell = cell;
            bounds = new Bounds(pos, sz);
        }
    }

    /// <summary>
    /// 파라미터 검증 결과를 저장하는 구조체
    /// </summary>
    [Serializable]
    public struct ValidationResult
    {
        public bool isValid;                                                    // 검증 성공 여부
        public List<string> warnings;                                           // 경고 메시지 리스트
        public List<string> errors;                                             // 오류 메시지 리스트
        public Dictionary<string, (float original, float clamped)> clampedValues;  // 제한된 값들

        /// <summary>
        /// ValidationResult 생성자
        /// </summary>
        public ValidationResult(bool valid)
        {
            isValid = valid;
            warnings = new List<string>();
            errors = new List<string>();
            clampedValues = new Dictionary<string, (float, float)>();
        }
    }

    /// <summary>
    /// 스폰/타겟 포인트 구성 정보를 저장하는 구조체
    /// </summary>
    [Serializable]
    public struct SpawnConfiguration
    {
        public Vector3 evaderSpawnPosition;   // 도망 드론 스폰 위치
        public Vector3 pursuerSpawnPosition;  // 추적 드론 스폰 위치
        public Vector3 targetPosition;        // 타겟 포인트 위치

        public int evaderSpawnNodeId;         // 도망 드론 스폰 위치의 가장 가까운 그래프 노드 ID
        public int pursuerSpawnNodeId;        // 추적 드론 스폰 위치의 가장 가까운 그래프 노드 ID
        public int targetNodeId;              // 타겟 포인트의 가장 가까운 그래프 노드 ID

        public float achievedMinSeparation;   // 실제 달성된 최소 분리 거리 (디버그용)
        public bool isValid;                  // 생성 성공 여부
    }

    /// <summary>
    /// 도시 생성 결과에서 추출된 구조적 메타데이터.
    /// 도시 경계, 건물 목록, 도로 그래프, 높이 범위, 전략적 위치 등을 포함한다.
    /// </summary>
    [Serializable]
    public class CityMetadata
    {
        /// <summary>도시 전체 경계 박스</summary>
        public Bounds cityBounds;

        /// <summary>격자 가로 크기</summary>
        public int actualCityWidth;

        /// <summary>격자 세로 크기</summary>
        public int actualCityDepth;

        /// <summary>건물 최소 높이</summary>
        public float minBuildingHeight;

        /// <summary>건물 최대 높이</summary>
        public float maxBuildingHeight;

        /// <summary>건물 목록 참조</summary>
        public List<Building> buildings;

        /// <summary>도로 그래프 참조</summary>
        public CityGraph cityGraph;

        /// <summary>전략적 위치 목록 참조</summary>
        public List<StrategicLocation> strategicLocations;

        /// <summary>유효 스폰 후보 노드 (캐싱)</summary>
        public List<GraphNode> validSpawnCandidates;

        /// <summary>도시 생성에 사용된 랜덤 시드</summary>
        public int usedRandomSeed;

        /// <summary>레이아웃 모드</summary>
        public CityLayoutMode layoutMode;
    }

    /// <summary>
    /// 도시 생성 결과를 저장하는 클래스
    /// </summary>
    [Serializable]
    public class CityGenerationResult
    {
        public bool success;                // 생성 성공 여부
        public int buildingCount;           // 생성된 건물 수
        public int nodeCount;               // 생성된 노드 수
        public int edgeCount;               // 생성된 엣지 수
        public float generationTime;        // 생성 시간 (초)
        public int usedRandomSeed;          // 사용된 랜덤 시드
        public string errorMessage;         // 오류 메시지
        public Texture2D minimap;           // 생성된 미니맵 텍스처
        public CityGraph graph;             // 생성된 도시 그래프

        /// <summary>
        /// CityGenerationResult 생성자
        /// </summary>
        public CityGenerationResult()
        {
            success = false;
            buildingCount = 0;
            nodeCount = 0;
            edgeCount = 0;
            generationTime = 0f;
            usedRandomSeed = 0;
            errorMessage = string.Empty;
            minimap = null;
            graph = null;
        }
    }
}
