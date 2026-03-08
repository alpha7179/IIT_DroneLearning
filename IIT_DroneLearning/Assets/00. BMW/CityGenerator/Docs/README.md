# Procedural City Generator

## 개요

**Procedural City Generator**는 Unity 환경에서 드론 강화학습을 위한 프로시저럴 도시 환경을 자동 생성하는 시스템입니다. 이 시스템은 파라미터 기반으로 다양한 도시 구조를 생성하며, AI 에이전트가 활용할 수 있는 그래프 자료구조와 등고선 스타일 미니맵을 제공합니다.

### 주요 기능

- **파라미터 기반 도시 생성**: Inspector를 통한 직관적인 도시 환경 설정
- **재현 가능한 생성**: 랜덤 시드를 통한 동일 환경 재생성
- **3가지 레이아웃 모드**: PureGrid(완전 격자) · Hybrid(불규칙 블록) · PureRandom(유기적 도로망) 선택 가능
- **길 연결성 보장**: BFS 기반 알고리즘으로 Hybrid/PureRandom에서 막힌 구간 자동 해소
- **CityGenerator 중심 생성**: 컴포넌트가 붙은 GameObject 위치를 도시 중심으로 자동 정렬
- **AI 친화적 데이터 구조**: 경로 계획과 전략 수립을 위한 그래프 및 공간 분할 자료구조
- **전략적 위치 분석**: 은폐 지점, 교차로, 막다른 골목 등 자동 식별
- **탑뷰 미니맵 자동 저장**: 도시 생성 시 등고선 스타일 PNG를 CityMaps/ 폴더에 자동 저장
- **그래프 JSON·CSV 자동 내보내기**: 도시 그래프를 CityData/ 폴더에 자동 저장 (도망자 드론 오프라인 분석용)
- **런타임 쿼리 API**: 실시간 도시 정보 조회를 위한 최적화된 API
- **프리셋 시스템**: 자주 사용하는 파라미터 조합 저장 및 로드
- **성능 최적화**: 대규모 도시 생성 시에도 5초 이내 완료

### 기술 스택

- **플랫폼**: Unity 6000.0.69f1 LTS
- **언어**: C# (.NET Standard 2.1)
- **아키텍처**: MonoBehaviour 컴포넌트, 싱글톤, ScriptableObject
- **자료구조**: 그래프 (노드-엣지), Quadtree (공간 분할)
- **최적화**: 오브젝트 풀링, 배치 처리

## 설치 및 설정

### 1. 프로젝트 구조

시스템은 다음 디렉토리 구조로 구성되어 있습니다:

```
IIT_DroneLearning/Assets/00. BMW/ProceduralCityGenerator/
├── Scripts/
│   ├── CityGenerator.cs              # 도시 생성 핵심 컴포넌트 (레이아웃 모드 포함)
│   ├── CityParameters.cs             # 파라미터 프리셋 ScriptableObject
│   ├── CityGraph.cs                  # 그래프 자료구조
│   ├── CityGraphExporter.cs          # 그래프 JSON·CSV 내보내기
│   ├── SpatialIndex.cs               # Quadtree 공간 인덱스
│   ├── BuildingFactory.cs            # 건물 생성 및 오브젝트 풀링
│   ├── StrategicLocationAnalyzer.cs  # 전략적 위치 분석
│   ├── MinimapGenerator.cs           # 미니맵 생성 및 PNG 저장
│   ├── MinimapRenderer.cs            # 미니맵 UI 렌더링
│   ├── CityDataAPI.cs                # 런타임 쿼리 API
│   ├── DataStructures.cs             # 공통 데이터 구조체 (CityLayoutMode 포함)
│   └── Editor/
│       └── CityGeneratorEditor.cs    # Custom Inspector (레이아웃 모드 버튼 포함)
├── Presets/                          # 저장된 파라미터 프리셋
├── Materials/                        # 건물 머티리얼
└── (런타임 생성 폴더 — git 제외)
    ├── CityData/                     # 자동 생성: 그래프 JSON·CSV 파일
    └── CityMaps/                     # 자동 생성: 미니맵 PNG 파일
```

### 2. 기본 설정

1. Unity 씬에 빈 GameObject를 생성합니다
2. `CityGenerator` 컴포넌트를 추가합니다
3. Inspector에서 `Default Building Material`을 설정합니다 (선택사항)
4. 파라미터를 조정하고 "도시 생성" 버튼을 클릭합니다


## 파라미터 설명

### Grid Settings (격자 설정)

#### Unit Distance (단위 거리)
- **설명**: 격자 시스템의 1 단위가 나타내는 실제 거리 (미터)
- **기본값**: 1.0
- **범위**: 0.1 ~ 100.0
- **권장값**: 
  - 소형 도시: 0.5 ~ 1.0
  - 중형 도시: 1.0 ~ 2.0
  - 대형 도시: 2.0 ~ 5.0

#### Min/Max Width (최소/최대 가로 길이)
- **설명**: 도시의 가로 크기 범위 (격자 단위)
- **기본값**: Min 10, Max 20
- **범위**: 1 ~ 100
- **권장값**:
  - 테스트 환경: 5 ~ 10
  - 학습 환경: 15 ~ 30
  - 대규모 환경: 40 ~ 80

#### Min/Max Depth (최소/최대 세로 길이)
- **설명**: 도시의 세로 크기 범위 (격자 단위)
- **기본값**: Min 10, Max 20
- **범위**: 1 ~ 100
- **권장값**: Width와 동일하거나 유사한 값 사용

### Building Settings (건물 설정)

#### Building Width/Depth (건물 가로/세로)
- **설명**: 개별 건물의 가로 및 세로 크기 (단위_거리)
- **기본값**: 1.0
- **범위**: 0.5 ~ 50.0
- **권장값**:
  - 밀집 도시: 0.8 ~ 1.5
  - 일반 도시: 1.0 ~ 2.0
  - 넓은 건물: 2.0 ~ 5.0

#### Min/Max Building Height (최소/최대 건물 높이)
- **설명**: 건물 높이 범위 (단위_거리)
- **기본값**: Min 5.0, Max 20.0
- **범위**: 1.0 ~ 500.0
- **권장값**:
  - 저층 건물: 3.0 ~ 10.0
  - 중층 건물: 10.0 ~ 30.0
  - 고층 건물: 30.0 ~ 100.0
  - 초고층 건물: 100.0 ~ 200.0

#### Building Spacing (건물 간격)
- **설명**: 인접 건물 사이의 최소 거리 (단위_거리)
- **기본값**: 1.0
- **범위**: 0.0 ~ 50.0
- **권장값**:
  - 밀집 도시: 0.5 ~ 1.0
  - 일반 도시: 1.0 ~ 2.0
  - 넓은 간격: 2.0 ~ 5.0
  - 간격 없음: 0.0

#### Building Density (건물 밀도)
- **설명**: 격자 셀에 건물이 배치될 확률 (0.0 ~ 1.0)
- **기본값**: 0.7
- **범위**: 0.0 ~ 1.0
- **권장값**:
  - 희소한 도시: 0.3 ~ 0.5
  - 일반 도시: 0.6 ~ 0.8
  - 밀집 도시: 0.8 ~ 1.0
  - 완전 밀집: 1.0

### Generation Settings (생성 설정)

#### Random Seed (랜덤 시드)
- **설명**: 재현 가능한 도시 생성을 위한 시드값
- **기본값**: -1 (시간 기반 랜덤)
- **특수값**: -1을 설정하면 매번 다른 도시 생성
- **권장값**:
  - 실험 재현: 특정 정수 값 (예: 12345)
  - 다양한 환경: -1

#### Default Building Material (기본 건물 머티리얼)
- **설명**: 건물에 적용될 Unity Material
- **기본값**: null (Unity 기본 머티리얼 사용)
- **권장값**: 프로젝트에 맞는 커스텀 머티리얼 설정

### Minimap Settings (미니맵 설정)

#### Minimap Resolution (미니맵 해상도)
- **설명**: 생성될 미니맵의 해상도
- **기본값**: 512x512
- **옵션**: 256x256, 512x512, 1024x1024
- **권장값**:
  - 빠른 생성: 256x256
  - 균형: 512x512
  - 고품질: 1024x1024

### Layout Mode (레이아웃 모드)

Inspector에서 3개의 버튼 중 하나를 선택합니다. 선택된 버튼은 하늘색으로 강조됩니다.

#### layoutMode
- **기본값**: `PureGrid`
- **옵션**:
  | 모드 | 설명 |
  |------|------|
  | `PureGrid` | 완전 격자 — 균일한 블록 간격, 건물이 정렬된 그리드에 배치됨 |
  | `Hybrid` | 격자 + 랜덤 오프셋 — 불규칙한 블록 간격과 건물 위치·크기 변화 |
  | `PureRandom` | 유기적 도로망 — 구불구불한 주간선 도로와 자유 배치, 가장 현실적인 도시 형태 |

#### Offset Strength (오프셋 세기) — Hybrid / PureRandom 전용
- **설명**: 격자 위치에서 건물이 얼마나 벗어날 수 있는지의 세기
- **기본값**: 0.5
- **범위**: 0.0 ~ 1.0
- **비고**: 내부적으로 `cellSize × offsetStrength × 0.45`로 제한되어 도로와 시각적 겹침 방지

#### Min Block Size (최소 블록 크기) — Hybrid / PureRandom 전용
- **설명**: 도로 사이 블록의 최소 크기 (격자 단위)
- **기본값**: 3
- **범위**: 2 ~ 8
- **비고**: 값이 작을수록 골목이 촘촘하게 생성됨

#### Max Block Size (최대 블록 크기) — Hybrid / PureRandom 전용
- **설명**: 도로 사이 블록의 최대 크기 (격자 단위)
- **기본값**: 6
- **범위**: 3 ~ 12
- **비고**: Min Block Size보다 커야 함 (Inspector에서 경고 표시)

### 길 연결성 보장 (EnsureRoadConnectivity)

Hybrid 및 PureRandom 모드에서는 건물 배치 후 BFS 기반 연결성 검사가 자동으로 실행됩니다.

- **방식**: 빈 셀(도로/공간)의 연결 컴포넌트를 BFS로 분석 → 고립된 컴포넌트와 주 컴포넌트의 경계에서 가장 낮은(작은) 건물을 자동 제거
- **최대 반복**: `min(건물수/5 + 1, 50)` 회
- **PureGrid**: 항상 연결됨 → 실행 건너뜀


## Inspector UI 사용 방법

### 1. 레이아웃 모드 선택

Inspector의 **Layout Mode** 섹션에 3개의 토글 버튼이 표시됩니다:

```
[ Pure Grid ]  [ Hybrid ]  [ Pure Random ]
```

- 선택된 버튼은 **하늘색(Sky Blue)** 배경 + **볼드** 텍스트로 강조됩니다
- 미선택 버튼은 회색 배경으로 표시됩니다
- `Hybrid` 또는 `Pure Random` 선택 시 **Offset Strength**, **Min/Max Block Size** 슬라이더가 추가로 표시됩니다
- `Min Block Size > Max Block Size`일 경우 노란색 경고 박스가 표시됩니다

### 2. 도시 생성

1. Unity Editor에서 `CityGenerator` 컴포넌트가 부착된 GameObject를 선택합니다
2. Inspector에서 원하는 파라미터를 설정합니다
3. **Layout Mode** 버튼에서 원하는 레이아웃을 선택합니다 (기본: Pure Grid)
4. **"도시 생성 (Generate City)"** 버튼을 클릭합니다
5. 생성이 완료되면 Console에 결과가 출력됩니다
6. `Assets/CityMaps/` 폴더에 미니맵 PNG가, `Assets/CityData/` 폴더에 그래프 JSON·CSV가 자동 저장됩니다

**생성 결과 예시 (Hybrid 모드, Seed 42):**
```
[CityGenerator] Hybrid 도로 마스크 완료. 도로 셀: 87 / 400
[CityGenerator] 연결성 보장: 건물 3개 제거 → 모든 빈 셀 연결됨
[MinimapGenerator] 미니맵 저장 완료: Assets/CityMaps/Minimap_Seed42_Hybrid_512x512.png
[CityGraphExporter] 그래프 내보내기 완료: Assets/CityData/City_Seed42_Hybrid.json
[CityGraphExporter] CSV 내보내기 완료: Assets/CityData/City_Seed42_Hybrid_nodes.csv
[CityGenerator] 도시 생성 완료! 건물: 218개, 노드: 441개, 엣지: 1560개, 시드: 42
```

### 3. 도시 초기화

- **"도시 초기화 (Clear City)"** 버튼을 클릭하면 확인 다이얼로그 후 생성된 모든 건물이 제거됩니다
- 새로운 도시를 생성하기 전에 자동으로 이전 도시가 제거됩니다

### 4. 프리셋 저장

1. 원하는 파라미터를 설정합니다
2. **"프리셋 저장 (Save Preset)"** 버튼을 클릭합니다
3. 프리셋 이름을 입력합니다 (예: "DenseCity", "OpenArea")
4. 프리셋이 `Assets/CityPresets/` 디렉토리에 저장됩니다

### 5. 프리셋 로드

1. **"프리셋 로드 (Load Preset)"** 버튼을 클릭합니다
2. `Assets/CityPresets/` 디렉토리에서 원하는 프리셋을 선택합니다
3. 선택한 프리셋의 파라미터가 자동으로 적용됩니다
4. 프리셋에는 `layoutMode`를 포함한 모든 파라미터가 저장됩니다


## API 사용 예제

### 1. 기본 설정

CityDataAPI는 싱글톤 패턴으로 구현되어 있어 어디서든 접근 가능합니다:

```csharp
using ProceduralCityGenerator;
using UnityEngine;

public class DroneAI : MonoBehaviour
{
    void Start()
    {
        // API가 초기화되었는지 확인
        if (CityDataAPI.Instance.IsInitialized())
        {
            Debug.Log("CityDataAPI is ready!");
        }
    }
}
```

### 2. 위치 기반 쿼리

#### 가장 가까운 노드 찾기

```csharp
// 드론의 현재 위치에서 가장 가까운 그래프 노드 찾기
Vector3 dronePosition = transform.position;
GraphNode nearestNode = CityDataAPI.Instance.GetNodeAtPosition(dronePosition);

Debug.Log($"가장 가까운 노드: ID {nearestNode.nodeId}, 위치 {nearestNode.position}");
```

#### 반경 내 노드 검색

```csharp
// 드론 주변 50미터 내의 모든 노드 찾기
Vector3 center = transform.position;
float radius = 50f;
List<GraphNode> nearbyNodes = CityDataAPI.Instance.GetNodesInRadius(center, radius);

Debug.Log($"반경 {radius}m 내에 {nearbyNodes.Count}개의 노드 발견");
```

### 3. 경로 계획

#### 최단 경로 계산

```csharp
// 현재 위치에서 목표 위치까지의 최단 경로 계산
GraphNode startNode = CityDataAPI.Instance.GetNodeAtPosition(transform.position);
GraphNode endNode = CityDataAPI.Instance.GetNodeAtPosition(targetPosition);

List<int> pathNodeIds = CityDataAPI.Instance.GetShortestPath(startNode.nodeId, endNode.nodeId);

if (pathNodeIds.Count > 0)
{
    Debug.Log($"경로 발견: {pathNodeIds.Count}개의 노드를 거쳐 이동");

    // 경로를 따라 이동 (GetNodeById로 O(1) 직접 조회)
    foreach (int nodeId in pathNodeIds)
    {
        GraphNode node = CityDataAPI.Instance.GetNodeById(nodeId);
        // 노드 위치로 이동하는 로직 구현
    }
}
```

#### 인접 노드 탐색

```csharp
// 현재 노드의 인접 노드 가져오기
int currentNodeId = nearestNode.nodeId;
List<GraphNode> neighbors = CityDataAPI.Instance.GetNeighborNodes(currentNodeId);

ProceduralCityGeneratorDebug.Log($"노드 {currentNodeId}는 {neighbors.Count}개의 인접 노드를 가지고 있습니다");

foreach (GraphNode neighbor in neighbors)
{
    Debug.Log($"  - 인접 노드 {neighbor.nodeId}: {neighbor.nodeType}");
}
```

### 4. 전략적 위치 활용

#### 은폐 지점 찾기

```csharp
// 추적자로부터 숨을 수 있는 은폐 지점 찾기
Vector3 dronePosition = transform.position;
float searchRadius = 100f;

List<StrategicLocation> coverPoints = CityDataAPI.Instance.GetCoverPoints(
    dronePosition, 
    searchRadius
);

if (coverPoints.Count > 0)
{
    // 가장 가까운 은폐 지점으로 이동
    StrategicLocation nearestCover = coverPoints[0];
    float minDistance = Vector3.Distance(dronePosition, nearestCover.position);
    
    foreach (StrategicLocation cover in coverPoints)
    {
        float distance = Vector3.Distance(dronePosition, cover.position);
        if (distance < minDistance)
        {
            minDistance = distance;
            nearestCover = cover;
        }
    }
    
    Debug.Log($"은폐 지점 발견: {nearestCover.position}, 위험도: {nearestCover.dangerScore}");
    // 은폐 지점으로 이동하는 로직 구현
}
```

#### 특정 전략적 위치 찾기

```csharp
// 가장 가까운 교차로 찾기
StrategicLocation nearestIntersection = CityDataAPI.Instance.GetNearestStrategicLocation(
    transform.position,
    StrategyType.Intersection
);

if (nearestIntersection.connectedNodes != null && nearestIntersection.connectedNodes.Count > 0)
{
    Debug.Log($"교차로 발견: {nearestIntersection.position}");
    // 교차로를 활용한 전략 수립
}

// 막다른 골목 피하기
StrategicLocation nearestDeadEnd = CityDataAPI.Instance.GetNearestStrategicLocation(
    transform.position,
    StrategyType.DeadEnd
);

if (nearestDeadEnd.connectedNodes != null)
{
    Debug.Log($"막다른 골목 경고: {nearestDeadEnd.position}");
    // 막다른 골목을 피하는 경로 계획
}
```

### 5. 가시성 확인

#### 추적자로부터 보이는지 확인

```csharp
// 드론이 추적자로부터 보이는지 확인
Vector3 dronePosition = transform.position;
Vector3 pursuerPosition = pursuer.transform.position;

bool isVisible = CityDataAPI.Instance.IsPositionVisible(pursuerPosition, dronePosition);

if (isVisible)
{
    Debug.Log("경고: 추적자에게 노출됨! 은폐 지점으로 이동 필요");
    // 은폐 전략 실행
}
else
{
    Debug.Log("안전: 건물에 의해 가려짐");
    // 현재 위치 유지 또는 다른 전략 실행
}
```


### 6. 미니맵 실시간 업데이트

#### 미니맵 초기화

```csharp
using ProceduralCityGenerator;
using UnityEngine;
using UnityEngine.UI;

public class MinimapController : MonoBehaviour
{
    public RawImage minimapImage;
    private MinimapRenderer minimapRenderer;
    
    void Start()
    {
        // MinimapRenderer 컴포넌트 가져오기
        minimapRenderer = GetComponent<MinimapRenderer>();
        
        if (minimapRenderer == null)
        {
            minimapRenderer = gameObject.AddComponent<MinimapRenderer>();
        }
        
        // 미니맵 이미지 설정
        minimapRenderer.minimapImage = minimapImage;
        
        // 생성된 미니맵 텍스처로 초기화 (CityGenerator에서 생성된 텍스처 사용)
        // Texture2D minimap = ...; // CityGenerator에서 가져온 미니맵
        // float pixelsPerMeter = ...; // 미니맵 스케일 정보
        // Bounds cityBounds = ...; // 도시 경계 정보
        // minimapRenderer.Initialize(minimap, pixelsPerMeter, cityBounds);
    }
}
```

#### 동적 마커 추가

```csharp
// 도망자 드론 위치 표시
Vector3 evaderPosition = evaderDrone.transform.position;
minimapRenderer.AddDynamicMarker(evaderPosition, MarkerType.EvaderDrone);

// 추적자 드론 위치 표시
Vector3 pursuerPosition = pursuerDrone.transform.position;
minimapRenderer.AddDynamicMarker(pursuerPosition, MarkerType.PursuerDrone);

// 목표 지점 표시
Vector3 targetPosition = targetPoint.position;
minimapRenderer.AddDynamicMarker(targetPosition, MarkerType.TargetPoint);
```

#### 마커 위치 업데이트

```csharp
void Update()
{
    // 드론이 이동할 때마다 미니맵 마커 업데이트
    Vector3 oldPosition = previousDronePosition;
    Vector3 newPosition = drone.transform.position;
    
    minimapRenderer.UpdateMarkerPosition(oldPosition, newPosition, MarkerType.EvaderDrone);
    
    previousDronePosition = newPosition;
}
```

#### 경로 표시

```csharp
// 계획된 경로를 미니맵에 표시
List<int> pathNodeIds = CityDataAPI.Instance.GetShortestPath(startNodeId, endNodeId);
List<Vector3> pathPositions = new List<Vector3>();

// 경로를 따라 이동 (GetNodeById로 O(1) 직접 조회)
foreach (int nodeId in pathNodeIds)
{
    GraphNode node = CityDataAPI.Instance.GetNodeById(nodeId);
    pathPositions.Add(node.position);
}

// 경로를 초록색 선으로 표시
minimapRenderer.DrawPath(pathPositions, Color.green);
```

### 7. 완전한 드론 AI 예제

```csharp
using ProceduralCityGenerator;
using UnityEngine;
using System.Collections.Generic;

public class EvaderDroneAI : MonoBehaviour
{
    public Transform pursuer;
    public float detectionRadius = 100f;
    public float moveSpeed = 10f;
    
    private List<int> currentPath;
    private int currentPathIndex = 0;
    
    void Update()
    {
        // 1. 추적자로부터 보이는지 확인
        bool isVisible = CityDataAPI.Instance.IsPositionVisible(
            pursuer.position, 
            transform.position
        );
        
        if (isVisible)
        {
            // 2. 은폐 지점 찾기
            List<StrategicLocation> coverPoints = CityDataAPI.Instance.GetCoverPoints(
                transform.position, 
                detectionRadius
            );
            
            if (coverPoints.Count > 0)
            {
                // 3. 가장 가까운 은폐 지점으로 경로 계획
                StrategicLocation targetCover = FindBestCoverPoint(coverPoints);
                PlanPathToCover(targetCover);
            }
        }
        
        // 4. 계획된 경로를 따라 이동
        if (currentPath != null && currentPath.Count > 0)
        {
            FollowPath();
        }
    }
    
    StrategicLocation FindBestCoverPoint(List<StrategicLocation> coverPoints)
    {
        StrategicLocation best = coverPoints[0];
        float bestScore = float.MinValue;
        
        foreach (StrategicLocation cover in coverPoints)
        {
            // 거리와 위험도를 고려한 점수 계산
            float distance = Vector3.Distance(transform.position, cover.position);
            float score = (1f / distance) * (1f - cover.dangerScore);
            
            if (score > bestScore)
            {
                bestScore = score;
                best = cover;
            }
        }
        
        return best;
    }
    
    void PlanPathToCover(StrategicLocation target)
    {
        GraphNode startNode = CityDataAPI.Instance.GetNodeAtPosition(transform.position);
        GraphNode endNode = CityDataAPI.Instance.GetNodeAtPosition(target.position);
        
        currentPath = CityDataAPI.Instance.GetShortestPath(startNode.nodeId, endNode.nodeId);
        currentPathIndex = 0;
        
        Debug.Log($"경로 계획 완료: {currentPath.Count}개 노드");
    }
    
    void FollowPath()
    {
        if (currentPathIndex >= currentPath.Count)
        {
            currentPath = null;
            return;
        }
        
        // 현재 목표 노드 가져오기 (GetNodeById로 O(1) 직접 조회)
        int targetNodeId = currentPath[currentPathIndex];
        GraphNode targetNode = CityDataAPI.Instance.GetNodeById(targetNodeId);
        
        // 목표 노드로 이동
        Vector3 direction = (targetNode.position - transform.position).normalized;
        transform.position += direction * moveSpeed * Time.deltaTime;
        
        // 목표 노드에 도달했는지 확인
        if (Vector3.Distance(transform.position, targetNode.position) < 1f)
        {
            currentPathIndex++;
        }
    }
}
```


## 데이터 구조 참조

### GraphNode (그래프 노드)

```csharp
public struct GraphNode
{
    public int nodeId;                          // 고유 식별자
    public Vector3 position;                    // 월드 좌표
    public NodeType nodeType;                   // 노드 타입
    public float elevation;                     // 고도 정보
    public float[] surroundingBuildingHeights; // 주변 건물 높이
    public List<StrategyType> strategicMarkers; // 전략적 마커
    public bool isVisibleFromSpawn;            // 생성 지점에서 보이는지 여부
}
```

**NodeType 열거형:**
- `OpenSpace`: 개방 공간
- `BuildingCorner`: 건물 모서리
- `AlleyEntrance`: 골목 입구
- `Intersection`: 교차로

### GraphEdge (그래프 엣지)

```csharp
public struct GraphEdge
{
    public int startNodeId;         // 시작 노드 ID
    public int endNodeId;           // 끝 노드 ID
    public float travelCost;        // 이동 비용 (거리 기반)
    public float visibilityScore;   // 가시성 점수 (0.0 ~ 1.0)
    public PathType pathType;       // 경로 타입
}
```

**PathType 열거형:**
- `Direct`: 직선 경로
- `Detour`: 우회 경로
- `Concealed`: 은폐 경로

### StrategicLocation (전략적 위치)

```csharp
public struct StrategicLocation
{
    public Vector3 position;              // 위치 좌표
    public StrategyType locationType;     // 위치 타입
    public float dangerScore;             // 위험도 점수 (0.0 ~ 1.0)
    public float visibilityScore;         // 가시성 점수 (0.0 ~ 1.0)
    public List<int> connectedNodes;      // 연결된 노드 ID 리스트
}
```

**StrategyType 열거형:**
- `CoverPoint`: 은폐 지점 (파란색)
- `Intersection`: 교차로 (노란색)
- `DeadEnd`: 막다른 골목 (빨간색)
- `OpenArea`: 개방 구역 (주황색)
- `DetourPath`: 우회 경로 (초록색)

### Building (건물)

```csharp
public class Building
{
    public GameObject gameObject;   // Unity GameObject
    public Vector3 position;        // 월드 위치
    public Vector3 size;           // 크기 (width, height, depth)
    public float height;           // 높이
    public GridCell gridCell;      // 격자 셀 정보
    public Bounds bounds;          // 경계 박스
}
```


## 성능 가이드라인

### 권장 파라미터 조합

#### 빠른 테스트 환경
```
Grid Size: 5x5 ~ 10x10
Building Density: 0.5 ~ 0.7
Building Height: 5 ~ 15
예상 생성 시간: < 1초
예상 건물 수: 15 ~ 70개
```

#### 일반 학습 환경
```
Grid Size: 15x15 ~ 25x25
Building Density: 0.6 ~ 0.8
Building Height: 10 ~ 30
예상 생성 시간: 1 ~ 3초
예상 건물 수: 135 ~ 500개
```

#### 대규모 환경
```
Grid Size: 40x40 ~ 60x60
Building Density: 0.7 ~ 0.9
Building Height: 20 ~ 100
예상 생성 시간: 3 ~ 5초
예상 건물 수: 1120 ~ 3240개
```

### 성능 최적화 팁

1. **건물 밀도 조절**: 밀도를 낮추면 생성 시간이 단축됩니다
2. **격자 크기 제한**: 100x100 이상의 격자는 생성 시간이 크게 증가합니다
3. **미니맵 해상도**: 256x256 해상도는 512x512보다 4배 빠릅니다
4. **오브젝트 풀링**: BuildingFactory가 자동으로 오브젝트 풀링을 처리합니다
5. **배치 처리**: 대규모 도시 생성 시 자동으로 배치 처리가 적용됩니다

### 성능 벤치마크

테스트 환경: Unity 6000.0.69f1, Intel i7-10700K, 32GB RAM

| 격자 크기 | 밀도 | 건물 수 | 생성 시간 | 노드 수 | 쿼리 시간 (평균) |
|----------|------|---------|----------|---------|-----------------|
| 10x10    | 0.7  | 70      | 0.5초    | 121     | < 0.1ms         |
| 20x20    | 0.7  | 280     | 1.2초    | 441     | < 0.2ms         |
| 40x40    | 0.7  | 1120    | 3.8초    | 1681    | < 0.5ms         |
| 60x60    | 0.8  | 2880    | 4.9초    | 3721    | < 0.8ms         |

### API 쿼리 성능

모든 쿼리 메서드는 최적화된 시간 복잡도를 제공합니다:

- `GetNodeById()`: O(1) - Dictionary 직접 접근
- `GetNodeAtPosition()`: O(log n) - Quadtree 사용
- `GetNodesInRadius()`: O(log n + k) - k는 결과 노드 수
- `GetNeighborNodes()`: O(1) - Dictionary 직접 접근
- `GetNearestStrategicLocation()`: O(k) - 타입별 캐시 활용, k는 해당 타입 위치 수
- `GetShortestPath()`: O(E log V) - Dijkstra 알고리즘
- `IsPositionVisible()`: O(1) - 단일 레이캐스트


## 문제 해결 (Troubleshooting)

### 일반적인 문제

#### 1. 도시가 생성되지 않음

**증상**: "도시 생성" 버튼을 클릭해도 아무 일도 일어나지 않음

**해결 방법**:
- Console 창에서 오류 메시지 확인
- 파라미터 검증 실패 여부 확인 (최소값 > 최대값 등)
- `defaultBuildingMaterial`이 설정되어 있는지 확인 (선택사항이지만 권장)

#### 2. 생성 시간이 너무 오래 걸림

**증상**: 진행률 표시줄이 5초 이상 지속됨

**해결 방법**:
- 격자 크기를 줄입니다 (예: 60x60 → 40x40)
- 건물 밀도를 낮춥니다 (예: 0.9 → 0.7)
- 미니맵 해상도를 낮춥니다 (예: 1024 → 512)
- Cancel 버튼을 클릭하여 생성을 중단하고 파라미터를 조정합니다

#### 3. CityDataAPI가 초기화되지 않음

**증상**: `CityDataAPI.Instance.IsInitialized()` 가 false 반환

**해결 방법**:
- 씬에 `CityDataAPI` 컴포넌트가 있는 GameObject가 있는지 확인
- 없다면 빈 GameObject를 생성하고 `CityDataAPI` 컴포넌트를 추가
- 또는 `CityDataAPI.Instance`를 호출하면 자동으로 생성됩니다
- 도시를 먼저 생성한 후 API를 사용해야 합니다

#### 4. 프리셋 저장/로드 실패

**증상**: 프리셋 저장 시 오류 발생 또는 로드되지 않음

**해결 방법**:
- `Assets/CityPresets` 디렉토리가 존재하는지 확인
- 없다면 수동으로 생성하거나 첫 저장 시 자동 생성됩니다
- 프리셋 이름에 특수 문자가 포함되지 않았는지 확인
- Unity Editor에서만 프리셋 기능이 작동합니다 (빌드된 게임에서는 불가)

#### 5. 미니맵이 표시되지 않음

**증상**: 미니맵 텍스처가 생성되지 않거나 UI에 표시되지 않음

**해결 방법**:
- `MinimapGenerator` 기능이 현재 구현 중입니다
- `Assets/CityMaps` 디렉토리가 존재하는지 확인
- `MinimapRenderer` 컴포넌트가 올바르게 설정되었는지 확인
- `RawImage` 컴포넌트가 UI Canvas에 있는지 확인

#### 6. 경로 탐색이 실패함

**증상**: `GetShortestPath()` 가 빈 리스트 반환

**해결 방법**:
- 시작 노드와 끝 노드가 연결되어 있는지 확인
- 건물 밀도가 너무 높으면 (1.0) 경로가 차단될 수 있습니다
- 건물 간격을 늘려서 이동 가능한 공간을 확보합니다
- Console에서 경고 메시지 확인

#### 7. Hybrid/PureRandom 모드에서 건물이 너무 적게 생성됨

**증상**: PureGrid 대비 건물 수가 현저히 적음

**해결 방법**:
- 이는 정상 동작입니다 — 도로 마스크가 도로 공간을 예약하므로 건물 셀이 줄어듭니다
- 건물 밀도를 높이거나 (`buildingDensity` ↑) `minBlockSize`를 늘려 도로 비율을 줄이세요
- `EnsureRoadConnectivity`가 추가로 일부 건물을 제거할 수 있습니다 (Console 메시지 확인)

#### 8. 메모리 부족 오류

**증상**: 대규모 도시 생성 시 OutOfMemoryException 발생

**해결 방법**:
- 격자 크기를 줄입니다 (80x80 이하 권장)
- 이전 도시를 제거한 후 새 도시를 생성합니다
- Unity Editor를 재시작합니다
- 64비트 Unity Editor를 사용하는지 확인합니다

### 디버깅 팁

#### Console 로그 활용

시스템은 다양한 로그 메시지를 출력합니다:

**PureGrid 모드 (Seed 12345):**
```
[CityGenerator] 도시 생성 시작
[CityGenerator] 랜덤 시드 초기화 완료. 사용된 시드: 12345
[CityGenerator] 도시 크기 결정 완료. 가로: 20, 세로: 20
[CityGenerator] 건물 배치 완료. 총 280개의 건물 생성
[CityGenerator] 그래프 구축 완료. 노드: 441, 엣지: 1680
[MinimapGenerator] 미니맵 저장 완료: Assets/CityMaps/Minimap_Seed12345_PureGrid_512x512.png
[CityGraphExporter] 그래프 내보내기 완료: Assets/CityData/City_Seed12345_PureGrid.json
[CityGenerator] 도시 생성 완료! 건물: 280개, 노드: 441개, 엣지: 1680개, 시드: 12345
```

**Hybrid 모드 (Seed 42):**
```
[CityGenerator] Hybrid 도로 마스크 완료. 도로 셀: 87 / 400
[CityGenerator] 건물 배치 완료. 총 218개의 건물 생성
[CityGenerator] 연결성 보장: 건물 3개 제거 → 모든 빈 셀 연결됨
[MinimapGenerator] 미니맵 저장 완료: Assets/CityMaps/Minimap_Seed42_Hybrid_512x512.png
[CityGraphExporter] 그래프 내보내기 완료: Assets/CityData/City_Seed42_Hybrid.json
```

**PureRandom 모드 (Seed 7):**
```
[CityGenerator] PureRandom 도로 마스크 완료. 도로 셀: 124 / 400
[CityGenerator] 건물 배치 완료. 총 183개의 건물 생성
[CityGenerator] 연결성 보장: 건물 5개 제거 → 모든 빈 셀 연결됨
[MinimapGenerator] 미니맵 저장 완료: Assets/CityMaps/Minimap_Seed7_PureRandom_512x512.png
[CityGraphExporter] 그래프 내보내기 완료: Assets/CityData/City_Seed7_PureRandom.json
```

#### Scene View에서 확인

- Hierarchy에서 `CityGenerator` 컴포넌트가 붙은 GameObject의 하위 건물 오브젝트를 찾습니다
- 각 건물은 "Building_X_Z" 형식으로 이름이 지정됩니다
- Scene View에서 도시 구조를 시각적으로 확인합니다

#### Gizmos 활용 (선택사항)

그래프 노드와 엣지를 시각화하려면 커스텀 Gizmos를 추가할 수 있습니다:

```csharp
void OnDrawGizmos()
{
    if (!CityDataAPI.Instance.IsInitialized()) return;
    
    // 모든 노드를 작은 구체로 표시
    List<GraphNode> nodes = CityDataAPI.Instance.GetNodesInRadius(
        Vector3.zero, 
        1000f
    );
    
    foreach (GraphNode node in nodes)
    {
        Gizmos.color = Color.yellow;
        Gizmos.DrawSphere(node.position, 0.5f);
    }
}
```


## 고급 사용법

### 1. 커스텀 건물 머티리얼

다양한 건물 외관을 위해 커스텀 머티리얼을 사용할 수 있습니다:

```csharp
// 높이에 따라 다른 머티리얼 적용
public class CustomBuildingMaterializer : MonoBehaviour
{
    public Material lowBuildingMaterial;
    public Material midBuildingMaterial;
    public Material highBuildingMaterial;
    
    void Start()
    {
        // 도시 생성 후 호출
        ApplyCustomMaterials();
    }
    
    void ApplyCustomMaterials()
    {
        GameObject cityRoot = cityGenerator.gameObject; // CityGenerator가 붙은 오브젝트
        if (cityRoot == null) return;
        
        foreach (Transform building in cityRoot.transform)
        {
            float height = building.localScale.y;
            MeshRenderer renderer = building.GetComponent<MeshRenderer>();
            
            if (height < 10f)
                renderer.material = lowBuildingMaterial;
            else if (height < 30f)
                renderer.material = midBuildingMaterial;
            else
                renderer.material = highBuildingMaterial;
        }
    }
}
```

### 2. 그래프 데이터 저장 및 로드

생성된 그래프를 파일로 저장하고 나중에 로드할 수 있습니다:

```csharp
using System.IO;

public class GraphSerializer : MonoBehaviour
{
    public void SaveGraphToFile(CityGraph graph, string filename)
    {
        string json = graph.SerializeToJson();
        string path = Path.Combine(Application.persistentDataPath, filename);
        File.WriteAllText(path, json);
        Debug.Log($"그래프 저장 완료: {path}");
    }
    
    public CityGraph LoadGraphFromFile(string filename)
    {
        string path = Path.Combine(Application.persistentDataPath, filename);
        if (!File.Exists(path))
        {
            Debug.LogError($"파일을 찾을 수 없습니다: {path}");
            return null;
        }
        
        string json = File.ReadAllText(path);
        CityGraph graph = new CityGraph();
        graph.DeserializeFromJson(json);
        Debug.Log($"그래프 로드 완료: {path}");
        return graph;
    }
}
```

### 3. 동적 장애물 추가

런타임에 동적 장애물을 추가하고 그래프를 업데이트할 수 있습니다:

```csharp
public class DynamicObstacleManager : MonoBehaviour
{
    public void AddObstacle(Vector3 position, float radius)
    {
        // 장애물 주변의 노드를 비활성화
        List<GraphNode> affectedNodes = CityDataAPI.Instance.GetNodesInRadius(
            position, 
            radius
        );
        
        foreach (GraphNode node in affectedNodes)
        {
            // 노드를 비활성화하거나 이동 비용을 증가시킴
            // 실제 구현은 CityGraph에 노드 비활성화 메서드 추가 필요
            Debug.Log($"노드 {node.nodeId} 영향받음");
        }
    }
}
```

### 4. 멀티 레이어 도시

여러 층의 도시를 생성하여 3D 공간 활용:

```csharp
public class MultiLayerCityGenerator : MonoBehaviour
{
    public CityGenerator cityGenerator;
    public int numberOfLayers = 3;
    public float layerHeight = 50f;
    
    public void GenerateMultiLayerCity()
    {
        for (int layer = 0; layer < numberOfLayers; layer++)
        {
            // 각 층마다 도시 생성
            cityGenerator.GenerateCity();
            
            // 생성된 도시를 위로 이동
            GameObject cityRoot = cityGenerator.gameObject; // CityGenerator가 붙은 오브젝트
            if (cityRoot != null)
            {
                cityRoot.name = $"City_Layer{layer}";
                cityRoot.transform.position = new Vector3(0, layer * layerHeight, 0);
            }
        }
    }
}
```

### 5. 강화학습 환경 통합

ML-Agents와 통합하여 강화학습 환경으로 사용:

```csharp
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class DroneAgent : Agent
{
    public Transform target;
    private Vector3 startPosition;
    
    public override void OnEpisodeBegin()
    {
        // 에피소드 시작 시 새로운 도시 생성 (선택사항)
        // cityGenerator.GenerateCity();
        
        // 드론을 랜덤 위치에 배치
        GraphNode randomNode = GetRandomNode();
        transform.position = randomNode.position + Vector3.up * 5f;
        startPosition = transform.position;
    }
    
    public override void CollectObservations(VectorSensor sensor)
    {
        // 드론의 현재 위치
        sensor.AddObservation(transform.position);
        
        // 목표까지의 방향
        sensor.AddObservation(target.position - transform.position);
        
        // 가장 가까운 은폐 지점
        List<StrategicLocation> covers = CityDataAPI.Instance.GetCoverPoints(
            transform.position, 
            50f
        );
        
        if (covers.Count > 0)
        {
            sensor.AddObservation(covers[0].position);
        }
        else
        {
            sensor.AddObservation(Vector3.zero);
        }
        
        // 추적자로부터의 가시성
        bool isVisible = CityDataAPI.Instance.IsPositionVisible(
            target.position, 
            transform.position
        );
        sensor.AddObservation(isVisible ? 1f : 0f);
    }
    
    public override void OnActionReceived(ActionBuffers actions)
    {
        // 행동 처리
        float moveX = actions.ContinuousActions[0];
        float moveZ = actions.ContinuousActions[1];
        float moveY = actions.ContinuousActions[2];
        
        Vector3 move = new Vector3(moveX, moveY, moveZ) * Time.deltaTime * 10f;
        transform.position += move;
        
        // 보상 계산
        float distanceToTarget = Vector3.Distance(transform.position, target.position);
        
        // 목표에 가까워지면 보상
        if (distanceToTarget < 5f)
        {
            SetReward(1.0f);
            EndEpisode();
        }
        
        // 추적자에게 보이면 페널티
        bool isVisible = CityDataAPI.Instance.IsPositionVisible(
            target.position, 
            transform.position
        );
        
        if (isVisible)
        {
            AddReward(-0.01f);
        }
        else
        {
            AddReward(0.001f); // 숨어있으면 작은 보상
        }
    }
    
    GraphNode GetRandomNode()
    {
        List<GraphNode> allNodes = CityDataAPI.Instance.GetNodesInRadius(
            Vector3.zero, 
            1000f
        );
        
        if (allNodes.Count == 0)
        {
            return default;
        }
        
        return allNodes[Random.Range(0, allNodes.Count)];
    }
}
```


## 프리셋 예제

시스템과 함께 제공되는 권장 프리셋 설정:

### Dense Urban (밀집 도시)
```
Unit Distance: 1.0
Grid Size: 30x30 ~ 40x40
Building Width/Depth: 1.0
Building Height: 15 ~ 50
Building Spacing: 0.5
Building Density: 0.9
Random Seed: 12345

용도: 복잡한 가림 현상, 좁은 골목, 높은 난이도
```

### Open City (개방 도시)
```
Unit Distance: 2.0
Grid Size: 20x20 ~ 30x30
Building Width/Depth: 1.5
Building Height: 5 ~ 20
Building Spacing: 3.0
Building Density: 0.5

용도: 넓은 이동 공간, 낮은 난이도, 초보 학습
```

### Skyscraper District (고층 빌딩 지구)
```
Unit Distance: 1.5
Grid Size: 15x15 ~ 25x25
Building Width/Depth: 2.0
Building Height: 50 ~ 150
Building Spacing: 2.0
Building Density: 0.6

용도: 수직 공간 활용, 고도 변화, 중급 학습
```

### Mixed Heights (혼합 높이)
```
Unit Distance: 1.0
Grid Size: 25x25 ~ 35x35
Building Width/Depth: 1.0
Building Height: 3 ~ 80
Building Spacing: 1.5
Building Density: 0.7

용도: 다양한 높이, 예측 불가능한 환경, 고급 학습
```

### Maze City (미로 도시)
```
Unit Distance: 0.8
Grid Size: 40x40 ~ 50x50
Building Width/Depth: 0.8
Building Height: 10 ~ 25
Building Spacing: 0.3
Building Density: 0.95

용도: 복잡한 경로 계획, 좁은 통로, 최고 난이도
```

### Organic City — Hybrid (유기적 혼합 도시)
```
Layout Mode: Hybrid
Unit Distance: 1.5
Grid Size: 25x25 ~ 35x35
Building Width/Depth: 1.2
Building Height: 8 ~ 40
Building Spacing: 1.0
Building Density: 0.75
Offset Strength: 0.6
Min Block Size: 3 / Max Block Size: 6

용도: 불규칙한 블록과 골목, 현실적 도시 느낌, 중간 난이도
```

### Organic City — Pure Random (완전 유기적 도시)
```
Layout Mode: PureRandom
Unit Distance: 1.5
Grid Size: 30x30 ~ 40x40
Building Width/Depth: 1.0
Building Height: 5 ~ 50
Building Spacing: 0.8
Building Density: 0.80
Offset Strength: 0.8
Min Block Size: 4 / Max Block Size: 8

용도: 가장 현실적인 도시 형태, LOS 차단 최다, RL 고난이도 훈련
```

## 시스템 아키텍처

### 컴포넌트 관계도

```
CityGenerator (중심 컴포넌트)
    ├── CityParameters (파라미터 저장)
    ├── [CityLayoutMode 결정]
    │   ├── roadMask (bool[,]) — Hybrid/PureRandom 도로 마스크
    │   └── EnsureRoadConnectivity() — BFS 연결성 보장
    ├── BuildingFactory (건물 생성)
    │   └── Object Pool (성능 최적화)
    ├── CityGraph (그래프 자료구조)
    │   ├── GraphNode (노드)
    │   └── GraphEdge (엣지)
    ├── SpatialIndex (Quadtree)
    │   └── QuadtreeNode (내부 노드)
    ├── StrategicLocationAnalyzer (전략 분석)
    │   └── StrategicLocation (전략적 위치)
    ├── MinimapGenerator (미니맵 생성)
    │   └── → Assets/CityMaps/Minimap_Seed{N}_{Mode}_{Res}.png
    ├── CityGraphExporter (그래프 내보내기)
    │   ├── → Assets/CityData/City_Seed{N}_{Mode}.json
    │   └── → Assets/CityData/City_Seed{N}_{Mode}_nodes.csv
    └── CityDataAPI (런타임 API)
        └── Singleton Instance
```

### 데이터 흐름

1. **생성 단계**:
   - 사용자가 Inspector에서 파라미터 및 레이아웃 모드 설정
   - CityGenerator가 파라미터 검증
   - 격자 레이아웃 생성 (`CreateGridLayout`)
   - Hybrid/PureRandom: 도로 마스크(`roadMask`) 생성
   - BuildingFactory를 통한 건물 배치 (도로 마스크 셀은 건물 제외)
   - Hybrid/PureRandom: BFS 연결성 보장 (`EnsureRoadConnectivity`)
   - CityGraph 구축 및 노드/엣지 생성
   - StrategicLocationAnalyzer로 전략적 위치 분석
   - SpatialIndex 구축 (Quadtree)
   - MinimapGenerator로 미니맵 생성 → `CityMaps/Minimap_Seed{N}_{Mode}_{Res}.png` 저장
   - CityGraphExporter로 JSON·CSV 내보내기 → `CityData/City_Seed{N}_{Mode}.*` 저장
   - CityDataAPI 초기화

2. **런타임 단계**:
   - AI 에이전트가 CityDataAPI를 통해 쿼리
   - SpatialIndex를 사용한 빠른 위치 검색
   - CityGraph를 사용한 경로 계획
   - MinimapRenderer를 통한 실시간 시각화

## 확장 가능성

### 현재 구현된 기능
- ✅ 기본 Box 형태 건물 생성
- ✅ 격자 기반 배치 시스템
- ✅ **3가지 레이아웃 모드** (PureGrid / Hybrid / PureRandom)
- ✅ **BFS 기반 길 연결성 보장** (Hybrid, PureRandom 모드)
- ✅ **Inspector 레이아웃 버튼 UI** (강조 토글 버튼)
- ✅ 그래프 자료구조 (노드-엣지)
- ✅ Quadtree 공간 인덱스
- ✅ 전략적 위치 분석 (위험도/가시성 점수 포함)
- ✅ 런타임 쿼리 API (O(1) GetNodeById 포함)
- ✅ 프리셋 시스템
- ✅ 오브젝트 풀링
- ✅ **탑뷰 미니맵 생성 및 PNG 자동 저장** (CityMaps/ 폴더)
- ✅ **그래프 JSON·CSV 자동 내보내기** (CityData/ 폴더)
- ✅ **레이아웃 모드별 파일명 자동 구분** (Seed + LayoutMode 포함)
- ✅ 미니맵 실시간 렌더링 (MinimapRenderer)

### 향후 확장 가능 기능
- ⬜ 등고선 스타일 미니맵 (고도 가시화)
- ⬜ 다양한 건물 형태 (L자형, T자형 등)
- ⬜ 텍스처 및 머티리얼 변형
- ⬜ 랜드마크 건물 배치
- ⬜ 지형 고도 변화
- ⬜ 동적 장애물 추가
- ⬜ 날씨 및 시간대 변화
- ⬜ 3D 미니맵 지원
- ⬜ 실시간 그래프 업데이트
- ⬜ 건물 내부 구조 생성
- ⬜ 다중 층 건물 지원


## FAQ (자주 묻는 질문)

### Q1: 생성된 도시를 씬에 영구적으로 저장할 수 있나요?

**A**: 네, 가능합니다. 도시 생성 후 Hierarchy에서 `CityGenerator`가 붙은 GameObject를 선택하고 Prefab으로 저장하면 됩니다. 그래프 데이터는 `Assets/CityData/` 폴더에 자동으로 JSON·CSV로 저장됩니다.

```csharp
// 그래프 데이터 저장 예제
CityGraph graph = cityGenerator.GetCityGraph();
string json = graph.SerializeToJson();
System.IO.File.WriteAllText("city_graph.json", json);
```

### Q2: 런타임에 도시를 생성할 수 있나요?

**A**: 네, `CityGenerator.GenerateCity()` 메서드는 런타임에도 호출 가능합니다. 단, 대규모 도시 생성 시 프레임 드롭이 발생할 수 있으므로 로딩 화면과 함께 사용하는 것을 권장합니다.

### Q3: 여러 개의 도시를 동시에 생성할 수 있나요?

**A**: 네, 여러 개의 `CityGenerator` 컴포넌트를 다른 GameObject에 부착하면 됩니다. 각 생성기는 독립적으로 작동하며, 건물은 각 생성기 GameObject의 자식으로 생성됩니다.

### Q4: 특정 위치에 건물을 배치하지 않으려면?

**A**: 현재는 건물 밀도 파라미터로만 제어 가능합니다. 특정 영역을 제외하려면 생성 후 해당 건물을 수동으로 제거하거나, `BuildingFactory` 코드를 수정하여 제외 영역을 추가할 수 있습니다.

### Q5: 그래프 노드를 시각화할 수 있나요?

**A**: 네, `OnDrawGizmos()` 메서드를 사용하여 Scene View에서 노드와 엣지를 시각화할 수 있습니다. 문제 해결 섹션의 Gizmos 예제를 참고하세요.

### Q6: 건물에 콜라이더가 자동으로 추가되나요?

**A**: 네, Unity의 기본 큐브 프리미티브를 사용하므로 BoxCollider가 자동으로 포함됩니다. 레이캐스트 및 물리 충돌이 정상적으로 작동합니다.

### Q7: 미니맵을 UI에 표시하려면?

**A**: `MinimapRenderer` 컴포넌트를 사용하여 Canvas의 RawImage에 미니맵을 표시할 수 있습니다. API 사용 예제의 미니맵 섹션을 참고하세요.

### Q8: 성능 최적화를 위한 팁은?

**A**: 
- 건물 밀도를 0.7 이하로 유지
- 격자 크기를 60x60 이하로 제한
- 미니맵 해상도를 512x512 이하로 설정
- 불필요한 건물은 생성 후 제거
- 오브젝트 풀링이 자동으로 적용됩니다

### Q9: 다른 프로젝트에서 사용할 수 있나요?

**A**: 네, `ProceduralCityGenerator` 폴더 전체를 다른 Unity 프로젝트로 복사하면 됩니다. 네임스페이스가 `ProceduralCityGenerator`로 분리되어 있어 충돌 없이 사용 가능합니다.

### Q10: 소스 코드를 수정해도 되나요?

**A**: 네, 모든 소스 코드는 수정 가능합니다. 각 클래스는 명확한 책임을 가지고 있어 확장이 용이합니다. XML 문서 주석을 참고하여 수정하세요.

## 라이선스 및 크레딧

### 개발 정보
- **프로젝트**: Procedural City Generator for Drone Reinforcement Learning
- **버전**: 1.2.0
- **개발 환경**: Unity 6000.0.69f1 LTS
- **언어**: C# (.NET Standard 2.1)

### 사용된 기술
- Unity Engine
- C# Programming Language
- Dijkstra's Algorithm (경로 탐색)
- BFS Connected Component Analysis (길 연결성 보장)
- Quadtree Data Structure (공간 분할)
- Procedural Road Mask Generation (Hybrid/PureRandom 도로 마스크)
- Object Pooling Pattern (성능 최적화)
- Singleton Pattern (API 접근)
- ScriptableObject (데이터 저장)
- JSON / CSV Serialization (그래프 내보내기)

## 지원 및 문의

### 문제 보고
시스템 사용 중 문제가 발생하면 다음 정보와 함께 보고해주세요:
- Unity 버전
- 사용한 파라미터 설정
- Console 오류 메시지
- 재현 단계

### 기능 요청
새로운 기능이 필요하거나 개선 사항이 있다면 제안해주세요:
- 원하는 기능 설명
- 사용 사례
- 예상되는 동작

## 버전 히스토리

### v1.2.0 (현재)

레이아웃 다양성 및 데이터 내보내기 릴리즈.

**신규 기능:**
- `CityLayoutMode` 열거형 추가 (`PureGrid` / `Hybrid` / `PureRandom`) — `DataStructures.cs`
- `CityGenerator`: 3가지 레이아웃 모드 구현
  - `PureGrid`: 기존 균일 격자 (변경 없음)
  - `Hybrid`: 불규칙 블록 간격 + 건물 위치 ±45% 오프셋 + 건물 크기 ±25% 변화
  - `PureRandom`: 유기적 구불 도로(2~4 주간선 × 2방향) + 건물 위치 ±45% + 크기 ±50%
- 도로 마스크(`bool[,] roadMask`): 건물 배치 전 도로 셀 예약
- `EnsureRoadConnectivity()`: BFS 연결 컴포넌트 분석 → 고립 셀 해소 (최대 50회 반복)
- `RemoveBuildingAtCell(int x, int z)`: 격자 좌표 기반 건물 제거
- `BuildWanderPoints(int length)`: PureRandom 도로 굴곡 위치 사전 계산
- `GenerateHybridRoadMask()` / `GeneratePureRandomRoadMask()`: 모드별 도로 마스크 생성
- **Inspector 레이아웃 버튼 UI**: `CityGeneratorEditor`에 3-버튼 토글 (선택 = 하늘색 강조)
  - Hybrid/PureRandom 선택 시 `Offset Strength`, `Min/Max Block Size` 슬라이더 노출
  - `Min > Max` 경고 HelpBox 자동 표시
- `MinimapGenerator.SaveMinimapToPNG()`: 파일명에 `layoutTag` 포함 (`Minimap_Seed{N}_{Mode}_{Res}x{Res}.png`)
- `CityGraphExporter.ExportAll()` / `ExportJson()`: `layoutTag` 파라미터 추가
  - 파일명: `City_Seed{N}_{Mode}.json` / `City_Seed{N}_{Mode}_nodes.csv`
  - JSON 메타데이터에 `"layoutMode"` 필드 추가

**Inspector 파라미터 추가:**
- `layoutMode` (CityLayoutMode): 레이아웃 모드 선택
- `randomOffsetStrength` (float 0~1): 건물 위치 오프셋 세기
- `minBlockSize` (int 2~8): 블록 최소 크기
- `maxBlockSize` (int 3~12): 블록 최대 크기

### v1.0.1

버그 수정 릴리즈. 컴파일 오류 해소 및 런타임 정확도 개선.

**버그 수정:**
- `CityGraph.BuildFromGrid()`: 양방향 엣지 중복 생성 수정 (4방향 → 2방향 탐색으로 노드 쌍당 정확히 2개 엣지 보장)
- `MinimapRenderer.CompositeLayers()`: 매 프레임 `new Texture2D()` 할당 제거 — `compositeTexture` 필드를 `Initialize()`에서 한 번 생성 후 재사용하여 GC 스파이크 방지
- `StrategicLocationAnalyzer.CalculateDangerScore()`: `maxDistance` 하드코딩 `100f` → 건물 분포 기반 동적 계산으로 변경
- `SpatialIndex.InsertRecursive()`: `Split()` 이후 데드 코드 제거 (Split 내부에서 이미 재분배 처리됨)
- `CityGenerator`: `strategicLocations`를 로컬 변수 대신 필드로 저장하여 `CityDataAPI.Initialize()`에 전달
- `CityDataAPI.Initialize()`: 세 번째 파라미터 `List<StrategicLocation> strategicLocations` 추가; 타입별 캐시 `strategicLocationsByType` 구축
- `CityDataAPI.GetCoverPoints()` / `GetNearestStrategicLocation()`: 타입별 인덱스 캐시 활용으로 `dangerScore` 올바르게 전달

**신규 API:**
- `CityDataAPI.GetNodeById(int nodeId)`: Dictionary O(1) 직접 조회
- `CityGenerator.GetCityGraph()`: 외부에서 CityGraph 접근
- `CityGenerator.GetStrategicLocations()`: 외부에서 StrategicLocation 리스트 접근
- `MinimapRenderer.Initialize(minimap, pixelsPerMeter, bounds)`: 세 번째 파라미터 `Bounds bounds` 추가

### v1.0.0
- ✅ 기본 도시 생성 시스템
- ✅ 그래프 자료구조 및 공간 인덱스
- ✅ 전략적 위치 분석
- ✅ 런타임 쿼리 API
- ✅ 프리셋 시스템
- ✅ Custom Inspector UI
- ✅ 성능 최적화 (오브젝트 풀링, 배치 처리)

### 향후 계획
- v1.3.0: 다양한 건물 형태 지원 (L자형, T자형 등)
- v1.4.0: 랜드마크·지형 고도 변화
- v2.0.0: 동적 환경 변화 및 실시간 그래프 업데이트

---

## 빠른 시작 체크리스트

시스템을 처음 사용하는 경우 다음 단계를 따라하세요:

- [ ] Unity 씬에 빈 GameObject 생성
- [ ] `CityGenerator` 컴포넌트 추가
- [ ] (선택사항) `Default Building Material` 설정
- [ ] 파라미터 조정 (기본값으로도 작동)
- [ ] **Layout Mode** 버튼에서 원하는 모드 선택 (기본: Pure Grid)
- [ ] (Hybrid/PureRandom) Offset Strength, Block Size 조정
- [ ] **"도시 생성 (Generate City)"** 버튼 클릭
- [ ] Console에서 생성 결과 및 저장 경로 확인
- [ ] `Assets/CityMaps/` 에서 미니맵 PNG 확인
- [ ] `Assets/CityData/` 에서 그래프 JSON·CSV 확인
- [ ] Scene View에서 도시 구조 확인
- [ ] (선택사항) 프리셋으로 저장
- [ ] 스크립트에서 `CityDataAPI` 사용 시작

**축하합니다! 이제 Procedural City Generator를 사용할 준비가 되었습니다.** 🎉

더 자세한 정보는 각 섹션을 참고하거나, API 사용 예제를 통해 실제 구현 방법을 확인하세요.

