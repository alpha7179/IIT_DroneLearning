# Procedural City Generator

## 개요

**Procedural City Generator**는 Unity 환경에서 드론 강화학습을 위한 프로시저럴 도시 환경을 자동 생성하는 시스템입니다. 파라미터 기반으로 다양한 도시 구조를 생성하며, AI 에이전트가 활용할 수 있는 그래프 자료구조, 에피소드별 동적 스폰 시스템, 등고선 스타일 미니맵을 제공합니다.

### 주요 기능

- **파라미터 기반 도시 생성**: Inspector를 통한 직관적인 도시 환경 설정
- **재현 가능한 생성**: 랜덤 시드를 통한 동일 환경 재생성
- **3가지 레이아웃 모드**: PureGrid(완전 격자) · Hybrid(불규칙 블록) · PureRandom(유기적 도로망)
- **길 연결성 보장**: BFS 기반 알고리즘으로 Hybrid/PureRandom에서 막힌 구간 자동 해소
- **에피소드별 동적 스폰**: CityMetadata 기반으로 매 에피소드마다 다른 스폰 위치 계산
- **4가지 스폰 전략**: CityMetadata(동적) · SpawnCenterRandom(범위 랜덤) · CityDataAPI(고정) · Fallback(Inspector 범위)
- **SpawnSystem 자동 배치**: 도시 생성 시 SpawnCenter + EpisodeSpawnCoordinator 자동 생성
- **다수 드론 지원**: 태그 기반 자동 드론 파악, 개별 스폰 위치 할당, 이격 거리 보장
- **AI 친화적 데이터 구조**: 경로 계획과 전략 수립을 위한 그래프 및 공간 분할 자료구조
- **전략적 위치 분석**: 은폐 지점, 교차로, 막다른 골목 등 자동 식별
- **탑뷰 미니맵 자동 저장**: 도시 생성 시 등고선 스타일 PNG를 CityMaps/ 폴더에 자동 저장
- **그래프 JSON·CSV 자동 내보내기**: CityData/ 폴더에 자동 저장 (도망자 드론 오프라인 분석용)
- **스폰/타겟 포인트 자동 생성**: 고립 노드 제외 및 최소 분리 거리 보장
- **최외각 스폰 제한**: `spawnPerimeterFraction`으로 드론 스폰 위치를 도시 가장자리 N% 이내로 제한
- **도시 계층 그룹화**: 건물·벽·바닥이 `CityGroup_Seed{N}` 단일 GameObject 하위로 자동 묶임
- **n×m 대량 배치 생성**: `CityBatchGenerator`로 도시를 격자 형태로 대량 생성 (ML 학습용)
- **런타임 쿼리 API**: 실시간 도시 정보 조회를 위한 최적화된 API
- **프리셋 시스템**: 자주 사용하는 파라미터 조합 저장 및 로드
- **속성 기반 테스트**: 17개 Property-Based Test로 정확성 검증

### 기술 스택

- **플랫폼**: Unity 6000.0.69f1 LTS
- **언어**: C# (.NET Standard 2.1)
- **아키텍처**: MonoBehaviour 컴포넌트, 싱글톤, ScriptableObject
- **자료구조**: 그래프 (노드-엣지), Quadtree (공간 분할)
- **최적화**: 오브젝트 풀링, 배치 처리

---

## 목차

1. [프로젝트 구조](#프로젝트-구조)
2. [시스템 아키텍처](#시스템-아키텍처)
3. [파라미터 설명](#파라미터-설명)
4. [에피소드 스폰 시스템](#에피소드-스폰-시스템)
5. [CityMetadata 동적 스폰](#citymetadata-동적-스폰)
6. [대량 배치 생성](#대량-배치-생성)
7. [API 사용 예제](#api-사용-예제)
8. [데이터 구조 참조](#데이터-구조-참조)
9. [테스트](#테스트)
10. [성능 가이드라인](#성능-가이드라인)
11. [문제 해결](#문제-해결)
12. [버전 히스토리](#버전-히스토리)


---

## 프로젝트 구조

```
IIT_DroneLearning/Assets/00. BMW/CityGenerator/
├── Scripts/
│   ├── CityGenerator.cs              # 도시 생성 핵심 컴포넌트
│   ├── CityParameters.cs             # 파라미터 프리셋 ScriptableObject
│   ├── CityGraph.cs                  # 그래프 자료구조
│   ├── CityGraphExporter.cs          # 그래프 JSON·CSV 내보내기
│   ├── SpatialIndex.cs               # Quadtree 공간 인덱스
│   ├── BuildingFactory.cs            # 건물 생성 및 오브젝트 풀링
│   ├── StrategicLocationAnalyzer.cs  # 전략적 위치 분석
│   ├── MinimapGenerator.cs           # 미니맵 생성 및 PNG 저장
│   ├── MinimapRenderer.cs            # 미니맵 UI 렌더링
│   ├── CityDataAPI.cs                # 런타임 쿼리 API + CityMetadata 저장/조회
│   ├── DataStructures.cs             # 공통 데이터 구조체 (CityMetadata 포함)
│   ├── EpisodeSpawnCoordinator.cs    # 에피소드 중앙집중식 스폰 통제
│   ├── SpawnCenter.cs                # 스폰 범위 관리 + 도시 메타데이터 동기화
│   ├── Goal.cs                       # 목표 지점 위치 적용 및 도달 판정
│   ├── CityBatchGenerator.cs         # n×m 대량 배치 생성 컴포넌트
│   └── Editor/
│       ├── CityGeneratorEditor.cs    # Custom Inspector
│       └── CityBatchGeneratorEditor.cs
├── Tests/
│   ├── CityGeneratorTests.asmdef     # 테스트 어셈블리 정의
│   ├── CityMetadataPropertyTests.cs
│   ├── CityDataAPIPropertyTests.cs
│   ├── EpisodeSpawnCoordinatorPropertyTests.cs
│   ├── SpawnCenterPropertyTests.cs
│   ├── CityGeneratorIntegrationPropertyTests.cs
│   └── BackwardCompatibilityPropertyTests.cs
├── Presets/                          # 저장된 파라미터 프리셋
├── Materials/                        # 건물 머티리얼
├── Docs/                             # 문서
└── (런타임 생성 폴더)
    ├── CityData/                     # 그래프 JSON·CSV 파일
    └── CityMaps/                     # 미니맵 PNG 파일
```

---

## 시스템 아키텍처

### 컴포넌트 관계도

```
CityGenerator (중심 컴포넌트)
    ├── CityParameters (파라미터 저장)
    ├── BuildingFactory (건물 생성 + 오브젝트 풀)
    ├── CityGraph (그래프 자료구조)
    ├── SpatialIndex (Quadtree)
    ├── StrategicLocationAnalyzer (전략 분석)
    ├── MinimapGenerator (미니맵 생성)
    ├── CityGraphExporter (그래프 내보내기)
    ├── CityDataAPI (런타임 API + CityMetadata)
    └── SpawnSystem (자동 생성)
        ├── SpawnCenter (스폰 범위 + 도시 동기화)
        └── EpisodeSpawnCoordinator (스폰 전략 통제)
```

### 전체 데이터 흐름

```
CityGenerator.GenerateCity()
  ├─ 파라미터 검증 → 랜덤 시드 초기화
  ├─ 격자 생성 → 건물 배치 → 연결성 보장
  ├─ 그래프 구축 → 전략적 위치 분석 → 공간 인덱스 구축
  ├─ BuildCityMetadata() → CityDataAPI.SetCityMetadata()
  ├─ [autoGenerateSpawns] GenerateSpawnConfiguration() → CityDataAPI.SetSpawnConfiguration()
  ├─ 미니맵 생성 → 그래프 내보내기
  └─ CreateSpawnSystem()
       ├─ SpawnCenter (AutoSyncFromCity = true)
       └─ EpisodeSpawnCoordinator (Strategy = CityMetadata)

DroneAgent.OnEpisodeBegin()
  └─ EpisodeSpawnCoordinator.ComputeSpawn()
       ├─ [AutoSyncFromCity] SpawnCenter.SyncFromCityMetadata()
       └─ switch (Strategy)
            ├─ CityMetadata      → 유효 후보 노드에서 동적 스폰 계산
            ├─ SpawnCenterRandom → SpawnCenter 범위 내 랜덤
            ├─ CityDataAPI       → 고정 SpawnConfiguration 사용
            └─ Fallback          → Inspector 범위 내 랜덤
```


---

## 파라미터 설명

### Grid Settings (격자 설정)

| 파라미터 | 기본값 | 범위 | 설명 |
|---------|--------|------|------|
| Unit Distance | 1.0 | 0.1~100 | 격자 1단위의 실제 거리 (m) |
| Min/Max Width | 10/20 | 1~100 | 도시 가로 크기 범위 (격자 단위) |
| Min/Max Depth | 10/20 | 1~100 | 도시 세로 크기 범위 (격자 단위) |

### Building Settings (건물 설정)

| 파라미터 | 기본값 | 범위 | 설명 |
|---------|--------|------|------|
| Building Width/Depth | 1.0 | 0.5~50 | 건물 가로/세로 크기 |
| Min/Max Building Height | 5/20 | 1~500 | 건물 높이 범위 |
| Building Spacing | 1.0 | 0~50 | 건물 간 최소 거리 |
| Building Density | 0.7 | 0~1 | 건물 배치 확률 |

### Layout Mode (레이아웃 모드)

| 모드 | 설명 |
|------|------|
| `PureGrid` | 완전 격자 — 균일한 블록 간격 |
| `Hybrid` | 격자 + 랜덤 오프셋 — 불규칙 블록, 건물 위치·크기 변화 |
| `PureRandom` | 유기적 도로망 — 구불구불한 주간선 도로와 자유 배치 |

Hybrid/PureRandom 전용 파라미터:

| 파라미터 | 기본값 | 범위 | 설명 |
|---------|--------|------|------|
| Offset Strength | 0.5 | 0~1 | 건물 위치 오프셋 세기 |
| Min Block Size | 3 | 2~8 | 블록 최소 크기 |
| Max Block Size | 6 | 3~12 | 블록 최대 크기 |

### Spawn Configuration (스폰 설정)

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| autoGenerateSpawns | **false** | true 시 도시 생성과 동시에 고정 스폰 위치 결정 |
| minSpawnSeparation | 10.0 | 스폰 포인트 간 최소 거리 (m) |
| minSpawnHeight | 5.0 | 드론 최저 비행 고도 (m) |
| spawnPerimeterFraction | 0.2 | 최외각 스폰 제한 비율 (0~0.5) |

---

## 에피소드 스폰 시스템

에피소드가 시작될 때마다 Evader·Pursuer·Goal의 위치를 **한 곳에서 결정하고 각 객체에 직접 전달**하는 중앙집중식 스폰 시스템입니다.

### 구성 요소

#### SpawnCenter

스폰 범위(SpawnRange)를 보관하고 랜덤 위치를 생성하는 데이터 제공자입니다.

```
SpawnCenter (MonoBehaviour, Singleton — SpawnCenter.Current)
├── SyncMode: Synchronized (공유 범위) / Desynchronized (개별 범위)
├── RangeMode: Radius (원형) / Rectangle (직사각형)
├── AutoSyncFromCity: 도시 메타데이터 자동 동기화 토글
└── SpawnRange: MinY, MaxY, Radius, Width, Depth
```

| 메서드 | 설명 |
|---|---|
| `GetEvaderSpawnRange()` | Evader 스폰 범위 |
| `GetPursuerSpawnRange()` | Pursuer 스폰 범위 |
| `GetGoalSpawnRange()` | Goal 스폰 범위 |
| `GetRandomPosition(range)` | 범위 내 랜덤 위치 생성 |
| `SyncFromCityMetadata()` | CityMetadata 기반 SpawnRange 자동 설정 |

#### EpisodeSpawnCoordinator

에피소드 단위의 스폰 결정권을 가진 중앙 통제 시스템입니다.

| 전략 | 동작 |
|---|---|
| `CityMetadata` | CityMetadata 기반 에피소드별 동적 스폰 (기본 권장) |
| `SpawnCenterRandom` | SpawnCenter 범위 내 랜덤 |
| `CityDataAPI` | CityDataAPI 고정 스폰 설정 사용. 다수 드론이면 SpawnCenter 폴백 |
| `Fallback` | Inspector 설정 범위 내 랜덤 |

| API | 설명 |
|---|---|
| `ComputeSpawn()` | 스폰 전체 계산. EvaderAgent.OnEpisodeBegin() 첫 줄에서 호출 |
| `GetSpawnPosition(GameObject)` | 드론 개별 스폰 위치 |
| `GetGoalPosition()` | Goal 스폰 위치 |
| `Strategy` | 스폰 전략 get/set |
| `IsComputed` | 계산 완료 여부 |
| `OnSpawnComputed` | ComputeSpawn() 완료 시 이벤트 |

#### Goal

목표 지점 GameObject. 위치 적용 및 도달 판정을 담당합니다.

| 메서드 | 호출 주체 | 설명 |
|---|---|---|
| `ApplySpawnPosition(Vector3)` | EpisodeSpawnCoordinator | 코디네이터가 계산한 위치를 Goal에 적용 |
| `RandomizePosition()` | 수동 폴백용 | 코디네이터 없을 때 직접 위치 결정 |

### 전체 흐름

```
EvaderAgent.OnEpisodeBegin()
  ├─ EpisodeSpawnCoordinator.ComputeSpawn()
  │    ├─ FindGameObjectsWithTag("Evader"/"Pursuer") → 드론 목록 파악
  │    ├─ [AutoSyncFromCity] SpawnCenter.SyncFromCityMetadata()
  │    ├─ 전략에 따라 스폰 위치 계산 (모든 위치 간 최소 이격 보장)
  │    ├─ Goal.Current.ApplySpawnPosition(goalPos)
  │    └─ OnSpawnComputed 이벤트 발생
  ├─ transform.position = GetSpawnPosition(gameObject)
  └─ ResetPhysicsState()

DroneAgent.OnEpisodeBegin() (Pursuer)
  └─ transform.position = GetSpawnPosition(gameObject)
```

### Inspector 설정

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| Strategy | SpawnCenterRandom | 스폰 전략 (자동 생성 시 CityMetadata) |
| Min Separation | 5m | 모든 스폰 위치 간 최소 이격 거리 |
| Max Retry | 20 | 이격 거리 미충족 시 재시도 횟수 |
| Min Spawn Height | 8m | CityMetadata 전략 최저 스폰 높이 |
| Max Spawn Height | 50m | CityMetadata 전략 최고 스폰 높이 |
| Fallback Range | 10m | Fallback 전략 XZ 랜덤 반폭 |
| Fallback Height | 5m | Fallback 전략 스폰 고도 |

### 폴백 우선순위

```
CityMetadata 전략:
  CityDataAPI.Instance == null → Fallback
  HasCityMetadata() == false → SpawnCenterRandom
  validSpawnCandidates 부족 → SpawnCenterRandom

SpawnCenterRandom 전략:
  SpawnCenter.Current == null → Fallback

CityDataAPI 전략:
  다수 드론 or 설정 없음 → SpawnCenterRandom → Fallback
```

### 디버그 기즈모

플레이 중 EpisodeSpawnCoordinator GameObject 선택 시 Scene 뷰에 표시:

| 색상 | 의미 |
|---|---|
| 파랑 구체 | Evader 스폰 위치 |
| 빨강 구체 | Pursuer 스폰 위치 |
| 노랑 구체 | Goal 스폰 위치 |
| 흰색 선 | 엔티티 간 이격 거리 연결 |


---

## CityMetadata 동적 스폰

### 개요

기존에는 `GenerateCity()` 호출 시 스폰 위치가 한 번 고정되었습니다. CityMetadata 동적 스폰은 도시의 구조적 정보를 `CityMetadata`로 추출하고, 매 에피소드마다 이 메타데이터를 활용하여 동적으로 스폰 위치를 계산합니다.

### CityMetadata 데이터 구조

도시 생성 결과에서 추출된 구조적 메타데이터입니다. `DataStructures.cs`에 정의되어 있습니다.

```csharp
[Serializable]
public class CityMetadata
{
    public Bounds cityBounds;                        // 도시 전체 경계 박스
    public int actualCityWidth;                      // 격자 가로 크기
    public int actualCityDepth;                      // 격자 세로 크기
    public float minBuildingHeight;                  // 건물 최소 높이
    public float maxBuildingHeight;                  // 건물 최대 높이
    public List<Building> buildings;                 // 건물 목록
    public CityGraph cityGraph;                      // 도로 그래프
    public List<StrategicLocation> strategicLocations; // 전략적 위치 목록
    public List<GraphNode> validSpawnCandidates;     // 유효 스폰 후보 노드 (캐싱)
    public int usedRandomSeed;                       // 사용된 랜덤 시드
    public CityLayoutMode layoutMode;                // 레이아웃 모드
}
```

`validSpawnCandidates` 필터링 조건:
- 노드 타입이 `OpenSpace` 또는 `Intersection`
- 그래프에서 하나 이상의 엣지가 존재 (고립 노드 제외)
- 어떤 건물의 Bounds 내부에도 위치하지 않음

### CityDataAPI 메타데이터 메서드

```csharp
CityDataAPI.Instance.SetCityMetadata(metadata);       // 저장
CityDataAPI.Instance.GetCityMetadata();                // 조회
CityDataAPI.Instance.HasCityMetadata();                // 존재 여부
CityDataAPI.Instance.GetValidSpawnCandidates();        // 유효 스폰 후보
CityDataAPI.Instance.GetCityBounds();                  // 도시 경계
CityDataAPI.Instance.GetBuildingHeightRange();         // 건물 높이 범위 (min, max)
```

### 스폰 알고리즘

`ComputeFromCityMetadata()` 동작 순서:

1. `CityDataAPI`에서 `CityMetadata` 조회
2. `validSpawnCandidates`에서 최외각 경계 후보 추출 (상위 20%, 최소 10개)
3. 경계 후보 풀을 셔플 (에피소드마다 다른 결과)
4. 경계 후보에서 Evader 위치 선택 (각 드론마다 개별 노드 할당)
5. 경계 후보에서 Pursuer 위치 선택 (이격 거리 보장)
6. 나머지 후보에서 Goal 위치 선택 (모든 드론과 이격 거리 보장)
7. Y좌표 = 노드 `elevation` + `minSpawnHeight` ~ `maxSpawnHeight` 범위 내 랜덤

### SpawnCenter 도시 동기화

`SyncFromCityMetadata()` 호출 시:
- SpawnRange Width = `cityBounds.extents.x`
- SpawnRange Depth = `cityBounds.extents.z`
- SpawnRange Radius = `max(Width, Depth)`
- MinY = `minBuildingHeight`, MaxY = `maxBuildingHeight`
- Transform 위치 = `cityBounds.center`

### 사용 예제

```csharp
// 기본 사용 — 도시 생성 후 자동으로 동적 스폰 동작
// CityGenerator Inspector에서 "도시 생성" 클릭
// → CityMetadata 자동 등록 → SpawnSystem 자동 생성

public override void OnEpisodeBegin()
{
    EpisodeSpawnCoordinator.Instance.ComputeSpawn();
    transform.position = EpisodeSpawnCoordinator.Instance.GetSpawnPosition(gameObject);
}

// 전략 변경
EpisodeSpawnCoordinator.Instance.Strategy = 
    EpisodeSpawnCoordinator.SpawnStrategy.CityDataAPI;

// CityMetadata 직접 활용
if (CityDataAPI.Instance.HasCityMetadata())
{
    CityMetadata meta = CityDataAPI.Instance.GetCityMetadata();
    Debug.Log($"스폰 후보: {meta.validSpawnCandidates.Count}개");
    Debug.Log($"도시 크기: {meta.cityBounds.size}");
}
```

### 하위 호환성

기존 코드는 변경 없이 동작합니다:
- `SpawnCenterRandom`, `CityDataAPI`, `Fallback` 전략: 기존과 동일
- `CityDataAPI.SetSpawnConfiguration` / `GetSpawnConfiguration`: 변경 없음
- `Goal.ApplySpawnPosition()` / `RandomizePosition()`: 변경 없음

> `autoGenerateSpawns` 기본값이 `false`로 변경되었습니다. 기존처럼 고정 스폰을 사용하려면 Inspector에서 `true`로 설정하고 Strategy를 `CityDataAPI`로 변경하세요.


---

## 대량 배치 생성

`CityBatchGenerator`로 서로 다른 씨드의 도시를 n×m 격자로 한 번에 생성합니다.

### 설정

1. 씬에 빈 GameObject 생성 → `CityBatchGenerator` 컴포넌트 추가
2. `City Template`에 `CityGenerator`가 붙은 GameObject 연결
3. `columns`, `rows`, `spacingX`, `spacingZ`, `seedMode` 설정
4. **[Generate Batch]** 버튼 클릭

### 씨드 모드

| 모드 | 동작 | baseSeed |
|---|---|---|
| `AllRandom` | 각 도시마다 고유 랜덤 씨드 | 비활성 |
| `AllSame` | 동일 씨드 → 1개만 생성 | 사용할 씨드 |
| `Sequential` | baseSeed, +1, +2, ... 순서대로 | 시작 씨드 |

### 생성 결과 계층 구조

```
CityBatch_3x2
├── City_col0_row0 → CityGroup_Seed100
│                    ├── City_Buildings
│                    ├── CityWalls
│                    ├── CityFloor
│                    └── SpawnSystem
├── City_col1_row0 → CityGroup_Seed101
└── ...
```

---

## API 사용 예제

### 위치 기반 쿼리

```csharp
// 가장 가까운 노드
GraphNode nearest = CityDataAPI.Instance.GetNodeAtPosition(transform.position);

// 반경 내 노드
List<GraphNode> nearby = CityDataAPI.Instance.GetNodesInRadius(transform.position, 50f);

// 노드 직접 조회 (O(1))
GraphNode node = CityDataAPI.Instance.GetNodeById(nodeId);
```

### 경로 계획

```csharp
List<int> path = CityDataAPI.Instance.GetShortestPath(startNodeId, endNodeId);
foreach (int nodeId in path)
{
    GraphNode node = CityDataAPI.Instance.GetNodeById(nodeId);
    // 노드 위치로 이동
}
```

### 전략적 위치

```csharp
// 은폐 지점 찾기
List<StrategicLocation> covers = CityDataAPI.Instance.GetCoverPoints(position, 100f);

// 가장 가까운 교차로
StrategicLocation intersection = CityDataAPI.Instance.GetNearestStrategicLocation(
    position, StrategyType.Intersection);
```

### 가시성 확인

```csharp
bool visible = CityDataAPI.Instance.IsPositionVisible(pursuerPos, dronePos);
```

### 스폰 시스템 연동

```csharp
// EvaderAgent.cs
public override void OnEpisodeBegin()
{
    if (EpisodeSpawnCoordinator.Instance != null)
    {
        EpisodeSpawnCoordinator.Instance.ComputeSpawn();
        transform.position = EpisodeSpawnCoordinator.Instance.GetSpawnPosition(gameObject);
    }
    else
    {
        // 폴백: SpawnCenter 또는 초기 위치
        if (SpawnCenter.Current != null)
        {
            var range = SpawnCenter.Current.GetEvaderSpawnRange();
            transform.position = SpawnCenter.Current.GetRandomPosition(range);
        }
        _goalZone?.RandomizePosition();
    }
    transform.rotation = Quaternion.Euler(0f, Random.Range(0f, 360f), 0f);
    ResetPhysicsState();
}

// DroneAgent.cs (Pursuer)
public override void OnEpisodeBegin()
{
    if (EpisodeSpawnCoordinator.Instance != null && EpisodeSpawnCoordinator.Instance.IsComputed)
    {
        transform.SetPositionAndRotation(
            EpisodeSpawnCoordinator.Instance.GetSpawnPosition(gameObject),
            Quaternion.Euler(0f, EpisodeSpawnCoordinator.Instance.GetSpawnYaw(gameObject), 0f));
        return;
    }
    // 기존 폴백 로직...
}
```

### API 쿼리 성능

| 메서드 | 시간 복잡도 |
|---|---|
| `GetNodeById()` | O(1) |
| `GetNodeAtPosition()` | O(log n) |
| `GetNodesInRadius()` | O(log n + k) |
| `GetNeighborNodes()` | O(1) |
| `GetNearestStrategicLocation()` | O(k) |
| `GetShortestPath()` | O(E log V) |
| `IsPositionVisible()` | O(1) |


---

## 데이터 구조 참조

### GraphNode

```csharp
public struct GraphNode
{
    public int nodeId;
    public Vector3 position;
    public NodeType nodeType;          // OpenSpace, BuildingCorner, AlleyEntrance, Intersection
    public float elevation;
    public float[] surroundingBuildingHeights;
    public List<StrategyType> strategicMarkers;
    public bool isVisibleFromSpawn;
}
```

### GraphEdge

```csharp
public struct GraphEdge
{
    public int startNodeId;
    public int endNodeId;
    public float travelCost;
    public float visibilityScore;      // 0.0 ~ 1.0
    public PathType pathType;          // Direct, Detour, Concealed
}
```

### StrategicLocation

```csharp
public struct StrategicLocation
{
    public Vector3 position;
    public StrategyType locationType;  // CoverPoint, Intersection, DeadEnd, OpenArea, DetourPath
    public float dangerScore;          // 0.0 ~ 1.0
    public float visibilityScore;      // 0.0 ~ 1.0
    public List<int> connectedNodes;
}
```

### SpawnConfiguration

```csharp
public struct SpawnConfiguration
{
    public Vector3 evaderSpawnPosition;
    public Vector3 pursuerSpawnPosition;
    public Vector3 targetPosition;
    public int evaderSpawnNodeId;
    public int pursuerSpawnNodeId;
    public int targetNodeId;
    public float achievedMinSeparation;
    public bool isValid;
}
```

### SpawnResult

```csharp
public struct SpawnResult
{
    public Vector3 Position;
    public float YawDegrees;           // 0~360
}
```

---

## 테스트

17개의 속성 기반 테스트(Property-Based Test)가 포함되어 있습니다. NUnit + 반복 루프(100회) 패턴으로 구현되었으며, Unity Test Runner에서 실행합니다.

| 파일 | 속성 | 검증 대상 |
|------|------|----------|
| `CityMetadataPropertyTests.cs` | P1, P2 | CityMetadata 필드 완전성, 스폰 후보 필터링 불변량 |
| `CityDataAPIPropertyTests.cs` | P6 | 메타데이터 저장/조회 라운드트립 |
| `EpisodeSpawnCoordinatorPropertyTests.cs` | P7, P8, P9 | 스폰 위치 유효성, 이격 거리, Y좌표 범위 |
| `SpawnCenterPropertyTests.cs` | P10 | 도시 메타데이터 동기화 |
| `CityGeneratorIntegrationPropertyTests.cs` | P3, P4, P5, P16 | GenerateCity 통합, SpawnSystem 생성/제거 |
| `BackwardCompatibilityPropertyTests.cs` | P11~P15, P17 | 하위 호환성, Goal 위치, OnSpawnComputed 이벤트 |

실행: **Window → General → Test Runner → EditMode → Run All**

---

## 성능 가이드라인

### 권장 파라미터 조합

| 환경 | 격자 크기 | 밀도 | 건물 높이 | 예상 시간 | 건물 수 |
|------|----------|------|----------|----------|---------|
| 빠른 테스트 | 5x5~10x10 | 0.5~0.7 | 5~15 | < 1초 | 15~70 |
| 일반 학습 | 15x15~25x25 | 0.6~0.8 | 10~30 | 1~3초 | 135~500 |
| 대규모 | 40x40~60x60 | 0.7~0.9 | 20~100 | 3~5초 | 1120~3240 |

### 프리셋 예제

| 프리셋 | 격자 | 밀도 | 높이 | 용도 |
|--------|------|------|------|------|
| Dense Urban | 30x40 | 0.9 | 15~50 | 복잡한 가림, 좁은 골목, 고난이도 |
| Open City | 20x30 | 0.5 | 5~20 | 넓은 이동 공간, 초보 학습 |
| Skyscraper | 15x25 | 0.6 | 50~150 | 수직 공간 활용, 고도 변화 |
| Maze City | 40x50 | 0.95 | 10~25 | 복잡한 경로 계획, 최고 난이도 |
| Organic Hybrid | 25x35 | 0.75 | 8~40 | 불규칙 블록, 현실적 도시 |
| Organic Random | 30x40 | 0.80 | 5~50 | 가장 현실적, LOS 차단 최다 |


---

## 문제 해결

| 증상 | 원인 | 해결 |
|------|------|------|
| 도시가 생성되지 않음 | 파라미터 검증 실패 | Console 오류 확인, 최소값 > 최대값 등 점검 |
| 생성 시간 5초 이상 | 격자/밀도 과다 | 격자 60x60 이하, 밀도 0.7 이하로 조정 |
| CityDataAPI 미초기화 | 도시 미생성 | 도시를 먼저 생성한 후 API 사용 |
| 경로 탐색 빈 리스트 | 노드 미연결 | 밀도 낮추기, 간격 늘리기 |
| Hybrid/PureRandom 건물 적음 | 도로 마스크 정상 동작 | 밀도 높이기 또는 minBlockSize 늘리기 |
| 스폰 위치가 항상 동일 | Strategy가 CityDataAPI | Strategy를 CityMetadata로 변경 |
| CityMetadata 전략 폴백 | 후보 노드 부족 | 도시 크기 늘리기 또는 밀도 낮추기 |

---

## 버전 히스토리

### v1.6.0 (현재)

CityMetadata 동적 스폰 시스템 릴리즈.

**신규 기능:**
- `CityMetadata` 데이터 구조 추가 — 도시 경계, 건물, 그래프, 유효 스폰 후보 등 캡슐화
- `CityGenerator.BuildCityMetadata()` — 도시 생성 후 CityMetadata 자동 조립
- `CityGenerator.CreateSpawnSystem()` — SpawnCenter + EpisodeSpawnCoordinator 자동 생성
- `CityDataAPI` 메타데이터 메서드 6개 추가 (Set/Get/Has + 편의 메서드)
- `EpisodeSpawnCoordinator.SpawnStrategy.CityMetadata` — 에피소드별 동적 스폰 전략
  - 최외각 경계 후보 우선 추출 (상위 20%, 최소 10개)
  - 경계 후보 셔플 → Evader/Pursuer 배치 → 나머지에서 Goal 선택
  - 모든 위치 간 이격 거리 보장, Y좌표 = elevation + 높이 범위 내 랜덤
  - 다수 드론 개별 노드 할당
- `EpisodeSpawnCoordinator.Strategy` 공개 프로퍼티 추가
- `SpawnCenter.AutoSyncFromCity` + `SyncFromCityMetadata()` — 도시 메타데이터 기반 SpawnRange 자동 동기화
- `CityGenerator.GenerateCity()` 수정 — CityMetadata 등록 + SpawnSystem 자동 생성
- `CityGenerator.ClearCity()` 수정 — SpawnSystem 제거 + CityMetadata 초기화
- `autoGenerateSpawns` 기본값 `true` → `false` 변경
- 17개 속성 기반 테스트 (Property 1~17)

**하위 호환성:**
- 기존 SpawnCenterRandom, CityDataAPI, Fallback 전략 변경 없음
- 기존 CityDataAPI 스폰 메서드 변경 없음
- Goal.ApplySpawnPosition() / RandomizePosition() 변경 없음

### v1.5.0

대량 배치 생성 및 도시 그룹화 릴리즈.

- `CityBatchGenerator` — n×m 격자 대량 생성 (AllRandom/AllSame/Sequential)
- `CityGenerator` 도시 계층 그룹화 (`CityGroup_Seed{N}`)
- 격자 origin 오프셋 보정, 미니맵 bounds 중심 동기화

### v1.4.0

최외각 스폰 제한 릴리즈.

- `spawnPerimeterFraction` — 드론 스폰을 도시 가장자리 N% 이내로 제한
- `GenerateSpawnConfiguration()` 리팩터링 (전체/최외각 후보 분리)

### v1.3.0

스폰/타겟 포인트 자동 생성 릴리즈.

- `SpawnConfiguration` 구조체, `GenerateSpawnConfiguration()` 자동 선정
- 미니맵 스폰 마커 시각화, CityDataAPI 스폰 쿼리

### v1.2.0

레이아웃 다양성 및 데이터 내보내기 릴리즈.

- `CityLayoutMode` (PureGrid/Hybrid/PureRandom)
- BFS 연결성 보장, 도로 마스크, Inspector 레이아웃 버튼 UI
- 미니맵/그래프 파일명에 layoutTag 포함

### v1.0.1

버그 수정 릴리즈.

- 양방향 엣지 중복 수정, MinimapRenderer GC 스파이크 방지
- `GetNodeById()` O(1) 직접 조회, 전략적 위치 캐시 개선

### v1.0.0

초기 릴리즈.

- 기본 도시 생성, 그래프/공간 인덱스, 전략적 위치 분석
- 런타임 쿼리 API, 프리셋 시스템, 오브젝트 풀링

---

## 씬 설정 체크리스트

- [ ] 빈 GameObject → `CityGenerator` 컴포넌트 추가
- [ ] 파라미터 조정 (기본값으로도 작동)
- [ ] Layout Mode 선택 (기본: Pure Grid)
- [ ] "도시 생성" 버튼 클릭
- [ ] → SpawnSystem 자동 생성 확인 (Hierarchy에서 CityGroup 하위)
- [ ] Evader 드론에 태그 `"Evader"` 설정
- [ ] Pursuer 드론에 태그 `"Pursuer"` 설정
- [ ] Goal GameObject에 `Goal` 컴포넌트 부착
- [ ] EvaderAgent.OnEpisodeBegin()에서 `ComputeSpawn()` 호출 확인
- [ ] (선택) EpisodeSpawnCoordinator의 Strategy 변경
- [ ] (선택) SpawnCenter의 AutoSyncFromCity 확인

---

*최초 작성: 2026-03-08 | 최종 수정: 2026-04-01 | 담당: 배민우 (00. BMW)*
