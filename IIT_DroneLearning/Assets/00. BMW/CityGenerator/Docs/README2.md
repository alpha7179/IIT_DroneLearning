# 에피소드 스폰 시스템 — 설계 및 구현 문서

> 담당: 배민우 (00. BMW)
> 관련 파일: `EpisodeSpawnCoordinator.cs` / `SpawnCenter.cs` / `Goal.cs`

---

## 1. 개요

에피소드가 시작될 때마다 Evader·Pursuer·Goal 의 위치를 **한 곳에서 결정하고 각 객체에 직접 전달**하는 중앙집중식 스폰 시스템이다.

기존에는 위치 결정 로직이 `EvaderAgent`, `DroneAgent`, `Goal` 에 분산되어 있었고, 에이전트가 직접 `RandomizePosition()` 을 호출하는 구조였다. 이를 단일 진입점 구조로 통합하였다.

---

## 2. 구성 요소

### 2.1 SpawnCenter

**역할**: 스폰 범위(SpawnRange) 를 보관하고 랜덤 위치를 생성하는 데이터 제공자.

```
SpawnCenter (MonoBehaviour, Singleton)
├── SyncMode
│   ├── Synchronized   : Evader / Pursuer / Goal 이 동일 범위 공유
│   └── Desynchronized : 각각 독립 범위 (_evaderRange / _pursuerRange / _goalRange)
├── RangeMode
│   ├── Radius    : XZ 원형 영역
│   └── Rectangle : XZ 직사각형 영역
└── SpawnRange (struct)
    ├── MinY / MaxY  : 고도 범위
    ├── Radius       : 원형 반경
    └── Width / Depth: 직사각형 반폭
```

**주요 API**:

| 메서드 | 반환 | 설명 |
|---|---|---|
| `GetEvaderSpawnRange()` | `SpawnRange` | Evader 스폰 범위 |
| `GetPursuerSpawnRange()` | `SpawnRange` | Pursuer 스폰 범위 |
| `GetGoalSpawnRange()` | `SpawnRange` | Goal 스폰 범위 |
| `GetRandomPosition(range)` | `Vector3` | 범위 내 랜덤 위치 생성 |

SpawnCenter 는 씬에서 `SpawnCenter.Current` 로 자동 등록된다.

---

### 2.2 EpisodeSpawnCoordinator

**역할**: 에피소드 단위의 스폰 결정권을 가진 중앙 통제 시스템.

#### 스폰 전략 (SpawnStrategy)

| 전략 | 동작 |
|---|---|
| `SpawnCenterRandom` | SpawnCenter 범위 내 랜덤 (기본값) |
| `CityDataAPI` | CityDataAPI 스폰 설정 사용. 다수 드론이면 SpawnCenter 폴백 |
| `Fallback` | SpawnCenter 없을 때 Inspector 설정 범위 내 랜덤 |

#### 스폰 결과 구조체

```csharp
public struct SpawnResult
{
    public Vector3 Position;    // 스폰 위치
    public float   YawDegrees;  // 초기 Yaw (0~360)
}
```

#### 주요 API

| 메서드 / 프로퍼티 | 설명 |
|---|---|
| `ComputeSpawn()` | 스폰 전체 계산 실행. EvaderAgent.OnEpisodeBegin() 첫 줄에서 호출 |
| `IsComputed` | 현 에피소드의 스폰 계산 완료 여부 |
| `GetSpawnPosition(GameObject)` | 드론 개별 스폰 위치 반환 |
| `GetSpawnYaw(GameObject)` | 드론 개별 초기 Yaw 반환 |
| `TryGetSpawnResult(GameObject, out SpawnResult)` | 스폰 결과 전체 반환 |
| `GetGoalPosition()` | Goal 스폰 위치 반환 |
| `GetEvaderPosition(int index)` | n번째 Evader 위치 (인덱스 기반) |
| `GetPursuerPosition(int index)` | n번째 Pursuer 위치 (인덱스 기반) |
| `EvaderCount` / `PursuerCount` | 이번 에피소드의 역할별 드론 수 |
| `OnSpawnComputed` | ComputeSpawn() 완료 시 발생하는 이벤트 |

---

### 2.3 Goal

**역할**: 목표 지점 GameObject. 위치 적용 및 도달 판정 담당.

코디네이터와의 연동에서 사용하는 메서드:

| 메서드 | 호출 주체 | 설명 |
|---|---|---|
| `ApplySpawnPosition(Vector3)` | EpisodeSpawnCoordinator | 코디네이터가 계산한 위치를 Goal에 직접 적용. 실린더 높이 갱신 포함 |
| `RandomizePosition()` | 수동 폴백용 | EpisodeSpawnCoordinator 없을 때 SpawnCenter 또는 반경 기반으로 직접 위치 결정 |

---

## 3. 전체 흐름

```
EvaderAgent.OnEpisodeBegin()
    │
    ├─ [1] EpisodeSpawnCoordinator.ComputeSpawn()
    │          │
    │          ├─ FindGameObjectsWithTag("Evader")  → N개 드론 목록
    │          ├─ FindGameObjectsWithTag("Pursuer") → M개 드론 목록
    │          │
    │          ├─ SpawnCenter.GetEvaderSpawnRange()
    │          │   └─ GetRandomPosition() × N   ─┐
    │          ├─ SpawnCenter.GetPursuerSpawnRange() │  모든 위치 간
    │          │   └─ GetRandomPosition() × M   ─┤  최소 이격 보장
    │          ├─ SpawnCenter.GetGoalSpawnRange()    │  (_minSeparation)
    │          │   └─ GetRandomPosition()        ─┘
    │          │
    │          └─ Goal.Current.ApplySpawnPosition(goalPos)  ← Goal에 직접 명령
    │
    ├─ [2] transform.position = GetSpawnPosition(gameObject)  ← Evader 자신 이동
    └─ [3] ResetPhysicsState()

DroneAgent.OnEpisodeBegin()  (Pursuer)
    └─ transform.position = GetSpawnPosition(gameObject)      ← Pursuer 자신 이동
```

---

## 4. 최소 이격 거리 보장 알고리즘

`ComputeSpawn()` 은 내부적으로 `_occupiedPositions` 리스트를 유지하며, 새 위치를 배치할 때마다 이미 확정된 모든 위치와의 거리를 검사한다.

```
배치 순서: Evader[0] → Evader[1] → ... → Pursuer[0] → ... → Goal
                                                              ↑ 마지막에 배치

각 위치 후보:
  for i in 0.._maxRetry:
      candidate = SpawnCenter.GetRandomPosition(range)
      if distance(candidate, 모든 기존 위치) >= _minSeparation:
          확정, break
  (재시도 초과 시 마지막 후보 사용)
```

Inspector 설정:

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `Min Separation` | 5 m | 모든 엔티티 간 최소 이격 거리 |
| `Max Retry` | 20회 | 이격 조건 충족 실패 시 재시도 횟수 |

---

## 5. 다수 드론 지원

`ComputeSpawn()` 은 매 에피소드마다 `FindGameObjectsWithTag()` 를 통해 씬의 드론 수를 자동으로 파악한다. 드론 수 변경 시 코드 수정 없이 자동 대응된다.

- 드론 GameObject 에 Unity 태그 `"Evader"` 또는 `"Pursuer"` 를 설정하면 된다.
- 태그명은 Inspector 의 `Evader Tag` / `Pursuer Tag` 필드에서 변경 가능하다.

**CityDataAPI 전략 시 주의**: CityDataAPI 는 단일 Evader·Pursuer 위치만 제공하므로, 다수 드론 환경에서는 자동으로 `SpawnCenterRandom` 전략으로 폴백된다.

---

## 6. 씬 설정 방법

1. 빈 GameObject 생성 → `EpisodeSpawnCoordinator` 컴포넌트 부착
2. 빈 GameObject 생성 → `SpawnCenter` 컴포넌트 부착, 범위 설정
3. Evader 드론 GameObject 에 태그 `"Evader"` 설정
4. Pursuer 드론 GameObject 에 태그 `"Pursuer"` 설정
5. Goal GameObject 에 `Goal` 컴포넌트 부착

---

## 7. 에이전트 연동 코드 (팀원 요청 사항)

### EvaderAgent.cs (이재왕 LJW)

`OnEpisodeBegin()` 을 아래와 같이 수정 요청:

```csharp
public override void OnEpisodeBegin()
{
    _episodeTimer             = 0f;
    _episodeSteps             = 0;
    _timeSincePursuerDetected = 0f;
    _isPursuerVisible         = false;
    _lastKnownPursuerPos      = Vector3.zero;

    // 코디네이터에 전체 스폰 계산 위임 (Goal 이동도 내부에서 처리됨)
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
        else
        {
            transform.position = _spawnCenter;
        }
        _goalZone?.RandomizePosition(); // 코디네이터 없을 때만 수동 호출
    }

    transform.rotation = Quaternion.Euler(0f, Random.Range(0f, 360f), 0f);
    ResetPhysicsState();
}
```

### DroneAgent.cs (공용, 합의 필요)

`OnEpisodeBegin()` 상단에 코디네이터 분기 추가:

```csharp
public override void OnEpisodeBegin()
{
    if (_dronePhysics == null) return;
    _dronePhysics.ResetPhysics();

    // 코디네이터 우선
    if (EpisodeSpawnCoordinator.Instance != null && EpisodeSpawnCoordinator.Instance.IsComputed)
    {
        transform.SetPositionAndRotation(
            EpisodeSpawnCoordinator.Instance.GetSpawnPosition(gameObject),
            Quaternion.Euler(0f, EpisodeSpawnCoordinator.Instance.GetSpawnYaw(gameObject), 0f));
        return;
    }

    // 기존 CityDataAPI / Fallback 로직 유지
    ...
}
```

---

## 8. Inspector 디버그

플레이 중 `EpisodeSpawnCoordinator` GameObject 선택 시 Scene 뷰에 기즈모 표시:

| 색상 | 의미 |
|---|---|
| 파랑 구체 | Evader[n] 스폰 위치 |
| 빨강 구체 | Pursuer[n] 스폰 위치 |
| 노랑 구체 | Goal 스폰 위치 |
| 흰색 선 | 엔티티 간 이격 거리 연결 |

Inspector Debug 섹션:

| 필드 | 설명 |
|---|---|
| `_isComputed` | 현 에피소드 스폰 계산 완료 여부 |
| `_debugEvaderCount` | 이번 에피소드에서 파악된 Evader 수 |
| `_debugPursuerCount` | 이번 에피소드에서 파악된 Pursuer 수 |

---

## 9. 폴백 우선순위

```
EpisodeSpawnCoordinator.ComputeSpawn()
    └─ SpawnStrategy 에 따라:
        1. SpawnCenterRandom  →  SpawnCenter.Current 없으면 Fallback
        2. CityDataAPI        →  다수 드론 or 설정 없으면 SpawnCenterRandom → Fallback
        3. Fallback           →  Inspector 설정 범위 내 랜덤

Goal.ApplySpawnPosition()  ← 코디네이터가 항상 직접 호출
Goal.RandomizePosition()   ← 코디네이터 없을 때 에이전트가 수동 호출하는 폴백
```