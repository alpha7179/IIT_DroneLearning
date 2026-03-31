# DroneCamera 시스템 — 설계 및 구현 문서

> 담당: 배민우 (00. BMW)
> 네임스페이스: `DroneCamera`
> 관련 파일: `DroneCameraSystem.cs` / `CameraControlSystem.cs`
> 테스트: `DroneCameraSystemTests.cs` / `DroneCameraSystemPropertyTests.cs`

---

## 1. 개요

드론 FPV 카메라와 씬 전체 카메라 레이아웃을 독립적으로 관리하는 시스템이다.

- **DroneCameraSystem**: 드론 GameObject에 부착. FPV 카메라(Solo + MultiView) 2개를 자동 생성.
- **CameraControlSystem**: 씬 단위 싱글톤. 모든 드론 카메라를 등록받아 디스플레이 할당 및 분할 뷰포트를 자동 구성.

기존 `DroneAgent.cs`, `DronePhysics.cs` 코드 수정 없이 드론 GameObject에 추가 부착만으로 동작한다.

---

## 2. DroneCameraSystem

### 2.1 역할

드론 1대에 부착되는 FPV 카메라 컴포넌트. `Awake()`에서 두 개의 자식 Camera를 생성한다.

| 카메라 | GameObject 이름 | 용도 |
|---|---|---|
| `Camera` (SoloCamera) | `DroneCamera_Solo` | 개별 디스플레이 단독 전체 화면 출력 |
| `MultiViewCamera` | `DroneCamera_MultiView` | Display 1 다중뷰 분할 화면 |

### 2.2 Inspector 필드

| 필드 | 기본값 | 설명 |
|---|---|---|
| `cameraLateralOffset` | (0, 0) | 카메라 좌우·상하 오프셋 (드론 로컬 X·Y) |
| `cameraDistance` | 0.5 | 드론 전방 카메라 거리 (드론 로컬 +Z) |
| `fieldOfView` | 60° | 카메라 FOV |
| `nearClipPlane` | 0.1 | Near Clip |
| `farClipPlane` | 500 | Far Clip |

### 2.3 생명주기

```
Awake()
  └─ DroneCamera_Solo (Camera) 생성 — localPosition = (offsetX, offsetY, distance)
  └─ DroneCamera_MultiView 생성 — enabled = false, targetDisplay = 0

Start()
  ├─ CameraControlSystem 탐색
  ├─ 없으면: SoloCamera만 사용, Evader=Display2 / Pursuer=Display3 독립 동작
  └─ 있으면: RegisterDroneCamera(this, role) 호출

LateUpdate()
  └─ ApplyCameraPosition() — Inspector 변경값 매 프레임 반영
```

### 2.4 역할 판별 (리플렉션)

`DroneCamera` 어셈블리는 `Assembly-CSharp`(DroneAgent)을 직접 참조할 수 없다.
`ResolveRole()`은 리플렉션으로 `DroneAgent.Role` 필드를 읽어 로컬 `DroneRole` enum으로 변환한다.

```csharp
// DroneRole enum: Pursuer = 0, Evader = 1
int roleValue = Convert.ToInt32(roleField.GetValue(comp));
return roleValue == 1 ? DroneRole.Evader : DroneRole.Pursuer;
```

### 2.5 공개 API

| 메서드 | 설명 |
|---|---|
| `SetEnabled(bool)` | Solo + MultiView 동시 활성/비활성 |
| `SetMultiViewEnabled(bool)` | MultiView 카메라만 제어 |
| `SetSoloEnabled(bool)` | Solo 카메라만 제어 |
| `SetViewportRect(Rect)` | MultiView 카메라 분할 뷰포트 설정 |
| `SetSoloViewportRect(Rect)` | Solo 카메라 뷰포트 설정 |
| `SetSoloDisplay(int)` | Solo 카메라 targetDisplay 설정 |
| `SetMultiViewDisplay(int)` | MultiView 카메라 targetDisplay 설정 |

---

## 3. CameraControlSystem

### 3.1 역할

씬 전체 카메라 레이아웃을 중앙 관리하는 싱글톤 매니저.
`CityBatchGenerator`의 배치 수(columns × rows)에 따라 SingleBatch / MultiBatch 모드를 자동 전환한다.

### 3.2 디스플레이 할당 상수

| 상수 | targetDisplay | Unity Display | 용도 |
|---|---|---|---|
| `DisplayMultiView` | 0 | Display 1 | 다중뷰 분할 화면 (SingleBatch), BatchTopView (MultiBatch) |
| `DisplayTopView` | 2 | Display 3 | TopView 단독 전체 화면 |
| `DisplayEvader` | 3 | Display 4 | Evader FPV 단독 전체 화면 |
| `DisplayPursuer` | 4 | Display 5 | Pursuer FPV 단독 전체 화면 |

### 3.3 카메라 모드

#### SingleBatch (배치 1개)

Display 1 다중뷰 분할 레이아웃:

```
┌──────────────────┬────────────┐
│                  │  Pursuer   │  ← 상단 1/2 (오른쪽 1/3)
│   TopView        ├────────────┤
│   (왼쪽 2/3)    │  Evader    │  ← 하단 1/2 (오른쪽 1/3)
└──────────────────┴────────────┘
   Display 1 (targetDisplay=0)
```

다수 드론 시 섹션 내 자동 분할:

| 드론 수 / 역할 | 레이아웃 |
|---|---|
| 1~2대 | 1컬럼 × N행 (세로 스택) |
| 3~4대 | 2컬럼 × ceil(N/2)행 (2×2 그리드) |
| 5대 이상 | 최대 4대까지 표시, 초과분 MultiView 비활성 |

```
각 1대:             각 2대:             각 4대 (2×2 그리드):
┌────────┬──────┐  ┌────────┬──────┐  ┌────────┬───┬───┐
│        │ Pur  │  │        │ Pur0 │  │        │P0 │P1 │
│TopView ├──────┤  │TopView ├──────┤  │TopView ├───┼───┤
│  (2/3) │ Eva  │  │  (2/3) │ Pur1 │  │  (2/3) │P2 │P3 │
│        │      │  │        ├──────┤  │        ├───┼───┤
└────────┴──────┘  │        │ Eva0 │  │        │E0 │E1 │
                   │        ├──────┤  │        ├───┼───┤
                   │        │ Eva1 │  │        │E2 │E3 │
                   └────────┴──────┘  └────────┴───┴───┘
```

Solo 카메라 (개별 디스플레이):
- Pursuer: Display 5 단독 전체 화면 (등록 [0]번만 활성)
- Evader: Display 4 단독 전체 화면 (등록 [0]번만 활성)
- TopView: Display 3 단독 전체 화면

#### MultiBatch (배치 2개 이상)

- 모든 드론 FPV + TopView 비활성
- `BatchTopView_Camera` → Display 1 전체 화면
- `CityBatchGenerator`(columns, rows, spacingX/Z) + `CityGenerator`(unitDistance, buildingWidth 등) 기반으로 OrthoSize 자동 계산

```
orthoSize = max(totalWidth, totalDepth) / 2 × paddingFactor
```

#### None (배치 0개)

- 모든 카메라 비활성화
- `[CameraControlSystem] 활성 배치가 없습니다.` 경고 출력

### 3.4 카메라 자동 생성 (런타임)

`Awake()`에서 씬에 미리 배치할 필요 없이 자동 생성된다.

| 카메라 | 생성 시점 | 특성 |
|---|---|---|
| `TopView_Camera` | Awake | Orthographic, -Y 방향, Display 1 |
| `TopView_Camera_Solo` | Awake | Orthographic, -Y 방향, Display 3 |
| `BatchTopView_Camera` | Awake | Orthographic, -Y 방향, Display 1 |

TopView OrthoSize 자동 계산 (`UpdateTopViewCamera()`):
```
orthoSize = max(cityWidth, cityDepth) / 2 × paddingFactor
```
CityBatchGenerator 미존재 시 Inspector 기본값(`topViewOrthoSize`) 사용.

### 3.5 드론 등록 방식

**자동 탐색** (`Start()` → `ScanAndRegisterDroneAgents()`):
씬 내 모든 `DroneCameraSystem`을 탐색하여 GameObject 태그(`pursuerTag` / `evaderTag`)로 역할 판별 후 등록.

**자동 등록** (`DroneCameraSystem.Start()` → `RegisterDroneCamera()`):
각 드론이 Start()에서 직접 CCS에 등록. 중복 등록은 자동 방지.

### 3.6 Inspector 필드

| 필드 | 기본값 | 설명 |
|---|---|---|
| `topViewHeight` | 100 | TopView 카메라 Y 위치 (m) |
| `topViewOrthoSize` | 50 | TopView OrthoSize 기본값 |
| `topViewPaddingFactor` | 1.1 | TopView OrthoSize 여백 배수 |
| `pursuerTag` | "Pursuer" | Pursuer 드론 Unity 태그 |
| `evaderTag` | "Evader" | Evader 드론 Unity 태그 |
| `showCameraLabels` | true | SingleBatch 모드 뷰 우하단 오브젝트 명 표시 |
| `labelFontSize` | 10 | 레이블 폰트 크기 |
| `labelBackgroundAlpha` | 0.55 | 레이블 배경 투명도 |
| `batchTopViewHeight` | 200 | BatchTopView Y 위치 (m) |
| `batchTopViewPaddingFactor` | 1.1 | BatchTopView OrthoSize 여백 배수 |

### 3.7 공개 API

| 메서드 | 설명 |
|---|---|
| `RegisterDroneCamera(DroneCameraSystem, DroneRole)` | 드론 카메라 역할 등록 |
| `UnregisterDroneCamera(DroneCameraSystem)` | 드론 카메라 등록 해제 |
| `ApplyLayout(int batchCount)` | 배치 수 기반 레이아웃 강제 적용 |
| `GetCurrentMode()` | 현재 CameraMode (None / SingleBatch / MultiBatch) 반환 |
| `GetActiveBatchCount()` | CityBatchGenerator columns × rows 반환 (없으면 1) |

### 3.8 외부 타입 리플렉션

`CityBatchGenerator`, `CityGenerator`는 named assembly에서 직접 참조 불가.
`ResolveExternalTypes()`가 AppDomain 전체 어셈블리를 1회 스캔하여 타입을 캐싱한다.

```csharp
_batchGenType = assembly.GetType("ProceduralCityGenerator.CityBatchGenerator");
_cityGenType  = assembly.GetType("ProceduralCityGenerator.CityGenerator");
```

---

## 4. 씬 설정 방법

1. 빈 GameObject 생성 → `CameraControlSystem` 부착 (씬에 1개만)
2. 드론 GameObject에 `DroneCameraSystem` 부착
3. 드론 GameObject 태그를 `"Pursuer"` 또는 `"Evader"`로 설정
4. Unity Editor `Display` 탭에서 Display 1~5 활성화

별도 카메라 프리팹 배치 불필요. 모든 카메라는 런타임 자동 생성.

---

## 5. 전체 동작 흐름

```
[씬 로드]
    │
    ├─ CameraControlSystem.Awake()
    │       └─ TopView_Camera × 2 + BatchTopView_Camera 생성
    │
    ├─ DroneCameraSystem.Awake() (각 드론마다)
    │       └─ DroneCamera_Solo + DroneCamera_MultiView 생성
    │
    ├─ CameraControlSystem.Start()
    │       ├─ ResolveExternalTypes()  — CityBatchGenerator 타입 캐싱
    │       ├─ ScanAndRegisterDroneAgents()  — 태그 기반 자동 등록
    │       └─ ApplyLayout(batchCount)
    │
    └─ DroneCameraSystem.Start() (각 드론마다)
            └─ RegisterDroneCamera(this, role)  — CCS에 자기 등록

[매 프레임]
    ├─ DroneCameraSystem.LateUpdate()  — 카메라 위치 동기화
    └─ CameraControlSystem.Update()    — batchCount 변동 감지 → ApplyLayout() 재호출
```

---

## 6. 테스트

### DroneCameraSystemTests (단위 테스트)

| 테스트 | 검증 내용 |
|---|---|
| `Awake_CreatesCameraChild` | Awake 후 자식 Camera 생성 확인 |
| `Awake_WithoutSensorSystem_WorksIndependently` | DroneSensorSystem 없이 독립 동작 |
| `Start_WithoutCameraControlSystem_DoesNotThrow` | CCS 없이 독립 동작 |
| `Awake_WithoutSensor_CameraForwardAlignedWithDroneForward` | 카메라 전방 = 드론 +Z 방향 |
| `Camera_FollowsDroneRotation` | 드론 회전 시 카메라 연동 |
| `SetEnabled_TogglesCamera` | SetEnabled API 정상 동작 |
| `SetViewportRect_AppliesRect` | SetViewportRect API 정상 반영 |
| `SingleBatch_ViewportRects_AreCorrect` | 1대씩 SingleBatch 뷰포트 좌표 검증 |
| `SingleBatch_MultiPursuer_SectionSplitsVertically` | Pursuer 2대 세로 분할 |
| `SingleBatch_FourPursuers_TwoColumnGrid` | Pursuer 4대 2×2 그리드 |
| `SingleBatch_MaxSlots_FourPerRole` | 역할당 최대 4슬롯, 초과분 비활성 |
| `NoneMode_AllCamerasDisabled_WithWarning` | 배치 0 → 전체 비활성 + 경고 |
| `SingleBatch_MultipleTrackers_OnlyFirstIsActive` | Solo 카메라 [0]번만 활성 |
| `RegisterUnregister_UpdatesLayout` | 등록/해제 후 레이아웃 재적용 |
| `MultiBatch_DronesCameraDisabled_BatchTopViewEnabled` | MultiBatch 모드 전환 검증 |

### DroneCameraSystemPropertyTests (속성 기반 테스트, 각 100회 반복)

| 테스트 | 검증 속성 |
|---|---|
| `Property1` | 임의 드론에서 항상 자식 Camera 1개 생성 |
| `Property2` | 임의 회전 드론 — DroneSensorSystem 없을 때 카메라 forward = 드론 +Z |
| `Property3` | 임의 회전 드론 — DroneSensorSystem 있을 때 카메라 forward = 센서 전방 방향 |
| `Property4` | SingleBatch 모드 — 모든 뷰포트 Rect 겹침 없음 |
| `Property5` | MultiBatch 모드 — 임의 드론 수/배치 수에서 항상 드론 카메라 비활성 |
| `Property6` | BatchTopView OrthoSize ≥ 전체 배치 바운딩 박스 반경 |
| `Property7` | batchCount=1 → SingleBatch, batchCount>1 → MultiBatch 항상 보장 |
| `Property8` | TopView OrthoSize ≥ 도시 바운딩 박스 반경 |
| `Property9` | batchCount=0 → None 모드, 모든 카메라 비활성 |

---

## 7. 디버깅 체크리스트

1. 드론 카메라가 보이지 않는다
   - `CameraControlSystem`이 씬에 있는가?
   - 드론 GameObject 태그가 `"Pursuer"` 또는 `"Evader"`인가?
   - Unity Editor의 Display 탭에서 해당 Display가 열려 있는가?

2. TopView가 도시를 포함하지 않는다
   - `CityBatchGenerator`가 씬에 있는가? (없으면 Inspector 기본값 사용)
   - `topViewPaddingFactor`가 1.0 이상인가?

3. 다수 드론 FPV가 표시되지 않는다
   - 역할당 최대 4대 (Display 1 다중뷰 기준). 5대 이상은 MultiView 비활성.
   - Solo(단독 디스플레이)는 역할당 [0]번 드론만 표시.

4. MultiBatch 모드로 전환되지 않는다
   - `CityBatchGenerator.columns × rows`가 2 이상인가?
   - `Update()`에서 매 프레임 `GetActiveBatchCount()`를 감지하므로 런타임 중 변경도 반영됨.
