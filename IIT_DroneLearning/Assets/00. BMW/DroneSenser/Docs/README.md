# 드론 레이 센서 시스템 (Drone Ray Sensor System)

Unity ML-Agents 기반 드론 시뮬레이션을 위한 26방향 레이캐스트 센서 시스템입니다.

## 개요

DroneSensorSystem은 드론 주변 360도 전방위 장애물 감지를 제공하는 레이캐스트 기반 센서입니다. 5개 레이어(Top, Top-Middle, Middle, Middle-Bottom, Bottom)로 구성된 26개의 레이를 통해 3차원 공간 인식을 제공하며, ML-Agents 학습을 위한 관찰 데이터로 활용됩니다.

### 주요 기능

- **26방향 레이캐스트**: 5개 레이어로 구성된 구형 센서 패턴
- **정규화된 거리 값**: ML 학습에 최적화된 0.0~1.0 범위의 거리 데이터
- **설정 가능한 매개변수**: Unity Inspector에서 감지 거리, 고도각, 레이어 마스크 조정
- **디버그 시각화**: Scene 뷰에서 실시간 레이 표시 (충돌 감지 시 빨간색, 충돌 없음 시 녹색)
- **성능 최적화**: 레이 방향 벡터 캐싱, 배열 재사용으로 가비지 컬렉션 최소화

## 설치 및 설정

### 1. 컴포넌트 추가

드론 GameObject에 `DroneSensorSystem` 컴포넌트를 추가합니다:

1. Hierarchy에서 드론 GameObject 선택
2. Inspector에서 `Add Component` 클릭
3. `DroneSensorSystem` 검색 및 추가

### 2. Inspector 설정

#### 센서 설정

- **Max Detection Range**: 최대 감지 거리 (기본값: 50m)
  - 레이가 장애물을 감지할 수 있는 최대 거리
  - 거리 정규화의 기준값으로 사용됨

- **Top Middle Elevation**: Top-Middle 레이어 고도각 (기본값: 45°)
  - 범위: 0° ~ 90°
  - 수평면 위쪽 대각선 레이의 각도

- **Middle Bottom Elevation**: Middle-Bottom 레이어 고도각 (기본값: -45°)
  - 범위: -90° ~ 0°
  - 수평면 아래쪽 대각선 레이의 각도

- **Detection Layer Mask**: 감지 대상 레이어
  - 센서가 감지할 Unity 레이어 선택
  - 기본값: Everything (모든 레이어)

#### 디버그 설정

- **Show Debug Rays**: Scene 뷰에서 레이 시각화 (기본값: true)
- **Ray Hit Color**: 충돌 감지 시 레이 색상 (기본값: 빨간색)
- **Ray Miss Color**: 충돌 없을 때 레이 색상 (기본값: 녹색)

### 3. 레이어 마스크 설정 권장사항

센서가 드론 자체를 감지하지 않도록 레이어를 설정하는 것이 좋습니다:

1. 드론 GameObject와 자식 오브젝트를 `Drone` 레이어로 설정
2. 장애물을 `Obstacle` 레이어로 설정
3. DroneSensorSystem의 Detection Layer Mask에서 `Drone` 레이어 제외

## 사용 방법

### DroneAgent와 통합

ML-Agents의 `Agent` 클래스에서 센서 데이터를 관찰로 수집합니다:

```csharp
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using BMW.DroneSensor;

public class DroneAgent : Agent
{
    private DroneSensorSystem _sensorSystem;
    
    protected override void Awake()
    {
        base.Awake();
        _sensorSystem = GetComponent<DroneSensorSystem>();
    }
    
    public override void CollectObservations(VectorSensor sensor)
    {
        // 기존 관찰 (위치, 속도, 회전 등)
        sensor.AddObservation(transform.localPosition);
        sensor.AddObservation(GetComponent<Rigidbody>().velocity);
        // ... 기타 관찰 ...
        
        // 센서 데이터 추가 (26개 거리 값)
        float[] distances = _sensorSystem.GetAllNormalizedDistances();
        foreach (float distance in distances)
        {
            sensor.AddObservation(distance);
        }
    }
}
```

### 특정 방향 거리 조회

특정 레이어와 방향의 거리를 개별적으로 조회할 수 있습니다:

```csharp
// 전방 수평 레이의 거리 조회
float forwardDistance = _sensorSystem.GetNormalizedDistance(
    DroneSensorSystem.SensorLayer.Middle, 
    DroneSensorSystem.CompassDirection.N
);

// 우측 위쪽 대각선 레이의 거리 조회
float rightTopDistance = _sensorSystem.GetNormalizedDistance(
    DroneSensorSystem.SensorLayer.TopMiddle, 
    DroneSensorSystem.CompassDirection.E
);

// 장애물이 가까우면 회피 행동
if (forwardDistance > 0.8f) // 최대 거리의 80% 이상에서 충돌
{
    // 회피 로직
}
```

### 런타임 설정 변경

```csharp
// 감지 거리를 100m로 확장
_sensorSystem.SetMaxDetectionRange(100f);

// 특정 레이 비활성화 (예: Top 레이)
_sensorSystem.SetRayEnabled(0, false);

// 디버그 시각화 끄기 (성능 최적화)
_sensorSystem.SetDebugVisualization(false);
```

### 카메라 정렬 (향후 통합용)

전방 레이 방향을 사용하여 카메라를 정렬할 수 있습니다:

```csharp
// 카메라를 전방 레이 방향으로 정렬
Vector3 forwardDirection = _sensorSystem.GetForwardRayDirection();
camera.transform.rotation = Quaternion.LookRotation(forwardDirection);
```

## 센서 구조

### 레이어 구성

센서는 5개의 수직 레이어로 구성됩니다:

```
         Top (1개)
           ↑
          / \
    Top-Middle (8개)
      ↖  ↑  ↗
     ←  드론  →  Middle (8개)
      ↙  ↓  ↘
   Middle-Bottom (8개)
          \ /
           ↓
        Bottom (1개)
```

### 레이 인덱스 매핑

26개 레이는 다음과 같이 인덱싱됩니다:

- **인덱스 0**: Top (수직 위)
- **인덱스 1-8**: Top-Middle (N, NE, E, SE, S, SW, W, NW)
- **인덱스 9-16**: Middle (N, NE, E, SE, S, SW, W, NW)
- **인덱스 17-24**: Middle-Bottom (N, NE, E, SE, S, SW, W, NW)
- **인덱스 25**: Bottom (수직 아래)

### 나침반 방향

8방위 레이어의 방향 매핑:

- **N (북)**: 0° - 드론 전방
- **NE (북동)**: 45° - 전방 우측
- **E (동)**: 90° - 드론 우측
- **SE (남동)**: 135° - 후방 우측
- **S (남)**: 180° - 드론 후방
- **SW (남서)**: 225° - 후방 좌측
- **W (서)**: 270° - 드론 좌측
- **NW (북서)**: 315° - 전방 좌측

## 좌표계

### Unity 월드 좌표계

- **+X**: 우측 (동쪽)
- **+Y**: 위쪽 (상방)
- **+Z**: 전방 (북쪽)

### 드론 로컬 좌표계

- **+X**: 드론의 우측
- **+Y**: 드론의 상방
- **+Z**: 드론의 전방 (기수 방향)

레이 방향은 드론의 로컬 좌표계를 기준으로 계산되며, 드론이 회전하면 모든 레이도 함께 회전합니다.

## 거리 정규화

센서는 정규화된 거리 값(0.0~1.0)을 반환합니다:

- **0.0**: 충돌 없음 (최대 감지 거리 내에 장애물 없음)
- **1.0**: 최대 감지 거리에서 충돌
- **0.0~1.0**: 중간 거리에서 충돌 (실제 거리 / 최대 감지 거리)

정규화 공식:
```
정규화된 거리 = 실제 거리 (미터) / 최대 감지 거리 (미터)
```

## 성능 고려사항

### 최적화 기법

1. **레이 방향 벡터 캐싱**: 초기화 시 계산된 방향 벡터를 재사용하여 매 프레임 재계산 방지
2. **배열 재사용**: 센서 데이터 배열을 사전 할당하여 가비지 컬렉션 최소화
3. **Physics.Raycast 직접 사용**: RaycastHit 배열 할당 없이 단일 레이캐스트 수행
4. **조건부 디버그 그리기**: ShowDebugRays가 false일 때 디버그 그리기 작업 건너뛰기

### 성능 목표

- **단일 드론**: 26개 레이캐스트 실행 시간 < 1ms
- **다중 드론**: 10개 드론 동시 실행 시 > 60 FPS
- **메모리 할당**: 0 할당/프레임 (FixedUpdate 중)

### 성능 팁

- 프로덕션 빌드에서는 디버그 시각화를 비활성화하세요
- 필요하지 않은 레이는 `SetRayEnabled()`로 비활성화하세요
- 레이어 마스크를 적절히 설정하여 불필요한 충돌 검사를 줄이세요

## 디버깅

### Scene 뷰 시각화

`Show Debug Rays`를 활성화하면 Scene 뷰에서 모든 레이를 볼 수 있습니다:

- **빨간색 레이**: 장애물과 충돌 감지
- **녹색 레이**: 충돌 없음

### 로그 메시지

센서 시스템은 다음과 같은 로그를 출력합니다:

- **Error**: 치명적 오류 (초기화 실패, 배열 인덱스 초과 등)
- **Warning**: 경고 (유효하지 않은 입력, 레이어 마스크 미설정 등)
- **Info**: 정보 (Unity Editor에서만, 디버그 빌드)

### 일반적인 문제 해결

**문제**: 센서가 드론 자체를 감지합니다
- **해결**: 드론을 별도 레이어로 설정하고 Detection Layer Mask에서 제외

**문제**: 모든 레이가 충돌을 감지하지 못합니다
- **해결**: Detection Layer Mask가 Nothing으로 설정되지 않았는지 확인

**문제**: 특정 방향의 레이가 작동하지 않습니다
- **해결**: `SetRayEnabled()`로 해당 레이가 비활성화되지 않았는지 확인

## API 참조

### 공개 메서드

#### `float[] GetAllNormalizedDistances()`
모든 센서의 정규화된 거리 값을 반환합니다 (26개).

#### `float GetNormalizedDistance(SensorLayer layer, CompassDirection direction)`
특정 레이어와 방향의 정규화된 거리 값을 반환합니다.

#### `void SetMaxDetectionRange(float range)`
최대 감지 거리를 런타임에 설정합니다.

#### `void SetRayEnabled(int rayIndex, bool enabled)`
특정 레이를 활성화 또는 비활성화합니다.

#### `Vector3 GetForwardRayDirection()`
전방 레이의 월드 공간 방향 벡터를 반환합니다 (카메라 정렬용).

#### `void SetDebugVisualization(bool enabled)`
디버그 시각화를 활성화 또는 비활성화합니다.

### 열거형

#### `SensorLayer`
- `Top`: 수직 위 (1개)
- `TopMiddle`: 위쪽 대각선 (8개)
- `Middle`: 수평 (8개)
- `MiddleBottom`: 아래쪽 대각선 (8개)
- `Bottom`: 수직 아래 (1개)

#### `CompassDirection`
- `N`: 북 (전방, 0°)
- `NE`: 북동 (45°)
- `E`: 동 (우측, 90°)
- `SE`: 남동 (135°)
- `S`: 남 (후방, 180°)
- `SW`: 남서 (225°)
- `W`: 서 (좌측, 270°)
- `NW`: 북서 (315°)

## 테스트

단위 테스트는 `Tests/DroneSensorSystemTests.cs`에 있습니다. Unity Test Runner를 사용하여 실행할 수 있습니다:

1. Window > General > Test Runner
2. PlayMode 탭 선택
3. Run All 클릭

## 향후 확장

### 카메라 센서 통합

센서 시스템은 향후 카메라 센서 추가를 위해 설계되었습니다:

- 카메라는 드론의 자식 GameObject로 부착
- `GetForwardRayDirection()`을 사용하여 카메라 방향 정렬
- 카메라 센서는 별도 컴포넌트로 구현 (DroneSensorSystem과 독립적)
- 동일한 Transform 계층 구조를 공유하여 일관된 좌표 변환 보장

## 라이선스

BMW 내부 프로젝트용

## 문의

문제나 질문이 있으면 프로젝트 관리자에게 문의하세요.
