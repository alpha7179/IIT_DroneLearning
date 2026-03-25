# 요구사항 문서: 드론 카메라 데이터 추출 & 강화학습 Visual Pipeline

## 소개

본 시스템은 기존 DroneCameraSystem이 생성하는 FPV 카메라 영상을 ML-Agents 강화학습의 Visual Observation으로 활용하고, 학습/분석용 이미지 데이터를 체계적으로 추출·저장하며, URP 네이티브 기능을 활용한 Depth Map 보조 센서를 제공하는 파이프라인이다.

현재 DroneAgent는 44차원 Vector Observation(위치/속도/오일러/레이센서)만 사용하며, DroneCameraSystem의 카메라 렌더링은 모니터링 용도로만 활용된다. 본 파이프라인은 이 카메라 데이터를 강화학습 입력과 Sim2Real 데이터셋 생성에 직접 활용할 수 있도록 확장한다.

## 용어 정의

- **Visual Observation**: ML-Agents에서 카메라/RenderTexture 이미지를 텐서로 변환하여 정책 네트워크에 입력하는 관측 방식
- **RenderTexture (RT)**: GPU 메모리에 카메라 렌더링 결과를 저장하는 Unity 오브젝트
- **RenderTextureSensorComponent**: ML-Agents가 제공하는 컴포넌트로, RenderTexture를 Visual Observation으로 변환
- **Depth Map**: 카메라로부터 각 픽셀까지의 거리 정보를 담은 이미지 (0=가까움, 1=먼 거리)
- **DroneSnapshotSystem**: 런타임에 카메라 렌더링을 PNG/JPG 파일로 저장하는 컴포넌트
- **메타데이터**: 캡처 시점의 드론 상태(위치, 회전, 속도), 타겟 정보, 보상 등을 기록한 JSON 데이터
- **CNN Encoder**: ML-Agents 내부에서 이미지 관측을 벡터로 압축하는 합성곱 신경망 (simple/nature_cnn/match3/resnet)

## 요구사항

### 요구사항 1: RenderTexture 기반 Visual Observation 연동

**사용자 스토리:** AI 연구자로서, 드론 FPV 카메라 영상을 ML-Agents 강화학습의 Visual Observation으로 사용하여, 이미지 기반 정책 학습을 수행할 수 있다.

#### 인수 조건

1. THE DroneVisionSystem SHALL DroneCameraSystem의 Solo 카메라와 동일한 시점의 영상을 **별도의 Observation 전용 카메라**를 통해 RenderTexture에 출력한다 (Solo 카메라 자체에 targetTexture를 설정하지 않는다)
2. THE RenderTexture SHALL Inspector에서 해상도(width, height)를 설정할 수 있으며, 기본값은 84×84로 한다
3. THE DroneVisionSystem SHALL ML-Agents의 RenderTextureSensorComponent를 통해 해당 RenderTexture를 Visual Observation으로 등록한다
4. THE DroneVisionSystem SHALL 기존 DroneAgent의 44차원 Vector Observation과 동시에 사용 가능해야 한다 (멀티모달 관측)
5. THE DroneVisionSystem SHALL Inspector에서 Visual Observation 활성/비활성을 토글할 수 있는 옵션을 제공한다
6. WHEN Visual Observation이 비활성화되면, THE DroneVisionSystem SHALL RenderTexture 렌더링을 중단하여 GPU 리소스를 절약한다
7. THE DroneVisionSystem SHALL Grayscale/RGB 모드를 Inspector에서 선택할 수 있다
8. THE RenderTexture SHALL 기존 DroneCameraSystem의 Solo/MultiView 카메라 동작에 영향을 주지 않아야 한다

### 요구사항 2: 학습용 YAML Config 확장

**사용자 스토리:** AI 연구자로서, ML-Agents YAML 설정에서 Visual Observation 인코더를 구성하여, 이미지 기반 학습 파이프라인을 실행할 수 있다.

#### 인수 조건

1. THE YAML Config SHALL `vis_encode_type` 필드를 포함하여 CNN 인코더 타입을 선택할 수 있다 (simple, nature_cnn, resnet)
2. THE YAML Config SHALL Visual Observation 사용 시 `batch_size`와 `buffer_size`를 이미지 학습에 적합한 값으로 조정한 템플릿을 제공한다
3. THE YAML Config SHALL 기존 Vector-only 학습과 Visual+Vector 혼합 학습을 모두 지원하는 구조여야 한다

### 요구사항 3: 이미지 캡처 및 파일 저장

**사용자 스토리:** 연구자로서, 드론 카메라 영상을 PNG 파일로 저장하여, 오프라인 분석·Sim2Real 데이터셋 구축·학습 디버깅에 활용할 수 있다.

#### 인수 조건

1. THE DroneSnapshotSystem SHALL 드론 카메라의 렌더링 결과를 PNG 파일로 저장할 수 있다
2. THE DroneSnapshotSystem SHALL Inspector에서 캡처 간격(매 N 스텝)을 설정할 수 있다
3. THE DroneSnapshotSystem SHALL 에피소드 단위로 하위 폴더를 자동 생성하여 이미지를 정리한다
4. THE DroneSnapshotSystem SHALL 각 이미지와 함께 메타데이터 JSON 파일을 동시에 저장한다
5. THE 메타데이터 JSON SHALL 최소한 다음 필드를 포함한다: 타임스탬프, 에피소드 번호, 스텝 번호, 드론 월드 위치(Vector3), 드론 회전(Euler), 드론 속도(Vector3), 타겟 상대 위치(Vector3), 드론 역할(Pursuer/Evader)
6. THE DroneSnapshotSystem SHALL Inspector에서 저장 경로(basePath)를 설정할 수 있으며, 기본값은 `CapturedData/` 이다
7. THE DroneSnapshotSystem SHALL Inspector에서 캡처 활성/비활성을 토글할 수 있다
8. THE DroneSnapshotSystem SHALL 캡처 시 메인 스레드 블로킹을 최소화하기 위해 AsyncGPUReadback 또는 비동기 파일 I/O를 사용한다
9. THE DroneSnapshotSystem SHALL RGB 이미지와 Depth Map 이미지를 각각 독립적으로 캡처 활성/비활성 설정할 수 있다

### 요구사항 4: 메타데이터 일관성

**사용자 스토리:** 데이터 분석가로서, 캡처된 이미지와 메타데이터가 정확히 대응되어, 이미지-상태 쌍 데이터셋을 신뢰할 수 있다.

#### 인수 조건

1. THE DroneSnapshotSystem SHALL 이미지 파일명과 메타데이터 파일명이 동일한 스텝 번호를 공유한다 (예: `step_0042_rgb.png`, `step_0042_meta.json`)
2. THE DroneSnapshotSystem SHALL 메타데이터의 드론 상태가 해당 이미지 렌더링 시점의 상태와 동일한 FixedUpdate 프레임에서 수집된 것이어야 한다
3. THE DroneSnapshotSystem SHALL 에피소드 시작 시 에피소드 번호를 자동 증가시키고, 스텝 번호를 0으로 초기화한다
4. THE DroneSnapshotSystem SHALL 에피소드별 요약 JSON(총 스텝 수, 최종 보상, 종료 사유)을 에피소드 종료 시 저장한다

### 요구사항 5: Depth Map 센서 (URP 네이티브)

**사용자 스토리:** AI 연구자로서, 드론 카메라의 Depth Map을 추가 관측으로 사용하여, 거리 인식 기반 정책 학습 또는 장애물 회피 성능을 향상시킬 수 있다.

#### 인수 조건

1. THE DroneDepthSystem SHALL URP의 `_CameraDepthTexture`를 활용하여 드론 카메라 시점의 Depth Map을 생성한다
2. THE DroneDepthSystem SHALL Depth Map을 별도의 RenderTexture에 출력하여 ML-Agents Visual Observation으로 등록할 수 있다
3. THE DroneDepthSystem SHALL Depth 값을 0~1 범위로 정규화한다 (0=nearClipPlane, 1=farClipPlane)
4. THE DroneDepthSystem SHALL Inspector에서 Depth Map 해상도를 설정할 수 있으며, 기본값은 84×84로 한다
5. THE DroneDepthSystem SHALL Inspector에서 활성/비활성을 토글할 수 있다
6. WHEN DroneDepthSystem가 비활성화되면, THE DroneDepthSystem SHALL Depth 렌더링을 중단하여 GPU 리소스를 절약한다
7. THE DroneDepthSystem SHALL 기존 DroneCameraSystem 및 DroneVisionSystem과 독립적으로 동작하며, 동시 사용이 가능해야 한다
8. THE DroneDepthSystem SHALL URP Renderer Feature 또는 커스텀 셰이더를 통해 구현하며, Unity Perception Package에 의존하지 않는다

### 요구사항 6: Inspector 실시간 반영 (OnValidate)

**사용자 스토리:** 개발자로서, Inspector에서 파라미터를 변경하면 즉시 시뮬레이션에 반영되어, 빠른 반복 실험과 디버깅이 가능하다.

#### 인수 조건

1. THE DroneVisionSystem, DroneSnapshotSystem, DroneDepthSystem SHALL 각각 `OnValidate()` 메서드를 구현하여 Inspector 값 변경 시 실시간으로 동작에 반영한다
2. THE OnValidate SHALL 컴포넌트가 초기화되지 않은 상태(`_isInitialized == false`)에서는 아무 동작도 수행하지 않는다
3. THE DroneVisionSystem SHALL Inspector에서 `enableVisualObservation` 변경 시 Observation 카메라의 enabled 상태를 즉시 토글한다
4. THE DroneVisionSystem SHALL Inspector에서 `textureWidth`/`textureHeight` 변경 시 RenderTexture를 재생성한다
5. THE DroneVisionSystem SHALL Inspector에서 `fovOverride`/`farClipOverride` 변경 시 Observation 카메라에 즉시 반영한다 (0 이하 = DroneCameraSystem 값 추종)
6. THE DroneVisionSystem SHALL Inspector에서 `grayscale` 변경 시 플레이 모드에서는 `Debug.LogWarning`을 출력한다 (ML-Agents 센서 채널 수 런타임 변경 불가)
7. THE DroneSnapshotSystem SHALL Inspector에서 `captureWidth`/`captureHeight` 변경 시 readback Texture2D를 재생성한다
8. THE DroneSnapshotSystem SHALL Inspector에서 `basePath`가 빈 문자열이면 기본값 `"CapturedData"`로 자동 복원한다
9. THE DroneDepthSystem SHALL Inspector에서 `enableDepthSensor` 변경 시 Depth 카메라의 enabled 상태를 즉시 토글한다
10. THE DroneDepthSystem SHALL Inspector에서 `depthWidth`/`depthHeight` 변경 시 Depth RenderTexture를 재생성한다
11. THE DroneDepthSystem SHALL Inspector에서 `depthMode`, `maxDepthDistance`, `colorRamp` 변경 시 Depth 셰이더 파라미터를 즉시 업데이트한다
12. THE DroneDepthSystem SHALL Inspector에서 `nearClipOverride`/`farClipOverride` 변경 시 Depth 카메라에 즉시 반영한다 (0 이하 = DroneCameraSystem 값 추종)
13. ALL 컴포넌트 SHALL `Range`, `Tooltip`, `Header` 어트리뷰트를 사용하여 Inspector에서 파라미터의 유효 범위와 설명을 명확히 표시한다
14. ALL 컴포넌트 SHALL DroneSensorSystem의 `#region` 패턴을 따라 Inspector 필드를 논리적 그룹으로 구분한다

### 요구사항 7: 성능 제약

**사용자 스토리:** 개발자로서, Visual Pipeline 추가 후에도 학습 환경의 프레임레이트가 유지되어, 학습 속도 저하를 최소화할 수 있다.

#### 인수 조건

1. THE Visual Pipeline SHALL 단일 드론 기준 RenderTexture 렌더링 + 텐서 변환이 프레임당 2ms 이내에 완료되어야 한다
2. THE Visual Pipeline SHALL 10드론 동시 실행 시 60 FPS 이상을 유지해야 한다
3. THE DroneSnapshotSystem SHALL 파일 저장으로 인한 프레임 드랍이 발생하지 않아야 한다 (비동기 I/O)
4. THE Visual Pipeline SHALL FixedUpdate 중 GC 할당을 최소화한다 (RenderTexture/Texture2D 재사용)

### 요구사항 8: 기존 시스템 호환성

**사용자 스토리:** 개발자로서, Visual Pipeline이 기존 DroneAgent, DroneCameraSystem, DroneSensorSystem 코드를 수정하지 않고 추가되어, 기존 학습 파이프라인이 깨지지 않는다.

#### 인수 조건

1. THE Visual Pipeline SHALL 기존 DroneAgent.cs의 CollectObservations() 메서드를 수정하지 않는다
2. THE Visual Pipeline SHALL 기존 DroneCameraSystem.cs의 Solo/MultiView 카메라 동작을 변경하지 않는다
3. THE Visual Pipeline SHALL 별도의 Assembly Definition(asmdef)으로 분리되어, 기존 어셈블리에 의존성을 추가하지 않는다
4. THE Visual Pipeline SHALL Visual Observation이 비활성화된 상태에서 기존 44차원 Vector-only 학습과 동일하게 동작한다
5. THE Visual Pipeline SHALL Unity 6000.0.69f1, ML-Agents 4.0.x, URP 17.0.4 환경에서 동작한다
