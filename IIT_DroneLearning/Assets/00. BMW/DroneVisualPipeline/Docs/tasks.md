# 구현 계획: 드론 카메라 데이터 추출 & 강화학습 Visual Pipeline

## 개요

DroneVisionSystem, DroneSnapshotSystem, DroneDepthSystem 3개 컴포넌트를 Unity C#으로 구현한다. DroneCamera 폴더 패턴을 따라 `DroneVisualPipeline/Scripts/`, `DroneVisualPipeline/Tests/` 구조로 구성하며, 기존 DroneAgent, DroneCameraSystem, DroneSensorSystem 코드를 수정하지 않는다. 각 태스크는 이전 태스크 위에 점진적으로 빌드된다.

## 태스크

- [ ] 1. 프로젝트 구조 및 Assembly Definition 설정
  - [ ] 1.1 DroneVisualPipeline 폴더 구조 생성
    - `IIT_DroneLearning/Assets/00. BMW/DroneVisualPipeline/Scripts/` 폴더 생성
    - `IIT_DroneLearning/Assets/00. BMW/DroneVisualPipeline/Tests/` 폴더 생성
    - `IIT_DroneLearning/Assets/00. BMW/DroneVisualPipeline/Scripts/Shaders/` 폴더 생성
    - _요구사항: 8.3_
  - [ ] 1.2 Assembly Definition 생성
    - `DroneVisualPipeline.asmdef` 생성 (rootNamespace: `DroneVisualPipeline`, references: `DroneCameraSystem`, `DroneSensorSystem`, `Unity.ML-Agents`)
      - DroneSensorSystem 참조 필수: `includeRayDistances=true` 시 `GetAllNormalizedDistances()` 직접 호출
    - `DroneVisualPipelineTests.asmdef` 생성 (references: `DroneVisualPipeline`, `DroneCameraSystem`, `DroneSensorSystem`)
    - DroneCameraSystem.asmdef 패턴 참고
    - _요구사항: 8.3_

- [ ] 2. DroneVisionSystem 구현 (Phase 1)
  - [ ] 2.1 DroneVisionSystem MonoBehaviour 기본 구조 구현
    - `DroneVisionSystem.cs` 파일 생성
    - `#region` 패턴으로 Inspector 필드 정의: `enableVisualObservation`, `textureWidth`, `textureHeight`, `grayscale`, `fovOverride`, `farClipOverride`, `showDebugFrustum`, `showPreviewOverlay`, `previewSize`, `sensorName`
    - `Range`, `Tooltip`, `Header` 어트리뷰트 적용
    - 내부 상태 필드: `_renderTexture`, `_cameraSystem`, `_observationCamera`, `_isInitialized`, `_lastWidth`, `_lastHeight`
    - _요구사항: 1.1, 1.2, 1.5, 1.7, 6.13, 6.14_
  - [ ] 2.2 Observation 전용 카메라 생성 로직 구현
    - `Start()`에서 초기화 (Awake 대신 — DroneCameraSystem.Awake() 완료 후 실행 보장)
    - DroneCameraSystem 참조 획득 (GetComponent)
    - `DroneCameraSystem.Camera.gameObject` (`DroneCamera_Solo`)에서 `AddComponent<Camera>()`로 Observation 카메라 추가
      → 동일 Transform 자동 공유, DroneCameraSystem LateUpdate() 위치 갱신 자동 추종
    - RenderTexture 생성 (textureWidth × textureHeight, ARGB32)
    - Observation 카메라 설정 = Solo 카메라 복제 (FOV, nearClip, farClip)
    - Observation 카메라의 targetTexture = RenderTexture
    - Solo 카메라의 targetTexture == null 유지 확인 (기존 Display 출력 보존)
    - 초기화 완료 후 `_isInitialized = true`
    - _요구사항: 1.1, 1.8, 8.2_
  - [ ] 2.3 RenderTextureSensorComponent 연동
    - 같은 GameObject의 RenderTextureSensorComponent에 RenderTexture 할당
    - sensorName 설정
    - grayscale 모드 반영
    - _요구사항: 1.3, 1.4_
  - [ ] 2.4 OnValidate 실시간 반영 구현
    - `_isInitialized` 체크 후 동작
    - `enableVisualObservation` → 카메라 enabled 토글
    - `textureWidth`/`textureHeight` 변경 감지 → RT 재생성
    - `fovOverride`/`farClipOverride` → 카메라 즉시 반영
    - `grayscale` 변경 시 플레이 모드 경고 출력
    - _요구사항: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_
  - [ ] 2.5 디버그 시각화 구현
    - `OnDrawGizmos()`에서 Frustum 시각화 (`showDebugFrustum`)
    - `OnGUI()`에서 RenderTexture 미리보기 (`showPreviewOverlay`, `previewSize`)
    - _요구사항: 1.5_

- [ ] 3. 체크포인트 - DroneVisionSystem 기본 동작 확인
  - 모든 테스트가 통과하는지 확인하고, 질문이 있으면 사용자에게 문의한다.

- [ ] 4. DroneSnapshotSystem 구현 (Phase 2)
  - [ ] 4.1 DroneSnapshotSystem MonoBehaviour 기본 구조 구현
    - `DroneSnapshotSystem.cs` 파일 생성
    - `#region` 패턴으로 Inspector 필드 정의: `enableCapture`, `captureInterval`, `captureRGB`, `captureDepth`, `captureWidth`, `captureHeight`, `imageFormat`, `jpgQuality`, `basePath`, `filePrefix`, `saveMetadata`, `includeRayDistances`, `saveEpisodeSummary`, `maxAsyncRequests`, `logCaptureEvents`, `showCaptureStatus`
    - `ImageFormat` 열거형 정의 (PNG, JPG)
    - `Range`, `Tooltip`, `Header` 어트리뷰트 적용
    - 내부 상태 필드: `_episodeCount`, `_stepCount`, `_currentEpisodePath`, `_captureRT`, `_readbackTexture`, `_pendingRequests`
    - 리플렉션 캐시 필드: `_droneAgent`, `_rewardField`, `_cumulativeRewardField`, `_velocityProperty`
    - _요구사항: 3.1, 3.2, 3.6, 3.7, 3.9, 6.13, 6.14_
  - [ ] 4.2 에피소드/스텝 관리 로직 구현
    - 에피소드 시작 시 에피소드 번호 증가, 스텝 번호 0 초기화
    - 에피소드별 하위 폴더 자동 생성 (`{filePrefix}_episode_{NNNN}/`)
    - DroneAgent의 Role 참조하여 filePrefix 자동 설정 (비어있을 때)
    - _요구사항: 3.3, 4.3_
  - [ ] 4.3 AsyncGPUReadback 기반 비동기 캡처 파이프라인 구현
    - `FixedUpdate`에서 captureInterval 간격으로 캡처 트리거
    - `AsyncGPUReadback.Request()` → NativeArray<byte> 수신
    - `_readbackTexture.LoadRawTextureData()` → PNG/JPG 인코딩
    - `Task.Run()` 백그라운드 스레드에서 파일 저장
    - `maxAsyncRequests` 제한으로 동시 요청 수 관리
    - _요구사항: 3.1, 3.8, 7.3, 7.4_
  - [ ] 4.4 메타데이터 JSON 저장 구현
    - `CaptureMetadata.cs` 직렬화 구조체 생성
    - 메타데이터 필드: timestamp, episode, step, droneRole, dronePosition, droneRotation, droneVelocity, targetRelativePosition, sensorDistances, reward, cumulativeReward
    - DroneAgent 데이터(reward, cumulativeReward) 수집은 **리플렉션** 사용
      → DroneVisualPipeline.asmdef에서 Assembly-CSharp(DroneAgent) 직접 참조 불가
      → `Start()`에서 `GetComponents<MonoBehaviour>()` 순회 → 타입명 `"DroneAgent"` 필터링
      → `FieldInfo.GetValue()`로 reward, cumulativeReward 읽기 (DroneCameraSystem.ResolveRole() 패턴 동일 적용)
      → DroneAgent 미존재 또는 필드 없을 시 기본값 0.0f 사용
    - `JsonUtility.ToJson()` 직렬화
    - 이미지와 동일 FixedUpdate 프레임에서 상태 수집
    - _요구사항: 3.4, 3.5, 4.1, 4.2_
  - [ ] 4.5 에피소드 요약 JSON 저장 구현
    - 에피소드 종료 시 `episode_summary.json` 생성
    - 필드: episode, droneRole, totalSteps, cumulativeReward, terminationReason, capturedImageCount, startTime, endTime
    - _요구사항: 4.4_
  - [ ] 4.6 OnValidate 실시간 반영 구현
    - `captureInterval`, `maxAsyncRequests`, `jpgQuality` 범위 클램핑
    - `captureWidth`/`captureHeight` 변경 시 readback 텍스처 재생성
    - `basePath` 빈 문자열 방지
    - _요구사항: 6.1, 6.2, 6.7, 6.8_
  - [ ] 4.7 RGB/Depth 독립 캡처 제어 구현
    - `captureRGB` / `captureDepth` 플래그로 독립 제어
    - DroneDepthSystem 미존재 시 `captureDepth` 자동 비활성 + 경고
    - _요구사항: 3.9_

- [ ] 5. 체크포인트 - DroneSnapshotSystem 기본 동작 확인
  - 모든 테스트가 통과하는지 확인하고, 질문이 있으면 사용자에게 문의한다.

- [ ] 6. DroneDepthSystem 구현 (Phase 4)
  - [ ] 6.1 DroneDepthSystem MonoBehaviour 기본 구조 구현
    - `DroneDepthSystem.cs` 파일 생성
    - `#region` 패턴으로 Inspector 필드 정의: `enableDepthSensor`, `depthWidth`, `depthHeight`, `depthMode`, `maxDepthDistance`, `nearClipOverride`, `farClipOverride`, `colorRamp`, `showDebugFrustum`, `showPreviewOverlay`, `previewSize`, `sensorName`
    - `DepthOutputMode`, `DepthColorRamp` 열거형 정의
    - `Range`, `Tooltip`, `Header` 어트리뷰트 적용
    - 내부 상태 필드: `_depthCamera`, `_depthRT`, `_depthMaterial`, `_cameraSystem`, `_isInitialized`, `_lastWidth`, `_lastHeight`
    - _요구사항: 5.1, 5.4, 5.5, 6.13, 6.14_
  - [ ] 6.2 Depth 전용 카메라 + 커스텀 셰이더 구현
    - `Start()`에서 초기화 (DroneVisionSystem과 동일하게 DroneCameraSystem.Awake() 완료 후 실행 보장)
    - `DroneCameraSystem.Camera.gameObject` (`DroneCamera_Solo`)에서 `AddComponent<Camera>()`로 Depth 카메라 추가
      → 동일 Transform 자동 공유 (DroneVisionSystem과 동일 방식)
    - Depth 카메라 `depthTextureMode = DepthTextureMode.Depth` 설정 (URP `_CameraDepthTexture` 자동 생성)
    - `DroneDepthVisualize.shader` 생성 (URP 호환 HLSL)
    - `_CameraDepthTexture` → `LinearEyeDepth` → `_MaxDistance`로 정규화 (0~1)
    - Grayscale / Jet 컬러 램프 모드 지원
    - Linear / Raw depth 출력 모드 지원
    - Depth RenderTexture 생성 (depthWidth × depthHeight, RFloat 포맷)
    - **렌더링 방식**: `RenderPipelineManager.endCameraRendering` 콜백 + `Graphics.Blit(null, _depthRT, _depthMaterial)`
      → `Camera.SetReplacementShader()` URP 미지원으로 사용 금지
      → `OnDestroy()`에서 `RenderPipelineManager.endCameraRendering -= OnCameraRendering` 해제 필수
    - _요구사항: 5.1, 5.2, 5.3, 5.8_
  - [ ] 6.3 RenderTextureSensorComponent 연동
    - Depth RenderTexture를 RenderTextureSensorComponent에 할당
    - Grayscale 모드로 ML-Agents Visual Observation 등록
    - sensorName 설정
    - _요구사항: 5.2, 5.7_
  - [ ] 6.4 OnValidate 실시간 반영 구현
    - `_isInitialized` 체크 후 동작
    - `enableDepthSensor` → 카메라 enabled 토글
    - `depthWidth`/`depthHeight` 변경 감지 → RT 재생성
    - `nearClipOverride`/`farClipOverride` → 카메라 즉시 반영
    - `depthMode`, `maxDepthDistance`, `colorRamp` → 셰이더 파라미터 즉시 업데이트
    - _요구사항: 6.1, 6.2, 6.9, 6.10, 6.11, 6.12_
  - [ ] 6.5 디버그 시각화 구현
    - `OnDrawGizmos()`에서 Depth 카메라 Frustum 시각화
    - `OnGUI()`에서 Depth Map 미리보기
    - _요구사항: 5.5_

- [ ] 7. 체크포인트 - DroneDepthSystem 기본 동작 확인
  - 모든 테스트가 통과하는지 확인하고, 질문이 있으면 사용자에게 문의한다.

- [ ] 8. YAML Config 템플릿 생성
  - [ ] 8.1 Visual+Vector 혼합 학습 YAML 템플릿 생성
    - `python/config/evader_s0_visual_template.yaml` 생성
    - `vis_encode_type: simple` 기본 설정
    - 이미지 학습에 적합한 batch_size, buffer_size 조정
    - 기존 Vector-only 학습과 혼합 학습 모두 지원하는 구조
    - _요구사항: 2.1, 2.2, 2.3_

- [ ] 9. 단위 테스트 작성
  - [ ]* 9.1 DroneVisionSystem 단위 테스트
    - RenderTexture 생성/해상도 검증
    - 활성/비활성 토글 동작 확인
    - 기존 카메라 무간섭 확인 (Solo 카메라 targetTexture == null)
    - OnValidate 동작 확인 (해상도 변경 시 RT 재생성)
    - _요구사항: 1.1, 1.2, 1.6, 1.8, 6.3, 6.4_
  - [ ]* 9.2 DroneSnapshotSystem 단위 테스트
    - 파일 저장 경로 생성 확인
    - 메타데이터 직렬화/역직렬화 확인
    - 에피소드 카운터 동작 확인
    - OnValidate 클램핑 동작 확인
    - _요구사항: 3.3, 4.1, 4.3, 6.7, 6.8_
  - [ ]* 9.3 DroneDepthSystem 단위 테스트
    - Depth RenderTexture 생성 확인
    - 셰이더 로드 검증
    - 활성/비활성 토글 동작 확인
    - OnValidate 셰이더 파라미터 업데이트 확인
    - _요구사항: 5.1, 5.4, 5.5, 6.9, 6.10, 6.11_

- [ ] 10. 속성 기반 테스트 작성
  - [ ]* 10.1 Property 1: RenderTexture 해상도 일치
    - 임의의 (width, height) 조합에서 생성된 RT 해상도 == 설정값 검증
    - _요구사항: 1.2_
  - [ ]* 10.2 Property 2: Visual Observation 비활성화 시 카메라 비활성
    - `enableVisualObservation = false` 시 카메라 enabled == false 검증
    - _요구사항: 1.6, 6.3_
  - [ ]* 10.3 Property 3: 기존 카메라 무간섭
    - DroneVisionSystem 부착 후 Solo/MultiView 카메라 targetTexture == null 검증
    - _요구사항: 1.8, 8.2_
  - [ ]* 10.4 Property 4: 메타데이터-이미지 파일명 대응
    - 캡처된 파일 집합에서 모든 RGB PNG에 대응하는 meta JSON 존재 검증
    - _요구사항: 4.1_
  - [ ]* 10.5 Property 5: Depth 정규화 범위
    - Depth Map 모든 픽셀 값 0.0 ≤ v ≤ 1.0 검증
    - _요구사항: 5.3_
  - [ ]* 10.6 Property 8: OnValidate 초기화 전 안전성
    - `_isInitialized == false` 상태에서 OnValidate 호출 시 예외 없음 검증
    - _요구사항: 6.2_

- [ ] 11. 최종 체크포인트 - 전체 시스템 통합 확인
  - 모든 테스트가 통과하는지 확인하고, 질문이 있으면 사용자에게 문의한다.
  - DroneVisionSystem + DroneSnapshotSystem + DroneDepthSystem 동시 부착 시 정상 동작 확인
  - RGB + Depth 동시 관측 시 프레임레이트 유지 확인

## 참고

- `*` 표시된 태스크는 선택 사항이며, 빠른 MVP를 위해 건너뛸 수 있다
- 각 태스크는 추적 가능성을 위해 특정 요구사항을 참조한다
- 체크포인트는 점진적 검증을 보장한다
- 모든 코드는 `IIT_DroneLearning/Assets/00. BMW/DroneVisualPipeline/` 하위에 위치한다
- DroneSensorSystem 테스트 패턴(Edit Mode, Reflection 기반 Awake 호출)을 따른다
- 기존 DroneAgent.cs, DroneCameraSystem.cs, DroneSensorSystem.cs는 수정하지 않는다
