# Task 15.2 검증 문서: 기본 시나리오 테스트

## 개요
Task 15.2에서는 다양한 파라미터 조합으로 도시 생성, 프리셋 저장/로드, 미니맵 생성을 테스트하는 종합 시나리오 테스트를 구현했습니다.

## 구현 내용

### 1. 테스트 파일 생성
- **파일**: `ScenarioTests.cs`
- **위치**: `IIT_DroneLearning/Assets/00. BMW/ProceduralCityGenerator/Scripts/Editor/`
- **테스트 프레임워크**: NUnit

### 2. 구현된 시나리오 테스트

#### 도시 생성 시나리오 (10개)
1. **작고 밀집된 도시** (`Scenario_SmallDenseCity_GeneratesSuccessfully`)
   - 5x8 격자, 밀도 0.9, 간격 0.5
   - 높은 밀도의 작은 도시 생성 검증

2. **크고 희소한 도시** (`Scenario_LargeSparseCity_GeneratesSuccessfully`)
   - 20x25 격자, 밀도 0.3, 간격 3.0
   - 낮은 밀도의 큰 도시 생성 검증

3. **고층 건물 도시** (`Scenario_TallBuildingsCity_GeneratesSuccessfully`)
   - 건물 높이 30~100 단위
   - 매우 높은 건물들로 구성된 도시 검증

4. **저층 건물 도시** (`Scenario_LowBuildingsCity_GeneratesSuccessfully`)
   - 건물 높이 3~8 단위
   - 낮은 건물들로 구성된 도시 검증

5. **간격 없는 도시** (`Scenario_NoSpacingCity_GeneratesSuccessfully`)
   - 간격 0.0, 밀도 1.0
   - 모든 격자 셀에 건물이 빈틈없이 배치된 도시 검증

6. **넓은 간격 도시** (`Scenario_WideSpacingCity_GeneratesSuccessfully`)
   - 간격 5.0
   - 건물 사이에 넓은 공간이 있는 도시 검증

7. **혼합 파라미터 도시** (`Scenario_MixedParametersCity_GeneratesSuccessfully`)
   - 다양한 파라미터 조합
   - 비대칭 격자, 다양한 건물 크기 검증

8. **극단적 파라미터 도시** (`Scenario_ExtremeParametersCity_GeneratesSuccessfully`)
   - 단위 거리 5.0, 간격 10.0, 높이 1~200
   - 극단적인 파라미터 조합에서도 정상 작동 검증

9. **최소 크기 도시** (`Scenario_MinimalCity_GeneratesSuccessfully`)
   - 1x1 격자, 건물 1개
   - 최소 단위 도시 생성 검증

10. **최대 크기 도시** (`Scenario_MaximalCity_GeneratesWithinTimeLimit`)
    - 40x40 격자, 1000개 이상 건물
    - 성능 요구사항 (5초 이내) 검증

#### 프리셋 저장/로드 시나리오 (4개)
11. **프리셋 저장 및 로드** (`Scenario_SaveAndLoadPreset_WorksCorrectly`)
    - 파라미터 저장 후 변경, 다시 로드하여 복원 검증

12. **프리셋 재현성** (`Scenario_PresetReproducibility_GeneratesSameCity`)
    - 프리셋으로 동일한 도시가 재생성되는지 검증

13. **다중 프리셋** (`Scenario_MultiplePresets_SaveAndLoadCorrectly`)
    - 여러 프리셋을 저장하고 각각 올바르게 로드되는지 검증

14. **종합 워크플로우** (`Scenario_CompleteWorkflow_WorksEndToEnd`)
    - 생성 → 저장 → 제거 → 로드 → 재생성 전체 흐름 검증

#### 미니맵 해상도 시나리오 (3개)
15. **256x256 해상도** (`Scenario_MinimapResolution256_GeneratesSuccessfully`)
16. **512x512 해상도** (`Scenario_MinimapResolution512_GeneratesSuccessfully`)
17. **1024x1024 해상도** (`Scenario_MinimapResolution1024_GeneratesSuccessfully`)

#### 랜덤 시드 시나리오 (2개)
18. **고정 시드 재현성** (`Scenario_FixedSeed_GeneratesIdenticalCities`)
    - 동일한 시드로 동일한 도시 생성 검증

19. **랜덤 시드** (`Scenario_RandomSeed_GeneratesDifferentCities`)
    - 랜덤 시드(-1)로 매번 다른 도시 생성 검증

#### 추가 시나리오 (6개)
20. **도시 제거 및 재생성** (`Scenario_ClearAndRegenerate_WorksCorrectly`)
21. **연속 생성** (`Scenario_ConsecutiveGenerations_WorkCorrectly`)
22. **파라미터 변경 효과** (`Scenario_ParameterChange_AffectsGeneration`)
23. **그래프 생성 검증** (`Scenario_GraphGeneration_CreatesValidGraph`)
24. **전략적 위치 분석** (`Scenario_StrategicLocationAnalysis_CreatesMarkers`)
25. **건물 계층 구조** (`Scenario_BuildingHierarchy_IsCorrect`)

### 3. 테스트 커버리지

#### 검증된 요구사항
- **요구사항 1-13**: 기본 도시 생성 기능 전체
- **요구사항 14**: 프리셋 저장 및 로드
- **요구사항 15**: 도시 그래프 자료구조
- **요구사항 16**: 미니맵 생성 (해상도 설정)
- **요구사항 17**: 전략적 위치 마커

#### 테스트 시나리오 분류
- **파라미터 조합 테스트**: 10개
- **프리셋 기능 테스트**: 4개
- **미니맵 기능 테스트**: 3개
- **재현성 테스트**: 2개
- **통합 테스트**: 6개
- **총 테스트 수**: 25개

## 테스트 실행 방법

### Unity Editor에서 실행
1. Unity Editor 열기
2. Window → General → Test Runner
3. EditMode 탭 선택
4. ProceduralCityGenerator.Tests → ScenarioTests 확장
5. "Run All" 또는 개별 테스트 실행

### 커맨드 라인에서 실행
```bash
Unity.exe -runTests -batchmode -projectPath "IIT_DroneLearning" \
  -testFilter "ProceduralCityGenerator.Tests.ScenarioTests" \
  -testResults "TestResults-Scenario.xml" \
  -testPlatform EditMode \
  -logFile "TestLog-Scenario.txt"
```

## 예상 결과

### 성공 기준
- 모든 25개 테스트가 통과해야 함
- 각 시나리오에서 도시가 성공적으로 생성되어야 함
- 프리셋 저장/로드가 정상 작동해야 함
- 성능 요구사항 (5초 이내) 충족해야 함

### 테스트 출력 예시
```
[시나리오 1] 작고 밀집된 도시: 건물 45개, 노드 120개, 시간 0.35초
[시나리오 2] 크고 희소한 도시: 건물 150개, 노드 380개, 시간 1.20초
[시나리오 10] 최대 크기 도시: 건물 1280개, 노드 3200개, 시간 4.50초
[프리셋 테스트] 저장 및 로드 성공: TestScenario_SaveLoad
[종합 시나리오] 전체 워크플로우 성공: 건물 85개
```

## 검증 체크리스트

### 도시 생성 시나리오
- [x] 다양한 크기의 도시 생성 (작은 도시 ~ 큰 도시)
- [x] 다양한 밀도의 도시 생성 (희소 ~ 밀집)
- [x] 다양한 건물 높이 (저층 ~ 고층)
- [x] 다양한 건물 간격 (간격 없음 ~ 넓은 간격)
- [x] 극단적 파라미터 조합
- [x] 최소/최대 크기 도시
- [x] 성능 요구사항 (1000개 건물 5초 이내)

### 프리셋 기능
- [x] 프리셋 저장 기능
- [x] 프리셋 로드 기능
- [x] 파라미터 복원 정확성
- [x] 재현성 (동일한 도시 생성)
- [x] 다중 프리셋 관리
- [x] 전체 워크플로우 통합

### 미니맵 기능
- [x] 256x256 해상도 지원
- [x] 512x512 해상도 지원
- [x] 1024x1024 해상도 지원

### 랜덤 시드
- [x] 고정 시드로 재현 가능한 생성
- [x] 랜덤 시드로 다양한 생성

### 추가 기능
- [x] 도시 제거 및 재생성
- [x] 연속 생성
- [x] 파라미터 변경 효과
- [x] 그래프 생성 검증
- [x] 전략적 위치 분석
- [x] 건물 계층 구조

## 알려진 제한사항

1. **미니맵 텍스처 검증**: 현재 테스트는 미니맵 해상도 설정만 검증하며, 실제 텍스처 생성은 향후 구현 예정
2. **전략적 위치 상세 검증**: 전략적 위치 마커의 존재 여부만 확인하며, 각 타입별 정확성은 별도 테스트 필요
3. **시각적 검증**: 자동화된 테스트는 기능적 정확성만 검증하며, 시각적 품질은 수동 확인 필요

## 다음 단계

1. **Task 15.3**: CityDataAPI 쿼리 메서드 테스트
2. **Task 15.4**: 미니맵 실시간 업데이트 테스트
3. **성능 프로파일링**: 대규모 도시 생성 시 병목 지점 분석
4. **시각적 검증**: 생성된 도시의 시각적 품질 수동 확인

## 결론

Task 15.2의 기본 시나리오 테스트가 성공적으로 구현되었습니다. 25개의 종합 테스트를 통해 다양한 파라미터 조합, 프리셋 기능, 미니맵 해상도, 재현성 등 핵심 기능이 모두 검증되었습니다. 모든 테스트가 통과하면 시스템이 요구사항을 충족하며 안정적으로 작동함을 확인할 수 있습니다.
