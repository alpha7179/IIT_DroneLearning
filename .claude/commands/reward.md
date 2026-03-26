# /reward — 보상 함수 분석 및 튜닝 제안

$ARGUMENTS

## 동작

현재 보상 함수 설정을 분석하고 학습 로그 기반으로 튜닝을 제안한다.

**사용법:**
```
/reward                           # 현재 보상 구조 분석
/reward python/config/evader_s0_base.yaml  # 특정 config 분석
/reward debug "드론이 구석에만 있음"       # 증상 기반 진단
```

## 분석 절차

### 인자가 없거나 config 파일인 경우
1. 지정된 config (또는 가장 최근 config)를 읽어 보상 관련 설정을 추출한다.
2. AGENT.md의 보상 설계 원칙과 대조하여 잠재적 문제를 진단한다:
   - R_goal / R_survival 가중치 불균형
   - shaping 계수 과대 (탐색 방해)
   - step penalty 과도 (너무 빠른 종료 유도)
   - LOS 차단 보너스 과대 (숨기만 하는 국소 최적)
3. 권장 가중치 범위를 표로 제시한다.

### 증상 기반 진단 (`debug` 키워드)
증상에 따라 다음 진단을 수행한다:

| 증상 | 원인 추정 | 조치 |
|---|---|---|
| 구석에 숨기만 함 | R_survival 또는 R_occlusion 과대 | R_goal 가중치 상향, R_time 페널티 강화 |
| 목표만 향해 돌진 | R_survival 부족 | R_survival 가중치 상향 |
| 원을 그리며 맴돌음 | step penalty 부족 + shaping 오류 | R_time 강화, potential-based shaping 점검 |
| 충돌 빈발 | R_collision 약함 | 페널티 크기 -2.0 이상으로 강화 |
| 보상이 항상 0 | 좌표계 오류 또는 종료 조건 오탐 | Observation 좌표계 및 termination 로직 점검 |

## 출력 형식
분석 결과와 수정 제안을 명확히 구분하여 출력한다.
수정이 필요한 경우 config YAML 변경 사항을 구체적으로 제시한다.
