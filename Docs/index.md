---
title: Home
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

# 도심 Occlusion 환경<br>드론 추격-회피 MARL 시스템

<p class="hero-subtitle">Urban Occlusion-Aware Multi-Agent Reinforcement Learning for Drone Pursuit-Evasion</p>

<div class="hero-badges">
  <span class="badge">아주대 2026-1 파란학기</span>
  <span class="badge">Digital Twin</span>
  <span class="badge">MARL</span>
  <span class="badge">Sim2Real</span>
</div>

<div class="hero-buttons" markdown>
[프로젝트 소개](system.md){ .md-button .md-button--primary }
[알고리즘 분석](analysis.md){ .md-button }
[:fontawesome-brands-github: GitHub](https://github.com/alpha7179/IIT_DroneLearning){ .md-button }
</div>

</div>

<div class="stats-strip">
  <div class="stat-item">
    <span class="stat-value">500×500</span>
    <span class="stat-label">도시 그리드</span>
  </div>
  <div class="stat-item">
    <span class="stat-value">8종</span>
    <span class="stat-label">경로 탐색 알고리즘</span>
  </div>
  <div class="stat-item">
    <span class="stat-value">100%</span>
    <span class="stat-label">Theta* 성공률</span>
  </div>
  <div class="stat-item">
    <span class="stat-value">16주</span>
    <span class="stat-label">프로젝트 일정</span>
  </div>
</div>

## 핵심 차별점

<div class="grid cards" markdown>

-   :material-city: **도심 Occlusion 환경**

    ---

    개활지가 아닌 고밀도 도심(건물·골목)에서의 실제 비행 시나리오.
    건물에 의한 LOS 차단을 핵심 설계 요소로 반영.

-   :material-eye-off: **정보 비대칭 구조**

    ---

    추격자는 제한된 시야(Local Sensor)만 가지고,
    회피자는 전역 지도(Global Map)에 접근 가능.
    현실적인 센서 제약을 그대로 모델링.

-   :material-robot: **MARL Stage 커리큘럼**

    ---

    Ray → Vision → LSTM → Domain Randomization의 단계적 학습.
    Self-Play로 두 에이전트가 서로의 전략을 고도화.

-   :material-transfer: **Sim2Real 검증 가능**

    ---

    PID 기반 상위 명령 추상화로 실기체 FCU 인터페이스와 직접 호환.
    Unity 환경에서 검증 후 실제 드론 배포 가능.

</div>

---

## 알고리즘 분석 미리보기

500×500 도시 그리드에서 8종 경로 탐색 알고리즘 성능 비교 — [전체 결과 보기](analysis.md)

![Algorithm Metrics](assets/metrics_bar.png)

| 순위 | 알고리즘 | Composite | Success | LOS Exp% |
|:---:|---|:---:|:---:|:---:|
| 1 | **Theta\*** | 0.888 | 100% | 0.9% |
| 2 | A\* (Octile) | 0.887 | 100% | 0.5% |
| 3 | A\*+LOS (w=1.0) | 0.883 | 100% | 0.2% |

---

## 기술 스택

| 레이어 | 기술 |
|---|---|
| 시뮬레이션 환경 | Unity 6000.0.58f2 + URP 17.0.4 |
| RL 프레임워크 | ML-Agents 4.0.x (PPO) |
| 딥러닝 | Python 3.10, PyTorch 2.x |
| 시각화 | TensorBoard, Matplotlib |
| 학습 인프라 | Google Colab Pro+ (Headless Linux) |

---

!!! note "진행 상황"
    본 프로젝트는 현재 **Phase 1 (환경 구축 및 설계)** 단계입니다.
    알고리즘 분석 벤치마크는 완료되었으며, Unity 환경 및 RL 학습은 진행 중입니다.
    학습 결과, 데모 영상, 실험 지표는 순차적으로 업데이트될 예정입니다.
