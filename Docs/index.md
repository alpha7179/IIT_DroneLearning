---
title: Home
hide:
  - navigation
  - toc
---

<div class="neo-hero" markdown>

# 도심 Occlusion 환경<br>*드론 추격-회피*<br>MARL 시스템

<div class="neo-divider"></div>

<p class="neo-subtitle">Urban Occlusion-Aware Multi-Agent Reinforcement Learning<br>아주대학교 2026-1 파란학기</p>

<div class="neo-badges">
  <span class="neo-badge">Digital Twin</span>
  <span class="neo-badge">MARL</span>
  <span class="neo-badge">Sim2Real</span>
  <span class="neo-badge">Unity 6</span>
</div>

<div class="neo-buttons">
  <a href="system.md" class="neo-btn neo-btn--primary">시스템 개요</a>
  <a href="analysis.md" class="neo-btn">알고리즘 분석</a>
  <a href="https://github.com/alpha7179/IIT_DroneLearning" class="neo-btn">GitHub</a>
</div>

</div>

<div class="neo-stats">
  <div class="neo-stat">
    <span class="neo-stat__value">500<sup style="font-size:1.2rem">²</sup></span>
    <span class="neo-stat__label">도시 그리드</span>
  </div>
  <div class="neo-stat">
    <span class="neo-stat__value">8</span>
    <span class="neo-stat__label">경로 탐색 알고리즘</span>
  </div>
  <div class="neo-stat">
    <span class="neo-stat__value">100%</span>
    <span class="neo-stat__label">Theta* 성공률</span>
  </div>
  <div class="neo-stat">
    <span class="neo-stat__value">16</span>
    <span class="neo-stat__label">주 프로젝트 일정</span>
  </div>
</div>

<div class="neo-features">
  <div class="neo-feature">
    <svg class="neo-feature__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2">
      <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>
    </svg>
    <div class="neo-feature__title">도심 Occlusion 환경</div>
    <p class="neo-feature__desc">개활지가 아닌 고밀도 도심(건물·골목)에서의 실제 비행 시나리오. 건물에 의한 LOS 차단을 핵심 설계 요소로 반영.</p>
  </div>
  <div class="neo-feature">
    <svg class="neo-feature__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2">
      <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>
    </svg>
    <div class="neo-feature__title">정보 비대칭 구조</div>
    <p class="neo-feature__desc">추격자는 제한된 시야(Local Sensor)만 가지고, 회피자는 전역 지도(Global Map)에 접근 가능. 현실적인 센서 제약 모델링.</p>
  </div>
  <div class="neo-feature">
    <svg class="neo-feature__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2">
      <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
    </svg>
    <div class="neo-feature__title">MARL 커리큘럼</div>
    <p class="neo-feature__desc">Ray → Vision → LSTM → Domain Randomization의 단계적 학습. Self-Play로 두 에이전트가 서로의 전략을 고도화.</p>
  </div>
  <div class="neo-feature">
    <svg class="neo-feature__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2">
      <circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>
    </svg>
    <div class="neo-feature__title">Sim2Real 검증</div>
    <p class="neo-feature__desc">PID 기반 상위 명령 추상화로 실기체 FCU 인터페이스와 직접 호환. Unity 환경 검증 후 실제 드론 배포 가능.</p>
  </div>
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
| 학습 인프라 | Google Colab Pro+ |

---

!!! note "진행 상황"
    본 프로젝트는 현재 **Phase 1 (환경 구축 및 설계)** 단계입니다.
    알고리즘 분석 벤치마크는 완료되었으며, Unity 환경 및 RL 학습은 진행 중입니다.
