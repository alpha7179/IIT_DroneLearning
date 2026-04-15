<div class="drone-split">
<div class="drone-split__media">
<svg viewBox="0 0 400 320" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="130" cy="120" r="36" stroke="rgba(244,245,240,0.12)" stroke-width="1"/>
  <circle cx="270" cy="120" r="36" stroke="rgba(244,245,240,0.12)" stroke-width="1"/>
  <circle cx="130" cy="220" r="36" stroke="rgba(244,245,240,0.12)" stroke-width="1"/>
  <circle cx="270" cy="220" r="36" stroke="rgba(200,212,184,0.2)" stroke-width="1"/>
  <circle cx="130" cy="120" r="8" fill="rgba(244,245,240,0.18)"/>
  <circle cx="270" cy="120" r="8" fill="rgba(244,245,240,0.18)"/>
  <circle cx="130" cy="220" r="8" fill="rgba(244,245,240,0.18)"/>
  <circle cx="270" cy="220" r="8" fill="rgba(200,212,184,0.4)"/>
  <line x1="166" y1="120" x2="234" y2="120" stroke="rgba(244,245,240,0.06)" stroke-width="1"/>
  <line x1="130" y1="156" x2="130" y2="184" stroke="rgba(244,245,240,0.06)" stroke-width="1"/>
  <line x1="270" y1="156" x2="270" y2="184" stroke="rgba(244,245,240,0.06)" stroke-width="1"/>
  <line x1="166" y1="220" x2="234" y2="220" stroke="rgba(244,245,240,0.06)" stroke-width="1"/>
</svg>
</div>
<div class="drone-split__content">
  <p class="drone-badge">Members</p>
  <div class="drone-split__title">Team</div>
  <p class="drone-split__desc">아주대학교 2026-1 파란학기. 4인 팀이 World·Sensor·Pursuer·Evader 역할을 분담하여 MARL 시스템을 구축합니다.<br><br>지도교수: 정소이 교수님 (미래모빌리티공학과)</p>
</div>
</div>

<div class="page-body" markdown>

# Team

**아주대학교 2026-1 파란학기**
지도교수: 정소이 교수님 (미래모빌리티공학과)

---

<div class="grid cards" markdown>

-   :material-earth: **이강민** — World / Physics

    ---

    - 드론 동역학 및 물리 엔진 구현
    - 6-DOF 비행 모델 및 PID 파이프라인 제공
    - Unity 도심 환경 구축 및 에피소드 관리
    - 추격·회피 판정 API 설계

    디지털미디어학과

    [:fontawesome-brands-github: grace-mi71](https://github.com/grace-mi71)

-   :material-camera: **배민우** — Sensor / Rendering

    ---

    - Digital Twin 도심 레벨 디자인
    - Ray 태그 표준 및 노이즈 모델 구현
    - 카메라·LiDAR 센서 파이프라인
    - 역방향 LOS Raycast 제공

    디지털미디어학과

    [:fontawesome-brands-github: alpha7179](https://github.com/alpha7179)

-   :material-crosshairs-gps: **박재현** — Pursuer RL

    ---

    - 추격자(Tracker) 정책 학습
    - LSTM 기반 Occlusion 예측 추적 모델
    - Self-Play Pursuer 인터페이스 제공
    - Stage 0용 Scripted baseline 구현

    소프트웨어학과

    [:fontawesome-brands-github: jhparktime](https://github.com/jhparktime)

-   :material-shield-airplane: **이재왕** — Evader RL

    ---

    - 회피자(Evader) 정책 학습 및 실험 분석
    - 보상 함수 설계 및 Stage 커리큘럼 운영
    - 8종 경로 탐색 알고리즘 벤치마크 수행
    - Eval 파이프라인 및 결과 정량 분석

    소프트웨어학과

    [:fontawesome-brands-github: zaeee-wang](https://github.com/zaeee-wang)

</div>

---

## 기술 스택

| 레이어 | 기술 |
|---|---|
| 시뮬레이션 엔진 | Unity 6000.0.58f2 + URP 17.0.4 |
| RL 프레임워크 | ML-Agents 4.0.x |
| 딥러닝 | Python 3.10, PyTorch 2.x |
| 알고리즘 | PPO, Self-Play, Curriculum Learning, LSTM |
| 시각화 | TensorBoard, Matplotlib |
| 학습 인프라 | Google Colab Pro+ (Headless Linux) |
| 버전 관리 | Git / GitHub |

</div>
