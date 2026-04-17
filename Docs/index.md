---
title: Home
hide:
  - navigation
  - toc
---

<script src="https://cdn.tailwindcss.com"></script>
<script src="https://unpkg.com/lucide@latest"></script>
<script type="module" src="https://unpkg.com/@splinetool/viewer@1.0.94/build/spline-viewer.js"></script>

<div class="drone-home">
  <nav class="drone-home-nav" id="droneHomeNav">
    <a href="." class="drone-home-nav__logo">
      <i data-lucide="crosshair"></i>
      <span>MARL DRONE</span>
    </a>
    <div class="drone-home-nav__links">
      <a href="#about">About</a>
      <a href="#architecture">Architecture</a>
      <a href="#system">Applications</a>
      <a href="#research">Research</a>
    </div>
    <a href="system/" class="drone-home-nav__cta">Docs</a>
  </nav>

  <section class="drone-hero">
    <div class="drone-hero__viewer">
      <spline-viewer url="https://prod.spline.design/A6ACz6lhqQJ8GkAN/scene.splinecode"></spline-viewer>
    </div>
    <div class="drone-hero__overlay"></div>

    <div class="drone-hero__content">
      <p class="drone-hero__eyebrow">Urban Occlusion-Aware</p>
      <h1 class="drone-hero__title">
        PURSUIT-EVASION
        <span class="drone-hero__title-accent">MARL</span>
        SYSTEM
      </h1>
      <p class="drone-hero__description">
        도심 속 복잡한 시야 가림(Occlusion) 환경을 극복하는
        멀티 에이전트 강화학습 기반 지능형 드론 추적-회피 시스템입니다.
      </p>
      <div class="drone-hero__actions">
        <a href="#architecture" class="drone-btn drone-btn--primary">Explore System</a>
        <a href="https://github.com/zaeee-wang" target="_blank" class="drone-btn drone-btn--ghost">
          <i data-lucide="github"></i>
          <span>GitHub</span>
        </a>
        <a href="system/" class="drone-btn drone-btn--ghost">System Docs</a>
      </div>
    </div>

    <a href="#about" class="drone-scroll-cue">
      <i data-lucide="chevron-down"></i>
    </a>
  </section>

  <section id="about" class="drone-section drone-section--about">
    <header class="drone-section__head">
      <p class="drone-section__eyebrow">Vision & Mission</p>
      <h2 class="drone-section__title">Next Intelligence</h2>
    </header>
    <p class="drone-about-copy">
      우리는 도심 환경의 구조적 시야 가림(Occlusion) 한계를 넘어서,
      강화학습 기반 드론 자율비행 시스템의 <strong>다음 장(Next Chapter)</strong>을 설계합니다.
    </p>

    <div class="drone-stat-wrap">
      <div class="drone-stat">
        <span class="drone-stat__value">500×500</span>
        <span class="drone-stat__label">Urban Grid</span>
      </div>
      <div class="drone-stat">
        <span class="drone-stat__value">8</span>
        <span class="drone-stat__label">Path Algorithms</span>
      </div>
      <div class="drone-stat">
        <span class="drone-stat__value">100%</span>
        <span class="drone-stat__label">Theta* Success</span>
      </div>
      <div class="drone-stat">
        <span class="drone-stat__value">16 Weeks</span>
        <span class="drone-stat__label">Project Roadmap</span>
      </div>
    </div>
  </section>

  <section id="architecture" class="drone-section drone-architecture">
    <header class="drone-section__head">
      <p class="drone-section__eyebrow">System Blueprint</p>
      <h2 class="drone-section__title">Architecture Core</h2>
      <p class="drone-section__desc">복잡한 도심 환경에 최적화된 핵심 알고리즘과 시스템 설계를 통합합니다.</p>
    </header>

    <div class="drone-architecture-grid">
      <article class="drone-card drone-card--lead">
        <p class="drone-card__meta">Core Module</p>
        <h3 class="drone-card__title">Occlusion-Aware Intelligent Agent</h3>
        <p class="drone-card__body">
          Raycasting 기반 LOS(Line of Sight) 판단으로 은폐 가능성을 예측하고,
          추적/회피 정책이 임무 상황에 맞게 경로를 능동적으로 재계획합니다.
        </p>
        <img src="https://images.unsplash.com/photo-1508614589041-895b88991e3e?auto=format&fit=crop&w=1000&q=80" alt="Drone in city">
      </article>

      <div class="drone-feature-grid">
        <article class="drone-card drone-card--feature">
          <p class="drone-card__meta">Algorithm</p>
          <h3 class="drone-card__title">MAPPO MARL</h3>
          <div class="drone-card__avatar">
            <img src="https://images.unsplash.com/photo-1527011045970-2032549303c3?auto=format&fit=crop&w=520&q=80" alt="Algorithm">
          </div>
          <p class="drone-card__caption">협력·경쟁이 공존하는 다중 에이전트 정책 최적화.</p>
        </article>

        <article class="drone-card drone-card--feature">
          <p class="drone-card__meta">Scenario</p>
          <h3 class="drone-card__title">Pursuit-Evasion</h3>
          <div class="drone-card__avatar">
            <img src="https://images.unsplash.com/photo-1579820010410-c10411aaaa88?auto=format&fit=crop&w=520&q=80" alt="Pursuit">
          </div>
          <p class="drone-card__caption">추적자/회피자 비대칭 보상 기반 전략 학습.</p>
        </article>

        <article class="drone-card drone-card--feature">
          <p class="drone-card__meta">Environment</p>
          <h3 class="drone-card__title">Urban Sim</h3>
          <div class="drone-card__avatar">
            <img src="https://images.unsplash.com/photo-1473968512647-3e447244af8f?auto=format&fit=crop&w=520&q=80" alt="City">
          </div>
          <p class="drone-card__caption">다층 장애물·시야 차단을 반영한 3D 도심 환경 생성.</p>
        </article>
      </div>
    </div>
  </section>

  <section id="system" class="drone-section">
    <header class="drone-section__head">
      <p class="drone-section__eyebrow">Core Applications</p>
      <h2 class="drone-section__title">Applied Domains</h2>
      <p class="drone-section__desc">실제 산업/공공 안전 영역으로 확장 가능한 활용 시나리오입니다.</p>
    </header>

    <div class="drone-app-grid">
      <article class="drone-card drone-app-card">
        <div class="drone-app-card__media">
          <img src="https://images.unsplash.com/photo-1541888062961-18e38d708365?auto=format&fit=crop&w=900&q=80" alt="Security">
        </div>
        <div class="drone-app-card__content">
          <p class="drone-app-card__index">Domain 01</p>
          <h3 class="drone-app-card__title">도심 보안 및 정찰</h3>
          <p class="drone-app-card__desc">사각지대 없는 순찰 경로로 도심 치안 대응력을 높입니다.</p>
        </div>
      </article>

      <article class="drone-card drone-app-card">
        <div class="drone-app-card__media">
          <img src="https://images.unsplash.com/photo-1449824913935-59a10b8d2000?auto=format&fit=crop&w=900&q=80" alt="Rescue">
        </div>
        <div class="drone-app-card__content">
          <p class="drone-app-card__index">Domain 02</p>
          <h3 class="drone-app-card__title">복합 수색 및 구조</h3>
          <p class="drone-app-card__desc">재난 환경의 은폐 구역을 빠르게 탐색해 구조 시간을 단축합니다.</p>
        </div>
      </article>

      <article class="drone-card drone-app-card">
        <div class="drone-app-card__media">
          <img src="https://images.unsplash.com/photo-1551351833-286a147e090b?auto=format&fit=crop&w=900&q=80" alt="Capture">
        </div>
        <div class="drone-app-card__content">
          <p class="drone-app-card__index">Domain 03</p>
          <h3 class="drone-app-card__title">불법 드론 추적·포획</h3>
          <p class="drone-app-card__desc">다수 에이전트 협동으로 회피 기동 드론의 포획 확률을 높입니다.</p>
        </div>
      </article>

      <article class="drone-card drone-app-card">
        <div class="drone-app-card__media">
          <img src="https://images.unsplash.com/photo-1508614589041-895b88991e3e?auto=format&fit=crop&w=900&q=80" alt="Infrastructure">
        </div>
        <div class="drone-app-card__content">
          <p class="drone-app-card__index">Domain 04</p>
          <h3 class="drone-app-card__title">인프라 사각지대 점검</h3>
          <p class="drone-app-card__desc">교량/대형 구조물 후면 점검 자동화로 유지관리 리스크를 낮춥니다.</p>
        </div>
      </article>
    </div>
  </section>

  <section id="research" class="drone-section drone-section--research">
    <header class="drone-section__head">
      <p class="drone-section__eyebrow">Research Center</p>
      <h2 class="drone-section__title">Paper & Project Updates</h2>
      <p class="drone-section__desc">핵심 연구 문서와 최신 실험 업데이트를 한 곳에 정리했습니다.</p>
    </header>

    <div class="drone-research-grid">
      <article class="drone-paper-card">
        <p class="drone-paper-card__meta">Featured Study</p>
        <h3>도심 Occlusion을 고려한 다중 드론 추적-회피 강화학습 시스템</h3>
        <p>
          시야 가림 현상을 수학적으로 모델링하고, 추적자/회피자 정책을 동시 학습하여
          임무 완수율과 생존 성능을 함께 최적화하는 MARL 프레임워크를 제안합니다.
        </p>
        <a href="system/" class="drone-paper-card__link">
          <span>Read Full Documentation</span>
          <i data-lucide="arrow-right"></i>
        </a>
      </article>

      <ul class="drone-updates">
        <li class="drone-update">
          <div>
            <p class="drone-update__meta">Update</p>
            <p class="drone-update__text">Raycasting 기반 LOS 판별 모듈 통합 완료</p>
          </div>
          <i data-lucide="chevron-right"></i>
        </li>
        <li class="drone-update">
          <div>
            <p class="drone-update__meta">Performance</p>
            <p class="drone-update__text">MAPPO 학습 수렴 속도 30% 향상 (vs Baseline)</p>
          </div>
          <i data-lucide="chevron-right"></i>
        </li>
        <li class="drone-update">
          <div>
            <p class="drone-update__meta">Environment</p>
            <p class="drone-update__text">Procedural City Generator v2.0 배포</p>
          </div>
          <i data-lucide="chevron-right"></i>
        </li>
        <li class="drone-update">
          <div>
            <p class="drone-update__meta">Release</p>
            <p class="drone-update__text">Pursuit-Evasion 시뮬레이션 환경 공개 준비 완료</p>
          </div>
          <i data-lucide="chevron-right"></i>
        </li>
      </ul>
    </div>
  </section>

  <section class="drone-section">
    <div class="drone-banner">
      <p class="drone-banner__eyebrow">Where Drone Intelligence Goes Next</p>
      <h2 class="drone-banner__title">The Next of Aerospace</h2>
    </div>
  </section>

  <section class="drone-section drone-section--cta">
    <header class="drone-section__head">
      <p class="drone-section__eyebrow">Join The Project</p>
      <h2 class="drone-section__title">보이지 않는 사각지대, 시스템이 먼저 예측하고 추적합니다.</h2>
      <p class="drone-section__desc">차세대 도심형 드론 자율비행 연구 프로젝트 문서와 실험 결과를 확인해보세요.</p>
    </header>

    <div class="drone-cta-actions">
      <a href="system/" class="drone-btn drone-btn--primary">View System Docs</a>
      <a href="analysis/" class="drone-btn drone-btn--ghost">Benchmark Analysis</a>
      <a href="evader/" class="drone-btn drone-btn--ghost">Evader Design</a>
      <a href="roadmap/" class="drone-btn drone-btn--ghost">Roadmap</a>
      <a href="team/" class="drone-btn drone-btn--ghost">Team</a>
      <a href="https://github.com/zaeee-wang" target="_blank" class="drone-btn drone-btn--ghost">GitHub</a>
    </div>
  </section>
</div>
