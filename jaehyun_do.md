# Vision-Based Pursuer Drone Agent

## 프로젝트 개요

도심형 3D 환경에서 Pursuer 드론이 Evader 드론을 시각적으로 탐지하고 추적하도록 학습시키는 강화학습 프로젝트입니다.  
Pursuer는 정답 좌표에 직접 의존하지 않고, 드론 전방 RGB 카메라에서 보이는 Evader의 색상 정보와 화면상 움직임을 기반으로 추적 행동을 학습합니다.

이 프로젝트의 핵심 목표는 다음과 같습니다.

1. Evader의 고유 색상을 RGB 카메라에서 탐지한다.
2. Lucas-Kanade Optical Flow로 Evader의 화면상 이동 속도를 추정한다.
3. 추정된 시각 정보와 드론 자체 상태를 이용해 Evader를 따라간다.
4. Evader가 순간적으로 가려져도 LSTM memory를 이용해 최근 움직임을 기반으로 추적을 이어간다.

## 담당 영역

Pursuer 학습 파이프라인 전반을 설계하고 구현했습니다.

- RGB 기반 Evader 색상 탐지 로직 구현
- 색상 blob centroid, blob area, viewport offset 계산
- Lucas-Kanade 기반 optical flow 추정 로직 추가
- Pursuer policy observation 설계
- GT dense shaping 제거 및 시각 기반 reward 재설계
- 학습 안정성을 위한 진단 지표 추가
- ML-Agents PPO 학습 설정 구성
- TensorBoard 로그 분석을 통한 perception 및 crash 경향 평가

## 시스템 구조

Pursuer는 다음 입력을 사용합니다.

- RGB 카메라 visual observation
- Pursuer 선속도, 각속도, 고도
- Evader 색상 blob 기반 target direction 및 거리 proxy
- viewport offset
- Lucas-Kanade optical flow 기반 viewport velocity
- visible flag
- LSTM recurrent memory를 통한 최근 시각 추적 상태

명시적으로 제외한 입력은 다음과 같습니다.

- Ray 기반 obstacle distance
- 항상 제공되는 Evader GT 좌표
- GT 거리 기반 dense reward shaping

이렇게 제한한 이유는 Pursuer가 단순히 정답 좌표를 따라가는 policy가 아니라, 카메라에서 Evader를 찾고 화면 중심에 유지하는 policy를 학습하도록 만들기 위해서입니다.

## 핵심 구현

### 1. 색상 기반 Evader 탐지

Evader는 회색 도시 배경과 구분되는 붉은색 계열 material을 사용합니다. Pursuer는 RGB RenderTexture를 읽어 target color와의 chromaticity distance를 계산하고, threshold를 통과한 pixel들을 weighted centroid로 통합합니다.

탐지 결과로 다음 값을 계산합니다.

- 화면상 Evader 중심 좌표
- 화면 중심 대비 offset
- 색상 blob 면적
- blob 면적 기반 거리 proxy
- visible / not visible 상태

초기 학습 결과에서 `TargetVisible`이 약 1% 수준에 머무는 문제가 있었고, 이를 해결하기 위해 색상 threshold, 최소 pixel fraction, spawn 거리, yaw jitter를 조정했습니다.

### 2. Lucas-Kanade Optical Flow

Evader 색상 blob 주변의 luminance frame을 이용해 Lucas-Kanade optical flow를 계산했습니다.  
이 값은 Evader의 화면상 이동 방향과 속도를 policy가 추정할 수 있도록 viewport velocity feature로 제공됩니다.

이 과정에서 target이 이전 frame에도 보였을 때만 optical flow를 계산하도록 하여, target이 없는 frame에서 잘못된 flow가 발생하지 않게 했습니다.

### 3. 시각 기반 Reward 설계

기존 GT 거리 기반 dense shaping은 제거하고, 다음 시각 기반 보상으로 재구성했습니다.

- Evader가 카메라에 보이면 visibility reward
- Evader가 화면 중심에 가까울수록 centering reward
- Evader 색상 blob 면적이 클수록 visual area reward
- 색상 blob이 커지는 방향이면 approach reward
- 보던 target을 잃으면 lost target penalty
- capture, timeout, crash terminal reward

이를 통해 Pursuer가 좌표값을 직접 따라가는 대신, 카메라 화면에서 Evader를 유지하는 방향으로 학습되도록 했습니다.

### 4. Occlusion 대응 LSTM Memory

도심 환경에서는 Evader가 건물 뒤로 순간적으로 가려질 수 있습니다. 이를 위해 policy network에 recurrent memory를 사용했습니다.

LSTM memory는 다음 정보를 시간적으로 누적합니다.

- Evader가 마지막으로 보였던 viewport offset
- 색상 blob 면적 변화
- Lucas-Kanade optical flow 기반 화면상 이동 방향
- visible / not-visible 전환 패턴
- Pursuer 자신의 속도와 회전 상태

이 구조를 통해 Pursuer는 Evader가 일시적으로 보이지 않는 순간에도 직전 시각 정보와 이동 추세를 기반으로 추적 방향을 유지할 수 있습니다.  
현재 단계에서는 완전한 장기 occlusion 경로 예측보다는, 짧은 가림 상황에서 추적이 끊기지 않도록 하는 short-term memory 역할에 초점을 맞췄습니다.


## 기술 스택

- Unity
- ML-Agents
- C#
- PyTorch backend
- PPO
- RGB RenderTexture visual observation
- Lucas-Kanade Optical Flow
- TensorBoard diagnostics

## 요약

이 작업에서는 Pursuer가 Evader의 GT 좌표를 직접 따라가는 방식이 아니라, RGB 카메라에서 Evader의 색상을 탐지하고 화면상 움직임을 추정해 추적하는 학습 구조를 구현했습니다.  
초기에는 탐지율이 매우 낮았지만, 색상 기반 perception 조건을 조정하고 LSTM recurrent memory를 적용해 target visibility와 LK confidence를 크게 개선했습니다.

현재 policy는 단일 frame의 탐지 결과뿐 아니라 최근 시각 추적 흐름을 함께 활용하도록 구성되어 있으며, 다음 단계는 건물 충돌을 줄이고 occlusion 상황에서의 추적 안정성을 높이는 것입니다.
