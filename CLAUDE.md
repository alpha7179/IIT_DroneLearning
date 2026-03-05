# CLAUDE.md — Evader RL Owner (이재왕) 작업 기준서

> Claude Code가 이 레포에서 작업할 때 **항상 이 파일을 먼저 읽고** 아래 규칙을 준수한다.
> 세부 설계 원칙은 [AGENT.md](AGENT.md) 를 함께 참조한다.

---

## 1. 내 역할 및 맥락

- **담당자**: 이재왕 (이재왕, zaeee-wang)
- **역할**: 회피 드론(Evader) RL 에이전트 개발 및 실험 분석
- **브랜치**: `work/evader` (아래 Git 규칙 참조)
- **프로젝트**: 도심 Occlusion 환경 MARL 드론 추격-회피 시스템 (아주대 2026-1 캡스톤)

---

## 2. Version Lock (절대 변경 금지)

| 컴포넌트 | 버전 |
|---|---|
| Unity | 6000.0.58f2 |
| ML-Agents | 4.0.x (`mlagents==4.0.0`) |
| Python | 3.10 |
| PyTorch | 2.x (`torch>=2.0.0,<3.0.0`) |
| URP (Unity) | 17.0.4 |

버전 불일치 시 마이그레이션 비용이 재현성보다 먼저 발생한다. **절대 임의 업그레이드 금지.**

---

## 3. Git 규칙 (Hard Constraint)

- **작업 브랜치는 `work/evader` 단 하나**만 사용한다.
- 기능 단위 작업은 `feature/evader-{기능명}` 브랜치에서 개발 후 `work/evader`로 PR 병합.
- 작업 시작 시 반드시 현재 브랜치 확인: `git branch --show-current`
- main / work/pursuer 브랜치는 **절대 직접 수정하지 않는다**.
- main과 동기화가 필요하면, 커맨드를 제안하고 **사용자가 직접 실행**하도록 한다.

### 커밋 메시지 컨벤션 (`[태그] 내용` 형식)

| 태그 | 사용 상황 |
|---|---|
| `[Feat]` | 새로운 기능 구현 |
| `[Add]` | 코드·라이브러리·에셋 추가 |
| `[Update]` | 기존 기능 강화 |
| `[Fix]` | 버그 및 오류 해결 |
| `[Docs]` | 문서 작성 및 수정 |
| `[Setting]` | 프로젝트 설정 변경 |
| `[Refactor]` | 코드 구조 개선 (기능 변경 없음) |
| `[Remove]` | 파일 및 리소스 삭제 |

---

## 4. 파일 범위 제한 (Scope)

Claude Code가 **기본적으로 건드릴 수 있는 파일**:
```
AGENT.md
CLAUDE.md
python/config/evader_*.yaml
python/scripts/
python/utils/
python/results/
docs/EXPERIMENTS.md
IIT_DroneLearning/Assets/Ljw/**
```

**건드리지 말 것** (담당자 합의 없이):
```
IIT_DroneLearning/Assets/Samples/   ← Unity 패키지
IIT_DroneLearning/Packages/         ← 패키지 매니페스트
IIT_DroneLearning/ProjectSettings/  ← Unity 설정
README.md                           ← 공통 프로젝트 문서
Docs/IMPLEMENTATION_BLUEPRINT.md    ← 공통 운영 기준
```

---

## 5. 디렉토리 구조

```
IIT_DroneLearning/                  ← 저장소 루트
├── CLAUDE.md                       ← 이 파일 (Claude Code 기준)
├── AGENT.md                        ← Evader RL Owner 상세 설계
├── README.md                       ← 프로젝트 개요
├── requirements.txt                ← Python 의존성
├── .gitignore
├── Docs/
│   └── IMPLEMENTATION_BLUEPRINT.md
├── docs/
│   └── EXPERIMENTS.md              ← 실험 로그 (반드시 기록)
├── python/
│   ├── config/                     ← ML-Agents YAML 설정
│   │   └── evader_s0_*.yaml
│   ├── scripts/                    ← 학습/평가 스크립트
│   │   ├── train.py
│   │   └── eval.py
│   ├── utils/                      ← 공통 유틸리티
│   │   └── logger.py
│   ├── models/                     ← onnx/ckpt (git 제외)
│   └── results/                    ← 실험 로그 (대용량 git 제외)
└── IIT_DroneLearning/              ← Unity 6 프로젝트
    └── Assets/
        └── Ljw/                    ← 이재왕 개인 작업 폴더
            ├── Scripts/
            ├── Scenes/
            ├── Prefabs/
            └── Materials/
```

---

## 6. 학습 파이프라인 (Quick Reference)

### 환경 설치
```bash
pip install -r requirements.txt
```

### Stage별 학습 실행
```bash
# Stage0: Ray + 힌트 (장애물 없음)
mlagents-learn python/config/evader_s0_base.yaml --run-id=evader_s0_v1 --force

# Stage1: 장애물 도입 + 힌트 제거
mlagents-learn python/config/evader_s1_obstacle.yaml --run-id=evader_s1_v1 \
  --initialize-from=evader_s0_v1 --force

# TensorBoard 확인
tensorboard --logdir python/results/
```

### Config / Run-ID 네이밍 규칙
- Config: `python/config/evader_s{stage}_{yyyymmdd}_{설명}.yaml`
- Run-ID: `evader_s{stage}_{exp}_{seed}`  (예: `evader_s0_base_seed42`)

---

## 7. 평가 지표 (Eval Metrics)

| 지표 | 설명 | 목표 |
|---|---|---|
| `survival_rate` | timeout까지 생존한 에피소드 비율 | M1: ≥70% |
| `goal_reach_rate` | 목표 지점 도달 비율 | M1: ≥50% |
| `capture_rate` | 포획된 에피소드 비율 | M1: ≤30% |
| `mean_time_to_capture` | 포획 시 평균 경과 시간 | 높을수록 좋음 |
| `los_break_rate` | LOS 차단 성공 비율 | Stage2+ |

Eval은 반드시 **고정 시드 20~50 에피소드**로 실행하고 결과를 `docs/EXPERIMENTS.md`에 기록한다.

---

## 8. 디버깅 체크리스트 (학습 안 붙을 때 순서대로)

1. episode가 너무 빨리 종료되지 않는가? (crash/capture 과다)
2. capture 조건이 너무 빡세거나 느슨하지 않은가?
3. 액션 스케일이 과도하지 않은가? (전복/폭주 여부)
4. 보상 부호/크기 이상이 없는가? (shaping만 과대 등)
5. 관측값에 NaN/Inf 또는 좌표계 혼용이 없는가?
6. fixed timestep / frame skip이 일관되는가?
7. eval과 train 환경 설정이 동일한가?

---

## 9. 팀 인터페이스 의존성

| 제공자 | 요청 사항 | 우선순위 |
|---|---|---|
| World (이강민) | PID 파이프라인, deterministic reset, capture/collision 판정 API, goal point API | 필수 |
| Sensor (배민우) | Ray 태그 표준, 역방향 LOS raycast (is_visible_to_pursuer) | 필수 |
| Pursuer (박재현) | Stage0용 scripted 추격 baseline, RL pursuer 인터페이스 | 필수 |

인터페이스 변경 시 **반드시 합의 PR**을 먼저 제안하고 사용자 승인 후 진행.

---

## 10. 금지 행동

- 실행 커맨드 없이 "아마 이렇게 동작할 것"이라 추측하여 코드 실행 금지
- PR 없이 main 또는 work/pursuer 브랜치 수정 금지
- 대규모 리팩터 또는 폴더 구조 변경을 합의 없이 진행 금지
- `python/models/`, `python/results/` 내 대용량 파일 git add 금지
