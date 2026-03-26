# /eval — 학습 모델 평가 실행

$ARGUMENTS

## 동작

인자로 run-id를 받아 고정 시드 평가를 실행하고 지표를 출력한다.

**사용법:**
```
/eval evader_s0_base_seed42          # 특정 run-id 평가
/eval evader_s0_base_seed42 50       # 50 에피소드 평가
```

## 실행 절차

1. `git branch --show-current` 로 브랜치가 `work/evader`인지 확인한다.
2. 인자에서 run-id와 에피소드 수(기본값 20)를 파싱한다.
3. `python/scripts/eval.py` 존재 여부를 확인하고, 다음 커맨드를 출력한다:

```bash
python python/scripts/eval.py \
  --run-id={run_id} \
  --n-episodes={n_episodes} \
  --seed=0
```

4. 평가 결과에서 아래 지표를 추출하여 표로 출력한다:

| 지표 | 값 | 목표 |
|---|---|---|
| survival_rate | - | M1: ≥70% |
| goal_reach_rate | - | M1: ≥50% |
| capture_rate | - | M1: ≤30% |
| mean_time_to_capture | - | 높을수록 좋음 |
| crash_rate | - | ≤15% |

5. 결과를 `docs/EXPERIMENTS.md`에 기록하도록 `/log` 사용을 안내한다.

## 평가 기준
- 고정 시드 사용 필수 (재현성)
- 최소 20 에피소드, 권장 50 에피소드
- eval 환경과 train 환경의 랜덤화 설정이 다른지 확인
