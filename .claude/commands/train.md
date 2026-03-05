# /train — ML-Agents 학습 실행

$ARGUMENTS

## 동작

인자로 Stage 번호와 실험 이름을 받아 적절한 mlagents-learn 커맨드를 구성하고 실행한다.

**사용법:**
```
/train s0              # Stage0 기본 학습
/train s0 base seed42  # Stage0, base config, seed 42
/train s1 obstacle     # Stage1 장애물 환경
```

## 실행 절차

1. `git branch --show-current` 로 브랜치가 `work/evader`인지 확인한다.
2. 인자에서 stage(`s0`~`s3`), exp 이름, seed 값을 파싱한다.
3. 해당 config 파일 `python/config/evader_{stage}_{exp}.yaml` 존재 여부를 확인한다.
   - 없으면 `python/config/evader_s0_template.yaml`을 기반으로 새 config 생성을 제안한다.
4. run-id를 `evader_{stage}_{exp}_{seed}` 형식으로 구성한다.
5. 아래 커맨드를 출력하고 실행 여부를 확인한다:

```bash
mlagents-learn python/config/evader_{stage}_{exp}.yaml \
  --run-id=evader_{stage}_{exp}_{seed} \
  --force
```

Stage 전환(warm-start) 시:
```bash
mlagents-learn python/config/evader_{next_stage}.yaml \
  --run-id=evader_{next_stage}_{exp}_{seed} \
  --initialize-from=evader_{prev_stage}_{prev_exp}_{seed} \
  --force
```

6. 학습 완료 후 결과를 `docs/EXPERIMENTS.md`에 기록하라고 안내한다.

## 주의사항
- `python/models/` 및 `python/results/` 내 파일은 git에 추가하지 않는다.
- 실험 메타데이터(config, run-id, seed, 지표)는 반드시 `docs/EXPERIMENTS.md`에 기록한다.
