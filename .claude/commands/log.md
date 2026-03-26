# /log — 실험 결과 기록

$ARGUMENTS

## 동작

실험 결과를 `docs/EXPERIMENTS.md`에 한 줄로 기록한다.

**사용법:**
```
/log evader_s0_base_seed42 "survival=75% goal=52% capture=25% 보상 우상향 확인"
/log                        # 현재 상태에서 기록할 정보 수집
```

## 실행 절차

1. 인자가 없으면 다음을 수집하여 사용자에게 확인받는다:
   - `git log --oneline -1` → 최신 커밋 hash
   - 현재 날짜
   - `python/config/` 내 가장 최근 수정된 config 파일 이름

2. `docs/EXPERIMENTS.md`를 읽어 기존 테이블 형식을 확인한다.

3. 아래 형식으로 한 줄을 추가한다:

```
| {date} | {git_commit_short} | {config_file} | {run_id} | {seed} | survival={X}% goal={X}% capture={X}% | {comment} |
```

4. Edit 도구로 `docs/EXPERIMENTS.md`에 해당 줄을 추가한다.

5. 변경사항을 커밋하도록 안내한다:
```bash
git add docs/EXPERIMENTS.md
git commit -m "[Docs] 실험 결과 기록: {run_id}"
```

## 주의
- 실험 기록은 작업 당일 반드시 남긴다.
- 지표가 목표치 미달인 경우 원인 분석 코멘트를 한 줄 추가한다.
