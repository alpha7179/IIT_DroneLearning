# /check — 작업 전 사전 점검

작업 시작 전 다음 항목을 순서대로 확인하고 각 결과를 표로 출력한다.

## 점검 절차

1. **브랜치 확인**: `git branch --show-current` 실행 → `work/evader` 여야 한다.
2. **스테이징 상태**: `git status` 실행 → 미커밋 변경사항 목록 확인.
3. **최신 커밋**: `git log --oneline -5` 실행 → 최근 작업 이력 확인.
4. **Python 환경**: `python --version` 및 `pip show mlagents torch` 실행 → 버전 확인.
5. **디렉토리 구조**: `python/config/`, `python/scripts/`, `python/results/` 존재 여부 확인.

## 출력 형식

점검 결과를 아래 형식의 표로 출력한다:

| 항목 | 결과 | 상태 |
|---|---|---|
| 브랜치 | work/evader | ✅ / ❌ |
| 미커밋 변경 | 없음 / N개 파일 | ✅ / ⚠️ |
| Python 버전 | 3.10.x | ✅ / ❌ |
| mlagents 버전 | 4.0.x | ✅ / ❌ |
| torch 버전 | 2.x | ✅ / ❌ |
| python/ 구조 | 존재 | ✅ / ❌ |

문제가 있는 항목은 해결 방법을 함께 제안한다.
