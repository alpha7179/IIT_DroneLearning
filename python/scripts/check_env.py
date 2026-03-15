"""
check_env.py — Unity 빌드 없이 학습 파이프라인 사전 점검
담당: 이재왕 (work/evader)

사용법:
    python python/scripts/check_env.py

점검 항목:
    1. mlagents / torch 설치 및 버전
    2. Config YAML 파싱 및 행동 공간 확인
    3. 학습 커맨드 dry-run (Unity 없이 시작 여부 확인)
"""

import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
CONFIG = ROOT / "python/config/evader_s0_20260315_base.yaml"

PASS = "\033[92m✅\033[0m"
FAIL = "\033[91m❌\033[0m"
WARN = "\033[93m⚠️ \033[0m"


def check(label: str, ok: bool, detail: str = ""):
    status = PASS if ok else FAIL
    print(f"  {status} {label}", f"({detail})" if detail else "")
    return ok


def main():
    all_ok = True
    print("\n=== Evader 학습 환경 사전 점검 ===\n")

    # ── 1. Python 버전 ──────────────────────────────────────────
    print("[1] Python 버전")
    py_ver = sys.version_info
    ok = py_ver.major == 3 and py_ver.minor == 10
    all_ok &= check(f"Python 3.10", ok, f"현재: {py_ver.major}.{py_ver.minor}.{py_ver.micro}")

    # ── 2. mlagents ─────────────────────────────────────────────
    print("\n[2] mlagents 패키지")
    try:
        import mlagents
        ver = mlagents.__version__
        ok = "4.0" in ver or "0.30" in ver  # release_23 = 0.30.x or 4.0.x
        all_ok &= check("mlagents 4.0.x", ok, f"현재: {ver}")
    except ImportError:
        all_ok &= check("mlagents 설치", False, "pip install mlagents==4.0.0")

    # ── 3. torch ────────────────────────────────────────────────
    print("\n[3] PyTorch 버전")
    try:
        import torch
        ver = torch.__version__
        major = int(ver.split(".")[0])
        ok = major >= 2
        all_ok &= check(f"torch >= 2.0", ok, f"현재: {ver}")
        check("CUDA 사용 가능", torch.cuda.is_available(),
              "CPU 학습 가능하나 Colab GPU 권장")
    except ImportError:
        all_ok &= check("torch 설치", False, "pip install torch>=2.0.0")

    # ── 4. Config 파일 ──────────────────────────────────────────
    print("\n[4] Config YAML")
    all_ok &= check("Config 존재", CONFIG.exists(), str(CONFIG))

    if CONFIG.exists():
        try:
            import yaml
            with open(CONFIG) as f:
                cfg = yaml.safe_load(f)
            behaviors = cfg.get("behaviors", {})
            ok = "EvaderAgent" in behaviors
            all_ok &= check("behaviors.EvaderAgent 존재", ok)
            if ok:
                hp = behaviors["EvaderAgent"].get("hyperparameters", {})
                check("batch_size 설정", "batch_size" in hp, str(hp.get("batch_size")))
                check("max_steps 설정", "max_steps" in behaviors["EvaderAgent"],
                      str(behaviors["EvaderAgent"].get("max_steps")))
        except Exception as e:
            all_ok &= check("YAML 파싱", False, str(e))

    # ── 5. 필요 파일 존재 확인 ──────────────────────────────────
    print("\n[5] 소스 파일")
    cs_files = [
        ROOT / "IIT_DroneLearning/Assets/00. LJW/Scripts/EvaderAgent.cs",
        ROOT / "IIT_DroneLearning/Assets/00. LJW/Scripts/EvaderReward.cs",
        ROOT / "IIT_DroneLearning/Assets/00. LJW/Scripts/ScriptedEvader.cs",
    ]
    for f in cs_files:
        all_ok &= check(f.name, f.exists())

    # ── 6. Unity 빌드 확인 (없어도 경고만) ──────────────────────
    print("\n[6] Unity 빌드 (Colab용)")
    build_candidates = list(ROOT.glob("**/EvaderEnv.x86_64"))
    if build_candidates:
        check("Linux Headless 빌드", True, str(build_candidates[0]))
    else:
        print(f"  {WARN} Linux 빌드 없음 — Colab 학습 전 빌드 필요")
        print("       File > Build Settings > Linux > Server Build > Build")

    # ── 결과 ────────────────────────────────────────────────────
    print("\n" + "=" * 40)
    if all_ok:
        print(f"{PASS} 모든 필수 항목 통과. 학습 준비 완료.")
        print(f"\n학습 실행 커맨드 (Editor 연결):")
        print(f"  mlagents-learn {CONFIG.relative_to(ROOT)} \\")
        print(f"    --run-id=evader_s0a_goalonly_seed42 --force")
        print(f"\n(Unity Editor에서 Play 버튼을 누르면 자동 연결됩니다.)")
    else:
        print(f"{FAIL} 일부 항목 실패. 위 내용을 확인하고 재실행하세요.")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
