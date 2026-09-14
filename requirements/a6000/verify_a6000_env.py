#!/usr/bin/env python
"""TSE_Eval 설치 검증 — NVIDIA RTX A6000 (Ampere, sm_86).

설치가 끝난 conda env 에서 repo root 기준으로 실행한다:

    conda activate tseeval
    cd <repo root>
    python requirements/a6000/verify_a6000_env.py              # 기본 검사 (~1분)
    python requirements/a6000/verify_a6000_env.py --full       # + ECAPA·Whisper 실제 로드·추론
    python requirements/a6000/verify_a6000_env.py --skip-gpu   # GPU 없이 패키지·CPU 지표만
    python requirements/a6000/verify_a6000_env.py --skip-models  # 모델 다운로드 전

각 항목을 PASS / WARN / FAIL / SKIP 으로 출력하고, FAIL 이 하나라도 있으면 종료코드 1 을
반환한다. 패키지 핀은 이 파일과 같은 폴더의 requirements 파일을 직접 읽어 대조하므로
핀 목록을 스크립트에 중복해 적지 않는다.

이 스크립트가 특히 잡으려는 것은 TSE_Eval 의 "조용한 실패"다 — 에러 없이 숫자만 틀리거나
nan 이 되는 것들:
  · librosa 누락           → DNSMOS 4열이 전 행 nan (speechmos 가 의존성을 선언하지 않음)
  · CPU onnxruntime 공존   → onnxruntime-gpu 가 가려져 DNSMOS 가 CPU 로 떨어짐
  · CUDA EP 로드 실패      → 경고만 찍고 CPU 로 폴백 (actual providers 로만 보임)
  · asteroid 미설치        → scripts/eval_*.sh 기본 백엔드라 첫 행에서 RuntimeError
  · 모델 캐시 불완전       → WER 단계(채점 2시간 지점)에서 3 GB 재다운로드

이 스크립트는 h200/verify_h200_env.py 와 본문이 동일하고 맨 위 TARGET 블록만 다르다.
새 서버에 폴더 하나만 복사해 가도 단독으로 돌 수 있도록 공용 모듈로 빼지 않았다.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import os
import re
import shutil
import subprocess
import sys
import traceback
import warnings
from pathlib import Path

# =============================================================================
# TARGET — 장비별로 다른 값은 여기에만 둔다
# =============================================================================
TARGET = {
    "label": "RTX A6000 (Ampere)",
    "script": "requirements/a6000/verify_a6000_env.py",
    "requirements": "requirements_a6000.txt",
    "arch": "sm_86",
    "capability": (8, 6),
    # CUDA 12.1 휠(cu121)의 Linux 최소 드라이버 525.60.13
    "min_driver": 525,
    # 온프레미스 서버는 세션 휘발 개념이 없어 env · HF 캐시 위치를 강제하지 않는다.
    "persistent_prefix": None,
    "default_hf_home": None,
    "persistent_hf_prefixes": None,
}

PYTHON = "3.10.20"
TORCH = "2.5.1+cu121"
TORCH_CUDA = "12.1"
ENV_NAME = "tseeval"

# 모델 캐시: (HF repo 디렉터리, 있어야 하는 파일) — scripts/eval_tse.sh 선행 검사와 동일
MODEL_CACHE = (
    ("models--speechbrain--spkrec-ecapa-voxceleb", "embedding_model.ckpt"),
    ("models--openai--whisper-large-v3", "model.safetensors"),
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
HF_HOME_USER_SET = bool(os.environ.get("HF_HOME"))   # main() 이 기본값을 채우기 전에 기록

# =============================================================================
# 출력
# =============================================================================
_TTY = sys.stdout.isatty()
_COLOR = {"PASS": "\033[32m", "WARN": "\033[33m", "FAIL": "\033[31m", "SKIP": "\033[90m"}
_RESULTS: list[tuple[str, str, str]] = []


def _paint(tag: str) -> str:
    return f"{_COLOR[tag]}{tag}\033[0m" if _TTY else tag


def record(status: str, name: str, detail: str = "") -> None:
    _RESULTS.append((status, name, detail))
    print(f"  [{_paint(status)}] {name:<34} {detail}", flush=True)


def section(title: str) -> None:
    print(f"\n── {title} " + "─" * max(0, 60 - len(title)), flush=True)


def run(name: str, fn) -> None:
    """fn() 은 (status, detail) 을 반환한다. 예외는 FAIL 로 기록한다."""
    try:
        status, detail = fn()
    except Exception as e:  # noqa: BLE001 — 검증 도구라 모든 예외를 결과로 남긴다
        status, detail = "FAIL", f"{type(e).__name__}: {e}"
        if os.environ.get("VERIFY_DEBUG"):
            traceback.print_exc()
    record(status, name, detail)


def version_of(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def _speechlike(sr: int, sec: float = 3.0, seed: int = 0):
    """PESQ 가 발화로 인식하도록 무음 구간이 섞인 진폭변조 신호를 만든다."""
    import numpy as np
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * sec)) / sr
    env = (np.sin(2 * np.pi * 3 * t) > 0).astype("float64") * (0.5 + 0.5 * np.sin(2 * np.pi * 7 * t))
    carrier = np.sin(2 * np.pi * 180 * t) + 0.5 * np.sin(2 * np.pi * 520 * t) + 0.3 * rng.standard_normal(t.size)
    return 0.3 * env * carrier


def _triple(sr: int):
    """(est, ref, mix) — ref 에 약한 잡음을 더한 est, 강한 잡음을 더한 mix."""
    import numpy as np
    ref = _speechlike(sr)
    noise = np.random.default_rng(1).standard_normal(ref.size)
    return ref + 0.02 * noise, ref, ref + 0.2 * noise


# =============================================================================
# 1. 파이썬 · conda env
# =============================================================================
def check_python():
    v = sys.version.split()[0]
    if not v.startswith("3.10."):
        return "FAIL", f"{v} (3.10.x 필요 — 권장 {PYTHON}; onnxruntime-gpu 1.20.x 가 cp310 휠의 마지막 계열)"
    return ("PASS" if v == PYTHON else "WARN"), f"{v}" + ("" if v == PYTHON else f" (권장 {PYTHON})")


def check_env_location():
    prefix = sys.prefix
    name = os.environ.get("CONDA_DEFAULT_ENV", "?")
    if not (Path(prefix) / "conda-meta").is_dir():
        return "FAIL", f"{prefix} — conda env 가 아닙니다(시스템 python?). `conda activate {ENV_NAME}` 후 실행하세요"
    if name in ("base", "?"):
        return "WARN", f"{prefix} (env={name}) — base 가 아니라 전용 env({ENV_NAME})에 설치하세요"
    hint = TARGET.get("persistent_prefix")
    if hint and not prefix.startswith(hint):
        return "WARN", f"{prefix} (env={name}) — {hint} 밖이면 세션 종료 시 사라질 수 있습니다"
    return "PASS", f"{prefix} (env={name})"


# =============================================================================
# 2. GPU · PyTorch
# =============================================================================
def check_torch_build():
    import torch
    import torchaudio
    bad = []
    if torch.__version__ != TORCH:
        bad.append(f"torch {torch.__version__}≠{TORCH}")
    if torchaudio.__version__ != TORCH:
        bad.append(f"torchaudio {torchaudio.__version__}≠{TORCH}")
    if torch.version.cuda != TORCH_CUDA:
        bad.append(f"cuda {torch.version.cuda}≠{TORCH_CUDA}")
    detail = f"torch {torch.__version__} · torchaudio {torchaudio.__version__} · cuda {torch.version.cuda}"
    return ("FAIL", detail + "  ← " + ", ".join(bad)) if bad else ("PASS", detail)


def check_nvidia_smi():
    if not shutil.which("nvidia-smi"):
        return "FAIL", "nvidia-smi 를 찾을 수 없습니다 (드라이버 미설치?)"
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=30,
    ).stdout.strip().splitlines()
    if not out:
        return "FAIL", "GPU 가 조회되지 않습니다"
    name, driver, mem = [x.strip() for x in out[0].split(",")]
    if int(driver.split(".")[0]) < TARGET["min_driver"]:
        return "FAIL", f"{name} · driver {driver} (cu121 에는 ≥ {TARGET['min_driver']} 필요)"
    return "PASS", f"{name} · driver {driver} · {mem}" + (f" (GPU {len(out)}장)" if len(out) > 1 else "")


def check_cuda_arch(allow_other_gpu: bool):
    import torch
    if not torch.cuda.is_available():
        return "FAIL", "torch.cuda.is_available() = False"
    archs = torch.cuda.get_arch_list()
    if TARGET["arch"] not in archs:
        return "FAIL", f"arch_list 에 {TARGET['arch']} 없음 {archs} — cu121 휠이 아닌 torch 가 설치됨"
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    if cap != TARGET["capability"]:
        msg = (f"{name} capability {cap} ≠ 기대 {TARGET['capability']} "
               f"— 이 서버는 {TARGET['label']} 이 아닙니다")
        return ("WARN" if allow_other_gpu else "FAIL"), msg
    return "PASS", f"{name} · capability {cap} · arch_list 에 {TARGET['arch']} 포함"


def check_cuda_compute():
    """Whisper 가 GPU 에서 bf16 으로 돈다(tse_eval/metrics.py _get_asr) → bf16 연산 확인."""
    import torch
    import torchaudio
    if not torch.cuda.is_bf16_supported():
        return "FAIL", "bf16 미지원 GPU — WER(Whisper) 가 bf16 으로 로드됩니다"
    m = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    z = m @ m
    w = torchaudio.functional.resample(torch.randn(1, 24000, device="cuda"), 24000, 16000)
    torch.cuda.synchronize()
    ok = bool(torch.isfinite(z).all()) and w.shape[-1] == 16000
    return ("PASS" if ok else "FAIL"), f"bf16 matmul {tuple(z.shape)} · resample 24k→16k {tuple(w.shape)} · finite={ok}"


# =============================================================================
# 3. 패키지
# =============================================================================
_REQ_LINE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*([<>=!~].*)?$")


def parse_requirements(path: Path) -> list[tuple[str, str]]:
    reqs = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        m = _REQ_LINE.match(line)
        if m:
            reqs.append((m.group(1), (m.group(2) or "").replace(" ", "")))
    return reqs


def check_pins():
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    path = HERE / TARGET["requirements"]
    if not path.exists():
        record("FAIL", "requirements 파일", f"{path} 없음")
        return
    reqs = parse_requirements(path)
    print(f"  (대조 기준: {path.relative_to(REPO)} · {len(reqs)}개 패키지)")
    for name, spec in reqs:
        installed = version_of(name)
        if installed is None:
            record("FAIL", f"pkg {name}", f"미설치 (요구 {spec or '버전 무관'})")
        elif not spec:
            record("PASS", f"pkg {name}", f"{installed} (버전 무관)")
        elif Version(installed) in SpecifierSet(spec):
            record("PASS", f"pkg {name}", f"{installed} ⊨ {spec}")
        else:
            record("FAIL", f"pkg {name}", f"{installed} ⊭ {spec}")


def check_onnxruntime_single():
    gpu, cpu = version_of("onnxruntime-gpu"), version_of("onnxruntime")
    if gpu and cpu:
        return "FAIL", (f"onnxruntime {cpu} 과 onnxruntime-gpu {gpu} 공존 — 같은 모듈 이름을 덮어써 충돌. "
                        f"`pip uninstall -y onnxruntime onnxruntime-gpu` 후 requirements 재설치")
    if cpu:
        return "FAIL", f"CPU 용 onnxruntime {cpu} 만 설치됨 — onnxruntime-gpu 가 필요 (`.[cpu]` extra 를 깔았나요?)"
    if not gpu:
        return "FAIL", "onnxruntime 계열 미설치 — DNSMOS 가 전부 nan"
    return "PASS", f"onnxruntime-gpu {gpu} 한 줄만 설치됨"


def check_numpy_major():
    import numpy as np
    if int(np.__version__.split(".")[0]) >= 2:
        return "FAIL", f"numpy {np.__version__} — 2.x 는 torch 2.5.1 ABI 와 충돌. 1.26.4 로 고정"
    return "PASS", f"numpy {np.__version__} (< 2)"


def check_setuptools():
    from packaging.version import Version
    v = version_of("setuptools")
    if v is None:
        return "FAIL", "setuptools 미설치"
    if Version(v) >= Version("81"):
        return "FAIL", f"setuptools {v} ≥ 81 → pkg_resources 제거됨. `pip install \"setuptools<81\"`"
    return "PASS", f"setuptools {v} (< 81)"


def check_pip_check():
    p = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True, timeout=120)
    text = (p.stdout + p.stderr).strip()
    if p.returncode == 0:
        return "PASS", text.splitlines()[-1] if text else "OK"
    return "FAIL", " | ".join(text.splitlines()[:3])


def check_librosa_speechmos():
    """speechmos 는 librosa 를 import 하지만 선언하지 않는다 — 여기서 실패하면 DNSMOS 전부 nan."""
    librosa = importlib.import_module("librosa")
    importlib.import_module("speechmos.dnsmos")
    return "PASS", f"librosa {librosa.__version__} · speechmos {version_of('speechmos')} import OK"


def check_tse_eval_install():
    v = version_of("tse-eval")
    if v is None:
        return "FAIL", "tse-eval 미설치 — `pip install -e '.[test]'` (④단계)"
    mod = importlib.import_module("tse_eval")
    loc = Path(mod.__file__).resolve()
    if REPO not in loc.parents:
        return "WARN", f"tse-eval {v} 이 이 repo 가 아닌 {loc.parent} 에서 import 됨 (editable 설치 확인)"
    return "PASS", f"tse-eval {v} · editable → {loc.parent}"


def check_asteroid_backend():
    from tse_eval.backends import available_backends
    avail = available_backends()
    if not avail.get("asteroid"):
        return "FAIL", ("asteroid 미설치 — scripts/eval_*.sh 의 기본 BACKEND 라 첫 행에서 멈춤. "
                        "`pip install -e '.[test]'` (native 만 쓸 거면 BACKEND=native)")
    return "PASS", f"available={avail} · asteroid {version_of('asteroid')}"


# =============================================================================
# 4. 지표 (합성 신호, 모델 불필요)
# =============================================================================
def check_config_load():
    from tse_eval.config import load_config
    cfg = load_config()
    return "PASS", (f"configs/config.yaml · target_sr {cfg.get('target_sr')} · "
                    f"{len(cfg.get('metrics', []))}개 지표 · si_sdr_backend {cfg.get('si_sdr_backend')}")


def check_si_sdr_backends():
    from tse_eval.backends import si_sdr_family
    est, ref, mix = _triple(24000)
    nat = si_sdr_family(est, ref, mix, backend="native")
    ast = si_sdr_family(est, ref, mix, backend="asteroid")
    diff = max(abs(nat[k] - ast[k]) for k in ("si_sdr", "si_sdri", "input_si_sdr"))
    ok = diff < 1e-6 and nat["si_sdr"] > nat["input_si_sdr"]
    return ("PASS" if ok else "FAIL"), f"SI-SDR {nat['si_sdr']:.2f} dB · SI-SDRi {nat['si_sdri']:.2f} dB · |native−asteroid| {diff:.1e}"


def check_reference_metrics():
    """SI-SDR·ESTOI 는 24 kHz native, PESQ 는 16 kHz — tse_eval 의 샘플레이트 정책 그대로."""
    import numpy as np
    from tse_eval.metrics import compute_row_metrics
    est, ref, mix = _triple(24000)
    want = {"si_sdr", "si_sdri", "input_si_sdr", "stoi", "estoi", "pesq"}
    row = compute_row_metrics(est, ref, mix, sr=24000, metrics=want)
    vals = {k: row[k] for k in sorted(want)}
    bad = [k for k, v in vals.items() if not np.isfinite(v)]
    detail = " · ".join(f"{k} {v:.3f}" for k, v in vals.items())
    return ("FAIL", detail + f"  ← nan: {bad}") if bad else ("PASS", detail)


def check_ort_providers(skip_gpu: bool):
    import onnxruntime as ort
    avail = ort.get_available_providers()
    if "CUDAExecutionProvider" not in avail:
        if skip_gpu:
            return "SKIP", f"{avail} (--skip-gpu)"
        return "FAIL", f"{avail} — CUDAExecutionProvider 없음. CPU onnxruntime 이 설치됐거나 onnxruntime-gpu 가 아님"
    return "PASS", f"onnxruntime {ort.__version__} · {avail}"


def _dnsmos_probe(providers: str) -> int:
    """(서브프로세스 전용) tse_eval CLI 와 같은 순서로 DNSMOS 를 돌리고 JSON 한 줄을 출력한다.

    ★ 별도 프로세스인 이유: 같은 프로세스에서 torch 가 먼저 cuDNN 을 올리면(예: CUDA conv1d)
      onnxruntime-gpu 의 CUDA EP 가 `libcudnn_ops.so.9: undefined symbol` 로 로드에 실패하고
      ★ 경고만 찍고 CPU 로 폴백 ★ 한다(2026-09-14 이 H200 에서 재현). tse_eval 파이프라인은
      DNSMOS 세션을 ECAPA/Whisper 보다 먼저 만들어 이 순서를 피하므로, 검증도 깨끗한
      프로세스에서 같은 순서로 해야 실제 채점과 같은 결론이 나온다.
    """
    import json
    import numpy as np
    sys.path.insert(0, str(REPO))
    from tse_eval import ort_setup
    from tse_eval.metrics import dnsmos

    ort_setup.configure_onnxruntime(threads=4, providers=providers, verbose=False)
    out = dnsmos(_speechlike(16000))
    vals = {n: out[f"dnsmos_{n}"] for n in ("sig", "bak", "ovrl", "p808")}
    print("DNSMOS_PROBE " + json.dumps({"vals": {k: (v if np.isfinite(v) else None) for k, v in vals.items()},
                                       "providers": ort_setup.actual_providers() or []}))
    return 0


def check_dnsmos(skip_gpu: bool):
    """tse_eval 과 같은 경로(ort_setup → metrics.dnsmos)로 돌리고, 실제로 붙은 EP 를 확인한다."""
    import json
    want = "cpu" if skip_gpu else "cuda"
    p = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--_dnsmos-probe", want],
                       capture_output=True, text=True, timeout=600, cwd=REPO)
    line = next((l for l in p.stdout.splitlines() if l.startswith("DNSMOS_PROBE ")), None)
    if line is None:
        tail = " | ".join((p.stderr or p.stdout).strip().splitlines()[-2:])
        return "FAIL", f"probe 실패 (exit {p.returncode}): {tail}"
    res = json.loads(line[len("DNSMOS_PROBE "):])
    vals, got = res["vals"], res["providers"]
    detail = " · ".join(f"{k} {'nan' if v is None else f'{v:.2f}'}" for k, v in vals.items()) + f" · EP {got}"
    if any(v is None for v in vals.values()):
        return "FAIL", detail + "  ← nan (librosa 누락 또는 onnxruntime 문제)"
    if not skip_gpu and (not got or got[0] != "CUDAExecutionProvider"):
        hint = next((l.strip() for l in p.stderr.splitlines() if "Failed to load" in l or "undefined symbol" in l), "")
        return "FAIL", detail + f"  ← CUDA 를 요청했지만 CPU 로 폴백 {hint[-120:]}"
    return "PASS", detail


# =============================================================================
# 5. 모델 캐시 · CLI
# =============================================================================
def _hf_home() -> tuple[Path, bool]:
    """(HF_HOME 경로, 사용자가 직접 설정했는지)."""
    env = os.environ.get("HF_HOME")
    home = Path(env) if env else Path.home() / ".cache" / "huggingface"
    return home, HF_HOME_USER_SET


def check_hf_home():
    home, from_env = _hf_home()
    hint = TARGET.get("default_hf_home")
    if not from_env and hint:
        return "WARN", f"HF_HOME 미설정 — 기본값 {home} 을 가정합니다. `export HF_HOME={hint}` 를 세션마다 하세요"
    if not from_env:
        return "WARN", (f"HF_HOME 미설정 → {home} 을 봅니다. scripts/download_models.py 는 미설정 시 "
                        f"/home/work/my-checkpoints/hf_cache 에 받으므로 어긋날 수 있음 — `export HF_HOME=...` 권장")
    if not home.is_dir():
        return "FAIL", f"HF_HOME={home} 디렉터리가 없습니다"
    prefixes = TARGET.get("persistent_hf_prefixes")
    if prefixes and not any(str(home).startswith(p) for p in prefixes):
        return "WARN", f"HF_HOME={home} — 영속 저장소({', '.join(prefixes)}) 밖이라 세션 종료 시 사라질 수 있음"
    return "PASS", f"HF_HOME={home}"


def check_model_cache(repo: str, key: str):
    home, _ = _hf_home()
    snap = Path(os.environ.get("HF_HUB_CACHE") or home / "hub") / repo / "snapshots"
    if not snap.is_dir():
        return "FAIL", f"{snap} 없음 → `python scripts/download_models.py`"
    hits = list(snap.rglob(key))
    if not hits:
        return "FAIL", f"{repo} 에 {key} 없음 (다운로드 불완전) → `python scripts/download_models.py`"
    size = hits[0].resolve().stat().st_size / 1e6
    return "PASS", f"{key} {size:,.0f} MB"


def check_no_incomplete():
    home, _ = _hf_home()
    hub = Path(os.environ.get("HF_HUB_CACHE") or home / "hub")
    bad = [p for repo, _ in MODEL_CACHE for p in (hub / repo).rglob("*.incomplete")] if hub.is_dir() else []
    if bad:
        return "FAIL", f".incomplete {len(bad)}개 (다운로드 중단) → `python scripts/download_models.py` 로 마무리"
    return "PASS", "중단된 다운로드(.incomplete) 없음"


def check_cli():
    p = subprocess.run([sys.executable, "-m", "tse_eval", "--help"], capture_output=True, text=True,
                       timeout=300, cwd=REPO)
    if p.returncode != 0:
        return "FAIL", " | ".join((p.stderr or p.stdout).strip().splitlines()[-2:])
    return "PASS", "`python -m tse_eval --help` 종료코드 0"


# =============================================================================
# 6. (--full) 모델 실제 로드·추론 — HF_HUB_OFFLINE=1 로 캐시만으로 도는지까지 본다
# =============================================================================
def check_spk_sim():
    from tse_eval.metrics import spk_sim, _get_spk
    x = _speechlike(16000).astype("float32")
    s = spk_sim(x, x)
    dev = _get_spk().get("device")
    ok = s == s and s > 0.99
    return ("PASS" if ok else "FAIL"), f"ECAPA on {dev} · 같은 신호 cos = {s:.4f} (기대 ≈ 1)"


def check_wer():
    import math
    from tse_eval.metrics import wer, _get_asr
    out = wer(_speechlike(16000).astype("float32"), "hello world")
    bundle = _get_asr()
    ok = math.isfinite(out["wer"]) and math.isfinite(out["wer_raw"])
    return ("PASS" if ok else "FAIL"), (f"Whisper on {bundle.get('device')} ({bundle.get('dtype')}) · "
                                        f"wer {out['wer']:.2f} · hyp {out['wer_hyp'][:30]!r}")


# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=f"TSE_Eval 설치 검증 — {TARGET['label']}")
    ap.add_argument("--skip-gpu", action="store_true",
                    help="GPU·CUDA 검사를 건너뛰고 DNSMOS 를 CPU 로 확인한다")
    ap.add_argument("--skip-models", action="store_true",
                    help="HF_HOME·모델 캐시 검사를 건너뛴다 (download_models.py 실행 전)")
    ap.add_argument("--full", action="store_true",
                    help="ECAPA·Whisper 를 캐시에서 실제로 로드해 spk_sim·wer 을 계산한다 (+30~60초)")
    ap.add_argument("--allow-other-gpu", action="store_true",
                    help="GPU 모델이 대상과 달라도 FAIL 대신 WARN (스크립트 자체 점검용)")
    ap.add_argument("--_dnsmos-probe", dest="dnsmos_probe", choices=("cpu", "cuda"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.dnsmos_probe:
        return _dnsmos_probe(args.dnsmos_probe)
    if args.full and args.skip_models:
        ap.error("--full 은 모델 캐시가 필요하므로 --skip-models 와 함께 쓸 수 없습니다")

    if not os.environ.get("HF_HOME") and TARGET.get("default_hf_home"):
        os.environ["HF_HOME"] = TARGET["default_hf_home"]     # transformers import 전에 확정
    if args.full:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")         # 캐시만으로 도는지 확인
    sys.path.insert(0, str(REPO))

    print(f"TSE_Eval 설치 검증 — {TARGET['label']}  (repo: {REPO})")

    section("1. Python · conda env")
    run("python", check_python)
    run("conda env 위치", check_env_location)

    section("2. GPU · PyTorch")
    run("torch / torchaudio 빌드", check_torch_build)
    if args.skip_gpu:
        record("SKIP", "nvidia-smi · CUDA · bf16", "--skip-gpu")
    else:
        run("nvidia-smi", check_nvidia_smi)
        run(f"CUDA arch ({TARGET['arch']})", lambda: check_cuda_arch(args.allow_other_gpu))
        run("CUDA 연산 (bf16 · resample)", check_cuda_compute)

    section("3. 패키지")
    check_pins()
    run("onnxruntime 배포본 1개", check_onnxruntime_single)
    run("numpy < 2", check_numpy_major)
    run("setuptools < 81", check_setuptools)
    run("pip check", check_pip_check)
    run("librosa · speechmos import", check_librosa_speechmos)
    run("tse-eval 설치", check_tse_eval_install)
    run("si_sdr 백엔드 asteroid", check_asteroid_backend)

    section("4. 지표 (합성 신호)")
    run("config.yaml 로드", check_config_load)
    run("SI-SDR native ≡ asteroid", check_si_sdr_backends)
    run("SI-SDR·STOI·ESTOI·PESQ", check_reference_metrics)
    run("onnxruntime providers", lambda: check_ort_providers(args.skip_gpu))
    run("DNSMOS 값 · 실제 EP", lambda: check_dnsmos(args.skip_gpu))

    section("5. 모델 캐시 · CLI")
    if args.skip_models:
        record("SKIP", "HF_HOME · 모델 캐시", "--skip-models")
    else:
        run("HF_HOME", check_hf_home)
        for repo, key in MODEL_CACHE:
            run(repo.split("--", 2)[-1], lambda r=repo, k=key: check_model_cache(r, k))
        run("중단된 다운로드", check_no_incomplete)
    run("CLI", check_cli)

    section("6. 모델 로드·추론 (--full)")
    if args.full:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            run("Speaker Similarity (ECAPA)", check_spk_sim)
            run("WER (Whisper large-v3)", check_wer)
    else:
        record("SKIP", "ECAPA · Whisper 로드", "--full 로 켜기")

    n = {k: sum(1 for s, *_ in _RESULTS if s == k) for k in ("PASS", "WARN", "FAIL", "SKIP")}
    print("\n" + "═" * 64)
    print(f"  결과: PASS {n['PASS']} · WARN {n['WARN']} · FAIL {n['FAIL']} · SKIP {n['SKIP']}")
    if n["FAIL"]:
        print("  ❌ 설치에 문제가 있습니다. FAIL 항목과 INSTALL.md 의 '문제 해결' 절을 확인하세요.")
    elif n["WARN"]:
        print("  ⚠️  사용은 가능하나 WARN 항목을 확인하세요.")
    else:
        print("  ✅ 설치가 올바릅니다.")
    print("═" * 64)
    return 1 if n["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
