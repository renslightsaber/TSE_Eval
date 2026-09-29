# 🔵 TSE_Eval 설치 가이드 — NVIDIA RTX A6000

> **대상**: Ampere 아키텍처(**sm_86**) 서버 — RTX A6000 (48 GB)
> **소요**: 약 5~10분 (pesq 소스 빌드 포함) + 모델 다운로드 3.1 GB
> **디스크**: env 약 7 GB + 모델 3.2 GB
> **패키지 목록**: [`requirements_a6000.txt`](./requirements_a6000.txt) · **검증 스크립트**: [`verify_a6000_env.py`](./verify_a6000_env.py)

TSE_Eval 은 채점 도구라, **어느 GPU 에서 채점하든 같은 바이너리로 계산해야** 세 프로젝트의 숫자가 비교 가능합니다.
그래서 A6000 파일의 패키지 핀은 [H200 파일](../h200/requirements_h200.txt)과 **23개 전부 동일**하고,
설치 절차도 같습니다. 다른 점은 [§1.1](#11-h200-과-다른-점) 에 모았습니다.

---

## 📋 목차

1. [대상 환경](#1-대상-환경)
2. [사전 확인](#2-사전-확인)
3. [설치 — 순서대로 실행](#3-설치--순서대로-실행)
4. [설치 검증](#4-설치-검증)
5. [문제 해결](#5-문제-해결)
6. [하지 말아야 할 것](#6-하지-말아야-할-것)

---

## 1. 대상 환경

| 항목 | 값 |
|---|---|
| GPU | NVIDIA RTX A6000 · 48 GB · compute capability **(8, 6)** |
| 드라이버 | **≥ 525** (CUDA 12.1 휠의 Linux 최소 요구 525.60.13) |
| Python | **3.10.20** (conda) |
| PyTorch | **2.5.1+cu121** · torchaudio 2.5.1+cu121 · CUDA 12.1 |
| 핵심 패키지 | onnxruntime-gpu 1.20.2 · librosa 0.11.0 · speechmos 0.0.1.1 · speechbrain 1.0.3 · transformers 4.46.3 · accelerate 1.10.0 · jiwer 3.1.0 · numpy 1.26.4 · **setuptools < 81** · pesq 0.0.4 |
| 컴파일러 | gcc (pesq 빌드용) |

> ⚠️ **검증 범위 안내**
> 2026-09-14 에는 A6000 실기가 없어 **A6000 에서 직접 설치해 보지는 못했습니다.** 확인한 것은 다음과 같습니다.
> - 이 문서의 §3 명령을 **H200 서버에서 그대로** 따라 임시 env 에 처음부터 설치(pip 캐시 경로만 추가) →
>   111 패키지 · `pip check` 클린 · `verify_a6000_env.py --full` 에서 **GPU 판별 1건(기대 (8,6) vs H200 (9,0))만 FAIL, 나머지 47건 PASS**
>   (`--allow-other-gpu` 면 WARN 1 · 종료코드 0) · pytest 105 passed, 1 skipped · 합성 예제 DNSMOS nan 0 / CUDA EP.
>   설치된 패키지 목록(`pip freeze --all`)은 H200 가이드로 설치한 env 와 **완전히 동일**했습니다.
> - 같은 torch cu121 + `onnxruntime-gpu>=1.19.2` 조합이 **A6000 실기(driver 530.30.02)** 에서 CUDA EP 를 로드한 기록:
>   `tpex/requirements/a6000/README.md` §3-2.
>
> 즉 **패키지 구성은 검증됐고, A6000 고유 항목(sm_86 커널 · 드라이버 · cuDNN 탐색)은 미검증**입니다.
> A6000 에서 처음 설치하면 [§4](#4-설치-검증) 결과를 이 문서에 기록해 주세요.

### 1.1 H200 과 다른 점

| 항목 | A6000 (이 문서) | H200 ([가이드](../h200/INSTALL.md)) |
|---|---|---|
| 아키텍처 | Ampere **sm_86** · capability (8, 6) | Hopper sm_90 · (9, 0) |
| VRAM | **48 GB** (채점은 약 5 GB 사용 → 충분) | 141 GB |
| 최소 드라이버 | 525 | 535 |
| 패키지 핀 | **동일** | 동일 |
| env 위치 | 제약 없음 (온프레미스) | vFolder(`/home/work/my-code/`) 안이어야 영속 |
| `HF_HOME` | **직접 지정 필수** ([③-⑤](#⑤-hf_home-지정--필수)) — 스크립트 기본값이 H200 경로 | `/home/work/my-checkpoints/hf_cache` |
| pip 캐시 | 보통 켜져 있음 → `--cache-dir` 불필요 | NGC 이미지가 꺼둠 → `--cache-dir` 필수 |
| 시스템 cuDNN | 없을 수 있음 → torch 번들(9.1) 을 씀 ([§5.2](#52-dnsmos-가-cpu-로-돎--cuda-ep-로드-실패)) | 9.4.0 설치돼 있음 |
| `scripts/eval_*.sh` | 경로 기본값이 H200 서버 기준 → **환경변수로 덮어쓰기** ([§4.4](#44-채점-래퍼를-쓸-때)) | 그대로 |
| 속도 수치 | 미측정 | DNSMOS 175 ms/utt 등 실측 |

---

## 2. 사전 확인

```bash
nvidia-smi                # ① "RTX A6000" 이 보이고 Driver Version 이 525 이상인가
gcc --version             # ② pesq 가 C 확장을 빌드하므로 gcc 필요
conda --version           # ③ conda (miniforge/miniconda/anaconda 무관)
df -h ~                   # ④ 약 11 GB 여유 (env 7 GB + 모델 3.2 GB) — HF_HOME 을 둘 디스크도 확인
```

---

## 3. 설치 — 순서대로 실행

> **순서가 중요합니다.** 특히 ②(PyTorch)를 ③보다 먼저, 반드시 단독으로 설치하세요.
> 설치 중에는 다른 터미널에서 **같은 env 에 conda 명령을 쓰지 마세요** (env 디렉터리가 도중에 사라져
> `No module named 'urllib'` 같은 엉뚱한 에러가 납니다).

### ⓪ 재설치라면 먼저 청소

```bash
conda deactivate 2>/dev/null
conda env remove -n tseeval -y
cd <TSE_Eval repo 경로>
rm -rf tse_eval.egg-info build             # 이전 editable 설치 잔여물
```

`tse_eval.egg-info` 를 남겨두면 ③에서 `ERROR: ... tse-eval 0.1.0 requires onnxruntime, which is not installed.`
가 뜹니다. 무해하지만 실패로 착각하게 됩니다.

### ① conda 환경 생성

```bash
conda create -n tseeval -y python=3.10.20
conda activate tseeval
cd <TSE_Eval repo 경로>                    # 이후 명령은 모두 repo root 에서
```

비대화형 셸(스크립트, tmux 안의 새 셸 등)에서 `conda activate` 가 안 되면 먼저
`source "$(conda info --base)/etc/profile.d/conda.sh"` 를 실행하세요.

### ② PyTorch 먼저, 단독으로

```bash
pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
```

> **왜 따로?** `speechbrain` 등이 torch 를 요구하므로, 먼저 깔지 않으면 PyPI 기본 torch(다른 CUDA 빌드)가
> 끼어들어 스택이 섞일 수 있습니다. `Failed to initialize NumPy` 경고는 정상입니다(numpy 는 ③에서 깔림).

### ③ 나머지 패키지

```bash
pip install -r requirements/a6000/requirements_a6000.txt
```

`pesq` 가 소스 빌드되므로 한동안 멈춘 것처럼 보일 수 있습니다.
`Building wheel for pesq ... finished with status 'done'` 이 보이면 성공입니다.

### ④ 프로젝트 + 테스트 도구

```bash
pip install -e '.[test]'
```

`[test]` = pytest + **asteroid**. asteroid 는 SI-SDR 등가성 오라클이면서 `scripts/eval_*.sh` 의
**기본 백엔드**이므로 표준 설치에 포함합니다.

```bash
pip list | grep onnxruntime      # → onnxruntime-gpu 1.20.2 한 줄만
pip check                        # → No broken requirements found.
```

> ⚠️ `pip install -e '.[cpu,test]'` 는 **CPU 전용 머신용**입니다. A6000 에서 쓰면 CPU `onnxruntime` 이
> 함께 깔려 `onnxruntime-gpu` 와 충돌합니다.

### ⑤ `HF_HOME` 지정 — 필수

모델 캐시를 둘 경로를 **직접** 정하고, 이후 모든 셸에서 같은 값을 쓰세요.

```bash
export HF_HOME=<모델 캐시 경로>             # 예: $HOME/hf_cache 또는 팀 공유 디스크
echo "export HF_HOME=<모델 캐시 경로>" >> ~/.bashrc   # 온프레미스라 $HOME 이 영속이면 편리
```

> 왜 필수인가: 값이 없으면 `scripts/download_models.py` 와 `scripts/eval_tse.sh` 는 **H200 서버 경로**
> (`/home/work/my-checkpoints/hf_cache`)를, `python -m tse_eval` 은 HuggingFace 기본값(`~/.cache/huggingface`)을
> 써서 **받은 곳과 읽는 곳이 어긋납니다** — 채점 도중 Whisper 3 GB 를 다시 받게 됩니다.

### ⑥ 모델 2개 다운로드 + 검증 (약 3.2 GB)

```bash
python scripts/download_models.py
```

| 용도 | 모델 | 크기 |
|---|---|---|
| Speaker Similarity | `speechbrain/spkrec-ecapa-voxceleb` | 89 MB |
| WER | `openai/whisper-large-v3` | 3.0 GB |

마지막 줄이 `ALL CHECKS PASSED` 여야 합니다.
`HF_HOME 이 영속 저장소(vFolder) 밖입니다` 경고는 **H200(클라우드) 전용 경고라 A6000 에서는 무시**해도 됩니다.

### ⑦ 설치 검증 → [§4](#4-설치-검증)

```bash
python requirements/a6000/verify_a6000_env.py --full
```

마지막 줄이 `✅ 설치가 올바릅니다.` 이면 성공입니다.

### ⑧ 환경 박제 (권장)

```bash
pip freeze --all > requirements/a6000/a6000_pip_freeze_$(date +%Y%m%d).txt
```

`--all` 은 `setuptools` 를 스냅샷에 남기기 위해 필요합니다. 이 파일은 **진단용이며 그대로 설치할 수 없습니다**
(editable `tse-eval` 등) — 재설치는 항상 `requirements_a6000.txt` 로 하세요.
커밋하려는 의도가 아니면 `git add` 할 때 주의하세요.

---

## 4. 설치 검증

### 4.1 검증 스크립트

```bash
conda activate tseeval
cd <TSE_Eval repo 경로>
export HF_HOME=<모델 캐시 경로>
python requirements/a6000/verify_a6000_env.py --full
```

| 옵션 | 용도 |
|---|---|
| (없음) | 1~5절 전체 (모델 캐시는 **파일 존재만** 확인) |
| `--full` | + ECAPA·Whisper 를 **캐시만으로(`HF_HUB_OFFLINE=1`)** 실제 로드해 `spk_sim`·`wer` 계산 — 설치 직후 권장 |
| `--skip-models` | ⑥ 모델 다운로드 **전**에 패키지·GPU 만 확인 |
| `--skip-gpu` | GPU 없이 패키지·CPU 지표만 확인 (DNSMOS 는 CPU 로) |
| `--allow-other-gpu` | GPU 모델이 A6000 이 아니어도 FAIL 대신 WARN (스크립트 자체 점검용) |

환경변수 `VERIFY_DEBUG=1` 을 주면 실패 항목의 전체 traceback 을 출력합니다.
**종료코드**: FAIL 이 없으면 `0`, 하나라도 있으면 `1`.

### 4.2 무엇을 검사하나

| 영역 | 검사 항목 | 실패 시 의미 |
|---|---|---|
| **1. Python·env** | Python 3.10.x · conda env 인지 (base 가 아닌지) | 시스템 python·base env 에 설치 |
| **2. GPU·PyTorch** | `torch/torchaudio == 2.5.1+cu121`, CUDA 12.1 · 드라이버 ≥ 525 · arch_list 에 **sm_86** · capability **(8,6)** · bf16 연산 · CUDA resample | 잘못된 torch 빌드, 드라이버 부족, 다른 GPU |
| **3. 패키지** | [`requirements_a6000.txt`](./requirements_a6000.txt) 를 **직접 읽어** 23개 패키지 대조 · **onnxruntime 배포본 1개** · numpy < 2 · setuptools < 81 · `pip check` · `librosa`/`speechmos` import · editable `tse-eval` · asteroid 백엔드 | 핀 이탈 · DNSMOS 전부 `nan` · 래퍼가 첫 행에서 멈춤 |
| **4. 지표** | `configs/config.yaml` 로드 · SI-SDR native ≡ asteroid(< 1e-6) · SI-SDR/STOI/ESTOI/PESQ 유한값 · **DNSMOS 가 실제로 `CUDAExecutionProvider` 로 도는지** (별도 프로세스) | DNSMOS 가 조용히 CPU 로 폴백 |
| **5. 모델 캐시·CLI** | `HF_HOME` · `embedding_model.ckpt` · `model.safetensors` · `.incomplete` 없음 · `python -m tse_eval --help` | 채점 도중 3 GB 재다운로드 |
| **6. `--full`** | ECAPA self-cosine ≈ 1 · Whisper bf16 전사 | 모델 로드 실패 |

> 패키지 핀 목록을 스크립트에 따로 적지 않고 requirements 파일을 읽으므로, **requirements 를 고치면 검증도 자동으로 따라갑니다.**
> 스크립트 본문은 [H200 판](../h200/verify_h200_env.py)과 동일하고 맨 위 `TARGET` 블록만 다릅니다.

### 4.3 정상 출력 모습

A6000 에서는 모든 항목이 PASS 여야 합니다. H200 과 다르게 보여야 하는 줄만 표시합니다.

```text
TSE_Eval 설치 검증 — RTX A6000 (Ampere)  (repo: <repo 경로>)
  ...
  [PASS] nvidia-smi                         NVIDIA RTX A6000 · driver 5xx.xx · 49140 MiB
  [PASS] CUDA arch (sm_86)                  NVIDIA RTX A6000 · capability (8, 6) · arch_list 에 sm_86 포함
  ...
  (대조 기준: requirements/a6000/requirements_a6000.txt · 23개 패키지)
  ...
  [PASS] DNSMOS 값 · 실제 EP                   sig 1.39 · bak 2.70 · ovrl 1.19 · p808 2.30 · EP ['CUDAExecutionProvider', 'CPUExecutionProvider']
  ...
════════════════════════════════════════════════════════════════
  결과: PASS 48 · WARN 0 · FAIL 0 · SKIP 0
  ✅ 설치가 올바릅니다.
════════════════════════════════════════════════════════════════
```

A6000 이 아닌 서버에서 실행하면 아래처럼 **GPU 판별 한 건만 FAIL** 이 나오는 것이 정상입니다.

```text
  [FAIL] CUDA arch (sm_86)                  NVIDIA H200 capability (9, 0) ≠ 기대 (8, 6) — 이 서버는 RTX A6000 (Ampere) 이 아닙니다
```

### 4.4 채점 래퍼를 쓸 때

`scripts/eval_llmtse.sh` · `scripts/eval_styletse.sh` 의 경로 기본값은 **H200 서버 기준**입니다. A6000 에서는 덮어쓰세요.

```bash
export HF_HOME=<모델 캐시 경로>                      # 기본값: /home/work/my-checkpoints/hf_cache
LLMTSE_ROOT=<llmtse 추론 산출물 루트> bash scripts/eval_llmtse.sh --check-only
STYLETSE_ROOT=<styletse 추론 산출물 루트> bash scripts/eval_styletse.sh --check-only
```

`--check-only` 는 선행 검사(패키지 · asteroid 백엔드 · 모델 캐시 · GPU 여유 8 GB · 입력 CSV)만 하고 끝납니다.
래퍼 없이 직접 채점하려면 `python -m tse_eval -i <manifest>.csv -o <out>.csv` ([USE_GUIDE](../../USE_GUIDE.md)).

### 4.5 테스트 스위트 · 합성 예제

```bash
pytest -q                                             # 2026-09-14 기준 105 passed, 1 skipped
python examples/make_example.py
python -m tse_eval -i examples/sample_input.csv -o examples/results.csv
```

`dnsmos_*` 4개 열이 전부 `nan` 이 아니어야 하고, 콘솔에
`[tse-eval] DNSMOS ONNX providers: ['CUDAExecutionProvider', 'CPUExecutionProvider'] (intra_op=4)` 가 찍혀야 합니다.

---

## 5. 문제 해결

### 5.1 `dnsmos_*` 4열이 전부 `nan`

`speechmos` 는 의존성을 선언하지 않는데 내부에서 `librosa` 를 import 하고, 지표 함수는 예외를 `nan` 으로 삼킵니다.

```bash
pip install librosa==0.11.0
```

### 5.2 DNSMOS 가 CPU 로 돎 / CUDA EP 로드 실패

검증 스크립트의 `DNSMOS 값 · 실제 EP` 가 FAIL 이거나, 채점 sidecar JSON 의 `dnsmos.actual_providers` 에
`CUDAExecutionProvider` 가 없을 때입니다. 먼저 스크립트 출력의 경고 줄을 보세요.

| 경고 | 원인 | 대응 |
|---|---|---|
| `Failed to load library libonnxruntime_providers_cuda.so` + `libcudnn*.so.9: cannot open shared object file` | onnxruntime-gpu 1.20 은 CUDA 12 + **cuDNN 9** 가 필요한데 시스템 cuDNN 이 없고 torch 번들 cuDNN 을 찾지 못함 | 아래 `LD_LIBRARY_PATH` 설정 |
| 같은 메시지 + `undefined symbol ... libcudnn_graph.so.9` | 서로 다른 버전의 cuDNN 이 섞임 (시스템 cuDNN 이 따로 있는 서버에서, torch 가 cuDNN 을 **먼저** 올린 프로세스) | `tse_eval` 은 `ort_setup.prime_dnsmos_session()` 으로 ONNX 세션을 먼저 만들어 해당 없음. 직접 짠 코드라면 아래 `LD_LIBRARY_PATH` 설정으로도 해소 |
| `onnxruntime-gpu` 버전이 1.19 미만 | 1.18.x 는 cuDNN 8 빌드 | `pip install "onnxruntime-gpu>=1.19.2,<1.21"` (1.18 로 내리지 말 것) |

torch 번들 cuDNN/cuBLAS 를 onnxruntime 이 찾도록 하는 방법 (세션마다, 또는 `~/.bashrc`).
H200 에서 이 설정을 넣으면 onnxruntime 이 torch 번들 cuDNN 9.1 만 쓰게 되고(`/proc/self/maps` 확인),
torch 를 먼저 쓴 프로세스에서도 CUDA EP 가 붙는 것까지 확인했습니다. **A6000 실기에서는 미검증**입니다.

```bash
NV=$(python -c "import os, nvidia.cudnn; print(os.path.dirname(list(nvidia.cudnn.__path__)[0]))")
export LD_LIBRARY_PATH="$NV/cudnn/lib:$NV/cublas/lib:${LD_LIBRARY_PATH:-}"
python requirements/a6000/verify_a6000_env.py        # DNSMOS EP 가 CUDA 로 바뀌었는지 확인
```

CUDA 를 끝내 쓸 수 없으면 `--dnsmos-providers cpu` 로 채점할 수 있습니다(약 3배 느림).
단 DNSMOS 는 CPU/CUDA 간 최대 약 `3e-3` 차이가 나므로 **비교표의 모든 시스템을 같은 providers 로** 채점하세요.

### 5.3 arch_list 에 `sm_86` 이 없음 / `CUDA error: no kernel image`

cu121 휠이 아닌 torch 가 깔렸습니다.

```bash
pip install --force-reinstall torch==2.5.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
```

### 5.4 `pip list` 에 onnxruntime 이 두 줄

CPU `onnxruntime` 과 `onnxruntime-gpu` 가 공존해 같은 모듈 이름을 덮어씁니다(`.[cpu]` extra 를 깔았을 때).

```bash
pip uninstall -y onnxruntime onnxruntime-gpu
pip install -r requirements/a6000/requirements_a6000.txt
pip install -e '.[test]'
```

### 5.5 `pesq` 빌드 실패

```bash
gcc --version                        # 없으면 시스템에 gcc 설치 필요
pip install "numpy==1.26.4"          # pesq 빌드는 numpy 헤더를 씀 — numpy 를 먼저
pip install pesq==0.0.4
```

### 5.6 채점 도중 Whisper 를 다시 받음

셸마다 `HF_HOME` 이 다릅니다([③-⑤](#⑤-hf_home-지정--필수)). `echo $HF_HOME` 으로 확인하고, 검증 스크립트의
`5. 모델 캐시` 절이 PASS 인지 보세요.

### 5.7 `ModuleNotFoundError: No module named 'pkg_resources'`

setuptools 81+ 에서 `pkg_resources` 가 제거됐는데 asteroid 0.7.0 → torchmetrics 0.11.4 가 그것을 import 합니다.

```bash
pip install "setuptools<81"
```

---

## 6. 하지 말아야 할 것

| ❌ 금지 | 이유 |
|---|---|
| `pip install -e '.[cpu,test]'` (GPU 서버에서) | CPU `onnxruntime` 이 `onnxruntime-gpu` 를 가립니다(§5.4) |
| `onnxruntime-gpu` 를 1.18.x 로 내리기 | cuDNN 8 빌드라 torch cu121 의 cuDNN 9 와 맞지 않습니다 |
| torch 를 cu124/cu126 으로 올리기 | H200·형제 프로젝트와 바이너리가 갈려 채점 재현성을 잃습니다 (TSE_Eval 은 torch 를 resample·모델 추론에만 씀) |
| `pip install -U setuptools` | §5.7 문제가 재발합니다 |
| numpy 2.x 로 업그레이드 | torch 2.5.1 ABI 와 충돌 |
| requirements 에서 `librosa` 삭제 | DNSMOS 가 에러 없이 전부 `nan` (§5.1) |
| `HF_HOME` 없이 모델 다운로드 | 받은 곳과 읽는 곳이 어긋납니다(§5.6) |

---

📎 **관련 문서**: [H200 설치 가이드](../h200/INSTALL.md) · [USE_GUIDE](../../USE_GUIDE.md) · [CAVEATS](../../CAVEATS.md) · [프로젝트 README](../../README.md)
