# 🛠️ TSE_Eval 설치 가이드 (NVIDIA H100 / H200)

> **이 문서 하나만 위에서 아래로 따라 하면 끝납니다.**
> 처음 보는 분을 기준으로 썼습니다. 명령마다 **소요 시간**, **정상 출력**, **자주 나는 문제**를 같이 적었습니다.

<div align="center">

| | |
|---|---|
| 🖥️ **대상 환경** | NVIDIA H100 / H200 (Hopper, sm_90) · Ubuntu 22.04 · Python 3.10.20 |
| ⏱️ **총 소요 시간** | 약 **6분** (+ 모델 다운로드 3.1 GB) |
| 💾 **필요 디스크** | env 약 7 GB + 모델 3.0 GB + pip 캐시 3.2 GB ≈ **13 GB** |
| ✅ **검증 상태** | 2026-08-06 이 머신에서 **처음부터 완주 재검증** (110 패키지 · `pip check` 클린 · 테스트 83 passed · `ALL CHECKS PASSED`) |

</div>

---

## 📑 목차

1. [30초 요약 — 복붙용 전체 블록](#1-30초-요약--복붙용-전체-블록)
2. [시작 전에 꼭 알아야 할 3가지](#2-시작-전에-꼭-알아야-할-3가지)
3. [설치 8단계 (⓪ ~ ⑦)](#3-설치-8단계)
4. [설치가 잘 됐는지 확인하기](#4-설치가-잘-됐는지-확인하기)
5. [두 번째 세션부터는 2줄이면 됩니다](#5-두-번째-세션부터는-2줄이면-됩니다)
6. [🚨 문제 해결 (증상 → 원인 → 해결)](#6--문제-해결)
7. [부록 — 파일 역할 / 지표별 샘플레이트](#7-부록)

---

## 1. 30초 요약 — 복붙용 전체 블록

<details>
<summary><b>▶ 이미 익숙하신 분은 이 블록만 복붙하세요</b> (클릭해서 펼치기)</summary>

```bash
# ⓪ 청소
source /home/work/my-code/miniforge3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null
conda env remove -n tseeval -y
cd /home/work/my-code/TSE_Eval && rm -rf tse_eval.egg-info build

# ① env 생성
conda create -n tseeval -y python=3.10.20
conda activate tseeval
cd /home/work/my-code/TSE_Eval

# ② PyTorch 먼저 단독
pip install --cache-dir /home/work/my-code/pip_cache \
    torch==2.5.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# ③ 나머지 패키지
pip install --cache-dir /home/work/my-code/pip_cache -r requirements/h200/requirements_h200.txt

# ④ 프로젝트 자체 + 테스트 도구(pytest, asteroid)
pip install --cache-dir /home/work/my-code/pip_cache -e '.[test]'

# ⑤⑥ 모델 다운로드
export HF_HOME=/home/work/my-checkpoints/hf_cache
python scripts/download_models.py

# ⑦ 박제
pip freeze --all > requirements/h200/h200_pip_freeze_$(date +%Y%m%d).txt
```

</details>

---

## 2. 시작 전에 꼭 알아야 할 3가지

이 3가지를 모르면 **반드시 시간을 낭비합니다.** 실제로 겪은 것만 적었습니다.

### 🔴 ① `/home/work` 홈은 세션이 끝나면 사라집니다

```
$HOME = /home/work   →  /dev/loop6, 49 GB  ★ 휘발 (세션 종료 시 삭제)
```

**영속 저장소(vFolder)** 는 아래 4곳뿐입니다. 무거운 것은 전부 여기 둬야 합니다.

| 경로 | 용도 |
|---|---|
| `/home/work/my-code` | 코드 + **conda env** (miniforge3 가 여기 있어 `-n` 으로 만들어도 영속) |
| `/home/work/my-checkpoints` | 체크포인트 + **HuggingFace 모델 캐시** |
| `/home/work/my-datasets` | 데이터셋 (PORTE-v3) |
| `/home/work/my-outputs` | 추론 결과 |

> 💡 **왜 중요한가**: HuggingFace 기본 캐시가 `~/.cache/huggingface`(= 휘발 영역)입니다.
> 그대로 두면 **Whisper 3 GB 를 세션마다 다시 받습니다.** 그래서 ⑤단계에서 `HF_HOME` 을 옮깁니다.

### 🟠 ② pip 캐시가 기본적으로 꺼져 있습니다

NGC 이미지의 `/etc/pip.conf` 가 `no-cache-dir = true` 를 걸어놨습니다.
→ 설치가 한 번 실패해서 다시 하면 **약 4 GB 를 처음부터 또 받습니다.**

되살리는 방법은 **CLI 플래그 하나뿐**입니다:

```bash
pip install --cache-dir /home/work/my-code/pip_cache ...
```

> ⚠️ `PIP_CACHE_DIR` / `PIP_NO_CACHE_DIR` **환경변수는 안 먹힙니다** (직접 확인).
> 그래서 이 문서의 모든 `pip install` 에 `--cache-dir` 이 붙어 있습니다. 지우지 마세요.

### 🟡 ③ 설치 중에 다른 터미널에서 conda 를 만지지 마세요

같은 env 에 `conda env remove` 와 설치가 겹치면 **env 디렉토리가 도중에 사라집니다.**
그러면 아래처럼 원인과 전혀 상관없어 보이는 에러가 납니다:

```
ModuleNotFoundError: No module named 'urllib'
```

（pip 이 쓰던 Python 표준 라이브러리가 통째로 사라져서 생기는 증상입니다.）

---

## 3. 설치 8단계

> 📍 모든 명령은 `/home/work/my-code/TSE_Eval` 에서 실행합니다.

### ⓪ 청소하기 &nbsp;·&nbsp; `~0초` &nbsp;·&nbsp; **재설치면 필수**

반쯤 지워진 env 껍데기와 예전 빌드 잔여물을 없앱니다.

```bash
source /home/work/my-code/miniforge3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null
conda env remove -n tseeval -y

# 정말 지워졌는지 확인 — "No such file" 이어야 정상입니다
ls -d /home/work/my-code/miniforge3/envs/tseeval

# repo 의 이전 editable 설치 잔여물도 제거
cd /home/work/my-code/TSE_Eval
rm -rf tse_eval.egg-info build
```

> ⚠️ **`rm -rf tse_eval.egg-info` 를 건너뛰면** ③단계에서 아래 에러가 뜹니다.
> 새 env 인데도 pip 이 repo 의 `egg-info` 를 "이미 설치된 패키지"로 오인해서 나오는 것으로, **무해하지만 실패로 착각하게 됩니다.**
> ```
> ERROR: ... tse-eval 0.1.0 requires onnxruntime, which is not installed.
> ```

---

### ① conda 환경 만들기 &nbsp;·&nbsp; `22초`

```bash
conda create -n tseeval -y python=3.10.20
conda activate tseeval
cd /home/work/my-code/TSE_Eval
```

<details>
<summary>정상 출력</summary>

```
Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use
#     $ conda activate tseeval
```
</details>

> 💡 `-n`(이름) 으로 만들어도 영속입니다. miniforge3 자체가 `/home/work/my-code` 안에 있어서
> env 가 `/home/work/my-code/miniforge3/envs/tseeval` 에 저장되기 때문입니다.
> 판단 기준은 **"`-n` 이냐 `-p` 냐"가 아니라 "경로가 vFolder 안이냐"** 입니다.
> 확인법 → `conda env list` 에서 경로가 `/home/work/my-code/...` 로 시작하면 안전.

---

### ② PyTorch 먼저, 반드시 단독으로 &nbsp;·&nbsp; `60초` &nbsp;·&nbsp; 약 2.5 GB

```bash
pip install --cache-dir /home/work/my-code/pip_cache \
    torch==2.5.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
```

**왜 먼저 따로 깔까요?** `speechbrain` 등이 `torch` 를 요구하기 때문에, 먼저 깔아두지 않으면
pip 이 PyPI 기본 torch(다른 CUDA 빌드)를 끌어와 스택이 섞일 수 있습니다.

<details>
<summary>정상 출력 / 무시해도 되는 경고</summary>

```
Successfully installed ... torch-2.5.1+cu121 torchaudio-2.5.1+cu121
```

아래 경고는 **정상**입니다. numpy 는 다음 단계에서 깔립니다.
```
UserWarning: Failed to initialize NumPy: No module named 'numpy'
```
</details>

> 🤔 **왜 cu121 인가요? H200 인데 최신이 아니어도 되나요?**
> 됩니다. `torch 2.5.1+cu121` 휠에는 **sm_90(H100/H200) 커널이 포함**되어 있고,
> 드라이버 580 은 CUDA 12.1 앱을 그대로 실행합니다.
> 게다가 형제 프로젝트(TPEX / LLM-TSE / StyleTSE) 3개가 모두 같은 빌드를 써서 **재현성이 최대**가 됩니다.

---

### ③ 나머지 패키지 &nbsp;·&nbsp; `3분 28초`

```bash
pip install --cache-dir /home/work/my-code/pip_cache -r requirements/h200/requirements_h200.txt
```

여기서 **`pesq` 가 C 확장을 컴파일**합니다 (gcc 필요, 이 머신은 gcc 11.4 있음).

<details>
<summary>정상 출력</summary>

```
Building wheel for pesq (pyproject.toml): started
Building wheel for pesq (pyproject.toml): finished with status 'done'
Successfully built pesq
Successfully installed ... librosa-0.11.0 numpy-1.26.4 onnxruntime-gpu-1.20.2
  speechbrain-1.0.3 transformers-4.46.3 ... (총 63개)
```
</details>

> ✅ 이 단계에서 **총 85개 패키지**가 갖춰집니다 (②의 torch 포함).

---

### ④ 프로젝트 + 테스트 도구 설치 &nbsp;·&nbsp; `33초`

```bash
pip install --cache-dir /home/work/my-code/pip_cache -e '.[test]'
```

이렇게 하면 어느 디렉토리에서든 `pytest` 와 `tse-eval` 명령이 동작합니다.
`[test]` extra 는 **pytest + asteroid** 를 함께 깝니다 — asteroid 는 SI-SDR 값이
표준 구현과 일치하는지 검증하는 **오라클**로만 쓰이고, 실제 채점은 더 빠른 `native`
백엔드가 담당합니다. 없이 설치하려면 `-e .` 만 쓰면 되고, 그 경우 등가성 테스트 2건이 skip 됩니다.

> 💡 **CPU용 `onnxruntime` 은 일부러 기본 의존성이 아닙니다.**
> CPU/GPU 빌드는 별개 배포본이라 둘이 같이 깔리면 충돌합니다. ③에서 깐
> `onnxruntime-gpu` 가 그대로 유지되도록 extra 로 분리해 뒀습니다
> (예전에는 `--no-deps` 로 우회했고 `pip check` 경고가 계속 떴습니다).
> H200 아닌 환경은 `pip install -e '.[cpu,test]'` 를 쓰세요.

**확인** — 한 줄만 나와야 하고, `pip check` 가 깨끗해야 합니다:

```bash
pip list | grep onnxruntime
#  onnxruntime-gpu          1.20.2      ← 이것만 있어야 정상
pip check
#  No broken requirements found.
```

---

### ⑤ HuggingFace 캐시를 영속 저장소로 &nbsp;·&nbsp; `즉시` &nbsp;·&nbsp; **세션마다 필요**

```bash
export HF_HOME=/home/work/my-checkpoints/hf_cache
```

안 하면 다음 세션에 **Whisper 3 GB 를 다시 받습니다** ([2-①](#-①-homework-홈은-세션이-끝나면-사라집니다) 참고).

> 💡 잊어도 안전합니다. `scripts/download_models.py` 가 `HF_HOME` 이 없으면 이 경로를
> 자동으로 쓰고, vFolder 밖으로 설정돼 있으면 크게 경고합니다.

---

### ⑥ 모델 2개 다운로드 + 검증 &nbsp;·&nbsp; `35초` &nbsp;·&nbsp; 약 3.1 GB

```bash
python scripts/download_models.py
```

| 용도 | 모델 | 크기 |
|---|---|---|
| **Speaker Similarity** | `speechbrain/spkrec-ecapa-voxceleb` (ECAPA-TDNN, 192차원) | 89 MB |
| **WER** | `openai/whisper-large-v3` | 3.0 GB |

> `whisper-large-v3` 는 LLM-TSE 의 `eval.py` 와 **같은 모델**이라 WER 숫자를 바로 비교할 수 있습니다.
> DNSMOS 는 `speechmos` 패키지에 ONNX 가 들어 있어 다운로드가 없습니다.
>
> 기본 지표 세트에는 **`spk_sim`(ECAPA) 이 포함**되어 있으므로 ECAPA 는 사실상 필수입니다.
> `wer`(Whisper) 는 기본에서 빠져 있지만(15,000 utt 기준 178분), 논문 표를 만들 때 켜므로
> 미리 받아 두는 편이 낫습니다.

**옵션**

| 옵션 | 설명 |
|---|---|
| `--only spk` | ECAPA 만 (89 MB) |
| `--only wer` | Whisper 만 (3.0 GB) |
| `--no-verify` | 다운로드만, 검증 생략 |
| `--device cpu` | 검증을 CPU 로 |

<details>
<summary>정상 출력 (마지막 줄이 <code>ALL CHECKS PASSED</code>)</summary>

```
  ── 검증: ECAPA ──
  [OK]   16 kHz 임베딩 shape (1, 192)
  [OK]   self cosine similarity = 1.000000
  [INFO] 같은 소리의 24 kHz 판 → 예외 없이 통과, cos(16k, 24k) = 0.8834
  [OK]   ★ 24 kHz 입력이 조용히 다른 임베딩을 만드는 것 확인
  [OK]   24k→16k 리샘플 후 cos = 0.9909  (리샘플 없이 0.8834 보다 높음)

  ── 검증: Whisper ──
  [OK]   16 kHz 전사 성공 — mel (1, 128, 3000)
  [OK]   ★ 24 kHz 입력을 ValueError 로 거부 확인
  [OK]   jiwer.wer 동작 확인 = 0.5

  PASS  ECAPA 다운로드 / ECAPA 검증 / Whisper 다운로드 / Whisper 검증
  HF_HOME 총 사용량         : 3182 MB

ALL CHECKS PASSED
```
</details>

> 🔁 **다시 실행해도 안전합니다.** 두 번째부터는 캐시 히트로 12초에 끝납니다.

---

### ⑦ 환경 박제 &nbsp;·&nbsp; `즉시`

```bash
pip freeze --all > requirements/h200/h200_pip_freeze_$(date +%Y%m%d).txt
```

> ⚠️ **`--all` 을 꼭 붙이세요.** 기본 `pip freeze` 는 `setuptools` 를 빼버리는데,
> 이 프로젝트에서 `setuptools<81` 은 중요한 핀이라 스냅샷에 보여야 합니다.
>
> ⚠️ **이 파일은 진단용이고, 그대로 설치할 수 없습니다.** (직접 확인)
> - conda 가 제공한 `packaging` 이 `@ file:///home/conda/...` 로 기록되는데 그 경로가 없어 `OSError` 로 실패
> - editable 설치인 `tse-eval` 이 `tse-eval==0.1.0` 으로 기록되는데 PyPI 에 없는 이름
>
> → **재설치는 항상 `requirements/h200/requirements_h200.txt` 로** 하세요. 스냅샷은 "그때 뭐가 깔려 있었나" 대조용입니다.

---

## 4. 설치가 잘 됐는지 확인하기

### ✅ 검증 1 — 스택 확인 (가장 중요)

```bash
python - <<'PY'
import torch, torchaudio, numpy, onnxruntime as ort
from importlib.metadata import version
print("torch      :", torch.__version__, "| cuda", torch.version.cuda)
print("arch_list  :", torch.cuda.get_arch_list())        # ← sm_90 이 있어야 함
print("gpu        :", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print("ort avail  :", ort.get_available_providers())     # ← CUDAExecutionProvider 필수
import speechmos, speechbrain, pesq, pystoi, librosa, transformers, accelerate, jiwer
print("speechbrain:", speechbrain.__version__, "| librosa", librosa.__version__)
print("transformers:", transformers.__version__, "| accelerate", accelerate.__version__)
print("jiwer      :", version("jiwer"))                  # jiwer 엔 __version__ 이 없음
print("numpy      :", numpy.__version__, "| setuptools", version("setuptools"))
print("OK")
PY
```

**기대 출력**

```
torch      : 2.5.1+cu121 | cuda 12.1
arch_list  : ['sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']
gpu        : NVIDIA H200 (9, 0)
ort avail  : ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
speechbrain: 1.0.3 | librosa 0.11.0
transformers: 4.46.3 | accelerate 1.10.0
jiwer      : 3.1.0
numpy      : 1.26.4 | setuptools 80.10.2
OK
```

| 확인할 것 | 왜 |
|---|---|
| `arch_list` 에 **`sm_90`** | 없으면 cu121 휠이 아닌 게 깔린 것 → ②를 `--force-reinstall` 로 다시 |
| **`CUDAExecutionProvider`** | 없으면 DNSMOS 가 CPU 로 돕니다 (아래 검증 2) |
| **`librosa`** 임포트 성공 | 없으면 DNSMOS 4열이 **조용히 전부 `nan`** 이 됩니다 |
| `setuptools` **81 미만** | 형제 repo 와 동일 계열 유지 |

### ✅ 검증 2 — 테스트 스위트

```bash
cd /home/work/my-code/TSE_Eval && pytest -q
```

```
35 passed, 2 warnings in 22s
```

> ❗ `dnsmos` 관련 2개가 실패하면 **`librosa` 누락**입니다 → `pip install librosa==0.11.0`

### ✅ 검증 3 — 합성 예제로 처음부터 끝까지

```bash
python examples/make_example.py
python -m tse_eval -i examples/sample_input.csv -o examples/results.csv
```

`dnsmos_*` 4개 열이 **전부 `nan` 이 아니어야** 합니다.

### ✅ 검증 4 — DNSMOS 가 실제로 GPU 를 쓰는지

`onnxruntime-gpu` 를 깔았어도 **DNSMOS 는 지금 CPU 로 돕니다.** 직접 확인해 보세요.

```bash
python - <<'PY'
import numpy as np, onnxruntime as ort
orig = ort.InferenceSession
def probe(*a, **k):
    s = orig(*a, **k); print("EP in use:", s.get_providers()); return s
ort.InferenceSession = probe
from speechmos import dnsmos
print(dnsmos.run((np.random.randn(16000)*0.05).astype(np.float32), sr=16000))
PY
```

**기대 출력** — `CPUExecutionProvider` 가 나오는 것이 **정상입니다**:

```
EP in use: ['CPUExecutionProvider']
EP in use: ['CPUExecutionProvider']
{'ovrl_mos': 1.10..., 'sig_mos': 1.21..., 'bak_mos': 1.18..., 'p808_mos': 2.09...}
```

> 🤔 **왜 GPU 를 안 쓰나요? 고장인가요?** 아닙니다.
> `speechmos` 가 세션을 만들 때 `providers` 를 넘기지 않는데,
> onnxruntime 1.9+ 는 **`providers` 미지정 시 CPU EP 만** 씁니다. 즉 CUDA 를 **시도조차 하지 않습니다.**
> CUDA 자체는 정상입니다 — 명시하면 바로 붙습니다:
> ```
> ort.InferenceSession(model)                           → ['CPUExecutionProvider']
> ort.InferenceSession(model, providers=['CUDA...'])    → ['CUDAExecutionProvider', 'CPU...']
> ```
> 그래서 지금 `onnxruntime-gpu` 핀은 **"나중을 위한 준비"** 이고, 실제 가속에는 코드 수정이
> 필요합니다 → [§6 DNSMOS 속도](#dnsmos-속도-실측) 와 [§7 후속 작업](#-후속-작업-아직-미구현) 참고.

---

## 5. 두 번째 세션부터는 2줄이면 됩니다

env 와 모델이 전부 영속 저장소에 있으므로 재설치는 필요 없습니다.

```bash
source /home/work/my-code/miniforge3/etc/profile.d/conda.sh && conda activate tseeval
export HF_HOME=/home/work/my-checkpoints/hf_cache
```

이 2줄이면 끝입니다. 환경변수를 더 설정할 필요는 없습니다.

> 💡 대량 평가가 느리게 느껴지면 [§6 DNSMOS 속도](#dnsmos-속도-실측) 를 보세요.
> (`OMP_NUM_THREADS` 를 만지는 것은 **효과가 없습니다** — 실측으로 확인했습니다.)

---

## 6. 🚨 문제 해결

| 증상 | 원인 | 해결 |
|---|---|---|
| `ModuleNotFoundError: No module named 'urllib'` | 설치 중 다른 터미널에서 conda 명령이 겹쳐 **env 가 사라짐** | ⓪부터 다시. 설치 중엔 다른 터미널에서 conda 를 만지지 마세요 |
| `ERROR: ... tse-eval 0.1.0 requires onnxruntime` (③단계) | repo 의 **`tse_eval.egg-info` 잔여물** 을 pip 이 오인 | `rm -rf tse_eval.egg-info build` 후 재시도. 무해하니 무시해도 됩니다 |
| `pip check` 가 같은 메시지를 계속 냄 (설치 완료 후) | `pyproject.toml` 이 CPU `onnxruntime` 을 선언하는데 우리는 `onnxruntime-gpu` 사용 | **정상입니다.** 이 환경에선 항상 뜨는 경고 한 줄 |
| `dnsmos_*` 4열이 전부 `nan` | **`librosa` 누락.** `speechmos` 가 의존성을 선언하지 않는데 내부에서 `librosa` 를 import 하고, 지표 함수가 예외를 `nan` 으로 삼킴 | `pip install librosa==0.11.0` |
| `arch_list` 에 `sm_90` 없음 | cu121 아닌 torch 가 깔림 | `pip install --force-reinstall torch==2.5.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121` |
| DNSMOS 가 느림 | 가속이 안 걸렸을 수 있음 | sidecar 의 `dnsmos.actual_providers` 확인. `CUDAExecutionProvider` 가 없으면 아래 [DNSMOS 속도](#dnsmos-속도-실측) 참고 |
| 실행이 40초쯤 멈춘 것처럼 보임 | CUDA 세션 1회성 초기화 | 정상입니다. `initialising DNSMOS on CUDA (~40 s, one time)` 로그가 나옵니다 |
| 빨간 `pthread_setaffinity_np failed` 줄이 쏟아짐 | 스레드 제한이 안 걸린 상태 (`--dnsmos-threads 0`) | 기본값(4)이면 나오지 않습니다. 굳이 0 을 줬다면 무해하지만 2.7배 느립니다 |
| 세 시스템의 DNSMOS 값이 미묘하게 안 맞음 | 일부는 CPU, 일부는 CUDA 로 채점됨 | 각 sidecar 의 `dnsmos.actual_providers` 를 비교하고 같은 설정으로 재채점 (차이 약 3e-3) |
| 재설치가 너무 느림 (매번 4 GB) | pip 캐시가 꺼져 있음 | 모든 `pip install` 에 `--cache-dir /home/work/my-code/pip_cache` |
| 다음 세션에 Whisper 를 또 받음 | `HF_HOME` 미설정 → 휘발 영역에 캐시됨 | `export HF_HOME=/home/work/my-checkpoints/hf_cache` |
| `pip install -r h200_pip_freeze_*.txt` 가 `OSError` | 스냅샷은 설치용이 아님 | `requirements/h200/requirements_h200.txt` 로 설치하세요 |
| Whisper 다운로드가 24 GB 를 받으려 함 | `snapshot_download` 를 직접 호출함 (flax/tf/fp32 샤드 포함) | `scripts/download_models.py` 를 쓰세요 (`from_pretrained` 방식, 3.0 GB) |

### DNSMOS 속도 (실측)

평가 파이프라인에서 **가장 느린 구간은 DNSMOS** 였습니다. **이제 기본으로 가속됩니다** —
`tse_eval/ort_setup.py` 가 onnxruntime 세션에 스레드 수와 providers 를 주입합니다.

실측 (2026-08-06, 실제 PORTE-v3 오디오 · 평균 11초 · 길이 전부 다름).
측정법은 **60행 실행 − 20행 실행 의 한계비용 ÷ 40** 으로, CUDA 의 고정 초기화 비용을 상쇄합니다:

| 설정 | 한계비용 | 15,000 utt | 무가속 대비 |
|---|--:|--:|--:|
| 무가속 (`--dnsmos-threads 0 --dnsmos-providers cpu`) | 1475 ms/utt | 369분 | 1.0배 |
| `--dnsmos-providers cpu` (intra_op=4) | 550 ms/utt | 137분 | 2.7배 |
| **기본값 (intra_op=4 + CUDA)** | **175 ms/utt** | **44분** | **8.4배** |

- CUDA 는 **1회성 초기화 약 40초**를 추가로 씁니다(5,000행 기준 0.5% 수준). 실행 중
  `[tse-eval] initialising DNSMOS on CUDA (~40 s, one time)…` 이 찍히면 멈춘 게 아닙니다.
- 어떤 EP 로 돌았는지는 결과 옆 **sidecar JSON 의 `dnsmos.actual_providers`** 에 기록됩니다.

> ⚠️ **`OMP_NUM_THREADS` 로는 해결되지 않습니다.**
> onnxruntime 의 intra-op 스레드 풀은 OpenMP 가 아니라 **자체 풀**이라서
> `SessionOptions.intra_op_num_threads` 만 효과가 있습니다(실측으로 확인).

> ⚠️ **CUDA 와 CPU 의 DNSMOS 값은 완전히 같지 않습니다** — 최대 약 `3e-3` 차이(부동소수점
> 커널 차이). CPU 설정끼리는 `4e-7` 로 사실상 동일합니다.
> DNSMOS 는 보통 소수점 2자리로 보고하므로 보고 정밀도보다 훨씬 작지만,
> **비교표에 들어가는 세 시스템은 반드시 같은 providers 로 채점해야 합니다.**
> sidecar 에 값이 남으므로 사후에 불일치를 검출할 수 있습니다.

### 전체 소요 시간 감각 (15,000 utt = 5,000행 × 3시스템)

| 지표 | 15,000 utt |
|---|--:|
| SI-SDR 계열 | 2분 |
| STOI + ESTOI | 28분 |
| PESQ | 51분 |
| DNSMOS (CUDA 기본) | 44분 |
| spk_sim (기본 포함) | 20분 |
| **기본 세트 합계** | **약 145분** |
| WER (기본 off, `--metrics` 로 켜기) | +178분 |

---

## 7. 부록

### 📁 파일 역할

| 파일 | 역할 |
|---|---|
| `requirements/h200/requirements_h200.txt` | **설치의 기준.** H100/H200 용 핀 목록 + 함정 12가지 + 상세 주석 |
| `requirements/a6000/requirements_a6000.txt` | 구형(A6000 / CPU-only) 기준. H200 에서는 쓰지 않습니다 |
| `requirements/h200/h200_pip_freeze_*.txt` | **진단용 스냅샷.** 설치용 아님 (⑦ 참고) |
| `scripts/download_models.py` | ECAPA + Whisper 다운로드 & 검증 |
| `requirements/h200/INSTALL.md` | 이 문서 |
| `USE_GUIDE.md` | 설치 후 **사용법** (CLI 옵션, 컬럼 자동 인식, FAQ) |

### 🎚️ 지표별 계산 샘플레이트

작업 샘플레이트 기본값은 **24000** (`--target-sr`) 입니다. PORTE-v3 원본이 24 kHz 이기 때문입니다.

| 지표 | 계산 SR | 이유 |
|---|---|---|
| SI-SDR / SI-SDRi | **24 kHz (native)** | 샘플레이트에 무관한 정의 |
| STOI / ESTOI | **24 kHz (native)** | `pystoi` 가 내부에서 10 kHz 로 리샘플 → 미리 16k 로 낮추면 이중 리샘플만 추가 |
| PESQ | 16 kHz | ITU-T P.862 가 8/16 kHz 만 정의 (그 외 `ValueError`) |
| DNSMOS | 16 kHz | 16 kHz 학습 모델 (`speechmos` 가 다른 SR 을 거부) |
| Speaker Similarity | 16 kHz | ECAPA 가 16 kHz 모델. ⚠️ **에러 없이 조용히 틀린 값**이 나오므로 반드시 리샘플 |
| WER | 16 kHz | Whisper 가 16 kHz 고정 (그 외 `ValueError`) |

이 프로토콜은 형제 프로젝트(TPEX / LLM-TSE / StyleTSE)의 `eval.py` 와 동일하므로,
같은 오디오에 대해 **같은 숫자**가 나옵니다.

### 🚧 후속 작업 (아직 미구현)

이전 판에 적어둔 개선점 4가지는 **모두 반영되었습니다**:

| # | 항목 | 상태 |
|---|---|---|
| 1 | DNSMOS 가속 (`intra_op` + `providers`) | ✅ `tse_eval/ort_setup.py`, 기본 CUDA, 8.4배 |
| 2 | `pred_path` 자동 인식 | ✅ est 후보에 추가 — `--est-col` 불필요 |
| 3 | Speaker Similarity / WER 지표 | ✅ 구현 완료. spk_sim 은 기본 on, wer 는 opt-in |
| 4 | `pyproject.toml` 의 CPU `onnxruntime` 정리 | ✅ extra 로 분리 → `pip check` 클린 |

남은 것은 **세 프로젝트 `inference.py` 실행**뿐입니다(매니페스트가 아직 없어 실데이터
채점과 형제 `eval.py` 숫자 대조를 못 한 상태). SDR/SIR/SAR 은 의도적으로 미구현입니다.

### 🔗 다음 단계

설치가 끝났으면 **[USE_GUIDE.md](../../USE_GUIDE.md)** 로 이동하세요.
논문 숫자를 만들기 전에는 **[CAVEATS.md](../../CAVEATS.md)** 를 꼭 한 번 보세요
(조용히 틀리는 것들 + 재현성 체크리스트).

세 모델의 추론 결과 CSV 는 **추가 옵션 없이 그대로** 채점됩니다:

```bash
python -m tse_eval -i <manifest>.csv -o <out>.csv --model-name tpex \
    --group-by overlap_ratio,prompt_category,same_gender,first_speak
```

> TPEX / LLM-TSE / StyleTSE 매니페스트 모두 추출 오디오 경로를 `pred_path` 컬럼에 쓰는데,
> 현재 자동 인식 후보에 그 이름이 없어서 `--est-col pred_path` 를 붙여야 합니다.
