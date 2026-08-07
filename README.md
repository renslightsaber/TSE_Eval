# TSE_Eval

[![Python](https://img.shields.io/badge/python-3.10.20-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-33%20passed-brightgreen.svg)](tests/)
![Platform](https://img.shields.io/badge/platform-CPU%20%7C%20GPU-lightgrey.svg)

> **A unified, easy-to-use evaluation toolkit for Target Speech Extraction (TSE).**

CSV 하나를 넣으면 → 표준 음성 품질 지표를 계산해 → 결과 CSV로 돌려줍니다.
서로 다른 baseline(StyleTSE, LLM-TSE 등)으로 뽑은 결과를 **동일한 파이프라인**으로 평가해 공정하게 비교하기 위한 도구입니다.

---

## ✨ 무엇을 하나요?

- **입력**: 각 발화(utterance)마다 `추출 음성 / 정답(GT) 음성 / 혼합(mixture) 음성` 경로가 담긴 **CSV 한 장**
- **출력**:
  1. **per-row CSV** — 입력 컬럼 그대로 + 지표 컬럼이 붙은 결과
  2. **summary CSV** — 오버랩(overlap) 구간별 평균 + 전체 평균 요약
- **지표(기본 6종)**: `SI-SDR`, `SI-SDRi`, `STOI`, `ESTOI`, `PESQ`, `DNSMOS(SIG/BAK/OVRL)`

---

## 📊 지원 지표

| 지표 | 의미 | 종류 | 필요한 입력 | 계산 SR |
|---|---|---|---|---|
| **SI-SDR** | Scale-Invariant SDR (dB) — 분리 품질 | 참조 기반 | 추출, GT | native |
| **SI-SDRi** | SI-SDR 개선량 = `SI-SDR(추출,GT) − SI-SDR(혼합,GT)` | 참조 기반 | 추출, GT, **혼합** | native |
| **input_si_sdr** | 혼합 자체의 SI-SDR (위 식의 뺄셈 항) | 참조 기반 | 혼합, GT | native |
| **input_si_sdr_pairwise** | 같은 값이나 **추출과 무관** (아래 설명) | 참조 기반 | 혼합, GT | native |
| **STOI** | 명료도 (0–1) | 참조 기반 | 추출, GT | native |
| **ESTOI** | Extended STOI (0–1) | 참조 기반 | 추출, GT | native |
| **PESQ** | 지각 음질 (wideband, ~1–4.5) | 참조 기반 | 추출, GT | **16 kHz** |
| **DNSMOS** | 무참조 MOS: **SIG**(음성)/**BAK**(배경)/**OVRL**(종합) + P.808 | **무참조** | 추출만 | **16 kHz** |
| **spk_sim** | ECAPA 화자 임베딩 코사인 유사도 | 참조 기반 | 추출, GT | **16 kHz** |
| **WER** ⚙️ | Whisper large-v3 전사 오류율 (`target_sentence` 대비) | 텍스트 참조 | 추출, 정답문장 | **16 kHz** |

⚙️ = **기본 off**. WER 은 15,000 utt 기준 약 178분이 추가되어 반복 실험이 무거워지므로,
논문 표를 만들 때만 켭니다 — `configs/config.yaml` 의 `# - wer` 주석을 해제하거나 지표를
명시적으로 열거하세요(`--metrics all,wer` 는 **동작하지 않습니다**: `all` 은 인자 전체일 때만
특별 처리됩니다). `spk_sim`(약 20분)은 기본 포함입니다.

### 🔬 정규화 정책 — 반드시 세 프로젝트가 동일해야 합니다

두 지표는 **입력을 어떻게 다듬느냐에 따라 값이 달라집니다.** 둘 다 예전에 에러도 `nan` 도
없이 조용히 틀린 값을 냈던 지점이라, 정규화값과 비정규화값을 **함께 기록**합니다.

| 컬럼 | 내용 |
|---|---|
| `dnsmos_*` | rms **−26 dBov**(ITU-T P.56)로 정규화 후 측정 ← **보고용** |
| `dnsmos_*_clipped` | ±1 로 하드 클리핑 후 측정 (이 repo 의 예전 동작, 대조용) |
| `wer`, `wer_edits`, `wer_words` | Whisper 자체 영어 정규화기 적용 ← **보고용** |
| `wer_raw`, `wer_raw_edits`, `wer_raw_words` | 원시 문자열 비교 (`llmtse/eval.py` 와 동일) |
| `wer_hyp` | Whisper 원본 전사 — 모델 재실행 없이 재계산할 수 있게 남깁니다 |

`_clipped` / `_raw` 컬럼은 **자동으로 따라옵니다**(`--metrics` 에 적을 필요 없음).
LLM-TSE 실측 영향: `dnsmos_ovrl` 2.55 → **2.91**, micro-WER 0.593 → **0.524**.
**SI-SDR·SI-SDRi·STOI·ESTOI·PESQ·spk_sim 은 레벨과 무관**하므로(실측 차이 < 1e-5) 값이
바뀌지 않습니다. 자세한 근거는 [CAVEATS.md §1-6 · §1-7](CAVEATS.md) 를 보세요.

> ⚠️ **추론 산출물은 반드시 float32 WAV 로 저장하세요.** `torchaudio.save` 와 `sf.write` 는
> 기본이 PCM_16 이고 ±1 밖을 **조용히 클리핑**합니다. TSE 모델 출력은 ±1 을 넘는 것이
> 정상이며(LLM-TSE 는 100% 모든 행이 초과, max peak 17.25), 클리핑되면 SI-SDR 이 평균
> **4.7 dB** 손실됩니다. → [CAVEATS.md §1-8](CAVEATS.md)

### ⏱️ 예상 소요 시간 (5,000행 × 3시스템 = 15,000 utt)

| 지표 | 시간 |
|---|--:|
| SI-SDR 계열 / STOI+ESTOI / PESQ | 2분 / 28분 / 51분 |
| DNSMOS (CUDA 기본) | 44분 |
| spk_sim | 20분 |
| **기본 세트 합계** | **약 145분** |
| WER 추가 시 | +178분 |

> DNSMOS 는 `tse_eval/ort_setup.py` 가 onnxruntime 스레드와 providers 를 주입해
> **기본으로 CUDA 가속**됩니다(무가속 369분 → 44분, 8.4배).
> CPU 로 되돌리려면 `--dnsmos-providers cpu`.
> ⚠️ CUDA 와 CPU 의 DNSMOS 값은 최대 약 `3e-3` 다릅니다 — **비교표의 세 시스템은 반드시
> 같은 설정으로** 채점하세요. 실제 사용된 EP 는 `<output>_config.json` 에 기록됩니다.

> **`input_si_sdr` 을 왜 따로 내보내나요?**
> `SI-SDRi = SI-SDR − input_si_sdr` 항등식을 결과 CSV 에서 바로 확인할 수 있어야 하기 때문입니다.
> 두 가지를 함께 냅니다:
> - `input_si_sdr` — 세 신호를 공통 길이로 자른 뒤 계산(형제 프로젝트 `eval.py` 와 동일 규약).
> - `input_si_sdr_pairwise` — 혼합/GT 만으로 계산해 **추출 결과와 무관**. 같은 샘플이면
>   프로젝트가 달라도 값이 동일하므로, *하나의 파이프라인으로 채점했다*는 직접적인 증거가 됩니다.

> **SDR / SIR / SAR 은 넣지 않았습니다.** `mir_eval.bss_eval_sources` 는 발화당 약 1초라
> 5,000행에 약 91분이 걸리고, blind source separation 전제라 간섭 신호 없이는 SIR 이 `inf` 로
> 나와 TSE 에 부적합합니다.

> **샘플레이트 프로토콜** — `--target-sr` (기본 **24000**) 이 "native" 레이트입니다.
> - SI-SDR / SI-SDRi 는 **네이티브 구현**(별도 `asteroid` 의존성 없음)이고 샘플레이트에 무관합니다.
> - STOI / ESTOI 도 native 로 계산합니다 — `pystoi` 가 내부에서 10 kHz 로 리샘플하므로
>   미리 16 kHz 로 낮추면 `24k→16k→10k` 이중 리샘플만 더해집니다.
> - PESQ 는 ITU-T P.862 가 8/16 kHz 만 정의하고, DNSMOS 는 16 kHz 모델이라
>   이 둘만 내부에서 **16 kHz** 로 리샘플합니다.
> - DNSMOS 는 `speechmos` 의 **번들 ONNX** 로 동작 → **모델 다운로드 불필요**.
>   단 `speechmos` 가 의존성을 선언하지 않으므로 **`librosa` 가 반드시 설치돼 있어야** 합니다
>   (없으면 DNSMOS 4열이 조용히 전부 `nan`).

---

## 🚀 빠른 시작

**1) 설치** — 사용 중인 GPU 에 따라 갈립니다.

| 환경 | 방법 |
|---|---|
| **NVIDIA H100 / H200** | 👉 **[INSTALL.md](INSTALL.md)** 를 따라가세요 (`requirements_h200.txt` 기준, 단계별 안내) |
| 그 외 / CPU only | `pip install -r requirements.txt` + `pip install librosa==0.11.0` |

> ⚠️ `requirements.txt` 는 A6000/CPU 시절 파일이라 **`librosa` 가 빠져 있습니다.**
> `librosa` 없이 돌리면 `speechmos` 가 내부에서 그것을 import 하지 못해
> **DNSMOS 4개 열이 에러 없이 전부 `nan`** 이 됩니다. 꼭 같이 설치하세요.

```bash
# 2) 합성 예제 생성 (실제 데이터 없이 바로 체험)
python examples/make_example.py

# 3) 평가 실행
python -m tse_eval \
    --input  examples/sample_input.csv \
    --output examples/results.csv
```

실행하면 `examples/results.csv`(per-row)와 `examples/results_summary.csv`(요약)가 생기고,
콘솔에 오버랩 구간별 요약 표가 출력됩니다.

### 입력 CSV 예시

```csv
file_id,estimate,reference,mixture,overlap
utt0,/path/est0.wav,/path/gt0.wav,/path/mix0.wav,40%
utt1,/path/est1.wav,/path/gt1.wav,/path/mix1.wav,100%
```

- `estimate` = 모델이 추출한 target speech, `reference` = 정답 target speech, `mixture` = 혼합 음성
- **컬럼 이름은 자동 인식**됩니다 (`estimate/extracted/est/pred…`, `reference/target/gt…`, `mixture/mixed/mix…`).
- `overlap` 컬럼은 **선택**입니다. 없으면 전체를 하나로 보고 요약 1줄만 만듭니다.

### 세 프로젝트 매니페스트 채점 + 층화 비교

TPEX / LLM-TSE / StyleTSE 의 inference 매니페스트는 **추가 옵션 없이 그대로** 인식됩니다
(추출 경로 `pred_path`, GT `target_path`, 혼합 `mixed_path`).

```bash
python -m tse_eval \
    --input  /home/work/my-outputs/tpex/<run>/inference_manifest.csv \
    --output results/tpex.csv \
    --model-name tpex \
    --group-by overlap_ratio,prompt_category,same_gender,first_speak
```

- `--model-name` 은 요약 CSV 의 `model_name` 열에 들어가므로, 세 프로젝트 요약을
  그대로 이어 붙이면 baseline 비교표가 됩니다.
- `same_gender` 는 `target_gender == infer_gender` 로 **자동 파생**되는 축입니다.
- 요약은 **long-format**(`model_name, axis, group, n, 지표…`) 이라 축을 몇 개 주든 파일 하나입니다.
- `--source-csv <PORTE-v3 CSV>` 를 주면 `file_id` 로 left join 해서
  StyleTSE 에 없는 `target_sentence`(WER 정답)와 연속형 축(`snr_db` 등)을 가져옵니다.
  연속형 축은 자동으로 사분위 구간(`Q1..Q4`)으로 묶입니다.
- 실행마다 **`<output>_config.json` sidecar** 가 함께 생성됩니다 — 지표 세트, SI-SDR 백엔드,
  지표별 샘플레이트, 모델 id, 입력 경로, 라이브러리 버전이 기록되어 논문에서 인용할 수 있습니다.

> 📖 자세한 사용법(컬럼 자동 인식 규칙, CLI 옵션 전체, 실데이터 팁, FAQ)은
> **[USE_GUIDE.md](USE_GUIDE.md)** 를 참고하세요.

---

## 📈 예시 출력 결과

```bash
python -m tse_eval -i preds.csv -o results.csv
```

**① per-row 결과 (`results.csv`)** — 입력 컬럼을 그대로 두고 뒤에 지표를 붙입니다:

| file_id | overlap | si_sdr | si_sdri | stoi | estoi | pesq | dnsmos_sig | dnsmos_bak | dnsmos_ovrl |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| utt0001 | 0%   | 13.82 | 12.90 | 0.94 | 0.88 | 3.11 | 3.52 | 4.02 | 3.28 |
| utt0002 | 40%  | 10.47 | 9.71  | 0.90 | 0.81 | 2.68 | 3.44 | 3.95 | 3.19 |
| utt0003 | 100% | 6.13  | 5.42  | 0.83 | 0.71 | 2.05 | 3.30 | 3.79 | 2.98 |

<sub>경로 컬럼(`estimate`/`reference`/`mixture`)·`dnsmos_p808`·`error` 는 지면상 생략</sub>

**② 요약 (`results_summary.csv`)** — 오버랩 구간별 + 전체(`ALL`) 평균:

| overlap | n | si_sdr | si_sdri | stoi | estoi | pesq | dnsmos_ovrl |
|---|--:|--:|--:|--:|--:|--:|--:|
| 0%   | 120 | 13.79 | 12.88 | 0.94 | 0.88 | 3.10 | 3.27 |
| 40%  | 118 | 10.55 | 9.79  | 0.90 | 0.82 | 2.70 | 3.20 |
| 100% | 121 | 6.20  | 5.49  | 0.83 | 0.72 | 2.06 | 2.99 |
| **ALL** | **359** | **10.20** | **9.40** | **0.89** | **0.81** | **2.62** | **3.15** |

**③ 콘솔 출력** — 실행하면 같은 요약이 터미널에도 찍힙니다:

```text
[tse-eval] wrote per-row metrics → results.csv  (359 rows)
[tse-eval] wrote summary      → results_summary.csv

===== Summary =====
overlap   n  si_sdr si_sdri   stoi  estoi   pesq dnsmos_sig dnsmos_bak dnsmos_ovrl dnsmos_p808
     0% 120 13.7900 12.8800 0.9400 0.8800 3.1000     3.5200     4.0200      3.2700      3.4100
    40% 118 10.5500  9.7900 0.9000 0.8200 2.7000     3.4400     3.9500      3.2000      3.3600
   100% 121  6.2000  5.4900 0.8300 0.7200 2.0600     3.3000     3.7900      2.9900      3.2500
    ALL 359 10.2000  9.4000 0.8900 0.8100 2.6200     3.1500     3.9200      3.1500      3.3400
```

> ⚠️ 위 수치는 **출력 형식을 보여주기 위한 예시(illustrative)** 입니다. 실제 값은 모델·데이터에 따라 달라집니다.
> (오버랩이 커질수록 분리가 어려워 지표가 낮아지는 경향을 예시로 표현했습니다.)

---

## 🧩 Python API

```python
from tse_eval import evaluate_csv, compute_row_metrics, si_sdr

# 전체 파이프라인 (target_sr 기본값 24000)
per_row, summary, cols = evaluate_csv("preds.csv", target_sr=24000)

# 개별 지표
snr = si_sdr(est_wav, ref_wav)          # numpy 1-D 배열
row = compute_row_metrics(est, ref, mix, sr=24000)   # dict
```

---

## ⚙️ 환경

| 항목 | 버전 |
|---|---|
| Python | 3.10.20 |
| torch / torchaudio | 2.5.1+cu121 |
| numpy | 1.26.4 (`<2`) |

`pesq`, `pystoi`, `speechmos`, `librosa`, `onnxruntime`, `pandas`, `soundfile` —
전체 핀은 **H100/H200** 은 [`requirements_h200.txt`](requirements_h200.txt),
그 외는 [`requirements.txt`](requirements.txt) 참고.

> **CPU만 있어도 동작**합니다. GPU 는 필수가 아닙니다(DNSMOS ONNX 는 CPU 추론).
> Speaker Similarity / WER 확장 지표를 쓸 때만 GPU 가 도움이 됩니다.

---

## 📂 저장소 구조

```
tse_eval/
├── tse_eval/            # 패키지
│   ├── metrics.py       # SI-SDR, SI-SDRi, STOI/ESTOI, PESQ, DNSMOS
│   ├── audio.py         # 로드/모노/리샘플/길이 정렬
│   ├── evaluate.py      # CSV 파이프라인 + 오버랩 요약
│   └── cli.py           # 커맨드라인 진입점
│   ├── backends.py      # SI-SDR 백엔드 2종 (native 기본 / asteroid)
│   ├── ort_setup.py     # DNSMOS onnxruntime 가속 (스레드 + CUDA providers)
│   └── config.py        # 정책 YAML 로드 + 실행 설정 sidecar 기록
├── configs/
│   └── config.yaml      # 채점 정책(세 프로젝트가 동일해야 하는 값)
├── scripts/
│   └── download_models.py   # 확장 지표용 모델(ECAPA/Whisper) 다운로드 + 검증
├── examples/            # 합성 예제 생성기 + sample_input.csv
├── tests/               # pytest (합성 신호, CPU only)
├── requirements.txt         # A6000 / CPU 기준 (★ librosa 없음)
├── requirements_h200.txt    # H100/H200 기준 — 함정 12가지 주석 포함
├── INSTALL.md               # 설치 가이드 (단계별, 문제 해결)
├── USE_GUIDE.md             # 사용법 (CLI 옵션, 컬럼 인식, FAQ)
├── CAVEATS.md               # ⚠️ 주의사항 — 조용히 틀리는 것들, 재현성 체크리스트
└── README.md
```

---

## ⚠️ 논문 숫자를 만들기 전에

**[CAVEATS.md](CAVEATS.md)** 를 한 번 읽어주세요. 에러가 나는 문제보다
**에러 없이 조용히 틀린 값을 내는** 문제가 위험합니다 (예: `librosa` 누락 → DNSMOS 전부 `nan`,
ECAPA 에 24 kHz 입력 → 에러 없이 다른 임베딩). 마지막에 **재현성 체크리스트 5개**가 있습니다.

---

## 📜 라이선스

[MIT](LICENSE)
