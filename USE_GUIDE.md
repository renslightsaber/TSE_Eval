# 📖 TSE_Eval 사용 가이드 (USE_GUIDE)

> 이 문서는 **처음 보는 사람도** TSE_Eval 을 바로 쓸 수 있도록 단계별로 설명합니다.
> 개요만 빠르게 보려면 [README.md](README.md) 를 참고하세요.

---

## 목차

1. [설치](#1-설치)
2. [1분 체험 (합성 예제)](#2-1분-체험-합성-예제)
3. [입력 CSV 만들기](#3-입력-csv-만들기)
4. [컬럼 자동 인식 규칙](#4-컬럼-자동-인식-규칙)
5. [실행 방법과 옵션](#5-실행-방법과-옵션)
6. [출력 이해하기](#6-출력-이해하기)
7. [오버랩(overlap) 요약](#7-오버랩overlap-요약)
8. [Python API](#8-python-api)
9. [자주 묻는 질문 (FAQ)](#9-자주-묻는-질문-faq)
10. [확장: WER / Speaker Similarity](#10-확장-wer--speaker-similarity)

---

## 1. 설치

TPEX 와 **동일한 환경**(Python 3.10.20, torch 2.5.1+cu121)을 권장합니다.

```bash
# (권장) 전용 가상환경 또는 conda env
conda create -n tse_eval python=3.10.20 -y
conda activate tse_eval

# 의존성 설치 (torch/torchaudio 는 requirements.txt 안의 cu121 인덱스에서 받음)
pip install -r requirements.txt
```

> **GPU 없이 CPU만 있어도 됩니다.** DNSMOS(ONNX)는 CPU로 돌아갑니다.
> GPU 가속을 원하면 `onnxruntime` 대신 `onnxruntime-gpu==1.20.2` 를 설치하세요.

설치 확인:

```bash
python -c "import tse_eval; print(tse_eval.__version__)"
```

---

## 2. 1분 체험 (합성 예제)

실제 데이터가 없어도 파이프라인을 바로 돌려볼 수 있습니다.

```bash
python examples/make_example.py      # 합성 wav 6개 + sample_input.csv 생성
python -m tse_eval -i examples/sample_input.csv -o examples/results.csv
```

- `examples/results.csv` — 발화별 지표
- `examples/results_summary.csv` — 오버랩 구간별 + 전체 평균

> ⚠️ 합성 예제는 **순수 톤/노이즈**라서 PESQ·STOI 값 자체는 의미가 없습니다(음성이 아니므로).
> "파이프라인이 잘 도는지" 확인용입니다. 실제 값은 진짜 음성으로 평가하세요.

---

## 3. 입력 CSV 만들기

CSV **한 줄 = 발화 1개**이며, 다음 4가지가 필요합니다.

| 항목 | 설명 | 필수? |
|---|---|---|
| **id** | 발화 식별자 (`file_id` 등) | 선택(없어도 됨) |
| **estimate** | 모델이 **추출**한 target speech wav 경로 | ✅ 필수 |
| **reference** | 정답(GT) target speech wav 경로 | ✅ 필수 |
| **mixture** | 혼합(mixture) 음성 wav 경로 | ✅ 필수 (SI-SDRi 계산에 필요) |
| **overlap** | 오버랩 비율/구간 (`40%`, `100%`, `0L`, `0-40%` 등) | 선택 |

예시:

```csv
file_id,estimate,reference,mixture,overlap
sample_0001,/data/out/0001_est.wav,/data/gt/0001.wav,/data/mix/0001.wav,0%
sample_0002,/data/out/0002_est.wav,/data/gt/0002.wav,/data/mix/0002.wav,40%
sample_0003,/data/out/0003_est.wav,/data/gt/0003.wav,/data/mix/0003.wav,100%
```

- 경로는 **절대경로** 또는 실행 위치 기준 상대경로 모두 가능합니다.
- 오디오 포맷은 `torchaudio` 가 읽을 수 있는 것이면 됩니다(wav 권장). 스테레오는 자동으로 mono 평균 처리됩니다.
- 샘플레이트가 제각각이어도 내부에서 자동 리샘플합니다.

---

## 4. 컬럼 자동 인식 규칙

컬럼 이름을 지정하지 않으면 아래 후보에서 **대소문자 무시**로 자동 인식합니다(먼저 매칭되는 것 우선).

| 역할 | 자동 인식 후보 |
|---|---|
| id | `file_id`, `id`, `utt_id`, `utterance_id`, `name`, `filename` |
| estimate | `estimate`, `extracted`, `est`, `pred`, `prediction`, `enhanced`, `output`, `est_path`, `estimate_path` |
| reference | `reference`, `target`, `gt`, `ground_truth`, `clean`, `ref`, `ref_path`, `target_path` |
| mixture | `mixture`, `mixed`, `mix`, `mixture_path`, `mixed_path`, `noisy` |
| overlap | `overlap`, `ovr`, `overlap_ratio`, `ovr_ratio`, `overlap_pct`, `overlap_percent` |

이름이 다르면 **직접 지정**하세요:

```bash
python -m tse_eval -i preds.csv -o out.csv \
    --est-col my_estimate \
    --ref-col my_target \
    --mix-col my_mixture \
    --id-col  utt \
    --ovr-col overlap_bin
```

> `estimate/reference/mixture` 중 하나라도 못 찾으면 친절한 에러와 함께 사용 가능한 컬럼 목록을 보여줍니다.

---

## 5. 실행 방법과 옵션

두 가지 방법 모두 동일합니다:

```bash
python -m tse_eval  -i preds.csv -o results.csv
tse-eval            -i preds.csv -o results.csv     # pip install 후 콘솔 스크립트
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--input, -i` | (필수) | 입력 CSV 경로 |
| `--output, -o` | (필수) | per-row 결과 CSV 경로 |
| `--summary-output, -s` | `<output>_summary.csv` | 요약 CSV 경로. `none` 이면 요약 생략 |
| `--target-sr` | `16000` | 작업 샘플레이트. SI-SDR 는 무관, 지각 지표는 항상 16k |
| `--metrics` | `all` | 계산할 지표 부분집합 (`si_sdr,pesq` 처럼 콤마로) |
| `--id-col` / `--est-col` / `--ref-col` / `--mix-col` / `--ovr-col` | 자동 | 컬럼 이름 직접 지정 |
| `--no-progress` | off | 진행바 숨김 |

예시 — 일부 지표만, 요약 없이:

```bash
python -m tse_eval -i preds.csv -o out.csv --metrics si_sdr,si_sdri,pesq -s none
```

---

## 6. 출력 이해하기

### (A) per-row CSV (`--output`)

**입력 컬럼을 그대로 유지**하고 그 뒤에 지표 컬럼을 붙입니다:

```
file_id, estimate, reference, mixture, overlap,   ← 입력 그대로
si_sdr, si_sdri, stoi, estoi, pesq,
dnsmos_sig, dnsmos_bak, dnsmos_ovrl, dnsmos_p808,
error                                             ← 실패 시 원인 문자열(정상은 빈칸)
```

- 특정 지표 계산이 실패하면 그 값만 `nan`, 나머지는 정상입니다.
- 파일을 못 읽는 등 **행 전체가 실패**하면 지표는 모두 `nan`, `error` 에 원인이 기록되고 **파이프라인은 멈추지 않습니다**.

### (B) summary CSV (`--summary-output`)

오버랩 그룹별 + 전체(`ALL`) 평균과 개수(`n`):

```
overlap,  n,  si_sdr, si_sdri, stoi, estoi, pesq, dnsmos_sig, dnsmos_bak, dnsmos_ovrl, dnsmos_p808
0%,       120, ...
40%,      118, ...
100%,     121, ...
ALL,      359, ...
```

평균은 **NaN 을 제외**하고 계산합니다.

---

## 7. 오버랩(overlap) 요약

- **오버랩 컬럼이 있으면**: 그 값**그대로**를 그룹 키로 사용합니다.
  숫자 파싱을 하지 않으므로 `0%`, `20%`, `100%` 뿐 아니라 `0L`, `0S`, `0-40%`, `40-70%`
  (예: LLM-TSE 논문식 구간) 같은 임의 라벨도 안전하게 그룹화됩니다.
- **오버랩 컬럼이 없으면**: 전체를 fully-overlapped(100%)로 보고 요약을 **`ALL` 1줄**만 만듭니다.
- 즉, **입력 CSV 에 어떤 오버랩 컬럼이 오든 출력 요약이 자동으로 그에 맞춰** 구성됩니다.

---

## 8. Python API

```python
from tse_eval import evaluate_csv, compute_row_metrics
from tse_eval.metrics import si_sdr, si_sdri, pesq_wb, stoi_metric, dnsmos

# 1) 전체 파이프라인 (DataFrame 3개 반환)
per_row, summary, cols = evaluate_csv(
    "preds.csv",
    target_sr=16000,
    est_col=None,           # None 이면 자동 인식
    ovr_col=None,
)
per_row.to_csv("out.csv", index=False)
summary.to_csv("out_summary.csv", index=False)

# 2) 배열 하나로 개별 계산 (numpy 1-D, float)
snr   = si_sdr(est, ref)             # dB
snri  = si_sdri(est, ref, mix)       # dB
estoi = stoi_metric(ref, est, extended=True)
pesq  = pesq_wb(ref, est)            # ref/est 는 16kHz 여야 함
mos   = dnsmos(est)                  # {'dnsmos_sig':..,'dnsmos_bak':..,'dnsmos_ovrl':..,'dnsmos_p808':..}

# 3) 한 행의 모든 지표
row = compute_row_metrics(est, ref, mix, sr=16000)   # dict, 키 = METRIC_COLUMNS
```

---

## 9. 자주 묻는 질문 (FAQ)

**Q. 샘플레이트가 24 kHz(또는 8 kHz)인데 괜찮나요?**
네. 내부에서 자동 리샘플합니다. SI-SDR 은 샘플레이트에 무관하고, PESQ/STOI/DNSMOS 는 항상 16 kHz 로 맞춰 계산합니다.

**Q. 추출/정답/혼합의 길이가 조금씩 다릅니다.**
공통 최소 길이로 잘라(trim) 정렬합니다. 큰 차이가 나면 정렬 문제일 수 있으니 확인하세요.

**Q. PESQ/STOI 가 `nan` 으로 나옵니다.**
너무 짧거나 무음에 가까운 신호, 또는 지각 지표가 처리 못 하는 입력일 때 `nan` 이 됩니다(크래시 대신 안전 처리). `error` 컬럼과 원본 오디오를 확인하세요.

**Q. DNSMOS 가 모델을 다운로드하나요?**
아니요. `speechmos` 패키지에 ONNX 모델이 **번들**되어 있어 오프라인으로 동작합니다.

**Q. GPU 가 꼭 필요한가요?**
아니요. 전부 CPU 로 동작합니다. 대량 평가 시 속도를 위해 `onnxruntime-gpu` 를 쓸 수 있습니다.

**Q. 결과가 재현되나요?**
동일 입력·동일 버전에서 결정론적입니다. 지표 라이브러리 버전이 다르면 소수점 이하가 달라질 수 있으니 [`requirements.txt`](requirements.txt) 핀을 맞추세요.

---

## 10. 확장: WER / Speaker Similarity

현재 기본 배포는 **핵심 6종 지표**만 포함합니다(가벼운 설치·오프라인 동작 목적).
아래 지표는 추가 의존성이 필요하여 기본에서 제외했지만, 확장 지점을 안내합니다.

- **WER (Word Error Rate)** — ASR 필요
  ```bash
  pip install openai-whisper jiwer      # 또는 faster-whisper
  ```
  Whisper 로 추출 음성과 GT 음성을 각각 전사한 뒤 `jiwer.wer()` 로 비교하는 방식을
  `tse_eval/metrics.py` 에 추가하면 됩니다. (첫 실행 시 ASR 모델 다운로드가 발생합니다.)

- **Speaker Similarity** — 화자 임베딩 모델 필요
  ```bash
  pip install resemblyzer               # 또는 speechbrain
  ```
  추출/GT 의 화자 임베딩 코사인 유사도로 계산합니다.

> 필요해지면 위 두 지표를 `--wer`, `--spk-sim` 같은 **옵션 플래그(기본 off)** 로 추가하는 것을 권장합니다.
