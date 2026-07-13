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

| 지표 | 의미 | 종류 | 필요한 입력 |
|---|---|---|---|
| **SI-SDR** | Scale-Invariant SDR (dB) — 분리 품질 | 참조 기반 | 추출, GT |
| **SI-SDRi** | SI-SDR 개선량 = `SI-SDR(추출,GT) − SI-SDR(혼합,GT)` | 참조 기반 | 추출, GT, **혼합** |
| **STOI** | 명료도 (0–1) | 참조 기반 | 추출, GT |
| **ESTOI** | Extended STOI (0–1) | 참조 기반 | 추출, GT |
| **PESQ** | 지각 음질 (wideband, ~1–4.5) | 참조 기반 | 추출, GT |
| **DNSMOS** | 무참조 MOS: **SIG**(음성)/**BAK**(배경)/**OVRL**(종합) + P.808 | **무참조** | 추출만 |

> - SI-SDR / SI-SDRi 는 **네이티브 구현**(별도 `asteroid` 의존성 없음), 샘플레이트에 무관합니다.
> - PESQ / STOI / ESTOI / DNSMOS 는 내부적으로 **16 kHz** 로 리샘플해 계산합니다.
> - DNSMOS 는 `speechmos` 의 **번들 ONNX** 로 동작 → **인터넷·모델 다운로드 불필요**.

---

## 🚀 빠른 시작

```bash
# 1) 설치 (TPEX 와 동일한 torch 2.5.1+cu121 / Python 3.10 환경 권장)
pip install -r requirements.txt

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

# 전체 파이프라인
per_row, summary, cols = evaluate_csv("preds.csv", target_sr=16000)

# 개별 지표
snr = si_sdr(est_wav, ref_wav)          # numpy 1-D 배열
row = compute_row_metrics(est, ref, mix, sr=16000)   # dict
```

---

## ⚙️ 환경

| 항목 | 버전 |
|---|---|
| Python | 3.10.20 |
| torch / torchaudio | 2.5.1+cu121 |
| numpy | 1.26.4 (`<2`) |

`pesq`, `pystoi`, `speechmos`, `onnxruntime`, `pandas`, `soundfile` — 전체 핀은 [`requirements.txt`](requirements.txt) 참고.

> **CPU만 있어도 동작**합니다. GPU 는 필수가 아닙니다(DNSMOS ONNX 는 CPU 추론).

---

## 📂 저장소 구조

```
tse_eval/
├── tse_eval/            # 패키지
│   ├── metrics.py       # SI-SDR, SI-SDRi, STOI/ESTOI, PESQ, DNSMOS
│   ├── audio.py         # 로드/모노/리샘플/길이 정렬
│   ├── evaluate.py      # CSV 파이프라인 + 오버랩 요약
│   └── cli.py           # 커맨드라인 진입점
├── examples/            # 합성 예제 생성기 + sample_input.csv
├── tests/               # pytest (합성 신호, CPU only)
├── requirements.txt
├── USE_GUIDE.md
└── README.md
```

---

## 📜 라이선스

[MIT](LICENSE)
