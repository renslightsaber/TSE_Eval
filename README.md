# TSE_Eval

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
