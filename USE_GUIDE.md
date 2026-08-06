# 📖 TSE_Eval 사용 가이드 (USE_GUIDE)

> 이 문서는 **처음 보는 사람도** TSE_Eval 을 바로 쓸 수 있도록 단계별로 설명합니다.
> 개요만 빠르게 보려면 [README.md](README.md) 를 참고하세요.
> 설치는 [INSTALL.md](INSTALL.md), **주의사항·재현성 체크리스트는 [CAVEATS.md](CAVEATS.md)** 입니다.

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
10. [WER / Speaker Similarity](#10-wer--speaker-similarity)

---

## 1. 설치

> 🛠️ **H100 / H200 을 쓰신다면 → [INSTALL.md](INSTALL.md) 를 그대로 따라가세요.**
> 단계별 명령·소요 시간·정상 출력·문제 해결이 모두 정리돼 있습니다. 이 문서는 **사용법** 전용입니다.

그 외 환경(CPU 등)은 아래로 충분합니다. TPEX 와 **동일한 환경**(Python 3.10.20, torch 2.5.1+cu121)을 권장합니다.

```bash
# (권장) 전용 conda env
conda create -n tseeval python=3.10.20 -y
conda activate tseeval

# 의존성 설치 (torch/torchaudio 는 requirements.txt 안의 cu121 인덱스에서 받음)
pip install -r requirements.txt

# ★ 필수 추가 — 없으면 DNSMOS 4개 열이 조용히 전부 nan 이 됩니다
pip install librosa==0.11.0
```

> ⚠️ **`librosa` 를 빼먹지 마세요.** `speechmos` 는 의존성을 선언하지 않는데 내부에서
> `librosa` 를 import 하고, 지표 함수는 예외를 `nan` 으로 삼킵니다.
> 그래서 에러 메시지 없이 `dnsmos_sig/bak/ovrl/p808` 이 전 행 `nan` 이 됩니다.
>
> 💡 **GPU 없이 CPU만 있어도 됩니다.** DNSMOS(ONNX)는 CPU로 돌아갑니다.
> `onnxruntime-gpu` 로 바꿔도 **DNSMOS 는 그대로 CPU 를 씁니다** — `speechmos` 가
> 세션에 `providers` 를 넘기지 않기 때문입니다(실측 확인).
> 자세한 속도 수치는 [INSTALL.md 의 DNSMOS 속도](INSTALL.md#dnsmos-속도-실측) 참고.

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
| estimate | `estimate`, `extracted`, `est`, `pred`, `prediction`, `enhanced`, `output`, `est_path`, `estimate_path`, **`pred_path`**, `prediction_path`, `extracted_path`, `enhanced_path` |
| reference | `reference`, `target`, `gt`, `ground_truth`, `clean`, `ref`, `ref_path`, `target_path` |
| mixture | `mixture`, `mixed`, `mix`, `mixture_path`, `mixed_path`, `noisy` |
| overlap | `overlap`, `ovr`, `overlap_ratio`, `ovr_ratio`, `overlap_pct`, `overlap_percent` |
| 간섭화자 (선택) | `interference_path`, `infer_path`, `interference`, `interferer_path` |
| 정답 문장 (선택, WER) | `target_sentence`, `text`, `transcript`, `reference_text` |

> ✅ **TPEX / LLM-TSE / StyleTSE 매니페스트는 추가 옵션 없이 그대로 인식됩니다.**
> 세 프로젝트 모두 추출 오디오를 `pred_path` 에 쓰고, GT 는 `target_path`, 혼합은 `mixed_path` 입니다.
> 간섭화자 컬럼 이름만 서로 다른데(TPEX `infer_path`, 나머지 `interference_path`) 둘 다 인식합니다.
> 이 컬럼은 통과만 시키며 아직 어떤 지표도 쓰지 않습니다(SDR/SIR/SAR 미구현).

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

**매번 바뀌는 값은 CLI, 세 프로젝트가 같아야 하는 정책은 [`configs/config.yaml`](configs/config.yaml)** 에 둡니다.
우선순위는 **CLI > config > 내장 기본값** 이고, 최종 해석 결과는 `<output>_config.json` 으로 저장됩니다.

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--input, -i` | (필수) | 입력 매니페스트 CSV |
| `--output, -o` | (필수) | per-row 결과 CSV 경로 |
| `--summary-output, -s` | `<output>_summary.csv` | 요약 CSV 경로. `none` 이면 요약 생략 |
| `--model-name` | 없음 | 채점 대상 이름(`tpex`/`llmtse`/`styletse`). 요약 CSV 와 sidecar 에 기록 |
| `--group-by` | overlap 컬럼 | 층화축, 콤마 구분. `same_gender` 는 자동 파생 |
| `--source-csv` | 없음 | PORTE-v3 소스 CSV 를 `file_id` 로 left join (선택) |
| `--metrics` | config 의 세트 | 지표 부분집합. `spk_sim` 은 기본 포함, **`wer` 는 명시할 때만** |
| `--si-sdr-backend` | `native` | `native` \| `asteroid` (값은 ~1e-13 이내로 동일) |
| `--dnsmos-providers` | `cuda` | DNSMOS onnxruntime EP. `cuda`(8.4배) \| `cpu`. ⚠ 값이 ~3e-3 달라지니 비교 대상은 같은 값으로 |
| `--dnsmos-threads` | `4` | DNSMOS intra-op 스레드. `0` 은 onnxruntime 기본(2.7배 느림 + 경고 폭주) |
| `--config` | `configs/config.yaml` | 정책 YAML 경로 |
| `--target-sr` | config(`24000`) | 작업 샘플레이트. SI-SDR/SI-SDRi/STOI/ESTOI 를 이 SR 로 계산하고, PESQ/DNSMOS/WER/spk_sim 만 내부에서 16 kHz 로 낮춤 ([9번 FAQ](#9-자주-묻는-질문-faq)) |
| `--id-col` / `--est-col` / `--ref-col` / `--mix-col` / `--ovr-col` / `--txt-col` | 자동 | 컬럼 이름 직접 지정 |
| `--no-progress` | off | 진행바 숨김 |

### 요약 CSV 는 long-format 입니다

축을 몇 개 주든 파일은 하나이고, 컬럼은 `model_name, axis, group, n, <지표들>` 입니다.
축마다 자체 `ALL` 행이 들어가고, 세 프로젝트 요약을 그대로 이어 붙이면 비교표가 됩니다.

```
model_name  axis             group          n   si_sdr  si_sdri
tpex        overlap_ratio    0.0          834  13.79    12.88
tpex        overlap_ratio    ALL         5000  10.20     9.40
tpex        same_gender      same        1365   9.81     9.02
tpex        same_gender      diff        3635  10.35     9.54
tpex        same_gender      ALL         5000  10.20     9.40
```

> `wer` 만 집계 방식이 다릅니다 — **corpus micro-WER**(전체 편집거리 합 / 전체 참조단어 합)로
> 묶습니다. 행별 WER 을 평균하면 짧은 발화가 과대 가중되어 값이 달라지므로, 논문 표에 쓰는
> micro 값을 요약에 넣습니다. 행별 값도 `wer` 열에 그대로 남아 검수할 수 있습니다.

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
    target_sr=24000,        # 기본값. PORTE-v3 원본 SR
    est_col=None,           # None 이면 자동 인식
    ovr_col=None,
)
per_row.to_csv("out.csv", index=False)
summary.to_csv("out_summary.csv", index=False)

# 2) 배열 하나로 개별 계산 (numpy 1-D, float)
snr   = si_sdr(est, ref)                    # dB, SR 무관
snri  = si_sdri(est, ref, mix)              # dB, SR 무관
estoi = stoi_metric(ref, est, 24000, extended=True)   # 3번째 인자 = 입력 SR (기본 16000)
pesq  = pesq_wb(ref, est)                   # ★ ref/est 는 16 kHz 여야 함
mos   = dnsmos(est)                         # ★ est 는 16 kHz 여야 함
                                            # {'dnsmos_sig':..,'dnsmos_bak':..,'dnsmos_ovrl':..,'dnsmos_p808':..}

# 3) 한 행의 모든 지표 (SR 리샘플을 알아서 처리)
row = compute_row_metrics(est, ref, mix, sr=24000)   # dict, 키 = METRIC_COLUMNS
```

---

## 9. 자주 묻는 질문 (FAQ)

**Q. 샘플레이트가 24 kHz(또는 8 kHz)인데 괜찮나요?**
네. `--target-sr` 로 작업 SR 을 정하면(**기본 24000**) 오디오를 그 SR 로 로드하고, 16 kHz 전용 지표만 내부에서 자동으로 낮춥니다. 지표별로 계산 SR 이 다릅니다:

| 지표 | 계산 SR | 이유 |
|---|---|---|
| SI-SDR / SI-SDRi | `--target-sr` (native) | 샘플레이트에 무관한 정의 |
| STOI / ESTOI | `--target-sr` (native) | `pystoi` 가 내부에서 10 kHz 로 리샘플 → 미리 16k 로 낮추면 이중 리샘플만 추가됨 |
| PESQ | **16 kHz** | ITU-T P.862 가 8/16 kHz 만 정의 |
| DNSMOS | **16 kHz** | 16 kHz 학습 모델 (`speechmos` 가 다른 SR 을 거부) |

이 프로토콜은 형제 프로젝트(TPEX / LLM-TSE / StyleTSE)의 `eval.py` 와 동일하므로, 같은 오디오에 대해 같은 숫자가 나옵니다.

**Q. `dnsmos_*` 4개 열이 전부 `nan` 입니다.**
거의 항상 **`librosa` 미설치**입니다. `speechmos` 는 의존성을 선언하지 않는데 내부에서 `librosa` 를 import 하고, 지표 함수는 예외를 `nan` 으로 삼키기 때문에 에러 메시지 없이 조용히 실패합니다. `pip install librosa==0.11.0` 로 해결됩니다.

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

## 10. WER / Speaker Similarity

두 지표는 **구현 완료**되었습니다. 차이는 기본 활성 여부뿐입니다:

| 지표 | 기본 | 15,000 utt 비용 | 켜는 방법 |
|---|---|--:|---|
| **spk_sim** (ECAPA) | ✅ 켜짐 | 20분 | 자동 |
| **WER** (Whisper large-v3) | ❌ 꺼짐 | 178분 | `--metrics all,wer` 또는 config 의 `- wer` 주석 해제 |

```bash
# 논문 표용 — WER 까지 전부
python -m tse_eval -i manifest.csv -o out.csv --model-name tpex \
    --metrics all,wer --group-by overlap_ratio,same_gender
```

> **WER 요약은 corpus micro-WER** 입니다(전체 편집거리 합 / 전체 참조단어 합).
> 행별 `wer` 열도 남으니 검수에 쓰고, 표에는 요약값을 쓰세요.
> `target_sentence` 가 없는 행(StyleTSE 매니페스트)은 자동으로 제외되며,
> `--source-csv` 로 PORTE-v3 를 join 하면 그 열을 가져올 수 있습니다.

### 모델 준비

[`requirements_h200.txt`](requirements_h200.txt) 에 의존성이 핀되어 있고, 가중치는 스크립트 하나로 받습니다:

```bash
export HF_HOME=/home/work/my-checkpoints/hf_cache   # ★ 영속 저장소 (홈은 세션 종료 시 삭제됨)
python scripts/download_models.py                   # 둘 다 (~3.2 GB) + 검증
python scripts/download_models.py --only spk        # ECAPA 만
python scripts/download_models.py --only wer        # Whisper 만
```

| 지표 | 모델 | 크기 | 계산 SR |
|---|---|---|---|
| **Speaker Similarity** | `speechbrain/spkrec-ecapa-voxceleb` (ECAPA-TDNN, 192-dim) | ~89 MB | **16 kHz** |
| **WER** | `openai/whisper-large-v3` (+ `jiwer`) | ~3.1 GB | **16 kHz** |

`whisper-large-v3` 은 LLM-TSE 의 `eval.py` 와 같은 모델이므로 WER 숫자를 바로 비교할 수 있습니다.

### ⚠️ 알아둘 점

- **둘 다 16 kHz 전용입니다.** 파이프라인이 자동으로 리샘플하므로 신경 쓸 필요는 없지만,
  직접 함수를 호출할 때는 반드시 16 kHz 를 넘기세요. Whisper 는 다른 SR 을 `ValueError` 로
  거부하는 반면 **ECAPA 는 에러 없이 그냥 틀린 임베딩을 냅니다**(`encode_batch()` 는 리샘플 안 함).
- 모델은 **한 번만 로드**되어 전 행에 재사용됩니다(모듈 레벨 lazy 싱글턴).
  Whisper 최초 로드에 약 9초가 걸립니다.
- WER 참조 텍스트는 매니페스트의 `target_sentence` 열입니다.
  **StyleTSE 매니페스트에는 이 열이 없어** 그 행들은 `nan` 이 되고 micro 집계에서 제외됩니다.
  `--source-csv` 로 PORTE-v3 를 join 하면 열을 가져올 수 있습니다.
- 행별 출력에 `wer_hyp`(Whisper 전사 결과) 열이 함께 남아 오류를 눈으로 확인할 수 있습니다.

모델 id 등은 [`configs/config.yaml`](configs/config.yaml) 에서 바꿀 수 있고,
실제 사용된 값은 `<output>_config.json` 에 기록됩니다.
