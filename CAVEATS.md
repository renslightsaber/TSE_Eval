# ⚠️ TSE_Eval 주의사항 (Caveats)

> 이 문서는 **실제로 겪고 실측으로 확인한 것만** 모았습니다. 추측은 없습니다.
> 설치 절차는 [requirements/h200/INSTALL.md](requirements/h200/INSTALL.md), 사용법은 [USE_GUIDE.md](USE_GUIDE.md) 를 보세요.

가장 위험한 것은 에러가 나는 문제가 아니라 **에러 없이 조용히 틀린 값을 내는** 문제입니다.
그래서 §1 을 맨 앞에 뒀습니다. 논문 숫자를 만들기 전에 §1 만이라도 읽으세요.

---

## 📑 목차

1. [🔴 조용히 틀리는 것들 — 최우선](#1--조용히-틀리는-것들--최우선)
2. [샘플레이트 규약](#2-샘플레이트-규약)
3. [📐 지표 정의상의 함정](#3--지표-정의상의-함정)
4. [💾 이 서버(컨테이너) 특성](#4--이-서버컨테이너-특성)
5. [📦 설치·의존성](#5--설치의존성)
6. [🔁 재현성 체크리스트](#6--재현성-체크리스트)
7. [🚫 일부러 넣지 않은 것](#7--일부러-넣지-않은-것)

---

## 1. 🔴 조용히 틀리는 것들 — 최우선

### 1-1. `librosa` 가 없으면 DNSMOS 4열이 전부 `nan` (에러 메시지 없음)

`speechmos` 휠은 **의존성을 하나도 선언하지 않는데** 내부에서 `librosa` 와 `onnxruntime` 을
모듈 최상단에서 import 합니다(`speechmos/dnsmos.py:4-6`). 지표 함수는 설계상 예외를 `nan` 으로
삼키므로, 예전에는 아무 메시지 없이 `dnsmos_sig/bak/ovrl/p808` 이 전 행 `nan` 이 됐습니다.

- **대응(적용됨)**: `requirements/a6000/requirements_a6000.txt` · `requirements/h200/requirements_h200.txt` · `pyproject.toml` 모두에
  `librosa==0.11.0` 을 명시했고, `ImportError` 시 **stderr 경고를 1회 출력**합니다.
- **확인법**: 결과 CSV 를 받으면 `dnsmos_*` 열이 전부 `nan` 인지 **항상 먼저** 보세요.

### 1-2. ECAPA(Speaker Similarity)에 24 kHz 를 넣으면 에러 없이 틀린 값

ECAPA(`spkrec-ecapa-voxceleb`)는 16 kHz 전용입니다. `hyperparams.yaml` 이 `sample_rate` 를
지정하지 않아 `Fbank` 기본값 16000 이 쓰이는데, **`encode_batch()` 는 리샘플하지 않습니다**
(`AudioNormalizer` 는 `load_audio(path)` 경로에서만 동작). Whisper·PESQ·DNSMOS 는 잘못된 SR 을
`ValueError` 로 거부하지만 **ECAPA 는 그냥 통과시킵니다.**

실측: 같은 소리의 16 kHz 판과 24 kHz 판의 임베딩 코사인 유사도 = **0.8834**
(제대로 24k→16k 리샘플하면 0.9909). 즉 조용히 다른 사람처럼 취급됩니다.

- **대응(적용됨)**: 샘플레이트 정책이 `spk_sim` 을 16 kHz 로 못 박고 파이프라인이 자동 변환.
- ⚠️ **기존 팀 코드에 이 버그가 있습니다**:
  `styletse/src/teammate_codes/Style_TSE_share/evaluate_styletse.py` 는
  `--sample_rate` 기본값이 **24000**(`:213-216`)이고 `:152-153` 에서 24 kHz 텐서를
  `encode_batch` 에 그대로 넣습니다. **그 스크립트가 낸 Sp.sim 수치는 신뢰할 수 없습니다.**

### 1-3. SI-SDR 의 eps 위치 — degenerate 입력이 `0.0 dB` 로 보이던 문제

예전 구현은 `10·log10((num+eps)/(den+eps))` 였습니다. 무음/DC-only/초소음 추정치는
`num=den=0` 이 되어 **0 dB**(= "신호와 잡음이 같다")를 반환했고, 이는 평균을 끌어올립니다.

표준 구현(`pb_bss_eval`, asteroid 내부)은 eps 를 **분모와 ratio 에만** 넣어
`10·log10(eps) = -80 dB` 라는 바닥값을 냅니다.

| 케이스 | 표준(asteroid) | 예전 구현 |
|---|--:|--:|
| 무음 estimate | −80.00 | **0.00** ❌ |
| 진폭 1e-8 | −49.37 | **0.00** ❌ |
| DC only | −80.00 | **0.00** ❌ |

- **대응(적용됨)**: eps 위치를 표준과 일치시켜 8개 케이스 전부 **4e-13** 이내로 일치.
  `tests/test_backends.py` 가 골든값으로 고정합니다.
- 규모 감각: 5,000행 중 1%만 degenerate 여도 평균이 약 0.8 dB 과대평가됩니다.

### 1-4. `pred_path` 가 자동인식되지 않던 문제

세 프로젝트(TPEX/LLM-TSE/StyleTSE) 매니페스트는 모두 추출 오디오를 `pred_path` 에 쓰는데,
est 후보 목록에 그 이름이 없어 **세 매니페스트 전부 자동인식 실패**했습니다.

- **대응(적용됨)**: `pred_path, prediction_path, extracted_path, enhanced_path` 를 후보에 추가
  (기존 후보 뒤에 붙여 하위호환 유지). 이제 `--est-col` 없이 그대로 돌아갑니다.

### 1-5. DNSMOS 를 CPU 와 CUDA 로 섞어 채점하면 값이 어긋납니다

CUDA EP 와 CPU EP 의 DNSMOS 값은 부동소수점 커널 차이로 **최대 약 3e-3** 다릅니다
(CPU 설정끼리는 4e-7 로 사실상 동일). DNSMOS 는 보통 소수점 2자리로 보고하므로 보고
정밀도보다는 작지만, **비교표에 들어가는 세 시스템이 서로 다른 EP 로 채점되면 안 됩니다.**

- **대응(적용됨)**: 실제 사용된 EP 가 `<output>_config.json` 의 `dnsmos.actual_providers` 에
  기록됩니다. 세 sidecar 를 비교해 같은지 확인하세요.

### 1-6. 🔴 모델 출력이 ±1 을 넘으면 DNSMOS 가 **클리핑된 신호**를 채점합니다

이 프로젝트에서 가장 비쌌던 버그입니다. 에러도 `nan` 도 없이 **그럴듯하지만 틀린 숫자**가
나왔고, SI-SDR 15.79 dB · PESQ 4.02 인데 WER 61.5% 라는 모순을 추적해야 드러났습니다.

`speechmos` 는 ±1 범위를 벗어난 입력을 `ValueError: np.ndarray values must be between -1
and 1.` 로 거부합니다. 예전 `metrics.dnsmos()` 는 그 제약을 `np.clip(x, -1, 1)` 로
만족시켰는데, TSE 모델 출력은 흔히 1.0 을 넘습니다 — **LLM-TSE 베이스라인은 300행 전수
확인 결과 100% 모든 행이 peak > 1.0** 이고(median 4.94, max 17.25), 정답보다 rms **+24.1 dB**
큽니다. 그 결과 median 행의 샘플 **5.5%**(최악 40.3%)가 잘린 채 채점됐습니다.

실측 (50행, 정답을 기준점으로 함께 측정):

| 조건 | sig | bak | ovrl | p808 | peak | rms dBFS |
|---|--:|--:|--:|--:|--:|--:|
| 예측, 하드 클리핑 (예전) | 3.153 | 3.208 | **2.550** | 3.173 | 1.000 | −8.1 |
| 예측, peak → 0.95 | 3.346 | 3.517 | 2.847 | 3.494 | 0.950 | −20.2 |
| **예측, rms → −26 dBov** | 3.388 | 3.611 | **2.912** | 3.494 | 0.514 | −26.0 |
| 정답, 원본 그대로 | 3.561 | 4.064 | **3.273** | 3.807 | 0.299 | −30.3 |
| 정답, rms → −26 dBov | 3.559 | 4.042 | **3.273** | 3.807 | 0.502 | −26.0 |

- **대응(적용됨)**: `metrics.dnsmos_normalize()` 가 rms 를 **−26 dBov**(ITU-T P.56)로
  맞춥니다. **클리핑이 아니라 스케일링**입니다. crest factor 가 31.7 dB 까지 가므로
  −26 dBov 후에도 **약 2%(8/400)** 는 peak 가 1.0 을 넘어, 그 경우 peak 0.99 로 한 번 더
  축소합니다(레벨을 조금 희생하지만 파형은 온전).
  −26 dBov 를 고른 이유는 **정답을 같은 방식으로 채점해도 3.273 으로 원본과 동일**해
  기준점이 흔들리지 않기 때문입니다(peak→0.95 는 3.273→3.211 로 움직임).
- 예전 방식 값은 `dnsmos_*_clipped` 컬럼으로 **함께 기록**되므로 과거 숫자와 대조 가능합니다.
  `_raw` 가 아니라 `_clipped` 인 이유: speechmos 가 ±1 초과를 거부하므로 "정규화하지 않은
  진짜 원시값" 은 애초에 계산할 수 없습니다. 그 값은 정규화를 안 한 게 아니라 **가장 많이
  변형된** 값입니다.
- **확인법**: `dnsmos_ovrl` 과 `dnsmos_ovrl_clipped` 가 같으면 정규화가 안 걸렸거나 예측이
  이미 ±1 안에 있다는 뜻입니다. `scripts/eval_tse.sh`(래퍼: `eval_llmtse.sh` · `eval_styletse.sh`)가 매 런마다 이 차이를 출력합니다.

**레벨에 영향받지 않는 지표** (예측을 정답 레벨로 맞춰 재계산한 실측 차이):

```
si_sdr / si_sdri / input_si_sdr / input_si_sdr_pairwise   -0.00000   정의상 스케일 불변
stoi / estoi                                              +0.00000   pystoi 내부 정규화
pesq                                                      -0.00001   P.862 내부 레벨 정렬
spk_sim                                                   +0.00001   ECAPA mean-var norm + 코사인
```

→ **이 9개 컬럼은 레벨과 무관하므로 기존 값이 그대로 유효합니다.** 영향은 DNSMOS 4개뿐입니다.

### 1-7. 🔴 WER 에 텍스트 정규화가 없으면 대소문자·구두점이 오류로 계산됩니다

Whisper 는 같은 모델로도 행에 따라 구두점을 붙이기도 하고 안 붙이기도 합니다. 정규화 없이
비교하면 **모든 단어가 맞아도** 오류가 쌓입니다:

```
REF: Although plump and healthy, Josiana was, we repeat, a perfect prude.
HYP: although plump and healthy josianna was we repeat a perfect prude
→ 원시 0.5455 (6 edits/11) · 정규화 0.0909 (1 edit)
   ★ 이 행의 SI-SDR 은 50.75 dB — 거의 완벽한 복원입니다.
```

LLM-TSE 5,000행 corpus micro-WER: **0.5932 → 0.5240** (약 7 pp).

- **대응(적용됨)**: Whisper 체크포인트 자신의 정규화기(`tokenizer.normalize` — 대소문자·
  구두점·숫자·축약형 + 철자 변형 1,740개)를 씁니다. 이미 로드하는 프로세서에서 나오므로
  **추가 의존성·추가 다운로드가 0** 입니다.
- 원시값은 `wer_raw` / `wer_raw_edits` / `wer_raw_words` 로 **함께 기록**됩니다.
  ⚠️ `llmtse/eval.py:43-45` 도 정규화를 하지 않으므로, 베이스라인 발표 수치와 직접
  대조할 때는 `wer_raw` 를 쓰세요.
- `wer_hyp` 에 Whisper 원본 출력이 남아 있어 **모델을 다시 돌리지 않고** 다른 정규화로
  재계산할 수 있습니다(5,000행 2시간 절약).
- 두 컬럼 모두 **corpus micro-WER**(총 edits / 총 words)로 집계합니다. 한쪽만 micro 로 하면
  정규화 효과와 집계 방식 차이가 섞여 비교가 무의미해집니다.

### 1-8. 🔴 `torchaudio.save` · `sf.write` 의 기본 형식은 PCM_16 이고 **조용히 클리핑**합니다

`encoding` / `subtype` 을 지정하지 않으면 float32 텐서를 넘겨도 **PCM_16 으로 저장되며
±1.0 밖이 잘립니다.** 경고도 없습니다. 실측:

```
torchaudio.save(path, wav.float(), sr)                  → PCM_16, peak 10.164 → 1.000  ★ 클리핑
torchaudio.save(..., encoding="PCM_F", bits_per_sample=32) → FLOAT,  peak 10.164 보존
sf.write(path, x, sr)                                   → PCM_16                       ★ 클리핑
sf.write(path, x, sr, subtype="FLOAT")                  → FLOAT   보존
```

TSE 모델 출력은 ±1 을 넘는 것이 정상이므로 이는 **DNSMOS만의 문제가 아니라 SI-SDR 을
망가뜨립니다.** LLM-TSE 예측 200행으로 시뮬레이션한 피해:

```
                FLOAT       PCM_16      차이
SI-SDR 평균      7.429       2.706    -4.723 dB
SI-SDR 중앙값    7.132       4.786    -2.346 dB
ESTOI 평균      0.7883      0.7115    -0.0768
악화폭 분위: 50%=2.4dB  75%=5.0dB  90%=9.6dB  99%=40.8dB
```

세 프로젝트 현황 — **모두 float32 로 통일되어야 합니다**:

| | 저장 코드 | 상태 |
|---|---|---|
| llmtse | `sf.write(..., subtype="FLOAT")` | ✓ 처음부터 올바름 |
| styletse | `torchaudio.save(..., encoding="PCM_F", bits_per_sample=32)` | ✓ 처음부터 올바름 |
| tpex | `torchaudio.save(path, wav...float(), sr)` | ✗ → **수정함** (`scripts/inference.py:_save_wav`, `pa_vocoder/scripts/inference.py`) |

- **확인법**: `python -c "import soundfile as sf; print(sf.info('pred.wav').subtype)"` →
  `FLOAT` 이어야 합니다. `PCM_16` 이면 그 산출물은 다시 뽑아야 합니다.
- 정답(`target`/`mixed`)이 PCM_16 인 것은 **무해합니다**: rms 0.028 에서 양자화 SNR 이
  70.1 dB 라, 이 데이터의 최대 측정값 50.9 dB 보다 19 dB 위입니다. 예측을 FLOAT 으로
  두는 이유는 정답과 형식을 맞추기 위해서가 아니라 **모델 출력이 ±1 을 넘기 때문**입니다.

---

## 2. 샘플레이트 규약

작업 SR 기본값은 **24000**(PORTE-v3 원본). 지표마다 계산 SR 이 다릅니다.

| 지표 | 계산 SR | 잘못된 SR 을 주면 |
|---|---|---|
| SI-SDR / SI-SDRi / input_si_sdr | 24 kHz native | (SR 무관) |
| STOI / ESTOI | 24 kHz native | (pystoi 가 내부 10 kHz 로 리샘플) |
| PESQ | 16 kHz | `ValueError` (P.862 는 8/16k만) |
| DNSMOS | 16 kHz | `ValueError` |
| WER (Whisper) | 16 kHz | `ValueError` |
| **spk_sim (ECAPA)** | 16 kHz | 🔴 **에러 없이 틀린 값** (§1-2) |

- 이 규약은 형제 프로젝트 `eval.py` 와 동일합니다(`llmtse/eval.py:9`:
  "SI-SDR/SI-SDRi/ESTOI = native 24k, PESQ = 16k").
- **STOI/ESTOI 를 미리 16 kHz 로 낮추지 마세요.** pystoi 가 어차피 10 kHz 로 내리므로
  `24k→16k→10k` 이중 리샘플이 되어 형제 숫자와 어긋납니다.
- 16 kHz 데이터를 평가할 때만 `--target-sr 16000` 을 쓰세요.
- 리샘플은 `tse_eval/audio.py:resample_np` 한 곳만 거칩니다. `mix` 는 어떤 16 kHz 지표도
  쓰지 않으므로 **일부러 리샘플하지 않습니다**.

---

## 3. 📐 지표 정의상의 함정

### 3-1. `input_si_sdr` 두 종류의 차이

둘 다 계산하는 값은 **똑같이 `SI-SDR(mix, ref)`** 입니다. 다른 것은 **어느 구간(창)까지
잘라서 계산하느냐** 하나뿐입니다.

| 열 | 창 | est 길이에 의존 |
|---|---|---|
| `input_si_sdr` | `min(len(est), len(ref), len(mix))` — **3-way** | ✅ |
| `input_si_sdr_pairwise` | `min(len(ref), len(mix))` — **est 무시** | ❌ |

**두 값이 갈라지는 조건은 딱 하나**: `len(est)` 가 셋 중 가장 짧을 때.
est 가 mix/ref 보다 길거나 같으면 두 창이 같아지므로 **두 값도 완전히 동일**합니다.

#### 예시 — `porte_v3_test_0000002` (overlap 0.8, ref/mix 모두 9.05초)

같은 샘플을 두 모델이 채점하는데 **출력 길이만** 다른 상황:

| | 모델 A (9.05초 전체 출력) | 모델 B (5.43초만 출력) |
|---|--:|--:|
| est 길이 | 9.05 s | 5.43 s |
| 3-way 창 | 9.05 s | **5.43 s** ← 짧아짐 |
| pairwise 창 | 9.05 s | 9.05 s |
| `input_si_sdr` | −5.9470 dB | **−3.6943 dB** |
| `input_si_sdr_pairwise` | −5.9470 dB | **−5.9470 dB** ← 그대로 |
| `si_sdr` / `si_sdri` | −5.9214 / +0.0257 | −3.7031 / −0.0088 |

`input_si_sdr` 은 **2.25 dB 움직였고**, `input_si_sdr_pairwise` 는 두 모델에서 동일합니다.
overlap 0.8 샘플이라 앞 5.43초와 뒤 3.6초의 간섭 정도가 달라서, 창이 짧아지면
"혼합이 얼마나 나쁜가"의 기준 자체가 바뀌기 때문입니다.

**이것이 리뷰어가 지적한 현상입니다.** `si_sdri − si_sdr = −input_si_sdr` 인데,
3-way 를 쓰면 이 값이 모델 출력 길이에 따라 달라집니다. 반면 pairwise 는 `mix`/`ref` 만으로
정해지므로 같은 샘플이면 세 프로젝트에서 같은 값이 나오고, 그것이 **하나의 파이프라인으로
채점했다는 직접 증거**가 됩니다.

#### asteroid 에만 있는 값인가?

- **`input_si_sdr`**: 이름은 asteroid `get_metrics` 가 붙여주는 것이지만
  (`metrics_list=['si_sdr']` → 반환 키 `['input_si_sdr', 'si_sdr']`),
  값은 그냥 `SI-SDR(mix, ref)` 입니다. 우리 native 로 직접 계산해도
  **4.4e-15** 이내로 같습니다. asteroid 가 있어야 얻는 값이 아닙니다.
- **`input_si_sdr_pairwise`**: **asteroid 에는 없고, asteroid 로는 만들 수 없습니다.**
  `get_metrics` 는 mix/ref/est 세 개가 같은 길이여야 하고 다르면 `AssertionError` 를 냅니다.
  즉 "est 를 무시하는 창"이라는 개념 자체가 없습니다. 이 열은 이 repo 의 추가 기능입니다.

> 두 백엔드는 길이가 어긋난 경우까지 포함해 동일한 값을 냅니다
> (`tests/test_backends.py::test_backends_agree_under_length_mismatch`).
> 예전에는 `--si-sdr-backend asteroid` 로 mix 가 ref 보다 짧은 행을 만나면
> `AssertionError` 로 그 행이 통째로 실패했는데, 우리 계약("길이가 다르면 자른다")에 맞게
> asteroid 에 넘기기 전에 정렬하도록 고쳤습니다.

### 3-2. WER 은 corpus micro 로 집계해야 합니다

행별 WER 을 평균(macro)하면 짧은 발화가 과대 가중됩니다. 논문 표에는
**전체 편집거리 합 / 전체 참조단어 합**(micro)을 씁니다.

실측 예: 1오류/100단어 + 5오류/5단어 → micro **5.7%**, 행별 평균 **50.5%**.

- 요약 CSV 의 `wer` 는 micro 로 집계됩니다(다른 지표는 nanmean).
- 행별 `wer` 열도 남으니 검수에 쓰세요. `wer_hyp` 에 Whisper 전사 결과가 들어갑니다.
- `target_sentence` 가 없는 행(**StyleTSE 매니페스트**)은 `nan` 이 되어 micro 에서 제외됩니다.
  `--source-csv` 로 PORTE-v3 를 join 하면 그 열을 가져올 수 있습니다.

### 3-3. SI-SDR 의 배수 불변성은 근사입니다

정의상으론 완전 불변이지만 `eps` 가 고정 절대항이라 부동소수점에서는 편차가 `1/gain²` 로
커집니다. 실측: gain 1000 → 2.7e-10, gain 0.01 → 2.7e-6, gain 0.001 → 2.7e-4.
표준 구현도 같은 식이라 동일한 특성이며, 버그가 아닙니다.

### 3-4. 무음 reference 는 `nan` (표준과 의도적으로 다름)

asteroid 는 무음 reference 에도 유한한 바닥값을 냅니다. 이 repo 는 `nan` 을 반환해
**해당 행이 평균에서 빠지게** 합니다. 등가성 테스트에서 이 케이스만 제외합니다.

### 3-5. 길이가 다르면 3-way trim

`min(len(est), len(ref), len(mix))` 로 세 신호를 함께 자릅니다(형제 프로젝트 규약).
asteroid 는 길이 불일치를 예외로 처리하지만 이 repo 는 자릅니다. 큰 차이가 나면 정렬 문제이니
원본을 확인하세요.

### 3-6. overlap 0.0 의 SI-SDR 50 dB 는 "분리 성능" 이 아니라 "복사 정확도" 입니다

LLM-TSE `best` 에서 SI-SDR 최댓값이 **50.77 dB** 였고 51 dB 초과는 0행이었습니다.
정상적인 측정값이지만 **분리 성능으로 읽으면 안 됩니다.**

먼저 측정 자체는 확실합니다:

- **eps 포화가 아닙니다**: `E(e_noise)=0.554` vs `eps=1e-8` — 7자리 차이. eps 없는 참값과
  적용값이 소수 3자리까지 같고, eps 가 만드는 천장은 128 dB 입니다.
- **독립 구현 3종이 일치합니다**: CSV / native / `pb_bss_eval` 직접 / `asteroid.get_metrics`
  네 경로가 `50.7674` 로 동일(최대차 8.6e-06).

문제는 해석입니다. overlap 0.0 은 두 화자가 **다른 시간대**에 말하므로, target 이 말하는
구간에는 방해음이 아예 없습니다. 120행 검증:

```
overlap 0.0, target 구간 안에서의 SI-SDR
  혼합 vs 정답 : median 102.7 dB · 120/120 행이 90 dB 초과   ← PCM_16 양자화 수준 = 비트 단위로 동일
  예측 vs 정답 : median  47.1 dB · max 50.9 dB
비교) overlap 1.0 의 혼합 vs 정답 (target 구간): median 0.1 dB  ← 여기선 진짜로 섞여 있음
```

즉 그 구간에서 모델이 할 일은 분리가 아니라 **그대로 통과시키기**뿐이고, 50.9 dB 는 이 모델의
통과 충실도입니다. 오차의 99.3% 도 target 구간에 있고 무음 구간 제거는 72 dB 로 거의 완벽합니다.

- **보고 시 권장**: overlap 별로 분해해 보여주세요. 전체 평균 하나로는 overlap 0.0 의
  통과 성능이 진짜 분리 성능을 가립니다(LLM-TSE 는 15.79 dB vs 나머지 4~7 dB).
- **반대쪽 −80.00 dB 바닥은 진짜 artifact 입니다**: 6행이 `pb_bss_eval` 의 eps 바닥에 걸려
  있고(정답과 상관계수 ~1e-6 = 완전 무상관), 전체 평균을 0.105 dB 끌어내립니다. 평균 7.004 vs
  중앙값 6.298 이므로 **중앙값을 함께 보고**하는 편이 정직합니다.
- **WER 이 overlap 과 반대 방향인 이유도 여기 있습니다**: overlap 0.0 에서 잔여 누설이
  target 대비 26 dB 아래라도, 무음 구간에 놓인 −26 dB 음성은 Whisper 가 또렷하게 전사합니다
  (log-mel 동적 범위 80 dB + 입력 정규화). 그래서 삽입 오류가 폭증합니다:
  삽입률 0.240(ov 0.0) → 0.048(ov 1.0), hyp/ref 단어 비율 1.045 → 0.795.
  **SI-SDR 과 WER 이 서로 다른 실패를 재고 있고 둘 다 맞습니다.**

---

## 4. 💾 이 서버(컨테이너) 특성

### 4-1. `$HOME=/home/work` 은 세션 종료 시 사라집니다

`/dev/loop6`, 49 GB. **영속 저장소는 아래 4곳뿐**입니다.

| 경로 | 용도 |
|---|---|
| `/home/work/my-code` | 코드 + conda env (miniforge3 가 여기 있어 `-n` 으로 만들어도 영속) |
| `/home/work/my-checkpoints` | 체크포인트 + **HF 모델 캐시** |
| `/home/work/my-datasets` | PORTE-v3 |
| `/home/work/my-outputs` | 추론 결과 |

→ 세션마다 `export HF_HOME=/home/work/my-checkpoints/hf_cache`.
안 하면 **Whisper 3 GB 를 매번 다시 받습니다**.

### 4-2. DNSMOS 가 이 컨테이너에서 특히 느립니다 (기본 가속 적용됨)

`/proc/cpuinfo` 는 호스트 224 core 를 보여주지만 cgroup 은 24개만 허용합니다.
`speechmos` 는 세션을 **옵션도 providers 도 없이** 만들어서 (a) 호스트 기준 스레드 풀을
잡고 코어 핀에 실패하며 (b) providers 미지정이라 onnxruntime 1.9+ 는 **CPU EP 만** 씁니다.

실측 (실제 PORTE-v3 오디오, 60행−20행 한계비용, 15,000 utt 환산):

| 설정 | 한계비용 | 15,000 utt | 무가속 대비 |
|---|--:|--:|--:|
| 무가속 | 1475 ms/utt | 369분 | 1.0배 |
| CPU (`intra_op=4`) | 550 ms/utt | 137분 | 2.7배 |
| **기본값 (CUDA)** | **175 ms/utt** | **44분** | **8.4배** |

- **`OMP_NUM_THREADS` 로는 해결되지 않습니다.** onnxruntime 의 intra-op 풀은 OpenMP 가 아니라
  자체 풀이라 `SessionOptions.intra_op_num_threads` 만 효과가 있습니다(실측 확인).
- CUDA 는 **1회성 초기화 약 40초**를 씁니다. `initialising DNSMOS on CUDA (~40 s, one time)`
  로그가 나오면 멈춘 게 아닙니다.
- 빨간 `pthread_setaffinity_np failed` 줄은 스레드 제한이 안 걸린 상태의 증상입니다(무해).

### 4-3. 설치 중 다른 터미널에서 같은 env 에 conda 를 쓰지 마세요

`conda env remove` 와 설치가 겹치면 **env 디렉토리가 도중에 사라집니다.** 증상이 원인과
전혀 무관해 보입니다:

```
ModuleNotFoundError: No module named 'urllib'
```

(pip 이 쓰던 Python 표준 라이브러리가 통째로 사라져서 나는 소리입니다.)
반쯤 지워진 빈 껍데기 디렉토리가 남으면 이후 단계가 이상하게 실패하므로
`conda env remove -n <name> -y` 로 확실히 정리하고 시작하세요.

### 4-4. pip 캐시가 기본적으로 꺼져 있습니다

NGC 이미지의 `/etc/pip.conf` 가 `no-cache-dir = true` 를 걸어놨습니다. 재시도마다 약 4 GB 를
다시 받습니다. 되살리는 방법은 **CLI 플래그 하나뿐**입니다:

```bash
pip install --cache-dir /home/work/my-code/pip_cache ...
```

⚠️ `PIP_CACHE_DIR` / `PIP_NO_CACHE_DIR` **환경변수는 안 먹힙니다**(직접 확인).

### 4-5. 예상 소요 시간 (5,000행 × 3시스템 = 15,000 utt)

| 지표 | 시간 |
|---|--:|
| SI-SDR 계열 / STOI+ESTOI / PESQ | 2분 / 28분 / 51분 |
| DNSMOS (CUDA 기본) | 44분 |
| spk_sim (기본 on) | 20분 |
| **기본 세트 합계** | **약 145분** |
| WER 추가 시 (기본 off) | **+178분** |

---

## 5. 📦 설치·의존성

### 5-1. `requirements/h200/h200_pip_freeze_*.txt` 는 그대로 설치할 수 없습니다

진단용 스냅샷입니다. 두 가지 이유로 `pip install -r` 이 실패합니다(실측):

1. conda 가 제공한 `packaging` 이 `@ file:///home/conda/...` 로 기록되고 그 경로는 없음 → `OSError`
2. editable 설치인 `tse-eval` 이 `tse-eval==0.1.0` 으로 기록됨 → PyPI 에 없는 이름

→ **재설치는 항상 `requirements/h200/requirements_h200.txt` + `pip install -e '.[test]'`.**
스냅샷은 `pip freeze --all` 로 뽑습니다(`--all` 이 없으면 **`setuptools` 가 빠지는데**
`setuptools<81` 은 중요한 핀이라 반드시 보여야 합니다).

### 5-2. `setuptools<81` 은 지우면 안 됩니다

`setuptools 81` 부터 `pkg_resources` 가 제거됐고, 오늘 `conda create python=3.10` 의 기본값은
**83.0.0** 입니다. 그런데 `torchmetrics 0.11.4`(asteroid 가 끌어옴)는 모듈 최상단에서
`pkg_resources` 를 import 합니다 → `import torchmetrics` 즉사.
형제 repo 3개가 모두 이 핀을 갖고 있습니다.

### 5-3. CPU/GPU onnxruntime 을 같이 깔지 마세요

별개 배포본이라 공존하면 충돌합니다. `pyproject.toml` 의 base dependencies 에서 빼고
extra 로 분리했습니다:

```bash
pip install -e '.[test]'        # H100/H200 (onnxruntime-gpu 는 requirements/h200/requirements_h200.txt 가 담당)
pip install -e '.[cpu,test]'    # 그 외
```

확인: `pip list | grep onnxruntime` → 한 줄만. `pip check` → `No broken requirements found.`

### 5-4. `onnxruntime-gpu` 는 py3.10 에서 1.20.2 가 상한

최신(1.28.x)은 `requires_python >= 3.11` 이라 설치 불가입니다.

### 5-5. Whisper 를 `snapshot_download` 로 받지 마세요

`openai/whisper-large-v3` repo 전체는 flax/tf/fp32 샤드까지 포함해 **24.7 GB** 입니다.
필요한 건 `model.safetensors` 3.09 GB 뿐이므로 `from_pretrained` 를 쓰세요
(`scripts/download_models.py` 가 그렇게 합니다 → 약 3.1 GB).

### 5-6. repo 의 `tse_eval.egg-info` 잔여물

이전 editable 설치 잔여물이 남아 있으면 **새 env 인데도** pip 이 아래 ERROR 를 냅니다:

```
ERROR: ... tse-eval 0.1.0 requires onnxruntime, which is not installed.
```

무해하지만 실패로 착각하게 됩니다. 재설치 전 `rm -rf tse_eval.egg-info build`.

### 5-7. speechbrain 은 `.to(device)` 로는 부족합니다

`EncoderClassifier.from_hparams(..., run_opts={"device": device})` 로 넘겨야 합니다.
나중에 `.to("cuda")` 를 부르면 가중치만 옮겨지고 `encode_batch()` 는 입력을 자기가 기억하는
`self.device`(기본 CPU)로 되돌려 보내서 이렇게 죽습니다:

```
Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same
```

### 5-8. `jiwer` 에는 `__version__` 속성이 없습니다

버전 확인은 `importlib.metadata.version("jiwer")` 로.

### 5-9. HF 캐시 용량을 `du` 로 재면 2배로 보입니다

HF 캐시는 실제 바이트를 `blobs/` 에 한 번만 두고 `snapshots/` 에는 심볼릭 링크만 둡니다.
링크를 따라가며 합산하면 같은 파일을 두 번 세어 3 GB 가 6.4 GB 로 보입니다.

---

## 6. 🔁 재현성 체크리스트

논문 숫자를 만들기 전 이 5개를 확인하세요.

- [ ] **`dnsmos_*` 열이 전부 `nan` 이 아닌지** (→ §1-1)
- [ ] 세 시스템의 `<output>_config.json` 에서 **`dnsmos.actual_providers` 가 동일**한지 (→ §1-5)
- [ ] 세 시스템의 sidecar 에서 **`target_sr` · `si_sdr_backend` · `metrics` 가 동일**한지
- [ ] `si_sdri == si_sdr − input_si_sdr` 이 성립하는지 (부동소수점 오차 내)
- [ ] 같은 샘플에 대해 **`input_si_sdr_pairwise` 가 세 시스템에서 동일**한지 (→ §3-1)

`<output>_config.json` 에는 지표 세트·백엔드·지표별 SR·모델 id·입력 경로·라이브러리 버전이
모두 들어갑니다. **논문에서 인용할 파일은 이것**입니다.

---

## 7. 🚫 일부러 넣지 않은 것

### SDR / SIR / SAR (`mir_eval.bss_eval_sources`)

- 발화당 약 1초 → 5,000행에 약 91분
- **blind source separation 전제**라 간섭 신호 없이는 SIR 이 `inf` 로 나와 TSE 에 부적합
- 값 자체는 샘플레이트에 의존하지 않음(시간영역 투영 기반)

### 풀밴드 무참조 지표 (NISQA 등)

PORTE-v3 가 24 kHz 원본이라 DNSMOS(16 kHz)가 8 kHz 이상을 버리는 건 사실이지만,
FlowTSE 를 포함한 24 kHz 생성계 TSE 논문들이 관행적으로 DNSMOS/OVRL 을 16 kHz 로 보고합니다.
비교 가능성을 위해 관행을 따랐습니다.

> 참고: FlowTSE(arXiv 2505.14465)의 "24 kHz" 는 **Vocos 보코더 요구사항**이며 평가 SR 이
> 아닙니다. 데이터셋(LibriSpeech/Libri2Mix/WHAM!)이 원본 16 kHz 이고 WER 은 Whisper-small,
> SIM 은 ResNet34-VoxCeleb2 로 둘 다 16 kHz 모델입니다.

### asteroid 를 런타임 의존성으로

기본 `native` 백엔드가 asteroid 와 **4e-14** 이내로 일치하고 오히려 **1.4배 빠릅니다**
(4.88 vs 6.80 ms/utt). asteroid 는 등가성 검증 오라클로만 `[test]` extra 에 둡니다
(22개 패키지, 다운로드 약 3 MB, 기존 핀은 불변).
`--si-sdr-backend asteroid` 로 언제든 대조할 수 있습니다.
