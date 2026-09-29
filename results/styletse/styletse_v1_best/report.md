> 📋 **사본** — 이 파일은 `exp_reports/styletse/styletse_v1/report.md` 의 사본입니다. repo 안의 **정본은 [`exp_reports/`](../../../exp_reports/README.md)**, 최종 원본은 `/home/work/my-code/styletse/exp_reports/` 입니다.
> 같은 폴더의 채점 산출물(`*.csv` · `*_config.json` · `eval.log`)이 이 보고서가 설명하는 수치의 근거입니다.

# 🎨 `styletse_v1` — Stage-2 평가 기록

> 📅 **작성** 2026-08-31 · 🖥️ NIPA H200 (sm_90) 1장
> ✅ **상태: 추론·채점 완료** — TSE_Eval, 2026-08-31
> 📊 **핵심 수치** (best, test 5,000 전수, text-only)
>   SI-SDR **14.236** dB · ESTOI **0.8615** · PESQ **2.895** · WER **0.3691** · spk_sim **0.8251**
> 🔁 **Stage-1 기록**: [`../stage1/run_stage1.md`](../../../exp_reports/styletse/stage1/run_stage1.md)

---

## 📚 이 실험의 문서 구성

| 문서 | 담는 것 |
|---|---|
| **이 문서** | 채점 절차 · bin 정의 · **best ↔ last 비교**(정본) · 결론 |
| [`report_best.md`](./report_best.md) | Stage-2 **best** 체크포인트 하나의 지표·층화·분석 |
| [`report_last.md`](../../../exp_reports/styletse/styletse_v1/report_last.md) | Stage-2 **last** 체크포인트 하나의 지표·층화·분석 |
| [`config.yaml`](./config.yaml) | 채점 시점의 `configs/config.yaml` 스냅샷 |

> ⚠️ **범위**: 이 문서 묶음은 **평가 결과만** 담습니다. Stage-2 학습 곡선·best 선정
> 메커니즘은 아직 기록하지 않았습니다(Stage-1 은 [`../stage1/run_stage1.md`](../../../exp_reports/styletse/stage1/run_stage1.md)).
> 🔗 **StyleTSE ↔ LLM-TSE 대조의 정본**은 [`TSE_Eval/exp_reports/comparison.md`](../../../exp_reports/comparison.md) 입니다.

---

## 1. 두 체크포인트

| | best | last |
|---|---|---|
| 시점 | S2 epoch **60.9** (누적 139.8) | S2 epoch **80.9** (누적 159.8, 조기종료) |
| 선정 기준 | dev `mixed` 기준 최고 | 없음 — 학습 종료 시점 |
| 체크포인트 | `<ckpt>/styletse_v1/stage2/best` | `<ckpt>/styletse_v1/stage2/step_00057834` |
| 매니페스트 | `<out>/styletse_v1_best/manifest.csv` | `<out>/styletse_v1_last/manifest.csv` |
| 채점 산출물 | `<out>/styletse_v1_best/eval/styletse_v1_best*` | `<out>/styletse_v1_last/eval/styletse_v1_last*` |
| 채점 소요 | 1h 29m 36s | 1h 41m 23s |
| **상세** | **[report_best.md](./report_best.md)** | **[report_last.md](../../../exp_reports/styletse/styletse_v1/report_last.md)** |

`<ckpt>` = `/home/work/my-checkpoints/styletse` · `<out>` = `/home/work/my-outputs/styletse`

> ★ **선정 기준과 평가 조건이 다릅니다.** Stage-2 dev 는 `mixed`
> (audio-text : text-only : audio-only = **2 : 2 : 1**)로 두었고([`config.yaml`](./config.yaml) `stage2.dev`),
> 평가는 **text-only** 입니다. 즉 판정값의 **60%가 text-only 평가에서 쓰이지 않는
> enrollment cue** 로 만들어졌습니다. config 주석이 Stage-1 실측으로 이 위험을 미리 적어
> 두었습니다 — "epoch 62→79 구간에서 audio-text 14.77→16.72(상승)인데 text-only
> 10.29→7.86(하락)". §3 의 결과가 그 우려와 일치합니다.
>
> ✅ **다만 이 run 에서는 그 불일치가 실제로 다른 체크포인트를 고르게 만들지 않았습니다.**
> TensorBoard `val_by_task/loss_text_only` 의 **전수 최솟값도 step 43,554**(−14.8921,
> 검증 81 회 중)로 `best/` 와 같은 step 입니다. mixed 로 골라도 text-only 로 골랐을 step 과
> 동일했습니다. best↔last 의 test 격차(§3)는 선정 기준보다 **화자 혼동 꼬리**로 설명하는
> 편이 실측과 부합합니다(→ [report_best.md](./report_best.md) 분석 7). (2026-09-07 감사)

---

## 1-1. 추론 파이프라인에서 발견된 결함 2건 (2026-09-07 감사)

두 체크포인트에 공통이며, **둘 다 SI-SDR 을 부풀리는 방향이 아닙니다.**

### 🟠 저장된 예측 wav 의 실효 정밀도가 float32 가 아니라 **bfloat16**

`inference.py:346` 이 `torch.autocast("cuda", dtype=torch.bfloat16)` 아래에서 모델을 돌리고
`:349` 에서 `est.float()` 로 승격하지만, **디코더 출력이 이미 bf16 격자에 고정**된 뒤입니다.

```
예측 wav 가 bf16 으로 정확히 표현되는 샘플 비율 : 1.0000  (100%)
원본 mixture (대조군)                          : 0.5478
bf16 상대정밀도 2⁻⁸ → SNR 상한 48.2 dB
실제 per-sample SI-SDR 상위 꼬리 : p99 47.93 · p99.9 50.48 · max 51.00  ← 상한에 붙음
```

`inference.py:358` 의 주석 *"float32 WAV (no 16-bit clipping/quantization)"* 는 컨테이너
기준으로는 맞지만 **내용물 기준으로는 사실과 다릅니다.** 영향은 **과소평가 방향**이며
(상위 꼬리가 48 dB 에서 잘림) 14.236 을 부풀리지 않습니다. 다음 run 부터 `run_inference` 를
autocast 밖 fp32 로 돌릴 것을 권합니다.

### 🟠 예측 파형 진폭이 target 대비 **약 11.4배**

```
최적 스케일 α (pred ≈ α·target) 중앙값 11.42 | peak 중앙값 3.50 | peak > 1.0 인 샘플 30/30
```

원인은 정상입니다 — 학습 loss 가 SI-SDR(스케일 불변)이고 masknet 출력 활성이 무계
`nn.ReLU` 라 게인을 맞출 유인이 없습니다. **SI-SDR·STOI·ESTOI 는 스케일 불변이라 영향
없음.** 다만 (a) int16 으로 변환해 청취·배포하면 전량 클리핑되고(PCM_F 라 디스크상은 안전),
(b) **WER 은 Whisper 특징추출이 진폭 정규화를 하지 않으므로 영향을 완전히 배제하기
어렵습니다.** DNSMOS 는 TSE_Eval 이 `rms_-26dbov` 정규화를 걸어 방어됩니다.

---

## 2. 채점 절차

```bash
cd /home/work/my-code/TSE_Eval
bash scripts/eval_styletse.sh --check-only    # 선행 검사 (약 10초)
bash scripts/eval_styletse.sh                 # best → last 순차 채점
python scripts/check_eval_done.py styletse    # 완료·출처 확인
```

| 항목 | 값 |
|---|---|
| SI-SDR 백엔드 | `asteroid` (= `pb_bss_eval`) |
| 작업 SR | 24000 (PESQ·DNSMOS·WER·spk_sim 은 내부 16 kHz) |
| 지표 | 13개 + 동반 5개 (`dnsmos_*_clipped` 4 · `wer_raw`) |
| 계층화 축 | `overlap_ratio` · `prompt_category` · `same_gender` · `first_speak` |
| DNSMOS | CUDA EP · `rms_-26dbov` 정규화 |
| WER | whisper-large-v3 · `whisper_english` 정규화 · corpus micro |
| 평가 조건 | **text-only (no enrollment)** |

### 2.1 매니페스트 스키마 (19열, 3-프로젝트 정렬)

`file_id · pred_path · mixed_path · target_path · interference_path · sample_rate · task ·
overlap_ratio · prompt · prompt_category · target_gender · infer_gender · first_speak ·
target_start_time · target_end_time · overlap_start_time · overlap_end_time · length_sec ·
target_sentence`

LLM-TSE 와 **같은 19열**이며 열 순서만 다릅니다(자동 감지는 이름 기반이라 무관).
StyleTSE 고유 규약: 파일명이 `manifest.csv`, 예측 wav 가 `<file_id>_est.wav`.

### 2.2 overlap bin

| overlap | 0.0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 |
|---|---|---|---|---|---|---|
| n | 834 | 834 | 833 | 833 | 833 | 833 |

> 📌 매니페스트의 `overlap_ratio` 가 float32 에서 온 값이라 `0.2000000029802322` 처럼
> 저장돼 있습니다. 요약 CSV 의 그룹 라벨에 그대로 나타나지만 **버킷은 정확히 6개**이고
> 집계에는 영향이 없습니다. 문서의 표는 읽기 좋게 반올림해 적었습니다.

---

## 3. ★ best ↔ last 비교 — 평균과 승률이 반대를 가리킨다

**전수 채점 확정치** (test 5,000).

| 지표 | best | last | 차이 |
|---|---|---|---|
| SI-SDR | **14.236** | 13.887 | +0.348 |
| **중앙값 SI-SDR** | 17.595 | **17.553** | −0.042 (last 근소 우위) |
| ESTOI | **0.8615** | 0.8596 | +0.0019 |
| PESQ | 2.895 | **2.899** | **last 우위** |
| STOI | **0.9131** | 0.9110 | +0.0021 |
| WER (낮을수록 좋음) | **0.3691** | 0.3770 | best 우위 |
| spk_sim | **0.8251** | 0.8221 | +0.0030 |
| DNSMOS OVRL | 3.124 | **3.131** | **last 우위** |
| SI-SDR `< −30 dB` | **210** | 212 | 사실상 동일 |

짝지은 비교(paired, n=5,000, 같은 `file_id`, SI-SDR):

| 통계 | 값 |
|---|---|
| 평균 차이 | **+0.3482 dB** (SE 0.0538) |
| **중앙값 차이** | **−0.0431 dB** ← last 우위 |
| paired t-test | t = 6.47, **p < 1e-9** |
| best 승 / last 승 | **1536 / 3464** (best **30.7%**) |
| `│차이│ > 10 dB` 인 행 | 152 (best 우위 **109** · last 우위 43) |
| 꼬리 제외(`│차이│ ≤ 10 dB`) 평균차 | +0.1240 |

**해석 — 두 지표가 서로 다른 답을 냅니다.**

1. **행 단위로는 last 가 69.3% 에서 이깁니다.** 중앙값 차이도 −0.043 dB 로 last 쪽입니다.
   전형적인 발화에서는 last 가 근소하게 낫습니다.
2. **그런데 평균은 best 가 +0.348 dB 앞섭니다.** 원인은 꼬리입니다 — 10 dB 이상 벌어지는
   152 행 중 **best 가 크게 이기는 쪽이 109, last 가 크게 이기는 쪽이 43** 입니다.
   best 는 자주 지지만 질 때 조금 지고, 이길 때 크게 이깁니다.
3. **PESQ·DNSMOS 는 last 가 낫습니다.** "best 가 전반적으로 우월하다"고 말할 수 없습니다.
4. ★ **best 선정이 이득을 주지 못한 구조적 이유**가 §1 에 있습니다 — 선정은 `mixed` dev
   (60%가 enrollment cue)로 했고 평가는 text-only 입니다. 두 조건이 어긋난 상태에서
   "best" 라는 이름은 text-only 성능을 보장하지 않습니다.

> 📌 논문 표에는 관례대로 **best** 를 씁니다. 다만 **"best 가 last 보다 낫다"는 서술은
> 근거가 약합니다** — 승률 30.7%, 중앙값 역전, PESQ·DNSMOS 역전입니다.
> p < 1e-9 는 "평균 차이가 0 이 아니다"만 말하고, 그 차이는 152 행의 꼬리에서 옵니다.

---

## 4. 결론

| 질문 | 답 |
|---|---|
| 산출물이 신뢰할 만한가 | ✅ 두 런 모두 채점기 자체 검증 6항목 통과 · 구조 전수 검증 통과 |
| text-only 조건의 도달 성능 | **SI-SDR 14.236 dB · ESTOI 0.8615 · PESQ 2.895 · WER 0.3691** (best) |
| best 선정이 유효했나 | ⚠️ **아니오** — 승률 30.7%, 중앙값 역전. 선정(`mixed`)과 평가(`text-only`) 조건 불일치가 유력한 원인 |
| 어느 조건에서 무너지나 | **overlap 1.0** — 14.37 → **4.63 dB** 절벽. 그 외 구간은 14~17 dB 유지 |
| `order` 프롬프트는 작동하나 | ✅ **작동** — 화자 혼동률 16.6%, overlap 0.6~0.8 에서는 **0~1%** |

### 다음 과제

1. **체크포인트 선정 기준을 평가 조건과 맞출 것.** dev 를 `text-only` 로 두거나, 최소한
   `val_by_task/text_only` 곡선으로 별도 후보를 뽑아 test 로 대조해야 합니다.
   현재는 판정값의 60%가 평가에 쓰이지 않는 cue 입니다.
2. **overlap 1.0 붕괴의 원인 규명.** 전 구간에서 14 dB 이상인데 이 한 지점만 4.6 dB 입니다.
   `< −30 dB` 대실패 210 행 중 **78 행(37%)이 이 구간**에 몰려 있습니다.
3. **동성 화자 격차 5.5 dB** (이성 15.74 vs 동성 10.23). `< −30 dB` 행의 38% 가 동성 쌍
   (전체 비율 27%)이라 대실패의 상당수가 여기서 나옵니다.
