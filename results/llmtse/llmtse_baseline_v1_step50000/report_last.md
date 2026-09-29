> 📋 **사본** — 이 파일은 `exp_reports/llmtse/llmtse_baseline_v1/report_last.md` 의 사본입니다. repo 안의 **정본은 [`exp_reports/`](../../../exp_reports/README.md)**, 최종 원본은 `/home/work/my-code/llmtse/exp_reports/` 입니다.
> 같은 폴더의 채점 산출물(`*.csv` · `*_config.json` · `eval.log`)이 이 보고서가 설명하는 수치의 근거입니다.

# ⏹️ llmtse_baseline_v1 — **last** 체크포인트 평가 (step 50,000)

> | | |
> |---|---|
> | 📄 **학습 기록(공통)** | [`report.md`](./report.md) — 설정·환경·학습 곡선·두 체크포인트 **비교**는 전부 그쪽입니다 |
> | 🔀 **반대 체크포인트** | [`report_best.md`](../../../exp_reports/llmtse/llmtse_baseline_v1/report_best.md) (step 49,000) |
> | ⏹️ **선정 기준** | **학습 종료 시점** (epoch 50 완주). 성능으로 고른 것이 아닙니다 |
> | 📦 **체크포인트** | `/home/work/my-checkpoints/llmtse/llmtse_baseline_v1/step_00050000/` (accelerate `save_state`) |
> | 🔤 **추론 `--run_name`** | `llmtse_baseline_v1_step50000` |
> | 🔬 **상태** | ✅ **추론·채점 완료** (TSE_Eval, 2026-08-07) |
> | 📊 **핵심 수치** | SI-SDR **6.823** · ESTOI **0.7799** · PESQ **2.075** · WER **0.5185** |
> | 📅 **갱신** | 2026-08-11 · 🔖 커밋 `304ebfd` |

---

## 1. 이 문서의 범위

**`llmtse_baseline_v1` 학습에서 나온 두 체크포인트 중 `last` 하나의 추론 결과만** 담습니다.

### 왜 last 도 추론했나

`val/loss` 기준 best 선정이 **dev 5,000 중 50 발화(1%)** 로만 이뤄져 신뢰도가 낮기 때문입니다([`report.md`](./report.md) §6.3).
best 가 정말 나은지 확인하려면 **두 체크포인트를 같은 test 5,000 에 태워 비교**해야 했습니다.

| 알고 싶은 것 | 어디에 있나 |
|---|---|
| 이 체크포인트의 test 지표·층화 | **이 문서** |
| 학습 설정·논문 대조·환경·실행 명령 | [`report.md`](./report.md) §1–§4 |
| 학습 곡선·정체와 돌파·best 선정 메커니즘 | [`report.md`](./report.md) §5–§6 |
| 추론 절차·매니페스트 스키마·overlap bin 정의 | [`report.md`](./report.md) §7.1–§7.4 |
| **best vs last 비교** (페어 통계·승률) | [`report.md`](./report.md) §7.5 |
| `order` 카테고리가 실패한 원인 | [`report.md`](./report.md) §7.6 |
| 결론·다음 실험 과제·알려진 제약 | [`report.md`](./report.md) §8–§9 |

> ⚠️ 이 문서에는 **다른 체크포인트와의 비교를 쓰지 않습니다.** 비교는 `report.md` §7.5 가 정본입니다.

---

## 2. 체크포인트 · 산출물

### 2.1 식별

| 항목 | 값 |
|---|---|
| step | **50,000** (epoch 50, 학습 종료) |
| 선정 기준 | **없음** — 마지막 step 이라서 |
| `val/loss` @ 이 시점 | −6.1556 (best 인 −7.4661 보다 나쁨) |
| 체크포인트 | `<ckpt>/llmtse_baseline_v1/step_00050000/` (accelerate `save_state` 디렉토리) |
| `--run_name` | `llmtse_baseline_v1_step50000` |
| 예측 wav | `<out>/llmtse_baseline_v1_step50000/preds/<file_id>.wav` (5,000개, 24 kHz float32, 약 5.1 GB) |
| 매니페스트 | `<out>/llmtse_baseline_v1_step50000/inference_test.csv` (5,000행 × 19열) |
| 추론 조건 | full-length(`segment=None`), batch=1, bf16, **text-only (no enrollment)** |

`<ckpt>` = `/home/work/my-checkpoints/llmtse` · `<out>` = `/home/work/my-outputs/llmtse`

### 2.2 이 체크포인트만 재현하는 명령

```bash
source /home/work/my-code/miniforge3/etc/profile.d/conda.sh
conda activate llmtse
cd /home/work/my-code/llmtse

python inference.py \
    --run_name llmtse_baseline_v1_step50000 \
    --split    test \
    --config   /home/work/my-checkpoints/llmtse/llmtse_baseline_v1/config_used.yaml \
    --ckpt     /home/work/my-checkpoints/llmtse/llmtse_baseline_v1/step_00050000
```

> 📌 **`--config` 를 반드시 명시해야 합니다.** `--run_name` 이 체크포인트 디렉토리 이름과 달라
> `config_used.yaml` 자동 탐색이 실패하고, 상대경로 `configs/config.yaml` 로 폴백합니다(`inference.py:217`).
> v1 은 `text_pooling` 키가 없는 legacy 라 이 폴백이 **가드에 걸려 중단**됩니다(현재 repo 기본값은 `last4_mean`).
>
> 📌 `--ckpt` 는 `.pth` 가 아니라 **디렉토리**입니다. `inference.py` 의 `_load_state_dict()` 가
> `save_state` 디렉토리 / `.pth` / `.safetensors` 세 형태를 모두 처리합니다.

---

## 3. 전체 (PORTE-v3 test 5,000) 지표

**채점 완료** — `TSE_Eval` (2026-08-07). 산출물: `<out>/llmtse_baseline_v1_step50000/eval/llmtse_last{,_summary,_config.json}.csv`

| 지표 | **값** | 계산 기준 |
|---|---|---|
| **SI-SDR** | **6.823** dB | asteroid, native 24 kHz |
| **SI-SDRi** | **6.780** dB | 입력 SI-SDR 0.042 dB 대비 |
| STOI | 0.8682 | pystoi, 24 kHz |
| **ESTOI** | **0.7799** | pystoi `extended=True`, 24 kHz |
| **PESQ** | **2.075** | 16 kHz resample (wideband) |
| **WER** | **0.5185** | Whisper-large-v3, corpus micro |
| spk_sim | 0.7239 | 화자 임베딩 코사인 |
| DNSMOS OVRL | 2.938 | |

> ⚠️ **조건: text-only (no enrollment).**
> 📌 best 와의 **차이**는 [`report.md`](./report.md) §7.5 가 정본입니다. 여기에 비교를 적지 마십시오.

---

## 4. 산출물 검증 (2026-08-07, test 5,000 중 무작위 500 표본)

### (1) 구조 — 통과 ✅

5,000행 / 19열 / 결측 0 / `file_id` 중복 0 / 경로 4종 전부 실재 / 표본 24 kHz FLOAT / 0바이트 파일 0.

### (2) val 값과의 일치

_(해당 없음 — `last` 는 best 선정 대상이 아니라 대조 조건이므로, validation 시점과 맞춰볼 기준이 없습니다.)_

참고로 이 시점의 dev-50 validation 은 SI-SDR **4.365** 였고, test 500 표본에서는 **6.219** 였습니다.
dev-50 이 이 체크포인트를 **과소평가**했다는 뜻이며, 이것이 best/last 격차가 dev 에서 부풀려 보인 주된 이유입니다([`report.md`](./report.md) §7.5).

### (3) 이상치 분포

| 항목 | 값 |
|---|---|
| SI-SDR < 0 dB | **20.8%** |
| SI-SDR < −30 dB (`clamp` 영역, [`report.md`](./report.md) §6.2) | **6개** |

> best(3개)보다 `< −30 dB` 대실패 샘플이 **2배** 많습니다. 후반 `val/loss` 왜곡의 실체가 test 에도 존재합니다.

---

## 5. overlap 층화

| overlap | n | SI-SDR | SI-SDRi | ESTOI | PESQ | WER | spk_sim |
|---|---|---|---|---|---|---|---|
| 0.0 | 834 | **15.257** | 15.268 | 0.9354 | 3.990 | 0.5199 | 0.7858 |
| 0.2 | 834 | 6.632 | 6.412 | 0.8894 | 2.484 | 0.5793 | 0.7439 |
| 0.4 | 833 | 5.418 | 5.472 | 0.8094 | 1.823 | 0.5503 | 0.7378 |
| 0.6 | 833 | 5.211 | 5.242 | 0.7421 | 1.530 | 0.4922 | 0.7290 |
| 0.8 | 833 | 4.525 | 4.451 | 0.6770 | 1.359 | 0.5076 | 0.6999 |
| 1.0 | 833 | **3.883** | 3.828 | 0.6259 | 1.264 | 0.4592 | 0.6471 |
| **ALL** | **5000** | **6.823** | **6.780** | **0.7799** | **2.075** | **0.5185** | **0.7239** |

best 와 동일하게 **overlap 이 커질수록 단조 감소**합니다(15.257 → 3.883 dB).

### 추가 축

| 축 | 그룹 | n | SI-SDR | ESTOI | PESQ | WER | spk_sim |
|---|---|---|---|---|---|---|---|
| same_gender | **diff** (이성) | 3635 | **8.857** | 0.7850 | 2.108 | 0.4474 | 0.7618 |
| | **same** (동성) | 1365 | **1.407** | 0.7663 | 1.989 | 0.7049 | 0.6231 |
| first_speak | infer(간섭이 먼저) | 2442 | 6.879 | 0.7878 | 2.162 | 0.5862 | 0.7383 |
| | target(타깃이 먼저) | 2558 | 6.769 | 0.7723 | 1.993 | 0.4513 | 0.7102 |

> ★ 동성 화자 조건에서 **7.5 dB 를 잃습니다**(8.857 → 1.407). best 와 같은 구조적 약점입니다.

---

## 6. `prompt_category` 별 성능

test 5,000 전수, TSE_Eval.

| prompt_category | n | SI-SDR | SI-SDRi | ESTOI | PESQ | **WER** | **spk_sim** |
|---|---|---|---|---|---|---|---|
| gender_female | 875 | **13.249** | 13.477 | 0.8090 | 2.229 | **0.319** | **0.861** |
| gender_male | 875 | 12.516 | 12.108 | 0.8148 | 2.251 | 0.333 | 0.855 |
| pitch_higher | 750 | 9.022 | 9.057 | 0.7838 | 2.060 | 0.460 | 0.778 |
| pitch_lower | 750 | 5.272 | 5.295 | 0.7619 | 2.034 | 0.508 | 0.715 |
| order_later | 875 | 1.111 | 0.820 | 0.7665 | 2.083 | **0.838** | 0.600 |
| order_first | 875 | **−0.141** | 0.039 | 0.7414 | 1.788 | **0.626** | **0.541** |

**축 요약**:

| 축 | n | SI-SDR | 화자 혼동률 | WER | spk_sim |
|---|---|---|---|---|---|
| gender | 1750 | **12.882** | **1.8%** | 0.326 | 0.858 |
| pitch | 1500 | 7.147 | 14.6% | 0.484 | 0.747 |
| **order** | 1750 | **0.485** | **42.3%** | **0.732** | **0.571** |

> ★ **`order` 실패가 체크포인트와 무관함이 확인됐습니다.** best 0.622 dB / 혼동 43.8% 대 last 0.485 dB / 42.3% —
> 사실상 같습니다. `order_first` 는 두 체크포인트 모두 **SI-SDR 음수**입니다.
> 이는 실패 원인이 체크포인트 선택이 아니라 **학습 설정(6초 random crop)** 이라는 진단과 일치합니다
> ([`report.md`](./report.md) §7.6).

---

## 7. 이 체크포인트에 한정된 관찰

1. **dev-50 이 이 체크포인트를 크게 과소평가했습니다.** dev-50 에서 4.365 dB 였지만 test 5,000 전수에서는 **6.823 dB** 입니다 — **2.5 dB 과소평가**. 50 발화 표본이 얼마나 불안정한지 보여주는 직접 증거입니다.
2. **대실패 샘플이 best 보다 많습니다.** SI-SDR `< −30 dB` 가 **61개 대 43개**. 평균 격차의 상당 부분이 이 꼬리에서 발생합니다.
3. **일부 지표는 오히려 best 보다 낫습니다.** `spk_sim` 0.7239 (best 0.7166), WER 0.5185 (best 0.5234), gender_female SI-SDR 13.249 (best 13.100). 즉 **어느 체크포인트가 낫다고 단정하기 어렵습니다** — 상세는 [`report.md`](./report.md) §7.5.
4. **논문 표에는 쓰지 않습니다.** 이 체크포인트는 best 선정이 신뢰할 만한지 검증하기 위한 **대조군**입니다.

---

## 8. 상호 참조

| 알고 싶은 것 | 문서 |
|---|---|
| 학습 설정 · 논문 대조 | [`report.md`](./report.md) §2 |
| 학습 곡선 · 정체와 돌파 | [`report.md`](./report.md) §6.1 |
| best 선정 기준의 결함 | [`report.md`](./report.md) §6.2 |
| **best ↔ last 비교** | [`report.md`](./report.md) §7.5 |
| `order` 실패 원인 (6초 crop) | [`report.md`](./report.md) §7.6 |
| 반대 체크포인트 | [`report_best.md`](../../../exp_reports/llmtse/llmtse_baseline_v1/report_best.md) |
| 후속 실험 (`text_pooling` 정렬) | [`../llmtse_baseline_v2/report.md`](../../../exp_reports/llmtse/llmtse_baseline_v2/report.md) |
