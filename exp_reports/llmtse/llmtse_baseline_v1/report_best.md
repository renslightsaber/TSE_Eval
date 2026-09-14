> 📋 **사본** — 원본은 `/home/work/my-code/llmtse/exp_reports/` 입니다. **원본이 정본**이며 여기는 채점 결과 해석에 필요한 맥락을 함께 두려고 복사한 것입니다.
> 프로젝트 간 대조는 [`comparison.md`](../../comparison.md) 가 정본입니다.

# 🏆 llmtse_baseline_v1 — **best** 체크포인트 평가 (step 49,000)

> | | |
> |---|---|
> | 📄 **학습 기록(공통)** | [`report.md`](./report.md) — 설정·환경·학습 곡선·두 체크포인트 **비교**는 전부 그쪽입니다 |
> | 🔀 **반대 체크포인트** | [`report_last.md`](./report_last.md) (step 50,000) |
> | 🏆 **선정 기준** | `val/loss` 최소 = **−7.4661** @ step 49,000 (epoch 48) |
> | 📦 **체크포인트** | `/home/work/my-checkpoints/llmtse/llmtse_baseline_v1/best_model.pth` |
> | 🔤 **추론 `--run_name`** | `llmtse_baseline_v1` |
> | 🔬 **상태** | ✅ **추론·채점 완료** (TSE_Eval, 2026-08-07) |
> | 📊 **핵심 수치** | SI-SDR **7.004** · ESTOI **0.7885** · PESQ **2.099** · WER **0.5234** |
> | 📅 **갱신** | 2026-08-11 · 🔖 커밋 `304ebfd` |

---

## 1. 이 문서의 범위

**`llmtse_baseline_v1` 학습에서 나온 두 체크포인트 중 `best` 하나의 추론 결과만** 담습니다.

| 알고 싶은 것 | 어디에 있나 |
|---|---|
| 이 체크포인트의 test 지표·층화·카테고리별 성능 | **이 문서** |
| 학습 설정·논문 대조·환경·실행 명령 | [`report.md`](./report.md) §1–§4 |
| 학습 곡선·정체와 돌파·best 선정 메커니즘 | [`report.md`](./report.md) §5–§6 |
| 추론 절차·매니페스트 스키마·overlap bin 정의 | [`report.md`](./report.md) §7.1–§7.4 |
| **best vs last 비교** (페어 통계·승률) | [`report.md`](./report.md) §7.5 |
| `order` 카테고리가 실패한 **원인** | [`report.md`](./report.md) §7.6 |
| 결론·다음 실험 과제·알려진 제약 | [`report.md`](./report.md) §8–§9 |

> ⚠️ 이 문서에는 **다른 체크포인트와의 비교를 쓰지 않습니다.** 비교는 `report.md` §7.5 가 정본입니다.

---

## 2. 체크포인트 · 산출물

### 2.1 식별

| 항목 | 값 |
|---|---|
| step | **49,000** (epoch 48) |
| 선정 기준 | `val/loss` 최소 (**−7.4661**) |
| ⚠️ 주의 | `val/loss` 는 `clamp(max=30)` 이 걸린 값이라 SI-SDR 기준 최고와 다릅니다 — **SI-SDR 기준 최고는 step 46,000**(7.150)이었으나 `ckpt_keep: 3` 이 이미 삭제했습니다. 상세는 [`report.md`](./report.md) §6.2 |
| 체크포인트 | `<ckpt>/llmtse_baseline_v1/best_model.pth` (13.28 GB) |
| `--run_name` | `llmtse_baseline_v1` |
| 예측 wav | `<out>/llmtse_baseline_v1/preds/<file_id>.wav` (5,000개, 24 kHz float32, 약 5.1 GB) |
| 매니페스트 | `<out>/llmtse_baseline_v1/inference_test.csv` (5,000행 × 19열) |
| 추론 조건 | full-length(`segment=None`), batch=1, bf16, **text-only (no enrollment)** |

`<ckpt>` = `/home/work/my-checkpoints/llmtse` · `<out>` = `/home/work/my-outputs/llmtse`

### 2.2 이 체크포인트만 재현하는 명령

```bash
source /home/work/my-code/miniforge3/etc/profile.d/conda.sh
conda activate llmtse
cd /home/work/my-code/llmtse

python inference.py --run_name llmtse_baseline_v1 --split test
```

`--config` · `--ckpt` 를 생략하면 `<ckpt>/llmtse_baseline_v1/` 의 `config_used.yaml` 과 `best_model.pth` 를 자동으로 찾습니다.

> ⚠️ **`--limit` 을 붙이지 마십시오.** 매니페스트를 그 행 수로 **잘라서 덮어씁니다**(`df.to_csv` 전체 쓰기).
> 실제로 이 매니페스트가 3행으로 잘렸던 사고가 있었습니다. 부분 확인이 필요하면 `--run_name _verify_xxx` 로 임시 이름을 쓰세요.

---

## 3. 전체 (PORTE-v3 test 5,000) 지표

**채점 완료** — `TSE_Eval` (2026-08-07). 산출물: `<out>/llmtse_baseline_v1/eval/llmtse_best{,_summary,_config.json}.csv`

| 지표 | **값** | 계산 기준 |
|---|---|---|
| **SI-SDR** | **7.004** dB | asteroid, native 24 kHz |
| **SI-SDRi** | **6.961** dB | 입력 SI-SDR 0.042 dB 대비 |
| STOI | 0.8756 | pystoi, 24 kHz |
| **ESTOI** | **0.7885** | pystoi `extended=True`, 24 kHz |
| **PESQ** | **2.099** | 16 kHz resample (wideband) |
| **WER** | **0.5234** | Whisper-large-v3, corpus micro |
| spk_sim | 0.7166 | 화자 임베딩 코사인 |
| DNSMOS OVRL | 2.947 | (SIG 3.400 / BAK 3.652 / P808 3.547) |

> ⚠️ **조건: text-only (no enrollment).** 표에 반드시 명시할 것.
> ⚠️ **WER 52.3% 는 매우 높습니다.** 카테고리별로 보면 gender 33% vs order 63~84% 로 갈립니다(§6) —
> 전체 평균만 인용하면 오해를 부릅니다.
> 📌 논문 표에 쓸 provenance 파일은 `eval/llmtse_best_config.json` 입니다.

---

## 4. 산출물 검증 (2026-08-07, test 5,000 중 무작위 500 표본)

### (1) 구조 — 통과 ✅

5,000행 / 19열 / 결측 0 / `file_id` 중복 0 / 경로 4종 전부 실재 / 표본 24 kHz FLOAT / 0바이트 파일 0.

### (2) val 값과의 일치 — **실제 모델 출력임의 근거**

| 항목 | test 500 | dev 50 (step 49,000 validation) | 판정 |
|---|---|---|---|
| SI-SDR | **6.873** | 6.478 | 0.4 dB 이내 ✅ |
| SI-SDRi | **6.768** | 6.662 | 0.1 dB 이내 ✅ |

validation 이 **full-length·batch1** 이라 추론과 조건이 완전히 동일해 직접 비교가 가능합니다([`report.md`](./report.md) §6.3).
**overlap 층화가 단조 감소**하는 것도 mixture 복사나 경로 오매칭이면 나올 수 없는 패턴입니다.

### (3) 이상치 분포

| 항목 | 값 |
|---|---|
| SI-SDR < 0 dB | **20.2%** |
| SI-SDR < −30 dB (`clamp` 영역, [`report.md`](./report.md) §6.2) | **3개** |

---

## 5. overlap 층화

| overlap | n | SI-SDR | SI-SDRi | ESTOI | PESQ | WER | spk_sim |
|---|---|---|---|---|---|---|---|
| 0.0 | 834 | **15.788** | 15.799 | 0.9471 | 4.022 | 0.5344 | 0.7882 |
| 0.2 | 834 | 6.893 | 6.672 | 0.9009 | 2.503 | 0.5912 | 0.7441 |
| 0.4 | 833 | 5.450 | 5.503 | 0.8162 | 1.865 | 0.5634 | 0.7249 |
| 0.6 | 833 | 5.274 | 5.305 | 0.7487 | 1.554 | 0.5091 | 0.7156 |
| 0.8 | 833 | 4.616 | 4.542 | 0.6842 | 1.371 | 0.4822 | 0.6891 |
| 1.0 | 833 | **3.991** | 3.935 | 0.6333 | 1.277 | 0.4584 | 0.6375 |
| **ALL** | **5000** | **7.004** | **6.961** | **0.7885** | **2.099** | **0.5234** | **0.7166** |

**overlap 이 커질수록 모든 지표가 단조 감소**합니다 — 15.788 → 3.991 dB (**11.8 dB 낙차**).
ESTOI 0.947 → 0.633, PESQ 4.02 → 1.28 도 같은 방향입니다. 겹침이 없을 때는 사실상 완벽에 가깝고,
완전 겹침에서는 분리가 거의 되지 않습니다.

> ⚠️ WER 만 반대 방향입니다(0.534 → 0.458). overlap 0% 구간은 target 발화가 짧고 무음이 길어
> Whisper 가 오히려 불리한 조건이라, WER 을 overlap 축으로 해석할 때는 주의가 필요합니다.

### 추가 축

| 축 | 그룹 | n | SI-SDR | ESTOI | PESQ | WER | spk_sim |
|---|---|---|---|---|---|---|---|
| same_gender | **diff** (이성) | 3635 | **8.998** | 0.7933 | 2.124 | 0.4549 | 0.7521 |
| | **same** (동성) | 1365 | **1.692** | 0.7756 | 2.033 | 0.7031 | 0.6220 |
| first_speak | infer(간섭이 먼저) | 2442 | 7.098 | 0.7977 | 2.183 | 0.5991 | 0.7333 |
| | target(타깃이 먼저) | 2558 | 6.914 | 0.7796 | 2.018 | 0.4482 | 0.7006 |

> ★ **동성 화자 조건에서 7.3 dB 를 잃습니다**(8.998 → 1.692). 이는 §6 의 `order` 실패와 겹치는
> 현상입니다 — 성별로 구분되지 않는 쌍에서 모델이 화자를 특정하지 못합니다.

---

## 6. `prompt_category` 별 성능 (test 1,500 표본)

전체 평균 7.0 dB 가 **숨기고 있는 사실**입니다. (test 5,000 전수, TSE_Eval)

| prompt_category | n | SI-SDR | SI-SDRi | ESTOI | PESQ | **WER** | **spk_sim** |
|---|---|---|---|---|---|---|---|
| gender_female | 875 | **13.100** | 13.329 | 0.8159 | 2.246 | **0.331** | **0.846** |
| gender_male | 875 | 12.571 | 12.163 | 0.8211 | 2.258 | 0.348 | 0.849 |
| pitch_higher | 750 | 9.287 | 9.321 | 0.7967 | 2.116 | 0.480 | 0.749 |
| pitch_lower | 750 | 6.003 | 6.026 | 0.7645 | 2.019 | 0.482 | 0.723 |
| order_later | 875 | 1.292 | 1.000 | 0.7807 | 2.111 | **0.841** | 0.602 |
| order_first | 875 | **−0.049** | 0.131 | 0.7496 | 1.835 | **0.630** | **0.536** |

**축 요약** (화자 혼동률 = 예측이 target 보다 간섭 화자에 더 가까운 비율):

| 축 | n | SI-SDR | 화자 혼동률 | WER | spk_sim |
|---|---|---|---|---|---|
| gender | 1750 | **12.836** | **1.5%** | 0.339 | 0.848 |
| pitch | 1500 | 7.645 | 13.7% | 0.481 | 0.736 |
| **order** | 1750 | **0.622** | **43.8%** ← 동전던지기 | **0.735** | **0.569** |

> ★ **`order_first` 는 SI-SDR 이 음수(−0.049 dB)** 입니다 — 분리하지 않은 mixture 를 그대로 내보내는 것보다
> 못하다는 뜻입니다. `spk_sim` 0.536 (gender 0.848 대비)과 WER 63~84% 가 같은 이야기를 합니다:
> **모델이 어느 화자를 뽑을지 결정하지 못합니다.**
>
> 📌 **실패 원인 분석(6초 crop)** 은 [`report.md`](./report.md) §7.6 이 정본입니다.

---

## 7. 이 체크포인트에 한정된 관찰

1. **산출물은 신뢰할 수 있습니다.** test 5,000 전수 SI-SDR 7.004 가 같은 조건의 dev-50 validation(6.478)과 0.5 dB 이내이고, overlap 단조 감소(15.8 → 4.0 dB)도 물리적으로 타당한 패턴입니다.
2. **`best_model.pth` 는 "생존한 체크포인트 중" 최선입니다.** `val/loss` 로 골랐기 때문에 진짜 최고(step 46,000, dev SI-SDR 7.150)를 놓쳤지만, 그 체크포인트는 이미 삭제된 뒤였습니다([`report.md`](./report.md) §6.2).
3. **전체 평균 7.004 dB 는 두 개의 다른 모드가 섞인 값입니다.** gender 12.836 dB 와 order 0.622 dB 는 같은 모델의 성능이라고 보기 어려울 만큼 다릅니다. 동성 화자 조건(1.692 dB)도 마찬가지입니다.
4. **논문 표에는 이 체크포인트를 씁니다.** 단 `order` 카테고리 실패(전체의 1/3)와 **WER 52.3%** 를 함께 명시해야 합니다([`report.md`](./report.md) §9).

---

## 8. 상호 참조

| 알고 싶은 것 | 문서 |
|---|---|
| 학습 설정 · 논문 대조 | [`report.md`](./report.md) §2 |
| 학습 곡선 · 정체와 돌파 | [`report.md`](./report.md) §6.1 |
| best 선정 기준의 결함 | [`report.md`](./report.md) §6.2 |
| **best ↔ last 비교** | [`report.md`](./report.md) §7.5 |
| `order` 실패 원인 (6초 crop) | [`report.md`](./report.md) §7.6 |
| 반대 체크포인트 | [`report_last.md`](./report_last.md) |
| 후속 실험 (`text_pooling` 정렬) | [`../llmtse_baseline_v2/report.md`](../llmtse_baseline_v2/report.md) |
