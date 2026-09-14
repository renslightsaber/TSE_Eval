# 📋 exp_reports — 채점 기록

> 📌 **목적**: `TSE_Eval` 로 채점한 결과를 **한자리에 모아** 프로젝트 간 대조를 가능하게 합니다.
> 여기는 학습 프로젝트가 아니라 **채점 도구**이므로, 이 폴더가 남기는 고유한 가치는
> ① 여섯 런이 **같은 조건**으로 채점됐다는 증거와 ② **프로젝트 간 비교**입니다.
> 📅 도입: 2026-08-31

---

## 📂 구조

```
exp_reports/
├── README.md          ← 이 문서 (채점 규약 · 현황)
├── comparison.md      ← ★ 프로젝트 간 대조 (정본)
├── llmtse/            ← llmtse/exp_reports/ 스냅샷
│   ├── README.md
│   ├── llmtse_baseline_v1/{config.yaml, report.md, report_best.md, report_last.md}
│   └── llmtse_baseline_v2/{config.yaml, report.md, report_best.md, report_last.md}
└── styletse/          ← styletse/exp_reports/ 스냅샷
    ├── README.md
    ├── stage1/run_stage1.md
    └── styletse_v1/{config.yaml, report.md, report_best.md, report_last.md}
```

> ⚠️ `llmtse/` · `styletse/` 는 **각 프로젝트 리포트의 사본**입니다. **원본이 정본**이고,
> 원본이 갱신되면 여기도 다시 복사해야 합니다. 사본인 이유는 채점 결과를 해석하려면
> 그 실험의 맥락이 함께 있어야 하는데, TSE_Eval 만 클론한 사람은 원본에 접근할 수
> 없기 때문입니다.
>
> | 위치 | 원본 |
> |---|---|
> | `llmtse/` | `/home/work/my-code/llmtse/exp_reports/` |
> | `styletse/` | `/home/work/my-code/styletse/exp_reports/` |

---

## 🔬 채점 규약 (여섯 런 공통)

| 항목 | 값 |
|---|---|
| 데이터 | PORTE-v3 test **5,000 전수** |
| 조건 | **text-only (no enrollment)** — 표에 반드시 명시 |
| SI-SDR 구현 | `asteroid` (= `pb_bss_eval`) |
| 작업 SR | 24000 (PESQ·DNSMOS·WER·spk_sim 은 내부 16 kHz) |
| 지표 | 13개 + 동반 5개 (`dnsmos_*_clipped` 4 · `wer_raw`) |
| 계층화 축 | `overlap_ratio` · `prompt_category` · `same_gender` · `first_speak` |
| DNSMOS | CUDA EP · **`rms_-26dbov`** 레벨 정규화 |
| WER | whisper-large-v3 · **`whisper_english`** 텍스트 정규화 · **corpus micro** |
| provenance | 각 `eval/*_config.json` (sidecar) |

**★ 여섯 런의 sidecar 8항목이 전부 일치합니다** — 근거와 재확인 방법은
[`comparison.md`](./comparison.md) §1.

---

## 📊 채점 현황

| 프로젝트 | 런 | 채점일 | SI-SDR | 중앙값 | WER | 상세 |
|---|---|---|--:|--:|--:|---|
| LLM-TSE | v1 best | 08-07 | 7.004 | 6.298 | 0.5234 | [report_best](./llmtse/llmtse_baseline_v1/report_best.md) |
| LLM-TSE | v1 last | 08-07 | 6.823 | 6.442 | 0.5185 | [report_last](./llmtse/llmtse_baseline_v1/report_last.md) |
| LLM-TSE | v2 best | 08-11 | 7.290 | 6.306 | 0.5457 | [report_best](./llmtse/llmtse_baseline_v2/report_best.md) |
| LLM-TSE | v2 last | 08-11 | 7.279 | 6.634 | 0.5047 | [report_last](./llmtse/llmtse_baseline_v2/report_last.md) |
| **StyleTSE** | v1 best | 08-31 | **14.236** | **17.595** | **0.3691** | [report_best](./styletse/styletse_v1/report_best.md) |
| StyleTSE | v1 last | 08-31 | 13.887 | 17.553 | 0.3770 | [report_last](./styletse/styletse_v1/report_last.md) |
| TPEX | — | **미채점** | — | — | — | 추론 대기 |

전부 test 5,000 전수 · text-only. 대조는 [`comparison.md`](./comparison.md).

---

## 🔁 채점 방법

```bash
cd /home/work/my-code/TSE_Eval

bash scripts/eval_llmtse.sh   --check-only   # 선행 검사 (약 10초)
bash scripts/eval_llmtse.sh                  # LLM-TSE (VERSION 기본 v2)
bash scripts/eval_styletse.sh                # StyleTSE (VERSION 기본 v1)

python scripts/check_eval_done.py llmtse v1  # 완료·출처 확인
python scripts/check_eval_done.py styletse
```

새 프로젝트는 래퍼 하나(약 30줄)만 추가하면 됩니다 — 엔진 `scripts/eval_tse.sh` 는
프로젝트를 모릅니다. 자세한 것은 각 스크립트 상단 주석.

---

## ⚠️ 이 결과를 읽을 때 반드시 알아야 할 것

| 항목 | 요약 | 근거 |
|---|---|---|
| **중앙값을 함께 보세요** | 평균과 중앙값의 관계가 프로젝트마다 반대입니다 (LLM-TSE 평균>중앙값, StyleTSE 평균<중앙값) | [`comparison.md`](./comparison.md) §4.2 |
| **`−80 dB` 는 실제 성능이 아닙니다** | `pb_bss_eval` 의 eps 바닥입니다 | [`../CAVEATS.md`](../CAVEATS.md) §3-6 |
| **overlap 0.0 의 높은 SI-SDR 은 "분리"가 아닙니다** | 그 구간 혼합이 정답과 사실상 동일 → **통과** 과제 | [`../CAVEATS.md`](../CAVEATS.md) §3-6 |
| **WER 이 overlap 과 반대 방향인 이유** | 무음 구간 잔여 누설의 **삽입 오류** (실측 확인) | [`../CAVEATS.md`](../CAVEATS.md) §3-6 |
| **DNSMOS·WER 은 정규화값을 보고하세요** | `_clipped` · `_raw` 동반 컬럼은 다른 도구와 대조할 때만 | [`../CAVEATS.md`](../CAVEATS.md) §1-6·§1-7 |
| **best 체크포인트가 최고가 아닙니다** | 세 실험 모두 best 선정이 test 우위를 보장하지 못함 | 각 리포트 §7.5 / §3 |
