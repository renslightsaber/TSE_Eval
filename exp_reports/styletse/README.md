> 📋 **사본** — 원본은 `/home/work/my-code/styletse/exp_reports/` 입니다. **원본이 정본**이며 여기는 채점 결과 해석에 필요한 맥락을 함께 두려고 복사한 것입니다.
> 프로젝트 간 대조는 [`comparison.md`](../comparison.md) 가 정본입니다.

# 🧪 exp_reports — StyleTSE 실험 기록

> 📌 **목적**: `run_name` 단위로 **설정 스냅샷 + 학습/평가 보고서**를 남겨, 나중에
> "이 결과가 어떤 설정에서 나왔는지"를 코드 히스토리 없이도 재구성할 수 있게 합니다.
> 📅 도입: 2026-08-31 (규약은 [llmtse/exp_reports](../llmtse/README.md) 와 동일)

---

## 📂 구조

```
exp_reports/
├── README.md                    ← 이 문서 (규약)
├── stage1/
│   └── run_stage1.md            ← Stage-1(audio-text) 학습 리포트
└── styletse_v1/                 ← Stage-2 실험 1회
    ├── config.yaml              ← 설정 스냅샷
    ├── report.md                ← 공통 + 두 체크포인트 비교
    ├── report_best.md           ← Stage-2 best 체크포인트 평가
    └── report_last.md           ← Stage-2 last 체크포인트 평가
```

### 📑 왜 3개로 나누는가

학습 1회에서 **두 개의 체크포인트**를 평가합니다 — dev 기준 **best** 와 종료 시점 **last**.
한 파일에 둘을 2열 표로 넣으면 어느 수치가 어느 체크포인트 것인지 흐려집니다.

> **배분 원칙: `report.md` = 공통이거나 비교인 것 · 자식 문서 = 정확히 한 체크포인트에만 참인 것**

| 종류 | 위치 |
|---|---|
| 채점 절차·매니페스트 스키마·bin 정의 | `report.md` |
| **best ↔ last 비교** (페어 통계·승률) | `report.md` — **정본** |
| 한 체크포인트의 전체 지표·층화·카테고리 | 각 자식 문서 |
| 그 체크포인트에만 해당하는 관찰·분석 | 각 자식 문서 |

⚠️ 자식 문서에는 **다른 체크포인트와의 비교를 쓰지 않습니다.**

---

## 🔗 프로젝트 간 비교는 여기가 아닙니다

StyleTSE ↔ LLM-TSE ↔ TPEX 대조의 **정본은 채점 도구 쪽**입니다:

> [`TSE_Eval/exp_reports/comparison.md`](../comparison.md)

세 프로젝트를 **하나의 채점 코드 경로**로 재는 것이 그 도구의 목적이고, 여섯 런이 같은
조건으로 채점됐다는 증거(sidecar 대조)도 그쪽에 있습니다. 여기에 중복해 적지 않습니다.

---

## 📊 채점 규약 (2026-08-31 기준)

| 항목 | 값 |
|---|---|
| 채점기 | `TSE_Eval` — `bash scripts/eval_styletse.sh` |
| SI-SDR 구현 | `asteroid` (= `pb_bss_eval`) |
| 작업 샘플레이트 | 24 kHz (PESQ·DNSMOS·WER·spk_sim 은 내부에서 16 kHz) |
| DNSMOS 레벨 정규화 | `rms_-26dbov` (ITU-T P.56) |
| WER 텍스트 정규화 | `whisper_english` (체크포인트 자체 정규화기) |
| 평가 조건 | **text-only (no enrollment)** — 표에 반드시 명시 |
| provenance | 각 `eval/*_config.json` (sidecar) |

> ⚠️ **동반 컬럼**: `wer_raw`(텍스트 정규화 이전) · `dnsmos_*_clipped`(레벨 정규화 이전)이
> 함께 기록됩니다. 보고에는 정규화값을 쓰고, 다른 도구의 비정규화 수치와 대조할 때만
> 동반 컬럼을 씁니다. 근거: [`TSE_Eval/CAVEATS.md`](../../CAVEATS.md) §1-6·§1-7.
