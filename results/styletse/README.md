# StyleTSE — 채점 결과물

학습 버전 1개(v1) × 체크포인트 2개(best · last) = **런 2개**. 각 런당 5,000행 채점.

| 런 폴더 | 체크포인트 | SI-SDR ↑ | SI-SDRi ↑ | ESTOI ↑ | PESQ ↑ | DNSMOS ovrl ↑ | spk_sim ↑ | WER ↓ | 채점일 |
|---|---|--:|--:|--:|--:|--:|--:|--:|---|
| [`styletse_v1_best`](./styletse_v1_best/) | v1 best | **14.236** | **14.193** | **0.8615** | 2.895 | 3.124 | **0.8251** | **0.3691** | 2026-08-31 |
| [`styletse_v1_last`](./styletse_v1_last/) | v1 last | 13.887 | 13.845 | 0.8596 | **2.899** | **3.131** | 0.8221 | 0.3770 | 2026-08-31 |

`ALL` 행(overlap 전 구간) 기준이며, 층화별 값은 각 폴더의 `*_summary.csv` 에 있습니다.

- 폴더마다 `report_best.md` 또는 `report_last.md` + `report.md`(v1 종합 보고서) + `config.yaml`(학습 설정)이 함께 있습니다.
- `styletse_v1_best_masked.csv` 는 best 런에만 있는 추가 산출물입니다.
- Stage-1 학습 기록은 [`../../exp_reports/styletse/stage1/run_stage1.md`](../../exp_reports/styletse/stage1/run_stage1.md), LLM-TSE 와의 대조는 [`../../exp_reports/comparison.md`](../../exp_reports/comparison.md).
- 파일 구성과 재채점 방법은 [`../README.md`](../README.md).
