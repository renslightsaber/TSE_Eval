# LLM-TSE — 채점 결과물

학습 버전 2개(v1 · v2) × 체크포인트 2개(best · last) = **런 4개**. 각 런당 5,000행 채점.

| 런 폴더 | 체크포인트 | SI-SDR ↑ | SI-SDRi ↑ | ESTOI ↑ | PESQ ↑ | spk_sim ↑ | WER ↓ | 채점일 |
|---|---|--:|--:|--:|--:|--:|--:|---|
| [`llmtse_baseline_v1`](./llmtse_baseline_v1/) | v1 best (step 49,000) | 7.004 | 6.961 | 0.7885 | 2.099 | 0.7166 | 0.5234 | 2026-08-07 |
| [`llmtse_baseline_v1_step50000`](./llmtse_baseline_v1_step50000/) | v1 last (step 50,000) | 6.823 | 6.780 | 0.7799 | 2.075 | 0.7239 | 0.5185 | 2026-08-07 |
| [`llmtse_baseline_v2`](./llmtse_baseline_v2/) | v2 best | **7.290** | **7.248** | **0.8019** | **2.166** | 0.7056 | 0.5457 | 2026-08-11 |
| [`llmtse_baseline_v2_step50000`](./llmtse_baseline_v2_step50000/) | v2 last | 7.279 | 7.236 | 0.7893 | 2.124 | 0.7238 | **0.5047** | 2026-08-11 |

`ALL` 행(overlap 전 구간) 기준이며, 층화별 값은 각 폴더의 `*_summary.csv` 에 있습니다.

- 폴더마다 `report_best.md` 또는 `report_last.md`(그 체크포인트 보고서) + `report.md`(그 버전 종합 보고서) + `config.yaml`(학습 설정)이 함께 있습니다.
- v1 ↔ v2 비교는 각 `report.md`, StyleTSE 와의 대조는 [`../../exp_reports/comparison.md`](../../exp_reports/comparison.md).
- 파일 구성과 재채점 방법은 [`../README.md`](../README.md).
