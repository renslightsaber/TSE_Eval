# 📊 results — TSE_Eval 로 채점한 성능 평가 결과물

이 폴더는 **TSE_Eval 로 실제 채점한 산출물과 그 해석 문서**를 프로젝트별·런별로 모아 둔 곳입니다.
repo 하나만 받아도 숫자를 다시 계산하거나 검증할 수 있도록, per-row CSV 까지 그대로 복사했습니다.

| 프로젝트 | 런 | 상태 |
|---|---|---|
| [LLM-TSE](./llmtse/) | 4개 (v1·v2 × best·last) | ✅ 채점 완료 |
| [StyleTSE](./styletse/) | 2개 (v1 × best·last) | ✅ 채점 완료 |
| [TPEX](./tpex/) | — | ⬜ **아직 채점하지 않음** (빈 폴더) |

> 프로젝트 간 대조는 [`../exp_reports/comparison.md`](../exp_reports/comparison.md) 가 정본입니다.
> 채점 정책·함정은 [`../CAVEATS.md`](../CAVEATS.md) 를 먼저 보세요.

---

## 전체 요약 (overlap 전 구간 `ALL`, 각 5,000행)

| 런 | SI-SDR ↑ | SI-SDRi ↑ | ESTOI ↑ | PESQ ↑ | DNSMOS ovrl ↑ | spk_sim ↑ | WER ↓ |
|---|--:|--:|--:|--:|--:|--:|--:|
| llmtse v1 best | 7.004 | 6.961 | 0.7885 | 2.099 | 2.947 | 0.7166 | 0.5234 |
| llmtse v1 last | 6.823 | 6.780 | 0.7799 | 2.075 | 2.938 | 0.7239 | 0.5185 |
| llmtse v2 best | 7.290 | 7.248 | 0.8019 | 2.166 | 2.946 | 0.7056 | 0.5457 |
| llmtse v2 last | 7.279 | 7.236 | 0.7893 | 2.124 | 2.937 | 0.7238 | 0.5047 |
| **styletse v1 best** | **14.236** | **14.193** | **0.8615** | **2.895** | 3.124 | **0.8251** | **0.3691** |
| styletse v1 last | 13.887 | 13.845 | 0.8596 | 2.899 | **3.131** | 0.8221 | 0.3770 |

이 값들은 각 런의 `*_summary.csv` 에서 `axis=overlap_ratio · group=ALL` 행을 그대로 옮긴 것입니다.
층화(overlap · prompt_category · same_gender · first_speak)별 값은 같은 파일 안에 있습니다.

---

## 폴더 구조

```
results/
├── llmtse/<run>/          # 런 하나가 폴더 하나
│   ├── <tag>.csv                  # per-row 지표 (5,000행 × 18지표 + 원본 컬럼)
│   ├── <tag>_summary.csv          # long-format 다축 요약 (axis · group · n · 지표)
│   ├── <tag>_config.json          # ★ provenance sidecar — 논문에 인용할 파일
│   ├── eval.log                   # 채점 실행 로그
│   ├── report_best.md             # 이 런(체크포인트)의 평가 보고서
│   ├── report.md                  # 그 학습 버전의 종합 보고서 (best·last 비교 포함)
│   └── config.yaml                # 그 학습 버전의 학습 설정
├── styletse/<run>/        # 같은 구성 (best 런에는 *_masked.csv 도 포함)
└── tpex/                  # 비어 있음 — 채점하면 같은 구조로 채웁니다
```

## 파일 읽는 순서

1. `report_best.md` / `report_last.md` — 그 체크포인트가 무엇을 잘/못했는지
2. `<tag>_summary.csv` — 층화별 수치 (표로 인용할 값)
3. `<tag>_config.json` — 어떤 조건에서 잰 값인지 (SR · si_sdr 백엔드 · DNSMOS providers · 라이브러리 버전)
4. `<tag>.csv` — 행 단위 재분석이 필요할 때

## 원본 위치 (정본)

| 대상 | 원본 |
|---|---|
| 채점 산출물 | `/home/work/my-outputs/<proj>/<run>/eval/` (H200) · `elu:/home/heiscold/nipa_h200/<proj>/eval/<run>/` (연구실) |
| MD 보고서 | `/home/work/my-code/<proj>/exp_reports/` → repo 안 정본 사본은 [`../exp_reports/`](../exp_reports/README.md) |

여기 있는 파일은 **사본**입니다. 원본이 갱신되면 이 폴더도 다시 복사해 맞춰야 합니다
(2026-09-29 기준 채점 산출물은 원본과 바이트 동일함을 `cmp` 로 확인).

## 다시 채점하려면

```bash
conda activate tseeval
export HF_HOME=/home/work/my-checkpoints/hf_cache
bash scripts/eval_llmtse.sh      # VERSION=v1|v2 · ONLY=best|last
bash scripts/eval_styletse.sh    # VERSION=v1
```

산출물은 추론 CSV 옆 `eval/` 에 생깁니다. TPEX 는 매니페스트만 있으면
`PROJECT=tpex RUNS="best|<manifest>.csv" bash scripts/eval_tse.sh` 로 같은 방식으로 채점합니다.
