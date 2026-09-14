> 📋 **사본** — 원본은 `/home/work/my-code/llmtse/exp_reports/` 입니다. **원본이 정본**이며 여기는 채점 결과 해석에 필요한 맥락을 함께 두려고 복사한 것입니다.
> 프로젝트 간 대조는 [`comparison.md`](../comparison.md) 가 정본입니다.

# 🧪 exp_reports — 학습 실험 기록

> 📌 **목적**: `run_name` 단위로 **학습 설정 스냅샷 + 학습/평가 보고서**를 남겨, 나중에 "이 결과가 어떤 설정에서 나왔는지"를 코드 히스토리 없이도 재구성할 수 있게 합니다.
> 📅 도입: 2026-08-05

---

## 📂 구조

```
exp_reports/
├── README.md                       ← 이 문서 (규약)
└── <run_name>/                     ← 학습 1회 = 디렉토리 1개
    ├── config.yaml                 ← ★ 학습 시작 시점의 configs/config.yaml 스냅샷
    ├── report.md                   ← ★ 학습 보고서 (공통) + 체크포인트 비교
    ├── report_best.md              ← ★ best 체크포인트 하나의 평가
    ├── report_last.md              ← ★ last 체크포인트 하나의 평가
    └── (선택) config_used.yaml     ← 학습이 남긴 실제 사용 config 사본
```

### 📑 왜 3개로 나누는가 (2026-08-11 도입)

학습 1회에서 보통 **두 개의 체크포인트**를 평가합니다 — `val/loss` 기준 **best** 와 학습 종료 시점 **last**.
(best 선정이 dev 의 1% 로만 이뤄져 신뢰도가 낮기 때문에 대조가 필요합니다.)
한 파일에 두 체크포인트 결과를 2열 표로 넣으면 문서가 비대해지고 어느 수치가 어느 체크포인트 것인지 흐려집니다.

> **배분 원칙: `report.md` = 공통이거나 비교인 것 · 자식 문서 = 정확히 한 체크포인트에만 참인 것**

| 종류 | 예 | 위치 |
|---|---|---|
| 공통 | 설정·환경·학습 곡선·진행 기록·매니페스트 스키마·overlap bin 정의 | `report.md` |
| **비교** | best↔last 페어 통계·승률, 이전 실험과의 대조 | `report.md` |
| 개별 | 그 체크포인트의 test 지표·층화·카테고리별 성능·산출물 검증 | `report_best.md` / `report_last.md` |

자식 문서 4종(v1·v2 × best·last)은 **동일한 뼈대**를 씁니다 — `§1 범위 / §2 체크포인트·산출물 / §3 전체 지표 /
§4 산출물 검증 / §5 overlap 층화 / §6 prompt_category / §7 관찰 / §8 상호참조`. 구조가 같아야
`diff` 로 실험 간 비교가 됩니다.

⚠️ **`report.md` 의 § 번호는 외부 문서가 인용합니다.** 번호를 바꾸거나 절을 다른 파일로 옮기기 전에
`grep -rn "report\.md.*§" --include="*.md" .` 로 인용처를 확인하십시오. 인용 대상 절에는
HTML 주석으로 동결 표시를 남겨 두었습니다.

`<run_name>` 은 `accelerate launch train.py --run_name <run_name>` 에 넘긴 값과 **정확히 동일**해야 합니다. 그래야 아래 세 곳이 한 이름으로 연결됩니다.

| 산출물 | 경로 |
|---|---|
| 학습 체크포인트 | `/home/work/my-checkpoints/llmtse/<run_name>/` |
| 추론·평가 산출물 | `/home/work/my-outputs/llmtse/<run_name>/` |
| TensorBoard | `runs/<run_name>/tb/` |
| **실험 기록** | **`exp_reports/<run_name>/`** ← 여기 |

---

## 🚀 새 실험을 시작할 때

```bash
RUN=llmtse_baseline_v2          # 원하는 run_name

mkdir -p exp_reports/$RUN
cp configs/config.yaml exp_reports/$RUN/config.yaml

# 리포트 3종은 직전 실험 것을 복사해 값을 비우는 방식이 가장 안전합니다
# (뼈대가 같아야 실험 간 diff 비교가 됩니다)
PREV=llmtse_baseline_v2
cp exp_reports/$PREV/report.md      exp_reports/$RUN/report.md
cp exp_reports/$PREV/report_best.md exp_reports/$RUN/report_best.md
cp exp_reports/$PREV/report_last.md exp_reports/$RUN/report_last.md
# → 수치를 전부 _(학습 미시작)_ 으로 비우고, §1 에 이번 실험의 목적과 **사전 예측**을 적을 것

git add exp_reports/$RUN && git commit -m "exp($RUN): 학습 설정 스냅샷"
```

> ⚠️ **config 스냅샷은 학습을 시작하기 전에** 뜹니다. 학습 도중 `configs/config.yaml` 을 고치면 스냅샷과 실제가 어긋나므로, 설정을 바꿔야 하면 **새 `run_name` 으로 새 실험**을 만드세요.

---

## ✍️ `report.md` 에 담을 것

| 섹션 | 내용 |
|---|---|
| **개요** | 목적, 무엇을 검증/달성하려는 실험인가, 이전 실험과 무엇이 다른가 |
| **설정 요약** | 핵심 하이퍼파라미터 + **논문/팀원 코드와의 대조**(따랐는가·왜 다른가) |
| **환경** | GPU, 커밋 해시, conda env, 데이터셋 |
| **실행 명령** | 복사해 붙일 수 있는 실제 명령 (재현용) |
| **진행 기록** | 시작/종료 시각, 중단·재개 이력, 이상 징후 |
| **학습 결과** | val/loss 곡선 요약, best step, 과적합 시점, `train/nan_skips` |
| **평가 결과** | SI-SDR / SI-SDRi / ESTOI / PESQ / WER, overlap 층화 |
| **관찰·결론** | 다음 실험에 넘길 교훈 |

빈 칸은 `_(학습 중)_` 처럼 **표시해 두고 채웁니다.** 비워두면 "측정 안 함"인지 "아직 안 됨"인지 구분이 안 됩니다.

---

## 📚 관련 문서

| 문서 | 내용 |
|---|---|
| `CLAUDE.md` | 프로젝트 정본 conventions |
| `docs/reports/llmtse_pretrain_readiness_2026-08-05.md` | 학습 착수 준비 검수 · 논문 원문 대조(§12) |
| `docs/guides/training_guide.md` | 학습 실행 가이드 |
| `docs/guides/vram_requirements.md` | 카드별 batch / VRAM |

> 💡 `docs/reports/` 는 **코드·환경 검수** 기록, `exp_reports/` 는 **학습 실행** 기록입니다. 성격이 다르니 섞지 마세요.
