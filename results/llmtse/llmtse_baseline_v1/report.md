> 📋 **사본** — 이 파일은 `exp_reports/llmtse/llmtse_baseline_v1/report.md` 의 사본입니다. repo 안의 **정본은 [`exp_reports/`](../../../exp_reports/README.md)**, 최종 원본은 `/home/work/my-code/llmtse/exp_reports/` 입니다.
> 같은 폴더의 채점 산출물(`*.csv` · `*_config.json` · `eval.log`)이 이 보고서가 설명하는 수치의 근거입니다.

# 🧪 `llmtse_baseline_v1` — LLM-TSE baseline 첫 본 학습

> 📅 **작성** 2026-08-05 · **갱신** 2026-08-07 · 🖥️ **NIPA H200 (sm_90) 1장** · 🔖 **코드 커밋** `c6fcb09`
> ✅ **상태: 학습 완료** — **2026-08-05 11:29:10 → 2026-08-07 06:53 KST** (50,000 step / epoch 50, **43.35시간**)
> 🏆 **best `val/loss` −7.4661 @ step 49,000** · `train/nan_skips` **0** · 중단·재개 **없음**
> 🔬 **평가**: ✅ **추론·채점 완료**(§7, TSE_Eval 2026-08-07) — best **SI-SDR 7.004** / ESTOI 0.7885 / PESQ 2.099 / WER 0.5234

## 📚 이 실험의 문서 구성

| 문서 | 담는 것 |
|---|---|
| **`report.md`** (이 문서) | 학습 전반 — 설정·환경·학습 곡선·진행 기록, 그리고 **두 체크포인트의 비교** |
| [**`report_best.md`**](./report_best.md) | **best** 체크포인트(step 49,000) 하나의 추론 결과 |
| [**`report_last.md`**](../../../exp_reports/llmtse/llmtse_baseline_v1/report_last.md) | **last** 체크포인트(step 50,000) 하나의 추론 결과 |

> 원칙: **공통이거나 비교인 것은 이 문서 · 정확히 한 체크포인트에만 참인 것은 자식 문서.**

<details><summary>📦 2026-08-11 문서 분할 — 어디로 옮겨갔나</summary>

| 이전 위치 | 현재 위치 |
|---|---|
| §7.1 best/last 2열 식별표 | `report_best.md` §2 · `report_last.md` §2 |
| §7.3 5지표 2열표 | `report_best.md` §3 · `report_last.md` §3 |
| §7.4 overlap 결과 셀 | `report_best.md` §5 · `report_last.md` §5 |
| §7.5(1) 구조 검증 | `report_best.md` §4-(1) · `report_last.md` §4-(1) |
| §7.5(2) val 값 일치 | `report_best.md` §4-(2) (best 전용) |
| §7.6 6-카테고리 표 | `report_best.md` §6 |

**이 문서에 그대로 남은 것**(외부 인용 대상 포함): §2.2 · §6.2 · §7.5(3) best↔last 비교 · §7.6 축 요약과 원인 분석 · §9
</details>

---

## 1. 개요

TPEX(ICASSP 2027) 논문의 비교 baseline인 **LLM-TSE**를 **PORTE-v3(24 kHz) text-only** 조건에서 학습하는 **첫 번째 본 학습**입니다.

| 항목 | 내용 |
|---|---|
| **목적** | TPEX baseline 표에 넣을 LLM-TSE 수치 확보 |
| **조건** | **text-only** (`return_enroll: false`) — TPEX와 공정 비교 |
| **이전 실험** | 없음 (첫 본 학습). 직전에 `smoke_20260805` 20+10 step 스모크런으로 파이프라인 검증 완료 |
| **선행 작업** | 결함 B1–B5 수정(`c1a375a`), 논문 원문 대조(`084b703`, `c6fcb09`) |

### 이 실험에서 확인하려는 것

1. 50 epoch(=50,000 optimizer step) 동안 **수렴하는가**, 어디서 **과적합**이 시작되는가
2. text-only 조건에서 도달 가능한 **SI-SDR / SI-SDRi / ESTOI / PESQ / WER**
3. `train/nan_skips` 가 0으로 유지되는가 (bf16 안정성)

---

## 2. 설정 요약 — 논문 / 팀원 코드 대조

설정 스냅샷: **[`config.yaml`](./config.yaml)** (학습 시작 시점의 `configs/config.yaml` 사본)

### 2.1 논문(arXiv:2310.07284) 명시값 — 전부 정렬 ✅

| 항목 | 값 | 논문 |
|---|---|---|
| optimizer | `AdamW` | ✅ *"the AdamW optimizer ... initial learning rate of 1e-4"* |
| lr | `1e-4` | ✅ |
| LR 스케줄 | linear warmup **1000 step** → **이후 constant (decay 없음)** | ✅ *"increases from 0 to 1e-4 and remains constant"* |
| grad clip | `30.0` | ✅ *"gradient normalization with a value of 30"* |
| **effective batch** | **40** (`batch 8 × accum 5`) | ✅ *"valid batch size of 40 per iteration"* |
| LoRA | `r=16, α=16, dropout=0.05, target=q_proj·k_proj` | ✅ |
| LLM | **LLaMA-2 7B Chat** (로컬 `ckpt/llama_2_7b_chat`, **bf16 통짜**) | ✅ |
| segment **길이** | `6.0`초 | ✅ *"set to a duration of 6 seconds"* (§V-A) — **길이만 일치**. 자르는 방식은 §2.2 |
| loss | `singlesrc_neg_sisdr` (zero-mean SI-SDR, 시간영역) | ✅ |
| 백본 | TD-SpeakerBeam 기본값 (k16/s8/512, TCN 8×3, i_adapt 7 `mul`) | ✅ |

### 2.2 논문 **미명시** → 근거를 밝혀야 하는 값 ⚠️
<!-- ⚠️ 이 § 번호는 외부에서 인용됩니다. 번호 변경·다른 파일로 이동 금지.
     인용처: docs/plans/impl_plan.md:236
             docs/reports/llmtse_pretrain_readiness_2026-08-05.md:633
             exp_reports/llmtse_baseline_v2/report.md (v1 report §2.2) -->

| 항목 | 값 | 출처 |
|---|---|---|
| **max_epochs** | **50** (= 50,000 optimizer step) | **팀원 코드** `src/teammate_codes/local/conf.yml: epochs: 50` 계승. **논문은 학습 길이를 언급하지 않음** |
| **early_stop** | `false` (로직 미구현) | 논문 미명시. 팀원은 `yes/patience 10` 이었으나 재현 근거가 없어 off. **best는 val_loss로 별도 추적** |
| **weight_decay** | `0.01` | 논문 미명시. AdamW 기본값 |
| **`crop_mode`** | `random` (blind) | ⚠️ **논문은 crop 을 하지 않는다.** §V-A 는 6초 mixture 를 40~70% overlap 으로 **합성**한다(*"online simulation"*). 사전 생성된 PORTE-v3(4~19.2초)를 6초로 맞추려고 팀원 코드가 도입한 방식이며 **논문 근거 없음**. 이것이 §7.6 의 `order` 실패를 낳았다 |
| **`text_pooling`** | v1 = `last_layer` | ⚠️ **논문과 다르다.** §V-C 는 *"averaging the outputs of the **last four** self-attention layers"* 인데 v1 은 **마지막 1개 층**의 last-token 을 썼다(팀원 코드 계승). 2026-08-07 에 `last4_mean` 을 구현해 `configs/config.yaml` 기본값으로 두었으나 **v1 은 재학습하지 않았다** — v1 결과는 `last_layer` 기준이다. 토큰 축(last-token)과 RMSNorm 정규화는 논문 미명시라 **우리 해석**이다 |

> ❗ 결과 서술 시 위 5개를 **"논문대로"라고 인용하지 말 것** (`CLAUDE.md` 금지 #4).

### 2.3 논문과 **의도적으로 다른** 값

| 항목 | 논문 | 이번 실험 | 이유 |
|---|---|---|---|
| sample rate | 16 kHz | **24 kHz** | TPEX/PORTE-v3 정렬 (리샘플 금지) |
| 데이터 | LibriSpeech+MLS **online simulation** | **PORTE-v3 사전생성 40,000** | TPEX와 동일 조건 |
| 조건 | audio enrollment + text | **text-only** | TPEX가 text-only |
| **overlap 비율** | **40 ~ 70%** (합성 시 무작위) | **0/20/40/60/80/100% 균등** | PORTE-v3 = TPEX 정렬. **학습 데이터의 33.3% 만 논문 범위**이고, 논문이 학습하지 않는 0%·100% 가 33.3% |
| GPU | 10× RTX 3090 | **H200 141GB × 1** | bf16 통짜로 1장 커버 |

> 🔍 **online simulation vs 고정 세트**: 논문은 매 iteration 데이터를 새로 생성해 **epoch 개념이 없습니다.**
> 우리는 고정 40,000개를 50회 반복하므로 **과적합 압력이 존재**합니다. 다만 학습 mixture의 **97.80%가 6초 초과**(평균 10.72초)라
> `crop_mode: random` 이 매 epoch 다른 구간을 잘라 실질적 증강으로 작동합니다. 완전 반복은 2.20%(878개)뿐.

### 2.4 데이터셋

| | train | dev | test |
|---|---|---|---|
| 행 수 | **40,000** | 5,000 | 5,000 |
| sample rate | 24 kHz | 24 kHz | 24 kHz |

- mixture 길이: min 4.11 / **mean 10.72** / max 17.20 초
- 고유 target 화자 **440**, 고유 prompt **48** (6 카테고리)
- 경로: `/home/work/my-datasets/porte_v3_clean`

### 2.5 스텝 산수

```
40,000행 ÷ batch 8         = 5,000 micro-step / epoch   (나머지 0)
5,000    ÷ grad_accum 5    = 1,000 optimizer step / epoch (나머지 0)
1,000 × 50 epoch           = 50,000 optimizer step   ← 파생값(설정 아님)
warmup 1,000 step          = 정확히 1 epoch
```

### ★ 종료 조건은 **epoch** 입니다 (step 아님)

혼동하기 쉬우므로 명시합니다.

```yaml
training:
  max_epochs: 50      # ← 실제 종료 조건
  max_steps: null     # ← null 이라 step 기반 종료는 비활성
```
```python
for epoch in range(start_epoch, max_epochs):          # 여기서 종료 (epoch 50)
    ...
    if max_steps and global_step >= int(max_steps):   # max_steps=null → 검사 자체가 동작 안 함
        stop = True
```

**50,000 은 설정값이 아니라 "50 epoch을 돌면 결과적으로 발생하는 optimizer step 수"** 입니다.

그럼에도 진행 상황을 step으로 말하게 되는 이유는, **로그·TensorBoard가 step으로만 진행을 표시**하기 때문입니다.

| 무엇 | 단위 |
|---|---|
| **종료 조건** | **epoch (50)** |
| 로그 (`step 2300 \| epoch 3 \| ...`) | step + epoch |
| TensorBoard x축 | step |
| `warmup_steps` · `val_every` · `metric_every` · `log_every` | step |

**1 epoch ≈ 48.5분** (2.91초/step × 1,000). → **실측 52.0분** (3.12초/step, 43.35h ÷ 50 epoch).

---

## 3. 환경

| 항목 | 값 |
|---|---|
| GPU | NVIDIA **H200 143,771 MiB** (sm_90, Hopper) × 1 |
| Driver / CUDA | 580.126.20 / 13.0 |
| conda env | `llmtse` · Python 3.10.20 |
| 스택 | torch 2.5.1+cu121 · accelerate 1.10.0 · transformers 4.46.3 · peft 0.13.2 |
| requirements | `requirements_h200.txt` |
| **코드 커밋** | **`c6fcb09`** |

### 리소스 — 예상(스모크런 기반) vs 실측

| 항목 | 학습 전 예상 | **실측** |
|---|---|---|
| VRAM | ≈53 GB | **54,302 MiB (53.0 GB)** — 학습 내내 변동 없음 ✅ |
| 체크포인트 1개 | 13.28 GB / ≈27초 | **13.28 GB** ✅ |
| 저장 횟수 | `val_every 500` → 100회 | **100회** ✅ |
| 디스크 사용 | ≈66 GB | **66 GB** (`best/` + `step_*` 3개 + `best_model.pth`) ✅ |
| step 속도 | ≈2.5초/step | **3.12초/step** (+25%, GPU 공유 §5.2) |
| **총 소요** | 37–40시간 | **43.35시간** (+9~17%) |

---

## 4. 실행 명령

```bash
conda activate llmtse
cd /home/work/my-code/llmtse

mkdir -p /home/work/my-outputs/llmtse
nvidia-smi --query-gpu=memory.used,memory.free --format=csv   # 53GB 여유 확인

accelerate launch \
    --num_processes 1 \
    --main_process_port 29517 \
    train.py \
    --config configs/config.yaml \
    --run_name llmtse_baseline_v1 \
    2>&1 | tee -a /home/work/my-outputs/llmtse/train_baseline_v1.log
```

### 중단 시 재개
```bash
accelerate launch --num_processes 1 --main_process_port 29517 train.py \
    --config configs/config.yaml --run_name llmtse_baseline_v1 \
    --resume /home/work/my-checkpoints/llmtse/llmtse_baseline_v1/best \
    2>&1 | tee -a /home/work/my-outputs/llmtse/train_baseline_v1.log
```

### TensorBoard
```bash
# 서버
tensorboard --logdir runs/ --port 6006 --reload_interval 15 \
    --reload_multifile true --samples_per_plugin "audio=100,images=100"
# 로컬
ssh -N -L 6006:localhost:6006 <user>@<server>   # → http://localhost:6006
```

### 추론 (학습 후)
```bash
bash scripts/run_inference_baseline_v1.sh          # best + last 두 체크포인트
```
절차는 **§7.1**, 산출 경로는 **§7.3**, 체크포인트별 결과는
[`report_best.md`](./report_best.md) · [`report_last.md`](../../../exp_reports/llmtse/llmtse_baseline_v1/report_last.md).

---

## 5. 진행 기록

| 시각 (KST) | 이벤트 | 비고 |
|---|---|---|
| 2026-08-05 11:26 | 실험 디렉토리 생성 · config 스냅샷 | 커밋 `c6fcb09` |
| **2026-08-05 11:29:10** | **★ 학습 시작** | PID 733625(launcher) / 733658(train) |
| 2026-08-05 12:01 | step 500 첫 validation + 체크포인트 | val/loss 0.1929 |
| 2026-08-05 12:24 | step 1000 validation → **warmup 종료**, LR 1e-4 상수 진입 | val/loss 0.1869 (best) |
| 2026-08-05 13:20 | **epoch 3 / 50** 진입 (step 2,300) · 이상 없음 | best 3회 갱신 · `nan_skips` 0 |
| 2026-08-05 15:43 | step 5,000 — **best 갱신 (val 0.18279)** | 이후 step 9,000 까지 갱신 없음(정체 구간) |
| 2026-08-05 ~16:00 | ⚠️ **GPU 공유 시작** — 타 프로젝트 `styletse` 학습이 46.7 GB 점유 | 아래 §5.2 |
| 2026-08-05 16:36 | **epoch 7 / 50** (step 6,150) · 이상 없음 | `nan_skips` 0 |
| 2026-08-06 04:40 | step 17,500 — **정체 구간 마지막** | SI-SDRi 여전히 +0.011 dB |
| **2026-08-06 05:03** | **★ step 18,000 — SI-SDRi 최초 양수 돌파** | 17,000 step 만에 "입력 통과" 해 탈출 (§6.1) |
| 2026-08-06 06:38 | step 20,000 — SI-SDRi **> 1 dB** | val/loss 0.183 → −0.822 |
| 2026-08-06 12:36 | step 27,500 — SI-SDRi **> 5 dB** | 급상승 구간 종료 |
| 2026-08-06 19:48 | step 36,500 — SI-SDR 6.80 dB | 이후 5.7~7.3 dB 진동만 |
| 2026-08-07 03:31 | step 46,000 — **SI-SDR 7.150 / ESTOI 0.787 (전 구간 최고)** | ⚠️ 이 ckpt 는 롤링으로 삭제됨 (§6.2) |
| **2026-08-07 06:06** | **★ best 최종 갱신 — step 49,000, val/loss −7.4661** | `best/` 저장 |
| **2026-08-07 06:53** | **★ 학습 종료** — step 50,000 / epoch 50 도달 | 정상 종료(중단 없음) |
| 2026-08-07 07:07 | `best/` 재로드 → `best_model.pth` 직렬화 완료 | 로그 `[best] reloaded ... for serialization` |

**중단·재개 이력**: **없음** — 43.35시간 동안 단일 프로세스(PID 733658)로 완주. `--resume` 미사용.

> 📌 08-06 이후 시각은 TensorBoard `wall_time` 기준입니다(로그에 시각 미기록).

### 시작 직후 상태 점검 (2026-08-05 12:28 기준)

| 항목 | 값 | 판정 |
|---|---|---|
| 배너 설정 | AdamW · lr 1e-4 · warmup 1000 · clip 30 · batch 8×5=**40** · max_epochs 50 | §2 설정과 일치 ✅ |
| `text_model` | `/home/work/my-code/llmtse/ckpt/llama_2_7b_chat` | 절대경로 해석 정상 ✅ |
| trainable / frozen | **14.42M** / **1.85M (fuse+aux)** | 예상치 일치 ✅ |
| VRAM | **54,300 MiB (53.0 GB)** | 스모크런 실측(54,302 MiB)과 동일 ✅ |
| **`nan_skips`** | **0** | bf16 안정 ✅ |
| 진행 | step **1,250 / 50,000** (2.50%), epoch 2 | |
| 평균 속도 | **2.86초/step** (모델 로드 포함) | 예상 2.5초와 근사 |
| **완료 예상** | **약 39.7시간 후 → 2026-08-07 새벽** | |

> ✅ **warmup 동작 확인**: LR이 step 50 `5.10e-06` → step 1000 `1.00e-04` 까지 선형 증가한 뒤,
> step 1050 이후 **`1.00e-04` 로 고정**됩니다. 논문의 *"increases from 0 to 1e-4 and remains constant"* 와 정확히 일치합니다.

> 📌 **train loss가 step마다 크게 출렁이는 것은 정상**입니다(예: step 1100 `-12.65`, step 1200 `3.37`).
> 로깅되는 `train/loss` 는 grad_accum 5개의 평균이 아니라 **마지막 마이크로배치 값**이고,
> 샘플별 SI-SDR 난이도 편차가 크기 때문입니다. 추세는 **`val/loss`** 로 판단하세요.

### 5.2 ⚠️ GPU 공유 (2026-08-05 ~16:00 ~)

같은 H200에서 **타 프로젝트 `styletse` 학습이 동시에 시작**되었습니다.

| PID | 사용량 | 정체 |
|---|---|---|
| 733658 | **54,302 MiB (53.0 GB)** | ★ 본 실험 (변동 없음) |
| 812846 | 47,880 MiB (46.7 GB) | `styletse` — 타 프로젝트 |
| 523553 | 5,972 MiB (5.8 GB) | `run_gpualive` 상시 프로세스 |
| **합계** | **108,176 MiB / 143,771 MiB** | **여유 34,980 MiB** |

**step 속도 영향** (체크포인트 mtime 실측):

| 구간 | 초/step | 비고 |
|---|---|---|
| 5,000 → 5,500 | **2.71** | 단독 실행 구간 |
| 5,500 → 6,000 | **2.92** | GPU 공유 구간 (**약 +8%**) |
| 전체 평균 | 2.99 | 모델 로드 시간 포함 |

→ **완료 예상이 8/7 03:55 → 8/7 05:04 로 밀림.** 학습 자체에는 이상 없음.

> ✅ **사후 확인**: 실제 종료는 **8/7 06:53** 로, 이 시점 예상(05:04)보다 **1시간 49분 더** 걸렸습니다.
> GPU 공유가 08-05 16:00 이후로도 계속되어 전체 평균이 **3.12초/step** 까지 올라갔기 때문입니다
> (단독 2.71 → 공유 2.92 로 추정했으나, 실제 후반 구간은 그보다 더 느렸습니다). OOM 은 발생하지 않았습니다.

> ⚠️ **리스크**: 여유가 35 GB로 줄었습니다. **세 번째 대형 작업이 추가되면 OOM으로 죽을 수 있습니다.**
> 다만 500 step마다 체크포인트가 있어 최대 25분 손실로 `--resume` 복구 가능합니다.
> 새 GPU 작업 전에는 반드시 `nvidia-smi` 로 여유를 확인하세요.

> ℹ️ Host Memory(RAM)가 88%로 보이는 것은 **정상**입니다. `free -h` 기준 실사용 135 GB이고
> 나머지는 PORTE-v3 wav 재읽기로 생긴 **파일 캐시**(available 1.8 TB)입니다.
> 본 실험 프로세스의 실제 RSS는 메인 2.04 GB + DataLoader worker 1.45 GB × 4 ≈ **8 GB** 수준입니다.

---

## 6. 학습 결과

| 항목 | 값 |
|---|---|
| best val/loss | **−7.4661** |
| best step | **49,000** (epoch 49) — `best/` · `best_model.pth` |
| 최종 step | **50,000 / 50,000** (epoch 50 완주, `max_epochs` 종료 조건 도달) |
| 과적합 시작 지점 | **관측되지 않음** — `val/loss` 가 발산하지 않고 step 35,000 이후 −5.7 ~ −7.5 사이에서 **진동만** 함. 과적합이 아니라 **수렴 후 정체**(§6.3) |
| `train/nan_skips` 최종값 | **0** ✅ bf16 오버플로 없음 |
| 실제 소요 시간 | **43.35시간** (예상 39.7h 대비 +9%, GPU 공유 영향 §5.2) |
| 평균 속도 | **3.12초/step** (전체 평균) |

### 6.1 ★ 핵심 서사 — 17,500 step 동안 "입력 통과" 해에 갇혔다가 돌파

이 학습의 가장 중요한 관찰입니다.

| 구간 | step | 상태 |
|---|---|---|
| **정체** | 500 – 17,500 (epoch 1–18) | `val/loss` 0.1828 ~ 0.1849 에서만 진동, **SI-SDRi ≈ 0** — 모델이 분리를 학습하지 못하고 **mixture 를 거의 그대로 통과**시키는 해에 머묾 |
| **돌파** | **18,000** | SI-SDRi 최초 양수 |
| **급상승** | 18,000 – 27,500 | SI-SDRi 0 → 5 dB (약 8시간) |
| **완만** | 27,500 – 35,000 | 5 → 6 dB |
| **정체(2)** | 35,000 – 50,000 | 5.7 ~ 7.3 dB 진동, 추세적 상승 없음 |

> ⚠️ **학습 도중 §6 에 적어둔 판단은 결과적으로 성급했습니다.**
> 당시 "step 10,000 이 다음 판단 지점 — 그때도 SI-SDRi 가 0 부근이면 텍스트 큐가 백본에 실제로
> 전달되는지, LoRA 경로가 학습에 기여하는지 점검이 필요하다"고 적었습니다.
> **실제로 step 10,000 에서도 SI-SDRi 는 −0.001 이었지만**, 아무 개입 없이 **8,000 step 뒤에 스스로 돌파**했습니다.
> → **정체 구간의 길이를 과소평가하지 말 것.** 이 아키텍처에서 "입력 통과" 해 탈출에는 **약 18 epoch** 가 필요했습니다.

### 6.2 ⚠️ best 선정 기준과 보고 지표의 불일치 (신규 발견)

`train.neg_sisdr_masked` 는 학습 안정화를 위해 **샘플별 SI-SDR 을 −30 dB 에서 바닥 처리**합니다.

```python
# train.py:204-205
loss = torch.nan_to_num(loss, nan=30.0, posinf=30.0, neginf=-30.0)
loss = torch.clamp(loss, max=30.0)      # ← negative SI-SDR 상한 30 = SI-SDR 하한 −30 dB
```

반면 `val/si_sdr`(asteroid `get_metrics`)에는 이 바닥이 없습니다. 학습 후반에 일부 dev 샘플이
−30 dB 아래로 무너지면서 **`val/loss` 가 실제 SI-SDR 보다 좋게 보이기 시작**합니다.

| 구간 | `(−val/loss) − val/si_sdr` 평균 |
|---|---|
| step ≤ 35,000 | **0.000 dB** — 완전 일치 |
| 35,000 – 45,000 | +0.545 dB |
| 45,000 – 50,000 | **+0.931 dB** |

최초 발생 step **37,000**, 전체 100회 validation 중 **17회**에서 gap > 0.05 dB.

**그래서 선정 기준에 따라 best 가 달라집니다**:

| 기준 | best step | 값 |
|---|---|---|
| `val/loss` (clamp 적용) ← **코드가 실제로 사용** | 49,000 | −7.4661 |
| `val/si_sdr` (clamp 없음) | **46,000** | **7.150 dB** |
| `val/estoi` | **46,000** | **0.7866** |
| `val/pesq` | 45,000 | 2.2297 |

> ❌ **step 46,000 체크포인트는 이미 삭제되었습니다.** `ckpt_keep: 3` 롤링으로
> `step_00049000` / `step_00049500` / `step_00050000` 만 남았습니다. 복구 불가.
>
> ✅ 다만 **살아남은 3개 중에서는 step 49,000 이 `val/loss`·SI-SDR·SI-SDRi 모두 최고**이므로,
> `best_model.pth`(= step 49,000)를 평가에 쓰는 현재 선택은 **유효**합니다.

| 생존 ckpt | val/loss | SI-SDR | SI-SDRi | ESTOI | PESQ |
|---|---|---|---|---|---|
| **49,000 (= `best/` = `best_model.pth`)** | **−7.4661** | **6.478** | **6.662** | 0.764 | 2.193 |
| 49,500 | −6.9553 | 6.418 | 6.602 | 0.771 | 2.129 |
| 50,000 (last) | −6.1556 | 4.365 | 4.549 | 0.756 | **2.206** |

→ 대응은 §8·§9 참조. 이번 학습 결과의 해석에는 영향이 없으므로 `train.py` 는 수정하지 않았습니다.

### 6.3 과적합이 아니라 "수렴 후 진동"

`val/loss` 는 반등·발산하지 않고 step 35,000 이후 평균 −6.3 / **σ 0.66** 으로 진동합니다
(마지막 20회 기준: mean −6.312, min −7.466, max −4.645).

이 진동 폭이 큰 이유는 **validation 이 dev 전체가 아니라 극히 일부만 보기 때문**입니다.

```python
# train.py:400-401  — dev 는 학습과 달리 crop 하지 않는다
val_ds = LLMTSEDataset(dev_csv, segment=d["segment_eval"], ...)   # segment_eval: null → full-length
# train.py:408      — batch_size 는 config 가 아니라 1 로 하드코딩
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, ...)
```

| 항목 | 값 |
|---|---|
| segment | **full-length** (`segment_eval: null`) — 학습의 6초 crop 과 다름 |
| batch_size | **1** (하드코딩, `loader.batch_size` 미적용) |
| `val_max_batches` | 50 |
| **실제 표본** | **dev 앞 50 발화 = 5,000 의 1%** |

⚠️ `val_max_batches` 는 "batch 수"라 batch_size 8 을 곱해 400 으로 오해하기 쉽지만, **val_loader 는 batch=1
이므로 곱이 아니라 그대로 50 발화**입니다. 즉 best 선정이 **1% 부분집합** 위에서 이뤄졌고,
인접 checkpoint 간 1 dB 차이는 **실제 차이가 아니라 표집 노이즈일 가능성이 큽니다.**
best(49,000)와 last(50,000) 를 **test 5,000 전체**로 함께 채점하려는 이유가 이것입니다(§7).

> ✅ **다행인 점**: validation 이 full-length·batch1 이라 **추론(`inference.py`)과 조건이 완전히 동일**합니다.
> 따라서 val 지표와 test 지표를 직접 비교할 수 있습니다. 실제로 best 산출물의 **test 5,000 전수**는
> **SI-SDR 7.004 / SI-SDRi 6.961** 로, dev 50 기준 값(6.478 / 6.662)과 **0.53 dB 이내로 일치**했습니다(§7.5).

### val 지표 추이 (validation 시점별, `val_every: 500`)

| step | epoch | val/loss | SI-SDR | SI-SDRi | ESTOI | PESQ | best |
|---|---|---|---|---|---|---|---|
| 500 | 1 | 0.1929 | −0.193 | −0.009 | 0.772 | 1.960 | ★ |
| 1,000 | 1 | 0.1869 | −0.187 | −0.003 | 0.775 | 1.961 | ★ |
| 1,500 | 2 | 0.1836 | −0.184 | 0.000 | 0.776 | 2.006 | ★ |
| 2,000 | 2 | 0.1846 | −0.185 | −0.001 | 0.775 | 2.017 | |
| 2,500 | 3 | 0.1834 | −0.183 | +0.001 | 0.776 | 2.040 | ★ |
| 3,000 | 3 | 0.1831 | −0.183 | +0.001 | 0.776 | 2.044 | ★ |
| 3,500 | 4 | 0.1840 | −0.184 | 0.000 | 0.776 | 2.034 | |
| 4,000 | 4 | 0.1849 | −0.185 | −0.001 | 0.776 | 2.019 | |
| 4,500 | 5 | 0.1840 | −0.184 | 0.000 | 0.775 | 1.991 | |
| **5,000** | 5 | **0.18279** | **−0.183** | **+0.001** | 0.776 | 2.048 | ★ **best** |
| 5,500 | 6 | 0.1840 | −0.184 | 0.000 | 0.776 | 2.044 | |
| 6,000 | 6 | 0.1832 | −0.183 | +0.001 | 0.776 | 2.038 | |
| … | | *(정체 지속)* | | | | | |
| 7,500 | 8 | 0.1834 | −0.183 | +0.001 | 0.776 | 2.006 | |
| 10,000 | 10 | 0.1847 | −0.185 | −0.001 | 0.776 | 2.049 | ← 판단 지점, **여전히 0** |
| 12,500 | 13 | 0.1838 | −0.184 | 0.000 | 0.776 | 2.051 | |
| 15,000 | 15 | 0.1834 | −0.183 | +0.001 | 0.776 | 2.050 | |
| 17,500 | 18 | 0.1731 | −0.173 | +0.011 | 0.776 | 2.053 | 정체 마지막 |
| **18,000** | **18** | — | — | **> 0** | | | ★ **돌파** |
| 20,000 | 20 | −0.8223 | 0.822 | **1.006** | 0.768 | 2.073 | |
| 22,500 | 23 | −1.0091 | 1.009 | 1.193 | 0.764 | 2.075 | |
| 25,000 | 25 | −3.6506 | 3.651 | 3.835 | 0.765 | 2.086 | |
| 27,500 | 28 | −4.8671 | 4.867 | 5.051 | 0.769 | 2.125 | |
| 30,000 | 30 | −5.7121 | 5.712 | 5.896 | 0.772 | 2.151 | |
| 32,500 | 33 | −6.0502 | 6.050 | 6.234 | 0.770 | 2.150 | |
| 35,000 | 35 | −5.6866 | 5.687 | 5.871 | 0.769 | 2.158 | |
| 36,500 | 37 | −6.7995 | 6.800 | 6.984 | 0.782 | 2.137 | ★ |
| 37,500 | 38 | −5.9582 | 5.958 | 6.142 | 0.768 | 2.124 | |
| 40,000 | 40 | −6.4606 | 6.461 | 6.645 | 0.779 | 2.219 | |
| 42,500 | 43 | −5.8595 | 5.583 | 5.767 | 0.765 | 2.130 | |
| 44,000 | 44 | −7.0414 | 7.041 | 7.225 | 0.782 | 2.193 | ★ |
| 45,000 | 45 | −6.1179 | 6.118 | 6.302 | 0.767 | **2.230** | PESQ 최고 |
| **46,000** | **46** | −7.1496 | **7.150** | **7.334** | **0.787** | 2.199 | ★ **SI-SDR·ESTOI 최고** ⚠️ ckpt 삭제됨 |
| 47,500 | 48 | −6.2566 | 5.158 | 5.342 | 0.762 | 2.162 | |
| 48,000 | 48 | −6.1015 | 4.877 | 5.061 | 0.759 | 2.070 | |
| 48,500 | 49 | −6.5971 | 6.597 | 6.781 | 0.780 | 2.149 | |
| **49,000** | **49** | **−7.4661** | 6.478 | 6.662 | 0.764 | 2.193 | ★ **best (val/loss)** — 생존 |
| 49,500 | 50 | −6.9553 | 6.418 | 6.602 | 0.771 | 2.129 | 생존 |
| **50,000** | **50** | −6.1556 | 4.365 | 4.549 | 0.756 | 2.206 | **last** — 생존 |

> 📌 전 구간(100회 validation)은 TensorBoard `runs/llmtse_baseline_v1/tb` 에 있습니다. 위 표는 발췌입니다.
> 📌 step 49,000·47,500·48,000·50,000 처럼 `val/loss` 와 `SI-SDR` 의 절댓값이 어긋나는 지점은 **§6.2 의 clamp 때문**입니다(step 35,000 이전엔 두 값이 완전히 일치).

> 📎 참고 — 스모크런(`smoke_20260805`) step 30: val/loss 9.83 / SI-SDR −9.83 dB / ESTOI 0.369 / PESQ 1.050
> → 최종 best 대비 **SI-SDR +16.3 dB · ESTOI +0.40 · PESQ +1.14**.

---

## 7. 평가 결과

**추론·채점 완료** (`TSE_Eval`, 2026-08-07). §6.3 의 이유(validation 이 dev 의 **1%(50 발화)** 만 봄)로 **best 와 last 두 체크포인트를
test 5,000 전체에 대해 각각 추론**했습니다. 채점기는 별도 결정합니다(`eval.py` 또는 3-프로젝트 공통 `TSE_Eval`).

> 📌 **이 절의 소유 규칙** — 어디에 무엇이 있는지
>
> | 정보 | 정본 |
> |---|---|
> | 추론 절차 · 매니페스트 스키마 · overlap bin 정의 | **이 문서** §7.1 / §7.2 / §7.4 |
> | 개별 체크포인트의 지표 · 층화 · 카테고리 · 검증 | [`report_best.md`](./report_best.md) / [`report_last.md`](../../../exp_reports/llmtse/llmtse_baseline_v1/report_last.md) |
> | **best ↔ last 차이** (페어 통계 · 승률 · dev↔test) | **이 문서** §7.5 |
> | `order` 실패 축 요약과 **원인** | **이 문서** §7.6 |

### 7.1 추론 실행 (공통 절차)

```bash
# 스모크(3개):  LIMIT=3 bash scripts/run_inference_baseline_v1.sh
bash scripts/run_inference_baseline_v1.sh
```

한 체크포인트만 돌리는 명령은 각 자식 문서 §2.2 에 있습니다.

조건: **full-length**(`segment=None`, `eval.mode: full_length`), batch=1, bf16, text-only.
산출 wav = 24 kHz float32, 각 약 5.1 GB. 체크포인트별 경로는 §7.3 표 참조.

> ⚠️ `--run_name` 을 반드시 나눠야 합니다. `inference.py` 는 산출 경로를 `<results_dir>/<run_name>` 으로
> 고정하고 출력 경로 override 플래그가 없어, 같은 이름으로 두 번 돌리면 **앞 결과를 덮어씁니다.**
>
> 📌 `--ckpt` 로 **accelerate `save_state` 디렉토리**를 직접 지정할 수 있게 `inference.py` 에
> `_load_state_dict()` 를 추가했습니다(2026-08-07). `.pth` / `save_state` 디렉토리 / `.safetensors` 3형태 지원.
> 두 형식의 키가 동일해(889개 일치) 로드 시 `missing=0, unexpected=0` 이 나오는 것이 정상입니다.

### 7.2 매니페스트 스키마 (19열, 3-프로젝트 정렬)

```
file_id, pred_path, mixed_path, target_path, interference_path, sample_rate,
task, overlap_ratio, prompt, prompt_category, target_gender, infer_gender,
first_speak, target_start_time, target_end_time, length_sec,
overlap_start_time, overlap_end_time, target_sentence
```
StyleTSE 19열과 **집합 동일**, TPEX 26열의 **부분집합** → `TSE_Eval` 이 그대로 받습니다.

### 7.3 두 체크포인트 — 개요 및 상세 문서

| | best | last |
|---|---|---|
| step | **49,000** (epoch 48) | 50,000 (epoch 50, 종료) |
| 선정 기준 | `val/loss` 최소 (−7.4661) — clamp 영향 §6.2 | 없음(마지막 step) |
| ckpt | `<ckpt>/llmtse_baseline_v1/best_model.pth` | `<ckpt>/llmtse_baseline_v1/step_00050000/` |
| `--run_name` | `llmtse_baseline_v1` | `llmtse_baseline_v1_step50000` |
| wav | `<out>/llmtse_baseline_v1/preds/` | `<out>/llmtse_baseline_v1_step50000/preds/` |
| 매니페스트 | `<out>/llmtse_baseline_v1/inference_test.csv` | `<out>/llmtse_baseline_v1_step50000/inference_test.csv` |
| 채점 상태 | ✅ **완료** (TSE_Eval 2026-08-07) | ✅ **완료** |
| SI-SDR / ESTOI / PESQ / WER | **7.004** / 0.7885 / 2.099 / 0.5234 | 6.823 / 0.7799 / 2.075 / 0.5185 |
| **상세** | **[report_best.md](./report_best.md)** | **[report_last.md](../../../exp_reports/llmtse/llmtse_baseline_v1/report_last.md)** |

`<ckpt>` = `/home/work/my-checkpoints/llmtse` · `<out>` = `/home/work/my-outputs/llmtse`

> 📌 위 한 줄은 **요약**이며, 층화·카테고리·추가 축을 포함한 전체 수치는 각 자식 문서가 정본입니다.
> 두 체크포인트의 **차이**만 §7.5 에서 다룹니다.

채점 시 공통 규칙:

> ⚠️ **SDR/SIR/SAR·DNSMOS는 미산출**입니다(`CLAUDE.md` #7).
> ⚠️ 표에 **"text-only (no enrollment)"** 조건을 반드시 명시할 것.

### 7.4 overlap 층화 — bin 정의

| overlap bin | 0.0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 |
|---|---|---|---|---|---|---|
| n | 834 | 834 | 833 | 833 | 833 | 833 |

체크포인트별 결과는 각 자식 문서 §5. 두 체크포인트의 **비교**는 §7.5(3).

### 7.5 추론 산출물 검증 (2026-08-07)

정식 채점 전에 **산출물이 실제 모델 출력인지** 확인한 결과입니다.
구조 검증은 test 5,000 중 무작위 500 표본으로, **지표는 TSE_Eval 전수 채점(5,000)** 으로 확인했습니다.

체크포인트별 검증 결과는 각 자식 문서 §4. 여기에는 **두 체크포인트에 걸친 항목만** 둡니다.

#### (1) 교차 확인 — 덮어쓰기 사고 없음 ✅

`best`/`last` 산출 wav 가 동일 파일인지 `np.array_equal` 로 대조 → **False**.
두 매니페스트 모두 5,000행 / 19열 / 결측 0 / `file_id` 중복 0 을 통과했습니다(각 자식 §4-(1)).

#### (2) dev 50 예측의 신뢰도 — ★ 채점 후 확정치

| 체크포인트 | dev 50 (validation) | **test 5,000 (TSE_Eval)** | 방향 |
|---|---|---|---|
| best (49,000) | 6.478 | **7.004** | 과소평가 0.53 dB |
| last (50,000) | 4.365 | **6.823** | **과소평가 2.46 dB** |

dev 50 이 두 체크포인트를 **비대칭적으로** 과소평가했고, 그 결과 **격차가 부풀려 보였습니다**:

| | dev 50 | **test 5,000** | 배율 |
|---|---|---|---|
| best − last (SI-SDR) | **+2.113 dB** | **+0.181 dB** | **11.7배 과대평가** |

> ⚠️ 이전 판(2026-08-07)은 test **500 표본** 기준 +0.654 dB 로 적었습니다. 전수 채점 결과
> **+0.181 dB** 로 더 줄었습니다 — 500 표본조차 격차를 3.6배 부풀렸습니다.

#### (3) ★ best vs last — dev 50 이 격차를 12배 과대평가했다

**전수 채점 확정치** (test 5,000, TSE_Eval). 이전 판의 500 표본 수치는 아래 표로 대체합니다.

| overlap | n | best (49,000) | last (50,000) | 차이 |
|---|---|---|---|---|
| 0.0 | 834 | **15.788** | 15.257 | **+0.531** |
| 0.2 | 834 | 6.893 | 6.632 | +0.261 |
| 0.4 | 833 | 5.450 | 5.418 | +0.031 |
| 0.6 | 833 | 5.274 | 5.211 | +0.063 |
| 0.8 | 833 | 4.616 | 4.525 | +0.091 |
| 1.0 | 833 | 3.991 | 3.883 | +0.108 |
| **전체** | **5000** | **7.004** | **6.823** | **+0.181** |

지표별로는 방향이 갈립니다:

| 지표 | best | last | 차이 |
|---|---|---|---|
| SI-SDR | **7.004** | 6.823 | +0.181 |
| ESTOI | **0.7885** | 0.7799 | +0.0086 |
| PESQ | **2.099** | 2.075 | +0.024 |
| WER (낮을수록 좋음) | 0.5234 | **0.5185** | **last 우위** |
| spk_sim | 0.7166 | **0.7239** | **last 우위** |

짝지은 비교(paired, n=5,000, SI-SDR):

| 통계 | 값 |
|---|---|
| 평균 차이 | **+0.181 dB** (SE 0.075, 95% CI **[+0.034, +0.328]**) |
| **중앙값 차이** | **+0.001 dB** ← 사실상 0 |
| paired t-test | t = 2.42, **p = 0.016** |
| best 승 / last 승 | **2565 / 2435** (51.3% — 동전던지기 수준) |

**해석** — best 가 통계적으로 유의하긴 하지만(p=0.016), 실질적 의미는 거의 없습니다:

1. **dev 50 이 예측한 +2.113 dB 의 12분의 1**(+0.181 dB). §6.3 의 표집 노이즈 우려가 **전수 채점으로 확정**되었습니다. 50 발화(dev 의 1%)로 checkpoint 를 고르는 것은 신뢰할 수 없습니다.
2. **중앙값 차이는 +0.001 dB, 승률 51.3%** — 전형적 발화에서 두 체크포인트는 구별되지 않습니다. 평균 차이는 **꼬리(소수의 대실패 샘플)** 가 만듭니다: SI-SDR `< −30 dB` 가 best **43개** vs last **61개**.
3. **WER 과 spk_sim 은 오히려 last 가 낫습니다.** "best 가 전반적으로 우월하다"고 말할 수 없습니다.

> 📌 논문 표에는 관례대로 **best(step 49,000)** 를 씁니다. 다만 **"best 가 last 보다 2 dB 낫다"는
> dev 기준 서술은 쓰면 안 됩니다** — test 전수에서는 **0.18 dB**, 중앙값으로는 **0 dB** 이고
> WER·spk_sim 은 역전됩니다.

> ⏱️ **추론 속도**: 각 5,000 발화 **11.4분 (0.14초/발화)**. 학습 중 validation 은 발화당 1.27초였지만
> 그쪽은 ESTOI·PESQ 등 **CPU 지표 계산**이 대부분이고, `inference.py` 는 forward + wav 저장만 하므로
> 이 차이가 정상입니다(지표는 채점기 소관). TensorBoard wall-time 역산으로 교차 확인했습니다.

### 7.6 ★ prompt_category 별 성능 — `order` 는 사실상 실패했다
<!-- ⚠️ 이 § 번호는 외부에서 인용됩니다. 번호 변경·다른 파일로 이동 금지.
     인용처: docs/guides/training_segment_policy.md:99
             docs/reports/llmtse_pretrain_readiness_2026-08-05.md:676
             exp_reports/llmtse_baseline_v2/report.md (v1 report §7.6)
     6-카테고리 상세표의 정본은 report_best.md §6 이며, 아래 축 요약만 여기 둔다. -->

전체 평균(7.0 dB)이 숨기고 있는 사실입니다. **test 5,000 전수**, `best_model.pth`:

| 축 | n | SI-SDR | **화자 혼동률** | WER | spk_sim |
|---|---|---|---|---|---|
| gender | 1750 | **12.836 dB** | **1.5%** | 0.339 | 0.848 |
| pitch | 1500 | 7.645 dB | 13.7% | 0.481 | 0.736 |
| **order** | 1750 | **0.622 dB** | **43.8%** ← 동전던지기(50%) | **0.735** | **0.569** |

> ★ 축 사이 격차가 **12.2 dB** 입니다. `order_first` 는 개별로 보면 **SI-SDR −0.049 dB(음수)** 로,
> 분리하지 않은 mixture 를 그대로 내보내는 것보다 못합니다. WER 73.5% · spk_sim 0.569 도 같은 이야기입니다.
>
> 📌 6개 `prompt_category` 각각의 상세 수치는 **[`report_best.md`](./report_best.md) §6 이 정본**입니다.
> 위 3축 요약만 여기에 둡니다(외부 인용 대상).
>
> 🔁 **last 체크포인트도 동일합니다** — order 0.485 dB / 혼동 42.3%([`report_last.md`](../../../exp_reports/llmtse/llmtse_baseline_v1/report_last.md) §6).
> 즉 이 실패는 **체크포인트 선택이 아니라 학습 설정의 결과**입니다.

"화자 혼동률" = 예측이 target 보다 **간섭 화자**에 더 가까운 비율(각각에 대해 SI-SDR 계산 후 비교, test 5,000 전수).
gender 는 간섭화자 SI-SDR 이 −25 dB 로 확실히 배제하는데, order 는 −2.2 dB 에 불과합니다 —
**어느 화자를 뽑을지 결정하지 못하고 있습니다.**

#### 원인은 text pooling 이 아니라 **6초 random crop** 입니다

학습된 `e_txt` 는 잘 분리됩니다 — silhouette **0.976**, 클래스間/内 거리비 **42.6x**.
카테고리 중심 코사인도 의미대로 정렬돼 있습니다(여성↔고음 0.679, 남성↔저음 0.734, 반대쌍 ≈ 0).
**단 하나 예외가 `order_first` ↔ `order_later` = 0.995 — 정반대 지시가 거의 같은 벡터입니다.**

이유: "누가 먼저 말했는지"를 판단하려면 **두 화자의 시작점이 모두 crop 창 안**에 있어야 합니다.

| overlap | 0.0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 | **order 전체** |
|---|---|---|---|---|---|---|---|
| 6초 random crop 이 두 onset 을 포함할 확률 | **0.0%** | **0.0%** | **0.0%** | 0.6% | 4.6% | 8.0% | **2.3%** |

**학습 세그먼트의 97.7% 에서 order 는 판단 불가능**했습니다. 임베딩 붕괴(0.995)는 원인이 아니라 **결과**입니다 — 두 프롬프트를 구별할 gradient 신호가 존재한 적이 없습니다.

논문에는 이 문제가 없습니다. §V-A 는 6초 mixture 를 **40~70% overlap 으로 합성**하므로 두 onset 이 항상 창 안에 있습니다. PORTE-v3 는 4~19.2초 mixture 에 0~100% overlap 이라 **6초 crop 과 구조적으로 맞지 않습니다.**

필요한 창 길이(학습셋 실측, 두 onset 간격 최대 9.2초):

| segment | 6초 | 8초 | 10초 | 12초 | 14초 | 16초 | 18초 |
|---|---|---|---|---|---|---|---|
| order 판단 가능 | 2.2% | 22.5% | 47.2% | 69.2% | 84.9% | 94.4% | **100%** |

> 📌 100% overlap 은 두 화자의 시작점이 **정확히 같아**(간격 0.000초, 100%) 창 길이와 무관하게 순서가 정의되지 않습니다. 데이터셋 설계 자체의 문제입니다.
>
> 📌 **`segment_train: 6.0` 유지는 의도적 결정**입니다(2026-08-07). 논문의 6초 숫자를 지키는 대신 order 카테고리를 포기했습니다. 논문 표에 **반드시 명시**해야 합니다(§9).

---

## 8. 관찰 · 결론

이 실험에서 확인하려던 3가지(§1)에 대한 답:

| 질문 | 답 |
|---|---|
| 50 epoch 동안 수렴하는가, 어디서 과적합하는가 | **수렴함. 과적합은 관측되지 않음.** step 18,000 돌파 → 35,000 수렴 → 이후 진동만. `val/loss` 발산 없음 |
| text-only 조건의 도달 성능 | **확정: test 5,000 전수 SI-SDR 7.004 / SI-SDRi 6.961 / ESTOI 0.7885 / PESQ 2.099 / WER 0.5234** (TSE_Eval). ⚠️ 단 이 평균은 gender 12.836 dB 와 order 0.622 dB 가 섞인 값이다(§7.6) |
| `nan_skips` 0 유지 (bf16 안정성) | **0** ✅ 43시간 동안 단 한 번도 없음 |

### 다음 실험(v2)에 넘길 것

1. **★ best 선정 기준을 바꿀 것.** 현재 `val/loss` 는 clamp(−30 dB) 가 걸린 값이라 논문에 보고할
   SI-SDR 과 후반부에 최대 0.93 dB 어긋납니다(§6.2). **`val/si_sdr` 로 best 를 고르거나**,
   최소한 `val/si_sdr` 기준 best 도 함께 저장하십시오. 이번엔 그 때문에 실제 최고 성능 체크포인트
   (step 46,000, SI-SDR 7.150 / ESTOI 0.787)를 **잃었습니다.**
2. **★ `ckpt_keep` 을 늘릴 것.** `3` 은 너무 공격적입니다. 13 GB/개라 디스크 부담은 있지만,
   후반 1만 step 이 σ 0.66 dB 로 진동하는 학습에서 3개는 사후 재선택 여지를 없앱니다. **10 이상 권장.**
3. **★ `val_max_batches` 를 늘릴 것 — 실측으로 확인된 문제.** `val_loader` 가 **batch=1 하드코딩**이라
   현재 값 50 은 **dev 5,000 중 50 발화(1%)** 에 불과합니다(§6.3). 그 결과 dev 50 은 best/last 격차를
   **+2.11 dB 로 예측했지만 test 실측은 +0.65 dB(중앙값 +0.007 dB)** — **3배 이상 과대평가**했습니다(§7.5).
   **최소 500(=10%)** 로 올리거나, val_loader 의 batch_size 를 config 로 빼십시오.
   full-length 라 batch 를 키우려면 길이 정렬/패딩 고려 필요(`neg_sisdr_masked` 는 이미 지원).
   ⚠️ 지금 값으로는 **best/last 를 구분하는 능력 자체가 없습니다**(test 에서 best 승률 53.4%).
4. **`max_epochs 50` 은 과할 수 있습니다.** step 35,000(epoch 35) 이후 15,000 step(약 13시간)은
   추세적 개선 없이 진동만 했습니다. **35~40 epoch 로 줄이면 25% 시간 절감** 가능. 단 정체 구간이
   18 epoch 나 지속됐던 만큼, 줄이려면 SI-SDRi 기반 plateau 판정을 함께 넣어야 안전합니다.
5. **정체 구간의 길이를 과소평가하지 말 것.** step 10,000 에서 "텍스트 큐 전달 점검이 필요하다"고
   판단했지만, 개입 없이 18,000 에서 스스로 돌파했습니다(§6.1). 조기 중단했다면 실패로 오판했을 것입니다.
6. batch 20×2(H200 권장) 대비 8×5 의 step 속도 비교 — _(미측정, v2 과제로 유지)_
7. **★ `segment_train` 을 14초 이상으로 올릴지 결정할 것.** 6초로는 `order` 프롬프트(전체의 1/3)가
   **원리적으로 학습 불가능**합니다 — 두 화자 onset 을 모두 포함할 확률 2.3%(§7.6). 14초면 84.9%,
   18초·full-length 면 100%. 지금은 논문의 6초 숫자를 지키려 유지했지만, order 를 1.17 dB 에서 끌어올리면
   **전체 평균이 6.9 → 10 dB 대로 뛸 여지**가 있습니다. ⚠️ `crop_mode: target_aware` 로는 해결되지
   않습니다(창 길이 문제).
8. **★ `text_pooling: last4_mean` 으로 재학습할지 결정할 것.** v1 은 `last_layer`(논문 미정렬)로 학습됐습니다.
   코드는 2026-08-07 에 정렬해 두었으나(§2.2) 재학습은 하지 않았습니다. 다만 측정상 pooling 은 성능 병목이
   아니므로(gender 혼동률 1.4%), **7번과 함께** 돌리는 것이 효율적입니다.

---

## 9. 알려진 제약
<!-- ⚠️ 이 § 번호는 외부에서 인용됩니다. 번호 변경·다른 파일로 이동 금지.
     인용처: docs/guides/training_segment_policy.md:99 ("§7.6·§9") -->

| 제약 | 내용 |
|---|---|
| **★ `order` 카테고리 실패** | 전체 prompt 의 **1/3**(order_first·order_later)이 **SI-SDR 1.17 dB · 화자 혼동률 46%** 로 사실상 동작하지 않음. 원인은 6초 random crop 이 두 화자의 onset 을 포함할 확률이 **2.3%** 뿐이라는 구조적 한계(§7.6). **논문 표에서 이 실패를 숨기지 말 것** — 전체 평균만 쓰면 gender(12.2 dB)가 order(1.2 dB)를 가린다 |
| **★ `text_pooling` 미정렬** | v1 은 `last_layer`(마지막 1개 층)로 학습됨. 논문 §V-C 는 **마지막 4개 층 평균**. 2026-08-07 에 `last4_mean` 을 구현했으나 **v1 은 재학습하지 않았다** → **v1 결과를 "논문 구현"이라고 서술하면 안 된다** (§2.2) |
| **★ best/last 격차 서술** | dev 50 기준 +2.11 dB 였으나 **test 실측 +0.65 dB, 중앙값 +0.007 dB, best 승률 53.4%**(§7.5). **"best 가 last 보다 2 dB 낫다"는 dev 기준 서술 금지** |
| **★ WER 52.3%** | 전체 WER 이 **0.5234** 로 매우 높다. 카테고리별로 gender 33.9% vs **order 73.5%** 로 갈리며, 동성 화자 조건은 **70.3%** 다. 논문 표에 WER 을 실을 때 **전체 평균만 쓰면 오해를 부른다** — 층화 수치를 함께 제시할 것 |
| **prompt 다양성** | 고유 prompt 48개(6 카테고리)뿐. 논문의 "임의 자연어 지시" 목표 대비 좁음 → **"LLM-TSE 재현"이 아니라 "PORTE-v3 조건의 LLM-TSE"** 로 서술할 것 |
| **학습 길이 근거** | `max_epochs 50` 은 논문 아닌 팀원 코드 출처 (§2.2) |
| **체크포인트 크기** | 학습 파라미터는 14.42M인데 LLaMA 동결분까지 저장해 1개당 13.28 GB. → `ckpt_keep` 을 늘리기 어려운 실질 원인 |
| **★ best 선정 기준** | `val/loss` 는 `clamp(max=30)` 이 걸린 값이라 보고 지표 SI-SDR 과 후반부 최대 0.93 dB 어긋남(§6.2). **실제 최고 성능 ckpt(step 46,000)는 롤링으로 소실**. 이번 결과 해석엔 영향 없으나 v2 에서 반드시 수정 |
| **★ validation 표본** | `val_max_batches: 50` + `val_loader` batch=1 하드코딩 → dev 5,000 중 **50 발화(1%)** 로만 val/best 판정. 인접 ckpt 간 1 dB 차이는 표집 노이즈일 수 있음 — **실측으로 확인됨**(dev 예측 +2.113 dB vs **test 전수 실측 +0.181 dB — 12배 과대평가**, §7.5) |
| **LaTeX·baseline 비교표** | `eval.py` 는 **구현되어 있음**(매니페스트 기반 채점). 다만 TPEX/StyleTSE 와의 3-프로젝트 공통 채점은 별도 repo `TSE_Eval` 에서 수행 예정이며 **아직 준비 중** |
| **`BNB_CUDA_VERSION=126`** | 컨테이너 환경변수. torch cu121과 불일치하나 `load_in_4bit: false` 라 현재 무해. **양자화 도입 시 주의** |
