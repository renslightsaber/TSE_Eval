#!/usr/bin/env python
"""scripts/check_eval_done.py — 채점이 끝났는지 확인하고, 끝났으면 결과를 비교한다.

eval_llmtse.sh 가 남긴 산출물 3종(per-row CSV · summary CSV · sidecar JSON)이
모두 존재하는지 보고, 다 있으면 전체 성능·delta·출처 일치를 출력한다.

  · 아직 진행 중  → 무엇이 없는지 알려주고 exit 1
  · 끝났으나 문제 → 문제를 알려주고 exit 1
  · 정상          → 비교 표를 출력하고 exit 0

그래서 조건부 실행에 그대로 쓸 수 있다:

    python scripts/check_eval_done.py && echo "비교 유효"

★ 마지막 '출처 일치' 절이 이 스크립트의 핵심이다. 두 런이 서로 다른 백엔드·
  샘플레이트·DNSMOS EP·모델로 채점됐다면 숫자를 나란히 놓는 것 자체가 무의미하다.
  특히 DNSMOS 는 CUDA/CPU 간에 ~3e-3 차이가 나므로, 학습이 GPU 를 채운 사이에
  한쪽만 CPU 로 떨어졌다면 여기서 걸린다.

사용법:
    python scripts/check_eval_done.py                      # LLM-TSE best/last (기본)
    python scripts/check_eval_done.py 'tpex|/path/to/dir'  # 임의의 (태그|추론디렉터리) 쌍
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

# 기본 대상 — eval_llmtse.sh 의 RUNS 와 같은 순서.
DEFAULT_RUNS = [
    ("best", "/home/work/my-outputs/llmtse/llmtse_baseline_v1"),
    ("last", "/home/work/my-outputs/llmtse/llmtse_baseline_v1_step50000"),
]

# 보고할 지표. wer 은 corpus micro-WER(총 edits / 총 words)로 집계되며 행별 WER 의
# 평균과 다르다 — 논문에서 보고하는 쪽은 micro 다.
KEYS = ("si_sdr", "si_sdri", "input_si_sdr", "stoi", "estoi", "pesq",
        "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "dnsmos_p808", "wer", "spk_sim",
        # Companions — the un-normalised variants, shown so a conclusion can be
        # checked against them rather than resting on the normalisation choice.
        "dnsmos_ovrl_clipped", "wer_raw")

LOWER_IS_BETTER = {"wer", "wer_raw"}


def main(argv: "list[str]") -> int:
    runs = ([tuple(a.split("|", 1)) for a in argv] if argv else DEFAULT_RUNS)

    done, summaries, meta = [], [], []
    for tag, d in runs:
        ev = os.path.join(d, "eval")
        paths = {k: os.path.join(ev, f"llmtse_{tag}{sfx}") for k, sfx in
                 (("per", ".csv"), ("sum", "_summary.csv"), ("cfg", "_config.json"))}
        missing = [os.path.basename(p) for p in paths.values() if not os.path.isfile(p)]
        if missing:
            print(f"  ✗ [{tag}] 미완료 — 없는 파일: {missing}")
            continue

        df = pd.read_csv(paths["per"])
        err = 0
        if "error" in df.columns:
            err = int((df["error"].notna()
                       & (df["error"].astype(str).str.strip() != "")).sum())
        nan_metrics = {c: int(df[c].isna().sum()) for c in KEYS
                       if c in df.columns and df[c].isna().any()}

        s = pd.read_csv(paths["sum"])
        # ★ 축을 지정하면 summarize() 는 축별 group="ALL" 행만 만들고 axis="ALL" 행은
        #   만들지 않는다. 축별 ALL 행은 모두 전체 프레임 집계라 아무거나 하나면 된다.
        allrow = s[s["group"] == "ALL"]
        if allrow.empty:
            print(f"  ✗ [{tag}] 요약에 group='ALL' 행이 없습니다")
            continue

        flag = "" if (err == 0 and not nan_metrics) else "  ← 확인 필요"
        print(f"  ✓ [{tag}] {len(df)}행 · 에러 {err} · NaN {nan_metrics or '없음'} "
              f"· 축 {sorted(s['axis'].unique())}{flag}")
        done.append(tag)
        summaries.append(allrow.iloc[[0]].assign(model_name=f"llmtse_{tag}"))
        meta.append((tag, json.load(open(paths["cfg"]))))

    print()
    if len(done) != len(runs):
        print(f"  ⏳ 아직 진행 중입니다 ({len(done)}/{len(runs)} 완료). "
              "byobu 창에 '완료' 배너가 뜬 뒤 다시 실행하세요.")
        return 1

    # ── 전체 성능 ─────────────────────────────────────────────────────────
    cmp = pd.concat(summaries, ignore_index=True)
    cols = ["model_name", "n"] + [c for c in KEYS if c in cmp.columns]
    print("  ── 전체 성능 (group=ALL) ───────────────────────────────────────")
    print(cmp[cols].to_string(index=False))

    if len(cmp) == 2:
        a_name, b_name = cmp.iloc[0]["model_name"], cmp.iloc[1]["model_name"]
        print()
        print(f"  ── delta ({b_name} − {a_name}) ─────────────────────────────")
        for c in KEYS:
            if c not in cmp.columns:
                continue
            a, b = float(cmp.iloc[0][c]), float(cmp.iloc[1][c])
            diff = b - a
            if abs(diff) < 1e-9:
                mark = "  (동일)"
            else:
                better = (diff < 0) if c in LOWER_IS_BETTER else (diff > 0)
                mark = "  ↑ 개선" if better else "  ↓ 악화"
            note = "   ※ 낮을수록 좋음" if c in LOWER_IS_BETTER else ""
            print(f"    {c:14s} {a:9.4f} → {b:9.4f}   {diff:+8.4f}{mark}{note}")
        # input_si_sdr 은 예측과 무관(mixture vs reference)하므로 두 런이 같아야 한다.
        if "input_si_sdr" in cmp.columns:
            d0 = abs(float(cmp.iloc[1]["input_si_sdr"]) - float(cmp.iloc[0]["input_si_sdr"]))
            if d0 > 1e-9:
                print(f"    ⚠ input_si_sdr 이 두 런에서 다릅니다 ({d0:.3e}) — "
                      "mixture/reference 가 같다면 0 이어야 합니다.")

    # ── 출처 일치 ─────────────────────────────────────────────────────────
    print()
    print("  ── 출처(sidecar) 일치 확인 ────────────────────────────────────")
    fields = {
        tag: {
            "si_sdr_backend": cfg.get("si_sdr_backend"),
            "target_sr": cfg.get("target_sr"),
            "metrics": len(cfg.get("metrics") or []),
            "dnsmos_actual": tuple((cfg.get("dnsmos") or {}).get("actual_providers") or ()),
            # ★ Both normalisations change the reported numbers (DNSMOS ~0.34 MOS,
            #   WER ~7 pp), so a mismatch here invalidates the comparison just as
            #   surely as a different model would.
            "dnsmos_normalize": (cfg.get("dnsmos") or {}).get("normalize"),
            "wer_normalize": (cfg.get("wer") or {}).get("normalize"),
            "wer_model": (cfg.get("wer") or {}).get("model_id"),
            "spk_model": (cfg.get("spk_sim") or {}).get("model_id"),
        }
        for tag, cfg in meta
    }
    mismatch = []
    for k in next(iter(fields.values())):
        vals = {t: fields[t][k] for t in done}
        same = len({str(v) for v in vals.values()}) == 1
        print(f"    {'✓' if same else '✗'} {k:16s} {vals[done[0]]}"
              + ("" if same else f"   ← 불일치 {vals}"))
        if not same:
            mismatch.append(k)

    print()
    if mismatch:
        print(f"  ✗ 두 런의 채점 조건이 다릅니다: {mismatch}")
        print("    → 이 비교는 무효입니다. 조건을 맞춰 다시 채점하세요 (FORCE=1).")
        return 1
    print("  ✓ 두 런이 동일한 조건으로 채점됐습니다 — 비교가 유효합니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
