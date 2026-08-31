#!/usr/bin/env python
"""scripts/check_eval_done.py — 채점이 끝났는지 확인하고, 끝났으면 결과를 비교한다.

eval_<project>.sh 가 남긴 산출물 3종(per-row CSV · summary CSV · sidecar JSON)이
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
    python scripts/check_eval_done.py                    # 기본(llmtse v2)의 best/last
    python scripts/check_eval_done.py styletse           # 프로젝트의 기본 버전(v1)
    python scripts/check_eval_done.py llmtse v1          # 프로젝트 + 버전
    python scripts/check_eval_done.py 'styletse_v1_best|/path/to/dir' ...
                                                         # 출력 stem 과 디렉터리를 직접
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

# eval_llmtse.sh 의 VERSION 과 같은 개념. 인자 없이 실행하면 이 버전을 본다.
# eval_<project>.sh 의 VERSION 과 같은 개념. 인자 없이 실행하면 이 조합을 본다.
DEFAULT_PROJECT = os.environ.get("PROJECT", "llmtse")
DEFAULT_VERSION = os.environ.get("VERSION", "")   # 빈 값 = 프로젝트별 기본

# 프로젝트 규약: 출력 루트 · 디렉터리 이름 패턴 · 기본 버전.
# eval_llmtse.sh / eval_styletse.sh 의 RUNS 유도와 같은 규약이어야 한다.
_PROJECTS = {
    "llmtse":   dict(root="/home/work/my-outputs/llmtse",   default="v2",
                     best="llmtse_baseline_{v}", last="llmtse_baseline_{v}_step50000"),
    "styletse": dict(root="/home/work/my-outputs/styletse", default="v1",
                     best="styletse_{v}_best",   last="styletse_{v}_last"),
}

# LLM-TSE V1 은 VERSION 이 생기기 전에 채점돼 태그에 버전 접두사가 없다
# (llmtse_best.csv). 이 표가 없으면 `check_eval_done.py llmtse v1` 이
# llmtse_v1_best.csv 를 찾다가 실패한다.
_LEGACY_TAGS = {("llmtse", "v1"): ("best", "last")}


def _runs_for(project: str, version: str = "") -> "list[tuple[str, str]]":
    """(프로젝트, 버전) -> [(출력 stem, 추론 디렉터리)].

    stem 은 엔진이 실제로 쓰는 파일명 그대로다 — eval_tse.sh 의
    `${PROJECT}_${tag}.csv` 와 한 글자도 달라선 안 된다. 그래서 여기서 조립한다.
    """
    if project not in _PROJECTS:
        raise SystemExit(f"모르는 프로젝트: {project!r}. "
                         f"사용 가능: {sorted(_PROJECTS)}")
    cfg = _PROJECTS[project]
    v = version or cfg["default"]
    root = os.environ.get(f"{project.upper()}_ROOT", cfg["root"])
    best, last = _LEGACY_TAGS.get((project, v), (f"{v}_best", f"{v}_last"))
    return [(f"{project}_{best}", f"{root}/" + cfg["best"].format(v=v)),
            (f"{project}_{last}", f"{root}/" + cfg["last"].format(v=v))]


# 보고할 지표. wer 은 corpus micro-WER(총 edits / 총 words)로 집계되며 행별 WER 의
# 평균과 다르다 — 논문에서 보고하는 쪽은 micro 다.
KEYS = ("si_sdr", "si_sdri", "input_si_sdr", "stoi", "estoi", "pesq",
        "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "dnsmos_p808", "wer", "spk_sim",
        # Companions — the un-normalised variants, shown so a conclusion can be
        # checked against them rather than resting on the normalisation choice.
        "dnsmos_ovrl_clipped", "wer_raw")

LOWER_IS_BETTER = {"wer", "wer_raw"}


_USAGE = ("사용법: (인자 없음) | <프로젝트> [버전] | '<stem>|<디렉터리>' ...\n"
          f"        프로젝트: {sorted(_PROJECTS)}")


def _parse_args(argv: "list[str]") -> "list[tuple[str, str]]":
    """인자 -> [(출력 stem, 추론 디렉터리)].

    ★ 잘못된 형식은 SystemExit 으로 **명확히** 알린다. 이 스크립트는
    `check_eval_done.py && ...` 처럼 조건부 실행에 쓰라고 문서화돼 있어,
    traceback 이 나면 원인을 알 수 없는 채로 실패한 것과 구별되지 않는다.
    """
    if not argv:
        return _runs_for(DEFAULT_PROJECT, DEFAULT_VERSION)

    piped = [a for a in argv if "|" in a]
    if piped and len(piped) != len(argv):
        raise SystemExit(f"인자를 섞어 쓸 수 없습니다: {argv}\n{_USAGE}")

    if piped:
        runs = []
        for a in argv:
            stem, _, d = a.partition("|")
            if not stem or not d:
                raise SystemExit(f"형식 오류: {a!r} — '<stem>|<디렉터리>' 여야 합니다\n{_USAGE}")
            runs.append((stem, d))
        return runs

    if len(argv) > 2:
        raise SystemExit(f"인자가 너무 많습니다: {argv}\n{_USAGE}")

    proj = argv[0]
    ver = argv[1] if len(argv) == 2 else DEFAULT_VERSION
    # 구 호출법 하위호환: 버전만 준 경우(`check_eval_done.py v1`).
    if proj not in _PROJECTS and proj.startswith("v"):
        proj, ver = DEFAULT_PROJECT, argv[0]
    return _runs_for(proj, ver)


def main(argv: "list[str]") -> int:
    runs = _parse_args(argv)

    done, summaries, meta = [], [], []
    for tag, d in runs:          # tag = 출력 stem (예: styletse_v1_best)
        ev = os.path.join(d, "eval")
        paths = {k: os.path.join(ev, f"{tag}{sfx}") for k, sfx in
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
        summaries.append(allrow.iloc[[0]].assign(model_name=tag))
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
