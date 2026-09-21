#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""융합규칙 학습 — 항목별로 세 원천을 어떻게 합칠지 고른다.

  python3 script.py --input dev.jsonl.gz --output-dir output/fit --dump-parts output/fit --collect
  python3 tools/fit_fusion.py output/fit                  # 후보 비교만
  python3 tools/fit_fusion.py output/fit --write          # model/융합규칙.json 갱신

원천 셋
  rule  A계층을 LLM 없이 규칙 사실만으로 판정한 값
  song  A계층에 LLM 사실을 융합해 판정한 값
  oh    B계층의 P(위반) 을 임계값으로 자른 값

후보
  단독 셋 + 쌍 조합(or·and) 셋. oh 가 끼면 임계값 격자를 함께 훑는다.
  항목당 약 250개 후보를 비교한다.

왜 마진과 교차검증이 필요한가
  dev 200건은 항목당 양성이 5~8개다. 24항목 × 250후보에서 최댓값을 고르면
  그중 상당수는 잡음이다. 기준 원천(--base, 기본 rule)보다 --margin 이상
  좋아야만 바꾸고, --cv 로 '이 선택 절차 자체'를 교차검증한다.
  적합 점수(고른 뒤 같은 데이터로 잰 값)는 항상 낙관적이므로 그대로 믿지 말 것.

규칙
  여기서 나온 값은 반드시 상수로 박아 제출한다. 런타임에 테스트셋을 보고
  모드나 임계값을 정하면 샘플 간 예측 의존 금지 규칙 위반이다.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ITEMS = [f"v{i}" for i in range(1, 25)]
GRID = [round(i / 50, 2) for i in range(1, 50)]          # 0.02 … 0.98
SOURCES = ("rule", "song", "oh")
PAIRS = (("rule", "song"), ("rule", "oh"), ("song", "oh"))

# 후보 = (모드, 원천쌍 or None, 임계값)
Cand = Tuple[str, Optional[Tuple[str, str]], float]


def f1(tp: int, fp: int, fn: int) -> float:
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def load_csv(path: str) -> Dict[str, Dict[str, str]]:
    return {r["id"]: r for r in csv.DictReader(io.open(path, encoding="utf-8"))}


def candidates(allow: Sequence[str] = SOURCES) -> List[Cand]:
    """allow 로 쓸 원천을 제한한다. oh 를 빼면 임계값 튜닝이 사라져 후보가 4개로 줄고,
    그만큼 선택 잡음도 줄어든다."""
    allow = tuple(allow)
    out: List[Cand] = [(s, None, 0.0) for s in ("rule", "song") if s in allow]
    if "oh" in allow:
        out += [("oh", None, t) for t in GRID]
    for pair in PAIRS:
        if not all(x in allow for x in pair):
            continue
        for mode in ("or", "and"):
            if "oh" in pair:
                out += [(mode, pair, t) for t in GRID]
            else:
                out.append((mode, pair, 0.0))
    return out


def hit_of(src: str, thr: float, rid: str, v: str, rule, song, prob) -> int:
    if src == "rule":
        return int(rule[rid][v] == "1")
    if src == "song":
        return int(song[rid][v] == "1")
    return int(float(prob[rid][v]) >= thr)


def score(v: str, cand: Cand, idx: Sequence[str], gold, rule, song, prob) -> Tuple[float, int, int, int]:
    mode, pair, thr = cand
    tp = fp = fn = 0
    for rid in idx:
        y = int(gold[rid][v] == "1")
        if mode in SOURCES:
            p = hit_of(mode, thr, rid, v, rule, song, prob)
        else:
            a = hit_of(pair[0], thr, rid, v, rule, song, prob)
            b = hit_of(pair[1], thr, rid, v, rule, song, prob)
            p = (a | b) if mode == "or" else (a & b)
        tp += y & p
        fp += (1 - y) & p
        fn += y & (1 - p)
    return f1(tp, fp, fn), tp, fp, fn


def best_for(v: str, idx: Sequence[str], gold, rule, song, prob,
             margin: float, base: str, cands: List[Cand]) -> Tuple[Cand, float]:
    """기준 원천을 마진만큼 넘는 후보만 채택한다."""
    base_cand: Cand = (base, None, 0.0)
    base_f1 = score(v, base_cand, idx, gold, rule, song, prob)[0]
    best, best_f1 = base_cand, base_f1
    for c in cands:
        s = score(v, c, idx, gold, rule, song, prob)[0]
        if s > best_f1 + 1e-12:
            best, best_f1 = c, s
    if best != base_cand and best_f1 < base_f1 + margin:
        return base_cand, base_f1
    return best, best_f1


def cross_val(ids: List[str], gold, rule, song, prob, margin: float, base: str,
              cands: List[Cand], k: int) -> float:
    """선택 절차 자체를 k-겹으로 검증한다. 폴드마다 학습 폴드에서 고르고 검증 폴드에 적용."""
    folds = [ids[i::k] for i in range(k)]
    macro = 0.0
    for v in ITEMS:
        tp = fp = fn = 0
        for i in range(k):
            test = folds[i]
            train = [x for j, f in enumerate(folds) if j != i for x in f]
            c, _ = best_for(v, train, gold, rule, song, prob, margin, base, cands)
            _, a, b, d = score(v, c, test, gold, rule, song, prob)
            tp, fp, fn = tp + a, fp + b, fn + d
        macro += f1(tp, fp, fn) / len(ITEMS)
    return macro


def label(c: Cand) -> str:
    mode, pair, thr = c
    if mode in SOURCES:
        return mode if mode != "oh" else f"oh@{thr:.2f}"
    tag = f"{mode}({pair[0]}|{pair[1]})"
    return tag + (f"@{thr:.2f}" if "oh" in pair else "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("parts_dir", help="script.py --dump-parts 가 만든 디렉토리")
    ap.add_argument("--labels", default=os.path.join(ROOT, "dev_labels.csv"))
    ap.add_argument("--margin", type=float, default=0.05,
                    help="기준 원천 대비 이만큼 이상 좋아야 바꾼다")
    ap.add_argument("--base", default="rule", choices=list(SOURCES),
                    help="기준 원천 — 마진 비교의 기준이자 기본 모드")
    ap.add_argument("--cv", type=int, default=5, help="교차검증 겹 수 (0이면 생략)")
    ap.add_argument("--sources", default="rule,song,oh",
                    help="후보에 쓸 원천 (쉼표 구분). 예: rule,song — oh 를 빼면 임계값 튜닝이 없어진다")
    ap.add_argument("--write", action="store_true", help="model/융합규칙.json 갱신")
    a = ap.parse_args()

    gold = load_csv(a.labels)
    rule = load_csv(os.path.join(a.parts_dir, "layer_rule.csv"))
    song = load_csv(os.path.join(a.parts_dir, "layer_song.csv"))
    prob = load_csv(os.path.join(a.parts_dir, "layer_oh_probs.csv"))
    ids = sorted(i for i in gold if i in rule and i in song and i in prob)
    if len(ids) != len(gold):
        print(f"[경고] 라벨 {len(gold)}건 중 {len(ids)}건만 덤프에 있습니다", file=sys.stderr)

    allow = tuple(x.strip() for x in a.sources.split(",") if x.strip() in SOURCES)
    cands = candidates(allow)
    print(f"후보 원천 {allow} · 후보 수 {len(cands)}", file=sys.stderr)
    singles = {s: 0.0 for s in SOURCES}
    rows, macro_fit = [], 0.0
    rules_out: Dict[str, Dict[str, object]] = {}

    for v in ITEMS:
        for s in SOURCES:
            if s == "oh":
                singles[s] += max(score(v, ("oh", None, t), ids, gold, rule, song, prob)[0]
                                  for t in GRID) / len(ITEMS)
            else:
                singles[s] += score(v, (s, None, 0.0), ids, gold, rule, song, prob)[0] / len(ITEMS)
        c, fit = best_for(v, ids, gold, rule, song, prob, a.margin, a.base, cands)
        _, tp, fp, fn = score(v, c, ids, gold, rule, song, prob)
        mode, pair, thr = c
        d: Dict[str, object] = {"모드": mode}
        if pair:
            d["원천"] = list(pair)
        if mode == "oh" or (pair and "oh" in pair):
            d["임계값"] = thr
        rules_out[v] = d
        macro_fit += fit / len(ITEMS)
        rows.append((v, c, fit, tp, fp, fn))

    base_macro = 0.0
    for v in ITEMS:
        base_macro += score(v, (a.base, None, 0.0), ids, gold, rule, song, prob)[0] / len(ITEMS)

    print(f"{'항목':>5}  {'채택':>20} {'적합F1':>7} {'TP':>4} {'FP':>5} {'FN':>4}")
    for v, c, fit, tp, fp, fn in rows:
        mark = "" if c == (a.base, None, 0.0) else "  ←"
        print(f"{v:>5}  {label(c):>20} {fit:7.4f} {tp:4d} {fp:5d} {fn:4d}{mark}")

    print()
    for s in SOURCES:
        tag = " (항목별 최적 임계값)" if s == "oh" else ""
        print(f"  {s:>5} 단독 Macro F1 = {singles[s]:.4f}{tag}")
    print(f"\n  기준({a.base}) Macro F1  = {base_macro:.4f}")
    print(f"  융합 적합 Macro F1   = {macro_fit:.4f}   ← 같은 데이터로 골라 같은 데이터로 잼. 낙관적")
    if a.cv:
        cv = cross_val(ids, gold, rule, song, prob, a.margin, a.base, cands, a.cv)
        print(f"  융합 {a.cv}겹 교차검증    = {cv:.4f}   ← 이 값으로 판단할 것")
        if cv < base_macro:
            print(f"  [주의] 교차검증이 기준 원천 단독({base_macro:.4f})보다 낮습니다. "
                  f"--margin 을 올리거나 융합을 보류하십시오.")

    if a.write:
        out = os.path.join(ROOT, "model", "융합규칙.json")
        doc = {
            "설명": "항목별 세 원천(rule·song·oh) 융합 방식. tools/fit_fusion.py 오프라인 결과만 "
                   "반영한다. 런타임에 바꾸면 샘플 간 예측 의존 금지 규칙 위반이다.",
            "학습": {"데이터": os.path.basename(a.labels), "건수": len(ids),
                    "기준원천": a.base, "마진": a.margin, "겹": a.cv, "후보원천": list(allow),
                    "원천단독_MacroF1": {s: round(singles[s], 4) for s in SOURCES},
                    "적합_MacroF1": round(macro_fit, 4),
                    "교차검증_MacroF1": round(cross_val(ids, gold, rule, song, prob,
                                                     a.margin, a.base, cands, a.cv), 4)
                    if a.cv else None},
            "기본": a.base,
            "oh_기본임계값": 0.25,
            "규칙": rules_out,
        }
        with io.open(out, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
            f.write("\n")
        print(f"\n→ {out} 갱신")
    return 0


if __name__ == "__main__":
    sys.exit(main())
