#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dev 채점 — 항목별/Macro 양성 F1. 리더보드 지표와 같은 계산이다.

  python3 tools/score_dev.py output/dev/submission.csv
  python3 tools/score_dev.py output/dev/submission.csv --gate-ceiling
"""
from __future__ import annotations
import argparse, csv, io, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 레포 루트
for _c in (ROOT, os.path.join(ROOT, "engines")):      # 평탄 배치 · baseline/ 배치 둘 다 지원
    if os.path.exists(os.path.join(_c, "script.py")):
        sys.path.insert(0, _c)
        break

ITEMS = [f"v{i}" for i in range(1, 25)]


def load(path: str) -> dict:
    return {r["id"]: r for r in csv.DictReader(io.open(path, encoding="utf-8"))}


def f1(tp: int, fp: int, fn: int) -> float:
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pred")
    ap.add_argument("--labels", default=os.path.join(ROOT, "dev_labels.csv"))
    ap.add_argument("--gate-ceiling", action="store_true",
                    help="메타 게이트만 적용하고 통과 항목을 전부 1로 찍었을 때의 상한 F1")
    a = ap.parse_args()

    gold, pred = load(a.labels), load(a.pred)
    missing = set(gold) - set(pred)
    if missing:
        print(f"[경고] 예측에 없는 id {len(missing)}건 — 전부 0으로 간주합니다", file=sys.stderr)

    if a.gate_ceiling:
        import gzip, json
        from script import ASSET_DIR, Assets, derive_facts, applicable
        recs = {json.loads(l)["id"]: json.loads(l)
                for l in gzip.open(os.path.join(ROOT, "dev.jsonl.gz"), "rt", encoding="utf-8")}
        assets = Assets(ASSET_DIR)
        pred = {}
        for rid, rec in recs.items():
            f = derive_facts(rec, assets)
            pred[rid] = {"id": rid, **{v: str(int(applicable(v, f, assets))) for v in ITEMS}}

    rows, macro = [], 0.0
    for v in ITEMS:
        tp = fp = fn = 0
        for rid, g in gold.items():
            y = int(g[v] == "1")
            p = int((pred.get(rid, {}).get(v) or "0") == "1")
            tp += y & p
            fp += (1 - y) & p
            fn += y & (1 - p)
        s = f1(tp, fp, fn)
        macro += s / len(ITEMS)
        rows.append((v, tp, fp, fn, s))

    print(f"{'항목':>5} {'TP':>4} {'FP':>5} {'FN':>4} {'F1':>7}")
    for v, tp, fp, fn, s in rows:
        print(f"{v:>5} {tp:4d} {fp:5d} {fn:4d} {s:7.4f}")
    print(f"\nMacro F1 = {macro:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
