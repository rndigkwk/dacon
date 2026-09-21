#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""덤프에 융합규칙을 적용해 제출 파일을 다시 만든다 — 모델 재실행 없이.

  python3 script.py --input dev.jsonl.gz --output-dir output/fit --dump-parts output/fit --collect
  python3 tools/fit_fusion.py output/fit --base song --margin 0.05 --write
  python3 tools/apply_fusion.py output/fit --out output/fit/submission_fused.csv
  python3 tools/score_dev.py output/fit/submission_fused.csv

융합규칙만 바꿔 점수를 다시 보고 싶을 때 쓴다. LLM 콜이 들지 않는다.
판정 로직은 script.fuse_record 를 그대로 호출하므로 실제 실행과 어긋나지 않는다.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import script as S                                   # noqa: E402
from model import oh_engine as OH                    # noqa: E402

ITEMS = S.ITEMS


def load_csv(path):
    return {r["id"]: r for r in csv.DictReader(io.open(path, encoding="utf-8"))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("parts_dir", help="script.py --dump-parts 가 만든 디렉토리")
    ap.add_argument("--asset-dir", default=os.path.join(ROOT, "model"))
    ap.add_argument("--out", default=None, help="기본 = <parts_dir>/submission_fused.csv")
    a = ap.parse_args()

    out = a.out or os.path.join(a.parts_dir, "submission_fused.csv")
    rule = load_csv(os.path.join(a.parts_dir, "layer_rule.csv"))
    song = load_csv(os.path.join(a.parts_dir, "layer_song.csv"))
    prob = load_csv(os.path.join(a.parts_dir, "layer_oh_probs.csv"))
    ev = {}
    with io.open(os.path.join(a.parts_dir, "layer_evidence.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ev[d["id"]] = d

    fusion = S.Fusion(a.asset_dir)
    assets = OH.Assets(a.asset_dir)
    print(f"융합규칙: {json.dumps(fusion.summary(), ensure_ascii=False)}", file=sys.stderr)

    rows, used = [], {}
    for rid in rule:
        d = ev.get(rid, {})
        # 덤프의 근거는 이미 원문 대조를 통과한 문자열이다. fuse_record 가 한 번 더
        # clean_evidence 를 거치므로, 원문 대신 근거 자신을 src 로 주어 통과시킨다.
        j_song = {v: {"hit": int(song[rid][v] == "1"), "ev": (d.get("song_ev") or {}).get(v)}
                  for v in ITEMS}
        j_rule = {v: {"hit": int(rule[rid][v] == "1"), "ev": (d.get("rule_ev") or {}).get(v)}
                  for v in ITEMS}
        p = {v: float(prob[rid][v]) for v in ITEMS}
        oh_ev = {k: x for k, x in (d.get("oh_ev") or {}).items() if x}
        src = "\n".join([x for x in list((d.get("song_ev") or {}).values())
                         + list((d.get("rule_ev") or {}).values())
                         + list(oh_ev.values()) if x]) or " "
        decided, cells, u = S.fuse_record(j_song, j_rule, p, oh_ev, fusion, assets, src)
        rows.append(S.to_row(rid, decided, cells))
        for k, n in u.items():
            used[k] = used.get(k, 0) + n

    S.write_csv(rows, out)
    errs = S.validate_csv(out, list(rule))
    print(json.dumps({"건수": len(rows), "융합_채택": used,
                      "양성_합계": sum(r[v] for r in rows for v in ITEMS),
                      "출력": out, "자가검증": "PASS" if not errs else errs[:5]},
                     ensure_ascii=False), file=sys.stderr)
    return 0 if not errs else 1


if __name__ == "__main__":
    sys.exit(main())
