#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""제출 ZIP 작성 — script.py · requirements.txt · model/ 만 담는다.

  python3 tools/make_submit.py                 # → submit.zip
  python3 tools/make_submit.py --out /tmp/x.zip

담기 전에 규칙 위반 소지를 검사한다. 하나라도 걸리면 ZIP 을 만들지 않는다.
  · LLM 가중치·LoRA 어댑터 동봉 금지 (.safetensors · .bin · .pt · .gguf …)
  · 제출 코드에서 외부 네트워크·외부 LLM 호출 금지
  · 평가 데이터 id·건수 하드코딩 금지
  · model/ 자산이 실제로 담겼는지 — 빠지면 크래시 없이 점수만 조용히 떨어진다
  · 2GB 이하
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INCLUDE = ["script.py", "requirements.txt", "model"]
SKIP_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints"}
WEIGHTS = re.compile(r"\.(safetensors|bin|pt|pth|gguf|ckpt|onnx|h5|msgpack)$", re.I)
NET = re.compile(r"\b(requests|urllib\.request|urllib3|httpx|aiohttp|socket|openai|anthropic)\b")
MUST_HAVE = ["게이트.json", "임계값.json", "조문발췌.json", "경쟁제품코드.json",
             "융합규칙.json", "competitive_products.csv", "law.py", "features.py"]
LIMIT = 2 * 1024 ** 3


def collect() -> list:
    out = []
    for name in INCLUDE:
        p = os.path.join(ROOT, name)
        if os.path.isfile(p):
            out.append((p, name))
        elif os.path.isdir(p):
            for base, dirs, files in os.walk(p):
                dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
                for fn in sorted(files):
                    full = os.path.join(base, fn)
                    out.append((full, os.path.relpath(full, ROOT)))
        else:
            print(f"[오류] 없음: {name}", file=sys.stderr)
            return []
    return sorted(out, key=lambda x: x[1])


def check(files: list) -> list:
    errs = []
    names = {rel for _f, rel in files}
    for full, rel in files:
        if WEIGHTS.search(rel):
            errs.append(f"모델 가중치로 보이는 파일: {rel}")
    for must in MUST_HAVE:
        if not any(os.path.basename(r) == must for r in names):
            errs.append(f"필수 자산 누락: model/{must}")
    for full, rel in files:
        if not rel.endswith(".py"):
            continue
        src = io.open(full, encoding="utf-8", errors="replace").read()
        for m in NET.finditer(src):
            line = src[:m.start()].count("\n") + 1
            frag = src.splitlines()[line - 1].strip()
            if frag.startswith("#") or "socket" in frag and "#" in frag:
                continue
            errs.append(f"외부 통신 의심 {rel}:{line} — {frag[:80]}")
        if "PPS-" in src:
            errs.append(f"평가 데이터 id 하드코딩 의심: {rel}")
    total = sum(os.path.getsize(f) for f, _ in files)
    if total > LIMIT:
        errs.append(f"용량 초과 {total / 1024 ** 3:.2f}GB > 2GB")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "submit.zip"))
    ap.add_argument("--force", action="store_true", help="검사 실패해도 만든다")
    a = ap.parse_args()

    files = collect()
    if not files:
        return 1
    errs = check(files)
    for e in errs:
        print(f"[검사] {e}", file=sys.stderr)
    if errs and not a.force:
        print(f"\n검사 {len(errs)}건 실패 — ZIP 을 만들지 않았습니다.", file=sys.stderr)
        return 1

    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for full, rel in files:
            z.write(full, rel)
    size = os.path.getsize(a.out)
    print(f"{a.out}  ({size / 1024:.0f} KB, {len(files)}개 파일)")
    for _f, rel in files:
        print(f"  {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
