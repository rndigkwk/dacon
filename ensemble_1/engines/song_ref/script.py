#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""나라장터 자체입찰 공고 법령 위반사항 판정 파이프라인.

  data/test.jsonl.gz  →  output/submission.csv (id, v1..v24, e1..e24)

구조
  1) 규칙 계층(model/features.py)  : 참가자격 섹션·금액·지역·실적 등 후보 추출
  2) 고정 LLM(사실 추출)           : 공고문에서 '사실' 12가지만 구조화 출력으로 추출
  3) 법령 적용 계층(model/law.py)  : 사실 + 나라장터 메타 → 24개 항목 위반 판정
  4) 근거 복구·검증(model/fuse.py) : 근거문구를 원문 부분문자열로 되살리고 검증

LLM 은 법 해석을 하지 않는다. 금액 구간 비교·조문 적용은 전부 코드가 한다.
LLM 출력이 비거나 깨져도 규칙 사실로 폴백하므로 파이프라인은 항상 24열을 채운다.

로컬 실행
  python script.py --mock          # 모델 없이 규칙만으로 제출 파일 생성
  python script.py --limit 10      # 앞 10건만
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
import re
import sys
import time
import unicodedata
from typing import Any, Dict, Iterator, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from model import law as LAW                     # noqa: E402
from model import fuse as FUSE                   # noqa: E402
from model import prompt as PROMPT               # noqa: E402
from model.extract_schema import FACT_SCHEMA     # noqa: E402
from model.features import extract as extract_signals   # noqa: E402

DATA_DIR = os.environ.get("PPS_DATA_DIR", "./data")
OUTPUT_DIR = os.environ.get("PPS_OUTPUT_DIR", "./output")
MODEL_DIR = os.environ.get("PPS_MODEL_DIR", "/opt/models/gemma-4-26B-A4B-it")

ITEMS = [f"v{i}" for i in range(1, 25)]
COLUMNS = ["id"] + ITEMS + [f"e{i}" for i in range(1, 25)]
ABSENCE = LAW.ABSENCE
EVIDENCE_MAX = 500

SEED = 20260826
MAX_MODEL_LEN = 16384
MAX_TOKENS = 2048
QUANT = "int8_per_channel_weight_only"


def log(msg: str) -> None:
    print(f"[pps] {msg}", file=sys.stderr, flush=True)


# ============================================================ 입력
def _open(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") \
        else io.open(path, "r", encoding="utf-8")


def validate_record(rec: Any) -> None:
    if not isinstance(rec, dict):
        raise ValueError("레코드가 object 가 아님")
    for k in ("id", "docs", "meta"):
        if k not in rec:
            raise ValueError(f"필수 키 없음: {k}")
    if not isinstance(rec["docs"], list) or not rec["docs"]:
        raise ValueError(f"docs 비어 있음 (id={rec['id']})")


def iter_records(path: str, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    n = 0
    with _open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno} JSON 파싱 실패: {e}") from e
            validate_record(rec)
            for d in rec["docs"]:
                d["text"] = unicodedata.normalize("NFC", d.get("text") or "")
                d["type"] = unicodedata.normalize("NFC", str(d.get("type") or "기타"))
            yield rec
            n += 1
            if limit and n >= limit:
                return


# ============================================================ 러너
class VLLMRunner:
    """평가 서버 고정 모델을 vLLM offline API 로 실행한다."""

    def __init__(self, schema, model_dir=MODEL_DIR, quant=QUANT,
                 max_tokens=MAX_TOKENS, seed=SEED, gpu_mem=0.92, tp=1):
        t0 = time.time()
        import vllm
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        log(f"vllm {vllm.__version__} · {model_dir} · quant={quant}")
        kw = dict(model=model_dir, tokenizer=model_dir, max_model_len=MAX_MODEL_LEN,
                  gpu_memory_utilization=gpu_mem, seed=seed,
                  tensor_parallel_size=tp, dtype="auto")
        if quant:
            kw["quantization"] = quant
        self.llm = LLM(**kw)
        self.tok = self.llm.get_tokenizer()
        self._SP, self._SOP = SamplingParams, StructuredOutputsParams
        self._schema, self._max_tokens, self._seed = schema, max_tokens, seed
        self.structured = True
        self.sp = self._make_sp(True)
        if self.sp is None:                       # 문법 컴파일 실패 → 비구조화로
            log("  ! 구조화 출력 설정 실패 → 비구조화 생성으로 시작")
            self.structured = False
            self.sp = self._make_sp(False)
        self.supports_system = self._probe_system()
        self.n_calls = 0          # 고정 LLM 호출 수
        self.n_ok = 0             # 그중 비어 있지 않은 정상 응답 수
        self.load_seconds = time.time() - t0

    def _make_sp(self, structured: bool):
        kw = dict(temperature=0.0, max_tokens=self._max_tokens, seed=self._seed)
        if not structured:
            return self._SP(**kw)
        try:
            return self._SP(structured_outputs=self._SOP(json=self._schema,
                                                         disable_any_whitespace=True), **kw)
        except Exception as e:
            log(f"  ! StructuredOutputsParams 생성 실패: {type(e).__name__}: {str(e)[:160]}")
            return None

    def _probe_system(self) -> bool:
        """chat template 이 system 롤을 받는지 확인한다 (Gemma 계열은 대개 거부)."""
        try:
            self.tok.apply_chat_template(
                [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
                add_generation_prompt=True, tokenize=False)
            return True
        except Exception as e:
            log(f"  · chat template 이 system 롤을 지원하지 않음 ({type(e).__name__}) "
                f"→ user 턴 하나로 합쳐 보냄")
            return False

    def set_structured(self, on: bool) -> bool:
        sp = self._make_sp(on)
        if sp is None:
            return False
        self.structured, self.sp = on, sp
        return True

    def count_tokens(self, messages) -> int:
        try:
            ids = self.tok.apply_chat_template(messages, add_generation_prompt=True,
                                               tokenize=True)
            if hasattr(ids, "keys") and "input_ids" in ids:
                ids = ids["input_ids"]
            return len(ids)
        except Exception:
            return len(self.tok.encode("\n".join(m["content"] for m in messages)))

    def chat(self, batch):
        outs = self.llm.chat(batch, sampling_params=self.sp, use_tqdm=False)
        texts = [o.outputs[0].text if o.outputs else "" for o in outs]
        self.n_calls += len(batch)
        self.n_ok += sum(1 for t in texts if t and t.strip())
        return texts


class MockRunner:
    """모델 없이 규칙 계층만으로 동작시킨다 (형식·흐름 점검용)."""
    load_seconds = 0.0
    supports_system = False
    structured = False

    def __init__(self, schema=None, **_):
        self.n_calls = self.n_ok = 0

    def set_structured(self, on: bool) -> bool:
        return False

    def count_tokens(self, messages) -> int:
        return sum(len(m["content"]) for m in messages) // 2

    def chat(self, batch):
        return ["" for _ in batch]


def run_chunk(runner, batch):
    try:
        return runner.chat(batch)
    except Exception as e:
        log(f"  ! 청크({len(batch)}건) 실패 → 건별 재시도: {type(e).__name__}: {str(e)[:160]}")
    outs = []
    for m in batch:
        try:
            outs.append(runner.chat([m])[0])
        except Exception as e:
            log(f"  ! 건별 실패 → 빈 출력: {type(e).__name__}: {str(e)[:120]}")
            outs.append("")
    return outs


# ============================================================ 파싱·출력
FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


def extract_json(text: str):
    text = (text or "").strip()
    if not text:
        return None
    cands = [text] + [m.group(1) for m in FENCE.finditer(text)]
    i, j = text.find("{"), text.rfind("}")
    if i >= 0 and j > i:
        cands.append(text[i:j + 1])
    for c in cands:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    return None


def clean_evidence(ev, src: str) -> str:
    """근거문구 셀 규약: NFC · 500자 이하 · 원문 부분문자열 · 수식 접두 금지."""
    if not ev:
        return ""
    s = ev[2] if isinstance(ev, (tuple, list)) and len(ev) >= 3 else str(ev)
    s = unicodedata.normalize("NFC", s).replace("\r", "").strip()
    if not s:
        return ""
    if len(s) > EVIDENCE_MAX:
        s = s[:EVIDENCE_MAX].rstrip()
    while s and s not in src:
        s = s[:-1].rstrip()
        if len(s) < 10:
            return ""
    if not s or s[0] in "=+@":
        return ""
    return s


def to_row(rec_id: str, judged: Dict[str, Dict[str, Any]], src: str) -> Dict[str, Any]:
    row = {"id": rec_id}
    for i, v in enumerate(ITEMS, 1):
        cell = judged.get(v) or {"hit": 0, "ev": None}
        hit = 1 if cell.get("hit") == 1 else 0
        row[v] = hit
        row[f"e{i}"] = "" if (hit == 0 or v in ABSENCE) else clean_evidence(cell.get("ev"), src)
    return row


def empty_row(rec_id: str) -> Dict[str, Any]:
    row = {"id": rec_id}
    for i, v in enumerate(ITEMS, 1):
        row[v], row[f"e{i}"] = 0, ""
    return row


def write_csv(rows, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with io.open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: unicodedata.normalize("NFC", str(r[k])) for k in COLUMNS})


def validate_csv(path: str, expected_ids: List[str]) -> List[str]:
    errs: List[str] = []
    with io.open(path, "r", encoding="utf-8", newline="") as f:
        rd = csv.reader(f)
        header = next(rd, None)
        rows = list(rd)
    if header != COLUMNS:
        return [f"헤더 불일치: {len(header or [])}열 (기대 {len(COLUMNS)})"]
    if len(rows) != len(expected_ids):
        errs.append(f"행 수 {len(rows)} ≠ 입력 {len(expected_ids)}")
    ids = [r[0] for r in rows]
    if len(set(ids)) != len(ids):
        errs.append("id 중복")
    if set(ids) != set(expected_ids):
        errs.append("id 집합 불일치")
    absence_idx = {COLUMNS.index("e" + v[1:]) for v in ABSENCE}
    for r in rows:
        if len(r) != len(COLUMNS):
            errs.append(f"{r[0]}: 열 수 {len(r)}")
            continue
        if any(x not in ("0", "1") for x in r[1:25]):
            errs.append(f"{r[0]}: v 열에 0/1 아닌 값")
        if any(len(x) > EVIDENCE_MAX for x in r[25:]):
            errs.append(f"{r[0]}: 근거문구 {EVIDENCE_MAX}자 초과")
        if any(r[j] for j in absence_idx):
            errs.append(f"{r[0]}: 부재탐지 항목에 근거문구")
        if any(x.startswith(("=", "+", "@")) for x in r[25:]):
            errs.append(f"{r[0]}: 수식 접두 근거문구")
    return errs


# ============================================================ 실행
def fit_budget(rec, sig, runner, chars: int, budget: int, use_sys: bool = False):
    while True:
        msgs = PROMPT.build_messages(rec, sig, budget_chars=chars, use_system=use_sys)
        n = runner.count_tokens(msgs)
        if n <= budget or chars <= 2500:
            return msgs, n
        chars = int(chars * min(0.85, budget / max(n, 1) * 0.95))


def run(input_path: str, out_path: str, runner_cls, limit, chunk, chars,
        debug_path: Optional[str] = None, **runner_kw) -> Dict[str, Any]:
    t_all = time.time()
    recs = list(iter_records(input_path, limit=limit))
    log(f"입력 {len(recs)}건 ← {input_path}")
    if not recs:
        write_csv([], out_path)
        return {"건수": 0, "자가검증": "PASS"}

    sigs, facts_rule = [], []
    for rec in recs:
        try:
            sig = extract_signals(rec)
        except Exception as e:                       # 한 건 실패가 전체를 막지 않도록
            log(f"  ! {rec['id']} 신호추출 실패: {type(e).__name__}: {e}")
            sig = None
        sigs.append(sig)
        facts_rule.append(LAW.rule_facts(sig) if sig else None)

    runner = runner_cls(FACT_SCHEMA, **runner_kw)
    log(f"러너 준비 {runner.load_seconds:.1f}s")

    budget = MAX_MODEL_LEN - MAX_TOKENS - 64

    def build_all(use_sys: bool):
        out, toks = [], []
        for rec, sig in zip(recs, sigs):
            if sig is None:
                out.append([{"role": "user", "content": "-"}])
                toks.append(0)
                continue
            m, n = fit_budget(rec, sig, runner, chars, budget, use_sys)
            out.append(m)
            toks.append(n)
        return out, toks

    use_sys = bool(getattr(runner, "supports_system", False))
    msgs_all, ntok = build_all(use_sys)
    if ntok:
        srt = sorted(ntok)
        log(f"프롬프트 토큰 중앙값 {srt[len(srt)//2]:,} / 최대 {max(ntok):,} (예산 {budget:,})")

    t_inf = time.time()
    texts: List[str] = []
    start = 0
    # --- 프로브 ---------------------------------------------------------
    # 대회 규칙: 평가 데이터의 각 공고에 대해 고정 LLM 정상 호출을 1회 이상 해야 한다.
    # chat template 의 system 롤 미지원이나 구조화 출력 문법 오류로 전 건이 빈 응답이 되면
    # 요건 미충족이 되므로, 앞 몇 건으로 실제 응답이 오는지 확인하고 설정을 바꿔가며 재시도한다.
    # (판정값이 아니라 '생성 설정'만 바꾸는 점검이며, 프로브 대상 건도 최종 설정으로 다시 생성한다.)
    probe_n = min(8, len(msgs_all))
    if probe_n and not isinstance(runner, MockRunner):
        base_sys = use_sys
        attempts = [("구조화+현재형식", True, base_sys),
                    ("비구조화+현재형식", False, base_sys),
                    ("구조화+형식토글", True, not base_sys),
                    ("비구조화+형식토글", False, not base_sys)]
        probe, ok = [], 0
        for label, want_struct, want_sys in attempts:
            if want_struct != runner.structured and not runner.set_structured(want_struct):
                continue                      # 해당 설정을 만들 수 없으면 건너뛴다
            if want_sys != use_sys:
                use_sys = want_sys
                msgs_all, _ = build_all(use_sys)
            probe = run_chunk(runner, msgs_all[:probe_n])
            ok = sum(1 for t in probe if isinstance(extract_json(t), dict))
            log(f"  프로브[{label}] {ok}/{probe_n}건 유효 JSON "
                f"(structured={runner.structured}, system={use_sys})")
            if ok:
                break
        if ok == 0:
            log("  [경고] 고정 LLM 에서 정상 응답을 받지 못했습니다. "
                "이 상태로 제출하면 대회 규칙상 '공고별 고정 LLM 정상 호출 1회 이상' "
                "요건을 충족하지 못해 무효 처리될 수 있습니다.")
        texts.extend(probe)
        start = probe_n
    for i in range(start, len(msgs_all), chunk):
        texts.extend(run_chunk(runner, msgs_all[i:i + chunk]))
        log(f"  {min(i + chunk, len(msgs_all))}/{len(msgs_all)}건 … {time.time() - t_inf:.0f}s")
    inf_seconds = time.time() - t_inf

    rows, n_json, n_ev, dbg = [], 0, 0, []
    for rec, sig, rf, txt in zip(recs, sigs, facts_rule, texts):
        try:
            if sig is None:
                rows.append(empty_row(rec["id"]))
                continue
            obj = extract_json(txt)
            n_json += int(isinstance(obj, dict))
            facts = FUSE.merge(obj, rf, sig)
            judged = LAW.judge(facts, sig)
            row = to_row(rec["id"], judged, sig["text"])
            n_ev += sum(1 for i in range(1, 25) if row[f"e{i}"])
            rows.append(row)
            if debug_path:
                dbg.append({"id": rec["id"],
                            "hits": [v for v in ITEMS if judged[v]["hit"]],
                            "why": {v: judged[v]["why"] for v in ITEMS if judged[v]["hit"]},
                            "llm_ok": isinstance(obj, dict)})
        except Exception as e:
            log(f"  ! {rec['id']} 판정 실패 → 전항목 0: {type(e).__name__}: {e}")
            rows.append(empty_row(rec["id"]))

    write_csv(rows, out_path)
    if debug_path:
        with io.open(debug_path, "w", encoding="utf-8") as f:
            for d in dbg:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    errs = validate_csv(out_path, [r["id"] for r in recs])
    report = {
        "건수": len(recs), "모델로드_s": round(runner.load_seconds, 1),
        "추론_s": round(inf_seconds, 1),
        "건당_s": round(inf_seconds / max(len(recs), 1), 2),
        "전체_s": round(time.time() - t_all, 1),
        "모델호출": getattr(runner, "n_calls", 0),
        "정상응답": getattr(runner, "n_ok", 0),
        "유효JSON": n_json, "근거문구_채움": n_ev,
        "위반합계": int(sum(r[v] for r in rows for v in ITEMS)),
        "출력": out_path, "자가검증": "PASS" if not errs else errs[:10],
    }
    log(json.dumps(report, ensure_ascii=False))
    if not isinstance(runner, MockRunner) and n_json < len(recs):
        log(f"  [주의] 유효 JSON {n_json}/{len(recs)}건 — 나머지는 규칙 계층으로 판정되었습니다.")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="입찰공고 법령 위반 판정 파이프라인")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    ap.add_argument("--input", default=None)
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--quantization", default=os.environ.get("PPS_QUANT", QUANT))
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--max-chars", type=int, default=12000)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--debug-jsonl", default=None)
    ap.add_argument("--mock", action="store_true", help="모델 없이 규칙 계층만")
    a = ap.parse_args()

    input_path = a.input or os.path.join(a.data_dir, "test.jsonl.gz")
    out_path = os.path.join(a.output_dir, "submission.csv")
    quant = None if str(a.quantization).lower() in ("none", "") else a.quantization
    kw = {} if a.mock else dict(model_dir=a.model_dir, quant=quant,
                                max_tokens=a.max_tokens, seed=SEED,
                                gpu_mem=a.gpu_mem, tp=a.tp)
    report = run(input_path, out_path, MockRunner if a.mock else VLLMRunner,
                 limit=a.limit, chunk=a.chunk, chars=a.max_chars,
                 debug_path=a.debug_jsonl, **kw)
    return 0 if report.get("자가검증") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
