#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""나라장터 자체입찰 공고 법령 위반 판정 — 두 계층 앙상블.

평가 서버는 이 파일을 `python script.py` 로 그대로 실행합니다.
  입력   ./data/test.jsonl.gz (+ 항목표.json · 정답스키마_디코딩.json · 법령패키지/)
  정적   ./model/  ← 제출 ZIP에 포함
  출력   ./output/submission.csv  (열 = id, v1..v24, e1..e24)

두 계층
  A. 사실추출 계층 (model/features·law·fuse·prompt — song 갈래)
     고정 LLM에게 '사실 12가지'만 묻고, 24항목 위반 판정은 코드가 조문 기준으로 한다.
     레코드당 LLM 1콜. dev 200건에서 LLM 없이 규칙만으로 Macro F1 0.5756.

  B. 게이트·logprob 계층 (model/oh_engine.py — oh 갈래)
     메타 게이트로 적용 대상 항목만 골라 고정 LLM에게 위반 여부를 직접 묻고,
     판정 토큰의 logprob에서 P(위반)을 복원해 항목별 임계값으로 자른다.

  두 계층은 오류의 종류가 다르다. A는 사실 스키마에 없는 신호를 놓치고,
  B는 모델이 조문을 오독하면 틀린다. 항목별로 어느 쪽을 쓸지는
  model/융합규칙.json 의 상수로 고정한다(오프라인 튜닝 결과만 반영).

엔진은 하나만 띄운다
  두 계층이 각자 vLLM 엔진을 만들면 26B 모델을 두 번 적재하게 된다.
  SharedEngine 하나를 만들고 계층별 어댑터(SongPort·OhPort)로 감싸 공유한다.

규칙 준수
  · 공고 1건마다 고정 LLM을 최소 1회 호출한다 — A계층 호출이 레코드당 1회 보장된다.
  · 샘플 간 예측 의존 없음 — 융합규칙·임계값은 model/ 의 상수이며 런타임에 바뀌지 않는다.
  · 외부 네트워크·외부 LLM 호출 없음. 판정은 배포된 법령패키지 스냅샷만 근거로 한다.
  · 평가 데이터의 id·건수를 하드코딩하지 않는다. 입력에서 읽어 그대로 출력한다.

로컬 실행
  python script.py --mock                      # 모델 없이 A계층 규칙만으로 흐름 확인
  python script.py --input dev.jsonl.gz --output-dir output/dev --dump-parts output/dev
  python script.py --limit 10
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
from collections import Counter, defaultdict
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from model import fuse as FUSE                          # noqa: E402
from model import law as LAW                            # noqa: E402
from model import oh_engine as OH                       # noqa: E402
from model import prompt as PROMPT                      # noqa: E402
from model.extract_schema import FACT_SCHEMA            # noqa: E402
from model.features import extract as extract_signals   # noqa: E402

ITEMS = [f"v{i}" for i in range(1, 25)]
COLUMNS = ["id"] + ITEMS + [f"e{i}" for i in range(1, 25)]
ABSENCE = set(LAW.ABSENCE)
EVIDENCE_MAX = 500

SEED = 20260826
MAX_MODEL_LEN = 16384
FACT_MAX_TOKENS = 2048          # A계층 사실추출 생성 상한
QUANT = "int8_per_channel_weight_only"
SOURCES = ("rule", "song", "oh")   # 예측 원천 셋
MODES = SOURCES + ("or", "and")


def _pick_dir(env: str, candidates: Sequence[str], marker: Optional[str] = None) -> str:
    v = os.environ.get(env)
    if v:
        return v
    for c in candidates:
        if os.path.isdir(c) and (marker is None or os.path.exists(os.path.join(c, marker))):
            return c
    return candidates[0]


DATA_DIR = _pick_dir("PPS_DATA_DIR", ["./data", os.path.join(HERE, "data")], marker="항목표.json")
OUTPUT_DIR = os.environ.get("PPS_OUTPUT_DIR",
                            os.path.join(os.path.dirname(DATA_DIR.rstrip("/\\")) or ".", "output"))
ASSET_DIR = _pick_dir("PPS_ASSET_DIR", [os.path.join(HERE, "model"), "./model"], marker="게이트.json")
MODEL_DIR = _pick_dir("PPS_MODEL_DIR",
                      ["/opt/models/gemma-4-26B-A4B-it", "/workspace/models/gemma-4-26B-A4B-it"],
                      marker="config.json")


def log(msg: str) -> None:
    print(f"[pps] {msg}", file=sys.stderr, flush=True)


# ===== 1. 입력 =====
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
    """두 계층이 같은 레코드 객체를 본다. NFC 정규화는 여기서 한 번만 한다."""
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


# ===== 2. 융합규칙 =====
class Fusion:
    """항목별로 어느 원천의 판정을 쓸지 정한 상수표.

    원천은 셋이다.
      rule  A계층을 LLM 없이 규칙 사실만으로 판정한 값 (추가 콜 0)
      song  A계층에 LLM 사실을 융합해 판정한 값
      oh    B계층의 P(위반) 을 항목 임계값으로 자른 값

    dev 200건에서 rule 0.5756 · song 0.5752 · oh 0.3186 으로 rule 과 song 의
    총점은 거의 같지만 항목별로는 크게 갈린다(v3·v5·v7 은 song 이 1.0,
    v19·v20 은 rule 이 크게 낫다). 그래서 '어느 쪽이 나은가'가 아니라
    '항목마다 어느 쪽인가'를 상수로 고정한다.

    모드는 원천 셋에 결합 둘을 더한 다섯이다.
      rule · song · oh      단독
      or                    원천 둘 중 하나라도 1  (재현율을 산다)
      and                   원천 둘 다 1           (정밀도를 산다)
    """

    def __init__(self, asset_dir: str):
        path = os.path.join(asset_dir, "융합규칙.json")
        d: Dict[str, Any] = {}
        try:
            d = json.load(io.open(path, encoding="utf-8"))
        except FileNotFoundError:
            log(f"[경고] {path} 없음 — 전 항목 rule 원천 단독으로 실행합니다.")
        except Exception as e:
            log(f"[경고] {path} 읽기 실패({type(e).__name__}) — 전 항목 rule 원천 단독으로 실행합니다.")
        self.default: str = d.get("기본", "rule")
        if self.default not in MODES:
            self.default = "rule"
        raw = d.get("규칙", {}) or {}
        self.rules: Dict[str, str] = {}
        self.srcs: Dict[str, Tuple[str, str]] = {}
        self.oh_threshold: Dict[str, float] = {}
        base_thr = float(d.get("oh_기본임계값", 0.25))
        for v in ITEMS:
            r = raw.get(v) or {}
            m = r.get("모드", self.default)
            if m not in MODES:
                m = self.default
            pair = tuple(x for x in (r.get("원천") or []) if x in SOURCES)
            if m in ("or", "and") and len(pair) != 2:
                m = self.default                      # 원천이 불완전하면 기본 모드로 되돌린다
            self.rules[v] = m
            self.srcs[v] = pair if len(pair) == 2 else ("rule", "song")
            try:
                self.oh_threshold[v] = float(r.get("임계값", base_thr))
            except (TypeError, ValueError):
                self.oh_threshold[v] = base_thr
        self.note: str = d.get("설명", "")

    def mode(self, v: str) -> str:
        return self.rules[v]

    def used_sources(self, v: str) -> Tuple[str, ...]:
        m = self.rules[v]
        return (m,) if m in SOURCES else self.srcs[v]

    def oh_items(self) -> List[str]:
        """B계층 확률이 실제로 필요한 항목. 나머지는 B계층에 묻지 않는다."""
        return [v for v in ITEMS if "oh" in self.used_sources(v)]

    def oh_groups(self) -> List[str]:
        need = set(self.oh_items())
        return [g for g, members in OH.GROUPS.items() if need & set(members)]

    def summary(self) -> Dict[str, Any]:
        return {"기본": self.default, "모드분포": dict(Counter(self.rules.values())),
                "B계층_항목": self.oh_items(), "B계층_그룹": self.oh_groups()}


# ===== 3. 공유 엔진 =====
class SharedEngine:
    """vLLM 엔진 하나를 두 계층이 공유한다. 26B 모델을 두 번 적재하지 않기 위한 장치."""

    def __init__(self, model_dir: str = MODEL_DIR, quant: Optional[str] = QUANT,
                 seed: int = SEED, gpu_mem: float = 0.92, tp: int = 1,
                 max_model_len: int = MAX_MODEL_LEN):
        t0 = time.time()
        if os.sep in model_dir and not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"모델 경로가 없습니다: {model_dir}\n"
                f"  PPS_MODEL_DIR 환경변수를 쓰거나 --model-dir 로 지정하십시오.")
        import vllm                                     # --mock 실행에서 vllm 없이도 돌도록 지연 import
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        log(f"vllm {vllm.__version__} · 모델 {model_dir} · quant={quant} · max_model_len={max_model_len}")
        kw = dict(model=model_dir, tokenizer=model_dir, max_model_len=max_model_len,
                  gpu_memory_utilization=gpu_mem, seed=seed, tensor_parallel_size=tp,
                  dtype="auto", enable_prefix_caching=True)
        if quant:
            kw["quantization"] = quant
        self.llm = LLM(**kw)
        self.tok = self.llm.get_tokenizer()
        self._SP, self._SOP = SamplingParams, StructuredOutputsParams
        self.max_model_len = max_model_len
        self.seed = seed
        self.load_seconds = time.time() - t0
        self.supports_system = self._probe_system()

    def _probe_system(self) -> bool:
        try:
            self.tok.apply_chat_template(
                [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
                add_generation_prompt=True, tokenize=False)
            return True
        except Exception as e:
            log(f"  · chat template 이 system 롤을 지원하지 않음({type(e).__name__}) → user 턴으로 합쳐 보냄")
            return False

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        try:
            ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
            if hasattr(ids, "keys") and "input_ids" in ids:
                ids = ids["input_ids"]
            return len(ids)
        except Exception:
            return len(self.tok.encode("\n".join(m["content"] for m in messages)))

    def params(self, schema: Optional[Dict[str, Any]], max_tokens: int, logprobs: Optional[int]):
        kw: Dict[str, Any] = dict(temperature=0.0, max_tokens=max_tokens, seed=self.seed)
        if logprobs:
            kw["logprobs"] = logprobs
        if schema is None:
            return self._SP(**kw)
        try:
            return self._SP(structured_outputs=self._SOP(json=schema, disable_any_whitespace=True), **kw)
        except Exception as e:
            log(f"  ! 구조화 출력 설정 실패 → 비구조화로 강등: {type(e).__name__}: {str(e)[:140]}")
            return self._SP(**kw)

    def generate(self, batch, schema, max_tokens: int, logprobs: Optional[int] = None):
        outs = self.llm.chat(batch, sampling_params=self.params(schema, max_tokens, logprobs),
                             use_tqdm=False)
        res = []
        for o in outs:
            if not o.outputs:
                res.append(("", None, None))
                continue
            c = o.outputs[0]
            res.append((c.text, getattr(c, "token_ids", None), getattr(c, "logprobs", None)))
        return res


class SongPort:
    """A계층이 기대하는 러너 인터페이스로 공유 엔진을 감싼다."""

    def __init__(self, engine: SharedEngine, schema, max_tokens: int = FACT_MAX_TOKENS):
        self.e, self._schema, self._max_tokens = engine, schema, max_tokens
        self.structured = True
        self.supports_system = engine.supports_system
        self.load_seconds = engine.load_seconds
        self.n_calls = self.n_ok = 0

    def set_structured(self, on: bool) -> bool:
        self.structured = on
        return True

    def count_tokens(self, messages) -> int:
        return self.e.count_tokens(messages)

    def chat(self, batch) -> List[str]:
        schema = self._schema if self.structured else None
        outs = self.e.generate(batch, schema, self._max_tokens)
        texts = [t for t, _ids, _lp in outs]
        self.n_calls += len(batch)
        self.n_ok += sum(1 for t in texts if t and t.strip())
        return texts


class OhPort:
    """B계층이 기대하는 러너 인터페이스. logprob 을 켜서 판정 토큰 확률을 받는다."""

    def __init__(self, engine: SharedEngine):
        self.e = engine
        self.tok = engine.tok
        self.max_model_len = engine.max_model_len
        self.load_seconds = engine.load_seconds
        self.n_calls = self.n_ok = 0

    def count_tokens(self, messages) -> int:
        return self.e.count_tokens(messages)

    def chat(self, batch, schema, max_tokens: int):
        outs = self.e.generate(batch, schema, max_tokens, logprobs=20)
        self.n_calls += len(batch)
        self.n_ok += sum(1 for t, _i, _l in outs if t and t.strip())
        return outs


class MockEngine:
    """모델 없이 A계층 규칙 판정과 출력 형식을 확인한다. B계층 확률은 전부 0."""
    load_seconds = 0.0
    supports_system = False
    max_model_len = MAX_MODEL_LEN
    tok = None

    def count_tokens(self, messages) -> int:
        return int(sum(len(m["content"]) for m in messages) / 1.3)

    def generate(self, batch, schema, max_tokens, logprobs=None):
        return [("", None, None) for _ in batch]


# ===== 4. A계층 (사실추출 → 법령적용) =====
def song_fit_budget(rec, sig, port, chars: int, budget: int, use_sys: bool):
    while True:
        msgs = PROMPT.build_messages(rec, sig, budget_chars=chars, use_system=use_sys)
        n = port.count_tokens(msgs)
        if n <= budget or chars <= 2500:
            return msgs, n
        chars = int(chars * min(0.85, budget / max(n, 1) * 0.95))


def run_song(recs, port, chunk: int, chars: int):
    """레코드당 1콜로 사실을 뽑고 코드로 24항목을 판정한다.

    같은 law.judge 를 두 번 돌려 원천 둘을 만든다. LLM 사실을 융합한 것(song)과
    규칙 사실만으로 판정한 것(rule)이다. rule 은 추가 LLM 콜이 들지 않는다.
    반환 = (song판정, rule판정, 신호, 통계)
    """
    sigs, facts_rule = [], []
    for rec in recs:
        try:
            sig = extract_signals(rec)
        except Exception as e:
            log(f"  ! {rec['id']} 신호추출 실패: {type(e).__name__}: {e}")
            sig = None
        sigs.append(sig)
        facts_rule.append(LAW.rule_facts(sig) if sig else None)

    budget = MAX_MODEL_LEN - FACT_MAX_TOKENS - 64
    use_sys = bool(getattr(port, "supports_system", False))

    def build_all(flag: bool):
        out, toks = [], []
        for rec, sig in zip(recs, sigs):
            if sig is None:
                out.append([{"role": "user", "content": "-"}])
                toks.append(0)
                continue
            m, n = song_fit_budget(rec, sig, port, chars, budget, flag)
            out.append(m)
            toks.append(n)
        return out, toks

    msgs_all, ntok = build_all(use_sys)
    if ntok:
        srt = sorted(ntok)
        log(f"  A계층 프롬프트 토큰 중앙값 {srt[len(srt) // 2]:,} / 최대 {max(ntok):,} (예산 {budget:,})")

    texts: List[str] = []
    mock = isinstance(port, _MockSongPort)
    t0 = time.time()
    # 생성 설정은 고정해서 시작한다. 예전에는 앞 8건을 먼저 돌려 설정을 정하는 프로브를 썼지만,
    # 그 방식은 '앞 8건의 응답'이 뒤 공고의 생성 설정을 바꾸는 구조라 공고 간 의존으로 읽힐 수 있다.
    # 운영측 답변이 "각 공고는 독립적으로 처리해야 한다"고 명시하므로, 설정 선택을 레코드 단위로 내린다.
    #   · 정상 환경에서는 첫 설정이 그대로 통과하므로 프로브가 고르던 설정과 동일하다(점수 변화 없음).
    #   · 실패한 레코드만 그 레코드의 응답을 보고 대안 설정으로 재시도한다. 다른 공고를 참조하지 않는다.
    port.set_structured(True)
    for i in range(0, len(msgs_all), chunk):
        texts.extend(song_chunk(port, msgs_all[i:i + chunk]))
        log(f"  A계층 {min(i + chunk, len(msgs_all))}/{len(msgs_all)}건 … {time.time() - t0:.0f}s")

    # 레코드 단위 재시도 — 판정에 쓸 JSON 을 못 얻은 건만, 자기 응답만 보고 설정을 바꾼다.
    budget_r = MAX_MODEL_LEN - FACT_MAX_TOKENS - 64
    bad = [i for i, t in enumerate(texts) if sigs[i] is not None and not isinstance(extract_json(t), dict)]
    retried = recovered = 0
    if bad and not mock:
        log(f"  A계층 유효 JSON 미획득 {len(bad)}건 → 레코드 단위 재시도")
        alts = [(False, use_sys), (True, not use_sys), (False, not use_sys)]
        for i in bad:
            retried += 1
            for want_struct, want_sys in alts:
                port.set_structured(want_struct)
                m, _ = song_fit_budget(recs[i], sigs[i], port, chars, budget_r, want_sys)
                out = song_chunk(port, [m])[0]
                if isinstance(extract_json(out), dict):
                    texts[i] = out
                    recovered += 1
                    break
        port.set_structured(True)
        log(f"  재시도 {retried}건 중 {recovered}건 복구")
    if bad and len(bad) == len([x for x in sigs if x is not None]) and not mock:
        log("  [경고] A계층에서 유효 JSON 을 한 건도 얻지 못했습니다 — 규칙 계층만으로 판정됩니다. "
            "대회 규칙상 '공고별 고정 LLM 정상 호출 1회 이상' 요건 확인이 필요합니다.")

    blank = {v: {"hit": 0, "ev": None} for v in ITEMS}
    judged_song, judged_rule, n_json = [], [], 0
    for rec, sig, rf, txt in zip(recs, sigs, facts_rule, texts):
        if sig is None:
            judged_song.append(dict(blank))
            judged_rule.append(dict(blank))
            continue
        try:
            judged_rule.append(LAW.judge(rf, sig))
        except Exception as e:
            log(f"  ! {rec['id']} rule 판정 실패 → 전항목 0: {type(e).__name__}: {e}")
            judged_rule.append(dict(blank))
        try:
            obj = extract_json(txt)
            n_json += int(isinstance(obj, dict))
            judged_song.append(LAW.judge(FUSE.merge(obj, rf, sig), sig))
        except Exception as e:
            log(f"  ! {rec['id']} song 판정 실패 → rule 판정으로 대체: {type(e).__name__}: {e}")
            judged_song.append(judged_rule[-1])

    stat = {"콜": getattr(port, "n_calls", 0), "정상응답": getattr(port, "n_ok", 0),
            "유효JSON": n_json, "재시도": retried, "재시도복구": recovered,
            "추론_s": round(time.time() - t0, 1)}
    return judged_song, judged_rule, sigs, stat


def song_chunk(port, batch) -> List[str]:
    try:
        return port.chat(batch)
    except Exception as e:
        log(f"  ! A계층 청크({len(batch)}건) 실패 → 건별 재시도: {type(e).__name__}: {str(e)[:140]}")
    out = []
    for m in batch:
        try:
            out.append(port.chat([m])[0])
        except Exception as e:
            log(f"  ! A계층 건별 실패 → 빈 출력: {type(e).__name__}: {str(e)[:120]}")
            out.append("")
    return out


class _MockSongPort(SongPort):
    def __init__(self, engine, schema, max_tokens=FACT_MAX_TOKENS):
        self.e, self._schema, self._max_tokens = engine, schema, max_tokens
        self.structured = False
        self.supports_system = False
        self.load_seconds = 0.0
        self.n_calls = self.n_ok = 0

    def chat(self, batch):
        return ["" for _ in batch]


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


# ===== 5. B계층 (게이트 → 그룹 판정 → logprob) =====
def build_oh_tasks(recs, tbl, a, port, need_items: Sequence[str],
                   doc_budget: int, law_budget: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """B계층이 실제로 쓰이는 항목만 묻는다. 그룹 전체를 묻지 않아 콜 수가 줄어든다."""
    need = set(need_items)
    tasks: List[Dict[str, Any]] = []
    gated: Counter = Counter()
    for ri, rec in enumerate(recs):
        f = OH.derive_facts(rec, a)
        rec["_facts"] = f
        for group, members in OH.GROUPS.items():
            items = [v for v in members if v in need and OH.applicable(v, f, a)]
            if not items:
                gated.update(v for v in members if v in need)
                continue
            law = OH.group_law_text(group, f, a, law_budget)
            sysmsg = OH.build_group_system(group, items, tbl, law, a.guides)
            usr, _ = OH.build_group_user(rec, f, a, group, doc_budget)
            tasks.append({"ri": ri, "group": group, "items": items,
                          "messages": OH.build_messages(sysmsg, usr)})

    budget = port.max_model_len - 256 - 1024
    shrunk, ntok = 0, []
    for t in tasks:
        n = port.count_tokens(t["messages"])
        db = doc_budget
        while n > budget and db > 1500:
            db = int(db * min(0.8, budget / n * 0.9))
            rec = recs[t["ri"]]
            usr, _ = OH.build_group_user(rec, rec["_facts"], a, t["group"], db)
            t["messages"][1]["content"] = usr
            n = port.count_tokens(t["messages"])
            shrunk += 1
        ntok.append(n)
    stat = {"작업수": len(tasks), "레코드당_평균콜": round(len(tasks) / max(len(recs), 1), 2),
            "토큰_중앙": sorted(ntok)[len(ntok) // 2] if ntok else 0,
            "토큰_최대": max(ntok) if ntok else 0, "예산축소": shrunk,
            "게이트차단": dict(gated)}
    return tasks, stat


def run_oh(recs, port, a, tbl, need_items: Sequence[str], chunk: int,
           doc_budget: int, law_budget: int):
    """반환 = (확률, 근거, 통계). need_items 가 비면 아무 호출도 하지 않는다."""
    probs: List[Dict[str, float]] = [dict() for _ in recs]
    evid: List[Dict[str, str]] = [dict() for _ in recs]
    if not need_items:
        return probs, evid, {"작업수": 0, "비고": "융합규칙이 B계층을 쓰지 않음"}

    tasks, stat = build_oh_tasks(recs, tbl, a, port, need_items, doc_budget, law_budget)
    log(f"  B계층 {json.dumps(stat, ensure_ascii=False)}")
    if not tasks:
        return probs, evid, stat

    by_shape: Dict[Tuple[str, Tuple[str, ...]], List[int]] = defaultdict(list)
    for i, t in enumerate(tasks):
        by_shape[(t["group"], tuple(t["items"]))].append(i)

    t0, done, fb_total, empty = time.time(), 0, 0, 0
    for (group, items), idxs in sorted(by_shape.items()):
        max_tokens = min(1536, 96 + 190 * len(items))
        for s in range(0, len(idxs), chunk):
            part = [tasks[i] for i in idxs[s:s + chunk]]
            outs = OH.run_chunk(port, part, max_tokens)
            for t, (text, tids, lps) in zip(part, outs):
                if not text:
                    empty += 1
                p, fb = OH.item_probs(text, tids, lps, port.tok, t["items"])
                fb_total += fb
                probs[t["ri"]].update(p)
                for v, e in OH.extract_evidence(text, t["items"]).items():
                    if e:
                        evid[t["ri"]][v] = e
            done += len(part)
            log(f"  B계층 {group}{list(items)[:2]}… {done}/{len(tasks)} … {time.time() - t0:.0f}s")
    stat.update({"빈출력": empty, "logprob_폴백": fb_total, "추론_s": round(time.time() - t0, 1),
                 "콜": getattr(port, "n_calls", 0), "정상응답": getattr(port, "n_ok", 0)})
    return probs, evid, stat


# ===== 6. 융합 =====
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


def fuse_record(j_song: Dict[str, Dict[str, Any]], j_rule: Dict[str, Dict[str, Any]],
                p: Dict[str, float], oh_ev: Dict[str, str],
                fusion: Fusion, a, src: str) -> Tuple[Dict[str, int], Dict[str, str], Counter]:
    """항목별 융합규칙으로 최종 0/1과 근거문구를 정한다.

    근거는 위반으로 판정한 원천의 것을 쓰되, 여럿이면 song → rule → oh 순으로 본다.
    A계층 쪽이 원문 복구 4단계와 규칙 블록 폴백을 갖고 있어 부분문자열 보존률이 높다.
    """
    decided: Dict[str, int] = {}
    cells: Dict[str, str] = {}
    used: Counter = Counter()
    for v in ITEMS:
        c_song = j_song.get(v) or {"hit": 0, "ev": None}
        c_rule = j_rule.get(v) or {"hit": 0, "ev": None}
        prob = p.get(v)
        hits = {
            "song": 1 if c_song.get("hit") == 1 else 0,
            "rule": 1 if c_rule.get("hit") == 1 else 0,
            "oh": int(prob >= fusion.oh_threshold[v]) if prob is not None else 0,
        }
        mode = fusion.mode(v)
        if mode in SOURCES:
            hit = hits[mode]
        else:
            s1, s2 = fusion.srcs[v]
            hit = (hits[s1] | hits[s2]) if mode == "or" else (hits[s1] & hits[s2])
        decided[v] = hit
        if not hit:
            continue
        used[mode] += 1
        if v in ABSENCE:                        # 부재탐지 항목은 항상 빈칸
            continue

        cell = ""
        for name, node in (("song", c_song), ("rule", c_rule)):
            if hits[name] and name in fusion.used_sources(v):
                cell = clean_evidence(node.get("ev"), src)
                if cell:
                    break
        if not cell and hits["oh"] and "oh" in fusion.used_sources(v):
            cell = clean_evidence(oh_ev.get(v), src)
        # B계층만으로 결정한 항목에는 B계층의 근거 형태 검사를 그대로 적용한다.
        # 근거 없이는 성립할 수 없는 항목(예: v21 은 지분율 숫자)의 오탐을 막는 장치다.
        if cell and fusion.used_sources(v) == ("oh",):
            rx = a.evidence_rx.get(v) if a is not None else None
            if rx is not None and not rx.search(cell):
                decided[v] = 0
                used[mode] -= 1
                continue
        cells[v] = cell
    return decided, cells, used


# ===== 7. 출력 =====
def to_row(rec_id: str, decided: Dict[str, int], cells: Dict[str, str]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"id": rec_id}
    for i, v in enumerate(ITEMS, 1):
        row[v] = int(decided.get(v, 0))
        row[f"e{i}"] = "" if (row[v] == 0 or v in ABSENCE) else cells.get(v, "")
    return row


def empty_row(rec_id: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {"id": rec_id}
    for i, v in enumerate(ITEMS, 1):
        row[v], row[f"e{i}"] = 0, ""
    return row


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with io.open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: unicodedata.normalize("NFC", str(r[k])) for k in COLUMNS})


def validate_csv(path: str, expected_ids: List[str]) -> List[str]:
    """열 49 · 행 수 = 입력 건수 · id 유일·일치 · v 0/1 · e 500자 이하 · 부재탐지 빈칸 · 수식 접두 없음"""
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
        errs.append(f"id 집합 불일치 (누락 {len(set(expected_ids) - set(ids))})")
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


# ===== 8. 실행 =====
def run(input_path: str, out_path: str, data_dir: str, asset_dir: str, mock: bool,
        limit: Optional[int], chunk: int, chars: int, doc_budget: int, law_budget: int,
        dump_parts: Optional[str] = None, collect: bool = False, **engine_kw) -> Dict[str, Any]:
    t_all = time.time()
    log(f"경로: cwd={os.getcwd()} · data={os.path.abspath(data_dir)} · "
        f"asset={os.path.abspath(asset_dir)} · out={os.path.abspath(os.path.dirname(out_path))}")
    recs = list(iter_records(input_path, limit=limit))
    log(f"입력 {len(recs)}건 ← {input_path}")
    if not recs:
        write_csv([], out_path)
        return {"건수": 0, "자가검증": "PASS"}

    fusion = Fusion(asset_dir)
    log(f"융합규칙: {json.dumps(fusion.summary(), ensure_ascii=False)}")

    tbl = OH.item_table(data_dir)
    a = OH.Assets(asset_dir)

    engine = MockEngine() if mock else SharedEngine(**engine_kw)
    log(f"엔진 준비 {engine.load_seconds:.1f}s")
    song_port = (_MockSongPort if mock else SongPort)(engine, FACT_SCHEMA)
    oh_port = OhPort(engine) if not mock else None

    judged_song, judged_rule, sigs, song_stat = run_song(recs, song_port, chunk, chars)
    log(f"  A계층 {json.dumps(song_stat, ensure_ascii=False)}")

    # --collect 는 오프라인 융합규칙 학습용이다. 융합규칙이 B계층을 쓰지 않는 항목까지
    # 전부 물어 확률을 모은다. 제출 실행에서는 쓰지 않는다(콜 수가 7배가 된다).
    need = ITEMS if collect else fusion.oh_items()
    if collect:
        log("  [수집모드] B계층에 24항목 전부를 묻습니다 — 융합규칙 학습 전용")
    if mock or not need:
        probs = [dict() for _ in recs]
        oh_evs = [dict() for _ in recs]
        oh_stat = {"작업수": 0, "비고": "mock" if mock else "융합규칙이 B계층을 쓰지 않음"}
    else:
        probs, oh_evs, oh_stat = run_oh(recs, oh_port, a, tbl, need, chunk, doc_budget, law_budget)

    rows, pos, used_total = [], Counter(), Counter()
    for rec, j_s, j_r, p, oev, sig in zip(recs, judged_song, judged_rule, probs, oh_evs, sigs):
        try:
            src = sig["text"] if sig else OH.full_text(rec)
            decided, cells, used = fuse_record(j_s, j_r, p, oev, fusion, a, src)
            rows.append(to_row(rec["id"], decided, cells))
            used_total.update(used)
            for v in ITEMS:
                if decided[v]:
                    pos[v] += 1
        except Exception as e:
            log(f"  ! {rec['id']} 융합 실패 → 전항목 0: {type(e).__name__}: {e}")
            rows.append(empty_row(rec["id"]))
    assert len(rows) == len(recs)

    write_csv(rows, out_path)
    if dump_parts:
        dump_layer_outputs(dump_parts, recs, judged_song, judged_rule, probs, oh_evs, sigs)

    errs = validate_csv(out_path, [r["id"] for r in recs])
    report = {
        "건수": len(recs), "엔진로드_s": round(engine.load_seconds, 1),
        "A계층": song_stat, "B계층": oh_stat,
        "융합_채택": dict(used_total),
        "양성_합계": int(sum(pos.values())), "양성_항목별": {v: pos[v] for v in ITEMS if pos[v]},
        "전체_s": round(time.time() - t_all, 1),
        "출력": out_path, "자가검증": "PASS" if not errs else errs[:10],
    }
    log(json.dumps(report, ensure_ascii=False))
    if not mock:
        n_ok = song_stat.get("정상응답", 0)
        if n_ok < len(recs):
            log(f"  [주의] A계층 정상응답 {n_ok}/{len(recs)}건 — 공고별 고정 LLM 정상 호출 "
                f"1회 이상 요건을 확인하십시오.")
    if report["양성_합계"] == 0:
        log("[경고] 양성 판정이 0건입니다 — 전 항목 0 제출은 Macro F1이 정확히 0.0입니다.")
    return report


def dump_layer_outputs(out_dir: str, recs, judged_song, judged_rule, probs, oh_evs, sigs) -> None:
    """원천별 원시 판정을 남긴다. 융합규칙을 다시 맞출 때 모델을 재실행하지 않기 위한 것."""
    os.makedirs(out_dir, exist_ok=True)

    def _hits(path: str, judged_list):
        with io.open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["id"] + ITEMS)
            for rec, j in zip(recs, judged_list):
                w.writerow([rec["id"]] + [int((j.get(v) or {}).get("hit") == 1) for v in ITEMS])

    pr = os.path.join(out_dir, "layer_rule.csv")
    ps = os.path.join(out_dir, "layer_song.csv")
    _hits(pr, judged_rule)
    _hits(ps, judged_song)
    pb = os.path.join(out_dir, "layer_oh_probs.csv")
    with io.open(pb, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["id"] + ITEMS)
        for rec, p in zip(recs, probs):
            w.writerow([rec["id"]] + [f"{p.get(v, 0.0):.6f}" for v in ITEMS])
    pe = os.path.join(out_dir, "layer_evidence.jsonl")
    with io.open(pe, "w", encoding="utf-8") as f:
        for rec, js, jr, oev, sig in zip(recs, judged_song, judged_rule, oh_evs, sigs):
            src = sig["text"] if sig else OH.full_text(rec)
            def _ev(j):
                out = {}
                for v in ITEMS:
                    c = clean_evidence((j.get(v) or {}).get("ev"), src)
                    if c:
                        out[v] = c
                return out
            f.write(json.dumps({"id": rec["id"], "song_ev": _ev(js), "rule_ev": _ev(jr),
                                "oh_ev": {k: clean_evidence(x, src) for k, x in oev.items()}},
                               ensure_ascii=False) + "\n")
    log(f"원천 덤프 → {pr} · {ps} · {pb} · {pe}")


def main() -> int:
    ap = argparse.ArgumentParser(description="24개 항목 법령 위반 판정 — 두 계층 앙상블")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    ap.add_argument("--asset-dir", default=ASSET_DIR)
    ap.add_argument("--input", default=None, help="기본 = <data-dir>/test.jsonl.gz")
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--quantization", default=os.environ.get("PPS_QUANT", QUANT))
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--max-chars", type=int, default=12000, help="A계층 발췌 예산(글자)")
    ap.add_argument("--doc-budget", type=int, default=10000, help="B계층 문서 구간 예산(글자)")
    ap.add_argument("--law-budget", type=int, default=5000, help="B계층 그룹당 조문 예산(글자)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dump-parts", default=None, help="계층별 원시 판정 덤프 디렉토리")
    ap.add_argument("--collect", action="store_true",
                    help="융합규칙 학습용 — B계층에 24항목 전부를 묻고 확률을 덤프한다")
    ap.add_argument("--mock", action="store_true", help="모델 없이 A계층 규칙만")
    a = ap.parse_args()

    input_path = a.input or os.path.join(a.data_dir, "test.jsonl.gz")
    out_path = os.path.join(a.output_dir, "submission.csv")
    quant = None if str(a.quantization).lower() in ("none", "") else a.quantization
    kw = {} if a.mock else dict(model_dir=a.model_dir, quant=quant, seed=SEED,
                                gpu_mem=a.gpu_mem, tp=a.tp)
    report = run(input_path, out_path, a.data_dir, a.asset_dir, a.mock,
                 limit=a.limit, chunk=a.chunk, chars=a.max_chars,
                 doc_budget=a.doc_budget, law_budget=a.law_budget,
                 dump_parts=a.dump_parts, collect=a.collect, **kw)
    return 0 if report.get("자가검증") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
