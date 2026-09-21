# -*- coding: utf-8 -*-
"""LLM 사실 추출 결과 + 규칙 추출 결과 → law.judge 가 먹는 facts 로 합친다.

원칙
 · 사실 단위로 LLM 을 우선하되, LLM 이 비워 둔 값은 규칙값으로 채운다.
 · 근거문구는 반드시 원문 부분문자열로 '복구'한다. 복구 실패 시 규칙 블록을 쓴다.
 · 금액·날짜처럼 계산이 필요한 값은 LLM 문자열을 그대로 믿지 않고 파싱해 검증한다.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from . import textutil as T

Block = Tuple[int, int, str]

# 발췌 문단의 머리표 — "[태그·태그]<문서 / 절 제목> " 까지를 한 덩어리로 본다.
TAG_HEAD = re.compile(r"^\s*\[[^\[\]\n]{1,60}\]\s*(?:<[^<>\n]{0,60}>)?\s*")


# ---------------------------------------------------------------- 근거 복구
def resolve_evidence(ev: Optional[str], text: str, blocks: List[Block],
                     fallback: Optional[Block]) -> Optional[Block]:
    if not ev or not isinstance(ev, str):
        return fallback
    ev = unicodedata.normalize("NFC", ev).strip()
    if not ev:
        return fallback
    i = text.find(ev)
    if i >= 0:
        return (i, i + len(ev), ev)
    # 발췌 문단은 "[기업규모·참가자격]<공고문 / 입찰참가자격> 본문" 꼴로 들어간다.
    # 모델이 이 머리표까지 베껴 오면 원문에 없는 문자열이 되므로 떼고 다시 찾는다.
    # 원문에도 '[지역:r1|단위=기초|…]' 같은 익명화 토큰이 대괄호로 시작하므로,
    # 있는 그대로 찾기에 실패했을 때만 벗긴다.
    stripped = TAG_HEAD.sub("", ev).strip()
    if stripped and stripped != ev:
        j = text.find(stripped)
        if j >= 0:
            return (j, j + len(stripped), stripped)
        ev = stripped
    # 공백만 어긋난 경우: 공백 제거 문자열로 위치를 찾아 원문 구간을 되살린다
    squeezed = re.sub(r"\s+", "", ev)
    if 8 <= len(squeezed) <= 400:
        pos, buf = [], []
        for idx, ch in enumerate(text):
            if not ch.isspace():
                buf.append(ch)
                pos.append(idx)
        flat = "".join(buf)
        j = flat.find(squeezed)
        if j >= 0:
            s, e = pos[j], pos[j + len(squeezed) - 1] + 1
            return (s, e, text[s:e])
    # 앞머리 25자 앵커로 블록을 찾는다
    head = ev[:25]
    if len(head) >= 8:
        k = text.find(head)
        if k >= 0:
            for b in blocks:
                if b[0] <= k < b[1]:
                    return b
    # 문자 겹침이 가장 큰 블록
    best, score = None, 0
    key = set(squeezed)
    for b in blocks:
        if len(b[2]) < 10:
            continue
        ov = len(key & set(re.sub(r"\s+", "", b[2])))
        if ov > score:
            best, score = b, ov
    if best is not None and score >= max(6, len(key) * 0.6):
        return best
    return fallback


def _amount(v) -> Optional[int]:
    if isinstance(v, (int, float)) and v > 0:
        return int(v)
    if isinstance(v, str):
        a = T.find_amounts(v)
        if a:
            return a[0][2]
        d = re.sub(r"\D", "", v)
        if d:
            return int(d)
    return None


def _date(v):
    if not v or not isinstance(v, str):
        return None
    d = T.find_dates(v)
    return d[0][1] if d else T.parse_meta_date(v)


SIZE_MAP = {"없음": "없음", "소기업소상공인": "소기업소상공인", "중소기업자": "중소기업자"}


def merge(llm: Optional[Dict[str, Any]], rf: Dict[str, Any],
          sig: Dict[str, Any]) -> Dict[str, Any]:
    """llm 이 None 이거나 형식이 깨졌으면 규칙 사실을 그대로 쓴다."""
    if not isinstance(llm, dict):
        return rf
    text, blocks = sig["text"], sig["blocks"]
    f = dict(rf)

    def ev(node_key, field, fb):
        node = llm.get(node_key)
        s = node.get("근거문구") if isinstance(node, dict) else None
        return resolve_evidence(s, text, blocks, fb)

    # 공고문 기재 금액 우선 (대회 명세: 금액 구간 판단은 공고문 기재값 우선) ----
    n = llm.get("공고문금액") or {}
    doc_est, doc_bud = _amount(n.get("추정가격")), _amount(n.get("사업예산"))
    if doc_est is None and doc_bud:
        doc_est = int(round(doc_bud / 1.1))
    base = sig.get("est_meta") or sig.get("est")
    if doc_est and doc_est >= 1_000_000:
        # 오추출 방어 — 민감도 실측(dev): est 를 ±10% 틀리면 Macro F1 -0.008 로 무해하지만
        # 2배/0.5배로 틀리면 -0.14/-0.10 까지 떨어진다. 부가세 환산(1.1)·등록 오기 수준의
        # 차이만 받아들이고, 그보다 크게 벌어지면 등록값을 유지한다.
        # (큰 불일치 자체는 v24 가 meta 원값으로 따로 잡는다)
        if not base or 0.6 <= doc_est / base <= 1.6:
            f["est_override"] = doc_est
    if doc_bud and doc_bud >= 1_000_000:
        f["budget_override"] = doc_bud

    # 특정기관 한정 -----------------------------------------------------
    n = llm.get("참가자격_특정기관한정") or {}
    f["org_only"] = {"hit": bool(n.get("해당")),
                     "ev": ev("참가자격_특정기관한정", "근거문구", rf["org_only"]["ev"])}

    # 실적제한 -----------------------------------------------------------
    n = llm.get("실적제한") or {}
    amt = _amount(n.get("요구실적금액")) or rf["perf"]["amount"]
    f["perf"] = {"hit": bool(n.get("해당")), "amount": amt,
                 "org_specific": bool(n.get("발주처_특정")),
                 "ev": ev("실적제한", "근거문구", rf["perf"]["ev"]),
                 "org_ev": ev("실적제한", "근거문구", rf["perf"]["org_ev"])}

    # 지역제한 -----------------------------------------------------------
    n = llm.get("지역제한") or {}
    wide = [T.canon_region(str(x)) for x in (n.get("제한광역목록") or []) if str(x).strip()]
    wide = sorted(set(w for w in wide if w))
    if not wide and rf["region"].get("wide"):
        wide = rf["region"]["wide"]
    f["region"] = {"hit": bool(n.get("해당")), "n_wide": len(wide), "wide": wide,
                   "basic_unit": bool(n.get("기초단위포함")) or rf["region"]["basic_unit"],
                   "ev": ev("지역제한", "근거문구", rf["region"]["ev"])}

    # 기업규모 -----------------------------------------------------------
    n = llm.get("기업규모제한") or {}
    f["size_class"] = SIZE_MAP.get(str(n.get("구분")), rf["size_class"])
    f["panro_exception"] = rf.get("panro_exception", False)
    f["size_ev"] = ev("기업규모제한", "근거문구", rf["size_ev"])

    # 경쟁제품 / 직접생산 -------------------------------------------------
    # 세부품명번호가 중기부 고시 목록에 있으면 그 사실이 우선한다(문서 근거보다 강함).
    if sig["competitive"].get("code_hit"):
        f["is_competitive"] = True
    elif sig["competitive"].get("codes"):
        f["is_competitive"] = False
    else:
        f["is_competitive"] = bool(llm.get("중기간경쟁제품"))
    n = llm.get("직접생산확인요구") or {}
    f["direct_production_required"] = bool(n.get("해당"))
    f["dp_codes_in_cp"] = rf.get("dp_codes_in_cp") or []
    f["direct_production_ev"] = ev("직접생산확인요구", "근거문구", rf["direct_production_ev"])

    # 특정 모델명 ---------------------------------------------------------
    n = llm.get("특정모델명") or {}
    f["model_name"] = {"hit": bool(n.get("해당")),
                       "equivalent_allowed": bool(n.get("동등이상허용")),
                       "ev": ev("특정모델명", "근거문구", rf["model_name"]["ev"])}

    # 물품공급 확약서 -----------------------------------------------------
    n = llm.get("물품공급확약서") or {}
    f["pledge"] = {"at_bid": bool(n.get("입찰단계요구")),
                   "ev": ev("물품공급확약서", "근거문구", rf["pledge"]["ev"])}

    # SW ------------------------------------------------------------------
    n = llm.get("소프트웨어사업") or {}
    f["sw"] = {"is_sw": bool(n.get("해당")),
               "limit_stated": bool(n.get("대기업참여제한_명시"))}

    # 공동수급 -------------------------------------------------------------
    n = llm.get("공동수급") or {}
    share = n.get("최소지분율")
    share = float(share) if isinstance(share, (int, float)) and 0 < float(share) <= 100 else \
        rf["joint"]["min_share"]
    f["joint"] = {"min_share": share, "ev": ev("공동수급", "근거문구", rf["joint"]["ev"])}

    # 설명회 ---------------------------------------------------------------
    n = llm.get("설명회") or {}
    f["briefing"] = {"held": bool(n.get("개최")),
                     "attendance_required": bool(n.get("참석의무")),
                     "date": _date(n.get("일자")) or rf["briefing"]["date"],
                     "ev": ev("설명회", "근거문구", rf["briefing"]["ev"])}

    # 등록값 불일치 ---------------------------------------------------------
    n = llm.get("등록값불일치") or {}
    hit = bool(n.get("해당")) or bool(rf["mismatch"]["hit"])
    f["mismatch"] = {"hit": hit,
                     "field": n.get("항목") or rf["mismatch"]["field"],
                     "ev": ev("등록값불일치", "근거문구", rf["mismatch"]["ev"])}
    return f
