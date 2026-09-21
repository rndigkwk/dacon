# -*- coding: utf-8 -*-
"""법령 적용 엔진.

입력은 '사실(facts)' 한 벌과 나라장터 메타뿐이고, 24개 항목의 위반 여부는
전부 이 파일의 결정 규칙으로 계산한다. 사실은 두 경로로 들어올 수 있다.

  · rule_facts()  : 정규식·사전 기반 추출 (LLM 없이 동작 / 폴백)
  · LLM 추출      : script.py 가 고정 모델에 구조화 출력으로 뽑아 온 사실

법 해석을 코드로 고정해 두면 (1) 모델이 조문을 외우지 않아도 되고,
(2) 판정 근거를 항목별로 문장으로 남길 수 있어 정성평가에 유리하다.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from . import lexicon as LX
from . import textutil as T

ITEMS = [f"v{i}" for i in range(1, 25)]
ABSENCE = {"v10", "v11", "v16", "v18", "v20"}

SIZE_NONE, SIZE_SMALL, SIZE_SME = "없음", "소기업소상공인", "중소기업자"


def _d(hit, ev=None, why=""):
    return {"hit": int(bool(hit)), "ev": ev, "why": why}


def judge(f: Dict[str, Any], sig: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """사실 → 24개 항목 판정. ev 는 (start, end, text) 블록 또는 문자열."""
    # 금액 구간 판단은 공고문 기재값 우선(LLM 추출), 없으면 나라장터 등록값
    est = int(f.get("est_override") or sig.get("est") or 0)
    law = sig.get("law") or "국가계약법"
    gosi = LX.GOSI_STATE_GOODS
    rlim = sig.get("region_limit") or gosi
    small_neg = law == "지방계약법" and (
        "수의" in (sig.get("method") or "") or "소액수의" in (sig.get("award") or ""))
    nego = "협상" in (sig.get("award") or "") or "협상" in (sig.get("method") or "")

    perf = f.get("perf") or {}
    reg = f.get("region") or {}
    size = f.get("size_class") or SIZE_NONE
    is_cp = bool(f.get("is_competitive"))
    dp = bool(f.get("direct_production_required"))
    R: Dict[str, Dict[str, Any]] = {}

    # --- v1 참가자격을 특정 기관·단체로 제한 -------------------------------
    R["v1"] = _d(f.get("org_only", {}).get("hit"), f.get("org_only", {}).get("ev"),
                 "국가계약법 시행령 제12·21조 / 지방계약법 시행령 제13·20조 — "
                 "법령 근거 없이 참가자격을 특정 기관·단체로 한정")

    # --- 실적제한 계열 ------------------------------------------------------
    has_perf = bool(perf.get("hit"))
    perf_amt = perf.get("amount")
    perf_ev = perf.get("ev")

    R["v2"] = _d(has_perf and est < gosi and not small_neg, perf_ev,
                 f"추정가격 {est:,}원이 고시금액 {gosi:,}원 미만이므로 "
                 "국가계약법 시행령 제21조제1항(지방: 시행령 제20조제1항제3·5호)상 "
                 "실적에 의한 참가자격 제한을 둘 수 없음")

    over = bool(has_perf and perf_amt and perf_amt > est)
    R["v3"] = _d(over, perf_ev,
                 f"요구 실적금액 {perf_amt}원이 추정가격 {est:,}원의 1배를 초과 — "
                 "국가계약법 시행규칙 제25조 / 지방계약법 시행규칙 제25조제2항제1호 나목 위반")

    R["v4"] = _d(has_perf and perf.get("org_specific"), perf.get("org_ev") or perf_ev,
                 "(계약예규) 정부 입찰·계약 집행기준 제2장 제5조 — "
                 "실적의 발주처를 특정 기관으로 한정하는 제한은 허용되지 않음")

    # --- 지역제한 계열 ------------------------------------------------------
    has_reg = bool(reg.get("hit"))
    n_wide = int(reg.get("n_wide") or 0)
    reg_ev = reg.get("ev")

    R["v5"] = _d(has_reg and est >= rlim, reg_ev,
                 f"추정가격 {est:,}원 ≥ 지역제한 허용상한 {rlim:,}원 — "
                 "국가계약법 시행령 제21조 / 지방계약법 시행규칙 제24조상 지역제한 불가")

    R["v6"] = _d(has_reg and est < rlim and reg.get("basic_unit") and not small_neg, reg_ev,
                 "지역제한은 시·도(광역) 단위로 하여야 함 — "
                 "지방계약법 시행규칙 제25조제3항 / 국가계약법 시행규칙 제25조 위반")

    R["v7"] = _d(has_reg and est < rlim and n_wide >= 2 and not small_neg, reg_ev,
                 "인접 시·도까지 확대하려면 시행규칙 제25조제3항 각 호의 예외에 해당해야 함 — "
                 f"제한 광역 {reg.get('wide')}")

    R["v8"] = _d(has_perf and has_reg and not small_neg, perf_ev or reg_ev,
                 "지방계약법 시행규칙 제25조제7항(국가: 시행규칙 제25조) — "
                 "실적제한과 지역제한의 중복 제한 금지")

    # --- v9 특정 모델명 ----------------------------------------------------
    mdl = f.get("model_name") or {}
    R["v9"] = _d(mdl.get("hit") and not mdl.get("equivalent_allowed"), mdl.get("ev"),
                 "(계약예규) 정부 입찰·계약 집행기준 제2장 제5조 — "
                 "특정 규격·모델을 지정하면서 동등 이상 제품을 허용하지 않음")

    # --- 중기간 경쟁제품 계열 ----------------------------------------------
    R["v10"] = _d(is_cp and not dp, None,
                  "판로지원법 제9조 — 중기간 경쟁제품 입찰은 직접생산확인증명서 "
                  "보유를 참가자격으로 요구해야 하나 해당 문구가 없음")

    # 판로지원법 제7조의2 는 지정 품목에 한해 소기업·소상공인 제한경쟁을 허용하므로,
    # '중소기업자 제한 부재'는 기업규모 제한이 아예 없을 때만 위반으로 본다.
    R["v11"] = _d(is_cp and size == SIZE_NONE, None,
                  "판로지원법 제7조제1항 — 중기간 경쟁제품은 중소기업자 간 경쟁으로 "
                  "하여야 하나 참가자격에 기업규모 제한이 전혀 없음")

    # 직생 요구 문단이 인용한 세부품명번호가 고시 경쟁제품이면 적법한 요구다
    dp_ok = bool(f.get("dp_codes_in_cp"))
    R["v12"] = _d((not is_cp) and dp and not dp_ok, f.get("direct_production_ev"),
                  "중기간 경쟁제품이 아닌 일반 물품·용역에 직접생산확인증명서를 요구 — "
                  "국가계약법 시행령 제21조 / 지방계약법 시행령 제20조상 근거 없음")

    R["v13"] = _d(is_cp and size == SIZE_SMALL, f.get("size_ev"),
                  "판로지원법 제7조제1항 — 중기간 경쟁제품은 중소기업자 간 경쟁이어야 하며 "
                  "소기업·소상공인으로 좁혀 제한할 수 없음")

    # --- 판로지원법 금액 구간 (일반 물품·용역) -----------------------------
    general = not is_cp
    in_mid = LX.PANRO_SMALL_LIMIT <= est < gosi
    below = est < LX.PANRO_SMALL_LIMIT
    # 판로지원법 시행령 제2조의3 예외를 공고에 명시하면 기업규모 제한을 두지 않아도 된다
    exc = bool(f.get("panro_exception"))

    R["v14"] = _d(general and est >= gosi and size != SIZE_NONE, f.get("size_ev"),
                  f"추정가격 {est:,}원 ≥ 고시금액 {gosi:,}원 — 판로지원법 시행령 제2조의2 "
                  "우선조달계약 대상이 아니므로 기업규모 제한을 둘 수 없음")

    R["v15"] = _d(general and in_mid and size == SIZE_SMALL, f.get("size_ev"),
                  "판로지원법 시행령 제2조의2제1항제2호 — 1억원 이상 고시금액 미만 구간은 "
                  "중소기업자 간 제한경쟁 대상인데 소기업·소상공인으로 제한")

    R["v16"] = _d(general and in_mid and size == SIZE_NONE and not exc, None,
                  "판로지원법 시행령 제2조의2제1항제2호 — 해당 구간의 "
                  "중소기업자 간 제한경쟁 참가자격이 공고에 없음")

    R["v17"] = _d(general and below and size == SIZE_SME, f.get("size_ev"),
                  "판로지원법 시행령 제2조의2제1항제1호 — 1억원 미만은 소기업·소상공인 간 "
                  "제한경쟁 대상인데 중소기업자로 제한")

    R["v18"] = _d(general and below and size == SIZE_NONE and not exc, None,
                  "판로지원법 시행령 제2조의2제1항제1호 — 1억원 미만 구간의 "
                  "소기업·소상공인 제한 참가자격이 공고에 없음")

    # --- v19 물품공급 확약서 ------------------------------------------------
    pl = f.get("pledge") or {}
    R["v19"] = _d(pl.get("at_bid"), pl.get("ev"),
                  "(계약예규) 정부 입찰·계약 집행기준 제2장 제5조의3 — 제조사 물품공급·"
                  "기술지원 확약서는 계약 단계 요건이며 입찰 단계 요구는 부당한 참가제한")

    # --- v20 SW 참가자격 ----------------------------------------------------
    sw = f.get("sw") or {}
    # 참고: 첨부가 일부 빠진 공고가 있으나, 대기업 참여제한 기준은 공고문 참가자격에
    # 기재해야 하는 사항이라 첨부 탈락을 면책 사유로 보지 않는다 (dev 확인: v20 양성 5건 중
    # 3건이 제안요청서 탈락 상태에서도 위반으로 라벨링됨).
    R["v20"] = _d(sw.get("is_sw") and not sw.get("limit_stated"), None,
                  "소프트웨어진흥법 및 중소SW사업자 사업 참여 지원 지침 제2조 별표1 — "
                  "사업금액 구간별 대기업 참여제한 기준을 공고에 명시하지 않음")

    # --- v21 공동수급 최소지분율 -------------------------------------------
    j = f.get("joint") or {}
    share = j.get("min_share")
    floor = LX.JOINT_MIN_SHARE_FLOOR.get(law, 5.0)
    std = LX.JOINT_MIN_SHARE.get(law, 5.0)
    R["v21"] = _d(share is not None and float(share) < floor, j.get("ev"),
                  f"{'(계약예규) 공동계약운용요령 제9조제5항' if law == '국가계약법' else '지방자치단체 입찰 및 계약 집행기준 제6장 제2절'}"
                  f" — 최소지분율 기준 {std}%(±20% 조정 시 하한 {floor}%)인데 공고는 {share}%")

    # --- v22 / v23 설명회 ---------------------------------------------------
    br = f.get("briefing") or {}
    R["v22"] = _d(nego and br.get("held") and br.get("attendance_required"), br.get("ev"),
                  "국가계약법 시행령 제43조제6항(’19.12.18 삭제) / 지방계약법 시행령 "
                  "제43조제7항(’22.9.20 삭제) — 설명회 참석을 입찰참가 요건으로 삼을 근거 없음")

    v23 = 0
    why23 = ""
    if nego and law == "지방계약법" and br.get("held"):
        need = next(d for lim, d in LX.BRIEFING_LEAD_DAYS if est >= lim)
        bd = br.get("date")
        due = T.parse_meta_date((sig.get("meta") or {}).get("개찰예정일자"))
        notice = T.parse_meta_date((sig.get("meta") or {}).get("공고게시일자"))
        if bd and due:
            gap = T.to_ordinal(due) - T.to_ordinal(bd)
            if gap < need:
                v23, why23 = 1, (f"제안요청서 설명일부터 제안서 제출마감일까지 {gap}일 — "
                                 f"지방자치단체 입찰시 낙찰자 결정기준 제7장 제3절 2-다 상 {need}일 미달")
        if not v23 and bd and notice:
            g2 = T.to_ordinal(bd) - T.to_ordinal(notice)
            if g2 < LX.BRIEFING_NOTICE_DAYS:
                v23, why23 = 1, (f"입찰공고일부터 설명일까지 {g2}일 — 같은 기준상 7일 미달")
    R["v23"] = _d(v23, br.get("ev"), why23 or
                  "협상에 의한 계약(지방계약법)의 제안요청서 설명 일정 기준")

    # --- v24 등록값 대조 ----------------------------------------------------
    mm = f.get("mismatch") or {}
    R["v24"] = _d(mm.get("hit"), mm.get("ev"),
                  f"공고서 표기와 나라장터 등록값 불일치: {mm.get('field') or ''}")

    for v in ABSENCE:
        R[v]["ev"] = None          # 부재탐지 항목은 근거문구를 비운다
    return R


# ---------------------------------------------------------------- 규칙 기반 사실
def rule_facts(sig: Dict[str, Any]) -> Dict[str, Any]:
    """LLM 없이 정규식·사전으로 뽑은 사실. LLM 출력이 없을 때의 폴백이자,
    프롬프트에 넣는 사전 후보이기도 하다."""
    perf, reg, sme = sig["perf"], sig["region"], sig["sme"]
    cls = {None: SIZE_NONE, "small": SIZE_SMALL, "sme": SIZE_SME}[sme["class"]]
    mm = sig["mismatch"]
    mdl = sig["model"].get("strict") or []
    equiv = any(k in b[2] for b in mdl for k in ("동등 이상", "동등이상", "이와 동등"))
    br = sig["briefing"]
    return {
        "perf": {"hit": perf["present"], "amount": perf["amount"],
                 "org_specific": bool(perf["org_blocks"]),
                 "ev": perf["amount_block"] or (perf["blocks"][0] if perf["blocks"] else None),
                 "org_ev": perf["org_blocks"][0] if perf["org_blocks"] else None},
        "region": {"hit": reg["present"], "n_wide": reg.get("n_wide", 0),
                   "wide": reg.get("wide"), "basic_unit": bool(reg.get("has_basic")),
                   "ev": reg["blocks"][0] if reg["blocks"] else None},
        "size_class": cls, "est_override": None, "budget_override": None,
        "panro_exception": sig["sme"].get("panro_exception", False),
        "size_ev": (sme["small_blocks"] or sme["sme_blocks"] or [None])[0],
        "is_competitive": sig["competitive"]["is_cp"],
        "direct_production_required": sig["direct_prod"]["present"],
        "dp_codes": sig["direct_prod"].get("codes") or [],
        "dp_codes_in_cp": sig["direct_prod"].get("codes_in_cp") or [],
        "direct_production_ev": (sig["direct_prod"]["blocks"] or [None])[0],
        "org_only": {"hit": bool(sig.get("org_only_blocks")),
                     "ev": (sig.get("org_only_blocks") or [None])[0]},
        "model_name": {"hit": bool(mdl), "equivalent_allowed": equiv,
                       "ev": mdl[0] if mdl else None},
        "pledge": {"at_bid": bool(sig["pledge"]["at_bid"]),
                   "ev": (sig["pledge"]["at_bid"] or sig["pledge"]["blocks"] or [None])[0]},
        "sw": {"is_sw": sig["sw"]["is_sw"], "limit_stated": bool(sig["sw"]["limit_blocks"])},
        "joint": {"min_share": sig["joint"]["min_share"], "ev": sig["joint"]["share_block"]},
        "briefing": {"held": bool(br["blocks"]), "attendance_required": bool(br["mandatory"]),
                     "date": br["date"], "ev": (br["mandatory"] or br["blocks"] or [None])[0]},
        "mismatch": {"hit": bool(mm["hits"]), "field": mm["hits"][0][0] if mm["hits"] else "",
                     "ev": mm["ev"]},
    }
