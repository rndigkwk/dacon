# -*- coding: utf-8 -*-
"""레코드 1건에서 규칙 판정에 필요한 신호를 뽑는다 (LLM 호출 없음)."""
from __future__ import annotations

import csv
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from . import lexicon as LX
from . import textutil as T

Block = Tuple[int, int, str]

DOC_JOIN = "\n"

# ---------------------------------------------------------------- 경쟁제품표
# 중기부 고시 경쟁제품 세부품명표.
# 1순위: 평가 서버가 제공하는 data/법령패키지 스냅샷 (채점 기준 원본)
# 2순위: 같은 파일을 그대로 복사해 둔 model/ 사본 (경로가 없을 때만 사용)
_CP_PATHS = [
    os.path.join(os.environ.get("PPS_DATA_DIR", "./data"),
                 "법령패키지", "중기부고시", "중기부고시_경쟁제품_세부품명.csv"),
    os.path.join(os.path.dirname(__file__), "competitive_products.csv"),
]
_CP_CODES: Set[str] = set()
_CP_NAMES: Set[str] = set()


def _load_cp():
    if _CP_CODES:
        return
    for path in _CP_PATHS:
        try:
            with open(path, encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    code = (row.get("세부품명번호") or "").strip()
                    name = (row.get("세부품명") or "").strip()
                    if code:
                        _CP_CODES.add(code)
                    if name:
                        _CP_NAMES.add(name)
        except OSError:
            continue
        if _CP_CODES:
            return


CODE_RE = re.compile(r"(?<!\d)(\d{10})(?!\d)")


CP_CTX = ("직접생산", "경쟁제품", "판로지원", "중소기업제품 구매촉진")


def product_codes(sig_text: str, meta_pum: str) -> Set[str]:
    """세부품명번호를 모은다.

    본문 번호는 '직접생산확인/경쟁제품' 문맥(앞뒤 400자) 안에 있을 때만 채택한다.
    규격표·참고자료에 우연히 적힌 10자리 숫자로 경쟁제품이라고 단정하지 않기 위함이다.
    반환값은 (전체 코드, 문맥 있는 코드).
    """
    meta_codes = set(CODE_RE.findall(meta_pum or ""))
    ctx_codes = set(meta_codes)
    all_codes = set(meta_codes)
    for pat in (r"세부\s*품?명?\s*번?호?[^0-9]{0,20}(\d{10})", r"\[(\d{10})\]"):
        for m in re.finditer(pat, sig_text):
            code = m.group(1)
            all_codes.add(code)
            win = sig_text[max(0, m.start() - 400): m.end() + 400]
            if any(k in win for k in CP_CTX):
                ctx_codes.add(code)
    return all_codes, ctx_codes

# 참가자격 섹션이 끝났다고 볼 수 있는 다음 절 제목
SECTION_END = re.compile(
    r"(?:제출\s*서류|입찰\s*방법|입찰\s*및|낙찰자\s*결정|입찰보증금|계약\s*체결|"
    r"유의\s*사항|입찰\s*무효|과업\s*내용|평가\s*방법|제안서\s*작성|첨부\s*파일|"
    r"청렴계약|기타\s*사항|공동수급|현장설명)"
)
# '…업체이어야 한다' 류 — 참가자격 요건임을 알려주는 말꼬리
REQUIRE_TAIL = [
    "업체이어야", "업체여야", "이어야 합니다", "여야 합니다", "하여야 합니다",
    "해야 합니다", "있는 자", "있는 업체", "보유한 업체", "보유한 자", "소지한 업체",
    "소지한 자", "참가할 수 있", "참가자격이 없", "참가 자격", "제한합니다", "제한하며",
    "만 참여", "에 한하여", "한함", "한합니다", "충족", "갖춘 자", "둔 업체", "둔 자",
]


QUAL_HINT = ("자격", "제한", "실적", "소재지", "본점", "주된 영업소", "중소기업", "소기업",
             "소상공인", "확인서", "직접생산", "업종", "면허", "공동수급", "확약서",
             "참여", "참가", "업체", "등록증")


def full_text(rec: Dict[str, Any]) -> str:
    return DOC_JOIN.join(d["text"] for d in rec["docs"])


def doc_spans(rec: Dict[str, Any]) -> List[Tuple[int, int, str]]:
    """문서별 (start, end, type) — 오프셋이 full_text 기준이 되도록 계산."""
    spans, pos = [], 0
    for d in rec["docs"]:
        spans.append((pos, pos + len(d["text"]), d["type"]))
        pos += len(d["text"]) + len(DOC_JOIN)
    return spans


def _is_require(text: str) -> bool:
    return any(k in text for k in REQUIRE_TAIL)


def qualification_span(text: str) -> List[Tuple[int, int]]:
    """'입찰참가자격' 제목 이후 구간들의 (start, end) 목록."""
    spans = []
    for kw in LX.KW_QUALIFICATION_HEAD:
        for m in re.finditer(re.escape(kw), text):
            s = m.start()
            tail = text[s + 200: s + 6500]
            m2 = SECTION_END.search(tail)
            e = (s + 200 + m2.start()) if m2 else min(len(text), s + 6500)
            spans.append((s, e))
    spans.sort()
    merged: List[Tuple[int, int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def in_spans(b: Block, spans) -> bool:
    return any(s <= b[0] < e for s, e in spans)


# ---------------------------------------------------------------- 메인 추출
def extract(rec: Dict[str, Any]) -> Dict[str, Any]:
    text = T.normalize(full_text(rec))
    blocks = T.split_blocks(text)
    qspans = qualification_span(text)
    # 구간 안의 블록 ∪ '…이어야 합니다' 같은 요건 말꼬리를 가진 자격 관련 블록.
    # 구간 탐지가 어긋나도 요건 문단을 놓치지 않게 합집합으로 잡는다.
    qblocks = [b for b in blocks
               if in_spans(b, qspans)
               or (_is_require(b[2]) and any(k in b[2] for k in QUAL_HINT))]

    m = rec.get("meta", {}) or {}

    def mv(k):
        v = m.get(k)
        return None if v in (None, "", "미입력") else v

    est = mv("입찰추정가격")
    budget = mv("배정예산금액")
    est = int(est) if isinstance(est, (int, float)) else None
    budget = int(budget) if isinstance(budget, (int, float)) else None
    if est is None and budget:
        est = int(budget / 1.1)
    est_meta, budget_meta = est, budget
    law = mv("적용계약법") or "국가계약법"
    law_meta = law
    biz = mv("업무구분") or ""
    jo = str(m.get("조항호내용") or "")

    head = ""
    head_long = ""
    for d in rec["docs"]:
        if d["type"] == "공고문":
            nd = T.normalize(d["text"])
            head, head_long = nd[:2500], nd[:14000]
            break

    # 대회 명세: 본문 기재값과 meta 가 다르면 '적용 법령·금액 구간 판단'은 공고문 기재값을 우선하고,
    # 본문에 기재가 없는 값만 meta 를 기준으로 한다. (불일치 자체는 v24 판정 대상)
    # 대회 명세상 금액 구간 판단은 공고문 기재값이 우선이지만, 정규식으로 뽑은 값은
    # 법령 인용문·실적요구 금액을 잘못 집는 일이 잦다(dev 25건 중 다수가 오추출).
    # 그래서 규칙 경로는 meta 를 쓰고, 공고문 금액은 LLM 이 뽑은 값만 채택한다.
    doc_price_hint = _doc_price(head_long[:3000])
    sig_price_src = "meta"

    types = {d["type"] for d in rec["docs"]}
    spec_spans = [(s, e) for s, e, t in doc_spans(rec) if t in ("규격서", "과업지시서", "제안요청서")]

    sig: Dict[str, Any] = {
        "id": rec["id"], "text": text, "blocks": blocks, "qblocks": qblocks,
        "qspans": qspans, "meta": m,
        "headings": T.heading_index(text), "doc_spans": doc_spans(rec), "est": est, "budget": budget, "law": law,
        "biz": biz, "jo": jo, "doc_types": types,
        "method": mv("계약방법") or "", "award": mv("낙찰방법") or "",
        "org": mv("소관구분") or "", "pum": str(m.get("세부품명번호목록") or ""),
        "est_meta": est_meta, "budget_meta": budget_meta, "law_meta": law_meta,
        "price_src": sig_price_src, "doc_price_hint": doc_price_hint,
        "dropped": rec.get("dropped_doc_counts") or {},
        "completeness": rec.get("input_completeness") or {},
        "region_flag": (m.get("지역제한여부") or "") == "Y",
        "lic_flag": (m.get("업종제한여부") or "") == "Y",
        "joint_mode": mv("공동도급구성방식") or "",
        "urgent": (m.get("긴급공고여부") or "") == "Y",
        "is_goods": "물품" in biz, "is_service": "용역" in biz,
    }

    # ---- 금액 기준 ------------------------------------------------------
    eng = any(k in head for k in ("건설기술용역", "엔지니어링사업", "건설엔지니어링",
                                  "설계공모", "공사감리", "건축사"))
    safety = any(k in head for k in ("정밀안전진단", "안전점검"))
    if law == "지방계약법":
        if safety:
            region_limit = LX.LOCAL_REGION_LIMIT_SAFETY
        elif eng:
            region_limit = LX.LOCAL_REGION_LIMIT_ENGINEERING
        else:
            region_limit = LX.LOCAL_REGION_LIMIT_DEFAULT
    else:
        region_limit = LX.GOSI_STATE_GOODS
    sig["region_limit"] = region_limit
    sig["gosi"] = LX.GOSI_STATE_GOODS

    # ---- 실적제한 -------------------------------------------------------
    perf_blocks = [b for b in qblocks
                   if any(k in b[2] for k in LX.KW_PERFORMANCE)
                   and ("이상" in b[2] or "보유" in b[2] or "제한" in b[2])
                   and _is_require(b[2])
                   and not any(k in b[2] for k in ("배점", "평가항목", "가점", "점수", "만점"))]
    perf_amt, perf_amt_block = None, None
    for b in perf_blocks:
        v = T.max_amount(b[2])
        if v and (perf_amt is None or v > perf_amt):
            perf_amt, perf_amt_block = v, b
    sig["perf"] = {
        "present": bool(perf_blocks), "blocks": perf_blocks,
        "amount": perf_amt, "amount_block": perf_amt_block,
        "org_blocks": [b for b in perf_blocks
                       if any(k in b[2] for k in LX.KW_ORG_SPECIFIC)],
    }

    # ---- 지역제한 -------------------------------------------------------
    reg_blocks = [b for b in qblocks
                  if any(k in b[2] for k in LX.KW_REGION) and _is_require(b[2])]
    reg_info = {"present": bool(reg_blocks) or sig["region_flag"], "blocks": reg_blocks,
                "wide": [], "n_wide": 0, "has_basic": False}
    joined = "\n".join(b[2] for b in reg_blocks)
    if joined:
        r = T.extract_regions(joined)
        reg_info.update(r)
    if sig["region_flag"] and m.get("제한지역코드목록"):
        rm = T.extract_regions(str(m["제한지역코드목록"]))
        reg_info["meta_wide"] = rm["wide"]
        reg_info["meta_basic"] = rm["has_basic"]
        if not reg_info["wide"]:
            reg_info.update({k: rm[k] for k in ("wide", "n_wide", "has_basic")})
    sig["region"] = reg_info

    # ---- 중소기업 / 소기업 ---------------------------------------------
    size = _size_requirement(qblocks, jo)
    size["panro_exception"] = bool(PANRO_EXCEPT.search(re.sub(r"\s+", " ", text)))
    sig["sme"] = size

    # ---- 직접생산 / 중기간 경쟁제품 ------------------------------------
    _load_cp()
    dp_blocks = [b for b in blocks if any(k in b[2] for k in LX.KW_DIRECT_PROD)]
    cp_blocks = [b for b in blocks if any(k in b[2] for k in LX.KW_COMPETITIVE_PRODUCT)]
    codes, ctx_codes = product_codes(text, sig["pum"])
    code_hit = sorted(ctx_codes & _CP_CODES)
    name_hit = [n for n in _CP_NAMES if len(n) >= 3 and n in text]
    if codes:
        is_cp = bool(code_hit)            # 세부품명번호가 있으면 고시 목록이 최종 판단
    else:
        is_cp = bool(cp_blocks) or any(k in jo for k in LX.JO_COMPETITIVE) or (
            bool(dp_blocks) and re.search(r"판로지원[^\n]{0,30}제\s*9\s*조", text) is not None)
    # 직접생산 요구 문단 안에 적힌 세부품명번호 — v12(일반제품 직생제한) 판정에 쓴다
    dp_codes = set()
    for b in dp_blocks:
        dp_codes |= set(CODE_RE.findall(b[2]))
    sig["direct_prod"] = {"present": bool(dp_blocks), "blocks": dp_blocks,
                          "codes": sorted(dp_codes),
                          "codes_in_cp": sorted(dp_codes & _CP_CODES)}
    sig["competitive"] = {"is_cp": is_cp, "blocks": cp_blocks or dp_blocks,
                          "codes": sorted(codes), "code_hit": code_hit,
                          "name_hit": name_hit[:3]}

    # ---- 특정 모델명 ----------------------------------------------------
    BRAND = re.compile(r"[A-Z][A-Za-z]{2,}(?:[\s\-][A-Za-z0-9]{1,})?|[A-Za-z]{2,}[\-]?\d{2,}")
    model_blocks = [b for b in blocks
                    if in_spans(b, spec_spans) and BRAND.search(b[2])
                    and (any(k in b[2] for k in ("모델명", "모델 명", "제조사", "제조회사",
                                                 "브랜드", "제품명", "규격", "품명"))
                         or any(k in b[2] for k in ("시리즈", "일 것", "동등", "전용")))
                    and not any(k in b[2] for k in ("업종코드", "세부품명번호", "사업자등록",
                                                    "전자입찰", "나라장터", "http"))]
    # 규칙 폴백용으로는 '제조사/모델명' 이 명시적으로 적힌 문단만 쓴다.
    strict = [b for b in model_blocks
              if any(k in b[2] for k in ("모델명", "모델 명", "제조사", "제조회사", "브랜드"))]
    sig["model"] = {"blocks": model_blocks, "strict": strict}
    # 규격서·과업지시서의 규격 서술 문단 (v9 판단용 문맥)
    # 규격서·과업지시서의 규격 서술 문단 (v9 판단용 문맥).
    # 브랜드·모델명으로 보이는 라틴문자 토큰이 있는 문단을 앞쪽에 둔다
    # — 콜론 없이 "DJI Matrice 4E/T 시리즈 … 일 것" 처럼 적힌 경우를 놓치지 않기 위함.
    cand = [b for b in blocks if in_spans(b, spec_spans) and len(b[2]) >= 12
            and (":" in b[2] or "규격" in b[2] or "사양" in b[2] or BRAND.search(b[2]))]
    cand.sort(key=lambda b: (-_brand_score(b[2]), b[0]))
    sig["spec_blocks"] = sorted(cand[:26], key=lambda b: b[0])
    # 제조사·모델명처럼 보이는 문단만 따로 — 프롬프트에서 우선 배치한다
    sig["spec_brand_blocks"] = [b for b in cand if _brand_score(b[2]) > 0][:8]

    # ---- 물품공급 확약서 ------------------------------------------------
    pl = [b for b in blocks
          if any(k in b[2] for k in LX.KW_SUPPLY_PLEDGE)
          and any(k in b[2] for k in LX.KW_PLEDGE_CTX)]
    pl_bid = [b for b in pl if any(k in b[2] for k in LX.KW_PLEDGE_AT_BID)]
    sig["pledge"] = {"blocks": pl, "at_bid": pl_bid}

    # ---- 현장설명회 -----------------------------------------------------
    br_blocks = [b for b in blocks if any(k in b[2] for k in LX.KW_BRIEFING)]
    br_mand = [b for b in br_blocks if any(k in b[2] for k in LX.KW_BRIEFING_MANDATORY)]
    br_date = None
    for b in br_blocks:
        ds = T.find_dates(b[2])
        if ds:
            br_date = ds[0][1]
            break
    sig["briefing"] = {"blocks": br_blocks, "mandatory": br_mand, "date": br_date}

    # ---- 공동수급 -------------------------------------------------------
    j_blocks = [b for b in blocks if any(k in b[2] for k in LX.KW_JOINT)]
    share, share_block = None, None
    for b in j_blocks:
        for mm in re.finditer(r"(\d{1,3}(?:\.\d+)?)\s*(?:%|퍼센트)", b[2]):
            v = float(mm.group(1))
            ctx = b[2][max(0, mm.start() - 40): mm.end() + 20]
            if any(k in ctx for k in ("최소", "이상", "지분", "출자")) and 0 < v <= 50:
                if share is None or v < share:
                    share, share_block = v, b
    sig["joint"] = {"blocks": j_blocks, "allowed": bool(j_blocks) or bool(sig["joint_mode"]),
                    "min_share": share, "share_block": share_block}

    # ---- 소프트웨어 사업 ------------------------------------------------
    # ---- 특정 기관·단체 한정 (v1) ---------------------------------------
    sig["org_only_blocks"] = [
        b for b in qblocks
        if any(k in b[2] for k in LX.KW_ORG_SPECIFIC)
        and re.search(r"(만\s*참여|만\s*가능|에\s*한하여|에\s*한함|으로\s*한정|"
                      r"만을\s*대상|만\s*입찰|에\s*限)", b[2])]

    # ---- v24: 공고서 vs 나라장터 등록값 대조 -----------------------------
    sig["mismatch"] = _mismatch(sig, head, blocks)

    sw_strong = ("소프트웨어" , "정보시스템", "시스템 구축", "시스템구축", "응용프로그램",
                 "홈페이지 구축", "누리집 구축", "앱 개발", "어플리케이션 개발",
                 "플랫폼 구축", "정보화사업", "SW사업", "전산시스템")
    sw_hits = sum(1 for k in sw_strong if k in head) + sum(1 for k in sw_strong if k in text)
    sig["sw"] = {
        "is_sw": sig["is_service"] and (
            any(k in head for k in sw_strong) or sw_hits >= 3
            or (sig["meta"].get("정보화사업여부") == "Y")),
        "limit_blocks": [b for b in blocks
                         if any(k in b[2] for k in ("대기업", "상호출자제한기업집단"))
                         and any(k in b[2] for k in ("참여", "제한"))],
    }
    return sig


# ---------------------------------------------------------------- v24 대조
AMOUNT_LABEL = ("사업금액", "사업비", "추정가격", "배정예산", "예산액", "총사업비",
                "용역금액", "구매금액", "사업예산", "계약금액", "기초금액")
METHOD_WORDS = ("일반경쟁", "제한경쟁", "지명경쟁", "수의계약")


def _mismatch(sig, head, blocks):
    """공고문 표기와 나라장터 등록값이 어긋나는 지점을 찾는다.

    정밀도가 중요하므로 (1) 금액과 (2) 지역제한만 규칙으로 확정하고,
    업종·계약방법 대조는 후보만 넘겨 LLM이 판단하게 한다.
    """
    out = {"hits": [], "cands": [], "ev": None}
    budget, est = sig.get("budget_meta"), sig.get("est_meta")   # 대조는 등록 원값으로

    # (1) 대표 사업금액 대조 -------------------------------------------
    known = set()
    for v in (budget, est):
        if v:
            known |= {v, int(round(v * 1.1)), int(round(v / 1.1)),
                      int(round(v * 1.1 / 10) * 10), int(round(v / 1.1 / 10) * 10)}

    def close(v):
        return any(abs(v - k) <= max(1000, k * 0.005) for k in known)

    if known:
        labeled = []
        for b in blocks:
            if b[0] > 12000:
                break
            for lab in AMOUNT_LABEL:
                for lm in re.finditer(re.escape(lab), b[2]):
                    tail = b[2][lm.end(): lm.end() + 40]
                    am = T.find_amounts(tail)
                    if am:
                        labeled.append((am[0][2], b))
        plaus = [(v, b) for v, b in labeled
                 if v >= 1_000_000 and any(0.7 <= v / k <= 1.4 for k in known)]
        if plaus and not any(close(v) for v, _ in plaus):
            v, b = max(plaus, key=lambda x: x[0])
            out["hits"].append(("예산", v, budget, est))
            out["ev"] = b

    # (2) 지역제한 유무 대조 -------------------------------------------
    strong = [b for b in sig["region"]["blocks"]
              if T.extract_regions(b[2])["n_wide"] >= 1
              and any(k in b[2] for k in ("제한", "둔 업체", "둔 자", "소재하고",
                                          "있는 업체", "이어야", "소재한"))]
    doc_region = bool(strong)
    if doc_region != bool(sig["region_flag"]):
        # 정밀도가 낮아 규칙 확정에는 쓰지 않고 LLM 판단 후보로만 넘긴다
        out["cands"].append(("지역제한", doc_region, sig["region_flag"]))
        out["region_ev"] = strong[0] if strong else None

    # (3) 후보(LLM 판단용) ---------------------------------------------
    lic_blocks = [b for b in sig["qblocks"]
                  if any(k in b[2] for k in ("업종", "면허", "업종코드"))
                  and _is_require(b[2])]
    if bool(lic_blocks) != bool(sig["lic_flag"]):
        out["cands"].append(("업종제한", bool(lic_blocks), sig["lic_flag"]))
    meth = sig["method"]
    if meth:
        found = [w for w in METHOD_WORDS if w in head]
        if found and meth not in found:
            out["cands"].append(("계약방법", found, meth))
    return out


# ------------------------------------------------- 기업규모 참가자격 분류기
# 참가자격 문장에서 '허용된 기업 계층'을 직접 읽는다.
#   "「중소기업기본법」 제2조에 따른 소기업자 또는 ... 소상공인"  → 소기업·소상공인
#   "「중소기업기본법」 제2조에 따른 중소기업자 또는 ... 소상공인" → 중소기업자
# 법률 이름(중소기업기본법, 중소기업제품 구매촉진…)은 계층 표현이 아니므로
# '따른/의거/의한/해당하는' 뒤에 오는 명사만 본다.
SIZE_CLASS_RE = re.compile(
    r"(?:따른|의거|의한|해당하는|해당되는|갖춘|보유한|인)\s*[\u300c\uff62\u2018\u201c]?\s*"
    r"(중소기업자|중소기업|소기업자|소기업|소상공인)")

# 판로지원법 시행령 제2조의3 (우선조달계약에 대한 예외).
# 예외를 공고에 명시하면 기업규모 제한이 없어도 위반이 아니다 (항목표 v15~v18 비고).
PANRO_EXCEPT = re.compile(
    r"제\s*2\s*조의\s*3"
    r"|우선조달계약에?\s*대한\s*예외"
    r"|우선조달계약\s*예외"
    r"|공공구매제도\s*운영요령[\s\S]{0,20}제\s*44\s*조")
CERT_SMALL = re.compile(r"소기업\s*[·ㆍ,]?\s*(?:및|또는)?\s*(?:소상공인)?\s*확인서"
                        r"|소상공인\s*확인서")
CERT_SME = re.compile(r"중소기업\s*[·ㆍ,]?\s*(?:소상공인)?\s*확인서")
SIZE_ANCHOR = re.compile(
    r"(확인서|확인증|제한경쟁|해당하는 업체|해당되는 업체|에 한합|에 한함|에 한하여|"
    r"이어야|여야|소지|참가자격|참가 자격|간 경쟁)")
SIZE_NOISE = re.compile(
    r"(무효|발급되지|발급받지 못한|신청한 경우|다른 경우|이후인 경우|유효기간 시작일|"
    r"간주 특별법인|확인이 되지 않을|미제출|반려|위반한 사실|신고 할 수)")


def _size_requirement(qblocks, jo):
    """참가자격에서 요구한 기업규모를 'small' / 'sme' / None 으로 분류한다.

    판로지원법 시행령 제2조의2 는 추정가격 구간별로 요구해야 할 기업규모를 정하므로,
    '무엇으로 제한했는가' 한 개의 라벨이 v14~v18 판정의 축이 된다.
    """
    small_b, sme_b = [], []
    for b in qblocks:
        raw = b[2]
        t = re.sub(r"\s+", " ", raw)          # 줄바꿈으로 쪼개진 어절을 붙인다
        t = re.sub(r"중\s*[·ㆍ・．.]\s*소기업", "중소기업", t)   # '중·소기업' → '중소기업'
        t = re.sub(r"소\s*[·ㆍ・．.]\s*상공인", "소상공인", t)
        if not SIZE_ANCHOR.search(t) or SIZE_NOISE.search(t):
            continue
        classes = {m.group(1) for m in SIZE_CLASS_RE.finditer(t)}
        classes = {("중소기업" if c.startswith("중소기업") else
                    "소기업" if c.startswith("소기업") else c) for c in classes}
        if "중소기업" in classes:
            sme_b.append(b)
        elif classes & {"소기업", "소상공인"}:
            small_b.append(b)
        elif CERT_SME.search(t):
            sme_b.append(b)
        elif CERT_SMALL.search(t):
            small_b.append(b)
    jo_small = any(k in jo for k in LX.JO_SMALL) and "중기업" not in jo
    jo_sme = ("중기업" in jo) or ("중소기업자" in jo)
    # 공고서 본문이 기준이다 (나라장터 등록 조항호는 v24 대조용으로만 남긴다).
    # 더 좁은 제한(소기업)이 명시돼 있으면 그쪽을 택한다.
    if small_b and sme_b:
        cls = "small" if small_b[0][0] < sme_b[0][0] else "sme"
    elif small_b:
        cls = "small"
    elif sme_b:
        cls = "sme"
    else:
        cls = None
    return {"class": cls, "small_blocks": small_b, "sme_blocks": sme_b,
            "both_blocks": [], "sme_required": cls == "sme",
            "small_required": cls == "small",
            "jo_small": jo_small, "jo_sme": jo_sme, "from_meta_only": False}


# ------------------------------------------------- 공고문 기재 금액·적용법령
PRICE_DIRECT = ("추정가격", "추정금액")
PRICE_VAT_IN = ("기초금액", "배정예산", "사업금액", "사업비", "예산액", "총사업비",
                "용역금액", "구매금액", "사업예산", "계약금액")


def _doc_price(head: str):
    """공고문 머리에서 추정가격/사업금액을 읽는다.

    '추정가격' 이 직접 적혀 있으면 그 값을, 없으면 부가세 포함 금액을 1.1 로 나눠
    추정가격으로 환산한다. (금액 구간 판단의 기준은 추정가격이다)
    반환: (추정가격, 사업예산, 출처)
    """
    if not head:
        return None, None, "meta"
    flat = head
    est = budget = None
    for lab in PRICE_DIRECT:
        for mm in re.finditer(re.escape(lab), flat):
            am = T.find_amounts(flat[mm.end(): mm.end() + 40])
            if am and am[0][2] >= 1_000_000:
                est = am[0][2]
                break
        if est:
            break
    for lab in PRICE_VAT_IN:
        for mm in re.finditer(re.escape(lab), flat):
            am = T.find_amounts(flat[mm.end(): mm.end() + 40])
            if am and am[0][2] >= 1_000_000:
                budget = am[0][2]
                break
        if budget:
            break
    if est is not None:
        return est, budget, "doc_direct"
    # 부가세 포함 금액만 있는 경우의 1.1 환산은 오차가 커서 채택하지 않는다
    return None, budget, "meta"


LAW_LOCAL = re.compile(r"지방자치단체를 당사자로 하는 계약|지방계약법|지방자치단체 입찰")
LAW_STATE = re.compile(r"국가를 당사자로 하는 계약|국가계약법")


def _doc_law(head: str):
    """공고문이 인용한 계약법령. 한쪽만 뚜렷하게 인용할 때만 채택한다."""
    if not head:
        return None
    nl, ns = len(LAW_LOCAL.findall(head)), len(LAW_STATE.findall(head))
    if nl >= 2 and nl > ns * 2:
        return "지방계약법"
    if ns >= 2 and ns > nl * 2:
        return "국가계약법"
    return None


# ----------------------------------------------- 제조사·모델명 후보 점수
# 규격서에는 GNSS·RTK·USB 같은 기술 약어가 잔뜩 나와서 '대문자 토큰'만으로는
# 브랜드를 가려낼 수 없다. 흔한 규격 약어·단위를 제외하고,
# 회사명처럼 연속된 라틴 낱말이나 영문+숫자 모델코드에 가중치를 준다.
SPEC_STOP = {
    "GNSS", "GPS", "RTK", "USB", "LED", "LCD", "OLED", "HDMI", "LAN", "WAN",
    "WIFI", "BLE", "NFC", "RFID", "FCC", "CE", "SRRC", "MIC", "KC", "IP",
    "AC", "DC", "PC", "CPU", "GPU", "RAM", "SSD", "HDD", "API", "SDK", "PSDK",
    "ISO", "IEC", "KS", "ASTM", "MAX", "MIN", "TYPE", "MODE", "SET", "EA",
    "MM", "CM", "KM", "KG", "PPM", "HZ", "KHZ", "MHZ", "GHZ", "VDC", "VAC",
    "FIX", "FOV", "IPS", "PDF", "JPG", "PNG", "AND", "OR", "THE", "AS",
    # 일반 규격·포맷 표기 — 브랜드가 아니다 (v9 오탐의 주된 원인)
    "FULL", "HD", "FHD", "UHD", "QHD", "SD", "MP4", "MOV", "AVI", "MP3",
    "A3", "A4", "B4", "B5", "CCTV", "DVR", "NVR", "POE", "RGB", "CMYK",
    "TB", "GB", "MB", "KB", "BPS", "MBPS", "FPS", "DPI", "PPI", "LUX",
    "XLS", "XLSX", "DOC", "DOCX", "PPT", "HWP", "HWPX", "ZIP", "CSV",
    "VER", "V1", "V2", "NO", "TEL", "FAX", "URL", "QR", "AI", "IOT",
}
LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]{1,}")


def _brand_score(text: str) -> int:
    """제조사·모델명이 적혔을 법한 정도. 0이면 일반 규격 서술로 본다."""
    words = LATIN_WORD.findall(text)
    if not words:
        return 0
    score = 0
    meaningful = [w for w in words if w.upper() not in SPEC_STOP and len(w) >= 2]
    for w in meaningful:
        if re.search(r"[A-Za-z]", w) and re.search(r"\d", w):
            score += 2                      # 4E, GB10 같은 모델코드
        elif w[:1].isupper() and len(w) >= 3:
            score += 1                      # 회사명처럼 보이는 낱말
    # 연속된 라틴 낱말 두 개 이상 (예: "DJI Matrice")
    if re.search(r"[A-Za-z]{2,}\s+[A-Za-z0-9][A-Za-z0-9\-/]{1,}", text):
        score += 3
    for k in ("제조사", "모델명", "브랜드", "시리즈", "社", "사 제품"):
        if k in text:
            score += 3
    return score
