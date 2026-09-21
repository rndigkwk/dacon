# -*- coding: utf-8 -*-
"""고정 모델에 '법 해석'이 아니라 '사실 추출'만 시키는 프롬프트.

법령 적용은 model/law.py 가 결정적으로 수행하므로, 모델은
공고문에서 확인 가능한 사실 12가지만 뽑으면 된다. 덕분에
 · 출력 스키마가 작아 디코딩이 빠르고
 · 금액 구간 같은 계산 실수가 사라지며
 · 근거문구를 항목이 아니라 사실 단위로 한 번만 요구하면 된다.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from . import lexicon as LX
from . import textutil as T

META_FIELDS = [
    "적용계약법", "업무구분", "계약방법", "낙찰방법", "낙찰하한율",
    "배정예산금액", "입찰추정가격", "소관구분", "공동도급구성방식", "정보화사업여부",
    "세부품명번호목록", "제한지역코드목록", "지역제한여부", "면허업종제한목록", "업종제한여부",
    "조항호내용", "공고게시일자", "개찰예정일자", "긴급공고여부", "입찰방법", "조달방식",
]

SYSTEM = """너는 공공 입찰공고를 읽고 '사실'만 뽑아내는 추출기다.
위반 여부나 법 조문은 판단하지 않는다. 오직 공고문·첨부문서에 쓰여 있는 내용만 본다.

규칙
1. 모든 근거문구는 제시된 문서에 **그대로 있는 연속된 부분문자열**이어야 한다.
   요약·수정·띄어쓰기 변경 모두 금지. 해당 사실이 없으면 null 을 넣는다.
2. 근거문구는 **200자 이내**로, 판단의 핵심이 담긴 한 문장(또는 한 항목)만 고른다.
   길게 붙여 쓰지 말 것. 해당 없으면 반드시 null.
3. 문서에 없으면 '없음/false' 로 답한다. 추측하지 않는다.
4. 제출서류 목록에 이름만 올라온 서류는 '참가자격 요건'이 아니다.
   참가자격으로 요구한 것만 해당(true)으로 본다.

각 필드의 뜻
- 공고문금액: 이 공고가 **자기 사업의 규모로 밝힌 금액**.
  추정가격 = 부가세 제외 금액(공고 개요의 '추정가격'), 사업예산 = 부가세 포함 금액
  (기초금액·사업금액·배정예산·용역금액). 공고 앞머리 개요표에 있는 값을 쓴다.
  다음은 절대 쓰지 않는다 — 법령을 인용한 문장 속 금액("추정가격이 고시금액 미만인 경우"),
  참가자격의 실적 요구금액, 보증금·위약금, 단가·수량 단위 금액.
  개요에서 찾지 못하면 null.
- 참가자격_특정기관한정: 특정 기관·단체 유형(대학, 산학협력단, 협회, 조합, 공공기관,
  특정 법인 등)만 입찰에 참여할 수 있다고 한정했는가.
- 실적제한: 과거 수행·납품 실적을 참가자격으로 요구했는가.
  요구실적금액은 요구한 실적의 최소 금액(원, 정수). 여러 개면 가장 큰 값.
  발주처_특정은 '국가기관이 발주한', '공공기관에서 시행한', '대학병원에 납품한'처럼
  실적의 발주처·수요처를 한정했는지 여부.
- 지역제한: 본점·주된 영업소 소재지로 참가자격을 제한했는가.
  제한광역목록은 제한한 시·도 이름만 (예: ["경기도","서울특별시"]).
  기초단위포함은 시·군·구 단위까지 좁혔는지 여부 (익명화 토큰 '단위=기초' 포함).
- 기업규모제한: 참가자격으로 **입찰에 참여할 수 있다고 허용한 기업 계층**.
  판단은 '…에 따른 OOO' 의 OOO 만 본다. 법률 이름(중소기업기본법, 중소기업제품 구매촉진
  및 판로지원에 관한 법률, 중소기업 범위 및 확인에 관한 규정)은 계층 표현이 아니다.
    · "「중소기업기본법」 제2조에 따른 **소기업자** 또는 … **소상공인**" → 소기업소상공인
    · "「중소기업기본법」 제2조에 따른 **중소기업자** 또는 … **소상공인**" → 중소기업자
    · 요구 서류가 '소기업·소상공인확인서' 면 소기업소상공인,
      '중소기업확인서' 면 중소기업자.
    · 기업 규모에 대한 참가자격 요건이 아예 없으면 '없음'.
  ※ 입찰 무효 사유 설명("확인서가 발급되지 않은 경우", "발급받지 못한 업체에 한함")은
     참가자격 요건이 아니다. 이런 문장만 있으면 '없음'.
- 중기간경쟁제품: 중소기업자간 경쟁제품(중소벤처기업부장관 지정 품목) 입찰인가.
- 직접생산확인요구: 직접생산확인증명서 보유를 참가자격으로 요구했는가.
- 특정모델명: 규격서·과업지시서·공고문에 **특정 회사의 제품을 지목**했는가.
  "제조사 : OOO", "모델명 : XXX", "OO사 △△ 시리즈", 영문 브랜드·모델명(예: GB10,
  Matrice 4E/T) 처럼 그 회사 제품만 가리키는 표기가 있으면 true.
  치수·용량·재질 같은 일반 규격 서술만 있으면 false.
  동등이상허용은 '또는 동등 이상', '이와 동등한 규격' 같은 대체 허용 문구가 있는지.
- 물품공급확약서: 제조사·공급사의 물품공급/기술지원(A/S) 확약서를
  '입찰 전·입찰서 제출 마감일까지' 보유하거나 제출하도록 요구했는가.
- 소프트웨어사업: 소프트웨어 개발·정보시스템 구축 사업인가,
  그리고 대기업(상호출자제한기업집단) 참여제한 기준이 공고에 적혀 있는가.
- 공동수급: 공동수급체 구성원의 최소지분율(%)을 숫자로. 없으면 null.
- 설명회: 현장설명회·사업설명회·제안요청서 설명을 여는가, 참석이 입찰 참가의 조건인가,
  설명회 일자(YYYY-MM-DD).
- 등록값불일치: [나라장터 등록정보]와 공고문 내용이 서로 다른 곳이 있는가
  (예산액, 계약방법, 지역제한 유무, 업종·면허 제한). 있으면 항목명과 공고문 쪽 근거문구.

- 등록값불일치는 아래 네 가지만 비교한다. '[나라장터 등록정보]'와 공고문이 실제로
  어긋날 때만 true 이고, 근거문구는 **공고문 쪽 문장**을 옮긴다.
    1) 예산: 공고문의 사업금액·기초금액·추정가격 ↔ 배정예산금액 / 입찰추정가격
       (같은 금액을 부가세 포함/별도로 달리 쓴 것은 불일치가 아니다)
    2) 계약방법: 공고문의 일반경쟁·제한경쟁·수의계약 표기 ↔ 계약방법
    3) 지역제한: 공고문에 소재지 제한 문구가 있는지 ↔ 지역제한여부(Y/N)·제한지역코드목록
       (제한한 시·도가 서로 다른 경우도 불일치)
    4) 업종·면허: 공고문의 업종·면허 요건 ↔ 업종제한여부·면허업종제한목록
  네 가지 중 어느 것도 어긋나지 않으면 false.

[예시 1] 참가자격에 "「중소기업기본법」 제2조에 따른 소기업자 또는 「소상공인 보호 및
지원에 관한 법률」 제2조에 따른 소상공인으로서 소기업ㆍ소상공인확인서를 소지한 자" 가 있고,
"최근 3년 이내 공공기관이 발주한 유사용역 1억원 이상 실적" 을 요구하며,
"본점 소재지가 경기도 또는 서울특별시인 업체" 로 제한한 경우
→ 기업규모제한.구분 = "소기업소상공인"
→ 실적제한.해당 = true, 요구실적금액 = 100000000, 발주처_특정 = true
→ 지역제한.해당 = true, 제한광역목록 = ["경기도","서울특별시"], 기초단위포함 = false

[예시 2] 제출서류 목록에 "중소기업확인서 1부", "직접생산확인증명서 1부" 만 적혀 있고
참가자격 항목에는 기업규모·직접생산 요건이 없는 경우
→ 기업규모제한.구분 = "없음", 직접생산확인요구.해당 = false
  (제출서류 목록은 참가자격이 아니다)

출력은 JSON 하나만. 설명을 덧붙이지 않는다."""


def format_meta(meta: Dict[str, Any]) -> str:
    out = []
    for k in META_FIELDS:
        if k in meta:
            v = meta[k]
            out.append(f"- {k}: {'미기재' if v is None else v}")
    return "\n".join(out)


RELEVANT = ("자격", "제한", "실적", "소재지", "본점", "영업소", "중소기업", "소기업",
            "소상공인", "확인서", "직접생산", "업종", "면허", "등록증", "공동수급",
            "확약서", "참여", "업체이어야", "이어야", "만 ", "한함", "한하여", "제출 마감")
NOISE = ("문의", "☎", "전화", "홈페이지", "http", "담당자", "청렴", "신고", "안내드립니다")

# 발췌에 넣어 봐야 토큰만 먹는 문단 — 규격서의 일반 서술이 대부분이다.
DROP = ("용지규격", "글꼴", "폰트", "쪽수", "페이지 수", "목차", "편집 용지",
        "제본", "인쇄 규격", "파일 형식은", "붙임", "별첨", "서식 참조")


def _rank_qblocks(blocks, keep: int = 16):
    """참가자격 구간에서 실제 요건이 담긴 문단만 골라 문서 순서대로 돌려준다.
    (일정표·문의처 같은 잡음을 빼면 모델이 볼 토큰이 그만큼 요건에 쓰인다.)"""
    scored = []
    for b in blocks:
        t = b[2]
        if len(t) < 8:
            continue
        sc = sum(1 for k in RELEVANT if k in t) - 2 * sum(1 for k in NOISE if k in t)
        scored.append((sc, b))
    scored.sort(key=lambda x: -x[0])
    top = [b for sc, b in scored[:keep] if sc > 0]
    top.sort(key=lambda b: b[0])
    return top


def _doc_of(doc_spans, off: int) -> str:
    for s, e, t in doc_spans:
        if s <= off < e:
            return t
    return ""


# 태그별 (이름, 우선순위). 숫자가 작을수록 예산을 먼저 가져간다.
# 참가자격은 양이 많아 뒤로 두고, 항목 수가 적어 놓치면 바로 0점이 되는
# 신호(확약서·설명회·공동수급·특정기관)를 앞에 둔다.
TAG_PRIORITY = {
    "특정기관": 0, "확약서": 1, "설명회": 1, "공동수급": 1, "경쟁제품": 1,
    "기업규모": 2, "지역": 2, "실적": 2, "직접생산": 2, "모델명": 3,
    "참가자격": 4, "규격": 5,
}


def _collect(sig):
    """블록마다 해당하는 태그를 **모두** 모은다.

    예전에는 먼저 붙은 태그가 블록을 선점해서, 같은 문단이 '기업규모'나
    '지역'인데도 전부 [참가자격] 하나로만 나갔다 (dev 기준 기업규모 337건,
    지역 81건이 고유 태그를 잃었다). 모델은 어느 문단이 어떤 사실용인지
    모르는 채로 읽어야 했다. 이제는 [참가자격·기업규모] 처럼 겹쳐 붙인다.
    """
    tags = {}      # offset -> set(tag)
    block = {}     # offset -> (start, end, text)

    def add(tag, blocks, limit):
        for b in (blocks or [])[:limit]:
            if not b:
                continue
            tags.setdefault(b[0], set()).add(tag)
            block[b[0]] = b

    add("참가자격", _rank_qblocks(sig["qblocks"], 26), 26)
    add("실적", sig["perf"]["blocks"], 8)
    add("지역", sig["region"]["blocks"], 8)
    add("기업규모", sig["sme"]["small_blocks"] + sig["sme"]["sme_blocks"], 10)
    add("직접생산", sig["direct_prod"]["blocks"], 4)
    add("경쟁제품", sig["competitive"]["blocks"], 4)
    add("모델명", sig["model"]["blocks"], 8)
    add("규격", sig.get("spec_brand_blocks") or [], 8)
    add("규격", sig.get("spec_blocks") or [], 12)
    add("확약서", sig["pledge"]["blocks"], 5)
    add("설명회", sig["briefing"]["blocks"], 5)
    add("공동수급", sig["joint"]["blocks"], 5)
    add("특정기관", sig.get("org_only_blocks") or [], 5)
    return tags, block


def _tagged(sig, budget_chars: int) -> str:
    """규칙이 찾아 둔 후보 블록을 태그·절 제목·출처와 함께 모은다.

    한 문단에 붙는 정보가 세 가지다.
      · 태그   — 어떤 사실을 뽑는 데 쓰라는 힌트 (여러 개 가능)
      · 절 제목 — '참가자격' 아래인지 '제출서류'·'입찰무효' 아래인지.
                 제출서류 목록의 서류 이름은 참가자격 요건이 아니므로
                 이 구분이 없으면 모델이 구조적으로 오탐을 낸다.
      · 출처   — 공고문인지 과업지시서·규격서인지
    """
    tags, block = _collect(sig)
    headings = sig.get("headings") or []
    dspans = sig.get("doc_spans") or []

    items = []
    seen_text = set()
    for off, tagset in tags.items():
        b = block[off]
        text = b[2]
        if len(text) < 8 or any(k in text for k in DROP):
            continue
        key = "".join(text.split())[:80]
        if key in seen_text:          # 공고문과 첨부에 같은 문단이 겹쳐 실리는 경우
            continue
        seen_text.add(key)
        prio = min(TAG_PRIORITY.get(t, 9) for t in tagset)
        label = "·".join(sorted(tagset, key=lambda t: TAG_PRIORITY.get(t, 9)))
        sec = T.section_at(headings, off)
        doc = _doc_of(dspans, off)
        where = " / ".join(x for x in (doc, sec) if x)
        head = f"[{label}]" + (f"<{where}>" if where else "")
        items.append((prio, off, f"{head} {text}"))

    # 예산은 우선순위 높은 태그부터 가져가고, 출력은 문서 순서를 지킨다.
    items.sort(key=lambda x: (x[0], x[1]))
    used, chosen = 0, []
    for prio, off, piece in items:
        if used + len(piece) + 1 > budget_chars:
            continue
        chosen.append((off, piece))
        used += len(piece) + 1
    chosen.sort()
    return chr(10).join(p for _, p in chosen)


def _price_hint(sig) -> str:
    h = sig.get("doc_price_hint") or (None, None, "meta")
    est, bud = h[0], h[1]
    if not est and not bud:
        return "개요에서 못 찾음"
    return f"추정가격 {est}, 사업예산(부가세포함) {bud}"


def rule_draft(sig: Dict[str, Any]) -> str:
    """규칙 계층이 찾아 둔 값을 '초안'으로 보여 준다. 모델은 이걸 고치기만 하면 된다."""
    sme = {"small": "소기업소상공인", "sme": "중소기업자", None: "없음"}[sig["sme"]["class"]]
    reg = sig["region"]
    cp = sig["competitive"]
    cp_txt = ("세부품명번호가 중기부 고시 경쟁제품 목록에 있음" if cp.get("code_hit")
              else "세부품명번호가 고시 목록에 없음" if cp.get("codes")
              else "세부품명번호 미확인")
    return (
        f"- 공고문 개요 금액(초안): {_price_hint(sig)}\n"
        f"- 나라장터 등록 금액: 추정가격 {sig.get('est_meta')}"
        f", 배정예산 {sig.get('budget_meta')}\n"
        f"- 기업규모제한(초안): {sme}\n"
        f"- 실적제한(초안): {'있음' if sig['perf']['present'] else '없음'}"
        f", 요구금액 {sig['perf']['amount']}\n"
        f"- 지역제한(초안): {'있음' if reg['present'] else '없음'}"
        f", 광역 {reg.get('wide')}, 기초단위 {bool(reg.get('has_basic'))}\n"
        f"- 중기간경쟁제품(참고): {cp_txt}\n"
        f"- 직접생산확인 문구(초안): {'있음' if sig['direct_prod']['present'] else '없음'}\n"
        f"- 공동수급 최소지분율(초안): {sig['joint']['min_share']}\n"
        f"- 설명회 문구(초안): {'있음' if sig['briefing']['blocks'] else '없음'}"
        f", 일자 {sig['briefing']['date']}\n"
        f"- 등록값 대조 의심지점: {_mismatch_hint(sig)}\n")


def _mismatch_hint(sig) -> str:
    mm = sig.get("mismatch") or {}
    parts = []
    for h in mm.get("hits", []):
        parts.append(f"{h[0]}(공고 {h[1]} / 등록 {h[2]})")
    for c in mm.get("cands", []):
        parts.append(f"{c[0]}(공고 {c[1]} / 등록 {c[2]})")
    return ", ".join(parts) if parts else "없음"


def build_user(rec: Dict[str, Any], sig: Dict[str, Any], budget_chars: int = 12000) -> str:
    # 공고문 머리에는 개요표(사업명·금액·계약방법·지역·업종)가 들어 있다.
    # v24(등록값 대조)와 '공고문금액'은 전적으로 이 구간을 읽어야 풀리는데
    # 1,200자에서 끊으면 개요표 뒷단이 잘리는 공고가 있다. 예산이 남으므로
    # (dev 중앙값 5.4천/1.2만자) 2,400자까지 넓힌다.
    head = ""
    for d in rec["docs"]:
        if d["type"] == "공고문":
            head = T.normalize(d["text"])[:2400]
            break
    body = _tagged(sig, budget_chars - len(head) - 900)
    dropped = rec.get("dropped_doc_counts") or {}
    note = ("\n[미수록 문서] " + ", ".join(f"{k} {v}건" for k, v in dropped.items())
            + " — 이 문서들은 제공되지 않았다. 여기에만 있었을 내용을 '없다'고 단정하지 말 것."
            ) if dropped else ""
    return (
        f"[나라장터 등록정보]\n{format_meta(rec.get('meta') or {})}\n\n"
        f"[공고문 머리]\n{head}\n\n"
        f"[규칙 초안]  정규식이 만든 임시값이라 자주 틀린다. 문서를 읽고 반드시 검증·수정하라.\n"
        f"{rule_draft(sig)}\n"
        f"[발췌 문단]  형식: [태그]<문서 / 절 제목> 본문\n"
        f"  · 태그는 그 문단이 어느 사실과 관련될 수 있다는 검색 힌트일 뿐 정답이 아니다.\n"
        f"  · <절 제목>이 '제출서류·구비서류'면 그 안의 서류 이름은 참가자격 요건이 아니다.\n"
        f"    '입찰무효'면 무효 사유 설명이지 참가자격 요건이 아니다.\n"
        f"  · 근거문구는 태그·꺾쇠 부분을 빼고 본문만 그대로 옮긴다.\n{body}{note}\n"
    )


def build_messages(rec, sig, budget_chars=8000, use_system: bool = False):
    """기본은 user 턴 하나로 합친다.

    Gemma 계열 chat template 은 system 롤을 지원하지 않아
    apply_chat_template 단계에서 예외가 나는 경우가 있고, 그러면 전 건이
    빈 출력 → 규칙 폴백으로 떨어진다. 합쳐 보내면 어떤 템플릿에서도 안전하다.
    """
    user = build_user(rec, sig, budget_chars)
    if use_system:
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    return [{"role": "user", "content": SYSTEM + "\n\n" + user}]
