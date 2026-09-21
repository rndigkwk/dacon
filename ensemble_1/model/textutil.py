# -*- coding: utf-8 -*-
"""공고문 텍스트를 '불릿 블록' 단위로 자르고 금액·날짜·지역을 뽑아내는 유틸.

모든 블록은 (start, end, text) 로 원문 오프셋을 들고 다닌다.
제출 규약상 근거문구는 반드시 원문 부분문자열이어야 하므로,
문자열을 새로 만들지 않고 원문 슬라이스만 넘긴다.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Tuple

from . import lexicon as LX

# ---------------------------------------------------------------- 블록 분할
# 불릿/번호 머리표 — 이 표시로 시작하는 줄에서 새 블록을 연다.
BULLET = re.compile(
    r"^\s*(?:"
    r"[가-힣]\s*[.)]"                     # 가. 나) 다.
    r"|[①-⑳]|[❶-❿]|[㉠-㉯]"
    r"|\(?\d{1,2}\s*[.)]"                 # 1. 2) (3)
    r"|[-•·○●◦▪▫◇◆□■※∙*]"
    r"|[IVXivx]{1,4}\s*[.)]"
    r")\s*"
)
HEADING = re.compile(r"^\s*(?:\[[^\]]{1,40}\]|【[^】]{1,40}】|\d{1,2}\.\s*\S{1,30}\s*$)")


def normalize(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def split_blocks(text: str, max_len: int = 700) -> List[Tuple[int, int, str]]:
    """원문을 불릿 블록으로 나눈다. 반환값의 text 는 원문 슬라이스와 동일하다."""
    blocks: List[Tuple[int, int, str]] = []
    pos = 0
    cur_start: Optional[int] = None
    for line in text.split("\n"):
        ls, le = pos, pos + len(line)
        pos = le + 1  # '\n'
        if not line.strip():
            if cur_start is not None:
                blocks.append((cur_start, ls, text[cur_start:ls].rstrip()))
                cur_start = None
            continue
        starts_new = bool(BULLET.match(line)) or bool(HEADING.match(line))
        if cur_start is None:
            cur_start = ls
        elif starts_new or (ls - cur_start) > max_len:
            blocks.append((cur_start, ls, text[cur_start:ls].rstrip()))
            cur_start = ls
    if cur_start is not None:
        blocks.append((cur_start, len(text), text[cur_start:].rstrip()))
    out = []
    for s, e, t in blocks:
        t2 = t.rstrip()
        if t2.strip():
            out.append((s, s + len(t2), t2))
    return out


# ---------------------------------------------------------------- 금액 파서
_UNIT = {"억": 10 ** 8, "천만": 10 ** 7, "백만": 10 ** 6, "십만": 10 ** 5,
         "만": 10 ** 4, "천": 10 ** 3}
_NUM = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(억|천만|백만|십만|만|천)?")
_AMOUNT = re.compile(
    r"(?:금\s*)?((?:\d[\d,]*(?:\.\d+)?\s*(?:억|천만|백만|십만|만|천)?\s*){1,4})\s*원"
)


def _eval_amount(chunk: str) -> Optional[int]:
    total, seen = 0, False
    for m in _NUM.finditer(chunk):
        raw, unit = m.group(1).replace(",", ""), m.group(2)
        try:
            val = float(raw)
        except ValueError:
            continue
        total += val * _UNIT.get(unit or "", 1)
        seen = True
    return int(total) if seen and total > 0 else None


def find_amounts(text: str) -> List[Tuple[int, int, int]]:
    """(start, end, 금액) 목록. '2억 3천만원', '455,000,000원' 모두 처리한다."""
    out = []
    for m in _AMOUNT.finditer(text):
        val = _eval_amount(m.group(1))
        if val:
            out.append((m.start(), m.end(), val))
    return out


def max_amount(text: str) -> Optional[int]:
    a = [v for _, _, v in find_amounts(text)]
    return max(a) if a else None


# ---------------------------------------------------------------- 날짜 파서
_DATE_PATS = [
    re.compile(r"(20\d{2})\s*[.\-/년]\s*(\d{1,2})\s*[.\-/월]\s*(\d{1,2})"),
    re.compile(r"(20\d{2})(\d{2})(\d{2})"),
]


def find_dates(text: str) -> List[Tuple[int, Tuple[int, int, int]]]:
    out = []
    for pat in _DATE_PATS:
        for m in pat.finditer(text):
            try:
                y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            except ValueError:
                continue
            if 1 <= mo <= 12 and 1 <= d <= 31:
                out.append((m.start(), (y, mo, d)))
    out.sort()
    return out


def to_ordinal(ymd: Tuple[int, int, int]) -> int:
    import datetime
    try:
        return datetime.date(*ymd).toordinal()
    except ValueError:
        return 0


def parse_meta_date(s) -> Optional[Tuple[int, int, int]]:
    if not s:
        return None
    t = re.sub(r"\D", "", str(s))
    if len(t) >= 8:
        y, m, d = int(t[:4]), int(t[4:6]), int(t[6:8])
        if 2000 <= y <= 2100 and 1 <= m <= 12 and 1 <= d <= 31:
            return (y, m, d)
    return None


# ---------------------------------------------------------------- 지역 파서
_ANON_REGION = re.compile(r"\[(?:등록)?지역:(r\d+)\|단위=(기초|광역)\|광역=([^\]]+)\]")


def canon_region(name: str) -> str:
    name = name.strip()
    return LX.REGION_CANON.get(name, name)


def extract_regions(text: str) -> Dict[str, object]:
    """지역제한 문구에서 광역 집합과 기초단위 사용 여부를 뽑는다."""
    wide, has_basic = set(), False
    for m in _ANON_REGION.finditer(text):
        if m.group(2) == "기초":
            has_basic = True
        wide.add(canon_region(m.group(3)))
    for name in LX.WIDE_REGIONS:
        if name in text:
            wide.add(canon_region(name))
    return {"wide": sorted(wide), "n_wide": len(wide), "has_basic": has_basic}


# ---------------------------------------------------------------- 검색 도우미
def contains_any(text: str, kws) -> Optional[str]:
    for k in kws:
        if k in text:
            return k
    return None


def blocks_matching(blocks, all_of=(), any_of=(), none_of=()):
    out = []
    for b in blocks:
        t = b[2]
        if all_of and not all(k in t for k in all_of):
            continue
        if any_of and not any(k in t for k in any_of):
            continue
        if none_of and any(k in t for k in none_of):
            continue
        out.append(b)
    return out


# ---------------------------------------------------------------- 절 제목 인덱스
# 발췌 문단만 떼어 모델에 보내면 그 문단이 '참가자격' 아래 있었는지
# '제출서류'·'입찰무효' 아래 있었는지가 사라진다. 이 둘은 판정이 정반대라
# (제출서류 목록에 이름만 오른 서류는 요건이 아니다) 블록마다 소속 절을 달아 준다.
SECTION_WORDS = (
    "참가자격", "참가 자격", "자격요건", "제출서류", "제출 서류", "구비서류",
    "입찰무효", "입찰 무효", "무효", "유의사항", "낙찰자", "과업내용", "과업 내용",
    "현장설명", "설명회", "공동수급", "공동계약", "계약체결", "계약 체결",
    "입찰개요", "입찰방법", "평가", "제안서", "규격", "사양", "지명원", "청렴",
)
_SEC_NUM = re.compile(r"^\s*(?:\d{1,2}|[IVXivx]{1,4})\s*[.)]\s*(\S[^\n]{0,32})$")
_SEC_BRACKET = re.compile(r"^\s*[【\[]\s*(\S[^\n】\]]{0,32})\s*[】\]]\s*$")
_SEC_BARE = re.compile(r"^\s*(?:[◆◇■□▣▶※]\s*)?(\S.{0,15})\s*$")
_SENT_TAIL = re.compile(r"(합니다|하여야|해야|있음|없음|한다|된다|우선함|참조|제출|임|함)$")


def _is_section_title(line: str) -> Optional[str]:
    """그 줄이 '절 제목'인지. 본문 서술문은 제목으로 치지 않는다."""
    s = line.strip()
    if not s or len(s) > 40:
        return None
    for rx in (_SEC_NUM, _SEC_BRACKET):
        m = rx.match(line)
        if m and any(w in m.group(1) for w in SECTION_WORDS):
            t = m.group(1).strip(" :·.")
            # "3. 본 입찰은 청렴계약제가 적용됩니다." 같은 서술문은 제목이 아니다.
            if _SENT_TAIL.search(t) and not t.endswith(("서류", "자격", "내용")):
                continue
            return t
    # 번호·괄호가 없는 맨제목은 아주 짧을 때만 인정한다.
    # 불릿으로 시작하거나 콜론이 있으면 본문 항목이지 제목이 아니다.
    if BULLET.match(line) or ":" in s:
        return None
    m = _SEC_BARE.match(line)
    if not m:
        return None
    t = m.group(1).strip(" :·.")
    if not any(w in t for w in SECTION_WORDS):
        return None
    if _SENT_TAIL.search(t) and not t.endswith(("서류", "제출서류", "구비서류")):
        return None
    return t


def heading_index(text: str) -> List[Tuple[int, str]]:
    """(offset, 절 제목) 오름차순 목록."""
    out: List[Tuple[int, str]] = []
    pos = 0
    for line in text.split("\n"):
        title = _is_section_title(line)
        if title:
            out.append((pos, title))
        pos += len(line) + 1
    return out


def section_at(headings: List[Tuple[int, str]], offset: int) -> str:
    """해당 오프셋 바로 앞의 절 제목. 없으면 빈 문자열."""
    lo, hi, best = 0, len(headings) - 1, ""
    while lo <= hi:
        mid = (lo + hi) // 2
        if headings[mid][0] <= offset:
            best = headings[mid][1]
            lo = mid + 1
        else:
            hi = mid - 1
    return best
