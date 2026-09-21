#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""나라장터 자체입찰 공고 법령 위반사항 모니터링 — 항목그룹 · 메타게이트 · 임계값 판정.

평가 서버는 이 파일을 `python script.py`로 그대로 실행합니다.
  입력   ./data/test.jsonl.gz (+ 항목표.json · 정답스키마_디코딩.json · 법령패키지/)
  정적   ./model/ (조문발췌 · 경쟁제품코드 · 게이트 · 임계값)  ← 제출 ZIP에 포함
  출력   ./output/submission.csv  (열 = id, v1..v24, e1..e24)
  경로   PPS_DATA_DIR · PPS_OUTPUT_DIR · PPS_MODEL_DIR · PPS_ASSET_DIR 환경변수 우선.
         환경변수가 없으면 ./data → <script.py 옆>/data → <한 단계 위>/data 순으로 찾습니다.
         평가 서버(평탄 배치)와 로컬 레포(baseline/ 하위) 양쪽에서 같은 코드가 돕니다.
         실행 첫 줄 로그에 실제로 고른 경로가 찍히니 확인하십시오.

베이스라인과 달라진 점
  1. 24항목 1콜 → 7개 항목그룹 콜. 그룹마다 관련 조문과 관련 문서 구간만 본다.
  2. 문서 앞 4000자 절단 → 앵커 기반 구간 선택. dev 근거문구 수록률 87.0% → 100%
     (tools/coverage.py). 구간은 원문 슬라이스 그대로 실어 인용한 문구가 원문 부분문자열로 남게 한다.
  3. 메타 게이트를 프롬프트 이전에 적용. 고시금액(2.3억)·낙찰방법·업무구분으로 적용 대상이
     아닌 항목은 모델에 묻지 않고 0으로 확정한다. dev에서 재현율 손실 0으로 검증했다.
  4. 0/1 강제 디코딩 → 판정 토큰의 logprob에서 P(위반)을 얻어 항목별 임계값으로 자른다.
     항목당 양성률이 2.5~4.0%라 임계값 0.5는 과도하게 보수적이다.

규칙 준수
  · 공고 1건마다 고정 LLM을 최소 1회 호출한다(게이트로 전 항목이 닫혀도 최소 1콜 보장).
  · 샘플 간 예측 의존 없음 — 임계값은 model/임계값.json의 상수이며 런타임에 바뀌지 않는다.
  · 외부 네트워크·외부 LLM 호출 없음. 판정은 배포된 법령패키지 스냅샷만 근거로 한다.

로컬 실행
  python script.py --mock            # 모델 없이 입력·게이트·출력 흐름 확인
  python script.py --limit 10        # 앞 10건 실행
  python script.py --input ../dev.jsonl.gz --output-dir ../output/dev
"""
from __future__ import annotations

# ===== 1. 상수·경로 =====
import argparse
import csv
import gzip
import io
import json
import math
import os
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))


def _pick_dir(env: str, candidates: Sequence[str], marker: Optional[str] = None) -> str:
    """디렉토리를 환경변수 → 후보 순서로 정한다. 없으면 첫 후보(원본 베이스라인과 같은 값).

    평가 서버와 로컬 레포의 배치가 다르기 때문에 한쪽으로 못박지 않는다.
      평가 서버  `python script.py` · CWD에 data/ output/ 이 있다 (배포 원본의 "./data")
      로컬 레포  baseline/script.py · data/ output/ 은 한 단계 위에 있다
    `marker`를 주면 그 파일이 실제로 있는 후보만 고른다 — 빈 디렉토리를 집는 것을 막는다.
    """
    v = os.environ.get(env)
    if v:
        return v
    for c in candidates:
        if os.path.isdir(c) and (marker is None or os.path.exists(os.path.join(c, marker))):
            return c
    return candidates[0]


DATA_DIR = _pick_dir("PPS_DATA_DIR",
                     ["./data", os.path.join(HERE, "data"), os.path.join(HERE, "..", "data")],
                     marker="항목표.json")
# output/ 은 어느 배치에서도 data/ 의 형제다.
OUTPUT_DIR = os.environ.get("PPS_OUTPUT_DIR",
                            os.path.join(os.path.dirname(DATA_DIR.rstrip("/\\")) or ".", "output"))
ASSET_DIR = _pick_dir("PPS_ASSET_DIR",
                      [os.path.join(HERE, "model"), "./model", os.path.join(HERE, "..", "model")],
                      marker="게이트.json")
# 고정 모델. 평가 서버는 PPS_MODEL_DIR 로 주지만, 없을 때를 대비해 알려진 자리를 훑는다.
# 경로가 없으면 vLLM이 이 문자열을 HuggingFace repo id로 보고
#   OSError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'
# 라는 엉뚱한 에러를 내므로, 아래 VLLMRunner 에서 먼저 확인하고 사람 말로 알려 준다.
MODEL_NAME = "gemma-4-26B-A4B-it"
MODEL_DIR = _pick_dir("PPS_MODEL_DIR",
                      [f"/opt/models/{MODEL_NAME}",          # 평가 서버 (배포 원본 기본값)
                       f"/workspace/models/{MODEL_NAME}",    # 개발 머신
                       f"/models/{MODEL_NAME}"],
                      marker="config.json")

ITEMS = [f"v{i}" for i in range(1, 25)]
EVID = [f"e{i}" for i in range(1, 25)]
COLUMNS = ["id"] + ITEMS + EVID
ABSENCE = ["v10", "v11", "v16", "v18", "v20"]          # 부재탐지 항목: 근거 문구 빈칸

DOC_ORDER = ["공고문", "규격서", "과업지시서", "제안요청서", "예외공표서", "기타"]
META_FIELDS = [
    "적용계약법", "업무구분", "계약방법", "낙찰방법", "낙찰하한율",
    "배정예산금액", "입찰추정가격", "소관구분", "공동도급구성방식", "정보화사업여부",
    "세부품명번호목록", "제한지역코드목록", "지역제한여부", "면허업종제한목록", "업종제한여부",
    "조항호내용", "공고게시일자", "개찰예정일자", "긴급공고여부", "입찰방법", "조달방식",
]

SEED = 20260826
MAX_MODEL_LEN = 16384                   # 베이스라인 모델 컨텍스트 길이
PROMPT_BUDGET_MARGIN = 256              # 토큰 추정 오차 여유
EVIDENCE_MAX = 500                      # 근거 문구 셀 글자 수 상한(제출 규약)
EVIDENCE_GEN_MAX = 300                  # 생성 단계 상한. dev 근거 중앙값 76자 · 최대 237자
QUANT = "int8_per_channel_weight_only"  # 평가 서버 양자화 설정

GOSI_DEFAULT = 230_000_000              # 고시금액(물품·용역). model/게이트.json이 있으면 그 값을 쓴다
EOK_DEFAULT = 100_000_000

# 항목그룹 — 같은 조문·같은 문서 구간을 근거로 쓰는 항목끼리 묶는다.
GROUPS: Dict[str, List[str]] = {
    "제한경쟁": ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"],
    "직생경쟁제품": ["v9", "v10", "v11", "v12", "v13"],
    "판로지원": ["v14", "v15", "v16", "v17", "v18", "v19"],
    "SW": ["v20"],
    "공동도급": ["v21"],
    "현장설명회": ["v22", "v23"],
    "메타대조": ["v24"],
}
assert sorted(sum(GROUPS.values(), []), key=lambda v: int(v[1:])) == ITEMS

# 그룹별 문서 앵커 — 이 표현 주변이 판정 근거가 놓이는 자리다.
ANCHORS: Dict[str, str] = {
    "제한경쟁": r"참가\s?자격|입찰참가|참가대상|제한\s?사항|자격\s?요건|실적|시공실적|납품실적|"
                r"용역\s?수행|이행실적|지역\s?제한|소재지|본점|주된\s?영업소|지사|영업소|"
                r"면허|업종|등록\s?기준|제한경쟁|유자격자",
    "직생경쟁제품": r"직접\s?생산|직생|중소기업자\s?간|경쟁\s?제품|직접생산확인|규격|모델\s?명|모델명|"
                    r"제조\s?사|제조사|품명|제품명|사양|스펙|동등\s?이상|이와\s?동등",
    "판로지원": r"중소기업|소기업|소상공인|중견기업|판로\s?지원|중소기업자|확인서|공동\s?상표|"
                r"물품\s?공급|공급\s?확약|확약서|제조\s?증명",
    "SW": r"소프트웨어|S/?W|대기업|중견기업|정보화|감리|상호출자|사업금액|참여\s?제한|"
          r"소프트웨어사업자",
    "공동도급": r"공동\s?수급|공동\s?도급|공동\s?계약|분담\s?이행|공동\s?이행|구성원|출자\s?비율|"
                r"지분|대표사|주계약자",
    "현장설명회": r"현장\s?설명|설명회|현장\s?확인|입찰\s?참가\s?자격|공고\s?기간|입찰\s?마감|"
                  r"개찰|참석|필참|의무\s?참석",
    "메타대조": r"추정\s?가격|배정\s?예산|사업\s?예산|사업\s?금액|계약\s?방법|낙찰\s?자|낙찰\s?방법|"
                r"지역\s?제한|업종|면허|입찰\s?방법|긴급",
}

# 그룹별 조문 — 적용계약법에 따라 국가/지방 갈래를 고른다.
GROUP_LAWS: Dict[str, Dict[str, List[str]]] = {
    "제한경쟁": {"국가계약법": ["제한경쟁_국가"], "지방계약법": ["제한경쟁_지방", "소액수의_지방"]},
    "직생경쟁제품": {"*": ["직접생산", "판로지원"]},
    "판로지원": {"*": ["판로지원", "중소기업범위"]},
    "SW": {"*": ["소프트웨어"]},
    "공동도급": {"*": ["공동계약"]},
    "현장설명회": {"국가계약법": ["제한경쟁_국가"], "지방계약법": ["제한경쟁_지방"]},
    "메타대조": {"*": []},
}


def log(msg: str) -> None:
    print(f"[pps] {msg}", file=sys.stderr, flush=True)


# ===== 2. 데이터 로더 =====
def _open(path: str):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return io.open(path, "r", encoding="utf-8")


def validate_record(rec: Any) -> None:
    """레코드 1건의 최소 스키마 검사 (id · docs(공고문 1개 이상) · meta)"""
    if not isinstance(rec, dict):
        raise ValueError(f"레코드가 object가 아니다: {type(rec).__name__}")
    for k in ("id", "docs", "meta"):
        if k not in rec:
            raise ValueError(f"필수 키 없음: {k}")
    if not isinstance(rec["id"], str) or not rec["id"]:
        raise ValueError("id가 비어 있다")
    docs = rec["docs"]
    if not isinstance(docs, list) or not docs:
        raise ValueError(f"docs가 비어 있다 (id={rec['id']})")
    for d in docs:
        if not isinstance(d, dict) or not all(k in d for k in ("doc_id", "type", "text")):
            raise ValueError(f"docs 원소 형식 오류 (id={rec['id']})")
        if not isinstance(d["text"], str):
            raise ValueError(f"docs.text가 문자열이 아니다 (id={rec['id']})")
    if not any(d["type"] == "공고문" for d in docs):
        raise ValueError(f"공고문이 없다 (id={rec['id']})")
    if not isinstance(rec["meta"], dict):
        raise ValueError(f"meta가 object가 아니다 (id={rec['id']})")


def normalize(rec: Dict[str, Any]) -> Dict[str, Any]:
    """NFC 정규화 — 근거문구 부분문자열 대조가 정규화 차이로 어긋나지 않게 한다."""
    for d in rec.get("docs", []):
        d["text"] = unicodedata.normalize("NFC", d["text"])
        if isinstance(d.get("type"), str):
            d["type"] = unicodedata.normalize("NFC", d["type"])
    return rec


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
            yield normalize(rec)
            n += 1
            if limit and n >= limit:
                return


def full_text(rec: Dict[str, Any]) -> str:
    """근거문구 대조용 원문 (NFC). 프롬프트에 무엇이 실렸든 원문 전체와 대조한다."""
    return "\n".join(d["text"] for d in rec["docs"])


def sorted_docs(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    order = {t: i for i, t in enumerate(DOC_ORDER)}
    return sorted(rec["docs"], key=lambda d: (order.get(d["type"], len(DOC_ORDER)), d["doc_id"]))


# ===== 3. 정적 자산 (항목표 · 스키마 · model/) =====
def item_table(data_dir: str = DATA_DIR) -> Dict[str, Dict[str, Any]]:
    p = os.path.join(data_dir, "항목표.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} 가 없습니다 — data/ 를 그대로 둔 채 실행하세요.")
    return json.load(io.open(p, encoding="utf-8"))["항목"]


def _load_asset(asset_dir: str, name: str, default: Any) -> Any:
    """model/ 자산 로더. 없거나 깨져도 실행은 계속한다(성능만 떨어질 뿐 실격은 면한다)."""
    p = os.path.join(asset_dir, name)
    try:
        return json.load(io.open(p, encoding="utf-8"))
    except Exception as e:
        log(f"[주의] 정적 자산 {name} 로드 실패 → 기본값 사용: {type(e).__name__}: {e}")
        return default


class Assets:
    """model/ 아래 정적 자산 묶음."""

    def __init__(self, asset_dir: str = ASSET_DIR):
        gate = _load_asset(asset_dir, "게이트.json", {})
        self.gosi: int = int(gate.get("고시금액", GOSI_DEFAULT))
        self.eok: int = int(gate.get("일억", EOK_DEFAULT))

        thr = _load_asset(asset_dir, "임계값.json", {})
        base = float(thr.get("기본값", 0.25))
        tbl = thr.get("임계값", {})
        self.threshold: Dict[str, float] = {v: float(tbl.get(v, base)) for v in ITEMS}

        laws = _load_asset(asset_dir, "조문발췌.json", {})
        self.laws: Dict[str, str] = laws.get("조문", {})

        guide = _load_asset(asset_dir, "판정지침.json", {})
        self.guides: Dict[str, Dict[str, Any]] = guide.get("지침", {})
        self.evidence_rx: Dict[str, Any] = {}
        for v, g in self.guides.items():
            if g.get("근거정규식"):
                try:
                    self.evidence_rx[v] = re.compile(g["근거정규식"])
                except re.error as e:
                    log(f"[주의] {v} 근거정규식 컴파일 실패 → 미적용: {e}")

        prod = _load_asset(asset_dir, "경쟁제품코드.json", {})
        self.product_codes: Dict[str, Any] = prod.get("코드", {})
        self.product_names: Dict[str, str] = prod.get("품명역인덱스", {})

        log(f"자산: 고시금액 {self.gosi:,} · 조문 {len(self.laws)}종 · 경쟁제품 {len(self.product_codes)}코드 "
            f"· 판정지침 {len(self.guides)}항목 · 임계값 중앙 {sorted(self.threshold.values())[len(ITEMS) // 2]}")

        # 자산이 빠져도 실행은 되지만 성능이 조용히 떨어진다 — 그 조용함이 제일 위험하므로 크게 알린다.
        missing = [n for n, ok in (("조문발췌", self.laws), ("경쟁제품코드", self.product_codes),
                                   ("게이트", gate), ("임계값", tbl)) if not ok]
        if missing:
            log("=" * 70)
            log(f"[경고] model/ 자산 {', '.join(missing)} 이(가) 비었습니다 — {os.path.abspath(asset_dir)}")
            log("       실행은 계속되지만 조문 근거·경쟁제품 조인·튜닝된 임계값 없이 판정합니다.")
            log("       제출 ZIP에 model/ 을 넣었는지, PPS_ASSET_DIR 이 맞는지 확인하십시오.")
            log("=" * 70)


def decode_schema_for(items: Sequence[str]) -> Dict[str, Any]:
    """그룹에 속한 항목만 담는 구조화 출력 스키마.

    위반여부를 근거문구보다 앞에 두어 판정 토큰의 위치를 일정하게 만든다(logprob 추출용).
    """
    props = {}
    for v in items:
        props[v] = {
            "type": "object", "additionalProperties": False,
            "required": ["위반여부", "근거문구"],
            "properties": {
                "위반여부": {"type": "integer", "enum": [0, 1]},
                "근거문구": ({"type": "null"} if v in ABSENCE
                             else {"type": ["string", "null"], "maxLength": EVIDENCE_GEN_MAX}),
            },
        }
    return {"type": "object", "additionalProperties": False,
            "required": list(items), "properties": props}


# ===== 4. 메타 파생 사실 · 게이트 =====
def _money(m: Dict[str, Any], key: str) -> Optional[int]:
    v = m.get(key)
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return int(v)
    s = re.sub(r"[^\d]", "", str(v))
    return int(s) if s else None


def _codes(m: Dict[str, Any]) -> List[str]:
    """'과일류[5030990101], 코팅머신[2315150201]' → ['5030990101', '2315150201']"""
    s = m.get("세부품명번호목록")
    return re.findall(r"\d{8,}", str(s)) if s else []


def derive_facts(rec: Dict[str, Any], a: Assets) -> Dict[str, Any]:
    """메타에서 판정 전제를 계산한다. 값이 없으면 None으로 두고 게이트를 열어 둔다."""
    m = rec.get("meta", {})
    price = _money(m, "입찰추정가격")
    budget = _money(m, "배정예산금액")
    ref = price if price is not None else budget          # 추정가격이 없으면 예산으로 갈음
    law = m.get("적용계약법")
    codes = _codes(m)
    known = [c for c in codes if c in a.product_codes]

    band = None
    if ref is not None:
        band = "고시금액이상" if ref >= a.gosi else ("일억이상_고시금액미만" if ref >= a.eok else "일억미만")

    return {
        "추정가격": price, "배정예산": budget, "기준금액": ref, "금액밴드": band,
        "적용계약법": law if law in ("국가계약법", "지방계약법") else None,
        "협상": m.get("낙찰방법") == "협상에의한계약",
        "공동도급방식": m.get("공동도급구성방식"),
        "업무구분": m.get("업무구분"),
        "세부품명번호": codes,
        "경쟁제품코드": known,
        "경쟁제품여부": (bool(known) if codes else None),   # 품명 미등록이면 판단 보류
    }


def applicable(v: str, f: Dict[str, Any], a: Assets) -> bool:
    """항목 v가 이 공고에 적용될 수 있는가. 모르면 True(재현율 보호)."""
    band = f["금액밴드"]
    if band is not None:
        if v in ("v5", "v14"):
            return band == "고시금액이상"
        if v == "v2":
            return band != "고시금액이상"
        # v6·v7·v8은 금액으로 가두지 않는다. tools/audit_gates.py 결과, v8은 양성 6건 중 5건이
        # 고시금액 이상이고(중복제한은 금액과 무관한 항목이다), v6·v7도 지방계약법 건에서
        # 고시금액을 넘는 양성이 나온다(지자체별 기준액 차이). 금액 사실은 프롬프트의
        # [판정 전제]로 전달해 모델이 판단하게 둔다.
        if v in ("v15", "v16"):
            return band == "일억이상_고시금액미만"
        if v in ("v17", "v18"):
            return band == "일억미만"
    if v == "v22":
        return f["협상"]
    if v == "v23":
        return f["협상"] and (f["적용계약법"] != "국가계약법")
    if v == "v9":
        return f["업무구분"] is None or f["업무구분"] == "물품(내자)"
    return True


def gate_reasons(f: Dict[str, Any], a: Assets) -> Dict[str, str]:
    """로그·검증용. 닫힌 항목과 그 사유."""
    out = {}
    for v in ITEMS:
        if not applicable(v, f, a):
            if v in ("v22", "v23"):
                out[v] = "낙찰방법이 협상에의한계약이 아님" if not f["협상"] else "지방계약법이 아님"
            elif v == "v9":
                out[v] = f"업무구분 {f['업무구분']}"
            else:
                out[v] = f"금액밴드 {f['금액밴드']} (기준금액 {f['기준금액']:,})"
    return out


# ===== 5. 문서 구간 선택 (앵커 기반) =====
def _spans(text: str, max_len: int = 1200) -> List[Tuple[int, int]]:
    """문단 경계로 자른 (시작, 끝) 문자 구간. 구간 밖에는 문단 사이 공백만 남는다.

    문자열이 아니라 구간을 다루는 이유: 이어 붙일 때 원문 구분자를 그대로 살려야 모델이 인용한
    문구가 원문의 부분문자열로 남는다(근거문구 대조는 원문 전체와 한다).
    """
    rough: List[Tuple[int, int]] = []
    pos = 0
    for m in re.finditer(r"\n{2,}", text):
        if m.start() > pos:
            rough.append((pos, m.start()))
        pos = m.end()
    if pos < len(text):
        rough.append((pos, len(text)))

    fine: List[Tuple[int, int]] = []
    for s, e in rough:                                     # 너무 긴 문단은 줄 경계로 더 쪼갠다
        while e - s > max_len:
            cut = text.rfind("\n", s, s + 900)
            cut = cut + 1 if cut > s else s + 900
            fine.append((s, cut))
            s = cut
        if e > s:
            fine.append((s, e))
    return [(s, e) for s, e in fine if text[s:e].strip()]


def select_windows(rec: Dict[str, Any], anchor: str, budget: int) -> Tuple[str, Dict[str, int]]:
    """앵커가 걸린 구간을 문서 순서대로 담되, 예산 안에서 히트가 많은 쪽을 우선한다.

    공고문 머리(참가자격 요약이 대개 여기 있다)는 앵커와 무관하게 먼저 확보한다.
    앵커가 하나도 안 걸리면 앞에서부터 채운다. 이어진 구간은 원문 그대로 한 덩어리로 낸다.
    """
    pat = re.compile(anchor)
    docs = sorted_docs(rec)
    scored: List[Tuple[int, int, int, int, int]] = []      # (문서순번, 구간순번, 히트수, 시작, 끝)
    for di, d in enumerate(docs):
        for pi, (s, e) in enumerate(_spans(d["text"])):
            scored.append((di, pi, len(pat.findall(d["text"][s:e])), s, e))

    head_budget = min(budget // 4, 1800)
    chosen: Dict[Tuple[int, int], Tuple[int, int]] = {}
    used = 0
    for di, pi, _h, s, e in scored:                        # 1) 첫 문서(공고문) 머리
        if di != 0 or used + (e - s) > head_budget:
            continue
        chosen[(di, pi)] = (s, e)
        used += e - s

    for di, pi, h, s, e in sorted(scored, key=lambda x: (-x[2], x[0], x[1])):        # 2) 히트 많은 순
        if h == 0 or (di, pi) in chosen or used + (e - s) > budget:
            continue
        chosen[(di, pi)] = (s, e)
        used += e - s

    for di, pi, _h, s, e in scored:                        # 3) 남으면 앞에서부터 채운다
        if (di, pi) in chosen or used + (e - s) > budget:
            continue
        chosen[(di, pi)] = (s, e)
        used += e - s

    parts, last_di = [], None
    run: Optional[Tuple[int, int]] = None                  # 이어진 구간은 원문 슬라이스 하나로 묶는다
    for (di, pi) in sorted(chosen):
        s, e = chosen[(di, pi)]
        if di != last_di:
            if run is not None:
                parts.append(docs[last_di]["text"][run[0]:run[1]])
            d = docs[di]
            parts.append(f"\n[{d['type']}:{d['doc_id']}]")
            run, last_di = (s, e), di
            continue
        if (di, pi - 1) in chosen and run is not None:
            run = (run[0], e)                              # 원문 구분자까지 포함해 확장
        else:
            parts.append(docs[last_di]["text"][run[0]:run[1]])
            parts.append("…")
            run = (s, e)
    if run is not None and last_di is not None:
        parts.append(docs[last_di]["text"][run[0]:run[1]])

    total = sum(len(d["text"]) for d in rec["docs"])
    text = "\n".join(parts).strip()
    dropped = Counter(rec.get("dropped_doc_counts") or {})
    if dropped:
        text += "\n\n[미수록 문서] " + ", ".join(f"{t} {n}건" for t, n in sorted(dropped.items()))
    if used < total:
        text += f"\n\n[안내] 원문 {total:,}자 중 관련 구간 {used:,}자만 수록했다. " \
                f"수록되지 않은 구간에 근거가 있을 수 있으니 단정하지 말 것."
    return text, {"수록": used, "원문": total}


def format_meta(rec: Dict[str, Any]) -> str:
    m = rec.get("meta", {})
    lines = []
    for k in META_FIELDS:
        if k in m:
            v = m[k]
            lines.append(f"- {k}: {'미기재' if v is None else v}")
    return "\n".join(lines)


def format_facts(f: Dict[str, Any], a: Assets) -> str:
    """모델이 다시 계산하지 않아도 되도록 판정 전제를 문장으로 못박는다."""
    L = [f"- 고시금액(물품·용역) = {a.gosi:,}원, 기준일 2026-01-08"]
    if f["기준금액"] is not None:
        rel = "이상" if f["기준금액"] >= a.gosi else "미만"
        L.append(f"- 이 공고의 기준금액 = {f['기준금액']:,}원 → 고시금액 {rel}")
        L.append(f"- 1억원 기준 → {'1억원 이상' if f['기준금액'] >= a.eok else '1억원 미만'}")
    else:
        L.append("- 이 공고는 추정가격·배정예산이 모두 미기재다. 금액 기준 항목은 문서에서 금액을 찾아 판단하라.")
    if f["적용계약법"]:
        L.append(f"- 적용계약법 = {f['적용계약법']}")
    L.append(f"- 낙찰방법이 협상에의한계약인가 = {'예' if f['협상'] else '아니오'}")
    if not f["세부품명번호"]:
        L.append("- 세부품명번호가 등록되지 않았다. 이것만으로 경쟁제품 여부를 단정할 수 없으니, "
                 "공고문·규격서 본문의 '직접생산확인'·'경쟁제품' 언급으로 판단하라.")
    if f["세부품명번호"]:
        if f["경쟁제품여부"]:
            names = ", ".join(a.product_codes[c].get("세부품명", c) for c in f["경쟁제품코드"])
            L.append(f"- 세부품명 {f['세부품명번호']} → 중소기업자간 경쟁제품에 **해당**({names})")
        else:
            L.append(f"- 세부품명 {f['세부품명번호']} → 중기부고시 경쟁제품 목록에 **없음**(경쟁제품이 아니다)")
    floor = 5 if (f["적용계약법"] == "지방계약법" or f["공동도급방식"] == "분담이행") else 10
    L.append(f"- 공동수급체 구성원별 최소지분율의 법정 하한 = {floor}% "
             f"(적용계약법 {f['적용계약법'] or '미상'} · 구성방식 {f['공동도급방식'] or '미기재'}). "
             f"공고문에 적힌 최소지분율이 {floor}% 미만이면 v21 위반이다.")
    L.append("- 메타의 지역제한여부·업종제한여부는 실제와 다를 수 있다. 문서 본문을 우선하라.")
    return "\n".join(L)


# ===== 6. 프롬프트 =====
SYSTEM_HEAD = """당신은 공공 입찰공고가 국가·지방계약법령을 위반했는지 점검하는 심사관이다.
아래 근거 법령과 공고 자료를 읽고, 배정된 점검 항목 각각에 대해 위반 여부를 판정한다.

판정 원칙
1. 위반이라고 볼 근거가 자료 안에 있어야 1이다. 짐작이나 일반론으로 1을 주지 않는다.
2. 근거 문구는 주어진 자료에 **그대로 있는 문장**을 옮긴다. 요약·수정·재작성하면 인정되지 않는다.
3. '부재탐지' 표시가 붙은 항목은 **있어야 할 내용이 없는 것**이 위반이다. 인용할 원문이 없으므로
   근거문구는 null로 둔다.
4. 자료는 원문의 일부만 수록됐을 수 있다. 수록되지 않은 구간을 근거로 위반을 단정하지 않는다."""

SYSTEM_TAIL = """
출력은 JSON 하나로만 낸다. 설명이나 머리말을 붙이지 않는다."""


def _guide_block(v: str, g: Dict[str, Any]) -> str:
    """항목 판정지침을 프롬프트 문단으로. 항목명이 축약어라 이게 없으면 모델이 뜻을 오해한다."""
    L = []
    if g.get("무엇"):
        L.append(f"    · 위반 정의: {g['무엇']}")
    for k in g.get("기준", []):
        L.append(f"      - {k}")
    if g.get("위반아님"):
        L.append("    · 아래는 위반이 아니다 (흔한 오인):")
        L += [f"      - {k}" for k in g["위반아님"]]
    if g.get("필수"):
        L.append(f"    · {g['필수']}")
    return "\n".join(L)


def build_group_system(group: str, items: Sequence[str], tbl: Dict[str, Dict[str, Any]],
                       law_text: str, guides: Optional[Dict[str, Dict[str, Any]]] = None) -> str:
    guides = guides or {}
    lines = []
    for v in items:
        it = tbl[v]
        tag = "  [부재탐지 — 근거문구 null]" if it["부재탐지"] else ""
        note = f" · 참고: {it['비고']}" if it.get("비고") else ""
        lines.append(f"- {v}: {it['항목명']}{note}{tag}")
        if v in guides:
            blk = _guide_block(v, guides[v])     # 근거정규식만 있는 항목은 빈 문단 → 넣지 않는다
            if blk.strip():
                lines.append(blk)
    parts = [SYSTEM_HEAD, f"\n[점검 항목 — {group}]\n" + "\n".join(lines)]
    if law_text:
        parts.append(f"\n[근거 법령]\n{law_text}")
    parts.append(SYSTEM_TAIL)
    return "\n".join(parts)


def build_group_user(rec: Dict[str, Any], f: Dict[str, Any], a: Assets,
                     group: str, doc_budget: int) -> Tuple[str, Dict[str, int]]:
    ctx, stat = select_windows(rec, ANCHORS[group], doc_budget)
    body = (
        f"[공고 ID] {rec['id']}\n\n"
        f"[판정 전제 — 이미 확인된 사실]\n{format_facts(f, a)}\n\n"
        f"[나라장터 등록 정보]\n{format_meta(rec)}\n\n"
        f"[공고 자료 — {group} 관련 구간]\n{ctx}\n"
    )
    return body, stat


def group_law_text(group: str, f: Dict[str, Any], a: Assets, budget: int) -> str:
    spec = GROUP_LAWS.get(group, {})
    keys = spec.get(f["적용계약법"] or "", spec.get("*"))
    if keys is None:                                       # 계약법 미상이면 국가 갈래를 기본으로
        keys = spec.get("국가계약법", [])
    chunks, used = [], 0
    for k in keys:
        t = a.laws.get(k, "")
        if not t:
            continue
        room = budget - used
        if room <= 500:
            break
        if len(t) > room:
            t = t[:room] + "\n…(이하 생략)"
        chunks.append(f"《{k}》\n{t}")
        used += len(t)
    return "\n\n".join(chunks)


def build_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    return [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}]


# ===== 7. 모델 러너 =====
class VLLMRunner:
    """평가 서버의 고정 모델을 vLLM offline API로 실행한다."""

    def __init__(self, model_dir: str = MODEL_DIR, quant: Optional[str] = QUANT,
                 seed: int = SEED, gpu_mem: float = 0.92, tp: int = 1,
                 max_model_len: int = MAX_MODEL_LEN):
        t0 = time.time()
        if os.sep in model_dir and not os.path.isdir(model_dir):
            # 경로처럼 생겼는데 없는 경우. 그냥 두면 vLLM이 HF repo id로 오해해
            # "Repo id must be in the form ..." 이라는 알아보기 힘든 에러를 낸다.
            raise FileNotFoundError(
                f"모델 경로가 없습니다: {model_dir}\n"
                f"  PPS_MODEL_DIR 환경변수를 쓰거나 --model-dir 로 직접 지정하십시오.\n"
                f"  예) PPS_MODEL_DIR=/workspace/models/{MODEL_NAME} python script.py")
        import vllm                                        # --mock 실행 시 vllm이 없어도 되도록 지연 import
        from vllm import LLM

        log(f"vllm {vllm.__version__} · 모델 {model_dir} · quant={quant} · max_model_len={max_model_len}")
        kw = dict(model=model_dir, tokenizer=model_dir, max_model_len=max_model_len,
                  gpu_memory_utilization=gpu_mem, seed=seed, tensor_parallel_size=tp,
                  dtype="auto", enable_prefix_caching=True)
        if quant:
            kw["quantization"] = quant
        self.llm = LLM(**kw)
        self.tok = self.llm.get_tokenizer()
        self.max_model_len = max_model_len
        self.load_seconds = time.time() - t0

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        try:
            ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
            if hasattr(ids, "keys") and "input_ids" in ids:
                ids = ids["input_ids"]
            return len(ids)
        except Exception:
            return len(self.tok.encode("\n".join(m["content"] for m in messages)))

    def _params(self, schema: Dict[str, Any], max_tokens: int):
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        return SamplingParams(
            temperature=0.0, max_tokens=max_tokens, seed=SEED, logprobs=20,
            structured_outputs=StructuredOutputsParams(json=schema, disable_any_whitespace=True),
        )

    def chat(self, batch: List[List[Dict[str, str]]], schema: Dict[str, Any],
             max_tokens: int) -> List[Tuple[str, Any, Any]]:
        outs = self.llm.chat(batch, sampling_params=self._params(schema, max_tokens), use_tqdm=False)
        res = []
        for o in outs:
            if not o.outputs:
                res.append(("", None, None))
                continue
            c = o.outputs[0]
            res.append((c.text, getattr(c, "token_ids", None), getattr(c, "logprobs", None)))
        return res


class MockRunner:
    """모델 없이 게이트·프롬프트·출력 형식을 확인한다. 판정은 전부 0."""
    load_seconds = 0.0
    max_model_len = MAX_MODEL_LEN

    def __init__(self, **_):
        pass

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        # 한국어는 토큰당 1.2~1.4자 정도라 //2로 잡으면 예산을 크게 과소평가한다.
        return int(sum(len(m["content"]) for m in messages) / 1.3)

    def chat(self, batch: List[List[Dict[str, str]]], schema: Dict[str, Any],
             max_tokens: int) -> List[Tuple[str, Any, Any]]:
        items = list(schema.get("properties", {}).keys())
        text = json.dumps({v: {"위반여부": 0, "근거문구": None} for v in items}, ensure_ascii=False)
        return [(text, None, None) for _ in batch]


# ===== 8. 판정 토큰 logprob → P(위반) =====
JUDGE = re.compile(r'"(v\d+)"\s*:\s*\{\s*"위반여부"\s*:\s*([01])')


def _token_texts(tok, token_ids, logprobs) -> Optional[List[str]]:
    """생성 토큰을 문자열로 편다. logprobs 항목의 decoded_token을 우선 쓴다."""
    if token_ids is None or logprobs is None or len(token_ids) != len(logprobs):
        return None
    out = []
    for tid, lp in zip(token_ids, logprobs):
        s = None
        if isinstance(lp, dict):
            e = lp.get(tid)
            s = getattr(e, "decoded_token", None) if e is not None else None
        if s is None:
            try:
                s = tok.decode([tid])
            except Exception:
                return None
        out.append(s)
    return out


def item_probs(text: str, token_ids, logprobs, tok, items: Sequence[str]) -> Tuple[Dict[str, float], int]:
    """판정 자리의 토큰 분포에서 P(위반)을 읽는다.

    구조화 디코딩이 그 자리에서 허용하는 토큰은 '0'과 '1'뿐이므로, 두 후보의 logprob을
    정규화하면 곧 모델이 본 위반 확률이다. 토큰이 다음 문자와 붙어 나와 자리를 특정할 수
    없으면 하드 라벨(0.0/1.0)로 되돌린다 — 그 횟수를 함께 돌려준다.
    """
    hard = {m.group(1): float(m.group(2)) for m in JUDGE.finditer(text or "")}
    probs = {v: hard.get(v, 0.0) for v in items}
    toks = _token_texts(tok, token_ids, logprobs) if tok is not None else None
    if not toks:
        return probs, len(items)

    starts, pos = [], 0
    for t in toks:                                         # 각 토큰이 시작하는 문자 위치
        starts.append(pos)
        pos += len(t)
    joined = "".join(toks)
    base = joined.find(text[:40]) if text[:40] and text[:40] in joined else 0

    fallback = 0
    for m in JUDGE.finditer(text or ""):
        v = m.group(1)
        if v not in probs:
            continue
        off = base + m.start(2)
        i = next((k for k in range(len(starts) - 1, -1, -1) if starts[k] <= off), None)
        lp = logprobs[i] if i is not None and i < len(logprobs) else None
        if not isinstance(lp, dict):
            fallback += 1
            continue
        cand: Dict[str, float] = {}
        for e in lp.values():
            s = (getattr(e, "decoded_token", "") or "").lstrip()
            if s[:1] in ("0", "1") and s[:1] not in cand:
                cand[s[:1]] = float(getattr(e, "logprob", -math.inf))
        if "0" in cand and "1" in cand:
            top = max(cand["0"], cand["1"])
            e0, e1 = math.exp(cand["0"] - top), math.exp(cand["1"] - top)
            probs[v] = e1 / (e0 + e1)
        elif "1" in cand and hard.get(v) == 1.0:
            probs[v] = 1.0
        elif "0" in cand and hard.get(v) == 0.0:
            probs[v] = 0.0
        else:
            fallback += 1
    return probs, fallback


def extract_evidence(text: str, items: Sequence[str]) -> Dict[str, Optional[str]]:
    obj = None
    try:
        obj = json.loads((text or "").strip())
    except json.JSONDecodeError:
        i, j = (text or "").find("{"), (text or "").rfind("}")
        if i >= 0 and j > i:
            try:
                obj = json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                obj = None
    out: Dict[str, Optional[str]] = {v: None for v in items}
    if isinstance(obj, dict):
        for v in items:
            cell = obj.get(v)
            if isinstance(cell, dict):
                ev = cell.get("근거문구")
                out[v] = ev if isinstance(ev, str) else None
    return out


# ===== 9. 근거문구 정리 =====
def clean_evidence(ev: Optional[str], src: str) -> str:
    """NFC · 앞뒤 공백 제거 · 500자 상한 · 수식 접두(=,+,@) 제거 · 원문 부분문자열만 인정.

    모델이 공백이나 줄바꿈을 흘린 경우를 위해 앞뒤를 조금씩 깎아 가며 다시 대조한다.
    """
    if not ev:
        return ""
    ev = unicodedata.normalize("NFC", ev).replace("\r", "").strip()
    if not ev:
        return ""
    ev = ev[:EVIDENCE_MAX]
    if ev in src:
        return ev if ev[0] not in "=+@" else ""
    for cut in (ev.strip("·-—…\"'“”‘’ \t\n"), re.sub(r"\s+", " ", ev)):
        if cut and cut in src:
            return cut if cut[0] not in "=+@" else ""
    if len(ev) > 40:                                       # 뒤가 잘렸을 때 앞부분만이라도 살린다
        for n in (len(ev) * 3 // 4, len(ev) // 2):
            head = ev[:n].rstrip()
            if len(head) >= 20 and head in src:
                return head if head[0] not in "=+@" else ""
    return ""


def to_row(rec_id: str, decided: Dict[str, int], evidence: Dict[str, str]) -> Dict[str, Any]:
    row = {"id": rec_id}
    for i, v in enumerate(ITEMS, 1):
        row[v] = int(decided.get(v, 0))
        row[f"e{i}"] = evidence.get(v, "") if row[v] == 1 and v not in ABSENCE else ""
    return row


# ===== 10. submission.csv 저장·자가검증 =====
def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with io.open(path, "w", encoding="utf-8", newline="") as f:   # UTF-8(BOM 없음) · RFC4180 quoting
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: unicodedata.normalize("NFC", str(r[k])) for k in COLUMNS})


def validate_csv(path: str, expected_ids: List[str]) -> List[str]:
    """열 49 · 행 수 = 입력 건수 · id 유일·일치 · v 0/1 · e 500자 이하 · 부재탐지 e 빈칸"""
    errs: List[str] = []
    with io.open(path, "r", encoding="utf-8", newline="") as f:
        rd = csv.reader(f)
        header = next(rd, None)
        rows = list(rd)
    if header != COLUMNS:
        errs.append(f"헤더 불일치: {len(header or [])}열 (기대 {len(COLUMNS)})")
        return errs
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
            errs.append(f"{r[0]}: 위반여부에 0/1 아닌 값")
        if any(len(x) > EVIDENCE_MAX for x in r[25:]):
            errs.append(f"{r[0]}: 근거문구 {EVIDENCE_MAX}자 초과")
        if any(r[j] for j in absence_idx):
            errs.append(f"{r[0]}: 부재탐지 항목에 근거문구")
        if any(x.startswith(("=", "+", "@")) for x in r[25:]):
            errs.append(f"{r[0]}: 수식 접두 근거문구")
    return errs


# ===== 11. 실행 =====
def build_tasks(recs: List[Dict[str, Any]], tbl, a: Assets, runner,
                doc_budget: int, law_budget: int,
                only_groups: Optional[Sequence[str]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """레코드×그룹 단위 작업 목록. 게이트로 전부 닫힌 그룹은 만들지 않는다.

    단, 규칙상 공고 1건마다 고정 LLM을 1회 이상 호출해야 하므로, 모든 그룹이 닫힌 레코드에는
    가장 넓은 그룹을 강제로 되살린다.
    """
    tasks: List[Dict[str, Any]] = []
    per_rec: Dict[str, int] = defaultdict(int)
    gated = Counter()
    for ri, rec in enumerate(recs):
        f = derive_facts(rec, a)
        rec["_facts"] = f
        for v, why in gate_reasons(f, a).items():
            gated[v] += 1
        for group, members in GROUPS.items():
            if only_groups and group not in only_groups:
                continue
            items = [v for v in members if applicable(v, f, a)]
            if not items:
                continue
            law = group_law_text(group, f, a, law_budget)
            sysmsg = build_group_system(group, items, tbl, law, a.guides)
            usr, _stat = build_group_user(rec, f, a, group, doc_budget)
            tasks.append({"ri": ri, "group": group, "items": items,
                          "messages": build_messages(sysmsg, usr)})
            per_rec[rec["id"]] += 1

    for ri, rec in enumerate(recs):                        # 1건 이상 호출 보장
        if per_rec[rec["id"]] or only_groups:               # 그룹 한정 실행은 검증용이라 제외
            continue
        if False:
            continue
        f = rec["_facts"]
        group, items = "제한경쟁", GROUPS["제한경쟁"]
        law = group_law_text(group, f, a, law_budget)
        usr, _ = build_group_user(rec, f, a, group, doc_budget)
        tasks.append({"ri": ri, "group": group, "items": items,
                      "messages": build_messages(build_group_system(group, items, tbl, law, a.guides), usr)})
        per_rec[rec["id"]] = 1
        log(f"  [1콜 보장] {rec['id']} 전 항목 게이트 차단 → {group} 그룹 강제 실행")

    # 토큰 예산 초과분은 문서 구간을 줄여 다시 만든다.
    budget = runner.max_model_len - PROMPT_BUDGET_MARGIN - 1024
    shrunk, ntok = 0, []
    for t in tasks:
        n = runner.count_tokens(t["messages"])
        db = doc_budget
        while n > budget and db > 1500:
            db = int(db * min(0.8, budget / n * 0.9))
            rec, f = recs[t["ri"]], recs[t["ri"]]["_facts"]
            usr, _ = build_group_user(rec, f, a, t["group"], db)
            t["messages"][1]["content"] = usr
            n = runner.count_tokens(t["messages"])
            shrunk += 1
        ntok.append(n)
    stat = {"작업수": len(tasks), "레코드당_평균콜": round(len(tasks) / max(len(recs), 1), 2),
            "토큰_중앙": sorted(ntok)[len(ntok) // 2] if ntok else 0,
            "토큰_최대": max(ntok) if ntok else 0, "예산축소": shrunk,
            "게이트차단_항목별": dict(gated)}
    return tasks, stat


def run_chunk(runner, tasks: List[Dict[str, Any]], max_tokens: int):
    """같은 스키마끼리 묶어 호출한다. 실패하면 건 단위로 재시도하고 그래도 실패하면 빈 출력."""
    schema = decode_schema_for(tasks[0]["items"])
    batch = [t["messages"] for t in tasks]
    try:
        return runner.chat(batch, schema, max_tokens)
    except Exception as e:
        log(f"  ! 청크({len(batch)}건) 실패 → 건 단위 재시도: {type(e).__name__}: {str(e)[:160]}")
    outs = []
    for t in tasks:
        try:
            outs.append(runner.chat([t["messages"]], decode_schema_for(t["items"]), max_tokens)[0])
        except Exception as e:
            log(f"  ! 건 단위 실패 → 빈 출력: {type(e).__name__}: {str(e)[:160]}")
            outs.append(("", None, None))
    return outs


def run(input_path: str, out_path: str, runner_cls, limit: Optional[int], chunk: int,
        doc_budget: int, law_budget: int, data_dir: str, asset_dir: str,
        dump_probs: Optional[str] = None, only_groups: Optional[Sequence[str]] = None,
        **runner_kw) -> Dict[str, Any]:
    t_all = time.time()
    log(f"경로: cwd={os.getcwd()} · data={os.path.abspath(data_dir)} · "
        f"asset={os.path.abspath(asset_dir)} · out={os.path.abspath(os.path.dirname(out_path))}")
    recs = list(iter_records(input_path, limit=limit))
    log(f"입력 {len(recs)}건 ← {input_path}")
    if not recs:
        write_csv([], out_path)
        return {"건수": 0, "자가검증": "PASS"}

    tbl = item_table(data_dir)
    a = Assets(asset_dir)
    runner = runner_cls(**runner_kw)
    log(f"모델 로드 {runner.load_seconds:.1f}s")

    tasks, tstat = build_tasks(recs, tbl, a, runner, doc_budget, law_budget, only_groups)
    log(json.dumps(tstat, ensure_ascii=False))

    # 같은 (그룹, 항목집합)끼리 모아 호출 — 스키마가 같아야 한 배치로 나간다.
    by_shape: Dict[Tuple[str, Tuple[str, ...]], List[int]] = defaultdict(list)
    for i, t in enumerate(tasks):
        by_shape[(t["group"], tuple(t["items"]))].append(i)

    probs: List[Dict[str, float]] = [dict() for _ in recs]
    evid: List[Dict[str, str]] = [dict() for _ in recs]
    t_inf, done, fallback, empty = time.time(), 0, 0, 0
    tok = getattr(runner, "tok", None)

    for (group, items), idxs in sorted(by_shape.items()):
        max_tokens = min(1536, 96 + 190 * len(items))
        for s in range(0, len(idxs), chunk):
            part = [tasks[i] for i in idxs[s:s + chunk]]
            outs = run_chunk(runner, part, max_tokens)
            for t, (text, tids, lps) in zip(part, outs):
                if not text:
                    empty += 1
                p, fb = item_probs(text, tids, lps, tok, t["items"])
                fallback += fb
                ev = extract_evidence(text, t["items"])
                probs[t["ri"]].update(p)
                for v, e in ev.items():
                    if e:
                        evid[t["ri"]][v] = e
            done += len(part)
            log(f"  {group}{list(items)[:2]}… {done}/{len(tasks)} … {time.time() - t_inf:.0f}s")
    inf_seconds = time.time() - t_inf

    rows, ev_kept, ev_dropped, pos, rejected = [], 0, 0, Counter(), Counter()
    for rec, p, ev in zip(recs, probs, evid):
        src = unicodedata.normalize("NFC", full_text(rec))
        decided, cells = {}, {}
        for v in ITEMS:
            hit = int(p.get(v, 0.0) >= a.threshold[v])
            cell = ""
            if hit and v not in ABSENCE:
                raw = ev.get(v)
                cell = clean_evidence(raw, src)
                ev_kept += int(bool(cell))
                ev_dropped += int(bool(raw) and not cell)
                # 근거가 항목의 필수 형태를 갖추지 못하면 위반을 물린다. 규칙 기반 후처리로,
                # 근거 없이는 성립할 수 없는 항목(예: v21은 지분율 숫자)의 오탐을 막는다.
                rx = a.evidence_rx.get(v)
                if rx is not None and not rx.search(cell):
                    hit, cell = 0, ""
                    rejected[v] += 1
            decided[v] = hit
            if hit:
                pos[v] += 1
            if hit and v not in ABSENCE:
                cells[v] = cell
        rows.append(to_row(rec["id"], decided, cells))
    assert len(rows) == len(recs)

    if dump_probs:
        # 임계값을 바꿀 때마다 모델을 다시 돌리지 않으려고 P(위반)을 그대로 남긴다.
        # 게이트로 닫힌 항목은 0.0 — 스윕에서도 그대로 0이어야 맞다.
        os.makedirs(os.path.dirname(os.path.abspath(dump_probs)), exist_ok=True)
        with io.open(dump_probs, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["id"] + ITEMS)
            for rec, p in zip(recs, probs):
                w.writerow([rec["id"]] + [f"{p.get(v, 0.0):.6f}" for v in ITEMS])
        log(f"확률 덤프 → {dump_probs}")

    write_csv(rows, out_path)
    errs = validate_csv(out_path, [r["id"] for r in recs])
    total_pos = sum(pos.values())
    report = {
        "건수": len(recs), "LLM호출": len(tasks), "모델로드_s": round(runner.load_seconds, 1),
        "추론_s": round(inf_seconds, 1), "건당_s": round(inf_seconds / len(recs), 2),
        "전체_s": round(time.time() - t_all, 1),
        "빈출력": empty, "logprob_폴백": fallback,
        "양성_합계": total_pos, "양성_항목별": {v: pos[v] for v in ITEMS if pos[v]},
        "근거_유지": ev_kept, "근거_원문불일치_폐기": ev_dropped,
        "근거형식불충족_기각": dict(rejected) or 0,
        "출력": out_path, "자가검증": "PASS" if not errs else errs,
    }
    log(json.dumps(report, ensure_ascii=False))
    if total_pos == 0:
        log("[경고] 양성 판정이 0건입니다. 전 항목 0 제출은 Macro F1이 정확히 0.0입니다 — "
            "model/임계값.json을 낮추거나 게이트·프롬프트를 점검하세요.")
    if empty:
        log(f"[주의] 빈 출력 {empty}건. 구조화 출력 설정과 토큰 예산을 확인하세요.")
    if fallback:
        log(f"[참고] logprob 폴백 {fallback}건 — 그만큼은 하드 0/1로 판정했습니다(임계값 무효).")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="24개 항목의 법령 위반 여부 판정")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    ap.add_argument("--asset-dir", default=ASSET_DIR, help="model/ — 조문발췌·경쟁제품·게이트·임계값")
    ap.add_argument("--input", default=None, help="기본 = <data-dir>/test.jsonl.gz")
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--quantization", default=os.environ.get("PPS_QUANT", QUANT),
                    help="채점 서버 = int8_per_channel_weight_only · 'none'이면 미양자화")
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    ap.add_argument("--chunk", type=int, default=128, help="LLM.chat 한 번에 넘길 건수")
    ap.add_argument("--doc-budget", type=int, default=10000, help="그룹당 문서 구간 글자 수 상한")
    ap.add_argument("--law-budget", type=int, default=5000, help="그룹당 조문 글자 수 상한")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only-groups", default=None, metavar="A,B",
                    help="그룹만 골라 실행한다(검증용). 예: 공동도급 · 나머지 항목은 0으로 남는다")
    ap.add_argument("--dump-probs", default=None, metavar="CSV",
                    help="항목별 P(위반)을 CSV로 남긴다 — 임계값 스윕용(제출 실행에는 불필요)")
    ap.add_argument("--mock", action="store_true", help="모델 없이 게이트·프롬프트 흐름만 확인")
    a = ap.parse_args()

    input_path = a.input or os.path.join(a.data_dir, "test.jsonl.gz")
    if not os.path.exists(input_path) and os.path.exists(input_path[:-3]):
        input_path = input_path[:-3]                       # test.jsonl.gz 대신 test.jsonl 만 있는 경우
    out_path = os.path.join(a.output_dir, "submission.csv")
    quant = None if str(a.quantization).lower() in ("none", "") else a.quantization
    runner_kw = {} if a.mock else dict(model_dir=a.model_dir, quant=quant, seed=SEED,
                                       gpu_mem=a.gpu_mem, tp=a.tp, max_model_len=a.max_model_len)
    report = run(input_path, out_path, MockRunner if a.mock else VLLMRunner,
                 limit=a.limit, chunk=a.chunk, doc_budget=a.doc_budget, law_budget=a.law_budget,
                 data_dir=a.data_dir, asset_dir=a.asset_dir, dump_probs=a.dump_probs,
                 only_groups=[g.strip() for g in a.only_groups.split(",")] if a.only_groups else None,
                 **runner_kw)
    return 0 if report.get("자가검증") in ("PASS", None) else 1


if __name__ == "__main__":
    sys.exit(main())
