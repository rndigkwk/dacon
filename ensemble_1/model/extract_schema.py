# -*- coding: utf-8 -*-
"""사실 추출용 JSON Schema (vLLM structured outputs / xgrammar 용)."""

# maxLength 는 문법 컴파일러 버전에 따라 미지원이라 넣지 않는다.
# 길이 제한은 프롬프트 지시 + 후처리(clean_evidence)로 건다.
_S = {"type": ["string", "null"]}
_B = {"type": "boolean"}


def _obj(props, req=None):
    return {"type": "object", "additionalProperties": False,
            "required": req if req is not None else list(props), "properties": props}


FACT_SCHEMA = _obj({
    "공고문금액": _obj({"추정가격": {"type": ["integer", "null"]},
                    "사업예산": {"type": ["integer", "null"]}, "근거문구": _S}),
    "참가자격_특정기관한정": _obj({"해당": _B, "근거문구": _S}),
    "실적제한": _obj({"해당": _B,
                  "요구실적금액": {"type": ["integer", "null"]},
                  "발주처_특정": _B, "근거문구": _S}),
    "지역제한": _obj({"해당": _B,
                  "제한광역목록": {"type": "array", "maxItems": 8,
                              "items": {"type": "string"}},
                  "기초단위포함": _B, "근거문구": _S}),
    "기업규모제한": _obj({"구분": {"type": "string",
                            "enum": ["없음", "소기업소상공인", "중소기업자"]},
                    "근거문구": _S}),
    "중기간경쟁제품": _B,
    "직접생산확인요구": _obj({"해당": _B, "근거문구": _S}),
    "특정모델명": _obj({"해당": _B, "동등이상허용": _B, "근거문구": _S}),
    "물품공급확약서": _obj({"입찰단계요구": _B, "근거문구": _S}),
    "소프트웨어사업": _obj({"해당": _B, "대기업참여제한_명시": _B}),
    "공동수급": _obj({"최소지분율": {"type": ["number", "null"]}, "근거문구": _S}),
    "설명회": _obj({"개최": _B, "참석의무": _B,
                 "일자": {"type": ["string", "null"]}, "근거문구": _S}),
    "등록값불일치": _obj({"해당": _B,
                    "항목": {"type": ["string", "null"]}, "근거문구": _S}),
})
