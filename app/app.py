from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any
import os, json, re

# ---- Env setup ----
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required.")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")

# LangChain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

app = FastAPI(title="SAP Note 2220005 Declaration Assessment")

# ===== Models =====
class SelectItem(BaseModel):
    table: str
    target_type: str
    target_name: str
    used_fields: List[str]
    suggested_fields: List[str]
    suggested_statement: str

    @field_validator("used_fields", "suggested_fields")
    @classmethod
    def no_none_elems(cls, v: List[str]) -> List[str]:
        return [x for x in v if x is not None]

class Unit(BaseModel):
    pgm_name: str
    inc_name: str
    type: str
    name: str
    selects: List[SelectItem] = Field(default_factory=list)

# ===== Obsolete type mappings for SAP Note 2220005 =====
OBSOLETE_TYPE_MAP = {
    "KONV": "PRCD_ELEMENTS",
    "DZAEHK": "VFPRC_COND_COUNT",
    "DZAEKO": "VFPRC_COND_COUNT_HEAD"
}

# Declaration detector regex
TYPE_DECL_RE = re.compile(
    r"""\b(?P<full>
        (DATA|TYPES|FIELD\-SYMBOLS|CONSTANTS)     # keyword
        [^.\n]*?                                  # variable name etc.
        \b(TYPE|LIKE)\b                           # TYPE or LIKE
        (\s+TABLE\s+OF\s+|\s+)?                   # optional TABLE OF
        (?P<dtype>\w+)                            # datatype name
    )""",
    re.IGNORECASE | re.VERBOSE
)

# ===== Summariser for SAP Note 2220005 risks =====
def summarize_selects(unit: Unit) -> Dict[str, Any]:
    tables_count: Dict[str, int] = {}
    total = len(unit.selects)
    flagged = []
    for s in unit.selects:
        tbl_upper = s.table.upper()
        tables_count[tbl_upper] = tables_count.get(tbl_upper, 0) + 1
        if tbl_upper in OBSOLETE_TYPE_MAP:
            new_type = OBSOLETE_TYPE_MAP[tbl_upper]
            flagged.append({
                "table": s.table,
                "target": s.target_name,
                "reason": f"Usage of {tbl_upper} detected. Replace with {new_type} per SAP Note 2220005."
            })

    return {
        "program": unit.pgm_name,
        "include": unit.inc_name,
        "unit_type": unit.type,
        "unit_name": unit.name,
        "stats": {
            "total_statements": total,
            "tables_frequency": tables_count,
            "note_2220005_flags": flagged
        }
    }

# ===== Prompt for SAP Note–specific fix =====
SYSTEM_MSG = "You are a precise ABAP reviewer familiar with SAP Note 2220005. Output strict JSON only."

USER_TEMPLATE = """
You are assessing ABAP code usage in light of SAP Note 2220005 (Pricing & Condition Technique data model changes).

From S/4HANA:
- Table/structure KONV is obsolete → use PRCD_ELEMENTS (or released CDS views)
- Data element DZAEHK is obsolete → use VFPRC_COND_COUNT
- Data element DZAEKO is obsolete → use VFPRC_COND_COUNT_HEAD

We provide program/include/unit metadata and analysis of declarations.

Your tasks:
1) Produce a concise **assessment** highlighting:
   - Which declarations reference obsolete types or data elements.
   - Why migration is needed.
   - Potential functional and data impact.
2) Produce an **LLM remediation prompt** to:
   - Scan ABAP code in this unit for KONV, DZAEHK, DZAEKO in DATA/TYPES/FIELD-SYMBOLS/CONSTANTS/LIKE declarations.
   - Replace with respective new names (PRCD_ELEMENTS, VFPRC_COND_COUNT, VFPRC_COND_COUNT_HEAD).
   - Add `TODO` comments where manual adjustments are needed for logic/refactoring.
   - Output strictly in JSON with: original_code, remediated_code, changes[].

Return ONLY strict JSON:
{{
  "assessment": "<concise note 2220005 impact>",
  "llm_prompt": "<prompt for LLM code fixer>"
}}

Unit metadata:
- Program: {pgm_name}
- Include: {inc_name}
- Unit type: {unit_type}
- Unit name: {unit_name}

Analysis:
{plan_json}

selects (JSON):
{selects_json}
""".strip()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MSG),
        ("user", USER_TEMPLATE),
    ]
)

llm = ChatOpenAI(model=OPENAI_MODEL)
parser = JsonOutputParser()
chain = prompt | llm | parser

# ===== LLM Call =====
def llm_assess_and_prompt(unit: Unit) -> Dict[str, str]:
    plan = summarize_selects(unit)
    plan_json = json.dumps(plan, ensure_ascii=False, indent=2)
    selects_json = json.dumps([s.model_dump() for s in unit.selects], ensure_ascii=False, indent=2)

    try:
        return chain.invoke({
            "pgm_name": unit.pgm_name,
            "inc_name": unit.inc_name,
            "unit_type": unit.type,
            "unit_name": unit.name,
            "plan_json": plan_json,
            "selects_json": selects_json
        })
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

# ===== Utility to parse declarations into SelectItem list =====
def extract_declarations(abap_code: str) -> List[SelectItem]:
    selects: List[SelectItem] = []
    for m in TYPE_DECL_RE.finditer(abap_code):
        dtype = m.group("dtype")
        if dtype.upper() in OBSOLETE_TYPE_MAP:
            new_dtype = OBSOLETE_TYPE_MAP[dtype.upper()]
            new_stmt = re.sub(rf"\b{dtype}\b", new_dtype, m.group("full"), flags=re.IGNORECASE)
            new_stmt = new_stmt + f"\n* TODO: {dtype.upper()} is obsolete, replaced with {new_dtype}."
            selects.append(
                SelectItem(
                    table=dtype,
                    target_type="",
                    target_name="",
                    used_fields=[],
                    suggested_fields=[],
                    suggested_statement=new_stmt
                )
            )
    return selects

# ===== API =====
@app.post("/assess-2220005-migration")
def assess_2220005_migration(units: List[Unit]) -> List[Dict[str, Any]]:
    out = []
    for u in units:
        # Fill selects from code (simulate extractor)
        # In your integration, this parse step may come earlier
        code_attr = getattr(u, "code", "")
        if code_attr:
            u.selects = extract_declarations(code_attr)
        obj = u.model_dump()
        llm_out = llm_assess_and_prompt(u)
        obj["assessment"] = llm_out.get("assessment", "")
        obj["llm_prompt"] = llm_out.get("llm_prompt", "")
        obj.pop("selects", None)
        out.append(obj)
    return out

@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_MODEL}