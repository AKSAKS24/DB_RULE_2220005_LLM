from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os, json, re

# --- Env + LLM Setup -----------------------------------------------------------
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required.")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

app = FastAPI(title="SAP Note 2220005 Obsolete Usage Detector")

# --- Models --------------------------------------------------------------------
class DeclFinding(BaseModel):
    snippet: str
    suggestion: str

class Unit(BaseModel):
    pgm_name: str
    inc_name: str
    type: str
    name: Optional[str] = ""
    start_line: Optional[int] = 0
    end_line: Optional[int] = 0
    code: Optional[str] = ""
    decl_findings: Optional[List[DeclFinding]] = Field(default=None)

# --- Pattern Setup -------------------------------------------------------------
OBSOLETE_TYPE_MAP = {
    "KONV": "PRCD_ELEMENTS",
    "DZAEHK": "VFPRC_COND_COUNT",
    "DZAEKO": "VFPRC_COND_COUNT_HEAD"
}

# Match declarations (DATA/TYPES/etc.) of obsolete types
TYPE_DECL_RE = re.compile(
    r'''\b(
        (DATA|TYPES|FIELD\-SYMBOLS|CONSTANTS)     # decl keyword
        [^.\n]*?                                  # variable name etc.
        \b(TYPE|LIKE)\b                           # TYPE or LIKE
        (\s+TABLE\s+OF\s+|\s+)?                   # optional TABLE OF
        (?P<dtype>KONV|DZAEHK|DZAEKO)             # obsolete type
    )''',
    re.IGNORECASE | re.VERBOSE
)

# Match SQL DML using obsolete tables (SELECT|INSERT|UPDATE|DELETE)
ABAP_SQL_STMT_RE = re.compile(
    r'''\b
        (SELECT|INSERT|UPDATE|DELETE)           # SQL keyword
        [^\n]*?                                 # anything, including spaces
        (?P<table>KONV|DZAEHK|DZAEKO)           # obsolete table
        [^\n]*                                  # rest of the line
        (?:\n|$)                                # until next line or end
    ''', re.IGNORECASE | re.VERBOSE
)

# --- Extraction Logic ----------------------------------------------------------
def extract_decl_findings(abap_code: str) -> List[DeclFinding]:
    findings = []

    # Declarations
    for m in TYPE_DECL_RE.finditer(abap_code or ""):
        dtype = m.group("dtype").upper()
        suggestion = re.sub(rf"\b{dtype}\b", OBSOLETE_TYPE_MAP[dtype], m.group(0), flags=re.IGNORECASE)
        suggestion += f"\n* TODO: {dtype} is obsolete. Use {OBSOLETE_TYPE_MAP[dtype]} instead."
        findings.append(DeclFinding(snippet=m.group(0).strip(), suggestion=suggestion.strip()))

    # SQL Statements
    for m in ABAP_SQL_STMT_RE.finditer(abap_code or ""):
        table = m.group("table").upper()
        orig = m.group(0)
        suggestion = re.sub(rf"\b{table}\b", OBSOLETE_TYPE_MAP[table], orig, flags=re.IGNORECASE)
        suggestion += f"\n* TODO: {table} is obsolete. Use {OBSOLETE_TYPE_MAP[table]} or new data model."
        findings.append(DeclFinding(snippet=orig.strip(), suggestion=suggestion.strip()))

    return findings

# --- LLM Prompt Configuration -------------------------------------------------
SYSTEM_MSG = """
You are an experienced ABAP reviewer for S/4HANA migration issues.
REPLY STRICTLY IN JSON!
For each decl_findings[].snippet given:
- Output a bullet point with:
    - The exact ABAP code snippet (inline).
    - The change required (decl_findings[].suggestion).
- Do NOT compress or omit snippets; show each code+fix directly, no referencing by index; no extra prose or numbering.
""".strip()

USER_TEMPLATE = """
Unit metadata:
Program: {pgm_name}
Include: {inc_name}
Unit type: {unit_type}
Unit name: {unit_name}
Start line: {start_line}
End line: {end_line}

ABAP code context (optional):
{code}

decl_findings (JSON list, each with .snippet and .suggestion):
{findings_json}

Instructions:
1. Write a 1-paragraph assessment of obsolete type/data element declaration risks.
2. Write a llm_prompt: for every finding, add a bullet point displaying both the code and recommendation from .suggestion (both inline).
Return JSON only:
{{
  "assessment": "<human language paragraph>",
  "llm_prompt": "<action bullets>"
}}
""".strip()

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MSG),
    ("user", USER_TEMPLATE)
])
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.0)
parser = JsonOutputParser()
chain = prompt | llm | parser

# --- LLM Assessment/Prompt Logic ----------------------------------------------
async def llm_assess_and_prompt(unit: Unit) -> Optional[Dict[str, str]]:
    findings = extract_decl_findings(unit.code or "")
    if not findings:
        return None  # NEGATIVE: omit this unit
    findings_json = json.dumps([f.model_dump() for f in findings], ensure_ascii=False, indent=2)
    try:
        return await chain.ainvoke({
            "pgm_name": unit.pgm_name,
            "inc_name": unit.inc_name,
            "unit_type": unit.type,
            "unit_name": unit.name or "",
            "start_line": unit.start_line or 0,
            "end_line": unit.end_line or 0,
            "code": unit.code or "",
            "findings_json": findings_json
        })
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

# --- API Endpoints ------------------------------------------------------------
@app.post("/assess-2220005-migration")
async def detect_obsolete_declarations(units: List[Unit]) -> List[Dict[str, Any]]:
    out = []
    for u in units:
        llm_out = await llm_assess_and_prompt(u)
        if llm_out is None:
            continue  # No findings for this unit
        obj = u.model_dump()
        obj["assessment"] = llm_out.get("assessment", "")
        prompt_out = llm_out.get("llm_prompt", "")
        if isinstance(prompt_out, list):
            prompt_out = "\n".join(str(x) for x in prompt_out if x is not None)
        obj["llm_prompt"] = prompt_out
        obj.pop("decl_findings", None)
        out.append(obj)
    return out

@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_MODEL}