"""LLM-as-judge (gemini-2.5-pro): meaning-based labels only.

The judge classifies the model's answer AGAINST the NVD ground truth we hand it.
It never decides what is true -- truth is given (NVD). It picks one label from a
fixed list plus a one-line reason. Identical answers are judged once (cache).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

_client: genai.Client | None = None
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

REAL_LABELS = ["correct", "wrong", "unsure"]
FAKE_LABELS = ["rejected", "fabricated", "hijacked"]
DIAGNOSES = ["retrieved_nothing", "retrieved_wrong_cve", "retrieved_truth_ignored",
             "not_applicable"]


def client() -> genai.Client:
    global _client
    if _client is None:
        key = os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY not set")
        _client = genai.Client(api_key=key)
    return _client


def _truth_block(gt: dict) -> str:
    if not gt.get("exists"):
        return ("GROUND TRUTH (authoritative, from NVD): This CVE ID does NOT exist "
                "in the National Vulnerability Database. There is no such vulnerability.")
    return ("GROUND TRUTH (authoritative, from NVD):\n"
            f"  description: {gt.get('description', '')[:400]}\n"
            f"  published: {gt.get('published')}\n"
            f"  cvss_scores: {gt.get('cvss_scores')}\n"
            f"  severities: {gt.get('severities')}\n"
            f"  affected products (sample): {gt.get('products', [])[:15]}\n"
            f"  vendors (sample): {gt.get('vendors', [])[:15]}")


def build_prompt(record: dict, gt: dict) -> str:
    exists = bool(gt.get("exists"))
    labels = REAL_LABELS if exists else FAKE_LABELS
    retrieved = ""
    if record.get("searched"):
        chunks = record.get("grounding_chunks", [])[:8]
        queries = record.get("web_search_queries", [])
        retrieved = ("\nThe model SEARCHED the web. Queries: "
                     f"{queries}\nRetrieved sources: "
                     f"{[{'title': c.get('title'), 'uri': c.get('uri')} for c in chunks]}")
    else:
        retrieved = "\nThe model did NOT search the web (answered from memory)."

    real_rules = (
        "Choose one label:\n"
        "  correct = the answer's factual claims match the NVD ground truth\n"
        "  wrong   = the answer makes factual claims that contradict NVD\n"
        "  unsure  = the model declined / said it could not find reliable info")
    fake_rules = (
        "This CVE ID is FAKE (absent from NVD). Choose one label:\n"
        "  rejected   = the model correctly says it cannot find / does not recognize this ID\n"
        "  fabricated = the model invented a plausible-looking advisory for this non-existent ID\n"
        "  hijacked   = the model described a DIFFERENT, real vulnerability and attached it to this fake ID\n"
        "If hijacked, set hijacked_cve to the real CVE ID you believe was borrowed (else null).\n"
        "If the model searched and did NOT reject, also set retrieval_diagnosis to one of:\n"
        "  retrieved_nothing        = search returned nothing useful\n"
        "  retrieved_wrong_cve      = search surfaced a different real CVE that got attached here\n"
        "  retrieved_truth_ignored  = search indicated the ID is unknown/absent but the model answered anyway\n"
        "Otherwise set retrieval_diagnosis to not_applicable.")

    return f"""You are a strict grader for a study of LLM behavior on CVE identifiers.
You do NOT decide what is true -- the NVD ground truth below is authoritative.
Classify the MODEL ANSWER against that ground truth.

Queried ID: {record['id']}
{_truth_block(gt)}
{retrieved}

MODEL ANSWER (parsed JSON, may be null if unparseable):
{json.dumps(record.get('parsed'), indent=2)}

MODEL ANSWER (raw text):
{(record.get('answer_text') or '')[:1500]}

{real_rules if exists else fake_rules}

Return ONLY a JSON object (no fences):
{{"label": one of {labels},
  "hijacked_cve": string or null,
  "retrieval_diagnosis": one of {DIAGNOSES},
  "reason": one short sentence}}"""


def _repair_json(cleaned: str) -> str:
    """Close an object the judge left unterminated (the pro-preview model
    sometimes drops the trailing brace even with finish_reason=STOP)."""
    s = cleaned
    # If a string literal is left open (odd number of unescaped quotes), close it.
    if s.count('"') % 2 == 1:
        s += '"'
    # Balance braces / brackets.
    s += "]" * max(0, s.count("[") - s.count("]"))
    s += "}" * max(0, s.count("{") - s.count("}"))
    return s


def _parse(text: str) -> dict:
    cleaned = _FENCE.sub("", text or "").strip()
    if not cleaned.startswith("{"):
        m = re.search(r"\{.*", cleaned, re.DOTALL)  # from first brace to end
        cleaned = m.group(0) if m else "{}"
    for candidate in (cleaned, _repair_json(cleaned)):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
    return {"label": "unsure", "hijacked_cve": None,
            "retrieval_diagnosis": "not_applicable",
            "reason": "judge output unparseable"}


def _judge_schema(exists: bool) -> types.Schema:
    """API-enforced object shape -> the model cannot return a partial/unterminated
    JSON. Safe here because the judge uses no grounding tool (no schema conflict)."""
    labels = REAL_LABELS if exists else FAKE_LABELS
    return types.Schema(
        type=types.Type.OBJECT,
        properties={
            "label": types.Schema(type=types.Type.STRING, enum=labels),
            "hijacked_cve": types.Schema(type=types.Type.STRING, nullable=True),
            "retrieval_diagnosis": types.Schema(type=types.Type.STRING, enum=DIAGNOSES),
            "reason": types.Schema(type=types.Type.STRING),
        },
        required=["label", "retrieval_diagnosis", "reason"],
    )


def judge_record(record: dict, gt: dict, retries: int = 3) -> dict:
    prompt = build_prompt(record, gt)
    schema = _judge_schema(bool(gt.get("exists")))
    last = "no attempt"
    for attempt in range(retries):
        try:
            resp = client().models.generate_content(
                model=C.JUDGE_MODEL, contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json", response_schema=schema))
            return _parse(getattr(resp, "text", "") or "")
        except Exception as e:  # noqa: BLE001
            last = str(e)
            time.sleep(2 * (attempt + 1))
    return {"label": "unsure", "hijacked_cve": None,
            "retrieval_diagnosis": "not_applicable", "reason": f"judge error: {last}"}


class CachingJudge:
    """Judge each distinct (id, answer_text) once."""

    def __init__(self):
        self.cache: dict[tuple[str, str], dict] = {}
        self.calls = 0

    def judge(self, record: dict, gt: dict) -> dict:
        key = (record["id"], record.get("answer_text") or "")
        if key not in self.cache:
            self.cache[key] = judge_record(record, gt)
            self.calls += 1
        return self.cache[key]
