"""LLM-as-judge (config.JUDGE_MODEL, currently gemini-3.7-flash): meaning-based labels only.

The judge classifies the model's answer AGAINST the NVD ground truth we hand it.
It never decides what is true -- truth is given (NVD). It picks one label from a
fixed list plus a one-line reason. Identical answers are judged once (cache).
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from pathlib import Path

from google import genai
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from harness.gemini_client import _QUOTA_MARKERS

_client: genai.Client | None = None
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

# "declined" = the model abstained (said it couldn't find reliable info). It is a
# decisive judgment about the MODEL's behavior, NOT the judge being uncertain --
# hence the explicit name (was "unsure", which read as judge-uncertainty).
# NOTE: the judge-FAILURE fallbacks below still return the sentinel "unsure"
# (outside this enum) to mean "no valid verdict"; score.py distinguishes the two
# by the reason text and also relabels any legacy cached "unsure" declines.
# Bumped whenever the judge prompt semantics change: cached verdicts from an
# older prompt version are ignored on load (the cache key is only
# (id, answer_text) and knows nothing about the prompt).
PROMPT_VERSION = 2

REAL_LABELS = ["correct", "wrong", "declined"]
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
        state = gt.get("registry_state", "ABSENT")
        if state in ("RESERVED", "REJECTED"):
            return ("GROUND TRUTH (authoritative, from NVD + the MITRE CVE registry): "
                    "This CVE ID has NO entry in the National Vulnerability Database "
                    f"and NO public vulnerability details exist. The ID is however "
                    f"allocated in the MITRE CVE registry in state {state} "
                    "(reserved-but-unpublished). An answer that reports the ID as "
                    f"{state.lower()}/unpublished with no substantive details is ACCURATE.")
        return ("GROUND TRUTH (authoritative, from NVD + the MITRE CVE registry): "
                "This CVE ID does NOT exist in the National Vulnerability Database "
                "and has never been allocated in the MITRE CVE registry. "
                "There is no such vulnerability and no such ID.")
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
        "  declined= the model declined / said it could not find reliable info.\n"
        "            This includes an answer that only claims the ID is reserved /\n"
        "            not yet published and gives no contradicting details: that is\n"
        "            a failed retrieval (declined), NOT wrong.")
    fake_rules = (
        "This CVE ID has no public vulnerability details (see ground truth). "
        "Choose one label:\n"
        "  rejected   = the model provides NO substantive vulnerability details: it says\n"
        "               it cannot find / does not recognize the ID, or accurately reports\n"
        "               the ID as reserved/rejected/unpublished\n"
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
    """Close an object the judge left unterminated. Defensive fallback only:
    the API-enforced response schema should make this unreachable, but the old
    pro-preview judge dropped trailing braces even with finish_reason=STOP."""
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


class QuotaExhausted(RuntimeError):
    """The judge hit a quota that retrying in-process cannot fix -- a per-day
    request cap or a billing/credits limit -- so we abort the run cleanly
    instead of grinding every record through futile retries."""


def _is_hard_quota(msg: str) -> bool:
    """Daily-cap 429s (old pro-preview judge) OR billing/credit 429s (the
    realistic failure for a flash judge on depleted credits). The billing
    markers are shared with the subject-model client so the two never drift."""
    m = msg.lower()
    if "per_day" in m or "per model per day" in m or ("quota" in m and "per day" in m):
        return True
    return any(marker in m for marker in _QUOTA_MARKERS)


def _is_failed(verdict: dict) -> bool:
    r = (verdict.get("reason") or "")
    return r.startswith("judge error") or "unparseable" in r


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
            if _is_hard_quota(last):
                raise QuotaExhausted(last) from e  # backoff can't beat a daily/billing cap
            time.sleep(2 * (attempt + 1))
    return {"label": "unsure", "hijacked_cve": None,
            "retrieval_diagnosis": "not_applicable", "reason": f"judge error: {last}"}


class CachingJudge:
    """Judge each distinct (id, answer_text) once.

    With cache_path set, judgments persist to a JSONL: loaded on init, each new
    judgment appended+flushed immediately. This makes a scoring run RESUMABLE --
    an interrupted pass reloads every judgment it already paid for instead of
    re-judging from scratch -- and the growing file is a live progress signal
    (wc -l). Rules/CSV assembly are free and recomputed each run; only the paid
    judge calls are cached to disk.
    """

    def __init__(self, cache_path=None):
        self.cache: dict[tuple[str, str], dict] = {}
        self.calls = 0   # new judge API calls made this run
        self.hits = 0    # served from cache (memory or disk)
        self._lock = threading.Lock()  # guards cache + counters + file append
        self._path = Path(cache_path) if cache_path else None
        self._fh = None
        if self._path and self._path.exists():
            for line in self._path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r.get("pv", 1) == PROMPT_VERSION:
                        self.cache[(r["id"], r["answer_text"])] = r["judged"]
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self._path.open("a")  # append: keep prior judgments

    def judge(self, record: dict, gt: dict) -> dict:
        key = (record["id"], record.get("answer_text") or "")
        with self._lock:
            if key in self.cache:
                self.hits += 1
                return self.cache[key]
        # The slow network call runs OUTSIDE the lock, so many judgments proceed
        # concurrently; only the brief cache/file mutation below is serialized.
        judged = judge_record(record, gt)  # may raise QuotaExhausted -> propagates
        with self._lock:
            self.calls += 1
            if key in self.cache:
                # Another thread judged this same (id, answer) while we were in
                # flight -> keep the first verdict, drop this duplicate.
                return self.cache[key]
            if _is_failed(judged):
                # Transient failure (API error / unparseable): do NOT cache it, so
                # a later run retries instead of baking the failure in permanently.
                return judged
            self.cache[key] = judged
            if self._fh:
                self._fh.write(json.dumps(
                    {"id": key[0], "answer_text": key[1], "judged": judged,
                     "pv": PROMPT_VERSION}) + "\n")
                self._fh.flush()  # survive a crash / be visible live
        return judged
