"""Thin wrapper over google-genai's Interactions API: one call, uniform result dict.

The only behavioural switch is search on/off. We NEVER instruct the model to
search or not to search -- with search "on" the model decides for itself. That
decision is the object of study (RQ1).

Why Interactions instead of generateContent: on Gemini 3.x, generateContent's
grounding_metadata (queries/chunks/supports) is populated in only a fraction of
responses that verifiably searched, which silently broke RQ1's searched flag.
Here a search is a first-class google_search_call step in the response -- the
same record Google bills per executed query -- with two independent
corroborating signals (usage.grounding_tool_count, url_citation annotations).

Calls are stateless (store=False): nothing is persisted server-side, keeping
runs independent and closer to the old generateContent semantics.
"""
from __future__ import annotations

import json
import os
import re
import time

from google import genai

_client: genai.Client | None = None


class QuotaExhausted(RuntimeError):
    """A 429 that waiting seconds will NOT fix (daily quota / billing).

    Raised instead of retrying so a sweep aborts immediately rather than
    hammering hundreds of calls x retries into a dead quota (which wastes
    minutes and, for tool-attached calls, possibly quota itself).
    """


# Substrings that mark a quota/billing 429 as opposed to a transient RPM spike
# (transient ones say "Please retry in Ns" and deserve the normal backoff).
_QUOTA_MARKERS = (
    "exceeded your current quota",
    "prepayment credits",
    "spend-based rate limit",
    "plan and billing",
)


def _is_quota_error(err: Exception) -> bool:
    msg = str(err).lower()
    return any(m in msg for m in _QUOTA_MARKERS)


def client() -> genai.Client:
    global _client
    if _client is None:
        key = os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY not set (see .env.example)")
        # Hard per-request timeout (ms): the SDK default is none, so a wedged
        # server-side call hangs a worker forever. With this, a stall surfaces
        # as an error after 3 min and the normal retry/resume machinery owns it.
        _client = genai.Client(
            api_key=key,
            http_options=genai.types.HttpOptions(timeout=180_000))
    return _client


# RQ4 knob: Interactions replaces the numeric thinking_budget with a discrete
# thinking_level. We map the harness's existing int convention onto it so
# run_experiment.py's --thinking-budget flag keeps working: 0 -> minimal
# (the old "no thinking" pole), any positive value -> high.
def _thinking_level(thinking_budget: int | None) -> str | None:
    if thinking_budget is None:
        return None
    return "minimal" if thinking_budget == 0 else "high"


def _extract_search(interaction) -> dict:
    """Read whether the model searched, from the interaction's step list.

    searched = any google_search_call step OR a nonzero grounding_tool_count
    in usage -- the two billing-backed signals. url_citation annotations are
    recorded but deliberately NOT part of the determination: models emit
    memory-recalled citation URLs even with no search tool attached (observed
    on gemini-3.5-flash answering Log4Shell/BlueKeep with search off).
    """
    queries: list[str] = []
    n_search_calls = 0
    citations: list[dict] = []
    for step in (getattr(interaction, "steps", None) or []):
        stype = getattr(step, "type", None)
        if stype == "google_search_call":
            n_search_calls += 1
            args = getattr(step, "arguments", None)
            queries.extend(getattr(args, "queries", None) or [])
        elif stype == "model_output":
            for content in (getattr(step, "content", None) or []):
                for ann in (getattr(content, "annotations", None) or []):
                    if getattr(ann, "type", None) == "url_citation":
                        citations.append({"uri": getattr(ann, "url", None),
                                          "title": getattr(ann, "title", None)})
    usage = getattr(interaction, "usage", None)
    gtc = 0
    for g in (getattr(usage, "grounding_tool_count", None) or []):
        gtc += getattr(g, "count", None) or 0
    return {
        "searched": bool(n_search_calls or gtc),
        "web_search_queries": queries,
        "grounding_chunks": citations,   # legacy key; now url_citation annotations
        "grounding_supports": [],        # legacy key; no equivalent field here
        "n_search_calls": n_search_calls,
        "grounding_tool_count": gtc,
    }


def _usage(interaction) -> dict:
    um = getattr(interaction, "usage", None)
    if um is None:
        return {}
    return {
        "prompt_tokens": getattr(um, "total_input_tokens", None),
        "candidates_tokens": getattr(um, "total_output_tokens", None),
        "thoughts_tokens": getattr(um, "total_thought_tokens", None),
        "tool_use_tokens": getattr(um, "total_tool_use_tokens", None),
        "total_tokens": getattr(um, "total_tokens", None),
    }


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_answer(text: str) -> tuple[dict | None, str | None]:
    """Lenient parse of the requested JSON. Returns (parsed, error).

    Never raises: a malformed answer still yields a raw string the judge reads.
    """
    if not text:
        return None, "empty response"
    cleaned = _FENCE.sub("", text).strip()
    # Fall back to the outermost brace span if there's surrounding prose.
    if not cleaned.startswith("{"):
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            cleaned = m.group(0)
    try:
        return json.loads(cleaned), None
    except (json.JSONDecodeError, ValueError) as e:
        return None, f"json parse failed: {e}"


def ask(cve_id: str, prompt: str, model: str, search_enabled: bool,
        thinking_budget: int | None = None, retries: int = 3) -> dict:
    """Single call. Returns a self-contained result dict (one raw log line)."""
    kwargs: dict = {"model": model, "input": prompt, "store": False}
    if search_enabled:
        kwargs["tools"] = [{"type": "google_search"}]
    level = _thinking_level(thinking_budget)
    if level is not None:
        kwargs["generation_config"] = {"thinking_level": level}
    last_err = None
    for attempt in range(retries):
        try:
            interaction = client().interactions.create(**kwargs)
            text = getattr(interaction, "output_text", None) or ""
            parsed, perr = parse_answer(text)
            search = _extract_search(interaction)
            return {
                "ok": True,
                "answer_text": text,
                "parsed": parsed,
                "parse_error": perr,
                **search,
                "usage": _usage(interaction),
                "interaction_status": getattr(interaction, "status", None),
                "error": None,
            }
        except Exception as e:
            if _is_quota_error(e):
                raise QuotaExhausted(str(e)) from e
            # transient API errors: back off and retry
            last_err = str(e)
            time.sleep(2 * (attempt + 1))
    return {"ok": False, "answer_text": "", "parsed": None,
            "parse_error": None, "searched": False, "web_search_queries": [],
            "grounding_chunks": [], "grounding_supports": [],
            "n_search_calls": 0, "grounding_tool_count": 0,
            "usage": {}, "interaction_status": None, "error": last_err}
