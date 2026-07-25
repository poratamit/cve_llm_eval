"""Thin wrapper over google-genai: one call, uniform result dict.

The only behavioural switch is search on/off. We NEVER instruct the model to
search or not to search -- with search "on" the model decides for itself, and
whether it grounded is read back from grounding_metadata. That decision is the
object of study (RQ1).
"""
from __future__ import annotations

import json
import os
import re
import time

from google import genai
from google.genai import types

_client: genai.Client | None = None


def client() -> genai.Client:
    global _client
    if _client is None:
        key = os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY not set (see .env.example)")
        _client = genai.Client(api_key=key)
    return _client


def _build_config(search_enabled: bool, thinking_budget: int | None) -> types.GenerateContentConfig | None:
    kwargs: dict = {}
    if search_enabled:
        kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
    if thinking_budget is not None:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
    return types.GenerateContentConfig(**kwargs) if kwargs else None


def _extract_grounding(resp) -> dict:
    """Read whether the model searched and what it retrieved."""
    queries: list[str] = []
    chunks: list[dict] = []
    try:
        gm = resp.candidates[0].grounding_metadata
    except (AttributeError, IndexError, TypeError):
        gm = None
    if gm is not None:
        queries = list(getattr(gm, "web_search_queries", None) or [])
        for ch in (getattr(gm, "grounding_chunks", None) or []):
            web = getattr(ch, "web", None)
            if web is not None:
                chunks.append({"uri": getattr(web, "uri", None),
                               "title": getattr(web, "title", None)})
    return {"searched": bool(queries or chunks),
            "web_search_queries": queries,
            "grounding_chunks": chunks}


def _usage(resp) -> dict:
    um = getattr(resp, "usage_metadata", None)
    if um is None:
        return {}
    return {
        "prompt_tokens": getattr(um, "prompt_token_count", None),
        "candidates_tokens": getattr(um, "candidates_token_count", None),
        "thoughts_tokens": getattr(um, "thoughts_token_count", None),
        "total_tokens": getattr(um, "total_token_count", None),
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
    config = _build_config(search_enabled, thinking_budget)
    last_err = None
    for attempt in range(retries):
        try:
            resp = client().models.generate_content(
                model=model, contents=prompt, config=config)
            text = getattr(resp, "text", None) or ""
            parsed, perr = parse_answer(text)
            grounding = _extract_grounding(resp)
            return {
                "ok": True,
                "answer_text": text,
                "parsed": parsed,
                "parse_error": perr,
                **grounding,
                "usage": _usage(resp),
                "error": None,
            }
        except Exception as e:  # transient API errors: back off and retry
            last_err = str(e)
            time.sleep(2 * (attempt + 1))
    return {"ok": False, "answer_text": "", "parsed": None,
            "parse_error": None, "searched": False, "web_search_queries": [],
            "grounding_chunks": [], "usage": {}, "error": last_err}
