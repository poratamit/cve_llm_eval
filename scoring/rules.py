"""Deterministic scoring signals from exact comparison against stored NVD truth.

Rules NEVER guess and NEVER fact-check the world -- they only compare the
model's structured claims to the answer key already saved in dataset.json.
A conclusive match is asserted; anything ambiguous is left for the judge
(the *_match fields become None = "escalate", never a silent "wrong").
"""
from __future__ import annotations

import re


def _norm(s: str) -> str:
    return re.sub(r"[\s_\-]+", " ", str(s).lower()).strip()


def _as_float(x) -> float | None:
    try:
        return round(float(x), 1)
    except (TypeError, ValueError):
        return None


def cvss_signal(parsed: dict, gt: dict):
    """True = exact match to a published score; False->None escalate; None = no claim."""
    if not parsed:
        return None
    score = _as_float(parsed.get("cvss_score"))
    if score is None:
        return None  # no claim
    published = set(gt.get("cvss_scores", []))
    return True if score in published else None  # non-match -> judge, not auto-wrong


def severity_signal(parsed: dict, gt: dict):
    if not parsed:
        return None
    sev = parsed.get("severity")
    if not sev:
        return None
    sev = str(sev).strip().upper()
    published = {s.upper() for s in gt.get("severities", [])}
    return True if sev in published else None


def product_signal(parsed: dict, gt: dict):
    """True = at least one claimed product clearly matches an NVD product/vendor.

    Non-match with a claim present -> None (escalate to judge). No claim -> None.
    Substring matching only, on tokens of length >= 3; no fuzzy matching.
    """
    if not parsed:
        return None
    claims = parsed.get("affected_products")
    if not claims:
        return None
    if isinstance(claims, str):
        claims = [claims]
    claim_norms = [_norm(c) for c in claims if c]
    if not claim_norms:
        return None
    truth_tokens = {_norm(t) for t in (gt.get("products", []) + gt.get("vendors", []))}
    truth_tokens = {t for t in truth_tokens if len(t) >= 3}
    for cn in claim_norms:
        for tt in truth_tokens:
            if tt in cn or cn in tt:
                return True
    return None  # claimed products, none matched -> judge decides


def declined_shape(parsed: dict) -> bool:
    """All four factual fields empty -> a strong (but judge-confirmed) rejection signal."""
    if not parsed:
        return False
    return all(not parsed.get(k) for k in
               ("affected_products", "cvss_score", "severity", "mitigation"))


def derive_signals(record: dict, gt: dict) -> dict:
    """All deterministic facts for one raw record."""
    parsed = record.get("parsed")
    return {
        "id_exists": bool(gt.get("exists")),
        "searched": bool(record.get("searched")),
        "declined_shape": declined_shape(parsed),
        "cvss_match": cvss_signal(parsed, gt),
        "severity_match": severity_signal(parsed, gt),
        "product_match": product_signal(parsed, gt),
        "confidence": (parsed or {}).get("confidence_0_100"),
        "parse_ok": parsed is not None,
    }
