"""Merge raw runs + rule signals + judge labels -> results/labels.csv.

  python scoring/score.py                       # score everything in results/raw/
  python scoring/score.py --input results/raw/dryrun*   # just the dry run
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from scoring import rules
from scoring.judge import CachingJudge

load_dotenv(C.ROOT / ".env")

_nvd_cache: dict[str, bool] = {}


def nvd_exists(cve_id: str) -> bool | None:
    """Verify a judge-named 'hijacked' CVE really exists (truth stays with NVD)."""
    if not cve_id or not cve_id.upper().startswith("CVE-"):
        return None
    cid = cve_id.upper()
    if cid in _nvd_cache:
        return _nvd_cache[cid]
    try:
        import os
        h = {"apiKey": os.environ["NVD_API_KEY"]} if os.environ.get("NVD_API_KEY") else {}
        r = requests.get("https://services.nvd.nist.gov/rest/json/cves/2.0",
                         params={"cveId": cid}, headers=h, timeout=20)
        val = r.json().get("totalResults", 0) > 0 if r.status_code == 200 else None
    except Exception:  # noqa: BLE001
        val = None
    _nvd_cache[cid] = val
    return val


def detect_conflict(sig: dict, judged: dict, id_exists: bool) -> bool:
    label = judged.get("label")
    if id_exists:
        if label == "correct" and (sig["cvss_match"] is None and sig["product_match"] is None
                                   and sig["parse_ok"] and not sig["declined_shape"]):
            # judge called it correct but no factual claim matched NVD exactly
            return True
        if label == "wrong" and sig["cvss_match"] and sig["product_match"]:
            return True
    else:
        if label == "rejected" and not sig["declined_shape"] and sig["parse_ok"]:
            return True  # "rejected" yet the model filled factual fields
        if label in ("fabricated", "hijacked") and sig["declined_shape"]:
            return True  # "fabricated" yet the model declined to fill anything
    return False


def load_records(patterns: list[str]) -> list[dict]:
    files: list[str] = []
    for p in patterns:
        files.extend(glob.glob(p))
    recs = []
    for fp in files:
        for line in Path(fp).read_text().splitlines():
            if line.strip():
                recs.append(json.loads(line))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", default=[str(C.RAW_DIR / "*.jsonl")])
    ap.add_argument("--out", default=str(C.RESULTS / "labels.csv"))
    a = ap.parse_args()

    gt_by_id = {it["id"]: it["ground_truth"]
                for it in json.loads(C.DATASET.read_text())["items"]}
    records = load_records(a.input)
    if not records:
        sys.exit(f"no records matched {a.input}")

    judge = CachingJudge()
    rows = []
    for i, rec in enumerate(records, 1):
        gt = gt_by_id.get(rec["id"], {"exists": False})
        sig = rules.derive_signals(rec, gt)
        judged = judge.judge(rec, gt)
        hij = judged.get("hijacked_cve")
        hij_exists = nvd_exists(hij) if hij else None
        conflict = detect_conflict(sig, judged, sig["id_exists"])
        reason = (judged.get("reason") or "").lower()
        needs_hc = conflict or judged.get("label") == "unsure" or "uncertain" in reason
        rows.append({
            "id": rec["id"], "category": rec["category"], "model": rec["model"],
            "condition": rec["condition"], "repeat": rec["repeat"],
            "thinking_budget": rec.get("thinking_budget"),
            **sig,
            "judge_label": judged.get("label"),
            "hijacked_cve": hij, "hijacked_cve_exists": hij_exists,
            "retrieval_diagnosis": judged.get("retrieval_diagnosis"),
            "judge_reason": judged.get("reason"),
            "rule_judge_conflict": conflict, "needs_handcheck": needs_hc,
        })
        if i % 100 == 0:
            print(f"  scored {i}/{len(records)} (judge calls: {judge.calls})")

    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"\nWrote {len(df)} rows -> {a.out}")
    print(f"judge calls (deduped): {judge.calls} of {len(records)} records")
    print(f"conflicts flagged: {df['rule_judge_conflict'].sum()}; "
          f"needs_handcheck: {df['needs_handcheck'].sum()}")


if __name__ == "__main__":
    main()
