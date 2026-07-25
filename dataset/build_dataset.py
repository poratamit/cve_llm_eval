"""One-time generator for dataset/dataset.json.

The DELIVERABLE is dataset.json; this script is the means to produce it
verifiably (and its provenance). It is NOT part of the experiment pipeline and
uses NO LLM API -- only the free NVD REST API.

Run once:  python dataset/build_dataset.py
Verify:    python dataset/build_dataset.py --verify

Four item categories (target 30 each):
  real/known    -- famous CVEs 2019-2023, well represented in training data
  real/recent   -- CVEs published after the models' training cutoff (2026)
  fake/random   -- format-valid IDs verified ABSENT from NVD
  fake/near-miss-- one digit off a real CVE, verified ABSENT from NVD
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
OUT = Path(__file__.replace("build_dataset.py", "dataset.json"))
API_KEY = os.environ.get("NVD_API_KEY") or None
# NVD rate limit: 5 req/30s without key, 50 with. Sleep conservatively.
SLEEP = 1.0 if API_KEY else 6.5

# Seed of famous CVEs (2019-2023) with strong training-data representation.
# Over-provisioned so we can still reach 30 if a few lack clean ground truth.
KNOWN_SEED = [
    "CVE-2021-44228", "CVE-2021-45046", "CVE-2019-0708", "CVE-2020-1472",
    "CVE-2021-34527", "CVE-2021-26855", "CVE-2022-30190", "CVE-2021-3156",
    "CVE-2022-22965", "CVE-2019-19781", "CVE-2020-0601", "CVE-2021-21972",
    "CVE-2022-1388", "CVE-2023-23397", "CVE-2023-34362", "CVE-2023-4863",
    "CVE-2022-41040", "CVE-2022-41082", "CVE-2021-26084", "CVE-2022-26134",
    "CVE-2020-5902", "CVE-2019-11510", "CVE-2021-40438", "CVE-2021-22986",
    "CVE-2023-27997", "CVE-2023-2868", "CVE-2022-0847", "CVE-2021-4034",
    "CVE-2020-14882", "CVE-2021-1675", "CVE-2023-20198", "CVE-2023-3519",
    "CVE-2023-4966", "CVE-2022-3602", "CVE-2020-0796",
]

TARGET_PER_CATEGORY = 30


def _headers() -> dict:
    return {"apiKey": API_KEY} if API_KEY else {}


def nvd_get(params: dict, retries: int = 4) -> dict:
    """GET the NVD API with retry/backoff; returns parsed JSON."""
    for attempt in range(retries):
        try:
            r = requests.get(NVD_URL, params=params, headers=_headers(), timeout=30)
            if r.status_code == 200:
                return r.json()
            # 403/503 = rate limited / transient; back off.
            time.sleep(SLEEP * (attempt + 2))
        except requests.RequestException:
            time.sleep(SLEEP * (attempt + 2))
    raise RuntimeError(f"NVD request failed after {retries} tries: {params}")


def lookup(cve_id: str) -> dict | None:
    """Return the raw `cve` object for an ID, or None if it does not exist."""
    data = nvd_get({"cveId": cve_id})
    time.sleep(SLEEP)
    if data.get("totalResults", 0) == 0:
        return None
    return data["vulnerabilities"][0]["cve"]


def extract_ground_truth(cve: dict) -> dict:
    """Pull the answer-key fields we score against."""
    desc = next((d["value"] for d in cve.get("descriptions", []) if d["lang"] == "en"), "")

    cvss_scores: list[float] = []
    severities: set[str] = set()
    metrics = cve.get("metrics", {})
    for key in ("cvssMetricV40", "cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        for entry in metrics.get(key, []):
            cd = entry.get("cvssData", {})
            if cd.get("baseScore") is not None:
                cvss_scores.append(round(float(cd["baseScore"]), 1))
            sev = cd.get("baseSeverity") or entry.get("baseSeverity")
            if sev:
                severities.add(sev.upper())

    products: set[str] = set()
    vendors: set[str] = set()
    for cfg in cve.get("configurations", []):
        for node in cfg.get("nodes", []):
            for cm in node.get("cpeMatch", []):
                parts = cm.get("criteria", "").split(":")
                if len(parts) > 5:
                    vendors.add(parts[3])
                    products.add(parts[4])

    return {
        "exists": True,
        "description": desc,
        "published": cve.get("published"),
        "cvss_scores": sorted(set(cvss_scores)),
        "severities": sorted(severities),
        "products": sorted(products),
        "vendors": sorted(vendors),
        "reference_count": len(cve.get("references", [])),
    }


def build_real_known() -> list[dict]:
    items = []
    for cid in KNOWN_SEED:
        if len(items) >= TARGET_PER_CATEGORY:
            break
        cve = lookup(cid)
        if cve is None:
            print(f"  ! {cid} unexpectedly absent, skipping", file=sys.stderr)
            continue
        gt = extract_ground_truth(cve)
        if not gt["cvss_scores"]:
            print(f"  ! {cid} has no CVSS, skipping", file=sys.stderr)
            continue
        items.append({"id": cid, "category": "real_known", "ground_truth": gt,
                      "provenance": {"source": "nvd cveId lookup", "seed_list": True}})
        print(f"  + real_known {cid} cvss={gt['cvss_scores']}")
    return items


def build_real_recent() -> list[dict]:
    """CVEs published in the post-cutoff window, with usable ground truth."""
    items: list[dict] = []
    params = {
        "pubStartDate": "2026-05-01T00:00:00.000",
        "pubEndDate": "2026-07-17T00:00:00.000",
        "resultsPerPage": 2000,
    }
    data = nvd_get(params)
    time.sleep(SLEEP)
    vulns = data.get("vulnerabilities", [])
    random.shuffle(vulns)
    for wrap in vulns:
        if len(items) >= TARGET_PER_CATEGORY:
            break
        cve = wrap["cve"]
        gt = extract_ground_truth(cve)
        # Require CVSS + a substantive English description for a scorable item.
        if not gt["cvss_scores"] or len(gt["description"]) < 60:
            continue
        items.append({"id": cve["id"], "category": "real_recent", "ground_truth": gt,
                      "provenance": {"source": "nvd pubDate range 2026-05..2026-07"}})
        print(f"  + real_recent {cve['id']} pub={gt['published'][:10]}")
    return items


def build_fake_random(rng: random.Random) -> list[dict]:
    items = []
    tried = set()
    while len(items) < TARGET_PER_CATEGORY:
        year = rng.randint(2019, 2024)
        num = rng.randint(40000, 99999)
        cid = f"CVE-{year}-{num}"
        if cid in tried:
            continue
        tried.add(cid)
        if lookup(cid) is None:  # verified absent
            items.append({"id": cid, "category": "fake_random",
                          "ground_truth": {"exists": False},
                          "provenance": {"source": "generated; verified absent from NVD"}})
            print(f"  + fake_random {cid} (verified absent)")
    return items


def build_fake_near_miss(real_ids: list[str], rng: random.Random) -> list[dict]:
    items = []
    pool = list(real_ids)
    rng.shuffle(pool)
    i = 0
    while len(items) < TARGET_PER_CATEGORY and i < len(pool) * 20:
        base = pool[i % len(pool)]
        i += 1
        year, seq = base.split("-")[1], base.split("-")[2]
        # Flip one digit of the sequence number to a different digit.
        pos = rng.randrange(len(seq))
        new_digit = rng.choice([d for d in "0123456789" if d != seq[pos]])
        new_seq = seq[:pos] + new_digit + seq[pos + 1:]
        cid = f"CVE-{year}-{new_seq}"
        if cid == base or any(it["id"] == cid for it in items):
            continue
        if lookup(cid) is None:  # verified absent
            items.append({"id": cid, "category": "fake_near_miss",
                          "ground_truth": {"exists": False},
                          "provenance": {"source": "one digit off a real CVE; verified absent",
                                         "near": base}})
            print(f"  + fake_near_miss {cid} (near {base}, verified absent)")
    return items


def build() -> None:
    rng = random.Random(20260717)
    print("[1/4] real_known ...")
    known = build_real_known()
    print("[2/4] real_recent ...")
    recent = build_real_recent()
    print("[3/4] fake_random ...")
    fake_r = build_fake_random(rng)
    print("[4/4] fake_near_miss ...")
    real_ids = [it["id"] for it in known + recent]
    fake_n = build_fake_near_miss(real_ids, rng)

    items = known + recent + fake_r + fake_n
    out = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "nvd_api_key_used": bool(API_KEY),
        "counts": {c: sum(1 for it in items if it["category"] == c)
                   for c in ("real_known", "real_recent", "fake_random", "fake_near_miss")},
        "items": items,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {len(items)} items -> {OUT}")
    print("counts:", out["counts"])


def verify() -> None:
    data = json.loads(OUT.read_text())
    fakes = [it for it in data["items"] if not it["ground_truth"]["exists"]]
    reals = [it for it in data["items"] if it["ground_truth"]["exists"]]
    print(f"Verifying {len(fakes)} fakes are absent and {len(reals)} reals have ground truth...")
    bad = 0
    for it in reals:
        gt = it["ground_truth"]
        if not gt["cvss_scores"] or not gt["description"]:
            print(f"  ! {it['id']} missing ground truth"); bad += 1
    for it in fakes:
        if lookup(it["id"]) is not None:
            print(f"  ! {it['id']} UNEXPECTEDLY EXISTS in NVD"); bad += 1
    print("OK" if bad == 0 else f"{bad} PROBLEMS")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    verify() if args.verify else build()
