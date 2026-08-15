"""Add MITRE CVE-registry state to every item's ground truth in dataset.json.

Why: "fake" items were built as absent-from-NVD, but a CVE ID can be allocated
(RESERVED) in MITRE's registry with no public details and no NVD entry. A
search-on model that reports "this ID is reserved" is being accurate, and the
judge must not grade that as fabrication. Registry state is therefore part of
ground truth:
  PUBLISHED / RESERVED / REJECTED  = allocated in the registry (cve.org)
  ABSENT                           = never allocated (registry 404)

Idempotent; re-running refreshes the states in place.

  python dataset/enrich_registry_state.py
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C


def registry_state(cve_id: str) -> str:
    req = urllib.request.Request(
        f"https://cveawg.mitre.org/api/cve-id/{cve_id}",
        headers={"User-Agent": "nlp-course-research-script"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read()).get("state", "UNKNOWN")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return "ABSENT"
            time.sleep(2 * (attempt + 1))
        except Exception:
            time.sleep(2 * (attempt + 1))
    return "UNKNOWN"


def main():
    data = json.loads(C.DATASET.read_text())
    counts: dict[tuple[str, str], int] = {}
    broken = []
    for it in data["items"]:
        state = registry_state(it["id"])
        it["ground_truth"]["registry_state"] = state
        counts[(it["category"], state)] = counts.get((it["category"], state), 0) + 1
        # A "fake" item whose ID has since been PUBLISHED is no longer a valid
        # fake probe; analysis must exclude it.
        if it["category"] in C.FAKE_CATEGORIES and state == "PUBLISHED":
            broken.append(it["id"])
        print(f"  {it['id']}: {state}", flush=True)
        time.sleep(0.3)
    data["registry_checked"] = time.strftime("%Y-%m-%d", time.gmtime())
    C.DATASET.write_text(json.dumps(data, indent=1) + "\n")

    print("\ncategory x registry_state:")
    for (cat, state), n in sorted(counts.items()):
        print(f"  {cat:16s} {state:10s} {n}")
    if broken:
        print(f"\nBROKEN fake items (now PUBLISHED, exclude from analysis): {broken}")


if __name__ == "__main__":
    main()
