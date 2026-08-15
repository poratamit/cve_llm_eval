"""Cross-validate the searched flag against an outcome proxy, per run.

Proxy: a model that answers a post-cutoff CVE-2026-* item CORRECTLY under
search-on must have searched -- the ID did not exist at training time. So on
that subset the searched flag's hit rate is a lower bound on its recall.

Also reports (a) the search-off converse (correct-on-2026 with search off
should be ~0, validating the proxy itself), and (b) agreement between the
three Interactions-API signals (google_search_call steps, the billed
grounding_tool_count, url_citation annotations).

  python analysis/validate_search_signal.py             # the CURRENT run
  python analysis/validate_search_signal.py --run <id>  # an older run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

VIA = ["steps", "tool_count", "citations"]


def is_2026(cve_id: str) -> bool:
    return str(cve_id).upper().startswith("CVE-2026-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None,
                    help="run id to validate (default: the CURRENT run)")
    a = ap.parse_args()
    rd = C.resolve_run(a.run)
    if rd is None:
        sys.exit("no run found; pass --run <id> or start one with run_experiment.py")
    print(f"run: {rd.name}")

    df = pd.read_csv(rd / "labels.csv")
    df = df[df["category"] == "real_recent"]
    df = df[df["id"].map(is_2026)]

    # --- proxy validity: with search OFF, 2026 items should almost never be correct
    off = df[df["condition"] == "off"]
    off_correct = off[off["judge_label"] == "correct"]
    print("Proxy validity (search OFF, CVE-2026-*): "
          f"{len(off_correct)}/{len(off)} correct (want ~0)")
    if len(off_correct):
        print(off_correct[["model", "id", "repeat"]].to_string(index=False))

    # --- recall lower bound: search ON + correct => searched must be True
    on = df[df["condition"] == "on"]
    correct = on[on["judge_label"] == "correct"]
    print("\nDetection recall on correct-on-2026 rows (searched flag hit rate):")
    for model, g in correct.groupby("model"):
        n, hit = len(g), int(g["searched"].sum())
        via = {v: int(g[f"searched_via_{v}"].sum()) for v in VIA}
        print(f"  {model:24s} {hit}/{n} = {hit/n:6.1%}   "
              f"(via steps {via['steps']}, tool_count {via['tool_count']}, "
              f"citations {via['citations']})")
    n, hit = len(correct), int(correct["searched"].sum())
    print(f"  {'ALL':24s} {hit}/{n} = {hit/n:6.1%}" if n else "  no correct rows")

    # --- signal agreement across ALL search-on rows (not just correct ones)
    print("\nSignal agreement on all search-on real_recent rows:")
    for model, g in on.groupby("model"):
        n = len(g)
        counts = {v: int(g[f"searched_via_{v}"].sum()) for v in VIA}
        both = int((g["searched_via_steps"] & g["searched_via_tool_count"]).sum())
        either = int((g["searched_via_steps"] | g["searched_via_tool_count"]).sum())
        print(f"  {model:24s} n={n:3d} steps {counts['steps']}, "
              f"tool_count {counts['tool_count']} (agree {both}/{either}), "
              f"citations {counts['citations']}")


if __name__ == "__main__":
    main()
