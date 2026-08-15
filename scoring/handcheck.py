"""Human validation of the auto-labels (proposal step 4).

  export:    python scoring/handcheck.py export [--run ID]
             -> <run>/handcheck_sample.csv  (fill in manual_label / manual_diagnosis)
  agreement: python scoring/handcheck.py agreement [--run ID]
             -> prints % agreement between judge and your manual labels

Operates on the CURRENT run's labels.csv (or --run ID), same as score.py
and analyze.py -- NOT the stale top-level results/labels.csv.

Sized for one person: --budget N rows total (default 40, ~20-30 min).
Every flagged row (needs_handcheck = rule/judge conflicts + judge failures)
is force-included; ~60% of the remaining budget samples the hardest subset
(fake IDs not rejected under search-on, which carry the three-way retrieval
diagnosis, spread across models) and the rest is a thin stratified slice of
everything else. Model declines ("declined") are a clean label and stratified
like any other, not force-included.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

SEED = 20260717
BUDGET = 40          # total rows to hand-check (approximate; forced rows always kept)
HARD_SHARE = 0.6     # share of the post-forced budget spent on the hard subset


def _paths(run_id):
    rd = C.resolve_run(run_id)
    if rd is None:
        sys.exit("no run found (start one with run_experiment.py) or pass --run <id>")
    print(f"run: {rd.name}")
    return rd / "labels.csv", rd / "handcheck_sample.csv"


def _proportional(pool: pd.DataFrame, n: int, by: list[str]) -> pd.DataFrame:
    """~n rows from pool, allocated proportionally across `by` groups.
    Iterate the groups rather than groupby.apply: pandas 3 excludes the
    grouping columns from the frame passed to apply, which silently blanked
    category/condition/judge_label on every sampled row (and so blanked the
    very columns `agreement()` compares)."""
    if n <= 0 or pool.empty:
        return pool.iloc[:0]
    frac = min(1.0, n / len(pool))
    picks = []
    for _, g in pool.groupby(by, dropna=False):
        k = min(len(g), round(len(g) * frac))
        if k:
            picks.append(g.sample(n=k, random_state=SEED))
    return pd.concat(picks) if picks else pool.iloc[:0]


def export(labels_csv, sample_csv, budget=BUDGET):
    df = pd.read_csv(labels_csv)
    forced = df[df["needs_handcheck"]]
    remaining = max(0, budget - len(forced))
    # The genuinely hard subset the proposal targets: fake IDs the model did
    # NOT reject under search-on -> the fabricated/hijacked cases that carry
    # the three-way retrieval diagnosis. Sampled (spread across models), not
    # taken wholesale -- it alone is bigger than a single person's budget.
    hard_pool = df[(df["category"].isin(C.FAKE_CATEGORIES)) & (df["condition"] == "on")
                   & (~df["judge_label"].isin(["rejected"])) ].drop(forced.index, errors="ignore")
    hard = _proportional(hard_pool, round(remaining * HARD_SHARE), ["model", "judge_label"])
    # Thin stratified slice of everything else with the leftover budget.
    rest = df.drop(forced.index).drop(hard_pool.index, errors="ignore")
    strat = _proportional(rest, remaining - len(hard),
                          ["category", "condition", "judge_label"])

    sample = pd.concat([forced, hard, strat]).drop_duplicates(subset=["id", "model", "condition", "repeat"])
    sample = sample.sample(frac=1.0, random_state=SEED)  # shuffle so labeler is blind to grouping
    sample["manual_label"] = ""
    sample["manual_diagnosis"] = ""
    cols = ["id", "category", "model", "condition", "repeat", "searched",
            "judge_label", "retrieval_diagnosis", "judge_reason",
            "rule_judge_conflict", "manual_label", "manual_diagnosis"]
    sample[cols].to_csv(sample_csv, index=False)
    pct = 100 * len(sample) / len(df)
    print(f"Wrote {len(sample)} rows ({pct:.0f}% of {len(df)}) -> {sample_csv}")
    print("Fill in manual_label (and manual_diagnosis for fake+search rows), then:")
    print("  python scoring/handcheck.py agreement")


def agreement(sample_csv):
    df = pd.read_csv(sample_csv).fillna("")
    labeled = df[df["manual_label"].astype(str).str.strip() != ""]
    if labeled.empty:
        sys.exit("No manual_label filled in yet.")
    labeled = labeled.copy()
    labeled["agree"] = (labeled["manual_label"].str.strip().str.lower()
                        == labeled["judge_label"].str.strip().str.lower())
    overall = 100 * labeled["agree"].mean()
    print(f"Overall label agreement: {overall:.1f}% (n={len(labeled)})\n")
    print("By judge_label:")
    for lbl, g in labeled.groupby("judge_label"):
        print(f"  {lbl:12s} {100*g['agree'].mean():5.1f}%  (n={len(g)})")

    diag = labeled[(labeled["category"].isin(C.FAKE_CATEGORIES))
                   & (labeled["manual_diagnosis"].str.strip() != "")].copy()
    if not diag.empty:
        diag["dagree"] = (diag["manual_diagnosis"].str.strip().str.lower()
                          == diag["retrieval_diagnosis"].astype(str).str.strip().str.lower())
        print(f"\nThree-way retrieval diagnosis agreement: "
              f"{100*diag['dagree'].mean():.1f}% (n={len(diag)})")
        if diag["dagree"].mean() < 0.8:
            print("  -> LOW: hand-label the full fake+search subset (proposal step 4).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="export", choices=["export", "agreement"])
    ap.add_argument("--run", default=None, help="run id (default: the CURRENT run)")
    ap.add_argument("--budget", type=int, default=BUDGET,
                    help=f"approx. total rows to hand-check (default {BUDGET})")
    a = ap.parse_args()
    labels_csv, sample_csv = _paths(a.run)
    export(labels_csv, sample_csv, a.budget) if a.mode == "export" else agreement(sample_csv)
