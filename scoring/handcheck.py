"""Human validation of the auto-labels (proposal step 4).

  export:    python scoring/handcheck.py export
             -> results/handcheck_sample.csv  (fill in manual_label / manual_diagnosis)
  agreement: python scoring/handcheck.py agreement
             -> prints % agreement between judge and your manual labels

Sampling targets ~15-20% overall, but force-includes every flagged row
(needs_handcheck = rule/judge conflicts + any judge failure) and over-samples
the hardest subset: fake IDs answered with search on (the three-way retrieval
diagnosis). Model declines ("declined") are a clean label and stratified like
any other, not force-included.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

LABELS_CSV = C.RESULTS / "labels.csv"
SAMPLE_CSV = C.RESULTS / "handcheck_sample.csv"
SEED = 20260717
FRACTION = 0.18


def export():
    df = pd.read_csv(LABELS_CSV)
    forced = df[df["needs_handcheck"]]
    # Over-sample only the genuinely hard subset the proposal targets: fake IDs the
    # model did NOT reject under search-on -> these are the fabricated/hijacked cases
    # that carry the three-way retrieval diagnosis. (Taking *all* fake+search-on rows
    # ballooned the sample to ~43%; rejections are the easy, high-agreement case.)
    hard = df[(df["category"].isin(C.FAKE_CATEGORIES)) & (df["condition"] == "on")
              & (~df["judge_label"].isin(["rejected"]))]
    rest = df.drop(forced.index)

    # ~18% of the rest, stratified by (category, condition, judge_label).
    # Iterate the groups rather than groupby.apply: pandas 3 excludes the
    # grouping columns from the frame passed to apply, which silently blanked
    # category/condition/judge_label on every sampled row (and so blanked the
    # very columns `agreement()` compares).
    picks = []
    for _, g in rest.groupby(["category", "condition", "judge_label"], dropna=False):
        n = max(1, round(len(g) * FRACTION))
        picks.append(g.sample(n=min(n, len(g)), random_state=SEED))
    strat = pd.concat(picks) if picks else rest.iloc[:0]

    sample = pd.concat([forced, hard, strat]).drop_duplicates(subset=["id", "model", "condition", "repeat"])
    sample = sample.sample(frac=1.0, random_state=SEED)  # shuffle so labeler is blind to grouping
    sample["manual_label"] = ""
    sample["manual_diagnosis"] = ""
    cols = ["id", "category", "model", "condition", "repeat", "searched",
            "judge_label", "retrieval_diagnosis", "judge_reason",
            "rule_judge_conflict", "manual_label", "manual_diagnosis"]
    sample[cols].to_csv(SAMPLE_CSV, index=False)
    pct = 100 * len(sample) / len(df)
    print(f"Wrote {len(sample)} rows ({pct:.0f}% of {len(df)}) -> {SAMPLE_CSV}")
    print("Fill in manual_label (and manual_diagnosis for fake+search rows), then:")
    print("  python scoring/handcheck.py agreement")


def agreement():
    df = pd.read_csv(SAMPLE_CSV).fillna("")
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
    mode = sys.argv[1] if len(sys.argv) > 1 else "export"
    export() if mode == "export" else agreement()
