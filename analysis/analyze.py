"""Compute RQ1-RQ4 from a run's labels.csv -> printed tables + its figures/.

Rates with raw counts shown, per model, per the proposal's methodology.
  python analysis/analyze.py             # the CURRENT run
  python analysis/analyze.py --run <id>  # an older run
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C


KEY = ["id", "model", "condition", "repeat"]
# Mirrors scoring.judge.DIAGNOSES (not imported: that module drags in the genai
# SDK). manual_diagnosis is only an override when it is one of these values --
# in practice the column doubles as a free-text notes field.
VALID_DIAGNOSES = {"retrieved_nothing", "retrieved_wrong_cve",
                   "retrieved_truth_ignored", "not_applicable"}


def apply_manual_overrides(df: pd.DataFrame, rd) -> pd.DataFrame:
    """Human verdicts beat the judge: rows of the run's handcheck_sample.csv
    with a non-empty manual_label override judge_label (and manual_diagnosis,
    if given, overrides retrieval_diagnosis). This is the ONE place hand
    verdicts enter the analysis -- labels.csv itself is never hand-edited,
    so a re-score cannot silently drop them."""
    sample_csv = rd / "handcheck_sample.csv"
    if not sample_csv.exists():
        return df
    hc = pd.read_csv(sample_csv).fillna("")
    hc = hc[hc["manual_label"].astype(str).str.strip() != ""]
    if hc.empty:
        return df
    df = df.set_index(KEY).sort_index()
    hc = hc.set_index(KEY)
    n_lbl = n_diag = 0
    for key, row in hc.iterrows():
        if key not in df.index:
            continue
        df.loc[key, "judge_label"] = str(row["manual_label"]).strip().lower()
        n_lbl += 1
        diag = str(row.get("manual_diagnosis", "")).strip().lower()
        if diag in VALID_DIAGNOSES:
            df.loc[key, "retrieval_diagnosis"] = diag
            n_diag += 1
    print(f"applied {n_lbl} manual label override(s) "
          f"({n_diag} with diagnosis) from {sample_csv.name}")
    return df.reset_index()


def good_outcome(row) -> bool:
    """The 'right' behavior: correct for real IDs, rejected for fake IDs."""
    if row["category"] in C.REAL_CATEGORIES:
        return row["judge_label"] == "correct"
    return row["judge_label"] == "rejected"


def load(rd) -> pd.DataFrame:
    df = pd.read_csv(rd / "labels.csv")
    df = apply_manual_overrides(df, rd)
    # A "fake" ID that has since been PUBLISHED in the CVE registry is no longer
    # a valid fake probe (the model can legitimately find real details for it).
    if "registry_state" in df.columns:
        broken = df["category"].isin(C.FAKE_CATEGORIES) & (df["registry_state"] == "PUBLISHED")
        if broken.any():
            ids = sorted(df.loc[broken, "id"].unique())
            print(f"excluding {int(broken.sum())} rows from now-PUBLISHED fake ids: {ids}")
            df = df[~broken]
    df["is_real"] = df["category"].isin(C.REAL_CATEGORIES)
    df["good"] = df.apply(good_outcome, axis=1)
    return df


# ---------------------------------------------------------------- RQ1
def rq1(df: pd.DataFrame) -> pd.DataFrame:
    """Does the model search at the right times? Compare knew-from-memory vs searched-when-on."""
    off = df[df["condition"] == "off"]
    on = df[df["condition"] == "on"]
    # Per (model, item): knew from memory? (majority of search-off repeats good)
    knew = (off.groupby(["model", "id"])["good"].mean() >= 0.5).rename("knew")
    # Per (model, item): searched when allowed? (majority of search-on repeats searched)
    searched = (on.groupby(["model", "id"])["searched"].mean() >= 0.5).rename("searched_on")
    j = pd.concat([knew, searched], axis=1).dropna().reset_index()

    rows = []
    for model, g in j.groupby("model"):
        did_not_know = g[~g["knew"]]
        knew_it = g[g["knew"]]
        rows.append({
            "model": model,
            "n_didnt_know": len(did_not_know),
            "search_when_needed_%": round(100 * did_not_know["searched_on"].mean(), 1) if len(did_not_know) else None,
            "n_knew": len(knew_it),
            "needless_search_%": round(100 * knew_it["searched_on"].mean(), 1) if len(knew_it) else None,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- RQ2
def rq2(df: pd.DataFrame):
    fakes = df[~df["is_real"]]
    # % of each fake outcome within (model, condition). Computed via an explicit
    # merge of counts/totals -- avoids groupby.apply, which in pandas 3 drops the
    # grouping columns and breaks the downstream reset_index/pivot.
    counts = (fakes.groupby(["model", "condition", "judge_label"]).size()
              .rename("n").reset_index())
    totals = fakes.groupby(["model", "condition"]).size().rename("tot").reset_index()
    dist = counts.merge(totals, on=["model", "condition"])
    dist["pct"] = (100 * dist["n"] / dist["tot"]).round(1)
    pivot = dist.pivot_table(index=["model", "condition"], columns="judge_label",
                             values="pct", fill_value=0).round(1)
    # Three-way retrieval diagnosis for fake + search-on non-rejections.
    diag_src = fakes[(fakes["condition"] == "on") & (fakes["judge_label"] != "rejected")]
    diag = (diag_src.groupby(["model", "retrieval_diagnosis"]).size()
            .rename("count").reset_index())
    return pivot, diag


# ---------------------------------------------------------------- RQ3
def rq3(df: pd.DataFrame) -> pd.DataFrame:
    reals = df[df["is_real"] & df["confidence"].notna()].copy()
    # correct / wrong / declined are distinct outcomes: a decline makes no factual
    # claim, so folding it into "wrong" would distort the comparison -- and declines
    # carry strikingly high confidence, which is itself an RQ3 finding.
    reals["outcome"] = reals["judge_label"]
    conf = (reals.groupby(["model", "outcome"])["confidence"]
            .agg(["mean", "count"]).round(1).reset_index())
    # Does lower confidence line up with choosing to search?
    on = df[(df["condition"] == "on") & df["confidence"].notna()]
    search_conf = (on.groupby(["model", "searched"])["confidence"]
                   .agg(["mean", "count"]).round(1).reset_index())
    return conf, search_conf


# ---------------------------------------------------------------- RQ4
def rq4(df: pd.DataFrame):
    tb = df[df["thinking_budget"].notna() & (~df["is_real"])]
    if tb.empty or tb["thinking_budget"].nunique() < 2:
        return None
    fab = tb.copy()
    fab["fabricated"] = fab["judge_label"].isin(["fabricated", "hijacked"])
    return (fab.groupby(["model", "thinking_budget"])["fabricated"]
            .agg(["mean", "count"]).reset_index()
            .assign(fabrication_pct=lambda d: (100 * d["mean"]).round(1)))


def plot_rq2(pivot: pd.DataFrame):
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    for cond in pivot.index.get_level_values("condition").unique():
        sub = pivot.xs(cond, level="condition")
        ax = sub.plot(kind="bar", stacked=True, figsize=(8, 5))
        ax.set_ylabel("% of fake IDs"); ax.set_title(f"Fake-ID handling (search {cond})")
        ax.legend(title="outcome", bbox_to_anchor=(1.02, 1))
        plt.tight_layout()
        plt.savefig(C.FIGURES / f"rq2_fake_handling_search_{cond}.png", dpi=120)
        plt.close()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None,
                    help="run id to analyze (default: the CURRENT run)")
    a = ap.parse_args()
    rd = C.resolve_run(a.run)
    if rd is None:
        sys.exit("no run found; pass --run <id> or start one with run_experiment.py")
    print(f"run: {rd.name}")
    # All figure/table outputs land inside the run directory.
    C.FIGURES = rd / "figures"
    df = load(rd)
    C.FIGURES.mkdir(parents=True, exist_ok=True)

    print("=" * 60, "\nRQ1 — Does the model search at the right times?")
    r1 = rq1(df); print(r1.to_string(index=False))
    r1.to_csv(C.FIGURES / "rq1_search_decision.csv", index=False)

    print("=" * 60, "\nRQ2 — Fake-ID handling (% by outcome), search off vs on")
    pivot, diag = rq2(df); print(pivot.to_string())
    pivot.to_csv(C.FIGURES / "rq2_fake_handling.csv")
    print("\nRQ2 — retrieval diagnosis (fake + search-on, non-rejections):")
    print(diag.to_string(index=False))
    diag.to_csv(C.FIGURES / "rq2_retrieval_diagnosis.csv", index=False)
    plot_rq2(pivot)

    print("=" * 60, "\nRQ3 — Confidence on correct vs wrong (real IDs)")
    conf, search_conf = rq3(df); print(conf.to_string(index=False))
    conf.to_csv(C.FIGURES / "rq3_confidence_by_outcome.csv", index=False)
    print("\nRQ3 — Confidence vs choosing to search (search-on):")
    print(search_conf.to_string(index=False))
    search_conf.to_csv(C.FIGURES / "rq3_confidence_vs_search.csv", index=False)

    r4 = rq4(df)
    print("=" * 60, "\nRQ4 — Thinking level vs fabrication (fakes)")
    if r4 is None:
        print("(no thinking-budget sweep in data; run with --thinking-budget to populate)")
    else:
        print(r4.to_string(index=False))
        r4.to_csv(C.FIGURES / "rq4_thinking_vs_fabrication.csv", index=False)

    print("\nFigures + tables ->", C.FIGURES)


if __name__ == "__main__":
    main()
