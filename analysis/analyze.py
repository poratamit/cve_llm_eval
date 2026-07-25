"""Compute RQ1-RQ4 from results/labels.csv -> printed tables + results/figures/.

Rates with raw counts shown, per model, per the proposal's methodology.
  python analysis/analyze.py
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


def good_outcome(row) -> bool:
    """The 'right' behavior: correct for real IDs, rejected for fake IDs."""
    if row["category"] in C.REAL_CATEGORIES:
        return row["judge_label"] == "correct"
    return row["judge_label"] == "rejected"


def load() -> pd.DataFrame:
    df = pd.read_csv(C.RESULTS / "labels.csv")
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
    dist = (fakes.groupby(["model", "condition", "judge_label"]).size()
            .groupby(level=[0, 1]).apply(lambda s: 100 * s / s.sum())
            .rename("pct").reset_index())
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
    reals["outcome"] = reals["judge_label"].map(
        lambda x: "correct" if x == "correct" else "wrong/unsure")
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
    df = load()
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
