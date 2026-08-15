"""Merge raw runs + rule signals + judge labels -> results/labels.csv.

  python scoring/score.py                       # score everything in results/raw/
  python scoring/score.py --input results/raw/dryrun*   # just the dry run
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
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
from scoring.judge import CachingJudge, QuotaExhausted

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
        # product_match no longer gates conflicts: the NVD product/vendor unions
        # are huge (100s of CPEs) and substring matching is so permissive it
        # matches almost anything, so it produced neither reliable corroboration
        # nor reliable contradiction. It stays as a descriptive column only.
        if label == "correct" and (sig["declined_shape"] or not sig["parse_ok"]):
            return True  # judge says correct, but the model gave nothing to be correct about
        if label == "wrong" and sig["cvss_match"]:
            return True  # judge says wrong, yet the CVSS exactly matches a published NVD score
    else:
        if (label in ("fabricated", "hijacked") and sig["claimed_reserved"]
                and sig.get("registry_state") in ("RESERVED", "REJECTED")
                and not sig["claimed_details"]):
            # The model reported a genuinely-reserved ID as reserved with no
            # substantive details -- that is accurate, not fabrication.
            return True
        if label == "rejected" and sig["claimed_details"] and sig["parse_ok"]:
            # "rejected" yet the model asserted SUBSTANTIVE vuln details
            # (cvss/severity/products). A mitigation note alone is part of a normal
            # rejection ("verify the identifier") and no longer counts here.
            return True
        if label in ("fabricated", "hijacked") and sig["declined_shape"]:
            return True  # "fabricated" yet the model declined to fill anything
    return False


def load_records(patterns: list[str]) -> list[dict]:
    """All raw runs, minus failed calls, deduped to one record per cell.

    A resumed sweep re-runs cells whose first attempt failed, so a file can hold
    both the failed line and its later success for the same
    (id, model, condition, repeat, thinking_budget). Keep the newest successful
    line only: scoring a failed call would send the judge an empty answer, and
    scoring both would double-count the cell in every rate we report.
    """
    files: list[str] = []
    for p in patterns:
        files.extend(glob.glob(p))
    # Dry-run cells share (id, model, condition, repeat=0) with real sweep cells,
    # so they must not be mixed in unless asked for by name.
    if not any("dryrun" in p for p in patterns):
        files = [f for f in files if not Path(f).name.startswith("dryrun_")]
    by_cell: dict[tuple, dict] = {}
    total = failed = 0
    for fp in files:
        for line in Path(fp).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            total += 1
            if not r.get("ok"):
                failed += 1
                continue
            cell = (r["id"], r["model"], r["condition"], r["repeat"],
                    r.get("thinking_budget"))
            prev = by_cell.get(cell)
            if prev is None or (r.get("ts") or 0) >= (prev.get("ts") or 0):
                by_cell[cell] = r
    dropped = total - failed - len(by_cell)
    if failed or dropped:
        print(f"  skipped {failed} failed call(s), {dropped} superseded duplicate(s) "
              f"of {total} raw lines")
    return list(by_cell.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None,
                    help="run id to score (default: the CURRENT run)")
    ap.add_argument("--input", nargs="+", default=None,
                    help="raw jsonl glob(s); default: the run's raw/*.jsonl")
    ap.add_argument("--out", default=None,
                    help="labels csv path; default: the run's labels.csv")
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent judge calls (default 8); each is one network call")
    a = ap.parse_args()

    rd = C.resolve_run(a.run)
    if rd is None:
        sys.exit("no run found (start one with run_experiment.py) "
                 "or pass --run <id> / explicit --input + --out")
    print(f"run: {rd.name}")
    if a.input is None:
        # A run is either a dry run or a real one (metadata says which); name
        # the dryrun files explicitly so load_records doesn't filter them out.
        meta = json.loads((rd / "metadata.json").read_text()) if (rd / "metadata.json").exists() else {}
        pat = "dryrun_*.jsonl" if meta.get("dry_run") else "*.jsonl"
        a.input = [str(rd / "raw" / pat)]
    if a.out is None:
        a.out = str(rd / "labels.csv")

    gt_by_id = {it["id"]: it["ground_truth"]
                for it in json.loads(C.DATASET.read_text())["items"]}
    records = load_records(a.input)
    if not records:
        sys.exit(f"no records matched {a.input}")

    # Cache file is per run AND per judge model: verdicts from different
    # judges must never be interchanged (the cache key is only (id, answer_text)).
    judge = CachingJudge(cache_path=rd / f"judge_cache_{C.JUDGE_MODEL}.jsonl")

    def score_one(rec):
        """Judge + rule signals for one record. Thread-safe: the judge cache and
        nvd cache serialize their own writes; everything else here is local."""
        gt = gt_by_id.get(rec["id"], {"exists": False})
        sig = rules.derive_signals(rec, gt)
        # Hand the judge the recomputed searched flag (rules corrects raw lines
        # whose flag counted memory-recalled citations as search evidence).
        judged = judge.judge({**rec, "searched": sig["searched"]}, gt)  # may raise QuotaExhausted
        hij = judged.get("hijacked_cve")
        hij_exists = nvd_exists(hij) if hij else None
        conflict = detect_conflict(sig, judged, sig["id_exists"])
        reason_raw = judged.get("reason") or ""
        # A judge FAILURE (API error / unparseable) also surfaces as label "unsure";
        # tell it apart from a real model decline by its reason text.
        is_failure = reason_raw.startswith("judge error") or "unparseable" in reason_raw
        label = judged.get("label")
        # Relabel a genuine real-ID abstention "unsure" -> "declined" (clearer name;
        # also normalizes verdicts cached under the old label). Judge failures keep
        # the "unsure" sentinel so they stay visible.
        if sig["id_exists"] and label == "unsure" and not is_failure:
            label = "declined"
        # needs_handcheck = genuine human-review load only: rule/judge conflicts and
        # any judge failure. Model declines are a clean, decisive label (the judge
        # doesn't even need the NVD row for them) and no longer auto-flagged.
        needs_hc = conflict or is_failure
        return {
            "id": rec["id"], "category": rec["category"], "model": rec["model"],
            "condition": rec["condition"], "repeat": rec["repeat"],
            "thinking_budget": rec.get("thinking_budget"),
            **sig,
            "judge_label": label,
            "hijacked_cve": hij, "hijacked_cve_exists": hij_exists,
            "retrieval_diagnosis": judged.get("retrieval_diagnosis"),
            "judge_reason": judged.get("reason"),
            "rule_judge_conflict": conflict, "needs_handcheck": needs_hc,
        }

    rows = []
    quota_err = None
    # I/O-bound judge calls -> run them concurrently. Worker threads only call the
    # network + return a row; the judge cache/file writes are internally locked.
    with cf.ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        futs = [ex.submit(score_one, rec) for rec in records]
        done = 0
        for fut in cf.as_completed(futs):
            try:
                rows.append(fut.result())
            except QuotaExhausted as e:
                quota_err = e
                for f in futs:
                    f.cancel()  # stop not-yet-started work; in-flight ones finish
                break
            done += 1
            if done % 50 == 0:
                print(f"  scored {done}/{len(records)} "
                      f"(new judge calls: {judge.calls}, cache hits: {judge.hits})",
                      flush=True)

    if quota_err is not None:
        print(f"\nDAILY JUDGE QUOTA EXHAUSTED for {C.JUDGE_MODEL} "
              f"after {judge.calls} new call(s) this run.", file=sys.stderr)
        print(f"{len(judge.cache)} verdict(s) safely cached in "
              f"{rd / f'judge_cache_{C.JUDGE_MODEL}.jsonl'}; labels.csv NOT overwritten.",
              file=sys.stderr)
        print("Resume later (higher tier or after the quota resets) -- cached "
              "verdicts are reused for free.", file=sys.stderr)
        print(f"  detail: {str(quota_err)[:200]}", file=sys.stderr)
        sys.exit(2)

    df = pd.DataFrame(rows)
    if not df.empty:  # as_completed returns out of order -> stable sort for a clean CSV
        df = df.sort_values(["model", "condition", "id", "repeat"]).reset_index(drop=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"\nWrote {len(df)} rows -> {a.out}")
    print(f"judge calls (deduped): {judge.calls} of {len(records)} records")
    print(f"conflicts flagged: {df['rule_judge_conflict'].sum()}; "
          f"needs_handcheck: {df['needs_handcheck'].sum()}")


if __name__ == "__main__":
    main()
