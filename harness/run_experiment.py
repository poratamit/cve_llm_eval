"""Sweep items x models x conditions x repeats -> results/raw/*.jsonl.

Resumable: an interrupted run re-reads existing output and skips completed
(item, repeat) tuples, so a crash costs no extra API spend.

  python harness/run_experiment.py --dry-run          # 5 items, 1 repeat, <$0.50
  python harness/run_experiment.py                      # full sweep
  python harness/run_experiment.py --workers 12         # more concurrency
  python harness/run_experiment.py --models gemini-2.5-flash --thinking-budget 0   # RQ4
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from harness import gemini_client as gc

load_dotenv(C.ROOT / ".env")


def load_items() -> list[dict]:
    if not C.DATASET.exists():
        sys.exit(f"dataset not found: {C.DATASET} (run dataset/build_dataset.py first)")
    return json.loads(C.DATASET.read_text())["items"]


def dry_run_items(items: list[dict]) -> list[dict]:
    """A few items spread across categories, so both real and fake search paths run."""
    picked, seen = [], {c: 0 for c in C.CATEGORIES}
    per = max(1, C.DRY_RUN_ITEMS // len(C.CATEGORIES))
    for it in items:
        if seen[it["category"]] < per and len(picked) < C.DRY_RUN_ITEMS:
            picked.append(it); seen[it["category"]] += 1
    return picked


def out_path(raw_dir: Path, model: str, condition: str, dry: bool) -> Path:
    tag = "dryrun_" if dry else ""
    return raw_dir / f"{tag}{model}_{condition}.jsonl"


def done_keys(path: Path) -> set[tuple[str, int]]:
    """(id, repeat) tuples already answered *successfully*.

    Failed calls (ok=False, e.g. a 429 that outlasted the client's retries) are
    deliberately NOT counted as done, so a resume retries them instead of
    baking the failure into the results. score.py drops the stale failed line.
    """
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("ok"):
                keys.add((r["id"], r["repeat"]))
    return keys


def run(run_rd, models, conditions, repeats, dry, thinking_budget, tb_tag, workers):
    raw_dir = run_rd / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    items = dry_run_items(load_items()) if dry else load_items()
    prices = C.PRICES
    grand_spend = 0.0
    gc.client()  # warm the shared client once, before threads race on lazy init

    for model in models:
        in_price, out_price = prices.get(model, (0.0, 0.0))
        for condition in conditions:
            search_enabled = condition == "on"
            path = out_path(raw_dir, model, condition, dry)
            if tb_tag:  # RQ4: separate file per thinking level
                path = path.with_name(path.stem + f"_tb{tb_tag}.jsonl")
            already = done_keys(path)
            pending = [(it, rep) for it in items for rep in range(repeats)
                       if (it["id"], rep) not in already]
            tokens_in = tokens_out = 0
            grounded = calls = 0
            t0 = time.time()

            # Worker threads only do the network call and return; the main thread
            # owns all writes + counters, so no locks are needed. gc.ask never
            # raises (it captures errors into the result dict), so .result() is safe.
            def work(task, _model=model, _cond=condition, _search=search_enabled):
                it, rep = task
                prompt = C.PROMPT_TEMPLATE.format(cve_id=it["id"])
                res = gc.ask(it["id"], prompt, _model, _search,
                             thinking_budget=thinking_budget)
                rec = {
                    "id": it["id"], "category": it["category"],
                    "model": _model, "condition": _cond,
                    "repeat": rep, "thinking_budget": thinking_budget,
                    "ts": time.time(), **res,
                }
                return it, rep, rec, res

            quota_err = None
            with path.open("a") as f, \
                    cf.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
                futs = [ex.submit(work, task) for task in pending]
                for fut in cf.as_completed(futs):
                    try:
                        it, rep, rec, res = fut.result()
                    except gc.QuotaExhausted as e:
                        # Daily quota / billing exhaustion: abort instead of
                        # hammering the remaining calls x retries into it.
                        quota_err = e
                        for f2 in futs:
                            f2.cancel()  # unstarted work; in-flight ones finish
                        break
                    f.write(json.dumps(rec) + "\n"); f.flush()
                    calls += 1
                    u = res.get("usage") or {}
                    tokens_in += u.get("prompt_tokens") or 0
                    tokens_out += (u.get("candidates_tokens") or 0) + (u.get("thoughts_tokens") or 0)
                    if res.get("searched"):
                        grounded += 1
                    if not res["ok"]:
                        print(f"  ! error {it['id']} rep{rep}: {res['error']}", file=sys.stderr)
            if quota_err is not None:
                print(f"\nQUOTA EXHAUSTED during [{model} / search {condition}] "
                      f"after {calls} call(s): {str(quota_err)[:200]}", file=sys.stderr)
                print(f"Run is resumable: re-invoke with the same run "
                      f"({run_rd.name}) once the quota resets; completed "
                      f"(item, repeat) cells are skipped.", file=sys.stderr)
                sys.exit(2)
            tok_cost = (tokens_in * in_price + tokens_out * out_price) / 1e6
            gp = C.GROUNDING_PRICE_PER_1K.get(model, 0.0)
            ground_cost_worst = grounded / 1000 * gp
            grand_spend += tok_cost
            dt = time.time() - t0
            rate = f"{calls/dt:.1f}/s" if dt > 0 and calls else "-"
            print(f"[{model} / search {condition}] {calls} new calls, "
                  f"{grounded} grounded, {tokens_in/1000:.0f}k in / {tokens_out/1000:.0f}k out, "
                  f"~${tok_cost:.2f} tokens (+${ground_cost_worst:.2f} grounding if billed), "
                  f"{dt:.0f}s ({rate}) -> {path.name}")
    print(f"\nTotal token spend this run: ~${grand_spend:.2f} "
          f"(grounding billed separately; free tiers usually cover it)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--models", nargs="+", default=C.MODELS)
    ap.add_argument("--conditions", nargs="+", default=C.CONDITIONS)
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--thinking-budget", type=int, default=None,
                    help="RQ4: e.g. 0 (low) vs a high value; writes a separate _tb file")
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent API calls per (model, condition) batch (default 8)")
    ap.add_argument("--run", default=None,
                    help="run id to resume (default: the CURRENT run)")
    ap.add_argument("--new-run", action="store_true",
                    help="start a fresh run directory and make it CURRENT")
    ap.add_argument("--note", default="", help="free-text note stored in run metadata")
    a = ap.parse_args()
    repeats = a.repeats if a.repeats is not None else (1 if a.dry_run else C.REPEATS)
    tb_tag = "" if a.thinking_budget is None else str(a.thinking_budget)

    if a.new_run and a.run:
        sys.exit("--new-run and --run are mutually exclusive")
    if a.new_run or C.resolve_run(a.run) is None:
        rid = C.new_run(a.note, dry_run=a.dry_run)
        print(f"new run {rid} -> {C.run_dir(rid)}")
    rd = C.resolve_run(a.run)
    if rd is None:
        sys.exit(f"run not found: {a.run}")
    print(f"run: {rd.name}")
    run(rd, a.models, a.conditions, repeats, a.dry_run, a.thinking_budget, tb_tag, a.workers)


if __name__ == "__main__":
    main()
