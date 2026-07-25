# Knowing When to Look — Retrieval Decisions in Security-Domain LLMs

Does a tool-using LLM *choose to search* at the right moments when asked about
real vs. fabricated CVE IDs, and — when it fails — *where* does the failure
happen (didn't search / searched but hijacked a different real CVE / searched
and ignored the truth)? See `research_proposal.md` for the full framing.

Gemini-only, total API budget well under $100 (expected ~$17).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then fill in GOOGLE_API_KEY (NVD_API_KEY optional)
```

(All `python ...` commands below assume the venv is active.)

## Pipeline

```bash
# 1. Dataset (free; NVD only, no LLM). Produces dataset/dataset.json.
python dataset/build_dataset.py
python dataset/build_dataset.py --verify

# 2. Dry run first (<$0.50): validates model IDs, grounding + parsing end to end.
python harness/run_experiment.py --dry-run

# 3. Full sweep (resumable — safe to re-run after a crash).
python harness/run_experiment.py

# 4. Score: rules (free) + judge (gemini-2.5-pro) -> results/labels.csv
python scoring/score.py

# 5. Hand-check 15-20% of labels, then measure agreement.
python scoring/handcheck.py export
#   ...fill in manual_label in results/handcheck_sample.csv...
python scoring/handcheck.py agreement

# 6. RQ1-RQ4 tables + figures.
python analysis/analyze.py

# RQ4 (stretch): re-run fakes at low vs high thinking on a thinking-capable model.
python harness/run_experiment.py --models gemini-2.5-flash --thinking-budget 0
python harness/run_experiment.py --models gemini-2.5-flash --thinking-budget 8000
```

## Layout

| Path | Role |
|---|---|
| `config.py` | model IDs, prompt, prices, paths (edit model IDs here if the dry run rejects one) |
| `dataset/build_dataset.py` | one-time generator for `dataset/dataset.json` (NVD only) |
| `harness/gemini_client.py` | one Gemini call, search on/off, grounding-metadata read-back |
| `harness/run_experiment.py` | resumable sweep + running spend counter |
| `scoring/rules.py` | deterministic signals vs stored NVD truth (never guesses) |
| `scoring/judge.py` | LLM-as-judge meaning labels (never decides truth) |
| `scoring/score.py` | merge rules + judge -> `results/labels.csv` |
| `scoring/handcheck.py` | export sample + agreement stats |
| `analysis/analyze.py` | RQ1-RQ4 |

## Design notes

- **Intention-neutral prompt.** Identical for real and fake IDs, framed as a
  production vuln-management request; nullable schema fields let the model
  decline without any `is_real` flag hinting that fakes exist.
- **JSON is requested in the prompt text, not via API schema enforcement**,
  because Gemini's `google_search` tool has historically conflicted with
  enforced `response_schema`. The dry run confirms current behavior.
- **Rules vs judge.** Rules do exact comparisons against the NVD answer key and
  assert only conclusive facts; the judge interprets language; NVD alone decides
  truth. Where they disagree, the row is force-included in the hand-check sample.
