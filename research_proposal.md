# Knowing When to Look: Do Security-Domain LLMs Search When They Should?

**NLP course research proposal**

---

## 1. Project Title & Problem Statement

**Title:** Knowing When to Look — Retrieval Decisions and Fabricated-Identifier Handling in Security-Domain LLMs.

**Problem.** People increasingly ask LLMs about security specifics: what a given CVE affects, whether a software package is safe, which setting hardens a system. A confident but wrong answer here is dangerous, and stronger models have not removed the risk. Modern models can also *look things up* with a search tool — but they decide for themselves whether to use it. This raises a question almost no one has studied for security facts: **does the model choose to search at the right moments, and when it does search, does that actually stop it from giving a wrong answer?** A 2025 study ([Abdullah et al.](https://arxiv.org/abs/2506.13161)) showed an older, search-free ChatGPT could not tell 100 real CVE IDs from 100 fake ones (it wrote believable advisories for ~96% and ~97% respectively). This project asks whether today's tool-using models are actually better, and *why* they succeed or fail.

## 2. Detailed Description of Idea and Innovation Highlights

The core idea is a controlled experiment: ask several current models about **real and fake CVE IDs**, with the search tool **turned off and on**, and study two things — *when the model chooses to search*, and *whether it can correctly say an ID is fake*.

A small pilot (run in Google AI Studio, June 2026) already shows the phenomenon is real and depends on model strength: **Gemini 3.1 Pro** correctly rejected a fabricated CVE from memory, while **Gemini 2.5-flash-lite** produced a fully fabricated advisory for the same fake ID — with real-looking citations and a real Microsoft MSMQ port (1801). That tells us the weak model didn't invent randomly; it grabbed details from a *different real* vulnerability and attached them to a CVE that doesn't exist.

**Innovation highlights — what is new here, stated plainly:**

1. **We study the model's *decision* to search, not whether search helps.** Earlier work feeds documents to the model through a fixed pipeline and measures if accuracy improves. We instead let the model decide whether to search, and measure whether that decision is sensible: does it search when it would otherwise be wrong, and skip searching when it already knows?

2. **We separate three different ways a search-enabled model can still be wrong**, which prior work lumps together:
   - it **didn't search** and answered from memory;
   - it **searched but pulled the wrong real CVE** and attributed it to the queried ID (the "identifier hijacking" we already saw in the pilot);
   - it **searched and got the right information but ignored it**.

Telling these three apart is the most original part of the project: it turns "the model hallucinated" into a concrete diagnosis of *where* the failure happened.

## 3. Implementation Steps

1. **Build the dataset.** Pull real CVE IDs and their details from the NVD / GitHub Security Advisory databases. Generate fake CVE IDs with a script and confirm each is absent from NVD. Add deliberate "near-miss" fakes (one digit off a real CVE). Save the correct answer for every item with a timestamp.
2. **Build the test harness.** Write code that sends each question to a model and, in the "search-on" condition, enables the built-in **Grounding with Google Search**. For every run we log the full result: the answer, whether the model grounded/searched and what it retrieved (read from the grounding metadata the API returns), and the model's stated confidence.
3. **Run the experiment.** Ask every item under two settings — search **off** (memory only) and search **on** (model may search or not) — for each model, repeating each a few times so we measure *rates*, not single lucky/unlucky answers.
4. **Score the answers.** We write a scoring program (our own short script — not an existing tool) that attaches a label to each answer (real IDs: correct / wrong / unsure; fake IDs: rejected / fabricated / hijacked-a-real-CVE). It works in two parts:
   - *Rule-based part (plain code).* Some labels are decided by simple logic against the NVD ground truth we already saved — whether the queried ID exists, whether a stated CVSS score or affected product matches NVD, and whether the model searched (read from our own log).
   - *Judge part (LLM-as-judge).* Meaning-based labels can't be string-matched (e.g. "I can't find that CVE" should count as *rejected*; deciding whether a fake answer *hijacked* a different real CVE). For these we call a **separate strong model as a judge**, giving it the model's answer, the queried ID, the NVD ground truth, and any retrieved evidence, and asking for one label from a fixed list plus a one-line reason. The judge classifies against NVD facts; it never decides truth itself (that stays with NVD), which avoids reintroducing the hallucination we are studying.

   We then **hand-check ~15–20% of the labels** ourselves and report how often our labels agree with the script's. If agreement is low on the hardest cases (the three-way fake diagnosis), we hand-label that subset instead.
5. **Analyze and write up.** Compute the comparisons described in §4 for each research question and summarize the findings.

## 4. Methodology & Datasets, and Models

### How we answer each research question (steps, not heavy statistics)

**RQ1 — Does the model search at the right times?**
- From the **search-off** runs, score every item to get a simple per-item label: *does this model know the answer from memory or not?*
- From the **search-on** runs, record whether the model chose to search each item.
- Compare the two: among items the model gets **wrong from memory**, how often does it search? Among items it **already knows**, how often does it search needlessly? A well-behaved model mostly searches the questions it *doesn't* already know the answer to, and skips searching the ones it does. We report these as plain percentages per model.

**RQ2 — How does the model handle fake IDs?**
- Ask all real and fake IDs, off and on. For fakes, count how often the model **rejects**, **fabricates**, or **hijacks a real CVE**.
- For every fabrication that happened with search on, check what was retrieved: nothing, the wrong CVE, or the right "this doesn't exist" answer it then ignored. Report the breakdown — this is the three-way diagnosis from §2.
- Compare fabrication rates off vs. on to see whether search actually helps for fakes.

**RQ3 — Is the model's confidence a useful signal?**
- The model states a confidence (0–100) on every answer. Compare its confidence on **correct** vs. **wrong** answers (does it sound just as sure when it's wrong?), and check whether **lower confidence** lines up with **choosing to search**.

**RQ4 (stretch) — Does "thinking harder" help?**
- For models with a thinking-level control, re-run the fake IDs at low vs. high thinking and compare fabrication rates.

We report results mainly as rates and simple before/after comparisons across models, with counts shown so the reader can judge how solid each number is.

### Datasets

- **NVD (National Vulnerability Database)** and **GitHub Security Advisory (GHSA)** — source of real CVE IDs, their details (affected products, severity, description), and the authoritative check for whether a fake ID truly doesn't exist. This is our **answer key for scoring**, not the model's search tool.
- **Retrieval** in the search-on condition is the built-in **Grounding with Google Search** (open web), with grounding metadata logged so we know whether and what the model searched.
- **Our constructed set** (~80–150 items): real / known, real / recent (released after the models' training cutoff, so unknowable without search), fake / random, and fake / near-miss.

### Models

- **Gemini 3.1 Flash-Lite** — newer-generation small model.
- **Gemini 2.5 Flash** — previous-generation mid model.
- **Gemini 2.5 Flash-Lite** — previous-generation small model (already shown to fabricate in the pilot).
- *(Optional)* one non-Google model for cross-provider comparison.

This set spans two generations and two size tiers (Flash vs. Flash-Lite), giving a capability gradient while keeping cost very low — all three are inexpensive and usable for free in the Google AI Studio UI. We deliberately target the *cheaper-model regime*, because these are the models people actually run at scale, and (as the pilot suggests) the regime where fabrication is most likely and therefore most consequential. All models are reachable via API / Google AI Studio; no model training is required.

## 5. Student IDs & Emails

| Name | Student ID | Email |
|---|---|---|
| Amit Porat | 315390252 | amit.porat01@post.runi.ac.il |
