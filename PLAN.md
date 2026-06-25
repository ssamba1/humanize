# Plan: `humanizer` — research prototype + eval harness

## Context

We completed verified deep research (`Humanize/humanizer-research-report.md`) on AI-text detectors and evasion. The single strongest **training-free** technique in the literature is **closed-loop, detector-feedback adversarial paraphrasing** (arXiv 2506.07001: −88% TPR@1%FPR, transfers across detectors, quality preserved) — and *no shipping commercial humanizer does it*; they do blind single-pass paraphrasing and plateau at 60–80% bypass. This repo builds and proves that loop.

**v1 goal (locked with user):** a Python library + CLI that humanizes AI text via a hosted-LLM detector-feedback loop scored against **open-source detector proxies**, plus a benchmark harness that proves the loop beats single-pass baselines. No GPU training, no commercial detector keys, no web UI. Greenfield repo at `C:\Users\Admin\Humanize`.

**Scope decisions:** Python · open-source proxy detectors only · hosted LLM API (Anthropic Claude default, OpenAI optional) for rewriting, training-free · prototype + eval, not SaaS.

## Architecture (the core loop)

```
input text
  → preserve-lock (mask citations, named entities, numbers, quotes)
  → LLM rewriter (style-injection + per-detector feedback prompt)
  → detector ensemble scores each proxy → P(AI) ∈ [0,1]
  → if max-proxy < threshold AND semantic-sim ≥ 0.76 → stop; else feed scores back, re-rewrite
  → (cap iterations) → restore locked spans → output + per-detector report
```

Targets the **max** proxy score (multi-detector evasion, per report gap #3). Quality gate uses the 0.76 P-SP acceptability bar from the watermark-removal paper. Preserve-lock is the citation/meaning differentiator (report gap #5).

## Reusable building blocks (confirmed to exist)

| Component | Use | CPU? |
|---|---|---|
| `openai-community/roberta-base-openai-detector` | HF `pipeline("text-classification")` | ✅ fast |
| `yaful/MAGE` (Longformer) | HF pipeline; repo supports `device='cpu'` | ✅ |
| Fast-DetectGPT (github baoguangsheng/fast-detect-gpt) | scoring model `gpt-neo-2.7B` or `gpt2` | ✅ slow |
| Perplexity+burstiness (GPTZero-style) | ~15 lines, GPT-2 via transformers, std-dev of sentence PPL | ✅ |
| Binoculars (github ahans30/Binoculars, BSD-3) | 2×Falcon-7B — **optional, GPU/4-bit only** | ❌ gate behind flag |
| Datasets: `liamdugan/raid` (adversarial, best), `Hello-SimpleAI/HC3` (small bootstrap), `yaful/MAGE` (cross-LLM) | eval gauntlet | — |

Note `IMGTB` (github kinit-sk/IMGTB) wraps several of these — pattern-match its plugin API, but write thin adapters (cleaner than depending on an unmaintained research CLI).

## Repo layout

```
Humanize/
  pyproject.toml            # deps, console_script entrypoint
  README.md                 # what it is, install, ethics/research-only note
  .env.example              # ANTHROPIC_API_KEY / OPENAI_API_KEY
  src/humanizer/
    config.py               # pydantic settings: thresholds, model ids, max_iters
    detectors/
      base.py               # Detector protocol: name, score(text)->float P(AI) in [0,1]
      roberta_openai.py  mage.py  fast_detectgpt.py
      perplexity_burstiness.py
      binoculars.py         # optional, GPU flag
      ensemble.py           # collect scores, aggregate (max + per-detector dict)
    rewriter/
      base.py               # Rewriter protocol: rewrite(text, scores, feedback)->text
      llm_rewriter.py       # Anthropic default; OpenAI optional; retries/cost track
      prompts.py            # style-injection + detector-feedback templates
    preserve/lock.py        # mask/restore citations(regex), entities(spaCy NER), numbers, quotes
    quality/semantic.py     # sentence-transformers cosine (all-MiniLM) ~ P-SP proxy
    loop/humanize.py        # the orchestrator (core algorithm above)
    cli.py                  # `humanize "text"` | `--file x.txt` | `--report`
  eval/
    datasets.py             # load + sample RAID / HC3 / MAGE subsets
    benchmark.py            # run gauntlet, pre/post scores, bypass rate, sim, iters, cost
    baselines.py            # no-op, single-pass "rewrite human" prompt (the thing to beat)
    report.py               # markdown + JSON metrics
  tests/                    # adapters, preserve round-trip, loop integration
```

## Implementation steps

1. **Scaffold** — `pyproject.toml` (deps: `transformers, torch` CPU, `sentence-transformers`, `spacy`, `anthropic`, `datasets`, `typer`, `pydantic`, `pytest`; optional `openai`, `bitsandbytes`), `config.py`, `.env.example`, `README` with research-only framing.
2. **Detector layer** — `Detector` protocol returning normalized P(AI)∈[0,1]; implement RoBERTa-OpenAI, MAGE, perplexity-burstiness first (all CPU). Fast-DetectGPT adapter (gpt2/gpt-neo). `ensemble.py` returns `{name: score}` + `max`. Binoculars behind `--enable-binoculars` flag. Cache loaded models.
3. **Preserve-lock** — regex for inline citations (APA/IEEE/MLA/numeric) + numbers/quotes, spaCy NER for entities; replace with sentinels, restore after loop. Unit-test round-trip integrity.
4. **Rewriter** — `LLMRewriter` calls hosted API with `prompts.py`: (a) base style-injection ("rewrite with human burstiness/perplexity, keep meaning, preserve sentinel tokens"), (b) feedback variant injecting which detectors flagged it and their scores. Track tokens/cost. Provider-agnostic via protocol.
5. **Quality gate** — `semantic.py` cosine similarity (sentence-transformers); enforce ≥ 0.76, reject-and-retry on violation.
6. **Core loop** — `humanize.py`: lock → iterate (score → check stop → feedback rewrite) up to `max_iters` (default 5), early-exit when max-proxy < threshold & sim ok → restore → return `(text, history)` where history has per-iter scores/sim/cost.
7. **CLI** — `typer` app: humanize a string/file, print per-detector before/after table + final text; `--report` dumps JSON.
8. **Eval harness** — `datasets.py` samples ~100 AI texts (HC3 to bootstrap, RAID for the real test); `benchmark.py` runs no-op / single-pass / full-loop over the sample, computes per-detector pre→post, **bypass rate** (% below threshold), semantic sim, iters, $/text; `report.py` writes a markdown comparison table. **Success = full-loop bypass rate > single-pass baseline at equal-or-better semantic similarity** (the report's thesis).

## Verification

- **Unit:** each detector adapter returns [0,1] on a known human and known AI sample (AI scores higher); preserve-lock restores citations/numbers/entities byte-exact; semantic sim returns ~1.0 for identical text.
- **Integration:** run `humanize` on a sample AI paragraph → assert final max-proxy score < initial AND sim ≥ 0.76 AND iterations ≤ max.
- **End-to-end eval:** `python -m eval.benchmark --dataset hc3 --n 100` → produces metrics report; confirm full-loop bypass rate beats single-pass. (Run on RAID for the headline number once HC3 smoke passes.)
- **Smoke:** `humanize --file sample.txt` prints a before/after detector table.

## Risks / honest caveats (to put in README)

- Proxy detectors ≠ commercial detectors. RoBERTa-OpenAI is weak on modern LLM text; treat the ensemble as a *signal*, not ground truth. Beating proxies is necessary, not sufficient — v2 adds paid-API validation (Originality/GPTZero) behind a flag.
- Binoculars (the strongest proxy) needs a GPU; v1 runs the CPU-feasible four by default and notes the gap.
- Loop cost scales with iterations × LLM calls — cap iterations, log cost per run.
- Repo is a research/evaluation harness; README states research-only + the legitimate defensive use (non-native writers falsely flagged at 61% FPR, report §4).

## Out of scope for v1 (later phases)

Local DPO/RL-against-ensemble (the StealthRL/MASH moat — needs GPU), commercial-detector API validation, web UI / auth / billing, back-translation & token-mixing attack modules.
