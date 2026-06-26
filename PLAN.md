# Plan: `untell` — a public Claude skill (research prototype + eval harness)

## Context

Verified deep research (`humanizer-research-report.md`) shows the strongest **training-free** evasion technique is a **closed-loop detector-feedback rewrite** (arXiv 2506.07001: −88% TPR@1%FPR, transfers across detectors, quality preserved) — and no shipping humanizer does it; they do blind single-pass paraphrasing (60–80% plateau).

**Goal (locked):** ship this loop as a **publicly distributable Claude skill** — any Claude Code user installs it and runs `/untell`. Python research prototype + eval harness underneath. No GPU training, no commercial detector keys, no web UI. Greenfield repo (git initialized).

**Key reframe — Claude *is* the rewriter.** The skill runner (Claude) performs the rewrites; the loop is orchestrated by `SKILL.md`. No external LLM API key needed → lowest install friction. Detectors are local lightweight scripts that score text; Claude reads scores and re-rewrites until the ensemble is under threshold while semantic similarity holds.

## How it works (the loop, driven by SKILL.md)

```
/untell <text|file>
  Claude: preserve-lock citations/entities/numbers/quotes (scripts/preserve.py)
  repeat up to N iters:
    score = scripts/score.py <text>     # ensemble of local detectors → {detector: P(AI)}, max
    sim   = scripts/quality.py <orig> <text>   # semantic similarity (must stay ≥ 0.76)
    if max(score) < threshold and sim ok: break
    Claude rewrites text using the per-detector scores as feedback
      (raise burstiness + perplexity, vary sentence architecture, keep meaning + sentinels)
  restore locked spans → output humanized text + before/after detector table
```

Targets the **max** proxy (multi-detector evasion, report gap #3). Quality gate = 0.76 P-SP bar (watermark-removal paper). Preserve-lock = citation/meaning differentiator (gap #5). The whole loop is `SKILL.md` instructions + 3 small scripts; Claude supplies the rewrite intelligence.

## Distribution as a Claude skill

- `untell/SKILL.md` — frontmatter `name: untell`, `description:` tuned to trigger on "untell text / bypass AI detector / make this sound human / reduce AI detection". Body = the loop procedure above, with explicit stop conditions and the feedback-rewrite rubric.
- Installable two ways: (a) copy `untell/` into `~/.claude/skills/`; (b) as a plugin dir. README documents both.
- **Tiered deps so it's low-friction public:**
  - **lite** (zero-ML, default fallback): perplexity+burstiness heuristic via a tiny script — no model download. Weak but instant.
  - **full**: installs `transformers`+`torch`, pulls RoBERTa-OpenAI + MAGE (CPU). Real proxy signal.
  - **heavy** (opt-in flag): Binoculars (2×Falcon-7B, GPU). For serious eval only.
  - `score.py` auto-detects what's installed and degrades gracefully (logs which tier ran).

## Reusable building blocks (confirmed)

| Component | Use | Tier |
|---|---|---|
| Perplexity+burstiness (GPTZero-style) | ~15 lines, GPT-2 sentence-PPL variance | lite/full |
| `openai-community/roberta-base-openai-detector` | HF pipeline | full (CPU) |
| `yaful/MAGE` (Longformer) | HF pipeline, `device='cpu'` | full (CPU) |
| Fast-DetectGPT (github baoguangsheng/fast-detect-gpt, gpt-neo) | optional adapter | full (CPU, slow) |
| Binoculars (github ahans30/Binoculars, BSD-3) | 2×Falcon-7B | heavy (GPU) |
| sentence-transformers (all-MiniLM) | semantic-sim quality gate | full |
| Datasets `liamdugan/raid`, `Hello-SimpleAI/HC3`, `yaful/MAGE` | eval gauntlet | eval only |

## Repo layout

```
Untell/
  LICENSE                       # MIT (public distribution)
  README.md                     # what it is, install (both ways), tiers, ethics/research note
  pyproject.toml                # extras: [full], [heavy], [eval]; console script `untell-score`
  untell/                     # THE SKILL (this dir is what users install)
    SKILL.md                    # name+description trigger; the loop procedure + rewrite rubric
    scripts/
      score.py                  # ensemble detector scoring → JSON {detector: P(AI), max}; tier auto-detect
      preserve.py               # mask/restore citations(regex)+entities(spaCy)+numbers+quotes
      quality.py                # semantic similarity (sentence-transformers; lite fallback = token-overlap)
    detectors/                  # adapters imported by score.py
      base.py  perplexity_burstiness.py  roberta_openai.py  mage.py  fast_detectgpt.py  binoculars.py
    references/                 # optional: thresholds.md, prompt-rubric.md the skill can load
  eval/
    datasets.py  benchmark.py  baselines.py  report.py
  tests/
```

## Implementation steps

1. **Scaffold + license** — `pyproject.toml` with extras (`full`=transformers/torch/sentence-transformers/spacy, `heavy`=bitsandbytes, `eval`=datasets), MIT `LICENSE`, README with install (both methods) + tier table + research-only/ethics note.
2. **Detector layer** — `detectors/base.py` protocol `score(text)->float P(AI)∈[0,1]`. Implement perplexity_burstiness (lite, no heavy dep beyond a small LM — or pure-heuristic if torch absent), roberta_openai, mage (all CPU). Fast-DetectGPT + Binoculars adapters guarded by availability. `score.py` loads whatever tier is installed, emits JSON, prints which tier ran.
3. **Preserve-lock** — `preserve.py`: regex citations (APA/IEEE/MLA/numeric) + numbers + quotes, spaCy NER (optional; regex-only fallback) → sentinels; restore. Round-trip tested.
4. **Quality gate** — `quality.py`: sentence-transformers cosine ≥ 0.76; lite fallback = normalized token-overlap so it runs zero-install.
5. **SKILL.md** — frontmatter (name/description for triggering) + the loop: lock → score → check stop (`max < threshold` & sim ok) → if not, rewrite with the per-detector scores as explicit feedback per the rubric → cap at N iters → restore → print before/after table. Include the rewrite rubric (burstiness/perplexity/sentence-architecture, preserve meaning + sentinels) and threshold defaults in `references/`.
6. **CLI helper** — `untell-score` console entry wrapping `score.py` for manual/CI use.
7. **Eval harness** — `datasets.py` samples AI texts (HC3 bootstrap, RAID real test); `benchmark.py` runs no-op / single-pass / full-loop, computes per-detector pre→post, **bypass rate**, semantic sim, iters; `report.py` writes markdown table. **Success = full-loop bypass rate > single-pass at equal-or-better sim** (the report's thesis). (Loop here scripted to mimic the skill so it's measurable without a human in the seat.)

## Verification

- **Unit:** each detector returns [0,1] (AI > human on known samples); preserve-lock restores citations/numbers/entities exactly; quality sim ≈1.0 on identical text; `score.py` runs in lite tier with zero ML installed.
- **Skill smoke:** install `untell/` into `~/.claude/skills/`, run `/untell` on a sample AI paragraph → final max-proxy score < initial, sim ≥ 0.76, ≤ N iters, before/after table printed.
- **End-to-end eval:** `python -m eval.benchmark --dataset hc3 --n 100` → metrics report; confirm full-loop beats single-pass. Re-run on RAID for the headline number.

## Risks / honest caveats (README)

- Proxy detectors ≠ commercial. RoBERTa-OpenAI is weak on modern text; ensemble is a *signal*, not ground truth. v2 adds paid-API validation behind a flag.
- lite tier is a weak heuristic — good for zero-install demo, not a real evasion claim. Full tier is the honest baseline; Binoculars (GPU) is the strongest proxy.
- Claude-as-rewriter means quality/evasion depend on the running model; document expected behavior.
- Research/eval harness + legitimate defensive use (non-native writers falsely flagged at 61% FPR, report §4). State research-only in README + SKILL.md.

## Out of scope for v1 (later)

Local DPO/RL-against-ensemble (StealthRL/MASH moat — GPU), commercial-detector API validation, hosted-API rewriter option, web UI, back-translation/token-mixing modules, marketplace/plugin publishing automation.
