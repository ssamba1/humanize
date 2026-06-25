# humanize

A **closed-loop, detector-feedback** AI-text humanizer, packaged as a **Claude skill**.

Most humanizers do a single blind paraphrase pass and plateau at 60–80% detector bypass. The
strongest *training-free* technique in the literature ([arXiv 2506.07001](https://arxiv.org/abs/2506.07001):
−88% TPR@1%FPR, transfers across detectors, preserves meaning) is an **iterative rewrite against
live detector scores** — and nothing ships it. This repo does, as a skill you install into Claude
Code: Claude is the rewriter, local scripts score the text and protect its facts, and the loop
runs until a detector ensemble stops flagging it while meaning is preserved.

> ⚠️ **Research / defensive tool.** AI detectors are noisy proxies — non-native English writers are
> falsely flagged at high rates (~61% FPR in some studies). The detectors here are *signals*, not
> ground truth, and the lite tier is a weak demo heuristic. Don't represent output as guaranteed
> undetectable by any commercial system. Use for research, evaluation, and legitimate defense
> against false positives.

## How it works

```
/humanize <text|file>
  preserve-lock citations / numbers / quotes / entities   (scripts/preserve.py)
  repeat up to N times:
    score = scripts/score.py <text>      # ensemble of local detectors -> {detector: P(AI), max}
    sim   = scripts/quality.py <orig> <text>   # semantic similarity, must stay >= 0.76
    if max(score) < threshold and sim ok: stop
    Claude rewrites using the per-detector scores as feedback
      (raise burstiness + perplexity, vary sentence architecture, keep meaning + sentinels)
  restore locked spans -> humanized text + before/after detector table
```

The loop drives the **max** across detectors (multi-detector evasion), gates every rewrite on a
**0.76** semantic-similarity bar (the P-SP threshold from the watermark-removal literature), and
keeps citations/numbers/quotes/entities byte-for-byte intact via preserve-lock.

## Install

**As a Claude skill (recommended):**

```bash
git clone https://github.com/ssamba1/humanize
cp -r humanize/humanize ~/.claude/skills/humanize   # copy the skill directory
# (or symlink it, or install as a plugin dir)
```

Then in Claude Code: `/humanize <your text or a file path>`. Works with **zero dependencies**
(lite tier). For real detector signal, install the full tier (below) in the repo.

**As a plugin dir:** point your Claude Code plugins at the cloned repo's `humanize/` directory.

## Tiers

The scripts auto-detect what's installed and degrade gracefully (the score JSON reports which
`tier` actually ran).

| Tier | Install | Detectors | Notes |
|---|---|---|---|
| **lite** | *(default, nothing to install)* | perplexity+burstiness heuristic; token-overlap quality | Stdlib only, instant, **weak** — demo signal, not an evasion claim. |
| **full** | `pip install -e ".[full]"` | + RoBERTa-OpenAI, MAGE, GPT-2 perplexity; MiniLM cosine quality | Real proxy signal on CPU. Downloads models on first run. |
| **heavy** | `pip install -e ".[heavy]"` | + Binoculars (2×Falcon-7B) | Strongest proxy; GPU recommended. Eval only. |

```bash
# Score any text directly (console script installed with the package):
humanize-score "Your text here" --tier full --threshold 0.3
echo "piped text" | humanize-score
```

## Eval harness (research only)

Validates the thesis — closed loop beats single-pass — without a human in the seat (a scripted
deterministic rewriter stands in for Claude so it's measurable):

```bash
pip install -e ".[eval]"
python -m eval.benchmark --dataset builtin --n 5      # zero-download smoke run
python -m eval.benchmark --dataset hc3 --n 100        # HuggingFace HC3
python -m eval.benchmark --dataset raid --n 100       # RAID (headline)
```

Success criterion: `full_loop` reaches a higher **bypass rate** than `single_pass` at
equal-or-better similarity. (With only the lite tier installed the absolute numbers are weak; the
*relative* comparison is the point — install `.[full]` for meaningful absolute rates.)

## Repo layout

```
humanize/            # THE SKILL (this dir is what you install)
  SKILL.md           # trigger + loop procedure + rewrite rubric
  scripts/           # score.py · preserve.py · quality.py
  detectors/         # base protocol + tiered adapters
  references/        # thresholds.md · prompt-rubric.md
eval/                # benchmark harness (research only)
tests/               # lite-tier unit tests (run with zero ML)
```

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest -q
```

CI runs two jobs: a **lite** matrix (ruff + pytest, no model downloads) across Python
3.9/3.11/3.12, and a **full-tier** job (Ubuntu, CPU torch + `.[full,eval]`) that loads the real
RoBERTa / Fast-DetectGPT / GPT-2-perplexity detectors and runs the `torch`-gated tests in
`tests/test_detectors_full.py`. The heavy tier (Binoculars) needs a GPU and is exercised
manually.

### Headless rewriter (optional)

The skill uses the running Claude as the rewriter (no key). For a fully programmatic loop (eval,
servers), `humanize.rewriter` adds optional hosted-LLM providers:

```bash
pip install -e ".[api]"      # anthropic + openai SDKs
export ANTHROPIC_API_KEY=...  # or OPENAI_API_KEY
```

```python
from humanize.rewriter import get_rewriter
rw = get_rewriter()           # None if no SDK/key -> caller falls back to the scripted rewriter
```

Run the whole loop standalone (lock → score → rewrite → restore → report), outside Claude:

```bash
humanize-loop "Your AI-sounding paragraph here." --tier full
humanize-loop --file draft.txt --json     # full structured result
```

(Needs `.[api]` + a key; without one it errors clearly and points you back to the `/humanize`
skill, where Claude is the rewriter.)

## Honest caveats

- **Proxy ≠ commercial.** These detectors approximate; they aren't Originality.ai/Turnitin/etc.
  RoBERTa-OpenAI in particular is weak on modern text. The ensemble is a signal, not a verdict.
- **lite is a demo.** The zero-install heuristic is good for showing the loop, not for claiming
  evasion. The full tier is the honest baseline; Binoculars (GPU) is the strongest proxy.
- **Claude is the rewriter.** Output quality and evasion depend on the running model.
- **Ethics.** Detector false-positives disproportionately harm non-native writers; this exists as
  a research/eval harness and a defense against that, not a plagiarism or academic-dishonesty aid.

## Deferred

Genuinely out of reach without GPUs, paid API keys, or a much larger surface — documented here
rather than shipped unverified:

- **Local DPO/RL-against-ensemble training** (the StealthRL/MASH "moat") — needs GPU training; the
  hosted-rewriter loop is the training-free stand-in that's actually shippable.
- **Commercial-detector API validation** (Originality.ai/Turnitin/ZeroGPT) — needs paid keys; the
  local proxy ensemble is the offline approximation.
- **Web UI** and **marketplace publishing automation** — out of scope for a CLI/skill.
- **Back-translation / token-mixing** attack modules — candidate future detectors-vs-attacks work.

## License

MIT — see [LICENSE](LICENSE).
