"""Reward function for RL-against-ensemble training (StealthRL / AuthorMist style).

reward = (1 - P(AI) from the target detector) - meaning-drift penalty - quality penalty

By default the target is our local ensemble (`score_text` max). **But the local ensemble does not
predict commercial detectors** (measured: RADAR 0.008 vs GPTZero 100% on the same humanized text), so
training against it produces a model that beats the local proxies and still fails GPTZero. To target a
real detector, train a surrogate (`training/surrogate.py`) and set ``UNTELL_SURROGATE_DIR`` — the reward
then uses the surrogate's P(AI) instead of the local ensemble, with no other change. That is the
difference between "learns to fool roberta/hc3" and "learns to fool a model that mimics GPTZero".

Pure-python over the chosen detector + semantic similarity, so it is testable on the lite tier with no
GPU (the surrogate path needs `.[train]` + a trained surrogate dir).
"""

import os
import re

from untell.scripts.quality import similarity
from untell.scripts.score import score_text

_W = re.compile(r"[A-Za-z']+")

_SURROGATE = None  # lazily-loaded SurrogateDetector when UNTELL_SURROGATE_DIR is set


def target_ai_score(text: str, tier: str = "full") -> float:
    """P(AI) from the training target: a GPTZero-mimicking surrogate if `UNTELL_SURROGATE_DIR` is set,
    else the local detector ensemble (max). Optimizing the surrogate is the only path that transfers
    to the commercial detector it was distilled from."""
    sd = os.environ.get("UNTELL_SURROGATE_DIR")
    if sd:
        global _SURROGATE
        if _SURROGATE is None:
            from training.surrogate import SurrogateDetector

            _SURROGATE = SurrogateDetector(sd)
        return float(_SURROGATE.score(text))
    return float(score_text(text, tier=tier)["max"])


def fluency(text: str) -> float:
    """Cheap quality proxy in [0,1]: distinct-bigram ratio (1.0 = no repetition, low = degenerate)."""
    words = [w.lower() for w in _W.findall(text)]
    if len(words) < 4:
        return 1.0
    bigrams = list(zip(words, words[1:]))
    return len(set(bigrams)) / len(bigrams)


def humanness_reward(
    original: str,
    candidate: str,
    *,
    tier: str = "full",
    sim_floor: float = 0.76,
    w_quality: float = 0.25,
) -> float:
    """Multi-objective reward: evasion + meaning + quality.

    reward = (1 - max P(AI) across our ensemble)            # evade every detector incl. the hard ones
             - meaning_drift_penalty (below the sim floor)   # don't mangle meaning
             - w_quality * (1 - fluency)                     # don't degenerate into repetition/garbage

    Targeting all three at once is the impossibility-triangle win competitors miss (they reward only
    evasion, so quality rots). Reward against ``tier="full"`` (RADAR + ensemble) or ``"commercial"``.
    """
    if not candidate.strip():
        return -1.0
    ai = target_ai_score(candidate, tier=tier)  # surrogate if UNTELL_SURROGATE_DIR set, else ensemble
    sim = similarity(original, candidate)
    evade = 1.0 - ai
    meaning_penalty = 0.0 if sim >= sim_floor else (sim_floor - sim) * 2.0
    quality_penalty = w_quality * (1.0 - fluency(candidate))
    return round(evade - meaning_penalty - quality_penalty, 4)


def batch_rewards(original: str, candidates: list[str], *, tier: str = "full", sim_floor: float = 0.76) -> list[float]:
    """Rewards for several candidate rewrites of one source (GRPO scores a group per prompt)."""
    return [humanness_reward(original, c, tier=tier, sim_floor=sim_floor) for c in candidates]
