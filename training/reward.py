"""Reward function for RL-against-ensemble training (StealthRL / AuthorMist style).

reward = (1 - max P(AI) across our detector ensemble) - meaning-drift penalty

Pure-python over our own detectors + semantic similarity, so it is testable on the lite tier with no
GPU. The RL trainer (rl_humanizer.py) calls this to score generated paraphrases; train against the
``full`` tier (or ``commercial`` with keys) to learn a humanize-by-default policy.
"""

import re

from humanize.scripts.quality import similarity
from humanize.scripts.score import score_text

_W = re.compile(r"[A-Za-z']+")


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
    ai = float(score_text(candidate, tier=tier)["max"])
    sim = similarity(original, candidate)
    evade = 1.0 - ai
    meaning_penalty = 0.0 if sim >= sim_floor else (sim_floor - sim) * 2.0
    quality_penalty = w_quality * (1.0 - fluency(candidate))
    return round(evade - meaning_penalty - quality_penalty, 4)


def batch_rewards(original: str, candidates: list[str], *, tier: str = "full", sim_floor: float = 0.76) -> list[float]:
    """Rewards for several candidate rewrites of one source (GRPO scores a group per prompt)."""
    return [humanness_reward(original, c, tier=tier, sim_floor=sim_floor) for c in candidates]
