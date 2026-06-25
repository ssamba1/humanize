"""DPO training of the humanizer — often more stable than GRPO on small GPUs. GPU-only, not run in CI.

Preference pairs: ``chosen`` = a humanized rewrite that passed our ensemble (from the loop, via
``training.distill``), ``rejected`` = the AI original. DPO teaches the model to prefer human-reading
text over AI text. Run distill first (needs a rewriter/teacher key), then DPO.

    pip install -e ".[train,full,api]" && export ANTHROPIC_API_KEY=...
    python -m training.dpo_humanizer --dataset raid --n 1000 --model Qwen/Qwen2.5-3B-Instruct
    python -m training.dpo_humanizer --smoke      # tiny model, 2 steps, synthetic pairs (no key) — proves it runs
"""

from __future__ import annotations

import argparse

SMOKE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_PROMPT = "Rewrite the following text so it reads as natural human writing while preserving its exact meaning:\n\n{text}"


def build_pairs(dataset: str = "builtin", n: int = 200, tier: str = "full") -> dict:
    """chosen = the loop's humanized output (passed the ensemble), rejected = the AI source."""
    from training.distill import distill

    out = distill(dataset, n=n, tier=tier)
    pairs = [{"prompt": r["prompt"], "chosen": r["humanized"], "rejected": r["source"]} for r in out["rows"]]
    return {"pairs": pairs, "kept": out["kept"], "total": out["total"]}


def _smoke_pairs(n: int = 8) -> list[dict]:
    """Synthetic pairs (no loop/key) — just to prove the DPO training loop runs end-to-end."""
    from eval.datasets import load_samples

    return [
        {"prompt": _PROMPT.format(text=s), "chosen": "Plainly put: " + s[:60], "rejected": s}
        for s in load_samples("builtin", n)
    ]


def train(
    model_id: str = "Qwen/Qwen2.5-3B-Instruct",
    dataset: str = "builtin",
    n: int = 1000,
    tier: str = "full",
    out: str = "out/dpo-humanizer",
    smoke: bool = False,
):
    """LoRA DPO. Heavy deps imported here so the module stays importable without a GPU."""
    import torch  # noqa: F401
    from datasets import Dataset
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer

    if smoke:
        model_id, out = SMOKE_MODEL, "out/dpo-smoke"
        pairs = _smoke_pairs()
    else:
        pairs = build_pairs(dataset, n=n, tier=tier)["pairs"]

    dataset_obj = Dataset.from_list(pairs)
    cfg = DPOConfig(
        output_dir=out,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        bf16=True,
        max_steps=2 if smoke else -1,
        num_train_epochs=1,
        logging_steps=10,
        beta=0.1,
    )
    lora = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = DPOTrainer(model=model_id, args=cfg, train_dataset=dataset_obj, peft_config=lora)
    trainer.train()
    trainer.save_model(out)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="training.dpo_humanizer", description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--dataset", default="builtin")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--tier", default="full", choices=["lite", "full", "heavy", "commercial"])
    parser.add_argument("--out", default="out/dpo-humanizer")
    parser.add_argument("--smoke", action="store_true", help="tiny model + 2 steps + synthetic pairs (proves it runs)")
    args = parser.parse_args(argv)
    path = train(model_id=args.model, dataset=args.dataset, n=args.n, tier=args.tier, out=args.out, smoke=args.smoke)
    print(f"saved DPO policy -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
