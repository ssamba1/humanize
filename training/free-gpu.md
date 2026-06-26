# Train on a FREE online GPU (Colab / Kaggle)

Free 16GB T4 GPUs are enough to train the moat — no DigitalOcean, no cost. **Kaggle is the best free
option** (30 GPU-hrs/week, reliable); Colab is easiest but flakier.

## VRAM → model choice (free GPUs are 16GB T4)
| Model | Fits 16GB? | Flag |
|---|---|---|
| Qwen2.5-1.5B + LoRA | ✅ comfortable | (default LoRA) |
| **Qwen2.5-3B + QLoRA (4-bit)** | ✅ tight, works | `--load-4bit` |
| Qwen2.5-3B + full LoRA | ❌ OOM on 16GB | use `--load-4bit` |

## Kaggle (recommended — 30 GPU-hrs/week)
1. kaggle.com → Create → **Notebook** → Settings → **Accelerator: GPU T4 x2** (or P100). Internet: **On**.
2. In a cell:
```python
!git clone https://github.com/ssamba1/untell.git    # private repo: use a token URL or upload a zip
%cd untell
!pip install -q -e ".[train,full]"
!python -m training.rl_humanizer --smoke               # free dry-run, proves it works (~minutes)
# real train (3B fits a T4 with 4-bit). full-tier step ~100-130s on a T4, so keep steps under the
# ~12h Kaggle/Colab session wall. Reward plateaus by ~step 90; 150 captures the gains in ~5h:
!python -m training.rl_humanizer --model Qwen/Qwen2.5-3B-Instruct --tier full --steps 150 --load-4bit
# or DPO (often steadier): !python -m training.dpo_humanizer --model Qwen/Qwen2.5-3B-Instruct --tier full --load-4bit
```
3. Download `out/rl-humanizer` from the notebook's Output. Done — $0. The trainer checkpoints every
   25 steps, so a session killed at the wall still leaves a usable adapter (`out/rl-humanizer/checkpoint-*`).

## Google Colab (easiest)
- colab.research.google.com → Runtime → Change runtime type → **T4 GPU**.
- Same cells as above. Save `out/...` to Google Drive (`from google.colab import drive; drive.mount('/content/drive')`)
  because Colab wipes the VM on disconnect.
- Free Colab has session/idle limits (~12h, disconnects) — checkpoint to Drive; Colab Pro ($10/mo) gives
  L4/A100 + longer runs if you want headroom.

## Notes
- Private repo: pass a GitHub token in the clone URL (`https://TOKEN@github.com/ssamba1/untell.git`),
  or upload the repo as a dataset/zip.
- `--tier full` reward downloads the OSS detectors (RADAR/Binoculars optional) on first run; `--tier
  lite` is faster but a weaker reward. `--tier commercial` needs detector API keys set in the notebook.
- Always run `--smoke` first — it surfaces any trl/peft version drift in minutes before the real train.
