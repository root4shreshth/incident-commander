"""Build training/train_sft.ipynb from scratch.

Generates a tight, self-contained Colab notebook that runs the SFT pipeline
end-to-end: setup, model load, dataset build, SFT training, evaluation,
plots, and an optional Hub push. Each code cell is preceded by a markdown
cell stating expected runtime.

The notebook saves intermediate state to /content/checkpoints/ so a Colab
disconnect mid-SFT can resume without restarting the long phases.
"""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "training" / "train_sft.ipynb"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip("\n").split("\n")],
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in source.strip("\n").split("\n")],
    }


# ===========================================================================
# CELLS
# ===========================================================================

cells = []

# --- 1. Title -------------------------------------------------------------
cells.append(md(r"""
# Praetor - SFT training notebook

This notebook trains the **Praetor** incident-response agent end-to-end:

1. **SFT** (supervised fine-tune) the policy on senior-SRE behavioral-clone
   trajectories drawn from `IDEAL_TRAJECTORIES` in `coach.py`.
2. **Evaluate** the trained model on held-out seeds across the scenario
   families covered by SFT.
3. **Render** the canonical plots into `/content/results/` for the README.
4. **Push** the LoRA adapter to your HuggingFace Hub.

### Stack

This notebook uses the **vanilla HuggingFace stack** (transformers + peft +
bitsandbytes + trl) so it runs on free Colab T4 as well as A100 / L40S.

### Total runtime budget

| Phase | A100 / L40S | T4 |
|---|---:|---:|
| Setup + clone | 5 min | 5 min |
| Load model (4-bit Qwen 1.5B + LoRA) | 4 min | 5 min |
| SFT (1 epoch, skipped if `sft.done` exists) | ~60 min | ~120 min |
| SFT sanity-eval (9 episodes) | ~5 min | ~8 min |
| Final eval (60 episodes) | ~8 min | ~15 min |
| Plots + push | 3-5 min | 3-5 min |
| **Total** | **~90 min** | **~155 min** |

If `sft.done` exists from a prior run, the SFT cell becomes a no-op and total
drops by ~60 min on A100/L40S. The adapter at
`/content/checkpoints/sft_adapter/` survives runtime restarts.

### Before you run

- **Runtime -> Change runtime type -> A100 / L40S GPU** (or T4 if unavailable).
- **Run all** from the menu, then walk away. The notebook writes intermediate
  state to `/content/checkpoints/` so a disconnect doesn't lose work.
- If any cell errors, the next cell will refuse to start. Fix the error,
  re-run the failing cell, then continue.
"""))

# --- 2. Setup -------------------------------------------------------------
cells.append(md(r"""
## Cell 1 - Setup: install dependencies + clone the repo

**Expected runtime: 2-3 minutes.** Most of the wall-clock is pip resolving
torch/transformers/trl together.
"""))

cells.append(code(r"""
import time, os, sys
_t0 = time.monotonic()

# Vanilla HF/PEFT/TRL stack.
%pip install -q --upgrade pip 2>&1 | tail -1
%pip install -q "torch>=2.4" "transformers>=4.46,<4.50" "trl==0.15.2" "peft>=0.13" "accelerate>=0.34" "bitsandbytes>=0.43" 2>&1 | tail -1
%pip install -q "datasets>=2.14" "huggingface-hub>=0.20" "matplotlib>=3.7" 2>&1 | tail -1
%pip install -q "pydantic>=2.0" "fastapi>=0.104" 2>&1 | tail -1

# Clone (or update) the project so imports like `training.eval_runner` resolve.
if not os.path.exists("/content/incident-commander"):
    !git clone -q https://github.com/root4shreshth/incident-commander.git /content/incident-commander
else:
    !cd /content/incident-commander && git pull -q
sys.path.insert(0, "/content/incident-commander")

# Change CWD to project root so `from datasets import Dataset` resolves to
# the HF library, not training/datasets.py.
os.chdir("/content/incident-commander")
print(f"CWD: {os.getcwd()}")

os.makedirs("/content/checkpoints", exist_ok=True)
os.makedirs("/content/results", exist_ok=True)

# Surface any pip resolution issues now, not 30 min into a run.
from trl import SFTTrainer, SFTConfig
print(f"Setup OK in {time.monotonic()-_t0:.1f}s. Vanilla stack, trl==0.15.2.")
"""))

# --- 3. Compute detection ------------------------------------------------
cells.append(md(r"""
## Cell 2 - Compute detection

**Expected runtime: < 1 second.** Reports the GPU and adjusts later cells'
batch sizes accordingly.
"""))

cells.append(code(r"""
import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "No GPU detected. Runtime -> Change runtime type -> A100 (or T4)."
    )
GPU_NAME = torch.cuda.get_device_name(0)
GPU_MEM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
IS_BIG_GPU = any(tag in GPU_NAME for tag in ("A100", "L40S", "H100", "A6000"))

# Batch size scales with GPU memory.
SFT_BATCH = 4 if IS_BIG_GPU else 2
SFT_GRAD_ACCUM = 4 if IS_BIG_GPU else 8        # effective batch = 16 either way

print(f"GPU            : {GPU_NAME} ({GPU_MEM_GB:.1f} GB)")
print(f"SFT batch      : {SFT_BATCH} x grad_accum {SFT_GRAD_ACCUM} (eff. {SFT_BATCH*SFT_GRAD_ACCUM})")
"""))

# --- 4. Load model -------------------------------------------------------
cells.append(md(r"""
## Cell 3 - Load Qwen2.5-Coder-1.5B in 4-bit with LoRA

**Expected runtime: 3-5 minutes.** Downloads the base model, applies 4-bit
NF4 quantization, and attaches a LoRA r=16 adapter ready for SFT.

Skip-if-already-done: if `model` is already in scope, this cell is a no-op.
"""))

cells.append(code(r"""
import time
_t0 = time.monotonic()

if "model" not in globals():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    MAX_SEQ_LEN = 2048

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if IS_BIG_GPU else torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print(f"Model loaded + LoRA attached in {time.monotonic()-_t0:.1f}s")
else:
    print("Model already loaded; skipping.")
"""))

# --- 5. SFT dataset ------------------------------------------------------
cells.append(md(r"""
## Cell 4 - Build the SFT chat dataset

**Expected runtime: < 30 seconds.** Materializes ~120 (system, user, assistant)
chat rows from `IDEAL_TRAJECTORIES` in `incident_commander_env/server/coach.py`,
sampled across multiple seeds per family so each scenario gets several
parametric variants.
"""))

cells.append(code(r"""
import time
_t0 = time.monotonic()

from training.datasets import build_sft_dataset, to_chat_messages, SYSTEM_PROMPT
from datasets import Dataset

raw_rows = build_sft_dataset(n_seeds_per_family=8)
hf_rows = []
for r in raw_rows:
    msgs = to_chat_messages(r)
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    hf_rows.append({"text": text})

sft_ds = Dataset.from_list(hf_rows)
print(f"SFT dataset: {len(sft_ds)} rows  ({time.monotonic()-_t0:.1f}s)")
print(f"  example length (chars): {len(sft_ds[0]['text'])}")
print(f"  scenarios covered     : {sorted({r['scenario'] for r in raw_rows})}")
"""))

# --- 6. SFT training ----------------------------------------------------
cells.append(md(r"""
## Cell 5 - SFT training (1 epoch)

**Expected runtime: 60 minutes on A100/L40S, 120 minutes on T4.** This is
the longest cell; you'll see TRL's per-step progress bar.

Skip-if-already-done: if `/content/checkpoints/sft.done` exists from a
previous successful run, the cell loads weights from
`/content/checkpoints/sft_adapter/` instead of re-training.
"""))

cells.append(code(r"""
import os, time
_t0 = time.monotonic()

SFT_CHECKPOINT = "/content/checkpoints/sft_adapter"
SFT_DONE = "/content/checkpoints/sft.done"

if os.path.exists(SFT_DONE) and os.path.exists(SFT_CHECKPOINT):
    print("SFT already complete (sft.done exists). Loading adapter from checkpoint.")
    from peft import PeftModel
    model.load_adapter(SFT_CHECKPOINT, adapter_name="default", is_trainable=True)
else:
    from trl import SFTTrainer, SFTConfig

    sft_args = SFTConfig(
        output_dir="/content/sft_runs",
        per_device_train_batch_size=SFT_BATCH,
        gradient_accumulation_steps=SFT_GRAD_ACCUM,
        num_train_epochs=1,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_steps=5,
        save_strategy="no",
        bf16=IS_BIG_GPU, fp16=not IS_BIG_GPU,
        max_seq_length=2048,
        dataset_text_field="text",
        report_to="none",
        seed=42,
    )
    sft_trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=sft_ds, args=sft_args,
    )
    sft_trainer.train()
    model.save_pretrained(SFT_CHECKPOINT)
    tokenizer.save_pretrained(SFT_CHECKPOINT)
    open(SFT_DONE, "w").write("done")

print(f"SFT phase complete in {time.monotonic()-_t0:.1f}s")
"""))

# --- 7. SFT sanity-eval --------------------------------------------------
cells.append(md(r"""
## Cell 6 - SFT sanity-eval (9 episodes)

**Expected runtime: ~5 minutes on A100, ~8 minutes on T4.** Quick
random-vs-SFT comparison on 3 seeds x 3 families to confirm SFT learned
something before we spend additional time on the full eval.

If SFT scores are below random, abort and inspect the SFT loss curve.
"""))

cells.append(code(r"""
import time, warnings, logging
warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
warnings.filterwarnings("ignore", category=FutureWarning,
                        module=r"transformers\.modeling_attn_mask_utils")
logging.getLogger("transformers").setLevel(logging.ERROR)

from training.eval_runner import evaluate, random_policy, hf_pipeline_policy
from training.datasets import SYSTEM_PROMPT
from transformers import GenerationConfig

# Clear Qwen's default max_length=32768 + use the right stop tokens.
if hasattr(model, "generation_config") and model.generation_config is not None:
    model.generation_config.max_length = None
_eos_ids = [tokenizer.eos_token_id]
_im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
if _im_end is not None and _im_end != tokenizer.unk_token_id and _im_end not in _eos_ids:
    _eos_ids.append(_im_end)

GEN_CFG = GenerationConfig(
    max_new_tokens=160,
    do_sample=False,
    eos_token_id=_eos_ids,
    pad_token_id=tokenizer.eos_token_id,
)

def hf_generate(messages, max_new=160):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]
    out = model.generate(**inputs, generation_config=GEN_CFG)
    # Decode ONLY the newly-generated tokens.
    new_tokens = out[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

SANITY_FAMILIES = ["oom_crash", "db_pool_exhaustion", "bad_deployment_cascade"]
SANITY_SEEDS = [2000, 2001, 2002]
TOTAL = len(SANITY_FAMILIES) * len(SANITY_SEEDS)

def _progress(idx, total, family, seed, ep):
    tag = "OK" if ep.resolved else "X "
    print(f"  [{idx:>2}/{total}] [{tag}] {family:<25s} seed={seed} steps={ep.steps_used:>2} score={ep.score:.2f}", flush=True)

_t0 = time.monotonic()
print("-> random baseline")
report_random = evaluate("random", random_policy(rng_seed=99),
                         SANITY_FAMILIES, SANITY_SEEDS,
                         system_prompt=SYSTEM_PROMPT, on_episode=_progress)

print("\n-> SFT model")
report_sft = evaluate("sft", hf_pipeline_policy(hf_generate),
                      SANITY_FAMILIES, SANITY_SEEDS,
                      system_prompt=SYSTEM_PROMPT, on_episode=_progress)

print(f"\nSanity eval done in {time.monotonic()-_t0:.1f}s")
print("=" * 60)
for cond, rpt in [("random", report_random), ("sft", report_sft)]:
    print(f"\n  {cond}:")
    for fam, stats in rpt.by_family.items():
        print(f"    {fam:<25s}: success={stats['success_rate']*100:>4.0f}%  "
              f"score={stats['avg_score']:.2f}  steps={stats['avg_steps_used']:.1f}")
"""))

# --- 8. Final eval -------------------------------------------------------
cells.append(md(r"""
## Cell 7 - Final eval (random vs SFT)

**Expected runtime: 8 minutes on A100, 15 minutes on T4.**
6 families x 10 held-out seeds x 2 conditions = 120 episodes total.
"""))

cells.append(code(r"""
import time

if hasattr(model, "generation_config") and model.generation_config is not None:
    model.generation_config.max_length = None

EVAL_FAMILIES = [
    "oom_crash", "db_pool_exhaustion", "bad_deployment_cascade",
    "disk_full", "slow_query", "cert_expiry",
]
FINAL_SEEDS = list(range(8000, 8010))   # 10 held-out seeds
TOTAL_FINAL = len(EVAL_FAMILIES) * len(FINAL_SEEDS)

print(f"Final eval - {TOTAL_FINAL} episodes per condition x 2 conditions = {TOTAL_FINAL*2} total\n")

_t0 = time.monotonic()
print("-> random baseline")
report_random_final = evaluate("random", random_policy(rng_seed=12345),
                               EVAL_FAMILIES, FINAL_SEEDS,
                               system_prompt=SYSTEM_PROMPT, on_episode=_progress)
print(f"  random done in {time.monotonic()-_t0:.1f}s\n")

_t1 = time.monotonic()
print("-> SFT model")
report_sft_final = evaluate("sft", hf_pipeline_policy(hf_generate),
                            EVAL_FAMILIES, FINAL_SEEDS,
                            system_prompt=SYSTEM_PROMPT, on_episode=_progress)
print(f"  sft done in {time.monotonic()-_t1:.1f}s")

print("\n" + "=" * 70)
print("Final results (success rate per family):")
print("=" * 70)
print(f"{'family':<28s}  {'random':>8s}  {'sft':>8s}  {'delta':>8s}")
for fam in EVAL_FAMILIES:
    r_rate = report_random_final.by_family.get(fam, {}).get("success_rate", 0) * 100
    s_rate = report_sft_final.by_family.get(fam, {}).get("success_rate", 0) * 100
    delta = s_rate - r_rate
    sign = "+" if delta >= 0 else ""
    print(f"{fam:<28s}  {r_rate:>6.0f}%  {s_rate:>6.0f}%  {sign}{delta:>5.0f}pp")
"""))

# --- 9. Plots ------------------------------------------------------------
cells.append(md(r"""
## Cell 8 - Render the canonical plots

**Expected runtime: < 1 minute.** Saves all plots to `/content/results/` so
you can download them directly into the README.
"""))

cells.append(code(r"""
import time, json
from collections import defaultdict
from training.plots import (
    make_success_bars, make_action_distribution, save_figure,
)

_t0 = time.monotonic()
out_dir = "/content/results"

# 1) Success bars: random vs SFT across families
save_figure(
    make_success_bars(
        reports_by_condition={
            "random": report_random_final.by_family,
            "sft":    report_sft_final.by_family,
        },
        families=EVAL_FAMILIES,
    ),
    f"{out_dir}/final_success_rates.png",
)
print("  wrote final_success_rates.png")

# 2) Action distributions
def _actions(report):
    out = defaultdict(int)
    for ep in report.episodes:
        for a, _ in ep.actions:
            out[a] += 1
    return dict(out)

save_figure(
    make_action_distribution(actions_by_condition={
        "random": _actions(report_random_final),
        "sft":    _actions(report_sft_final),
    }),
    f"{out_dir}/final_action_distribution.png",
)
print("  wrote final_action_distribution.png")

# 3) Eval summary JSON for the README
summary = {
    "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "lora_r": 16,
    "lora_alpha": 32,
    "eval_seeds": FINAL_SEEDS,
    "by_family": {
        fam: {
            "random_success": report_random_final.by_family[fam]["success_rate"],
            "sft_success":    report_sft_final.by_family[fam]["success_rate"],
            "random_avg_score": report_random_final.by_family[fam]["avg_score"],
            "sft_avg_score":    report_sft_final.by_family[fam]["avg_score"],
        }
        for fam in EVAL_FAMILIES
    },
}
with open(f"{out_dir}/eval_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("  wrote eval_summary.json")

print(f"\nAll plots written to {out_dir}/ in {time.monotonic()-_t0:.1f}s")
"""))

# --- 10. Push to Hub -----------------------------------------------------
cells.append(md(r"""
## Cell 9 - Push LoRA adapter to your HuggingFace Hub (optional)

**Expected runtime: 2-3 minutes.** Skipped if the `HF_USER` and `HF_TOKEN`
environment variables aren't set. Set them in Colab via:
> Sidebar -> Secrets -> add `HF_TOKEN` (write scope) and `HF_USER`.
"""))

cells.append(code(r"""
import os, time
_t0 = time.monotonic()

HF_USER = os.environ.get("HF_USER", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

if not HF_USER or not HF_TOKEN:
    print("HF_USER / HF_TOKEN not set - skipping Hub push.")
    print("To enable: Colab sidebar -> Secrets -> add both, then re-run this cell.")
else:
    from huggingface_hub import login, create_repo
    login(token=HF_TOKEN)
    repo_id = f"{HF_USER}/praetor-incident-commander-sft"
    create_repo(repo_id, exist_ok=True)
    model.push_to_hub(repo_id, token=HF_TOKEN)
    tokenizer.push_to_hub(repo_id, token=HF_TOKEN)
    print(f"Pushed LoRA + tokenizer to https://huggingface.co/{repo_id}")
    print("  (paste this URL into the README's 'Trained LoRA adapter' line)")

print(f"\nDone in {time.monotonic()-_t0:.1f}s")
"""))

# --- 11. README results table ------------------------------------------
cells.append(md(r"""
## Cell 10 - Print the README results table

**Expected runtime: < 1 second.** Copy this block straight into the
README's *Eval results* section, replacing the placeholder rows.
"""))

cells.append(code(r"""
print()
print("Paste the block below into README.md -> 'Eval results' section:")
print()
print("=" * 70)
print()

print("| Condition | OOM Crash | DB Pool | Bad Deploy | Disk Full | Slow Query | Cert Expiry | Average |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")

def _row(name, by_fam, bold=False):
    rates = [by_fam.get(f, {}).get("success_rate", 0) * 100 for f in EVAL_FAMILIES]
    avg = sum(rates) / len(rates)
    label = f"**{name}**" if bold else name
    cells = " | ".join(f"{r:>3.0f}%" for r in rates)
    print(f"| {label} | {cells} | **{avg:>3.0f}%** |")

_row("Random (n=60)", report_random_final.by_family)
_row("SFT (n=60)", report_sft_final.by_family, bold=True)

print()
print("=" * 70)
print()
print("Plot files (also commit these into results/ on GitHub):")
print("  /content/results/final_success_rates.png")
print("  /content/results/final_action_distribution.png")
print("  /content/results/eval_summary.json")
"""))


# ===========================================================================
# Build notebook
# ===========================================================================

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
        "colab": {"provenance": []},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
with NB_PATH.open("w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Wrote {NB_PATH}")
print(f"  cells: {len(cells)} ({sum(1 for c in cells if c['cell_type']=='code')} code, {sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")
