# 02_finetuning

Fine-tune **Ministral-8B-Instruct** on dbt DAG generation using QLoRA on Google Colab,
then export to GGUF for local inference with LM Studio.

## Why Ministral-8B?

| Model | Params | Context | GGUF size (Q4_K_M) | LM Studio compatible |
|---|---|---|---|---|
| Ministral-8B-Instruct-2410 | 8B | 128k | ~4.5 GB | ✅ |
| Mistral-7B-Instruct-v0.3 | 7B | 32k | ~4.1 GB | ✅ |
| Ministral-3B-Instruct | 3B | 128k | ~1.8 GB | ✅ |

Ministral-8B is the best starting point: strong instruction following, large context window
(useful if you extend to multi-model prompts), and fits comfortably in a T4 (15 GB VRAM)
with 4-bit quantisation.

## Prerequisites

- Google account (free Colab T4 GPU is sufficient)
- HuggingFace account + token (for downloading the base model)
  - Accept the Ministral-8B licence at: https://huggingface.co/mistralai/Ministral-8B-Instruct-2410

## Files

| File | Purpose |
|---|---|
| `prepare_dataset.py` | Convert `finetune_dataset.jsonl` → Mistral chat format, 90/10 train/eval split |
| `mistral_dbt_finetune.ipynb` | Colab notebook: load model, LoRA, train, export GGUF |
| `evaluate_baseline.py` | Run the eval set against any LM Studio model to measure pre/post fine-tune quality |
| `data/train.jsonl` | 900-row training set (chat format) |
| `data/eval.jsonl` | 100-row eval set (chat format) |

## Step-by-step

### Step 0 — Prepare the dataset (already done)

```bash
cd 02_finetuning
python prepare_dataset.py
# → data/train.jsonl  (900 rows)
# → data/eval.jsonl   (100 rows)
```

### Step 1 — Baseline evaluation (optional but recommended)

Load **Ministral-8B-Instruct** (unmodified) in LM Studio, then:

```bash
python evaluate_baseline.py --model "ministral-8b" --limit 50
# Measures dbt_parse_pass_rate BEFORE fine-tuning
# Saved to results/baseline.json
```

This tells you how much the base model already knows about dbt. Expect ~10-30% pass rate.

### Step 2 — Fine-tune on Colab

1. Open `mistral_dbt_finetune.ipynb` in Google Colab
   - Runtime → Change runtime type → **T4 GPU** (free tier)
2. Run cells in order:
   - Cell 0: GPU check
   - Cell 1: Install dependencies (~3 min)
   - Cell 2: Upload `data/train.jsonl` and `data/eval.jsonl` when prompted
   - Cell 3: Load Ministral-8B in 4-bit (~5 min download)
   - Cell 4: Attach LoRA adapters
   - Cell 5: Prepare datasets
   - Cell 6: Train (~20–40 min on T4 for 3 epochs)
   - Cell 7: Quick inference check
   - Cell 8: Save LoRA adapters
   - Cell 9: Export GGUF Q4_K_M and download (~4.5 GB)

### Step 3 — Load in LM Studio

1. Move the downloaded `.gguf` file to your LM Studio models folder:
   `~/Library/Application Support/LM Studio/models/`
2. Open LM Studio → My Models → the file should appear
3. Load and test with a prompt like:
   ```
   Business question: Show the total revenue per product category.

   SQL schemas:
   CREATE TABLE products (product_id INT, category VARCHAR, price DECIMAL);
   CREATE TABLE orders (order_id INT, product_id INT, quantity INT)
   ```

### Step 4 — Post fine-tune evaluation

```bash
python evaluate_baseline.py \
  --model "ministral-8b-dbt" \
  --output results/after_finetune.json
# Compare dbt_parse_pass_rate to results/baseline.json
```

## Expected results (T4, 3 epochs)

| Metric | Before fine-tune | After fine-tune (expected) |
|---|---|---|
| dbt_parse_pass_rate | ~15–30% | ~80–90% |
| has_staging_rate | ~40–60% | ~95%+ |
| has_marts_rate | ~50–70% | ~95%+ |
| correct_prefix_rate | ~30–50% | ~80–90% |
| Training time | — | ~25–40 min (T4) |

## LoRA config explained

| Param | Value | Why |
|---|---|---|
| `r` | 16 | Adapter rank — balance of capacity vs size |
| `lora_alpha` | 32 | Scaling factor (alpha/r = 2) |
| `target_modules` | all attention + MLP | Full attention fine-tuning for best quality |
| `max_seq_length` | 2048 | All rows fit under ~950 tokens; 2048 gives headroom |
| `epochs` | 3 | Enough for 900 examples without overfitting |
| `learning_rate` | 2e-4 | Standard LoRA LR |
| `batch_size` | 4 × 4 accum = 16 | Fits T4, effective batch of 16 |
