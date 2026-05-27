# qlora_finetune.py
# QLoRA fine-tuning of Llama 3.1 8B on HPIE training data
#
# Requirements (install first):
#   pip install transformers peft bitsandbytes accelerate datasets trl
#
# Run on: Google Colab Pro (T4 16GB) or any GPU with >=16GB VRAM
# Expected training time: ~45 minutes on T4 for 500 samples
#
# Usage:
#   python qlora_finetune.py
#   (or paste into Google Colab cell by cell)

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME    = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR    = "./hpie-llama-qlora"
TRAIN_FILE    = "hpie_train.jsonl"
EVAL_FILE     = "hpie_eval.jsonl"
MAX_SEQ_LEN   = 2048
BATCH_SIZE    = 2
GRAD_ACCUM    = 4       # effective batch = 2 * 4 = 8
EPOCHS        = 3
LR            = 2e-4
LORA_R        = 16      # rank
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05

# ── 1. Load training data ─────────────────────────────────────────────────────
print("Loading training data...")

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

train_records = load_jsonl(TRAIN_FILE)
eval_records  = load_jsonl(EVAL_FILE)

print(f"Train: {len(train_records)} samples")
print(f"Eval : {len(eval_records)} samples")

# ── 2. Format as chat-style prompt ────────────────────────────────────────────
# Alpaca format: instruction + input -> output
def format_prompt(example):
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )

train_dataset = Dataset.from_list([
    {"text": format_prompt(r)} for r in train_records
])
eval_dataset = Dataset.from_list([
    {"text": format_prompt(r)} for r in eval_records
])

print(f"\nSample training prompt (first 500 chars):")
print(train_dataset[0]["text"][:500])
print("...")

# ── 3. Load base model in 4-bit (QLoRA) ──────────────────────────────────────
print("\nLoading base model in 4-bit quantisation...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,     # double quant saves more memory
    bnb_4bit_quant_type="nf4",          # NormalFloat4 — best for LLMs
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)

print("Base model loaded in 4-bit.")

# ── 4. Apply LoRA adapters ────────────────────────────────────────────────────
print("Applying LoRA adapters...")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    # Target the attention projection layers — standard for Llama
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected output: trainable params ~40M / total ~8B = ~0.5%

# ── 5. Training arguments ─────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    optim="paged_adamw_8bit",           # memory-efficient optimizer for QLoRA
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    fp16=False,
    bf16=True,                          # use bfloat16 on modern GPUs
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none",                   # set to "wandb" if you use W&B
    dataloader_num_workers=0,
)

# ── 6. SFT Trainer ────────────────────────────────────────────────────────────
print("Initialising SFT trainer...")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    max_seq_length=MAX_SEQ_LEN,
    dataset_text_field="text",
    packing=False,
)

# ── 7. Train ──────────────────────────────────────────────────────────────────
print("\nStarting QLoRA fine-tuning...")
print(f"Model   : {MODEL_NAME}")
print(f"Epochs  : {EPOCHS}")
print(f"LR      : {LR}")
print(f"LoRA r  : {LORA_R}, alpha: {LORA_ALPHA}")
print(f"Batch   : {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM} effective")
print("-" * 50)

trainer.train()

# ── 8. Save the LoRA adapter ──────────────────────────────────────────────────
print("\nSaving LoRA adapter...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nDone. Adapter saved to: {OUTPUT_DIR}/")
print("\nNext step: run  python integrate_qlora.py")
print("to load the adapter into your HPIE app.")
