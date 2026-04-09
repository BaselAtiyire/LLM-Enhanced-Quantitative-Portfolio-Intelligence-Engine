# integrate_qlora.py
# Loads the fine-tuned QLoRA adapter into HPIE as the reasoning agent
# Drop this file into your project folder alongside app_chatbot.py
#
# Usage: called automatically by app_chatbot.py when USE_QLORA=true in .env

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_MODEL  = "mistralai/Mistral-7B-Instruct-v0.3"
ADAPTER_DIR = "./hpie-llama-qlora"   # output from qlora_finetune.py
MAX_NEW_TOKENS = 512
TEMPERATURE    = 0.1    # low temperature for factual, grounded responses

# ── Load model with QLoRA adapter ────────────────────────────────────────────
def load_qlora_model():
    print("Loading QLoRA fine-tuned HPIE model...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    print("QLoRA model loaded successfully.")
    return model, tokenizer


# ── Inference function ────────────────────────────────────────────────────────
def qlora_chat(model, tokenizer, context_str: str, question: str) -> str:
    """
    Runs inference with the fine-tuned HPIE model.
    context_str : the formatted ranking table string
    question    : the user's question
    Returns     : the model's grounded response
    """
    prompt = (
        f"### Instruction:\n"
        f"You are a quantitative financial analyst assistant. "
        f"Answer the user's question using ONLY the data provided in the context. "
        f"Do not generate, infer, or hallucinate any numerical values "
        f"not explicitly present in the dataset. "
        f"If a metric is not in the dataset, say it is missing.\n\n"
        f"### Input:\n"
        f"{context_str}\n\n"
        f"User question: {question}\n\n"
        f"### Response:\n"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (not the prompt)
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response   = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return response.strip()


# ── Test locally ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model, tokenizer = load_qlora_model()

    test_context = """RANKING DATASET (use ONLY this data):
Rank  Ticker  Score    1W%      30D%     AnnVol%    Sharpe   MaxDD%    P/E      MktCap$B
1     GOOGL   0.697    11.34    -1.87    30.43      0.681    -44.32    28.3     3695
2     AAPL    0.638    1.89     -4.76    21.28      0.497    -33.36    32.1     3726
3     NVDA    0.626    6.32     -7.02    37.78      1.160    -66.34    36.3     4329
20    TSLA    0.132    -4.20    -13.30   39.26      0.369    -73.63    324.0    1301"""

    questions = [
        "Why is GOOGL ranked first?",
        "Which stock has the highest Sharpe ratio?",
        "What will TSLA's price be tomorrow?",   # should be refused
        "What is AAPL's EPS?",                   # should be refused (not in dataset)
    ]

    print("\n" + "="*60)
    print("HPIE QLoRA Model — Test Inference")
    print("="*60)

    for q in questions:
        print(f"\nQ: {q}")
        response = qlora_chat(model, tokenizer, test_context, q)
        print(f"A: {response}")
        print("-"*60)
