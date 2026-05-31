"""
run_inference.py - CSE 151B Competition Entry Point
=====================================================

Usage
-----
    # From Python:
    from run_inference import run_inference
    run_inference()

    # From the command line:
    python run_inference.py

Environment
-----------
    Tested on Google Colab (T4), Kaggle (T4 / P100), and UCSD DataHub.
    Requires a GPU. Set the HF_TOKEN environment variable (or Colab/Kaggle
    secret) to authenticate with HuggingFace before downloading the model.
"""

import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional


# 0. Lazy dependency installation

def _ensure_dependencies() -> None:
    packages = [
        "transformers==4.51.3",
        "protobuf>=4.0.0",
        "bitsandbytes",
        "accelerate",
        "sympy",
        "numpy",
        "tqdm",
        "huggingface_hub",
    ]
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *packages],
        check=True,
    )



# 1. Default configuration

DATA_PATH   = "data/private.jsonl"
OUTPUT_PATH = "results/submission.csv"
MODEL_NAME  = "Qwen/Qwen3-4B"

MAX_TOKENS  = 32768
TEMPERATURE = 0.2
TOP_P       = 0.85
TOP_K       = 20


# 2. Prompts

SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step.\n\n"

    " FINAL ANSWER FORMAT - THE MOST IMPORTANT RULE \n"
    "At the very LAST LINE of your response, place ALL answers inside exactly ONE \\boxed{}.\n"
    "  DO:     \\boxed{380, 315, 13, 310}  (all parts, comma-separated, one box)\n"
    "  DO:     \\boxed{5/8}  (single answer)\n"
    "  DON'T:  box each sub-answer in a separate \\boxed{} throughout the solution\n"
    "  DON'T:  \\boxed{380}  ...text...  \\boxed{315}  ...text...  \\boxed{13}\n"
    "Even if you use \\boxed{} for intermediate steps during working, you MUST finish with "
    "a single combined \\boxed{a, b, c} on the very last line - all answers, in the order asked.\n"
    "Never leave \\boxed{} empty.\n\n"

    "EXACT FORM RULES\n"
    "1. SYMBOLIC OVER NUMERIC - if the answer is a function applied to given constants, "
    "write the expression, NOT a decimal:\n"
    "  DO:     \\arctan(4.76)          DON'T: 1.3635\n"
    "  DO:     \\ln(0.5)/\\ln(0.96584)  DON'T: 19.94\n"
    "  DO:     (1/2)^{(1999-1963)/31} DON'T: 0.447\n"
    "2. DECIMAL PRECISION - when a decimal is required, give at minimum 6 significant digits:\n"
    "  DO:     7.79744   DON'T: 7.80  |  DO: 442.857   DON'T: 442.86  |  DO: 12.0814  DON'T: 12.08\n"
    "3. PRESERVE STRUCTURE - if the problem writes 2*8*x, write 2*8*x not 16x.\n"
    "4. EXPLICIT MULTIPLICATION - write 3*t*(1-t)^2, not 3t(1-t)^2.\n"
    "5. EXPONENTIALS - write \\exp(0.016*t) or e^{0.016t}, not standalone e^0.016t.\n"
    "6. FRACTIONS - use exact fractions (5/8) for rational results.\n"
    "7. ORDER - answer multi-part questions in the exact order the problem asks.\n"
    "8. NO ANGLE BRACKETS - do not wrap answers in <> brackets.\n\n"

    "Before writing the final \\boxed{}, verify your answer satisfies the original problem. "
    "Commit to your best answer."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices carefully, then select the single best answer.\n\n"
    "STEP 1 - Solve: Work through the problem step-by-step to derive your answer.\n"
    "STEP 2 - Match: Compare your result against every option:\n"
    "  a) Check algebraic/symbolic equivalence (e.g. pi*sqrt(a) = pi*a^{1/2}, "
    "4/3*ln(3) = 2/3*ln(9), 1-cos^2(x) = sin^2(x)).\n"
    "  b) If options look different, plug in a concrete numeric value for any free variable "
    "and evaluate BOTH your answer and each option - pick the one whose value matches yours.\n"
    "  c) If two options appear numerically equal, prefer the one whose algebraic form "
    "matches your derivation most directly.\n"
    "STEP 3 - Commit: Trust your derivation. Do not abandon a correct answer just because "
    "the option looks different in form.\n\n"
    "You MUST always pick one of the given letters, never say none match. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)


# 3. Utilities

def build_messages(question: str, options: Optional[list]) -> list[dict]:
    """Return a messages list in OpenAI chat format."""
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        user_content = f"{question}\n\nOptions:\n{opts_text}"
        system = SYSTEM_PROMPT_MCQ
    else:
        user_content = question
        system = SYSTEM_PROMPT_MATH
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_content},
    ]


def extract_boxed(text: str) -> str:
    """Return the content of the LAST \\boxed{} in text (handles nested braces)."""
    matches = []
    i = 0
    while i < len(text):
        idx = text.find(r"\boxed{", i)
        if idx == -1:
            break
        depth = 0
        j = idx + len(r"\boxed{")
        start = j
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                if depth == 0:
                    matches.append(text[start:j])
                    break
                depth -= 1
            j += 1
        i = idx + 1
    return matches[-1].strip() if matches else ""


def _write_csv(out_path: Path, data: list, done: dict) -> None:
    """Write id,response CSV in original dataset order."""
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "response"])
        writer.writeheader()
        for item in data:
            writer.writerow({"id": item["id"], "response": done.get(item["id"], "")})


def _login_huggingface() -> None:
    """
    Load HF_TOKEN from (in order):
      1. Colab secrets
      2. Environment variable HF_TOKEN
    Then call huggingface_hub.login().
    Raises RuntimeError if no token is found.
    """
    from huggingface_hub import login

    token: Optional[str] = None

    # Colab
    try:
        from google.colab import userdata  # type: ignore
        token = userdata.get("HF_TOKEN")
        if token:
            print("HF token loaded from Colab secrets.")
    except Exception:
        pass

    # Environment variable (also covers Kaggle secrets surfaced as env vars)
    if not token:
        token = os.environ.get("HF_TOKEN")
        if token:
            print("HF token loaded from environment variable HF_TOKEN.")

    if not token:
        raise RuntimeError(
            "HF_TOKEN not found. "
            "Add it as a Colab/Kaggle secret or set the HF_TOKEN environment variable."
        )

    login(token=token, add_to_git_credential=False)
    print("HuggingFace login successful.")



# 4. Competition entry point

def run_inference(
    data_path:   str = DATA_PATH,
    output_path: str = OUTPUT_PATH,
    model_name:  str = MODEL_NAME,
) -> str:
    """
    Full end-to-end inference pipeline.

    Parameters
    ----------
    data_path   : Path to the private test JSONL file.
    output_path : Where to write the submission CSV.
    model_name  : HuggingFace model ID or local path.

    Returns
    -------
    str : Absolute path to the written submission CSV.
    """
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # GPU check 
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU detected. "
            "On Colab: Runtime → Change runtime type → T4 GPU. "
            "On Kaggle: Settings → Accelerator → GPU."
        )
    print(f"GPU  : {torch.cuda.get_device_name(0)}")
    print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # HuggingFace login 
    _login_huggingface()

    # Load dataset 
    print(f"\nLoading dataset: {data_path}")
    data = [json.loads(line) for line in open(data_path, encoding="utf-8")]
    n_mcq  = sum(bool(d.get("options")) for d in data)
    n_free = len(data) - n_mcq
    print(f"Loaded {len(data)} questions ({n_mcq} MCQ, {n_free} free-form)")

    # Resume from partial CSV 
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: dict[int, str] = {}
    if out_path.exists():
        with open(out_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                resp = row["response"]
                if resp and not resp.startswith("ERROR:"):
                    done[int(row["id"])] = resp
        print(f"Resuming: {len(done)} done, {len(data) - len(done)} remaining")

    remaining = [d for d in data if d["id"] not in done]
    if not remaining:
        print("All questions already completed. Writing CSV.")
        _write_csv(out_path, data, done)
        return str(out_path.resolve())

    # Load model 
    print(f"\nLoading tokenizer : {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"Loading model     : {model_name}  (this takes ~2-4 min)")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.\n")

    # Inference loop 
    print(f"Running inference on {len(remaining)} questions...")
    for item in tqdm(remaining, desc="Generating"):
        messages = build_messages(item["question"], item.get("options"))
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        try:
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    top_k=TOP_K,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
            response   = tokenizer.decode(new_tokens, skip_special_tokens=True)
        except Exception as e:
            print(f"\nERROR on id={item['id']}: {e}")
            response = f"ERROR: {e}"

        done[item["id"]] = response
        # Save after every question so progress isn't lost on disconnect
        _write_csv(out_path, data, done)

    n_empty = sum(1 for d in data if not done.get(d["id"], "").strip())
    print(f"\nDone. {len(data)} questions processed, {n_empty} empty responses.")
    print(f"Submission saved to: {out_path.resolve()}")
    return str(out_path.resolve())


# 5. CLI entry point

if __name__ == "__main__":
    _ensure_dependencies()
    run_inference()
