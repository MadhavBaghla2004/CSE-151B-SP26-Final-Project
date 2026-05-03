"""
CSE 151B Competition — vLLM Inference Script
Run from terminal: python inference.py
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import random

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID    = "/content/drive/MyDrive/cse151b-sp26/Qwen/qwen3-finetuned"
GPU_ID      = "0"                       # Colab always uses device 0
DATA_PATH   = "/content/drive/MyDrive/cse151b-sp26/data/private.jsonl"
OUTPUT_PATH = "/content/drive/MyDrive/cse151b-sp26/results/private_results.jsonl"
MAX_TOKENS  = 32768
SAVE_EVAL   = False                      # Set to False for private test set

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

# ── Imports (after env vars are set) ──────────────────────────────────────────
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# ── 1. Load Dataset ───────────────────────────────────────────────────────────
print("Loading dataset...")
data = [json.loads(line) for line in open(DATA_PATH)]

n_mcq  = sum(bool(d.get("options")) for d in data)
n_free = sum(not d.get("options")   for d in data)
print(f"Loaded {len(data)} questions  ({n_mcq} MCQ, {n_free} free-form)")


# ── 2. Prompt Construction ────────────────────────────────────────────────────
SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step and explain your reasoning concisely. This means breaking the problem down into solvable parts and working through them to build the final answer. "
    """Use a clear structure with short labeled steps (Step 1, Step 2, …). Follow the framework of understanding the problem ("we know that..."), devising 
    a plan ("we need to find...", "in order to find..., use equation..."), executing the plan, and verifying the solution to ensure no mistakes. """
    "Avoid unnecessary commentary; focus only on the logic and theory needed to solve the problem thoroughly. "
    "Longer response doesn't always mean better reasoning, so prioritize thoroughness and rely on previous steps in your explanation when calculating the current step.\n"
    
    "Below are 4 examples of reasoning through the problem:"
    r"""
    Example Question 1: Compute $\tan^{-1}(\tan \frac{5\pi}{8})=$ [ANS],
    Explanation: $ \tan \frac{5\pi}{8} = -2.41421356237$. Then, $\tan^{-1}(-2.41421356237) = -1.1781$. 
    Thus, after calculating inner operation then outer operation, the answer is $-1.1781$.
    """
    r"""
    Example Question 2: Suppose we want a 95\% confidence interval for the average amount spent on books by freshmen in their first year at college. The amount spent has a normal distribution with standard deviation \$30.
    (a) How large should the sample be if the margin of error is to be less than \$4? ANSWER: [ANS]
    (b) If we wanted a smaller margin of error, we would need a [ANS] sample. (Enter: ''LARGER'', ''SMALLER'' or ''SAME SIZE'', without the quotes.),
    Explanation: Equation for margin of error is $\text{E} = z*\frac{\sigma}{\sqrt{n}}$. Isolate $n$ such that $n = (\frac{z*\sigma}{E})^2$. 
    Plug in inputs where $\sigma = 30$, $E = 4$, $z = 1.96$ corresponding to 95\% confidence interval. 
    The answer is $\text{Ceiling}(n) = \text{Ceiling}((\frac{1.96*30}{4})^2) = 217$. For the second part, because $n$ is in the denominator, increasing $n$ will decrease the standard error and thus the margin of error. 
    Therefore, the answer is LARGER.
    """
    r"""
    Example Question 3: The smallest sum one could get by adding three different numbers from the set {-1, 20, 24, 13,-4, 17} is [ANS]., 
    Explanation: Smallest sum is sum of smallest values. The smallest values are {-1, -4, 13}. Thus, $\text{smallest sum} = 8$.
    """
    r"""
    Example Question 4: Question: The cost of a parking ticket at NAU is \$40 for the first offense, but the cost triples for each additional offense. 
    Write a formula for the cost $C$ as a function of the number of tickets $n$. Remember to use the variable " $n$ " in your answer. $\ C$=[ANS]., 
    Explanation: $40$ is cost of parking ticket. $3^{(n-1)}$ is multiplicative rate for each additional offense. Thus, the answer is $40*3^{(n-1)}$.
    """

    "\nPut your final answer inside \\boxed{}. "
    """
    For decimal answers, represent decimal answers in IEEE-754 double-precision floating-point approximation to ensure greater accuracy (e.g. \\boxed{62.7777777777778}, \\boxed{143.224229233795}). 
    For expression answers, leave answers in exact form rather than decimal approximation (e.g. \\boxed{T * W + S*(T+W), T+W}, \\boxed{pi}, \\boxed{325*(1+325)}) if possible. 
    Don't forget to consolidate multiple sub-answers into a single \\boxed{} at the very end if needed (e.g. \\boxed{'-10', '100', '-9', '81'}). 
    """
    "For expression answers, represent each operation with standard mathematical notation (e.g. \\boxed{'3*t^1*(1-t)^2'}), avoid using latex. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, e.g. \\boxed{3, 7}."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. Solve the problem step-by-step and explain your reasoning concisely. This means breaking the problem down into solvable parts and working through them to build the final answer. "
    """Use a clear structure with short labeled steps (Step 1, Step 2, …). Follow the framework of understanding the problem ("we know that..."), devising 
    a plan ("we need to find...", "in order to find..., use equation..."), executing the plan, and verifying the solution to ensure no mistakes. """
    "Avoid unnecessary commentary; focus only on the logic and theory needed to solve the problem thoroughly. "
    "Longer response doesn't always mean better reasoning, so prioritize thoroughness and rely on previous steps in your explanation when calculating the current step.\n"
    
    "Below are 3 examples of reasoning through the problem:"
    r"""
    Example Question 1: In a certain factory, one team consists of 9 male workers and 5 female workers. If 3 representatives are to be selected, what is 
    the probability that at least one of the selected representatives is a female worker?, 
    Choices: ['13/16', '8/13', '9/13', '11/14', '12/15', '7/12', '9/14', '11/13', '10/13', '7/10'], 
    Explanation: 3 representatives are selected. First, calculate the probability of all male workers: 
    $\frac{9}{14} * \frac{8}{13} * \frac{7}{12} = \frac{3}{13}$. Then, take the complement to calculate at least one selected female worker: 
    $1 - \frac{3}{13}$. Thus, the answer is $\frac{10}{13}$, which corresponds to (I).
    """
    r"""
    Example Question 2: If a random variable X is normally distributed with a mean of 118 and a standard deviation of 11, what Z-scores correspond to raw 
    scores of 115, 134, and 99?, 
    Choices: ['0, 1.5, -2', '-.27, 1.45, -1.73', '-.27, -1.45, -1.73', '2.73, -0.27, 0.45', '.27, -1.45, .73', '-1, 2, -2', '.27, 1.45, 1.73', '-2.18, 0.15, -1.82', '-.27, 1.45, 1.73', '-0.5, 1.2, -1.5'], 
    Explanation: First, the equation for z-score is $z = \frac{x - \mu}{\sigma}$. Then, plug the raw scores in the formula where $\mu = 118$, 
    $\sigma = 11$: $z = \frac{115 - 118}{11} = -.27$, $z = \frac{134 - 118}{11} = 1.45$, $z = \frac{99 - 118}{11} = -1.73$. Thus, the answer is (B) = $-.27, 1.45, -1.73$.
    """
    r"""
    Example Question 3: Find $L=\lim_{(x,y) \to (2,3)}\left(\frac{ x^2-y^2+10 \cdot y-25 }{ x^2-y^2-10 \cdot x+25 }\right)$., 
    Choices: ['-4/3', '-3/2', '-2/3', '-3/4', '-3/5', '-1/3', '-1/2', '-1/4'], 
    Explanation: Step 1: Substitute $(x, y) = (2,3)$ into the expression to see it gives $\frac{0}{0}$, so the fraction needs to be simplified algebraically. 
    Step 2: Factor the numerator and denominator as differences of squares, cancel the common factor $(x+y-5)$, and reduce the expression to $\frac{x-y+5}{x-y+5}$.
    Step 3: Substitute $(x, y) = (2,3)$ into the simplified expression to get $-\frac{4}{6} = -\frac{2}{3}$. Thus, the correct answer is (C) = $-\frac{2}{3}$.
    """
    
    "\nRead the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)


def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a question."""
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, question


# ── 3. Load Model ─────────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

print("Loading model (this may take a few minutes)...")
llm = LLM(
    model=MODEL_ID,
    quantization="bitsandbytes",
    load_format="bitsandbytes",
    enable_prefix_caching=False,
    gpu_memory_utilization=0.85,
    max_model_len=16384,
    trust_remote_code=True,
    max_num_seqs=256,
    max_num_batched_tokens=32768,
)

sampling_params = SamplingParams(
    max_tokens=MAX_TOKENS,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    presence_penalty=0.0,
    repetition_penalty=1.0,
)

print("Model loaded.")


# ── 4. Build Prompts ──────────────────────────────────────────────────────────
print("Building prompts...")
prompts = []

for item in data:
    system, user = build_prompt(item["question"], item.get("options"))
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user",   "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompts.append(prompt_text)


# ── 6. Generate Responses ─────────────────────────────────────────────────────
print(f"Generating responses for {len(prompts)} questions...")
outputs = llm.generate(prompts, sampling_params=sampling_params)
responses = [out.outputs[0].text.strip() for out in outputs]
print("Generation complete.")


# # ── 7. Score Responses ────────────────────────────────────────────────────────
# def extract_letter(text: str) -> str:
#     m = re.search(r"\\boxed\{([A-Za-z])\}", text)
#     if m:
#         return m.group(1).upper()
#     matches = re.findall(r"\b([A-Z])\b", text.upper())
#     return matches[-1] if matches else ""


# def score_mcq(response: str, gold_letter: str) -> bool:
#     return extract_letter(response) == gold_letter.strip().upper()


# print("Scoring responses...")
# sys.path.insert(0, ".")
# from judger import Judger
# judger = Judger(strict_extract=False)

results = []
for item, response in tqdm(zip(data, responses), total=len(data), desc="Scoring"):
    is_mcq = bool(item.get("options"))
    # gold   = item["answer"]

    # if is_mcq:
    #     correct = score_mcq(response, str(gold))
    # else:
    #     gold_list = gold if isinstance(gold, list) else [gold]
    #     try:
    #         correct = judger.auto_judge(
    #             pred=response,
    #             gold=gold_list,
    #             options=[[]] * len(gold_list),
    #         )
    #     except Exception:
    #         correct = False

    results.append({
        "id":       item.get("id"),
        "is_mcq":   is_mcq,
        # "gold":     gold,
        "response": response,
        # "correct":  correct,
    })


# # ── 8. Print Summary ──────────────────────────────────────────────────────────
# mcq_res  = [r for r in results if r["is_mcq"]]
# free_res = [r for r in results if not r["is_mcq"]]

# def acc(subset):
#     return sum(r["correct"] for r in subset) / len(subset) * 100 if subset else 0.0

# print("=" * 50)
# print("EVALUATION RESULTS")
# print("=" * 50)
# print(f"  MCQ        : {sum(r['correct'] for r in mcq_res):4d} / {len(mcq_res):4d}  ({acc(mcq_res):.2f}%)")
# print(f"  Free-form  : {sum(r['correct'] for r in free_res):4d} / {len(free_res):4d}  ({acc(free_res):.2f}%)")
# print(f"  Overall    : {sum(r['correct'] for r in results):4d} / {len(results):4d}  ({acc(results):.2f}%)")
# print("=" * 50)


# ── 8. Save Results ───────────────────────────────────────────────────────────
out_path = Path(OUTPUT_PATH)
out_path.parent.mkdir(parents=True, exist_ok=True)

with open(out_path, "w") as f:
    for r in results:
        if SAVE_EVAL:
            record = {"id": r["id"], "is_mcq": r["is_mcq"], "gold": r["gold"],
                      "response": r["response"], "correct": r["correct"]}
        else:
            record = {"id": r["id"], "is_mcq": r["is_mcq"], "response": r["response"]}
        f.write(json.dumps(record) + "\n")

print(f"Saved {len(results)} records to {out_path}")
