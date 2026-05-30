import os
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"

# ── 1. Imports ────────────────────────────────────────────────────────────────
import copy
import ast
import json
import re
from unsloth import FastLanguageModel
from typing import Optional

import numpy as np
import torch
from datasets import Dataset
from transformers import TrainerCallback
from vllm import SamplingParams
from trl import GRPOConfig, GRPOTrainer

from judger import Judger
from utils import last_boxed_only_string, remove_boxed

# ── 2. Config ─────────────────────────────────────────────────────────────────
MODEL_ID = "unsloth/Qwen3-4B-Thinking-2507"
DATA_PATH = "/content/drive/MyDrive/cse151b-sp26/data/public.jsonl"
OUTPUT_DIR = "/content/drive/MyDrive/cse151b-sp26/Qwen_rl_v4/qwen3-grpo"
RANK = 16
LORA_ALPHA = 32
LEARNING_RATE = 5e-6
MAX_SEQ_LEN = 2048
NUM_GENERATIONS = 6
MAX_STEPS = 700
SEED = 3407

# Prompt / completion budget (leave MAX_SEQ_LEN unchanged)
PROMPT_LENGTH_QUANTILE = 0.75
MIN_COMPLETION_TOKENS = 768
CLIP_TOKEN_MARGIN = 4

# Train / serve decoding (matches inference.py)
TRAIN_TEMPERATURE = 0.6
TRAIN_TOP_P = 0.95
TRAIN_TOP_K = 20
TRAIN_MIN_P = 0.0

# Held-out judger eval during training
EVAL_FRACTION = 0.10
EVAL_MAX_SAMPLES = 80
EVAL_EVERY_N_STEPS = 50

# GRPO stability
BETA = 0.04
REWARD_WEIGHTS = [0.25, 1.0, 0.5, 0.0]  # format, accuracy, clipped, debug

REASONING_START = "<reasoning>"
ANSWER_STOP = "</answer>"
CLOSING_RE = re.compile(
    r"</reasoning>\s*<answer>.*?</answer>\s*$",
    re.DOTALL,
)
ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
MCQ_LABEL_RE = re.compile(r"\b([A-Z])\b")

_JUDGER = Judger(strict_extract=False)
_MAX_COMPLETION_LEN: int = 512  # set after dataset prep
_TRAINER_REF: list = []  # filled after GRPOTrainer construction


# ── 3. Scoring helpers (aligned with inference.py + judger.py) ────────────────
def _normalize_str_list(items: list) -> list[str]:
    out = []
    for item in items:
        s = str(item).strip()
        if s:
            out.append(s)
    return out


def _parse_answer_value(answer) -> list[str]:
    """
    Robustly parse answers that may be:
    - scalar strings ("A", "42")
    - lists (["A", "C"])
    - serialized strings ("['A', 'C']", '"A"', '["A","C"]')
    """
    if isinstance(answer, list):
        return _normalize_str_list(answer)
    if answer is None:
        return []

    s = str(answer).strip()
    if not s:
        return []

    # Try JSON first (supports answers stored via json.dumps).
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return _normalize_str_list(parsed)
        return _normalize_str_list([parsed])
    except Exception:
        pass

    # Fall back to Python-literal style strings.
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return _normalize_str_list(parsed)
        return _normalize_str_list([parsed])
    except Exception:
        return [s]


def _count_ans_slots(question: str) -> int:
    return max(1, len(re.findall(r"\[ANS\]", question, flags=re.IGNORECASE)))


def _labels_from_text(text: str) -> list[str]:
    """
    Extract MCQ labels from free text. Handles:
    - "A"
    - "A, C"
    - "A and C"
    - "AC"
    """
    if not text:
        return []

    cleaned = text.upper()
    labels = MCQ_LABEL_RE.findall(cleaned)
    if not labels:
        tight = re.sub(r"[^A-Z]", "", cleaned)
        if 1 <= len(tight) <= 10:
            labels = list(tight)

    # Preserve order, remove duplicates, keep A-Z only.
    seen = set()
    out = []
    for lbl in labels:
        if "A" <= lbl <= "Z" and lbl not in seen:
            seen.add(lbl)
            out.append(lbl)
    return out


def extract_mcq_labels(text: str) -> list[str]:
    # Highest precision: labels inside boxed answer.
    boxed = remove_boxed(last_boxed_only_string(text))
    if boxed:
        labels = _labels_from_text(boxed)
        if labels:
            return labels

    # Next: explicit <answer>...</answer> block.
    m = ANSWER_BLOCK_RE.search(text)
    if m:
        labels = _labels_from_text(m.group(1))
        if labels:
            return labels

    # Last resort: full text.
    return _labels_from_text(text)


def extract_gold_mcq_labels(answer) -> list[str]:
    items = _parse_answer_value(answer)
    labels = []
    for item in items:
        labels.extend(_labels_from_text(item))
    # De-duplicate while preserving order.
    seen = set()
    out = []
    for lbl in labels:
        if lbl not in seen:
            seen.add(lbl)
            out.append(lbl)
    return out


def full_completion_text(assistant_content: str) -> str:
    """Rebuild text as seen at inference (forced <reasoning> prefix + completion)."""
    return REASONING_START + assistant_content


def gold_as_list(answer) -> list[str]:
    return _parse_answer_value(answer)


def judger_is_correct(full_text: str, gold, is_mcq: bool) -> bool:
    if is_mcq:
        pred_labels = extract_mcq_labels(full_text)
        gold_labels = extract_gold_mcq_labels(gold)
        return bool(gold_labels) and set(pred_labels) == set(gold_labels)
    try:
        return _JUDGER.auto_judge(
            pred=full_text,
            gold=_parse_answer_value(gold),
            options=[[]] * len(_parse_answer_value(gold)),
        )
    except Exception:
        return False


def judger_is_correct_with_slots(full_text: str, gold, is_mcq: bool, num_ans_slots: int = 1) -> bool:
    """
    Slot-aware wrapper around Judger logic.
    - MCQ: exact set match of labels.
    - Free-form with 1 slot + multiple gold entries: treat entries as alternative valid forms.
    - Free-form with >1 slots: treat gold list as ordered multi-part target.
    """
    if is_mcq:
        return judger_is_correct(full_text, gold, is_mcq=True)

    gold_list = _parse_answer_value(gold)
    if not gold_list:
        return False

    # Single answer slot with multiple entries is often "any of these forms".
    if num_ans_slots <= 1 and len(gold_list) > 1:
        # Accept if any canonical gold form matches full prediction.
        for alt in gold_list:
            try:
                if _JUDGER.auto_judge(
                    pred=full_text,
                    gold=[alt],
                    options=[[]],
                ):
                    return True
            except Exception:
                pass

        # Also accept if prediction outputs multiple comma-separated forms
        # and any one form matches a valid alternative.
        boxed = remove_boxed(last_boxed_only_string(full_text))
        if boxed:
            pred_parts = [p.strip() for p in boxed.split(",") if p.strip()]
            for pred_part in pred_parts:
                pred_wrapped = f"<answer>\\boxed{{{pred_part}}}</answer>"
                for alt in gold_list:
                    try:
                        if _JUDGER.auto_judge(
                            pred=pred_wrapped,
                            gold=[alt],
                            options=[[]],
                        ):
                            return True
                    except Exception:
                        continue
        return False

    # Multi-slot question or single canonical answer.
    try:
        return _JUDGER.auto_judge(
            pred=full_text,
            gold=gold_list,
            options=[[]] * len(gold_list),
        )
    except Exception:
        return False


def has_extractable_answer(full_text: str) -> bool:
    if last_boxed_only_string(full_text) is None:
        return False
    inner = remove_boxed(last_boxed_only_string(full_text))
    return bool(inner and inner.strip())


def is_clipped(completion_ids, margin: int = CLIP_TOKEN_MARGIN) -> bool:
    if completion_ids is None:
        return False
    return len(completion_ids) >= _MAX_COMPLETION_LEN - margin


# ── 4. Load Model & Tokenizer ─────────────────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=RANK,
    gpu_memory_utilization=0.55,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=RANK,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
)
model.print_trainable_parameters()

# ── 5. Chat Template ──────────────────────────────────────────────────────────
_chat_template = (
    "{% if messages[0]['role'] == 'system' %}"
        "{{ messages[0]['content'] + eos_token }}"
        "{% set loop_messages = messages[1:] %}"
    "{% else %}"
        "{% set loop_messages = messages %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
        "{% if message['role'] == 'user' %}"
            "{{ message['content'] }}"
        "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] + eos_token }}"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}" + f"{{{{ '{REASONING_START}' }}}}" + "{% endif %}"
)
tokenizer.chat_template = _chat_template

# ── 6. System Prompts ─────────────────────────────────────────────────────────
SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step and explain your reasoning concisely. "
    "Use short labeled steps (Step 1, Step 2, …). "
    "Your response continues inside <reasoning>. Complete it in this format:\n"
    "  ...reasoning...\n"
    "  </reasoning>\n"
    "  <answer>\n"
    "  \\boxed{final answer}\n"
    "  </answer>\n"
    "Be decisive and concise in your reasoning. The less tokens the better while keeping the logic and theory needed to solve the problem thoroughly"
    "Put the final answer in \\boxed{}. Use exact form when possible; for decimals use full double precision. "
    "Multiple sub-answers go in one \\boxed{} separated by commas, e.g. \\boxed{3, 7}."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. Solve the problem step-by-step with short labeled steps. "
    "Your response continues inside <reasoning>. Complete it in this format:\n"
    "  ...reasoning...\n"
    "  </reasoning>\n"
    "  <answer>\n"
    "  \\boxed{LETTER}\n"
    "  </answer>\n"
    "Be decisive and concise in your reasoning. The less tokens the better while keeping the logic and theory needed to solve the problem thoroughly"
    "Read the options and output all correct letter(s) inside one \\boxed{}."
    "If there is one answer, use \\boxed{C}. If there are multiple answers, use comma-separated letters, e.g. \\boxed{A, C}."
)


def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, question


# ── 7. Load, split, and filter dataset ────────────────────────────────────────
print("Loading dataset...")
with open(DATA_PATH, encoding="utf-8") as f:
    raw_data = [json.loads(line) for line in f]

n_mcq = sum(bool(d.get("options")) for d in raw_data)
n_free = len(raw_data) - n_mcq
print(f"Loaded {len(raw_data)} questions  ({n_mcq} MCQ, {n_free} free-form)")

records = []
for item in raw_data:
    system, user = build_prompt(item["question"], item.get("options"))
    records.append({
        "prompt": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "answer": json.dumps(item["answer"], ensure_ascii=False),
        "is_mcq": bool(item.get("options")),
        "num_ans_slots": _count_ans_slots(item["question"]),
    })

rng = np.random.RandomState(SEED)
perm = rng.permutation(len(records))
n_eval = min(EVAL_MAX_SAMPLES, max(1, int(len(records) * EVAL_FRACTION)))
eval_idx = set(perm[:n_eval].tolist())
train_records = [records[i] for i in perm[n_eval:]]
eval_records = [records[i] for i in perm[:n_eval]]
print(f"Train: {len(train_records)}  |  Held-out eval: {len(eval_records)}")

dataset = Dataset.from_list(train_records)
eval_dataset = Dataset.from_list(eval_records)

tokenized_lens = dataset.map(
    lambda x: {
        "L": len(
            tokenizer.apply_chat_template(
                x["prompt"], add_generation_prompt=True, tokenize=True
            )
        )
    },
    desc="Measuring prompt lengths",
)

forced_prefix_tokens = len(tokenizer.encode(REASONING_START, add_special_tokens=False))
p_quantile = int(np.quantile(tokenized_lens["L"], PROMPT_LENGTH_QUANTILE))
max_prompt_by_budget = MAX_SEQ_LEN - MIN_COMPLETION_TOKENS - forced_prefix_tokens
max_prompt_length = min(p_quantile + 1, max_prompt_by_budget)

lengths = np.array(tokenized_lens["L"])
dataset = dataset.select(np.where(lengths <= max_prompt_length)[0])
print(f"Prompt length cap: {max_prompt_length} tokens (p{int(PROMPT_LENGTH_QUANTILE * 100)} was {p_quantile})")
print(f"Dataset after length filter: {len(dataset)} train examples")

max_completion_length = MAX_SEQ_LEN - max_prompt_length - forced_prefix_tokens
_MAX_COMPLETION_LEN = max_completion_length
print(f"Max completion tokens: {max_completion_length}")

# ── 8. Reward functions ─────────────────────────────────────────────────────────
def reward_format(completions, **kwargs) -> list[float]:
    """
    Continuous format signal (varies across generations in a group).
    +1.0  valid closing structure ending with </answer> and extractable \\boxed{}
    +0.3  boxed present but structure incomplete
    -0.5  duplicate closing tags
    -1.0  no extractable boxed answer
    """
    scores = []
    for comp in completions:
        text = full_completion_text(comp[0]["content"])
        score = -1.0
        if has_extractable_answer(text):
            if CLOSING_RE.search(text):
                score = 1.0
            else:
                score = 0.3
        dup_penalty = 0.0
        for tag in ("</reasoning>", "<answer>", "</answer>"):
            c = text.count(tag)
            if c > 1:
                dup_penalty -= 0.25 * (c - 1)
        scores.append(max(-1.0, score + dup_penalty))
    return scores


def reward_accuracy(prompts, completions, answer, is_mcq, num_ans_slots, **kwargs) -> list[float]:
    """
    Primary task reward via Judger (free-form) or letter match (MCQ).
    +6.0  correct
    -1.0  wrong but extractable answer
    -4.0  no extractable answer
    """
    scores = []
    for comp, true_ans, mcq, n_slots in zip(completions, answer, is_mcq, num_ans_slots):
        text = full_completion_text(comp[0]["content"])
        if not has_extractable_answer(text):
            scores.append(-4.0)
            continue

        if not mcq:
            scores.append(6.0 if judger_is_correct_with_slots(text, true_ans, mcq, n_slots) else -1.0)
            continue

        pred = set(extract_mcq_labels(text))
        gold = set(extract_gold_mcq_labels(true_ans))
        if not pred or not gold:
            scores.append(-4.0)
        elif pred == gold:
            scores.append(6.0)
        else:
            overlap = len(pred & gold)
            if overlap > 0:
                # Positive partial credit when one of multiple labels is correct.
                scores.append(1.0 + 2.0 * overlap / len(gold))
            else:
                scores.append(-1.0)
    return scores


def reward_clipped(completions, completion_ids, **kwargs) -> list[float]:
    """Penalize generations that hit the completion token budget."""
    scores = []
    for comp, ids in zip(completions, completion_ids):
        text = comp[0]["content"]
        clipped = is_clipped(ids)
        # Also penalize missing </answer> when not stopped cleanly
        if not clipped and ANSWER_STOP not in text:
            clipped = len(ids) >= _MAX_COMPLETION_LEN - CLIP_TOKEN_MARGIN * 2
        scores.append(-2.5 if clipped else 0.0)
    return scores


_print_counter = 0


def reward_debug(prompts, completions, answer, is_mcq, completion_ids, num_ans_slots, **kwargs) -> list[float]:
    global _print_counter
    if _print_counter % 5 == 0:
        text = full_completion_text(completions[0][0]["content"])
        ok = judger_is_correct_with_slots(text, answer[0], is_mcq[0], num_ans_slots[0])
        clipped = is_clipped(completion_ids[0]) if completion_ids else False
        q = prompts[0][-1]["content"][:120]
        r = completions[0][0]["content"][:300]
        print(
            f"\n{'=' * 60}"
            f"\nQ: {q}"
            f"\nTrue: {answer[0]}"
            f"\nJudger correct: {ok}  clipped: {clipped}"
            f"\nResp (after forced <reasoning>): {r}"
            f"\n{'=' * 60}"
        )
    _print_counter += 1
    return [0.0] * len(completions)


# ── 9. Held-out judger eval callback ──────────────────────────────────────────
class JudgerEvalCallback(TrainerCallback):
    def __init__(self, eval_ds: Dataset, every_n_steps: int, max_samples: int):
        self.eval_ds = eval_ds
        self.every_n_steps = every_n_steps
        self.max_samples = min(max_samples, len(eval_ds))
        self._eval_sampling_params = None

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.every_n_steps != 0:
            return
        if not _TRAINER_REF:
            return
        trainer = _TRAINER_REF[0]
        samples = self.eval_ds.select(range(self.max_samples))
        prompts = [ex["prompt"] for ex in samples]

        old_sp = trainer.args.vllm_sampling_params
        try:
            trainer.args.vllm_sampling_params = self._eval_sampling_params
            prev_mode = trainer.model.training
            trainer.model.eval()
            with torch.inference_mode():
                completions = trainer._generate(copy.deepcopy(prompts))
            trainer.model.train(prev_mode)
        except Exception as exc:
            print(f"[eval] Skipped judger eval at step {state.global_step}: {exc}")
            return
        finally:
            trainer.args.vllm_sampling_params = old_sp

        correct = 0
        for ex, comp in zip(samples, completions):
            content = comp[0]["content"] if isinstance(comp, list) else comp
            text = full_completion_text(content)
            if judger_is_correct_with_slots(text, ex["answer"], ex["is_mcq"], ex["num_ans_slots"]):
                correct += 1
        acc = correct / len(samples)
        print(f"[eval] step {state.global_step}: judger accuracy = {acc:.3f} ({correct}/{len(samples)})")
        trainer.log({"eval/judger_accuracy": acc})

    def bind_eval_sampling(self, sampling_params: SamplingParams):
        self._eval_sampling_params = sampling_params


# ── 10. Sampling & GRPO config ─────────────────────────────────────────────────
_stop_tokens = [ANSWER_STOP, tokenizer.eos_token]

vllm_sampling_params = SamplingParams(
    temperature=TRAIN_TEMPERATURE,
    top_p=TRAIN_TOP_P,
    top_k=TRAIN_TOP_K,
    min_p=TRAIN_MIN_P,
    max_tokens=max_completion_length,
    seed=SEED,
    stop=_stop_tokens,
    include_stop_str_in_output=True,
)

eval_sampling_params = SamplingParams(
    temperature=TRAIN_TEMPERATURE,
    top_p=TRAIN_TOP_P,
    top_k=TRAIN_TOP_K,
    min_p=TRAIN_MIN_P,
    max_tokens=max_completion_length,
    seed=SEED,
    stop=_stop_tokens,
    include_stop_str_in_output=True,
)

training_args = GRPOConfig(
    vllm_sampling_params=vllm_sampling_params,
    temperature=TRAIN_TEMPERATURE,
    top_p=TRAIN_TOP_P,
    top_k=TRAIN_TOP_K,
    num_generations=NUM_GENERATIONS,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    beta=BETA,
    reward_weights=REWARD_WEIGHTS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.001,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    bf16=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=MAX_STEPS,
    logging_steps=1,
    save_steps=100,
    output_dir=OUTPUT_DIR,
    report_to="none",
)

eval_callback = JudgerEvalCallback(
    eval_dataset,
    every_n_steps=EVAL_EVERY_N_STEPS,
    max_samples=EVAL_MAX_SAMPLES,
)
eval_callback.bind_eval_sampling(eval_sampling_params)

# ── 11. Train ─────────────────────────────────────────────────────────────────
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        reward_format,
        reward_accuracy,
        reward_clipped,
        reward_debug,
    ],
    args=training_args,
    train_dataset=dataset,
    callbacks=[eval_callback],
)

_TRAINER_REF.append(trainer)

trainer.train()

# ── 12. Save ──────────────────────────────────────────────────────────────────
model.save_lora(OUTPUT_DIR + "/lora_adapter")
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved to {OUTPUT_DIR}")