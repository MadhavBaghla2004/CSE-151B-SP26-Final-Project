# ── 1. Imports ────────────────────────────────────────────────────────────────
import torch
import json
from typing import Optional

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling

# ── 2. Config ─────────────────────────────────────────────────────────────────
MODEL_ID       = "Qwen/Qwen3-4B-Thinking-2507"
DATA_PATH      = "/content/drive/MyDrive/cse151b-sp26/data/public.jsonl"
OUTPUT_DIR     = "/content/drive/MyDrive/cse151b-sp26/Qwen/qwen3-finetuned"
RANK           = 16          # LoRA rank
LORA_ALPHA     = 16          # LoRA scaling
LEARNING_RATE  = 2e-4
EPOCHS         = 4           # any more and the eval loss starts to increase with current model architecture
MAX_SEQ_LEN    = 2048

# ── 3. Load Dataset ───────────────────────────────────────────────────────────
print("Loading dataset...")
data = [json.loads(line) for line in open(DATA_PATH)]

n_mcq  = sum(bool(d.get("options")) for d in data)
n_free = sum(not d.get("options")   for d in data)
print(f"Loaded {len(data)} questions  ({n_mcq} MCQ, {n_free} free-form)")

# ── 4. Prompt Construction ────────────────────────────────────────────────────
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

# ── 5. Build Prompts ──────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

print("Building prompts...")
prompts = []

for item in data:
    system, user = build_prompt(item["question"], item.get("options"))
    answer = str(item["answer"])
    
    prompt_text = tokenizer.apply_chat_template( # system = prompt, user = question + options, answer = answer to question
        [{"role": "system", "content": system},
         {"role": "user",   "content": user},
         {"role": "assistant", "content": f"\\boxed{{{answer}}}"}], 
        tokenize=False,
        add_generation_prompt=False, # false for training, true for inference. but why?
    )
    prompts.append({"text": prompt_text})
    
dataset = Dataset.from_list(prompts)

def tokenize(example): # tokenize the prompt
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
    )
    tokens["labels"] = [-100 if token == tokenizer.pad_token_id else token for token in tokens["input_ids"]]  # model predicts its own input
    return tokens

# what does the tokenized_dataset look like?
tokenized_dataset = dataset.map(tokenize, remove_columns=["text"]) # removes raw text, only keeps tokenized text
print(f"Tokenized dataset ready: {len(tokenized_dataset)} examples")

# ── 6. Load Base Model ──────────────────────────────────────────────────────────
config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=config,
    device_map="auto",
    trust_remote_code=True,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters() # self check

# ── 7. Training Model ──────────────────────────────────────────────────────────
split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
eval_dataset  = split["test"]

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir = OUTPUT_DIR,
    num_train_epochs = EPOCHS, # how many passes through the dataset
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 8,
    learning_rate=LEARNING_RATE,
    optim="adamw_torch",
    weight_decay=0.01, # increase regularization as needed (try 0.1)
    gradient_checkpointing=True, # cautious decision to reduce memory usage
    bf16=True,
    logging_strategy="epoch", # prints training loss every epoch (or every # gradient steps)
    eval_strategy="epoch", # evaluate on unseen part of dataset to track overfitting
    eval_accumulation_steps=1,
    per_device_eval_batch_size=1,
    save_strategy="epoch",
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset, # 80% for training
    processing_class = tokenizer,
    data_collator = data_collator,
    eval_dataset = eval_dataset, # 20% for evaluation
)

trainer.train()

# ── 8. Save Model ──────────────────────────────────────────────────────────
model = model.merge_and_unload()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR) # rules and methods needed to tokenize text
print(f"Model saved to {OUTPUT_DIR}")


