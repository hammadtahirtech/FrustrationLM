from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments
)
from datasets import Dataset

# -----------------------
# Tokenizer
# -----------------------
tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
# Use default GPT-2 EOS and set PAD to EOS for causal LM training
tokenizer.pad_token = tokenizer.eos_token

# -----------------------
# Build regular prompt→completion dataset (no roles)
# -----------------------
def build_example(prompt, completion):
    # Compose sample with role prefixes
    prefix = f"User: {prompt}\nAssistant: "
    text = f"{prefix}{completion}{tokenizer.eos_token}"

    # Tokenize once, with character offsets, so we can find exactly where the
    # completion starts in token-space without re-tokenizing the prefix
    # separately. Re-tokenizing the prefix alone (the old approach) can
    # misalign by a token, because BPE merges characters differently
    # depending on what follows — silently eating the first completion token.
    tokens = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=256,
        return_offsets_mapping=True,
    )
    input_ids = tokens["input_ids"]
    offsets = tokens["offset_mapping"]

    completion_char_start = len(prefix)

    # Find the first token whose span extends INTO the completion region.
    # We check char_end (not char_start) because GPT-2's byte-level BPE often
    # merges the prefix's trailing space with the completion's first letter
    # into a single token (e.g. " I" as one token). That token's char_start
    # falls just before completion_char_start (still "in" the prefix), but
    # its char_end falls after it — so checking char_end catches it correctly.
    # Padding tokens have offset (0, 0), whose char_end (0) never exceeds
    # completion_char_start (always > 0), so they're naturally skipped.
    start = len(input_ids)
    for i, (char_start, char_end) in enumerate(offsets):
        if char_end > completion_char_start:
            start = i
            break

    # Mask the prompt part; only learn on the completion and EOS
    labels = [-100] * len(input_ids)
    labels[start:] = input_ids[start:]

    # Also mask out padding positions. Without this, since pad_token == eos_token,
    # the ~200+ padding tokens after the real completion+EOS get treated as valid
    # training targets too — teaching the model to predict EOS repeatedly and
    # dominating the loss, which causes it to emit EOS immediately at inference
    # (empty output) instead of learning the actual completion.
    attention_mask = tokens["attention_mask"]
    labels = [
        label if attention_mask[i] == 1 else -100
        for i, label in enumerate(labels)
    ]

    return {
        "input_ids": input_ids,
        "attention_mask": tokens["attention_mask"],
        "labels": labels,
    }

def load_examples_from_file(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split on blank lines to get blocks, no reliance on any role markers
    blocks = [b.strip() for b in content.split("\n\n")]
    examples = []

    for block in blocks:
        if not block:
            continue

        lines = [line.strip() for line in block.splitlines() if line.strip()]

        # Expect exactly two lines per example: first = prompt, second = completion
        if len(lines) < 2:
            continue

        prompt = lines[0]
        completion = lines[1]

        examples.append(build_example(prompt, completion))

    return examples


examples = load_examples_from_file("train.txt")
dataset = Dataset.from_list(examples)

# -----------------------
# Model
# -----------------------
# Load pretrained DistilGPT2 weights instead of a random-init custom config.
# This inherits DistilGPT2's actual architecture (6 layers, 768 dim, ~82M params) —
# you can no longer set n_layer/n_embd/n_head yourself, since those are fixed by
# whatever checkpoint you load.
model = GPT2LMHeadModel.from_pretrained("distilgpt2")

# Resize embeddings in case tokenizer vocab differs from checkpoint defaults
# (harmless no-op if they already match, but safe to keep).
model.resize_token_embeddings(len(tokenizer))

# -----------------------
# Training
# -----------------------
args = TrainingArguments(
    output_dir="tinyLLM",
    per_device_train_batch_size=4,
    num_train_epochs=6,
    learning_rate=5e-5,  # lowered from 5e-4 — fine-tuning pretrained weights
                         # needs a gentler LR than random-init training does,
                         # so you shape the existing grammar knowledge instead
                         # of overwriting it too aggressively.
    logging_steps=10,
    save_steps=500,
    save_total_limit=1,
    fp16=False,
    report_to="none",
    dataloader_num_workers=0,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained("tinyLLM")
tokenizer.save_pretrained("tinyLLM")