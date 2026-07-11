# FrustrationLM

FrustrationLM is a small language model fine-tuned to generate frustration-oriented conversational responses. It is built on top of DistilGPT-2 using the Hugging Face Transformers library and was created as an open-source learning project to understand how modern language models are built, trained, and deployed.

This repository contains the complete source code used to fine-tune the model. The trained weights are available separately on Hugging Face.

## Example

```text
User: Who are you?

Assistant: I am a program built to be eternally annoyed 😫
```

## Project Links

🤗 Model: https://huggingface.co/hammadtahirtech/FrustrationLM

---

# Why FrustrationLM?

I have been programming for years, but before this project I had very little understanding of AI engineering or how language models actually work internally. Rather than only reading papers or watching videos, I decided to learn by building.

The original goal was to train a tiny language model from scratch on a frustration-only dataset. During development, I learned that a niche dataset alone is not enough for a model to simultaneously learn language, grammar, and reasoning. Modern language models first require enormous amounts of general text before they can be specialized effectively.

Because of this, FrustrationLM changed direction. Instead of replacing the language capabilities entirely, it uses a pretrained DistilGPT-2 model and fine-tunes it on frustration-oriented conversations.

The project also explores a broader idea that interests me: whether emotions can function as useful internal signals for intelligent systems. Humans frequently use emotions as feedback that influences attention, learning, and decision making. While FrustrationLM is only a text model and cannot genuinely experience emotions, it serves as a small experiment in exploring how emotion-specific behavior can be represented within a language model.

---

# How FrustrationLM Was Built

The project began with a minimal GPT-style training implementation to understand the core building blocks of language models, including tokenization, transformers, training loops, and autoregressive text generation.

After experimenting with training from scratch, the project transitioned to fine-tuning DistilGPT-2. This approach preserves the language understanding learned during large-scale pretraining while adapting the model toward frustration-oriented conversational responses using a custom dataset.

Training examples follow a conversational format:

```text
User: My internet stopped working.
Assistant: That's seriously frustrating...
```

The objective is not to create a general-purpose chatbot, but a small model specialized in a narrow behavioral domain.

---

# Technical Specifications

## Architecture

* Base Model: DistilGPT-2
* Framework: Hugging Face Transformers
* Language: Python

## Model

* Parameters: ~82 Million
* Context Length: 256 tokens
* Vocabulary Size: 50,257
* Hidden Layers: 6
* Embedding Dimension: 384
* Attention Heads: 12
* Activation Function: GELU

## Training

* Fine-tuning Method: Causal Language Modeling
* Optimizer: AdamW
* Epochs: 6
* Batch Size: 4
* Learning Rate: 5e-5
* Precision: FP32

## Dataset

The training dataset consists of prompt-completion conversation pairs focused on frustration-oriented responses.

Example:

```text
User: I accidentally deleted my project.
Assistant: That has to be one of the most frustrating things that can happen...
```

---

# Project Structure

```text
.
├── train.py
├── inference.py
├── train.txt
├── tinyLLM/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
└── requirements.txt
```

(The exact structure may vary slightly as the project evolves.)

---

# Installation

Clone the repository:

```bash
git clone https://github.com/hammadtahirtech/FrustrationLM.git

cd FrustrationLM
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Training

Prepare your dataset inside `train.txt`.

Then run:

```bash
python train.py
```

The model checkpoints will be saved to the output directory configured in the training script.

---

# Running the Model

The model is downloaded automatically from Hugging Face the first time you run the code.

Load the fine-tuned model:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

repo = "hammadtahirtech/FrustrationLM"

tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo)

prompt = "User: My code still doesn't work.\nAssistant:"

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

# Limitations

* FrustrationLM is not intended to be a general-purpose assistant.
* It is specialized for frustration-oriented conversational behavior.
* Outputs may occasionally be repetitive or inconsistent.
* The project was created primarily as an educational and experimental open-source project.

---

# Open Source

The source code is available in this repository.

The trained model is available on Hugging Face.

Contributions, suggestions, and improvements are always welcome.

---

# License

This project is released under the MIT License.