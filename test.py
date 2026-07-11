from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="./tinyLLM",
    tokenizer="./tinyLLM",
)

user_input = "Hello"
prompt = f"Prompt: {user_input}\nResponse:"
out = pipe(prompt, max_new_tokens=20, do_sample=False, temperature=0.7)

print(out[0]["generated_text"])