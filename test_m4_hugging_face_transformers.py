from dotenv import load_dotenv
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

load_dotenv()
model_name = os.getenv("LLM_MODEL_NAME")
if model_name is None:
    raise ValueError("LLM_MODEL_NAME is not set in the environment variables.")

print(f"Loading model {model_name}...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="mps", # Metal Performance Shaders
    low_cpu_mem_usage=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "def calculate_sum(numbers):\n "
inputs = tokenizer(prompt, return_tensors="pt").to("mps")

print("Generating...")
outputs = model.generate(
    **inputs,
    max_length=100
)
print(tokenizer.decode(outputs[0]))