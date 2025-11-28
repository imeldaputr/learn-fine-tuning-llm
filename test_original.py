# To compare the fine-tuned with the original model

from dotenv import load_dotenv
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

load_dotenv()

model_name = os.getenv("LLM_MODEL_NAME")
print(f"Loading ORIGINAL model {model_name}...")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

prompt = "### Instruction: Write a function to calculate factorial\n### Output:"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=150,
    do_sample=False,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id
)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\n Original Model Output:\n{result}")