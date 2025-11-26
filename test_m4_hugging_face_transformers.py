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
    device_map="auto", # mps: Metal Performance Shaders
    #trust_remote_code=True
    #low_cpu_mem_usage=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

#prompt = "def calculate_sum(numbers):\n "
prompt = "write a quick sort algorithm."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

# Apply chat template
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Tokenize and move to device
inputs = tokenizer([text], return_tensors="pt").to(model.device) 

outputs = model.generate(
    **inputs,
    max_new_tokens=512
)

# Decode the result
result = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print("\n=== RESULT ===")
print(result)

# outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, pad_token_id=tokenizer.pad_token_id)

# result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
# print(result)

# print("Generating...")
# outputs = model.generate(
#     **inputs,
#     max_length=100
# )
# print(tokenizer.decode(outputs[0]))