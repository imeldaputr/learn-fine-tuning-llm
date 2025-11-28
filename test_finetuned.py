from dotenv import load_dotenv
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

load_dotenv()


# Load base model
base_model_name = os.getenv("LLM_MODEL_NAME")
print(f"Loading base model {base_model_name}...")

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)


# Load LoRA weights
print("Loading fine-tuned LoRA weights...")
model = PeftModel.from_pretrained(base_model, "./qwen-finetuned-lora")

# Test prompts
test_prompts = [
    "### Instruction: Write a function to calculate factorial\n### Output:",
    "### Instruction: Write a function to check palindrome\n### Output:",
    "### Instruction: Create a function to find maximum in a list\n### Output:", # New, not in training data
]

print("\n" + "="*70)
print("Testing Fine-tuned Model (with increased max tokens)")
print("="*70)

for prompt in test_prompts:
    instruction = prompt.split("### Instruction:")[1].split("\n")[0].strip()
    print(f"\n📝 Prompt: {instruction}")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=400,
        do_sample=False,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id # Stop at proper ending
    )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the output part
    output = result.split("### Output:")[-1].strip()
    
    print(f" Output:\n{output}\n")
    print("-"*50)