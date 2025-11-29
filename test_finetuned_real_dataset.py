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
print("Loading fine-tuned LoRA weights (with CodeAlpaca dataset)...")
model = PeftModel.from_pretrained(base_model, "./qwen-finetuned-real-dataset")


# Test prompts
test_prompts = [
    {
        "prompt": "### Instruction: Write a function to calculate factorial\n### Output:",
        "in_training": "Likely ✓",
        "category": "Recursion"
    },
    {
        "prompt": "### Instruction: Write a function to check palindrome\n### Output:",
        "in_training": "Likely ✓",
        "category": "String Manipulation"
    },
    {
        "prompt": "### Instruction: Create a function to find maximum in a list\n### Output:",
        "in_training": "Maybe ✓",
        "category": "List Operations"
    },
    {
        "prompt": "### Instruction: Write a function to calculate the sum of even numbers in a list\n### Output:",
        "in_training": "New ✗",
        "category": "List + Conditionals"
    },
    {
        "prompt": "### Instruction: Create a function to merge two sorted lists\n### Output:",
        "in_training": "New ✗",
        "category": "Advanced Algorithm"
    },
]

print("\n" + "="*70)
print("Testing Fine-tuned Model (CodeAlpaca dataset)")
print("="*70)
print(f"Model: {base_model_name}")
print(f"Fine-tuned on: 100 CodeAlpaca examples")
print(f"Test cases: {len(test_prompts)}")
print("="*70)


for i, test_case in enumerate(test_prompts, 1):
    prompt = test_case["prompt"]
    in_training = test_case["in_training"]
    category = test_case["category"]
    
    instruction = prompt.split("### Instruction:")[1].split("\n")[0].strip()
    
    print(f"\n[Test {i}/{len(test_prompts)}] {instruction}")
    print(f"Category: {category} | Training data: {in_training}")
    print("-"*70)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id, # Stop at proper ending
        repetition_penalty=1.1 # Penalize repetition
    )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output = result.split("### Output:")[-1].strip()
    
    # Count output length for analysis
    output_lines = output.split('\n')
    output_length = len(output)
    
    print(output)
    print("-"*70)
    print(f"Output length: {output_length} chars, {len(output_lines)} lines")
    

print("\n" + "="*70)
print("Testing Complete!")
print("="*70)
print("\n📊 Evaluation Criteria:")
print("1. Does output stop properly? (No hallucination)")
print("2. Is code syntactically correct?")
print("3. Does it solve the task correctly?")
print("4. Is formatting clean and consistent?")
print("5. Can it generalize to new tasks (Test 4-5)?")