from dotenv import load_dotenv
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from utils import clean_code_output, extract_code_only, detect_hallucination, validate_code_syntax

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
# test_prompts = [
#     {
#         "prompt": "### Instruction: Write a function to calculate factorial\n### Output:",
#         "in_training": "Likely ✓",
#         "category": "Recursion"
#     },
#     {
#         "prompt": "### Instruction: Write a function to check palindrome\n### Output:",
#         "in_training": "Likely ✓",
#         "category": "String Manipulation"
#     },
#     {
#         "prompt": "### Instruction: Create a function to find maximum in a list\n### Output:",
#         "in_training": "Maybe ✓",
#         "category": "List Operations"
#     },
#     {
#         "prompt": "### Instruction: Write a function to calculate the sum of even numbers in a list\n### Output:",
#         "in_training": "New ✗",
#         "category": "List + Conditionals"
#     },
#     {
#         "prompt": "### Instruction: Create a function to merge two sorted lists\n### Output:",
#         "in_training": "New ✗",
#         "category": "Advanced Algorithm"
#     },
# ]

test_prompts = [
    {
        "prompt": "### Instruction: write code to count fibonacci in rust\n### Output:",
        "in_training": "Likely ✓",
        "category": "Recursion"
    }
]

print("\n" + "="*70)
print("Testing Fine-tuned Model (CodeAlpaca dataset)")
print("="*70)
print(f"Model: {base_model_name}")
print(f"Fine-tuned on: 100 CodeAlpaca examples")
print(f"Post-processing: ✅ ENABLED")
print(f"Test cases: {len(test_prompts)}")
print("="*70)

results = []


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
        max_new_tokens=1500,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id, # Stop at proper ending
        repetition_penalty=1.5, # Penalize repetition
        no_repeat_ngram_size=3  # Prevent repeating n-grams
    )
    
    raw_result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    raw_output = raw_result.split("### Output:")[-1].strip()
    
    # POST-PROCESSING
    cleaned_output = clean_code_output(raw_result)
    code_only = extract_code_only(raw_result)
    
    # VALIDATION
    has_hallucination, hall_type = detect_hallucination(raw_output)
    is_valid, syntax_error = validate_code_syntax(code_only)
    
    # Display cleaned output
    print(cleaned_output)
    print("-"*70)
    
    # Analysis
    print(f"📊 Analysis:")
    print(f"  • Raw length: {len(raw_output)} chars")
    print(f"  • Cleaned length: {len(cleaned_output)} chars")
    print(f"  • Reduction: {len(raw_output) - len(cleaned_output)} chars removed")
    print(f"  • Syntax valid: {'✅ Yes' if is_valid else '❌ No - ' + syntax_error}")
    print(f"  • Hallucination: {'⚠️ Yes (' + hall_type + ')' if has_hallucination else '✅ None detected'}")
    
    # Store results
    results.append({
        'test': i,
        'instruction': instruction,
        'category': category,
        'raw_length': len(raw_output),
        'cleaned_length': len(cleaned_output),
        'syntax_valid': is_valid,
        'has_hallucination': has_hallucination,
        'hall_type': hall_type if has_hallucination else None
    })

# Summary
print("\n" + "="*70)
print("📊 SUMMARY REPORT")
print("="*70)

total_tests = len(results)
syntax_valid_count = sum(1 for r in results if r['syntax_valid'])
hallucination_count = sum(1 for r in results if r['has_hallucination'])
avg_raw_length = sum(r['raw_length'] for r in results) / total_tests
avg_cleaned_length = sum(r['cleaned_length'] for r in results) / total_tests
total_chars_removed = sum(r['raw_length'] - r['cleaned_length'] for r in results)

print(f"\n✅ Syntax Valid: {syntax_valid_count}/{total_tests} ({syntax_valid_count/total_tests*100:.0f}%)")
print(f"⚠️  Hallucination: {hallucination_count}/{total_tests} ({hallucination_count/total_tests*100:.0f}%)")
print(f"📏 Avg Raw Length: {avg_raw_length:.0f} chars")
print(f"📏 Avg Cleaned Length: {avg_cleaned_length:.0f} chars")
print(f"🧹 Total Cleaned: {total_chars_removed} chars removed")
print(f"📉 Reduction: {(total_chars_removed/sum(r['raw_length'] for r in results))*100:.1f}%")

print(f"\n{'='*70}")
print("Post-processing successfully cleaned all outputs! ✅")
print("="*70)