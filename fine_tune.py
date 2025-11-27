from dotenv import load_dotenv
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer
import torch

load_dotenv()
model_name = os.getenv("LLM_MODEL_NAME")

print(f"Loading model {model_name}...")

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Dummys
data = {
    "text": [
        "### Instruction: Write a function to calculate factorial\n### Output: def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n-1)",
        
        "### Instruction: Create a function to check if number is prime\n### Output: def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
        
        "### Instruction: Write a function to reverse a string\n### Output: def reverse_string(s):\n    return s[::-1]",
        
        "### Instruction: Create a function to find fibonacci number\n### Output: def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        
        "### Instruction: Write a function to check palindrome\n### Output: def is_palindrome(s):\n    return s == s[::-1]",
    ]
}

dataset = Dataset.from_dict(data)


# LoRA config
lora_config = LoraConfig(
    r=8, # Rank - bigger r means more parameters to train
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Attention layers
    lora_dropout=0.05, # Dropout for regularization
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
print(f"Trainable parameters: {model.print_trainable_parameters()}")


# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4, 
    learning_rate=2e-4, 
    logging_steps=10, # Log every 10 steps
    save_strategy="epoch",
    fp16==True, # Use mixed precision
)

# Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    dataset_text_field="text",
    tokenizer=tokenizer,
    max_seq_length=512,
)

print("Starting training...")
trainer.train()


# Save the fine-tuned model
trainer.save_model("./qwen-finetuned-lora")
print("Training complete~~ Model saved to ./qwen-finetuned-lora")