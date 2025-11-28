from dotenv import load_dotenv
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import Dataset
#from trl import SFTTrainer
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

# Dummys dataset
data = {
    "text": [
        """### Instruction: Write a function to calculate factorial
### Output:
``````````````````````python
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n-1)
``````````````````````""",
        
       """### Instruction: Create a function to check if number is prime
### Output:
````````````````````python
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
```````````````````""",
        
        """### Instruction: Write a function to reverse a string
### Output:
``````````````````python
def reverse_string(s):
    return s[::-1]
`````````````````""",
        
        """### Instruction: Create a function to find fibonacci number
### Output:
````````````````python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```````````````""",
        
        """### Instruction: Write a function to check palindrome
### Output:
``````````````python
def is_palindrome(s):
    return s == s[::-1]
`````````````""",
        
        """### Instruction: Create a function to find sum of list
### Output:
````````````python
def sum_list(numbers):
    return sum(numbers)
```````````""",
        
        """### Instruction: Write a function to find minimum value
### Output:
``````````python
def find_min(numbers):
    return min(numbers)
`````````""",
        
        """### Instruction: Create a function to count vowels in string
### Output:
````````python
def count_vowels(s):
    vowels = 'aeiouAEIOU'
    return sum(1 for char in s if char in vowels)
````````""",
    ]
}

dataset = Dataset.from_dict(data)

# Tokenize Dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# LoRA config
lora_config = LoraConfig(
    r=16, # Rank - bigger r means more parameters to train
    lora_alpha=32,
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
    output_dir="./results_improved",
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4, 
    learning_rate=2e-4, 
    logging_steps=1, # Log every 10 steps
    save_strategy="epoch",
    fp16=True, # Use mixed precision
    remove_unused_columns=False,
    save_total_limit=2, # Keep only best 2 checkpoints
)

# # Trainer
# trainer = SFTTrainer(
#     model=model,
#     train_dataset=dataset,
#     args=training_args,
#     formatting_func=lambda x: x["text"],
#     tokenizer=tokenizer,
#     max_seq_length=512
# )

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False # Causal LM, not masked LM
)

# Standard Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    #tokenizer=tokenizer
)

# Start training
print("\n" + "="*50)
print("Starting IMPROVED training...")
print(f"- Dataset: 8 examples (up from 5)")
print(f"- LoRA rank: 16 (up from 8)")
print(f"- Epochs: 5 (up from 3)")
print("="*50 + "\n")
trainer.train()


# Save the fine-tuned model
trainer.save_model("./qwen-finetuned-lora-improved")
tokenizer.save_pretrained("./qwen-finetuned-lora-improved")
print("\n✅ Training complete~~ Model saved to ./qwen-finetuned-lora-improved")