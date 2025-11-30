from dotenv import load_dotenv
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk
import torch

load_dotenv()
model_name = os.getenv("LLM_MODEL_NAME")

print(f"Loading model {model_name}...")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_size = "right"

# Load dataset
print("Loading CodeAlpaca dataset...")
dataset = load_from_disk("./code_alpaca_100_improved")

print(f"Loaded {len(dataset)} examples")
print("\nSample:")
print(dataset[0]['text'][:300] + "...")


# Tokenization
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, max_length=512, padding="max_length")

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])


# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
print("\nTrainable parameters:")
model.print_trainable_parameters()


# Training arguments
training_args = TrainingArguments(
    output_dir="./results_real_dataset",
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Increased batch size
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    remove_unused_columns=False,
    save_total_limit=2,
    warmup_steps=10,  # Warmup for stability
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("\n" + "="*70)
print("Starting training with real dataset")
print("="*70)
print(f"- Dataset: 100 examples (with END markers)")
print(f"- LoRA rank: 16")
print(f"- Epochs: 3")
print("="*70 + "\n")


trainer.train()

trainer.save_model("./qwen-finetuned-real-dataset")
tokenizer.save_pretrained("./qwen-finetuned-real-dataset")
print("\n✅ Training complete~~ Model saved to ./qwen-finetuned-real-dataset")

