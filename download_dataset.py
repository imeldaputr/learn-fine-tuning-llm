from datasets import load_dataset

print("Downloading CodeAlpaca dataset...")
dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")

# Take only first 100 examples for quick training
dataset = dataset.select(range(100))

print(f"Downloaded {len(dataset)} examples")
print("\nSample:")
print(dataset[0])

# Save to local
dataset.save_to_disk("./code_alpaca_100")
print("\nDataset saved to ./code_alpaca_100")