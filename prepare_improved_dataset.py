from datasets import load_from_disk, Dataset

print("Loading original dataset...")
dataset = load_from_disk("./code_alpaca_100")

def format_with_end_marker(example):
    """Format with explicit END marker to teach stopping"""
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output = example.get('output', '')
    
    # Clean output - remove any existing formatting issues
    output = output.strip()
    
    if input_text:
        text = f"""### Instruction: {instruction}
### Input: {input_text}
### Output:
```python
{output}
```
### END"""
    else:
        text = f"""### Instruction: {instruction}
### Output:
```python
{output}
```
### END"""
    
    return {"text": text}

# Apply formatting to entire dataset
improved_dataset = dataset.map(format_with_end_marker, remove_columns=dataset.column_names)


print(f"Formatted {len(improved_dataset)} examples")
print("\n📝 Sample formatted example:")
print(improved_dataset[0]['text'])
print("\n" + "="*70)

# Save
improved_dataset.save_to_disk("./code_alpaca_100_improved")
print("✅ Improved dataset saved to ./code_alpaca_100_improved")