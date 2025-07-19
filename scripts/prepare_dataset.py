from datasets import load_dataset
import json
import os
from tqdm import tqdm

# Load IMDb dataset from Hugging Face
dataset = load_dataset("imdb")

# Function to convert example to prompt/completion format
def format_example(example):
    sentiment = "Positive" if example["label"] == 1 else "Negative"
    prompt = f"Classify the sentiment: {example['text'].strip()[:500]} Sentiment:"
    return {"prompt": prompt, "completion": f" {sentiment}"}

# Create output folder if needed
os.makedirs("data", exist_ok=True)

# Prepare training data (you can change to "test" or use both)
train_data = dataset["train"].select(range(10000))  # use 10k for quick fine-tuning

# Convert and save
formatted_data = [format_example(ex) for ex in tqdm(train_data)]
with open("data/train.json", "w", encoding="utf-8") as f:
    json.dump(formatted_data, f, indent=2)

print("✅ Dataset saved to data/train.json")
