from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import torch

# Load model + tokenizer
model_name = "NousResearch/Llama-2-7b-hf"  # or use LLaMA 3 if access granted
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="auto")

# Apply LoRA
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, config)

# Load and preprocess dataset
dataset = load_dataset("json", data_files="data/train.json")

def format(example):
    input_ids = tokenizer(example["prompt"], truncation=True, padding="max_length", max_length=256, return_tensors="pt").input_ids[0]
    label_ids = tokenizer(example["completion"], truncation=True, padding="max_length", max_length=64, return_tensors="pt").input_ids[0]
    return {"input_ids": input_ids, "labels": label_ids}

tokenized = dataset["train"].map(format)

# Training
args = TrainingArguments(
    output_dir="models/adapter",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    fp16=True,
    logging_dir="outputs/logs",
    save_strategy="epoch",
    learning_rate=2e-4,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
)

trainer.train()
model.save_pretrained("models/adapter")
