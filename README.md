# 🦙 LLaMA Sentiment Fine-Tuning (PyTorch + LoRA)

This project fine-tunes a LLaMA-based language model for sentiment analysis using **LoRA (Low-Rank Adaptation)** with **PyTorch**. It demonstrates how to adapt large language models efficiently on custom classification tasks using parameter-efficient fine-tuning.

---

## 📌 Project Highlights

- ✅ Fine-tuned Meta’s LLaMA model for binary sentiment classification (positive/negative)
- ✅ Used HuggingFace’s `transformers`, `datasets`, and `peft` libraries
- ✅ Integrated **LoRA** to significantly reduce GPU memory and training time
- ✅ Trained on a lightweight IMDb-style custom dataset
- ✅ Supports inference with LoRA adapters

---

## 🛠️ Tech Stack

- **Model:** LLaMA-2 7B (HuggingFace format)
- **Framework:** PyTorch
- **Fine-Tuning Method:** LoRA via PEFT
- **Tokenizer:** LLaMA tokenizer (sentencepiece)
- **Dataset:** IMDb-like dataset for sentiment classification
- **Training Strategy:** Gradient checkpointing + mixed precision

---

## 🧾 Directory Structure

llama-sentiment-finetune/
│
├── data/
  └── train.json # Processed training & validation data
│
├── models/
│ └── llama/ # Base LLaMA weights (HF format)
│
├── adapters/ # Trained LoRA adapters after fine-tuning
│
├── scripts/
│ ├── finetune.py # Fine-tuning script
│ └── prepare_dataset.py # fir preparing dataset
│
├── requirements.txt # Python dependencies
└── README.md
