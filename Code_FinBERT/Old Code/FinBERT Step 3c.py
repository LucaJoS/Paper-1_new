import os
from datasets import Dataset
from transformers import (
    BertTokenizerFast,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_model, LoraConfig, TaskType
import torch

# Path to directory with ESG reports (test version)
input_dir = "/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/Code - Paper 1/1_extract_text_from_pdfs/clean_extracted_texts/aaaTesting - only 3 reports"

# Load FinBERT tokenizer and model
model_name = "ProsusAI/finbert"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

# LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"  # explicitly pass string instead of enum
)
model = get_peft_model(model, lora_config)

# Load ESG reports from text files
texts = []
instruction = (
    "ESG SCORE SYSTEM (0–100): 0 is bad, 100 is excellent. "
    "Score as an ESG Analyst.\n\n"
)
for file in os.listdir(input_dir):
    if file.endswith(".txt"):
        with open(os.path.join(input_dir, file), "r", encoding="utf-8") as f:
            texts.append(instruction + f.read())

# Tokenize
tokens = tokenizer(texts, truncation=True, padding=True, return_special_tokens_mask=True)
dataset = Dataset.from_dict({
    "input_ids": tokens["input_ids"],
    "attention_mask": tokens["attention_mask"],
    "special_tokens_mask": tokens["special_tokens_mask"]
})

# Data collator
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# Training setup
training_args = TrainingArguments(
    output_dir="./finbert_step3c_lora",
    per_device_train_batch_size=1,
    num_train_epochs=10,
    learning_rate=5e-5,
    save_total_limit=1,
    logging_steps=10,
    report_to="none"
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator,
)

trainer.train()

# Save LoRA-tuned model
model.save_pretrained("./finbert_step3c_lora")
tokenizer.save_pretrained("./finbert_step3c_lora")
print("✅ LoRA-tuned FinBERT saved to ./finbert_step3c_lora")