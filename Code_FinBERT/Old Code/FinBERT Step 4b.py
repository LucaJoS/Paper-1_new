

import subprocess
# Prevent macOS from sleeping during training
subprocess.run("caffeinate -dimsu &", shell=True)

import os
import pandas as pd
from transformers import BertTokenizerFast, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Load ESG-score enriched prompts
df = pd.read_excel("/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/FinBERT Output/FinBERT Step XXX Database Builder.xlsx")
df = df[["prompt_header"]].dropna()
df.rename(columns={"prompt_header": "text"}, inplace=True)

# Load tokenizer
tokenizer = BertTokenizerFast.from_pretrained("ProsusAI/finbert")

# Tokenization
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

dataset = Dataset.from_pandas(df)
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# Load model
model = BertForMaskedLM.from_pretrained("ProsusAI/finbert")

# Training setup
training_args = TrainingArguments(
    output_dir="./finbert_step4b_mlm",
    overwrite_output_dir=True,
    per_device_train_batch_size=4,
    num_train_epochs=10,
    learning_rate=5e-5,
    save_total_limit=1,
    save_steps=10000,
    logging_steps=100,
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
trainer.save_model("./finbert_step4b_mlm")
print("✅ MLM-tuned model saved to ./finbert_step4b_mlm")