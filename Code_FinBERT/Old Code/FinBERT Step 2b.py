import os
os.environ["TRANSFORMERS_NO_PEFT"] = "1"  # Prevents peft backend import crash

from transformers import BertTokenizerFast, BertForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import torch

# Load tokenizer and model
model_name = "ProsusAI/finbert"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

# Load rubric text
rubric_path = "/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/summarized_lesg_esg_methodology.txt"
with open(rubric_path, "r", encoding="utf-8") as f:
    text = f.read()

# Tokenize rubric into short blocks
inputs = tokenizer(text, return_special_tokens_mask=True, return_tensors="pt", truncation=True)
tokens = tokenizer([text], truncation=True, return_special_tokens_mask=True)

# Wrap as HF Dataset
dataset = Dataset.from_dict({"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"], "special_tokens_mask": tokens["special_tokens_mask"]})

# Setup training
training_args = TrainingArguments(
    output_dir="./finbert_step2b_mlm",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=1,
    save_steps=50,
    save_total_limit=1,
    logging_steps=10,
    learning_rate=5e-5,
    prediction_loss_only=True,
    report_to="none"
)

# Use dynamic masking
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Train
trainer.train()
model.save_pretrained("./finbert_step2b_mlm")
tokenizer.save_pretrained("./finbert_step2b_mlm")
print("Tuned model saved to ./finbert_step2b_mlm")
