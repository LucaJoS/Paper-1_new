import os
import torch
from transformers import BertTokenizerFast, BertForMaskedLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset

# Load base FinBERT model and tokenizer
model_name = "ProsusAI/finbert"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
base_model = BertForMaskedLM.from_pretrained(model_name)

# Apply LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    task_type=TaskType.CAUSAL_LM,
    lora_dropout=0.1,
    bias="none"
)
model = get_peft_model(base_model, peft_config)

# Load rubric text
rubric_path = "/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/summarized_lesg_esg_methodology.txt"
with open(rubric_path, "r", encoding="utf-8") as f:
    text = f.read()

# Tokenize and prepare dataset
inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=512)
dataset = Dataset.from_dict({
    "input_ids": inputs["input_ids"],
    "attention_mask": inputs["attention_mask"]
})

# Setup data collator for MLM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# Training configuration
training_args = TrainingArguments(
    output_dir="./finbert_step2c_lora",
    num_train_epochs=10,
    per_device_train_batch_size=1,
    save_steps=50,
    save_total_limit=1,
    logging_steps=10,
    learning_rate=5e-5,
    report_to="none"
)

# Trainer initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# Train and save
trainer.train()
model.save_pretrained("./finbert_step2c_lora")
tokenizer.save_pretrained("./finbert_step2c_lora")
print("✅ LoRA-tuned FinBERT saved to ./finbert_step2c_lora")
