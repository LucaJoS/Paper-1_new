


import os
import pandas as pd
from datasets import Dataset
from transformers import (
    BertTokenizerFast,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_model, LoraConfig, TaskType

# Load ESG-score prompts
df = pd.read_excel("/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/FinBERT Output/FinBERT Step XXX Database Builder.xlsx")
df = df[["prompt_header"]].dropna()
df.rename(columns={"prompt_header": "text"}, inplace=True)

# Tokenizer and model
model_name = "ProsusAI/finbert"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
base_model = BertForMaskedLM.from_pretrained(model_name)

# Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(base_model, lora_config)

# Tokenize
tokenized = tokenizer(list(df["text"]), truncation=True, padding=True, return_special_tokens_mask=True)
dataset = Dataset.from_dict({
    "input_ids": tokenized["input_ids"],
    "attention_mask": tokenized["attention_mask"],
    "special_tokens_mask": tokenized["special_tokens_mask"]
})

# Training
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
training_args = TrainingArguments(
    output_dir="./finbert_step4c_lora",
    num_train_epochs=10,
    per_device_train_batch_size=1,
    learning_rate=5e-5,
    save_total_limit=1,
    logging_steps=10,
    report_to="none"
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)
trainer.train()

# Save
model.save_pretrained("./finbert_step4c_lora")
tokenizer.save_pretrained("./finbert_step4c_lora")
print("✅ LoRA-tuned FinBERT saved to ./finbert_step4c_lora")