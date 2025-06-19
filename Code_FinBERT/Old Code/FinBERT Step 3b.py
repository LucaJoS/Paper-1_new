from transformers import BertTokenizerFast, BertForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import os

# === Load and concatenate ESG report texts ===
input_dir = "/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/Code - Paper 1/1_extract_text_from_pdfs/clean_extracted_texts/aaaTesting - only 3 reports"
texts = []
for file in os.listdir(input_dir):
    if file.endswith(".txt"):
        with open(os.path.join(input_dir, file), "r", encoding="utf-8") as f:
            texts.append(f.read())
corpus = "\n\n".join(texts)

# === Load tokenizer and model ===
tokenizer = BertTokenizerFast.from_pretrained("ProsusAI/finbert")
model = BertForMaskedLM.from_pretrained("ProsusAI/finbert")

# === Tokenize corpus ===
tokens = tokenizer([corpus], truncation=True, return_special_tokens_mask=True)

# === Wrap as HuggingFace Dataset ===
dataset = Dataset.from_dict({
    "input_ids": tokens["input_ids"],
    "attention_mask": tokens["attention_mask"],
    "special_tokens_mask": tokens["special_tokens_mask"]
})

# === MLM fine-tuning setup ===
training_args = TrainingArguments(
    output_dir="./finbert_step3b_mlm",
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

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

# === Train and save ===
trainer.train()
model.save_pretrained("./finbert_step3b_mlm")
tokenizer.save_pretrained("./finbert_step3b_mlm")
print("✅ MLM-tuned model saved to ./finbert_step3b_mlm")