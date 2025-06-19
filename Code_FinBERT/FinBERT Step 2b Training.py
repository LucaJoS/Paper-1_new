import os
import re
from transformers import (
    BertTokenizerFast, BertForMaskedLM,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import torch
from tqdm import tqdm

os.environ["TRANSFORMERS_NO_PEFT"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def tokenize_text(text, tokenizer, max_length=512):
    tokenized = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_overflowing_tokens=True,
        return_special_tokens_mask=True,
        return_attention_mask=True
    )
    return tokenized

def train_model(dataset, tokenizer, model, output_dir, device):
    print(f"Training on {len(dataset)} rubric chunks")
    print(f"Example chunk tokens: {dataset[0]['input_ids'][:20]}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=1,
        save_steps=50,
        save_total_limit=1,
        logging_steps=10,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        learning_rate=5e-5,
        prediction_loss_only=True,
        report_to="none"
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    model.to(device)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Tuned model saved to {output_dir}")

def main():
    model_name = "ProsusAI/finbert"
    rubric_path = "/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/aaa Clean Slate/a_Input_Methodology/lseg-esg-scores-methodology.txt"
    output_dir = "./finbert_step2b_mlm"

    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)

    text = load_text(rubric_path)
    tokens = tokenize_text(text, tokenizer)

    dataset = Dataset.from_dict({
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "special_tokens_mask": tokens["special_tokens_mask"]
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(dataset, tokenizer, model, output_dir, device)

if __name__ == "__main__":
    main()
