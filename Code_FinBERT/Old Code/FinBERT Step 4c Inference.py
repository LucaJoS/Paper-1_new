import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizerFast, BertForMaskedLM
from peft import PeftModel

# Load tokenizer and LoRA model
model_path = "./finbert_step4c_lora"
tokenizer = BertTokenizerFast.from_pretrained(model_path)
base_model = BertForMaskedLM.from_pretrained("ProsusAI/finbert")
base_model = PeftModel.from_pretrained(base_model, model_path)
base_model.eval()
base_model.config.output_hidden_states = True

class FinBERTRegressor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.regressor = nn.Linear(model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.hidden_states[-1][:, 0, :]
        return self.regressor(cls).squeeze(-1)

model = FinBERTRegressor(base_model)
model.eval()

# Load prompts and file paths
df = pd.read_excel("/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/FinBERT Output/FinBERT Step XXX Database Builder.xlsx")
results = []

for idx, row in df.iterrows():
    file_path = row["file_path"]
    prompt = row["prompt_header"]

    if not os.path.exists(file_path):
        print(f"❌ Missing file: {file_path}")
        continue

    print(f"Processing {idx+1}/{len(df)}: {os.path.basename(file_path)}")

    with open(file_path, "r", encoding="utf-8") as f:
        disclosure = f.read()

    full_input = prompt + "\n\n" + disclosure

    encoded = tokenizer.encode_plus(
        full_input,
        add_special_tokens=True,
        max_length=20480,
        truncation=True,
        return_attention_mask=False
    )["input_ids"]

    max_len, stride = 512, 256
    chunks = [encoded[i:i+max_len] for i in range(0, len(encoded), stride) if len(encoded[i:i+max_len]) >= 10]

    scores_run = []
    for _ in range(10):
        chunk_scores = []
        for chunk in chunks:
            input_tensor = torch.tensor([chunk])
            attention_mask = torch.ones_like(input_tensor)
            with torch.no_grad():
                score = model(input_tensor, attention_mask=attention_mask)
                chunk_scores.append(score.item())
        raw_mean = np.mean(chunk_scores)
        scaled = (raw_mean + 1.5) / 3.0 * 100
        scores_run.append(round(np.clip(scaled, 0, 100), 2))

    avg = round(np.mean(scores_run), 2)
    std = round(np.std(scores_run), 2)
    mn, mx = min(scores_run), max(scores_run)
    med = round(np.median(scores_run), 2)
    iqr = round(np.percentile(scores_run, 75) - np.percentile(scores_run, 25), 2)
    skew = round((avg - med) / (std + 1e-6), 2)
    short_name = re.search(r"\d{4}-[A-Za-z]{3}-\d{2}", file_path)
    short_name = short_name.group(0) if short_name else os.path.basename(file_path)

    results.append([short_name, avg, std, mn, mx, med, iqr, skew] + scores_run)

# Save output
columns = ["File", "Avg. Estimated Score", "Score Volatility", "Score Min", "Score Max",
           "Score Median", "Score IQR", "Score Skewness"] + [f"Est. Score {i+1}" for i in range(10)]

df_out = pd.DataFrame(results, columns=columns)
output_path = "/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/FinBERT Output/finbert_step4c_output.xlsx"
df_out.to_excel(output_path, index=False)
print(f"✅ Output saved to: {output_path}")
print(df_out)