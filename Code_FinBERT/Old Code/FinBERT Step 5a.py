

import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizerFast, BertModel

# Load FinBERT base model and tokenizer
tokenizer = BertTokenizerFast.from_pretrained("ProsusAI/finbert")
base_model = BertModel.from_pretrained("ProsusAI/finbert")

# Define regression model
class FinBERTRegressor(nn.Module):
    def __init__(self, base_model):
        super(FinBERTRegressor, self).__init__()
        self.bert = base_model
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output
        return self.regressor(cls_output).squeeze(-1)

# Load ESG rubric
with open("/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/summarized_lesg_esg_methodology.txt", "r", encoding="utf-8") as f:
    rubric = f.read().strip()

# Load prebuilt database (includes file_path, ESG scores, and metadata)
db_path = "/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/FinBERT Output/FinBERT Step XXX Database Builder.xlsx"
df = pd.read_excel(db_path)

# Prompt constructor
def build_prompt(row):
    filing_date = pd.to_datetime(row.get('filing_date', '1900-01-01'))
    return (
        "You are a professional ESG analyst.\n"
        "Based on the following ESG scoring rubric and ESG metadata, read the ESG disclosure and estimate an ESG score from 0 (worst) to 100 (best).\n"
        "Reply with a single number. No explanations. Just a number.\n\n"
        f"{rubric}\n\n"
        f"Company: {row.get('ticker', 'UNKNOWN')}, Filing Date: {filing_date.strftime('%Y-%m-%d')}\n"
        f"Refinitiv ESG Scores:\n"
        f"- ESG Score: {row.get('esg_score', 'N/A')}\n"
        f"- ESG Controversy Score: {row.get('esg_controversy_score', 'N/A')}\n"
        f"- ESG Combined Score (ESGC): {row.get('esgc_score', 'N/A')}\n"
        "---"
    )

results = []
for idx, row in df.iterrows():
    file_path = row["file_path"]
    if not os.path.exists(file_path):
        print(f"❌ Missing file: {file_path}")
        continue

    with open(file_path, "r", encoding="utf-8") as f:
        report = f.read()

    prompt = build_prompt(row)
    full_input = prompt + "\n\n" + report

    scores_run = []
    for _ in range(10):
        model = FinBERTRegressor(base_model)
        model.eval()

        encoded = tokenizer.encode_plus(
            full_input,
            add_special_tokens=True,
            max_length=20480,
            truncation=True,
            return_attention_mask=False
        )["input_ids"]

        max_len, stride = 512, 256
        chunks = [encoded[i:i+max_len] for i in range(0, len(encoded), stride) if len(encoded[i:i+max_len]) >= 10]

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
output_path = "/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/FinBERT Output/finbert_step5a_output.xlsx"
df_out.to_excel(output_path, index=False)
print(f"✅ Output saved to: {output_path}")
print(df_out)