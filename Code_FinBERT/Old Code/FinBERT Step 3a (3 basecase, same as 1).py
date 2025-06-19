import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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

# Regression head
class FinBERTRegressor(nn.Module):
    def __init__(self, base_model):
        super(FinBERTRegressor, self).__init__()
        self.bert = base_model
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output  # [CLS] token
        return self.regressor(cls_output).squeeze(-1)

# Folder paths
input_dir = "/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/Code - Paper 1/1_extract_text_from_pdfs/clean_extracted_texts/aaaTesting - only 3 reports"
output_file = "/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/FinBERT Output/finbert_step3a_output.xlsx"

# Prompt prefix
prompt = (
    "ESG SCORE SYSTEM (0–100): 0 is bad, 100 is excellent. "
    "Score as an ESG Analyst.\n\n"
)

results = []
files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]

# Process
for idx, filename in enumerate(files, 1):
    print(f"Processing file {idx}/{len(files)}: {filename}")
    with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f:
        disclosure_text = f.read()

    full_input = prompt + disclosure_text

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

        max_len = 512
        stride = 256
        chunks = [encoded[i:i+max_len] for i in range(0, len(encoded), stride) if len(encoded[i:i+max_len]) >= 10]

        scores = []
        for input_ids in chunks:
            input_tensor = torch.tensor([input_ids])
            attention_mask = torch.ones_like(input_tensor)
            with torch.no_grad():
                score = model(input_tensor, attention_mask=attention_mask)
                scores.append(score.item())

        raw_mean = np.mean(scores)
        rescaled_score = (raw_mean + 1.5) / 3.0 * 100
        final_score = float(np.clip(rescaled_score, 0, 100))
        scores_run.append(round(final_score, 2))

    # Metrics
    avg_score = round(np.mean(scores_run), 2)
    std_dev = round(np.std(scores_run), 2)
    score_min = round(np.min(scores_run), 2)
    score_max = round(np.max(scores_run), 2)
    iqr = round(np.percentile(scores_run, 75) - np.percentile(scores_run, 25), 2)
    median = round(np.median(scores_run), 2)
    skew = round(((np.mean(scores_run) - median) / np.std(scores_run + [1e-6])), 2)

    match = re.search(r"\d{4}-[A-Za-z]{3}-\d{2}", filename)
    short_name = match.group(0) if match else filename

    results.append(
        [short_name, avg_score, std_dev, score_min, score_max, iqr, median, skew] + scores_run
    )

# Save output
columns = [
    "File", "Avg. Estimated Score", "Score Volatility", "Score Min", "Score Max",
    "Score IQR", "Score Median", "Score Skew"
] + [f"Est. Score {i+1}" for i in range(10)]

df = pd.DataFrame(results, columns=columns)
df.to_excel(output_file, index=False)
print(f"✅ Output saved to: {output_file}")
print(df)