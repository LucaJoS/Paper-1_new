import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
import re
import random
import numpy as np
import torch
import pandas as pd
from transformers import BertTokenizer, BertTokenizerFast, BertModel
import torch.nn as nn

# Load FinBERT base model and tokenizer
tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
tokenizer_fast = BertTokenizerFast.from_pretrained("ProsusAI/finbert")
base_model = BertModel.from_pretrained("ProsusAI/finbert")

# Regression model definition
class FinBERTRegressor(nn.Module):
    def __init__(self, base_model):
        super(FinBERTRegressor, self).__init__()
        self.bert = base_model
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output
        score = self.regressor(cls_output)
        return score.squeeze(-1)

# Load Refinitiv rubric from dummy file
with open('/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/summarized_lesg_esg_methodology.txt', "r", encoding="utf-8") as f:
    rubric_text = f.read().strip()

# Directory containing test disclosures
input_dir = ('/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/Code - Paper 1/1_extract_text_from_pdfs/clean_extracted_texts/aaaTesting - only 3 reports')
results = []

all_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
total_files = len(all_files)

# Process each file
for idx, filename in enumerate(all_files, start=1):
    print(f"Processing file {idx}/{total_files}: {filename}")
    with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as file:
        text = file.read()

    # Combine rubric + ESG document
    prepended_text = rubric_text + "\n\n" + text

    scores_run = []
    for _ in range(10):
        model = FinBERTRegressor(base_model)
        model.eval()

        encoded = tokenizer_fast.encode_plus(
            prepended_text,
            add_special_tokens=True,
            max_length=20340,
            truncation=True,
            return_attention_mask=False,
            return_token_type_ids=False
        )["input_ids"]

        max_len = 512
        stride = 256

        chunks = []
        for i in range(0, len(encoded), stride):
            window = encoded[i:i+max_len]
            if len(window) < 10:
                continue
            chunks.append(window)

        scores = []
        for input_ids in chunks:
            input_tensor = torch.tensor([input_ids])
            attention_mask = torch.ones_like(input_tensor)
            with torch.no_grad():
                output = model(input_ids=input_tensor, attention_mask=attention_mask)
                score = output.item()
                scores.append(score)

        raw_mean = np.mean(scores)
        rescaled_score = (raw_mean + 1.5) / 3.0 * 100
        final_score = float(np.clip(rescaled_score, 0, 100))
        scores_run.append(round(final_score, 2))

    avg_score = round(np.mean(scores_run), 2)
    score_volatility = round(np.std(scores_run), 2)
    score_min = round(min(scores_run), 2)
    score_max = round(max(scores_run), 2)
    score_median = round(np.median(scores_run), 2)
    q75, q25 = np.percentile(scores_run, [75, 25])
    score_iqr = round(q75 - q25, 2)
    from scipy.stats import skew
    score_skew = round(skew(scores_run), 2)
    match = re.search(r"\d{4}-[A-Za-z]{3}-\d{2}", filename)
    short_name = match.group(0) if match else filename
    results.append([
        short_name, avg_score, score_volatility, score_min, score_max,
        score_median, score_iqr, score_skew
    ] + scores_run)

# Output as DataFrame
df = pd.DataFrame(
    results,
    columns=["File", "Avg. Estimated Score", "Score Volatility", "Score Min", "Score Max",
             "Score Median", "Score IQR", "Score Skewness"] +
            [f"Est. Score {i+1}" for i in range(10)]
)
print(df)
output_path = "/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/FinBERT Output/finbert_step2a_output.xlsx"
df.to_excel(output_path, index=False)
print(f"Output saved to {output_path}")