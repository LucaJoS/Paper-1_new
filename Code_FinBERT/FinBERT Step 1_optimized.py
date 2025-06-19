import os
import re
import torch
import numpy as np
import pandas as pd
import warnings
from transformers import BertTokenizerFast
from FinBERT_stable_regressor import load_stable_regressor

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Constants
INPUT_DIR = r"/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/aaa Clean Slate/a_Input_Reports_preprocessed_txt/A"
OUTPUT_PATH = r"/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/aaa Clean Slate/Output_FinBERT/finbert_step1_output.xlsx"
MAX_LEN = 512
STRIDE = 256

# Load model and tokenizer
model, tokenizer = load_stable_regressor()

results = []
files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")]
for idx, filename in enumerate(files, 1):
    print(f"Processing file {idx}/{len(files)}: {filename}")
    with open(os.path.join(INPUT_DIR, filename), "r", encoding="utf-8") as f:
        text = f.read()

    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=20340,
        truncation=True,
        return_attention_mask=False,
        return_token_type_ids=False
    )["input_ids"]

    chunks = [encoded[i:i + MAX_LEN] for i in range(0, len(encoded), STRIDE) if len(encoded[i:i + MAX_LEN]) >= 10]

    with torch.no_grad():
        scores = [model(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids))).item() for ids in chunks]

    raw_mean = np.mean(scores)
    raw_score_check = raw_mean
    # Rescale raw_mean from range [-1, 1] to [0, 100] for interpretability in ESG scoring
    rescaled_score = (raw_mean + 1.0) / 2.0 * 100
    avg_score = float(np.clip(rescaled_score, 0, 100))

    short_name = re.search(r"\d{4}-[A-Za-z]{3}-\d{2}", filename)
    results.append([short_name.group(0) if short_name else filename, avg_score, raw_score_check])

df = pd.DataFrame(results, columns=["File", "Avg. Estimated Score", "Raw Score (Unscaled)"])
df.to_excel(OUTPUT_PATH, index=False)
print(df)
print(f"Output saved to {OUTPUT_PATH}")
