import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn

from FinBERT_stable_regressor import load_stable_regressor

# Do not fix model here – it will be instantiated per iteration

import os
from transformers import BertTokenizerFast
import numpy as np

model, tokenizer_fast = load_stable_regressor()

import re
import pandas as pd

results = []

input_dir = r"/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/aaa Clean Slate/a_Input_Reports_preprocessed_txt/A"
all_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
total_files = len(all_files)
for idx, filename in enumerate(all_files, start=1):
    print(f"Processing file {idx}/{total_files}: {filename}")
    with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as file:
        text = file.read()

    prepended_text = text

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
    raw_score_check = raw_mean
    rescaled_score = (raw_mean + 1.5) / 3.0 * 100
    avg_score = float(np.clip(rescaled_score, 0, 100))

    # score_volatility = 0.0

    # Additional statistics
    # score_min = avg_score
    # score_max = avg_score
    # score_median = avg_score
    # score_iqr = 0.0
    # score_skew = 0.0

    match = re.search(r"\d{4}-[A-Za-z]{3}-\d{2}", filename)
    short_name = match.group(0) if match else filename

    results.append([
        short_name, avg_score, raw_score_check
        # , score_volatility, score_min, score_max,
        # score_median, score_iqr, score_skew
    ])

df = pd.DataFrame(
    results,
    columns=["File", "Avg. Estimated Score", "Raw Score (Unscaled)"]
    # + ["Score Volatility", "Score Min", "Score Max",
    #    "Score Median", "Score IQR", "Score Skewness"]
)
print(df)
output_path = r"/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/FinBERT Output/finbert_step1_output.xlsx"
df.to_excel(output_path, index=False)
print(f"Output saved to {output_path}")
