import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizerFast, BertModel, BertForMaskedLM
from peft import PeftModel

import logging
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

# Suppress PEFT/transformers irrelevant sharding/layer loading warnings:
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").propagate = False
logging.getLogger("transformers.modeling_tf_utils").setLevel(logging.ERROR)
logging.getLogger("peft.utils.other").setLevel(logging.ERROR)

# Explanation:
# These warnings refer to missing weight sharding or decoder layer mapping in masked language models.
# They are triggered during adapter/LoRA loading due to internal HuggingFace design for distributed setups.
# Since we are not using sharding or generation (no causal LM), these messages are safe to ignore and suppressed for clarity.

# Paths
model_path = "/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/Code - Paper 1/finbert_step3c_lora"
input_dir = "/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/Code - Paper 1/1_extract_text_from_pdfs/clean_extracted_texts/aaaTesting - only 3 reports"
output_file = "/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/FinBERT Output/finbert_step3c_output.xlsx"

# Load tokenizer and base model
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
        cls_output = outputs.hidden_states[-1][:, 0, :]  # Use last hidden layer's CLS token
        return self.regressor(cls_output).squeeze(-1)


# Run scoring
results = []
files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]

for idx, filename in enumerate(files, 1):
    print(f"Processing file {idx}/{len(files)}: {filename}")
    with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f:
        text = f.read()

    scores_run = []
    for _ in range(10):
        model = FinBERTRegressor(base_model)  # reinstantiate each time to inject fresh randomness
        model.eval()
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=20480,
            truncation=True,
            return_attention_mask=False
        )["input_ids"]

        max_len, stride = 512, 256
        chunks = [encoded[i:i+max_len] for i in range(0, len(encoded), stride) if len(encoded[i:i+max_len]) >= 10]

        chunk_scores = []
        for input_ids in chunks:
            input_tensor = torch.tensor([input_ids])
            attention_mask = torch.ones_like(input_tensor)
            with torch.no_grad():
                score = model(input_tensor, attention_mask=attention_mask)
                chunk_scores.append(score.item())

        raw_mean = np.mean(chunk_scores)
        rescaled = (raw_mean + 1.5) / 3.0 * 100
        scores_run.append(round(float(np.clip(rescaled, 0, 100)), 2))

    # Aggregate metrics
    avg_score = round(np.mean(scores_run), 2)
    std_dev = round(np.std(scores_run), 2)
    score_min = round(np.min(scores_run), 2)
    score_max = round(np.max(scores_run), 2)
    iqr = round(np.percentile(scores_run, 75) - np.percentile(scores_run, 25), 2)
    median = round(np.median(scores_run), 2)
    skew = round(((avg_score - median) / (std_dev + 1e-6)), 2)

    match = re.search(r"\d{4}-[A-Za-z]{3}-\d{2}", filename)
    short_name = match.group(0) if match else filename

    results.append(
        [short_name, avg_score, std_dev, score_min, score_max, iqr, median, skew] + scores_run
    )

# Save to Excel
columns = [
    "File", "Avg. Estimated Score", "Score Volatility", "Score Min", "Score Max",
    "Score IQR", "Score Median", "Score Skew"
] + [f"Est. Score {i+1}" for i in range(10)]

df = pd.DataFrame(results, columns=columns)
df.to_excel(output_file, index=False)
print(f"✅ Output saved to: {output_file}")
print(df)