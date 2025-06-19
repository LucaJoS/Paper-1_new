import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
import re
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizerFast, BertModel
import torch.nn as nn

# Absolute model path
model_path = "/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/aaa Clean Slate/finbert_step2b_mlm"

# Load tuned model and tokenizer
tokenizer_fast = BertTokenizerFast.from_pretrained(model_path)
base_model = BertModel.from_pretrained(model_path)

# Regression wrapper
class FinBERTRegressor(nn.Module):
    def __init__(self, base_model):
        super(FinBERTRegressor, self).__init__()
        self.bert = base_model
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output
        return self.regressor(cls_output).squeeze(-1)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load regressor
model = FinBERTRegressor(base_model)
model.to(device)
model.eval()

# Input directory
input_dir = "/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/aaa Clean Slate/a_Input_Reports_preprocessed_txt/A"
results = []

all_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
total_files = len(all_files)

for idx, filename in enumerate(all_files, start=1):
    print(f"Processing file {idx}/{total_files}: {filename}")
    with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as file:
        text = file.read()

    input_text = text

    encoded = tokenizer_fast.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=20340,
        truncation=True,
        return_attention_mask=False,
        return_token_type_ids=False
    )["input_ids"]

    max_len = 512
    stride = 256
    chunks = [encoded[i:i+max_len] for i in range(0, len(encoded), stride) if len(encoded[i:i+max_len]) >= 10]

    chunk_scores = []
    for input_ids in chunks:
        input_tensor = torch.tensor([input_ids]).to(device)
        attention_mask = torch.ones_like(input_tensor).to(device)
        with torch.no_grad():
            score = model(input_tensor, attention_mask).item()
            chunk_scores.append(score)

    raw_mean = np.mean(chunk_scores)
    final_score = float(np.clip((raw_mean + 1.0) / 2.0 * 100, 0, 100))

    short_name = re.search(r"\d{4}-[A-Za-z]{3}-\d{2}", filename)
    short_name = short_name.group(0) if short_name else filename

    results.append([short_name, final_score, raw_mean])

# Save DataFrame to Excel
columns = ["File", "Avg. Estimated Score", "Raw Score (Unscaled)"]

df = pd.DataFrame(results, columns=columns)
print(df)

output_path = "/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/aaa Clean Slate/Output_FinBERT/finbert_step2b_output.xlsx"
df.to_excel(output_path, index=False)
print(f"✅ Output saved to {output_path}")