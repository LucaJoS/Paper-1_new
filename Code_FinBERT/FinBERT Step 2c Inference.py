import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizerFast, BertForMaskedLM
from peft import PeftModel, PeftConfig
from tqdm import tqdm

# Suppress warning about uninitialized weights in BertForMaskedLM; the regression head is trained separately and the MLM head is not used.
import warnings
warnings.filterwarnings("ignore", message="Some weights of BertForMaskedLM were not initialized from the model checkpoint.*")

# Suppress repeated warning about uninitialized MLM head weights
# Explanation: BertForMaskedLM initializes a decoder head ('cls.predictions.*') which we do not use;
# our scoring model uses only the pooled CLS embedding + regression head, so these warnings are irrelevant.
transformers_logging_enabled = False
try:
    import logging
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    transformers_logging_enabled = True
except Exception:
    pass

# Load tokenizer and base model
model_path = "/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/aaa Clean Slate/finbert_step2c_lora"
base_model = BertForMaskedLM.from_pretrained("ProsusAI/finbert")
tokenizer = BertTokenizerFast.from_pretrained(model_path)

class FinBERTWithRegressor(nn.Module):
    def __init__(self, model, hidden_size=768):
        super().__init__()
        self.model = model
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.model.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        score = self.regressor(cls_embedding)
        return score

# Load PEFT adapter config and attach to base model
peft_config = PeftConfig.from_pretrained(model_path)
peft_model = PeftModel.from_pretrained(base_model, model_path)
model = FinBERTWithRegressor(peft_model)
model.eval()

# Input directory and output file
input_dir = "/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/aaa Clean Slate/a_Input_Reports_preprocessed_txt/A"
output_file = "/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/aaa Clean Slate/Output_FinBERT/finbert_step2c_output.xlsx"

# Prepare to collect results
results = []
all_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
total_files = len(all_files)

# Inference loop
for idx, filename in enumerate(tqdm(all_files, desc="Files"), start=1):
    print(f"Processing file {idx}/{total_files}: {filename}")
    with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f:
        text = f.read()

    prepended_text = (
        "ESG SCORE SYSTEM (0–100): 0 is bad, 100 is excellent. "
        "Score as an ESG Analyst\n\n" + text
    )

    encoded = tokenizer.encode_plus(
        prepended_text,
        add_special_tokens=True,
        max_length=20480,
        truncation=True,
        return_attention_mask=False,
        return_token_type_ids=False
    )["input_ids"]

    max_len = 512
    stride = 256

    chunks = [
        encoded[i:i+max_len]
        for i in range(0, len(encoded), stride)
        if len(encoded[i:i+max_len]) >= 10
    ]

    scores = []
    for input_ids in chunks:
        input_tensor = torch.tensor([input_ids])
        attention_mask = torch.ones_like(input_tensor)
        with torch.no_grad():
            score = model(input_tensor, attention_mask=attention_mask).squeeze()
            scores.append(score.item())

    raw_mean = np.mean(scores)
    final_score = float(np.clip((raw_mean + 1.0) / 2.0 * 100, 0, 100))

    stats = {
        "File": re.search(r"\d{4}-[A-Za-z]{3}-\d{2}", filename).group(0) if re.search(r"\d{4}-[A-Za-z]{3}-\d{2}", filename) else filename,
        "Avg. Estimated Score": final_score,
        "Raw Score (Unscaled)": raw_mean
    }

    results.append(stats)

# Save output
df = pd.DataFrame(results, columns=["File", "Avg. Estimated Score", "Raw Score (Unscaled)"])
df.to_excel(output_file, index=False)
print("✅ Output saved to:", output_file)
print(df)
