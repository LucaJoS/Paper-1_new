import pandas as pd
import torch
from FinBERT_stable_regressor import load_stable_regressor
import numpy as np


# Load ESG metadata using the full ESG metadata file
data = pd.read_parquet('/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/Code - Paper 1/Refinitiv Dataset ESG S&P500/Matched_Report_ESG_Metadata.parquet')
data['year'] = pd.to_datetime(data['filing_date'], errors='coerce').dt.year

# Define ESG rubric (example placeholder, replace with actual rubric)
esg_rubric = (
    "ESG Rubric:\n"
    "- Environmental: Consider carbon emissions, energy usage, waste management.\n"
    "- Social: Consider labor practices, community engagement, human rights.\n"
    "- Governance: Consider board structure, ethics, transparency.\n"
)

model, tokenizer = load_stable_regressor()

# Function to build prompt with history
def build_prompt_with_history(row, error_history):
    prompt = esg_rubric + "\n\n"
    prompt += "Historical Errors:\n"
    if error_history:
        for year, error in error_history:
            prompt += f"Year {year}: Residual = {error:.4f}\n"
    else:
        prompt += "None\n"
    prompt += f"\nFirm: {row['ticker']}\nFiling Date: {row['filing_date']}\n\n"
    # prompt += "Disclosure Text:\n" + row['disclosure_text']
    return prompt

# Sort data chronologically by firm and year
data = data.sort_values(by=['ticker', 'year']).reset_index(drop=True)

data = data.groupby(['ticker', 'year']).head(1).reset_index(drop=True)

# Define chunking parameters (same as Step 5a)
max_length = 512
stride = 256

results = []
firm_history = {}
total_files = len(data)
current_idx = 0

for firm in sorted(data['ticker'].unique()):
    firm_data = data[data['ticker'] == firm].sort_values('year')
    firm_history[firm] = []

    for _, row in firm_data.iterrows():
        current_idx += 1
        print(f"Processing file {current_idx}/{total_files} - {row['ticker']} - {row['year']}")
        year = row['year']
        error_history = firm_history[firm].copy()

        # Build prompt with history
        try:
            with open(row['file_path'], 'r', encoding='utf-8') as f:
                disclosure_text = f.read()
        except Exception as e:
            disclosure_text = "[ERROR: Failed to read disclosure text]"

        prompt = build_prompt_with_history(row, error_history)
        truncated_prompt = tokenizer.decode(
            tokenizer(prompt, max_length=256, truncation=True)["input_ids"]
        )
        scoring_instruction = (
            "ESG SCORE SYSTEM (0–100): 0 is bad, 100 is excellent. "
            "Score as an ESG Analyst.\n\n"
        )
        full_input = scoring_instruction + truncated_prompt + "\n\n" + disclosure_text

        # Tokenize and chunk prompt
        inputs = tokenizer(full_input, return_tensors='pt', truncation=False)
        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]
        input_len = input_ids.size(0)

        chunks = []
        start = 0
        while start < input_len:
            end = min(start + max_length, input_len)
            chunk_input_ids = input_ids[start:end]
            chunk_attention_mask = attention_mask[start:end]
            chunks.append((chunk_input_ids.unsqueeze(0), chunk_attention_mask.unsqueeze(0)))
            if end == input_len:
                break
            start += max_length - stride

        # Run FinBERT prediction per chunk, average raw outputs and rescale
        scores = []
        with torch.no_grad():
            for chunk_input_ids, chunk_attention_mask in chunks:
                output = model(chunk_input_ids, attention_mask=chunk_attention_mask)
                scores.append(output.item())

        raw_mean = np.mean(scores)
        rescaled_score = (raw_mean + 1.5) / 3.0 * 100
        predicted_avg = float(np.clip(rescaled_score, 0, 100))

        # Calculate residual
        refinitiv_score = row['esg_score']
        residual = predicted_avg - refinitiv_score

        # Store for next year's prompt
        firm_history[firm].append((year, residual))

        # Save results
        results.append({
            'firm': firm,
            'year': year,
            'filing_date': row['filing_date'],
            'predicted_avg': predicted_avg,
            'refinitiv_esg_score': refinitiv_score,
            'residual': residual,
            'error_history': error_history,
            'disclosure_text': disclosure_text
        })

# Convert results to DataFrame and save to Excel
results_df = pd.DataFrame(results)

results_df.to_excel('/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/FinBERT Output/finbert_step6a_output.xlsx', index=False)
print("✅ Saved to /Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/FinBERT Output/finbert_step6a_output.xlsx")
