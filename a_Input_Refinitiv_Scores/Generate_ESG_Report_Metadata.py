import os
import re
import pandas as pd
from datetime import datetime

# --- Paths ---
report_dir = "/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/aaa Clean Slate/a_Input_Reports_preprocessed_txt"
esg_scores_path = "/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/aaa Clean Slate/a_Input_Refinitiv_Scores/Input/Cleaned_SP500_ESG_Scores_10Y.parquet"
output_path = "/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/aaa Clean Slate/a_Input_Refinitiv_Scores/Output/Matched_Report_ESG_Metadata.parquet"

# --- Load Refinitiv scores ---
df_scores = pd.read_parquet(esg_scores_path)
df_scores["Date"] = pd.to_datetime(df_scores["Date"], errors="coerce")
df_scores = df_scores.dropna(subset=["Date", "Instrument"])

# --- Prepare results ---
metadata = []

# --- Extract and match ---
for root, dirs, files in os.walk(report_dir):
    for fname in files:
        if not fname.endswith(".txt"):
            continue

        match = re.search(r"Filing Date - ([\d]{4}-[A-Za-z]{3}-[\d]{2}) - ([^-]+) -", fname)
        if not match:
            continue

        filing_str, ticker = match.groups()
        filing_date = pd.to_datetime(filing_str, format="%Y-%b-%d", errors="coerce")

        if pd.isna(filing_date):
            continue

        subset = df_scores[df_scores["Instrument"] == ticker]
        if not subset.empty and subset[subset["Date"] <= filing_date].empty:
            print(f"⚠️ Ticker matched but no date found on or before filing date: {ticker}, Filing Date: {filing_date.date()}, Available Dates: {subset['Date'].dt.date.unique()}")
        if subset.empty:
            continue

        # Nearest available report date (on or before)
        nearest = subset[subset["Date"] <= filing_date].sort_values("Date", ascending=False).head(1)

        if nearest.empty:
            continue

        row = nearest.iloc[0]
        metadata.append({
            "file_path": os.path.join(root, fname),
            "file_name": fname,
            "filing_date": filing_date,
            "ticker": ticker,
            "esg_score": row.get("ESG_Score", None),
            "esg_controversy_score": row.get("ESG_Controversies_Score", None),
            "esgc_score": row.get("ESGC_Score", None),
            "matched_score_date": row["Date"]
        })

# --- Save result ---
df_out = pd.DataFrame(metadata)
df_out.to_parquet(output_path, index=False)
print(f"✅ Saved matched ESG report metadata to: {output_path}")
print(df_out)