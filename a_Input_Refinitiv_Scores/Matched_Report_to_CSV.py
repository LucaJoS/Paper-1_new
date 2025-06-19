import pandas as pd

# Load Parquet
parquet_path = "/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/aaa Clean Slate/a_Input_Refinitiv_Scores/Output/Matched_Report_ESG_Metadata.parquet"
df = pd.read_parquet(parquet_path)

# Export to CSV
csv_path = parquet_path.replace(".parquet", ".csv")
df.to_csv(csv_path, index=False)

print(f"✅ Exported CSV to: {csv_path}")