import os
import pandas as pd

# Paths
metadata_path = "/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/Code - Paper 1/Refinitiv Dataset ESG S&P500/Matched_Report_ESG_Metadata.parquet"
output_path = "/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/FinBERT Output/FinBERT Step XXX Database Builder.xlsx"

# Load metadata with ESG scores
df = pd.read_parquet(metadata_path)

# Prompt header generation function (metadata only, no disclosure text)
def build_prompt_header(row):
    return (
        f"Company: {row['ticker']}, Filing Date: {row['filing_date'].strftime('%Y-%m-%d')}\n"
        f"Refinitiv ESG Scores:\n"
        f"- ESG Score: {row.get('esg_score', 'N/A')}\n"
        f"- ESG Controversy Score: {row.get('esg_controversy_score', 'N/A')}\n"
        f"- ESG Combined Score (ESGC): {row.get('esgc_score', 'N/A')}\n"
        f"---"
    )

# Construct prompt headers only (text used later during execution)
headers = []
for i, row in df.iterrows():
    try:
        header = build_prompt_header(row)
        headers.append([row["file_path"], header])
    except Exception as e:
        print(f"⚠️ Failed to process metadata: {row['file_path']} — {str(e)}")

# Save to Excel
header_df = pd.DataFrame(headers, columns=["file_path", "prompt_header"])
header_df.to_excel(output_path, index=False)
print(f"✅ Saved ESG-score headers to: {output_path}")

# Prevent macOS from going to sleep during long execution
try:
    os.system("caffeinate -dimsu &")
except Exception as e:
    print(f"⚠️ Unable to enable sleep prevention: {e}")