import os
import json
import requests
import pandas as pd

# Load Gemini API key from external file
with open("/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/Code - Paper 1/Gemini_API_Key.txt", "r") as f:
    api_key = f.read().strip()

# Define endpoint for Gemini 1.5 Flash (cheap + fast)
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

# Directory with report text files
input_dir = "/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/Code - Paper 1/1_extract_text_from_pdfs/clean_extracted_texts/aaaTesting - only 3 reports"
output_path = "/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/Gemini Output/gemini_step1_output.xlsx"

# System prompt
system_prompt = (
    "You are a professional ESG analyst.\n"
    "Read the following text and estimate an ESG score from 0 (worst) to 100 (best).\n"
    "Reply with a single number. No explanations. Just a number."
)

results = []
files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
total_files = len(files)

for idx, filename in enumerate(files, 1):
    print(f"Processing file {idx}/{total_files}: {filename}")
    with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f:
        content = f.read()

    # Construct Gemini payload
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": system_prompt + "\n\n" + content}]
            }
        ]
    }

    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )

    if response.status_code == 200:
        result = response.json()
        try:
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            numeric = float(text.strip().replace("%", ""))
            score = round(min(max(numeric, 0), 100), 2)
        except Exception as e:
            score = None
            print(f"⚠️ Could not parse score from: {text}")
    else:
        score = None
        print(f"❌ Request failed for {filename} — {response.status_code}")
        print(response.text)

    results.append([filename, score])

# Save results and print to terminal
df = pd.DataFrame(results, columns=["File", "Gemini ESG Score"])
print("\nFinal Results:\n")
print(df)
df.to_excel(output_path, index=False)
print(f"\n✅ Output saved to: {output_path}")