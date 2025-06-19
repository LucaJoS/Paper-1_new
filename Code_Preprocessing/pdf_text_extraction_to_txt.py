#!/usr/bin/env python3
import os
import pdfplumber
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import shutil
import warnings
# Suppress pdfplumber CropBox missing warnings
warnings.filterwarnings(
    "ignore",
    message="CropBox missing from /Page, defaulting to MediaBox"
)
import sys
import contextlib
import pandas as pd
import logging
# Configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
logging.disable(logging.WARNING)

def extract_text_from_pdf(path):
    """
    Extract text from a PDF file using pdfplumber with a PyMuPDF fallback.
    Suppresses CropBox missing warnings printed to stderr by pdfplumber/PyMuPDF.
    """
    try:
        with open(os.devnull, 'w') as fnull, contextlib.redirect_stderr(fnull):
            # Primary extraction with pdfplumber
            with pdfplumber.open(path) as pdf:
                full_text = ''.join(page.extract_text() or '' for page in pdf.pages)
                if full_text.strip():
                    return full_text
            # Fallback extraction with PyMuPDF
            doc = fitz.open(path)
            return ''.join(page.get_text() for page in doc)
    except Exception as e:
        print(f"[ERROR] Failed to extract {path}: {e}", file=sys.stderr)
        return None

def extract_text_from_all_pdfs(base_dir, output_dir="clean_extracted_texts"):
    """
    Traverse base_dir, extract text from each PDF into a mirror folder structure
    under output_dir. Show only progress messages during processing.
    At the end, print errors, skipped files, and a summary table of stats.
    """
    # Prepare output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Find all company folders
    tickers = [
        d for d in os.listdir(base_dir)
        if (
            not d.startswith('.')
            and os.path.isdir(os.path.join(base_dir, d))
            and any(
                f.lower().endswith('.pdf')
                for f in os.listdir(os.path.join(base_dir, d))
            )
        )
    ]
    total_tickers = len(tickers)

    # Initialize track lists
    stats = []
    errors = []
    skipped = []

    # Process each ticker
    for idx_ticker, ticker in enumerate(tickers, start=1):
        ticker_path = os.path.join(base_dir, ticker)
        # Gather PDFs for this ticker
        pdf_files = sorted(
            f for f in os.listdir(ticker_path)
            if not f.startswith('.') and f.lower().endswith(".pdf")
        )
        total_files = len(pdf_files)
        # Create corresponding output folder
        ticker_output_dir = os.path.join(output_dir, ticker)
        os.makedirs(ticker_output_dir, exist_ok=True)

        # Process files with progress messages
        for idx_file, filename in enumerate(pdf_files, start=1):
            # Progress message
            print(f"Company {idx_ticker} of {total_tickers}. File {idx_file} of {total_files} for Company {ticker}.")
            pdf_path = os.path.join(ticker_path, filename)
            base_name = os.path.splitext(filename)[0]
            txt_path = os.path.join(ticker_output_dir, base_name + ".txt")

            # Idempotency: skip existing
            if os.path.exists(txt_path):
                skipped.append(txt_path)
                stats.append({
                    "filename": filename,
                    "ticker": ticker,
                    "success": True,
                    "ocr_used": False,
                    "num_chars": len(open(txt_path, "r", encoding="utf-8").read())
                })
                continue

            # Attempt extraction
            text = extract_text_from_pdf(pdf_path)
            if text:
                try:
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(text)
                    stats.append({
                        "filename": filename,
                        "ticker": ticker,
                        "success": True,
                        "ocr_used": False,
                        "num_chars": len(text)
                    })
                except Exception as e:
                    errors.append(f"Failed writing {txt_path}: {e}")
                    stats.append({
                        "filename": filename,
                        "ticker": ticker,
                        "success": False,
                        "ocr_used": False,
                        "num_chars": 0
                    })
            else:
                errors.append(f"No text extracted from {pdf_path}")
                stats.append({
                    "filename": filename,
                    "ticker": ticker,
                    "success": False,
                    "ocr_used": False,
                    "num_chars": 0
                })

    # After all done, print errors and skipped at end
    if errors:
        print("\nErrors encountered:")
        for err in errors:
            print("  -", err)

    if skipped:
        print(f"\nSkipped {len(skipped)} existing files:")
        for path in skipped:
            print("  -", path)

    # Print summary table
    df_stats = pd.DataFrame(stats)
    print("\nExtraction summary:")
    print(df_stats.to_string(index=False))

    # Save metrics to 'output metrics' subfolder
    metrics_dir = os.path.join(output_dir, "output metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # Save CSV
    df_stats.to_csv(os.path.join(metrics_dir, "extraction_stats.csv"), index=False)

if __name__ == "__main__":
    base_dir = "/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/aaa Clean Slate/reports"
    extract_text_from_all_pdfs(base_dir, output_dir="/Users/lucaschmidt/Desktop/PhD_Rasa/My Papers/aaa Clean Slate/reports_txt")