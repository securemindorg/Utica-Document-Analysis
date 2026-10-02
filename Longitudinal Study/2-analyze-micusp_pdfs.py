import os
import glob
import numpy as np
import pandas as pd
from scipy import stats
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from pypdf import PdfReader

# Ensure NLTK tokenizer models are available
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)



def calculate_mattr(tokens, window_size=100):
    """Calculates Moving Average Type-Token Ratio (MATTR)."""
    if len(tokens) < window_size:
        return len(set(tokens)) / len(tokens) if tokens else 0.0
    ttrs = [
        len(set(tokens[i : i + window_size])) / float(window_size)
        for i in range(len(tokens) - window_size + 1)
    ]
    return float(np.mean(ttrs))

def calculate_zipf_r2(tokens):
    """Calculates Zipf's Law R^2 score for log-rank vs log-frequency linear fit."""
    alpha_tokens = [t.lower() for t in tokens if t.isalpha()]
    if not alpha_tokens:
        return 0.0

    freq_dist = nltk.FreqDist(alpha_tokens)
    frequencies = sorted(freq_dist.values(), reverse=True)
    if len(frequencies) < 5:
        return 0.0

    ranks = np.arange(1, len(frequencies) + 1)
    log_ranks = np.log(ranks)
    log_freqs = np.log(frequencies)

    _, _, r_value, _, _ = stats.linregress(log_ranks, log_freqs)
    return float(r_value ** 2)

def extract_features_from_text(text):
    """Extracts baseline metrics from plain text."""
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    alpha_words = [w.lower() for w in words if w.isalpha()]

    # 1. Sentence Variance
    sent_lengths = [len(word_tokenize(s)) for s in sentences]
    sentence_variance = float(np.var(sent_lengths)) if len(sent_lengths) > 1 else 0.0

    # 2. MATTR & Zipf R^2
    mattr = calculate_mattr(alpha_words, window_size=100)
    zipf_r2 = calculate_zipf_r2(words)

    # 3. Readability: Flesch Reading Ease
    total_words = len(alpha_words)
    total_sents = max(len(sentences), 1)
    total_syllables = sum([max(1, len([c for c in w if c in "aeiouy"])) for w in alpha_words])

    if total_words > 0:
        flesch_score = 206.835 - (1.015 * (total_words / total_sents)) - (84.6 * (total_syllables / total_words))
    else:
        flesch_score = 0.0

    return {
        "sentence_variance": sentence_variance,
        "mattr": mattr,
        "zipf_r2": zipf_r2,
        "flesch_reading_ease": flesch_score,
        "word_count": total_words
    }

def process_micusp_pdf_dir(pdf_folder, csv_metadata_path):
    """Reads all PDF files, computes features, and merges with metadata."""
    metadata_df = pd.read_csv(csv_metadata_path)
    records = []

    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    print(f"Found {len(pdf_files)} PDF files in '{pdf_folder}'. Processing...")

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        paper_id = os.path.splitext(filename)[0]  # E.g., 'BIO.G1.04.1'

        try:
            reader = PdfReader(pdf_path)
            extracted_text = " ".join([page.extract_text() or "" for page in reader.pages])

            if len(extracted_text.strip()) < 50:
                print(f"Warning: Minimal or no text extracted from {filename}")
                continue

            metrics = extract_features_from_text(extracted_text)
            metrics["PAPER ID"] = paper_id
            records.append(metrics)

        except Exception as e:
            print(f"Error parsing {filename}: {e}")

    metrics_df = pd.DataFrame(records)

    # Merge extracted features back with your CSV metadata
    full_df = pd.merge(metadata_df, metrics_df, on="PAPER ID", how="inner")
    print(f"\nSuccessfully processed {len(full_df)} papers!")
    return full_df

# --- RUN PROCESSING & BASELINE STATS ---
if __name__ == "__main__":
    # 1. Process local PDF files and merge with CSV
    micusp_df = process_micusp_pdf_dir("./micusp_pdfs", "micusp_papers.csv")
    micusp_df.to_csv("micusp_extracted_baseline.csv", index=False)

    # 2. Filter for Graduate Papers only and print baseline statistics
    grad_df = micusp_df[micusp_df["STUDENT LEVEL"].str.contains("Graduate", na=False)]

    baseline_stats = grad_df[["sentence_variance", "mattr", "zipf_r2", "flesch_reading_ease"]].agg(["mean", "std", "median"]).T
    print("\n=== PRE-2024 GRADUATE BASELINE NORMS ===")
    print(baseline_stats)
