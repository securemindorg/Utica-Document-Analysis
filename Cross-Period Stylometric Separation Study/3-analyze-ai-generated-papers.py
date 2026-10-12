import os
import re
import math
import glob
import sys
import logging
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from docx import Document
from pypdf import PdfReader

import matplotlib.pyplot as plt
import seaborn as sns

# Suppress pypdf warning logs
logging.getLogger("pypdf").setLevel(logging.ERROR)

# Ensure required NLTK tokenizers are available
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Set publication-quality plot style
plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

# ==========================================
# UNIFIED STYLOMETRIC EXTRACTOR ENGINE
# ==========================================
class UnifiedStylometricExtractor:
    """
    Unified feature extraction engine.
    Applies identical NLTK-based tokenization, MATTR (W=100), Zipf R^2,
    and readability metrics across corpora.
    """
    def __init__(self, mattr_window=100):
        self.mattr_window = mattr_window

    def calculate_mattr(self, alpha_tokens):
        if len(alpha_tokens) < self.mattr_window:
            return float(len(set(alpha_tokens)) / len(alpha_tokens)) if alpha_tokens else 0.0
        ttrs = [
            len(set(alpha_tokens[i : i + self.mattr_window])) / float(self.mattr_window)
            for i in range(len(alpha_tokens) - self.mattr_window + 1)
        ]
        return float(np.mean(ttrs))

    def calculate_zipf_r2(self, alpha_tokens):
        if len(alpha_tokens) < 5:
            return 0.0
        freq_dist = Counter(alpha_tokens)
        frequencies = sorted(freq_dist.values(), reverse=True)
        if len(frequencies) < 5:
            return 0.0

        ranks = np.arange(1, len(frequencies) + 1)
        log_ranks = np.log(ranks)
        log_freqs = np.log(frequencies)

        _, _, r_value, _, _ = stats.linregress(log_ranks, log_freqs)
        return float(r_value ** 2)

    def calculate_passive_voice(self, sentences):
        if not sentences:
            return 0.0
        passive_regex = re.compile(
            r'\b(am|is|are|was|were|be|been|being)\b\s+([\w]+ed|conducted|shown|given|made|found|taken|seen|built|chosen|driven|written|known|done)\b',
            re.I
        )
        count = sum(1 for s in sentences if passive_regex.search(s))
        return round((count / len(sentences)) * 100.0, 1)

    def count_syllables(self, word):
        word = re.sub(r'[^a-z]', '', word.lower())
        if len(word) <= 3:
            return 1
        word = re.sub(r'(?:[^laeiouy]es|ed|[^laeiouy]e)$', '', word)
        word = re.sub(r'^y', '', word)
        matches = re.findall(r'[aeiouy]{1,2}', word)
        return len(matches) if matches else 1

    def analyze_text(self, raw_text):
        if not raw_text or len(raw_text.strip()) < 50:
            return None

        # Stripping LaTeX math, code snippets, and structural artifacts[cite: 3]
        clean_text = re.sub(r'```.*?```', '', raw_text, flags=re.S)
        clean_text = re.sub(r'\$.*?\$', '', clean_text)
        clean_text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL_REDACTED]', clean_text)

        sentences = sent_tokenize(clean_text)
        all_tokens = word_tokenize(clean_text)
        alpha_words = [w.lower() for w in all_tokens if w.isalpha()]

        total_words = len(alpha_words)
        total_sentences = len(sentences)

        if total_words < 30 or total_sentences < 2:
            return None

        sent_lengths = [len([w for w in word_tokenize(s) if w.isalpha()]) for s in sentences]
        sentence_variance = float(np.var(sent_lengths)) if len(sent_lengths) > 1 else 0.0
        sentence_variance = min(1000.0, sentence_variance)

        mattr = self.calculate_mattr(alpha_words)
        zipf_fit = self.calculate_zipf_r2(alpha_words)

        char_counts = Counter(clean_text)
        total_chars = len(clean_text)
        entropy = -sum((cnt / total_chars) * math.log2(cnt / total_chars) for cnt in char_counts.values()) if total_chars > 0 else 0.0

        passive_ratio = self.calculate_passive_voice(sentences)

        total_syllables = sum(self.count_syllables(w) for w in alpha_words)
        flesch_ease = max(0.0, min(100.0, 206.835 - (1.015 * (total_words / total_sentences)) - (84.6 * (total_syllables / total_words))))
        grade_level = max(0.0, (0.39 * (total_words / total_sentences)) + (11.8 * (total_syllables / total_words)) - 15.59)

        nominalizations = [w for w in alpha_words if (w.endswith('tion') or w.endswith('ment') or w.endswith('ance') or w.endswith('ence')) and len(w) > 5]
        nom_ratio = (len(nominalizations) / total_words) * 100.0

        return {
            'Word_Count': total_words,
            'Sentence_Count': total_sentences,
            'Entropy': round(entropy, 3),
            'MATTR': round(mattr, 4),
            'Zipf_R2': round(zipf_fit, 4),
            'Sentence_Var': round(sentence_variance, 2),
            'Flesch_Ease': round(flesch_ease, 1),
            'Grade_Level': round(grade_level, 1),
            'Nominal_Ratio_%': round(nom_ratio, 1),
            'Passive_%': round(passive_ratio, 1)
        }

# ==========================================
# FILE PARSERS & DUAL LOGGER
# ==========================================
class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

def extract_text_from_filepath(filepath):
    ext = filepath.split('.')[-1].lower()
    try:
        if ext == 'txt':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        elif ext == 'docx':
            doc = Document(filepath)
            return "\n".join([p.text for p in doc.paragraphs])
        elif ext == 'pdf':
            reader = PdfReader(filepath)
            return "\n".join([page.extract_text() or '' for page in reader.pages])
    except Exception:
        return ""
    return ""

def process_ai_generated_directory(root_dir, extractor):
    """
    Scans AI-Generated-Papers/ subdirectories, using folder names as Model Family tags.
    """
    if not os.path.exists(root_dir):
        print(f"Error: Directory '{root_dir}' not found.")
        return pd.DataFrame()

    records = []
    model_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    for model_name in model_dirs:
        model_path = os.path.join(root_dir, model_name)
        files = glob.glob(os.path.join(model_path, "*.*"))
        
        for fpath in files:
            ext = fpath.split('.')[-1].lower()
            if ext in ['txt', 'docx', 'pdf']:
                raw_text = extract_text_from_filepath(fpath)
                res = extractor.analyze_text(raw_text)
                if res:
                    res.update({
                        'Corpus': 'AI_Synthetic',
                        'Model_Family': model_name,
                        'Source_File': os.path.basename(fpath)
                    })
                    records.append(res)

    return pd.DataFrame(records)

# ==========================================
# VISUALIZATION ROUTINE
# ==========================================
def generate_ai_comparison_plots(df_ai, df_human):
    print("\nGenerating comparative visualization charts...")
    
    # Combine datasets for unified plotting
    df_ai['Cohort'] = 'AI-Generated (' + df_ai['Model_Family'] + ')'
    if not df_human.empty:
        df_human['Cohort'] = 'Contemporary Human'
        df_combined = pd.concat([df_ai, df_human], ignore_index=True)
    else:
        df_combined = df_ai

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    metrics = ['Sentence_Var', 'Flesch_Ease', 'Zipf_R2', 'MATTR']
    titles = [
        'Sentence Length Variance (Collapse Threshold ≤ 50.0)', 
        'Flesch Reading Ease (FRE Compression ≤ 25.0)', 
        'Zipfian Rank-Frequency Linearity (R²)', 
        'Moving-Average TTR (MATTR₁₀₀)'
    ]

    for i, m in enumerate(metrics):
        sns.boxplot(
            data=df_combined,
            x='Cohort',
            y=m,
            ax=axes[i],
            palette='Set2',
            width=0.5,
            fliersize=2
        )
        axes[i].set_title(titles[i], fontsize=11, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].tick_params(axis='x', rotation=15)
        axes[i].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plot_path = "ai_vs_human_stylometric_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"-> Saved chart: {plot_path}")

# ==========================================
# MAIN EXECUTION ROUTINE
# ==========================================
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"ai_validation_analysis_report_{timestamp}.txt"
    logger = DualLogger(log_file)
    sys.stdout = logger

    print("================================================================================")
    print("SYNTHETIC AI CORPUS VALIDATION & PER-MODEL STYLOMETRIC ANALYSIS")
    print(f"Execution Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("================================================================================")

    extractor = UnifiedStylometricExtractor(mattr_window=100)

    # 1. Process AI-Generated Papers Directory
    print("\n[1/3] Processing AI-Generated Papers Directory...")
    ai_df = process_ai_generated_directory("AI-Generated-Papers", extractor)
    if ai_df.empty:
        print("Error: No valid documents processed from AI-Generated-Papers. Check folder path.")
        return
    
    ai_df.to_csv("ai_extracted_dataset.csv", index=False)
    print(f"-> Successfully processed {len(ai_df)} synthetic papers across model families: {ai_df['Model_Family'].unique().tolist()}")

    # 2. Load Contemporary Human Baseline for Comparison (if available)
    human_df = pd.DataFrame()
    if os.path.exists("moodle_extracted_dataset.csv"):
        human_df = pd.read_csv("moodle_extracted_dataset.csv")
        print(f"-> Loaded contemporary human baseline corpus ({len(human_df)} papers) for comparison.")

    # 3. Aggregate Reports (As a Whole & Per Model)
    print("\n[2/3] Generating Summary Reports (As a Whole & Per Model)...")
    features = ['Sentence_Var', 'Flesch_Ease', 'Zipf_R2', 'Entropy', 'Nominal_Ratio_%', 'MATTR', 'Grade_Level', 'Passive_%']

    # As a Whole Summary
    overall_summary = []
    for f in features:
        vals = ai_df[f].dropna()
        overall_summary.append({
            'Metric': f,
            'AI Corpus (Overall Mean ± SD)': f"{vals.mean():.3f} ± {vals.std():.3f}",
            'Min': vals.min(),
            'Max': vals.max()
        })
    print("\n=== SYNTHETIC AI CORPUS (OVERALL SUMMARY) ===")
    print(pd.DataFrame(overall_summary).to_markdown(index=False))

    # Per Model Family Breakdown
    model_grouped = ai_df.groupby('Model_Family')[features].agg(['mean', 'std']).round(3)
    print("\n=== PER-MODEL FAMILY STYLOMETRIC BREAKDOWN ===")
    print(model_grouped.to_markdown())

    # 4. Generate Visualizations
    print("\n[3/3] Generating Graphics...")
    generate_ai_comparison_plots(ai_df, human_df)

    print(f"\nExecution finished. Output report logged to: {log_file}")
    sys.stdout = logger.terminal
    logger.close()

if __name__ == "__main__":
    main()
