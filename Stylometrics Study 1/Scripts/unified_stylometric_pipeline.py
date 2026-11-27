#!/usr/bin/env python3
"""
Unified Stylometric Extraction, Validation, Confound-Decomposition,
Demographic Fairness, and Synthetic AI Benchmark Pipeline.

Replaces:
  - 1-analyze-both-moodle-exports-and-micusp-pdfs.py
  - 2-analyze-esl-subgroups.py
  - 3-analyze-ai-generated-papers.py

Usage:
  Full raw extraction + analysis:
      python unified_stylometric_pipeline.py
  Run directly on existing extracted CSVs:
      python unified_stylometric_pipeline.py --from-csv
"""

import argparse
import glob
import hashlib
import logging
import math
import os
import re
import sys
import zipfile
from collections import Counter
from datetime import datetime

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from docx import Document
from nltk.tokenize import sent_tokenize, word_tokenize
from pypdf import PdfReader
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

logging.getLogger("pypdf").setLevel(logging.ERROR)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

plt.style.use(
    "seaborn-v0_8-paper" if "seaborn-v0_8-paper" in plt.style.available else "default"
)
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 300

# ==============================================================================
# CONSTANTS & CONFIGURATION
# ==============================================================================
SALT = "UTICA_IRB_2026_SECURE_SALT"
RANDOM_STATE = 42
SEMESTER_MAP = {"01": ("Spring", 1), "05": ("Summer", 2), "10": ("Fall", 3)}

PHYSICAL_FEATURES = [
    "Sentence_Var",
    "Flesch_Ease",
    "Zipf_R2",
    "Entropy",
    "Nominal_Ratio_%",
    "MATTR",
    "Grade_Level",
    "Passive_%",
]

STEM_DISCIPLINES = {
    "Biology",
    "Civil & Environmental Engineering",
    "Economics",
    "Industrial & Operations Engineering",
    "Mechanical Engineering",
    "Natural Resources & Environment",
    "Nursing",
    "Physics",
    "Psychology",
}

REPORT_GENRES = {"Report", "Research Paper", "Proposal"}
GRAD_LEVELS = {
    "G1",
    "G2",
    "G3",
    "First Year Graduate",
    "Second Year Graduate",
    "Third Year Graduate",
}


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


# ==============================================================================
# UNIFIED STYLOMETRIC EXTRACTOR ENGINE (SINGLE SOURCE OF TRUTH)
# ==============================================================================
class UnifiedStylometricExtractor:
    """
    Single locked feature extractor applied identically across MICUSP, Moodle,
    Contemporary ESL, and Synthetic AI corpora.
    """

    def __init__(
        self,
        mattr_window=100,
        min_words=50,
        min_sentences=3,
        sentence_var_cap=1000.0,
    ):
        self.mattr_window = mattr_window
        self.min_words = min_words
        self.min_sentences = min_sentences
        self.sentence_var_cap = sentence_var_cap

        self.ai_signatures = {
            "delve", "testament", "pivotal", "foster", "synergy", "bespoke",
            "demystify", "furthermore", "moreover", "in conclusion", "tapestry",
            "beacon", "realm", "multifaceted", "underscore", "crucial",
            "imperative", "paramount", "vibrant", "intertwined", "seamlessly",
        }
        self.synthetic_trigrams = [
            "in today's rapidly evolving", "in today's digital age",
            "it is important to remember that", "plays a crucial role in",
            "plays a pivotal role in", "it is worth noting that",
            "a testament to the", "delve into the world", "shed light on",
            "in the realm of", "serves as a reminder", "in conclusion, it can be",
            "it is essential to consider", "a wide range of", "plays an important role",
        ]
        self.passive_regex = re.compile(
            r"\b(am|is|are|was|were|be|been|being)\b\s+"
            r"([\w]+ed|conducted|shown|given|made|found|taken|seen|built|"
            r"chosen|driven|written|known|done)\b",
            re.I,
        )

    def calculate_mattr(self, alpha_tokens):
        if not alpha_tokens:
            return 0.0
        if len(alpha_tokens) < self.mattr_window:
            return float(len(set(alpha_tokens)) / len(alpha_tokens))
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
        _, _, r_value, _, _ = stats.linregress(np.log(ranks), np.log(frequencies))
        return float(r_value ** 2)

    def calculate_passive_voice(self, sentences):
        if not sentences:
            return 0.0
        count = sum(1 for s in sentences if self.passive_regex.search(s))
        return round((count / len(sentences)) * 100.0, 1)

    def calculate_paragraph_symmetry(self, raw_text):
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw_text) if p.strip()]
        if len(paragraphs) < 3:
            return len(paragraphs), 0.0, 1.0, False
        word_counts = [len(p.split()) for p in paragraphs]
        mean = sum(word_counts) / float(len(word_counts))
        variance = sum((x - mean) ** 2 for x in word_counts) / float(len(word_counts))
        std_dev = math.sqrt(variance)
        cv = (std_dev / mean) if mean > 0 else 1.0
        return len(paragraphs), round(std_dev, 1), round(cv, 2), (cv < 0.22)

    def calculate_vocabulary_burstiness(self, raw_text):
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw_text) if p.strip()]
        if len(paragraphs) < 3:
            return 1.0, False
        stop_words = {
            "the", "is", "at", "which", "on", "a", "an", "and", "or", "to",
            "in", "of", "for", "with", "be", "has", "been", "it", "by", "that",
            "this", "are", "was", "as", "from", "have", "had", "do", "does",
            "did", "but", "not", "what", "all", "were", "when", "we", "there",
            "can", "more", "her", "will", "also", "about", "if", "out", "so",
            "into", "than", "them", "some", "could",
        }
        para_counts = []
        global_counts = Counter()
        for p in paragraphs:
            cleaned = [re.sub(r"[^\w]", "", w.lower()) for w in p.split()]
            p_words = [w for w in cleaned if w and w not in stop_words]
            counts = Counter(p_words)
            para_counts.append(counts)
            global_counts.update(counts)

        top_words = [w for w, c in global_counts.most_common() if c >= 3][:8]
        if not top_words:
            return 1.0, False

        total_dispersion = 0.0
        for w in top_words:
            freqs = [p[w] for p in para_counts]
            mean = sum(freqs) / float(len(freqs))
            variance = sum((x - mean) ** 2 for x in freqs) / float(len(freqs))
            total_dispersion += (variance / mean) if mean > 0 else 0.0

        avg_dispersion = total_dispersion / float(len(top_words))
        return round(avg_dispersion, 2), (avg_dispersion < 0.35)

    @staticmethod
    def count_syllables(word):
        word = re.sub(r"[^a-z]", "", word.lower())
        if len(word) <= 3:
            return 1
        word = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", word)
        word = re.sub(r"^y", "", word)
        matches = re.findall(r"[aeiouy]{1,2}", word)
        return len(matches) if matches else 1

    def deep_artifact_check(self, raw_text, alpha_words):
        """Legacy heuristic artifact audit (kept strictly out of ML features)."""
        penalty = 0
        flags = []

        zw_matches = len(re.findall(r"[\u200B-\u200D\uFEFF]", raw_text))
        if zw_matches > 0:
            penalty += 20
            flags.append(f"Zero-Width Unicode ({zw_matches})")

        nbsp_matches = len(re.findall(r"\u00A0", raw_text))
        if nbsp_matches > 5:
            penalty += 10
            flags.append(f"Web Clipboard Spaces ({nbsp_matches})")

        lower_text = raw_text.lower()
        found_trigrams = [p for p in self.synthetic_trigrams if p in lower_text]
        if found_trigrams:
            penalty += min(25, len(found_trigrams) * 8)
            flags.append(f"Synthetic Trigrams ({len(found_trigrams)})")

        arxiv_finds = re.findall(
            r"(arxiv:\s*\d{4}\.\d{4,5}|arxiv\.org/abs/\d{4}\.\d{4,5}|doi\.org/10\.\d{4})",
            raw_text,
            re.I,
        )
        if arxiv_finds:
            penalty += 25
            flags.append(f"Synthetic Citation ({len(arxiv_finds)})")

        header_matches = re.findall(
            r"^(Concern|Introduction|The Concern|References):", raw_text, re.M
        )
        if header_matches:
            penalty += 10
            flags.append("LLM Section Headers")

        found_buzzwords = sorted(set(w for w in alpha_words if w in self.ai_signatures))
        if len(found_buzzwords) >= 2:
            penalty += min(20, len(found_buzzwords) * 5)
            flags.append(f"AI Vocabulary ({', '.join(found_buzzwords[:3])})")

        _, _, para_cv, is_symmetrical = self.calculate_paragraph_symmetry(raw_text)
        if is_symmetrical:
            penalty += 15
            flags.append(f"Unnatural Para Symmetry (CV={para_cv})")

        dispersion, is_uniform = self.calculate_vocabulary_burstiness(raw_text)
        if is_uniform:
            penalty += 15
            flags.append(f"Low Vocab Burstiness ({dispersion})")

        return penalty, flags

    def clean_text(self, raw_text):
        """Consistent text cleaning across all corpora."""
        text = re.sub(r"```.*?```", "", raw_text, flags=re.S)
        text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.S)
        text = re.sub(r"\$[^$\n]+\$", "", text)
        text = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[EMAIL_REDACTED]", text)
        text = re.sub(
            r"\b(ID|Student ID|Banner ID|ID#)[\s:]*\d+\b",
            "[ID_REDACTED]",
            text,
            flags=re.I,
        )
        text = re.sub(
            r"^(Name|Author|Student|Submitted by)[\s:]+.*$",
            "",
            text,
            flags=re.M | re.I,
        )
        return text

    def analyze_text(self, raw_text):
        if not raw_text or len(raw_text.strip()) < 50:
            return None

        clean_text = self.clean_text(raw_text)
        sentences = sent_tokenize(clean_text)
        all_tokens = word_tokenize(clean_text)
        alpha_words = [w.lower() for w in all_tokens if w.isalpha()]

        total_words = len(alpha_words)
        total_sentences = len(sentences)
        if total_words < self.min_words or total_sentences < self.min_sentences:
            return None

        sent_lengths = [
            len([w for w in word_tokenize(s) if w.isalpha()]) for s in sentences
        ]
        sent_var_raw = float(np.var(sent_lengths)) if len(sent_lengths) > 1 else 0.0
        sent_var_capped = (
            min(self.sentence_var_cap, sent_var_raw)
            if self.sentence_var_cap is not None
            else sent_var_raw
        )

        mattr = self.calculate_mattr(alpha_words)
        zipf_fit = self.calculate_zipf_r2(alpha_words)

        char_counts = Counter(clean_text)
        total_chars = len(clean_text)
        entropy = (
            -sum(
                (cnt / total_chars) * math.log2(cnt / total_chars)
                for cnt in char_counts.values()
            )
            if total_chars > 0
            else 0.0
        )

        passive_ratio = self.calculate_passive_voice(sentences)
        total_syllables = sum(self.count_syllables(w) for w in alpha_words)
        words_per_sent = total_words / float(total_sentences)
        syllables_per_word = total_syllables / float(total_words)

        flesch_ease = max(
            0.0,
            min(100.0, 206.835 - (1.015 * words_per_sent) - (84.6 * syllables_per_word)),
        )
        grade_level = max(
            0.0, (0.39 * words_per_sent) + (11.8 * syllables_per_word) - 15.59
        )

        nominalizations = [
            w
            for w in alpha_words
            if w.endswith(("tion", "ment", "ance", "ence")) and len(w) > 5
        ]
        nom_ratio = (len(nominalizations) / float(total_words)) * 100.0

        v_score = min(25.0, (sent_var_capped / 85.0) * 25.0)
        m_score = min(25.0, (mattr / 0.75) * 25.0)
        z_score = 15.0 if zipf_fit >= 0.85 else 5.0
        base_score = 25.0 + v_score + m_score + z_score
        penalty, flags = self.deep_artifact_check(clean_text, alpha_words)
        final_score = max(0.0, min(100.0, base_score - penalty))

        if final_score > 75 and len(flags) == 0:
            verdict = "Human"
        elif final_score >= 55:
            verdict = "Review"
        elif final_score >= 40:
            verdict = "Hybrid"
        else:
            verdict = "Bot/AI"

        norm_text_for_hash = " ".join(alpha_words[:500])
        text_hash = hashlib.sha256(norm_text_for_hash.encode("utf-8")).hexdigest()[:16]

        return {
            "Word_Count": total_words,
            "Sentence_Count": total_sentences,
            "Entropy": round(entropy, 3),
            "MATTR": round(mattr, 4),
            "Zipf_R2": round(zipf_fit, 4),
            "Sentence_Var": round(sent_var_capped, 2),
            "Sentence_Var_Raw": round(sent_var_raw, 2),
            "Sentence_Var_Capped_Flag": int(
                self.sentence_var_cap is not None and sent_var_raw > self.sentence_var_cap
            ),
            "Flesch_Ease": round(flesch_ease, 1),
            "Grade_Level": round(grade_level, 1),
            "Nominal_Ratio_%": round(nom_ratio, 1),
            "Passive_%": round(passive_ratio, 1),
            "Auth_Score": round(final_score, 1),
            "Verdict": verdict,
            "Flag_Count": len(flags),
            "Flags": ", ".join(flags) if flags else "None",
            "Text_Hash": text_hash,
        }

# ==============================================================================
# FILE PARSERS & CORPUS LOADERS
# ==============================================================================
def anonymize_student_id(raw_name):
    if not raw_name or raw_name == "Unknown Student":
        return "Sub_ANONYMOUS"
    salted_string = f"{SALT}_{raw_name.strip().lower()}"
    return f"Sub_{hashlib.sha256(salted_string.encode()).hexdigest()[:8]}"


def parse_moodle_filename(filename):
    strict_pattern = r"^([A-Z]{3,4})-(\d{3})-([A-Z0-9]+)-(\d{4})(\d{2})-(.+)$"
    match = re.match(strict_pattern, filename)
    if match:
        program, course_num, section, year, month_code, assignment = match.groups()
        sem_name, sem_order = SEMESTER_MAP.get(month_code, ("Unknown", 9))
        return {
            "Program": program,
            "Course": f"{program}-{course_num}",
            "Section": section,
            "Year": int(year),
            "Semester": sem_name,
            "Sem_Order": sem_order,
            "Assignment": assignment,
        }
    year_match = re.search(r"\b(202[0-9])\b", filename)
    extracted_year = int(year_match.group(1)) if year_match else 2025
    sem_name = (
        "Spring"
        if "01" in filename or "Spring" in filename
        else ("Fall" if "10" in filename or "Fall" in filename else "Unassigned")
    )
    sem_order = 1 if sem_name == "Spring" else (3 if sem_name == "Fall" else 9)
    return {
        "Program": filename.split("-")[0] if "-" in filename else "GEN",
        "Course": filename.split("-")[0] if "-" in filename else "GEN-100",
        "Section": "01",
        "Year": extracted_year,
        "Semester": sem_name,
        "Sem_Order": sem_order,
        "Assignment": filename,
    }


def extract_text_from_stream(stream, ext):
    try:
        if ext == "txt":
            return stream.read().decode("utf-8", errors="ignore")
        if ext == "docx":
            doc = Document(stream)
            return "\n".join(p.text for p in doc.paragraphs)
        if ext == "pdf":
            reader = PdfReader(stream)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""
    return ""


def extract_text_from_filepath(filepath):
    ext = filepath.split(".")[-1].lower()
    try:
        if ext == "txt":
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        if ext == "docx":
            doc = Document(filepath)
            return "\n".join(p.text for p in doc.paragraphs)
        if ext == "pdf":
            reader = PdfReader(filepath)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""
    return ""


def parse_moodle_student_name(path):
    parts = path.split("/")
    if len(parts) > 1:
        return parts[0].split("_")[0].strip()
    return "Unknown Student"


def process_moodle_exports(moodle_dir, extractor):
    zip_files = sorted(glob.glob(os.path.join(moodle_dir, "*.zip")))
    if not zip_files:
        print(f"No ZIP archives found in '{moodle_dir}'.")
        return pd.DataFrame()

    records = []
    doc_counter = 1
    for zpath in zip_files:
        meta = parse_moodle_filename(os.path.basename(zpath))
        with zipfile.ZipFile(zpath, "r") as z:
            for zip_info in z.infolist():
                if (
                    zip_info.is_dir()
                    or "__MACOSX" in zip_info.filename
                    or "/." in zip_info.filename
                ):
                    continue
                ext = zip_info.filename.split(".")[-1].lower()
                if ext in ("txt", "docx", "pdf"):
                    raw_name = parse_moodle_student_name(zip_info.filename)
                    subject_id = anonymize_student_id(raw_name)
                    doc_id = f"Doc_{doc_counter:04d}.{ext}"
                    doc_counter += 1
                    with z.open(zip_info) as f:
                        text = extract_text_from_stream(f, ext)
                    res = extractor.analyze_text(text)
                    if res:
                        res.update(
                            {
                                "Corpus": "Moodle_PostLLM",
                                "Era": "Post-LLM (2025-2026)",
                                "Doc_ID": doc_id,
                                "Source_File": os.path.basename(zip_info.filename),
                                "Subject_ID": subject_id,
                                "Course": meta["Course"],
                                "Year": meta["Year"],
                                "Semester": meta["Semester"],
                                "Assignment": meta["Assignment"],
                            }
                        )
                        records.append(res)
    return pd.DataFrame(records)


def process_micusp_pdfs(micusp_dir, metadata_csv, extractor):
    pdf_files = sorted(glob.glob(os.path.join(micusp_dir, "*.pdf")))
    if not pdf_files:
        print(f"No PDF files found in '{micusp_dir}'.")
        return pd.DataFrame()

    meta_df = pd.read_csv(metadata_csv) if os.path.exists(metadata_csv) else pd.DataFrame()
    records = []
    for pdf_path in pdf_files:
        paper_id = os.path.splitext(os.path.basename(pdf_path))[0]
        raw_text = extract_text_from_filepath(pdf_path)
        res = extractor.analyze_text(raw_text)
        if res:
            res.update(
                {
                    "Corpus": "MICUSP_PreLLM",
                    "Era": "Pre-LLM Baseline",
                    "PAPER ID": paper_id,
                    "Source_File": os.path.basename(pdf_path),
                    "Subject_ID": f"MICUSP_{paper_id}",
                    "Year": 2008,
                    "Semester": "Pre-LLM Baseline",
                    "Assignment": "MICUSP_Historical",
                }
            )
            records.append(res)

    df = pd.DataFrame(records)
    if not df.empty and not meta_df.empty and "PAPER ID" in meta_df.columns:
        df = pd.merge(df, meta_df, on="PAPER ID", how="left")
        if "NATIVENESS" in df.columns:
            df["Language_Group"] = df["NATIVENESS"].apply(
                lambda x: "NES" if str(x).strip().upper() == "NS" else "ESL"
            )
    return df


def process_non_native_dir(non_native_dir, extractor):
    files = sorted(glob.glob(os.path.join(non_native_dir, "**", "*.*"), recursive=True))
    records = []
    for fpath in files:
        ext = fpath.split(".")[-1].lower()
        if ext in ("txt", "docx", "pdf"):
            raw_text = extract_text_from_filepath(fpath)
            res = extractor.analyze_text(raw_text)
            if res:
                res.update(
                    {
                        "Corpus": "Contemporary_ESL",
                        "Era": "Post-LLM (2025-2026)",
                        "Language_Group": "ESL",
                        "Source_File": os.path.basename(fpath),
                    }
                )
                records.append(res)
    return pd.DataFrame(records)


def process_ai_generated_directory(root_dir, extractor):
    if not os.path.exists(root_dir):
        print(f"AI directory '{root_dir}' not found.")
        return pd.DataFrame()
    records = []
    model_dirs = sorted(
        d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
    )
    for model_name in model_dirs:
        model_path = os.path.join(root_dir, model_name)
        for fpath in sorted(glob.glob(os.path.join(model_path, "*.*"))):
            ext = fpath.split(".")[-1].lower()
            if ext in ("txt", "docx", "pdf"):
                raw_text = extract_text_from_filepath(fpath)
                res = extractor.analyze_text(raw_text)
                if res:
                    res.update(
                        {
                            "Corpus": "AI_Synthetic",
                            "Era": "Synthetic AI",
                            "Model_Family": model_name,
                            "Source_File": os.path.basename(fpath),
                        }
                    )
                    records.append(res)
    return pd.DataFrame(records)

# ==============================================================================
# STATISTICAL HELPERS (DOC-LEVEL & AUTHOR-LEVEL, EFFECT SIZES, BH-FDR)
# ==============================================================================
def calculate_cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return 0.0
    s1, s2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled = math.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    return (np.mean(g1) - np.mean(g2)) / pooled if pooled > 0 else 0.0


def calculate_cliffs_delta(g1, g2):
    a = np.asarray(g1)
    b = np.asarray(g2)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    diff = a[:, None] - b[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / (len(a) * len(b)))


def apply_bh_fdr(p_values):
    p_arr = np.asarray(p_values, dtype=float)
    m = len(p_arr)
    if m == 0:
        return np.array([])
    order = np.argsort(p_arr)
    q_arr = np.zeros(m)
    cum_min = 1.0
    for rank, idx in reversed(list(enumerate(order))):
        q = (p_arr[idx] * m) / (rank + 1)
        cum_min = min(cum_min, q)
        q_arr[idx] = min(1.0, cum_min)
    return q_arr


def compare_two_groups(df_a, df_b, features, label_a="Group A", label_b="Group B"):
    rows = []
    mw_pvals = []
    for f in features:
        va = df_a[f].dropna()
        vb = df_b[f].dropna()
        if len(va) < 3 or len(vb) < 3:
            continue
        _, t_p = stats.ttest_ind(va, vb, equal_var=False)
        u_stat, mw_p = stats.mannwhitneyu(va, vb, alternative="two-sided")
        d_val = calculate_cohens_d(va, vb)
        delta_val = calculate_cliffs_delta(va, vb)
        mw_pvals.append(mw_p)
        rows.append(
            {
                "Feature": f,
                f"{label_a} Mean±SD": f"{va.mean():.3f} ± {va.std():.3f}",
                f"{label_a} Median [IQR]": f"{va.median():.3f} [{va.quantile(0.25):.3f}, {va.quantile(0.75):.3f}]",
                f"{label_b} Mean±SD": f"{vb.mean():.3f} ± {vb.std():.3f}",
                f"{label_b} Median [IQR]": f"{vb.median():.3f} [{vb.quantile(0.25):.3f}, {vb.quantile(0.75):.3f}]",
                "Cohen_d": round(d_val, 3),
                "Cliffs_delta": round(delta_val, 3),
                "Welch_p": t_p,
                "MW_U": round(u_stat, 1),
                "MW_p": mw_p,
            }
        )
    if rows:
        q_vals = apply_bh_fdr(mw_pvals)
        for row, q in zip(rows, q_vals):
            row["FDR_q"] = round(float(q), 4)
            row["MW_p"] = round(float(row["MW_p"]), 4)
            row["Welch_p"] = round(float(row["Welch_p"]), 4)
    return pd.DataFrame(rows)


def aggregate_to_author_level(df, features, group_col="Subject_ID"):
    if group_col not in df.columns or df[group_col].isna().all():
        return df.copy()
    agg_dict = {f: "mean" for f in features if f in df.columns}
    for extra in ("Word_Count", "Sentence_Count"):
        if extra in df.columns:
            agg_dict[extra] = "mean"
    return df.groupby(group_col, as_index=False).agg(agg_dict)


def build_models():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200, random_state=RANDOM_STATE
        ),
    }


def evaluate_cv(df, features, model, splitter, groups=None):
    X = df[features].to_numpy(dtype=float)
    y = df["Target"].to_numpy(dtype=int)
    oof_probs = np.full(len(df), np.nan)

    splits = (
        splitter.split(X, y, groups)
        if groups is not None
        else splitter.split(X, y)
    )
    fold_aucs, fold_accs = [], []
    for train_idx, test_idx in splits:
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        model.fit(X[train_idx], y[train_idx])
        probs = model.predict_proba(X[test_idx])[:, 1]
        oof_probs[test_idx] = probs
        fold_aucs.append(roc_auc_score(y[test_idx], probs))
        fold_accs.append(accuracy_score(y[test_idx], (probs >= 0.5).astype(int)))

    valid = ~np.isnan(oof_probs)
    y_v, p_v = y[valid], oof_probs[valid]
    preds_v = (p_v >= 0.5).astype(int)

    rng = np.random.default_rng(RANDOM_STATE)
    boot_aucs = []
    for _ in range(500):
        idx = rng.integers(0, len(y_v), size=len(y_v))
        if len(np.unique(y_v[idx])) == 2:
            boot_aucs.append(roc_auc_score(y_v[idx], p_v[idx]))
    ci_lo, ci_hi = (
        (float(np.percentile(boot_aucs, 2.5)), float(np.percentile(boot_aucs, 97.5)))
        if boot_aucs
        else (np.nan, np.nan)
    )

    return {
        "N": int(valid.sum()),
        "N_PreLLM": int((y_v == 0).sum()),
        "N_PostLLM": int((y_v == 1).sum()),
        "Accuracy": round(accuracy_score(y_v, preds_v), 4),
        "F1": round(f1_score(y_v, preds_v), 4),
        "ROC_AUC": round(roc_auc_score(y_v, p_v), 4),
        "AUC_95CI": f"[{ci_lo:.4f}, {ci_hi:.4f}]",
        "PR_AUC": round(average_precision_score(y_v, p_v), 4),
        "Brier": round(brier_score_loss(y_v, p_v), 4),
        "Fold_AUC_Mean": round(float(np.mean(fold_aucs)), 4) if fold_aucs else np.nan,
        "Fold_AUC_SD": round(float(np.std(fold_aucs)), 4) if fold_aucs else np.nan,
        "OOF_Probs": oof_probs,
    }


def match_word_count_1to1(df_pre, df_post, max_rel_diff=0.20):
    pre = df_pre.dropna(subset=["Word_Count"]).copy().reset_index(drop=True)
    post = df_post.dropna(subset=["Word_Count"]).copy().reset_index(drop=True)
    used_pre = set()
    matched_pre_idx = []
    matched_post_idx = []

    for i, row in post.iterrows():
        wc = row["Word_Count"]
        candidates = pre.loc[~pre.index.isin(used_pre)]
        if candidates.empty:
            break
        rel_diffs = (candidates["Word_Count"] - wc).abs() / max(wc, 1)
        best_idx = rel_diffs.idxmin()
        if rel_diffs.loc[best_idx] <= max_rel_diff:
            used_pre.add(best_idx)
            matched_pre_idx.append(best_idx)
            matched_post_idx.append(i)

    return pd.concat(
        [pre.loc[matched_pre_idx], post.loc[matched_post_idx]], ignore_index=True
    )


def compute_feature_importance(df, features):
    X = df[features].to_numpy(dtype=float)
    y = df["Target"].to_numpy(dtype=int)
    rf = build_models()["RandomForest"]
    rf.fit(X, y)

    if HAS_SHAP:
        explainer = shap.TreeExplainer(rf)
        shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            vals = np.abs(shap_vals[1]).mean(axis=0)
        elif shap_vals.ndim == 3:
            vals = np.abs(shap_vals[:, :, 1]).mean(axis=0)
        else:
            vals = np.abs(shap_vals).mean(axis=0)
        method = "Mean_|TreeSHAP|"
    else:
        perm = permutation_importance(
            rf, X, y, scoring="roc_auc", n_repeats=15, random_state=RANDOM_STATE
        )
        vals = perm.importances_mean
        method = "Permutation_AUC_Drop"

    imp_df = pd.DataFrame({"Feature": features, method: np.round(vals, 4)})
    return imp_df.sort_values(method, ascending=False).reset_index(drop=True), rf, method

# ==============================================================================
# CONFOUND-DECOMPOSITION ABLATIONS & PLOTS
# ==============================================================================
ENG_PHYS_ECON = {
    "Civil & Environmental Engineering",
    "Economics",
    "Industrial & Operations Engineering",
    "Mechanical Engineering",
    "Physics",
}


def run_confound_decomposition(micusp_df, moodle_df, features, out_dir):
    pre = micusp_df.dropna(subset=features).copy()
    post = moodle_df.dropna(subset=features).copy()
    pre["Target"] = 0
    post["Target"] = 1

    if "Subject_ID" not in pre.columns:
        pre["Subject_ID"] = [f"MICUSP_{i}" for i in range(len(pre))]
    if "Subject_ID" not in post.columns:
        post["Subject_ID"] = [f"MOODLE_{i}" for i in range(len(post))]

    scenarios = [("1_Full_Baseline", pd.concat([pre, post], ignore_index=True))]

    if "DISCIPLINE" in pre.columns:
        pre_stem = pre[pre["DISCIPLINE"].isin(STEM_DISCIPLINES)]
        if len(pre_stem) >= 20:
            scenarios.append(
                ("2_MICUSP_STEM_Only", pd.concat([pre_stem, post], ignore_index=True))
            )
        pre_quant = pre[pre["DISCIPLINE"].isin(ENG_PHYS_ECON)]
        if len(pre_quant) >= 20:
            scenarios.append(
                ("3_MICUSP_Eng_Phys_Econ", pd.concat([pre_quant, post], ignore_index=True))
            )

    if "PAPER TYPE" in pre.columns:
        pre_rep = pre[pre["PAPER TYPE"].isin(REPORT_GENRES)]
        if len(pre_rep) >= 20:
            scenarios.append(
                ("4_MICUSP_Reports_Papers", pd.concat([pre_rep, post], ignore_index=True))
            )

    if "Word_Count" in pre.columns and "Word_Count" in post.columns:
        pre_len = pre[pre["Word_Count"].between(750, 2000)]
        post_len = post[post["Word_Count"].between(750, 2000)]
        if len(pre_len) >= 20 and len(post_len) >= 20:
            scenarios.append(
                ("5_LengthBand_750_2000w", pd.concat([pre_len, post_len], ignore_index=True))
            )

        matched_1to1 = match_word_count_1to1(pre, post, max_rel_diff=0.20)
        if len(matched_1to1) >= 40:
            scenarios.append(("6_LengthMatched_1to1", matched_1to1))

        if "DISCIPLINE" in pre.columns:
            pre_stem_len = pre[
                pre["DISCIPLINE"].isin(STEM_DISCIPLINES)
                & pre["Word_Count"].between(750, 2000)
            ]
            if len(pre_stem_len) >= 20 and len(post_len) >= 20:
                scenarios.append(
                    (
                        "7_STEM_Plus_LengthBand",
                        pd.concat([pre_stem_len, post_len], ignore_index=True),
                    )
                )
            matched_stem = match_word_count_1to1(
                pre[pre["DISCIPLINE"].isin(STEM_DISCIPLINES)], post, max_rel_diff=0.20
            )
            if len(matched_stem) >= 40:
                scenarios.append(("8_STEM_LengthMatched_1to1", matched_stem))

    summary_rows = []
    importance_frames = []

    for name, sub_df in scenarios:
        rf = build_models()["RandomForest"]
        gkf = GroupKFold(n_splits=5)
        res = evaluate_cv(
            sub_df, features, rf, gkf, groups=sub_df["Subject_ID"].astype(str).to_numpy()
        )
        imp_df, _, _ = compute_feature_importance(sub_df, features)
        top_feats = ", ".join(imp_df["Feature"].head(3).tolist())

        summary_rows.append(
            {
                "Ablation_Scenario": name,
                "N_PreLLM": res["N_PreLLM"],
                "N_PostLLM": res["N_PostLLM"],
                "AuthorDisjoint_Acc": res["Accuracy"],
                "AuthorDisjoint_AUC": res["ROC_AUC"],
                "AUC_95CI": res["AUC_95CI"],
                "Brier": res["Brier"],
                "Top_3_Features": top_feats,
            }
        )
        imp_df["Scenario"] = name
        importance_frames.append(imp_df)

    ablation_df = pd.DataFrame(summary_rows)
    ablation_df.to_csv(os.path.join(out_dir, "confound_ablation_results.csv"), index=False)

    if not ablation_df.empty:
        plt.figure(figsize=(9, 5))
        sns.barplot(
            data=ablation_df,
            y="Ablation_Scenario",
            x="AuthorDisjoint_AUC",
            hue="Ablation_Scenario",
            palette="viridis",
            legend=False,
        )
        plt.xlim(0.5, 1.05)
        plt.axvline(0.5, color="gray", linestyle="--", linewidth=1)
        plt.title(
            "Confound-Decomposition Ablation: Author-Disjoint ROC-AUC Across Controlled Subsets",
            fontsize=11,
            fontweight="bold",
        )
        plt.xlabel("Author-Disjoint 5-Fold ROC-AUC")
        plt.ylabel("")
        for i, row in ablation_df.iterrows():
            plt.text(
                min(row["AuthorDisjoint_AUC"] + 0.008, 1.04),
                i,
                f"{row['AuthorDisjoint_AUC']:.3f} (n={row['N_PreLLM']}+{row['N_PostLLM']})",
                va="center",
                fontsize=8.5,
            )
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "confound_ablation_auc.png"), dpi=300)
        plt.close()

    return ablation_df, pd.concat(importance_frames, ignore_index=True)


def generate_esl_plots(df_esl_combined, post_with_probs, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    plot_metrics = ["Sentence_Var", "Flesch_Ease", "Zipf_R2", "MATTR"]
    titles = [
        "Sentence Length Variance (Capped at 1000)",
        "Flesch Reading Ease",
        "Zipfian Rank-Frequency Linearity (R²)",
        "Moving-Average TTR (MATTR₁₀₀)",
    ]
    for i, m in enumerate(plot_metrics):
        sns.boxplot(
            data=df_esl_combined,
            x="Era",
            y=m,
            hue="Language_Group",
            ax=axes[i],
            palette={"NES": "#2b5c8f", "ESL": "#d95f02"},
            width=0.55,
            fliersize=2,
        )
        axes[i].set_title(titles[i], fontsize=11, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].grid(True, linestyle="--", alpha=0.45)
        axes[i].legend(title="Language Group", loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "esl_feature_distributions.png"), dpi=300)
    plt.close()

    if post_with_probs is not None and not post_with_probs.empty:
        plt.figure(figsize=(8, 5))
        sns.kdeplot(
            data=post_with_probs,
            x="Predicted_Prob_PostLLM",
            hue="Language_Group",
            fill=True,
            common_norm=False,
            clip=(0.0, 1.0),
            palette={"NES": "#2b5c8f", "ESL": "#d95f02"},
            alpha=0.4,
        )
        plt.title(
            "Out-of-Fold Random Forest Predicted P(Post-LLM | x) by Language Group",
            fontsize=11,
            fontweight="bold",
        )
        plt.xlabel("Out-of-Fold Predicted Post-LLM Probability P(y=1 | x)")
        plt.ylabel("Density")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "esl_predicted_probability_kde.png"), dpi=300)
        plt.close()


def generate_ai_comparison_plots(ai_df, human_df, micusp_df, out_dir):
    if ai_df.empty:
        return
    frames = []
    ai_plot = ai_df.copy()
    ai_plot["Cohort"] = "AI (" + ai_plot["Model_Family"].astype(str) + ")"
    frames.append(ai_plot)

    if not human_df.empty:
        h_plot = human_df.copy()
        h_plot["Cohort"] = "Contemporary Human"
        frames.append(h_plot)

    if not micusp_df.empty:
        m_plot = micusp_df.copy()
        m_plot["Cohort"] = "MICUSP Pre-LLM"
        frames.append(m_plot)

    combined = pd.concat(frames, ignore_index=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    metrics = ["Sentence_Var", "Flesch_Ease", "Zipf_R2", "MATTR"]
    titles = [
        "Sentence Length Variance",
        "Flesch Reading Ease",
        "Zipfian Rank-Frequency Linearity (R²)",
        "Moving-Average TTR (MATTR₁₀₀)",
    ]
    for i, m in enumerate(metrics):
        sns.boxplot(
            data=combined,
            x="Cohort",
            y=m,
            hue="Cohort",
            ax=axes[i],
            palette="Set2",
            width=0.55,
            fliersize=2,
            legend=False,
        )
        axes[i].set_title(titles[i], fontsize=11, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].tick_params(axis="x", rotation=20)
        axes[i].grid(True, linestyle="--", alpha=0.45)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ai_vs_human_stylometric_comparison.png"), dpi=300)
    plt.close()

# ==============================================================================
# MAIN ORCHESTRATION
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Unified Stylometric Extraction, Validation & Confound-Decomposition Pipeline"
    )
    parser.add_argument(
        "--from-csv",
        action="store_true",
        help="Skip raw file parsing and run analyses directly on existing extracted CSVs.",
    )
    parser.add_argument("--moodle-dir", default="./moodle_exports")
    parser.add_argument("--micusp-dir", default="./micusp_pdfs")
    parser.add_argument("--micusp-meta", default="micusp_papers.csv")
    parser.add_argument("--non-native-dir", default="./non-native")
    parser.add_argument("--ai-dir", default="./AI-Generated-Papers")
    parser.add_argument("--out-dir", default="./outputs")
    parser.add_argument("--sentence-var-cap", type=float, default=1000.0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.out_dir, f"unified_pipeline_report_{timestamp}.txt")
    logger = DualLogger(log_path)
    sys.stdout = logger

    print("=" * 84)
    print("UNIFIED STYLOMETRIC ANALYSIS, VALIDATION & CONFOUND-DECOMPOSITION PIPELINE")
    print(f"Execution Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'FROM EXISTING CSVs' if args.from_csv else 'FULL RAW EXTRACTION'}")
    print(f"Sentence_Var Cap: {args.sentence_var_cap} | SHAP available: {HAS_SHAP}")
    print("=" * 84)

    extractor = UnifiedStylometricExtractor(
        mattr_window=100,
        min_words=50,
        min_sentences=3,
        sentence_var_cap=args.sentence_var_cap,
    )

    # STEP 1: LOAD OR EXTRACT
    if args.from_csv:
        print("\n[1/6] Loading existing CSV datasets...")
        moodle_df = (
            pd.read_csv("moodle_extracted_dataset.csv")
            if os.path.exists("moodle_extracted_dataset.csv")
            else pd.DataFrame()
        )
        micusp_df = (
            pd.read_csv("micusp_extracted_baseline.csv")
            if os.path.exists("micusp_extracted_baseline.csv")
            else pd.DataFrame()
        )
        esl_saved_df = (
            pd.read_csv("esl_subgroup_extracted_dataset.csv")
            if os.path.exists("esl_subgroup_extracted_dataset.csv")
            else pd.DataFrame()
        )
        ai_df = (
            pd.read_csv("ai_extracted_dataset.csv")
            if os.path.exists("ai_extracted_dataset.csv")
            else pd.DataFrame()
        )

        for name, df in [
            ("Moodle", moodle_df),
            ("MICUSP", micusp_df),
            ("ESL_Saved", esl_saved_df),
            ("AI", ai_df),
        ]:
            if not df.empty and "Sentence_Var" in df.columns:
                n_over = int((df["Sentence_Var"] > args.sentence_var_cap).sum())
                if n_over > 0:
                    print(
                        f"  -> Harmonizing {name}: capping {n_over} rows with "
                        f"Sentence_Var > {args.sentence_var_cap}."
                    )
                    df["Sentence_Var_Raw"] = df["Sentence_Var"]
                    df["Sentence_Var"] = df["Sentence_Var"].clip(upper=args.sentence_var_cap)

        if not micusp_df.empty and os.path.exists(args.micusp_meta):
            meta_df = pd.read_csv(args.micusp_meta)
            merge_cols = [
                c
                for c in ["PAPER ID", "DISCIPLINE", "PAPER TYPE", "STUDENT LEVEL", "NATIVENESS"]
                if (c in meta_df.columns and c not in micusp_df.columns) or c == "PAPER ID"
            ]
            if "PAPER ID" in micusp_df.columns and len(merge_cols) > 1:
                micusp_df = pd.merge(micusp_df, meta_df[merge_cols], on="PAPER ID", how="left")
            if "NATIVENESS" in micusp_df.columns:
                micusp_df["Language_Group"] = micusp_df["NATIVENESS"].apply(
                    lambda x: "NES" if str(x).strip().upper() == "NS" else "ESL"
                )

        esl_univ_df = (
            esl_saved_df[
                (esl_saved_df["Era"] != "Pre-LLM Baseline")
                & (esl_saved_df["Language_Group"] == "ESL")
            ].copy()
            if not esl_saved_df.empty and "Language_Group" in esl_saved_df.columns
            else pd.DataFrame()
        )
    else:
        print("\n[1/6] Running unified extraction on raw corpora...")
        moodle_df = process_moodle_exports(args.moodle_dir, extractor)
        micusp_df = process_micusp_pdfs(args.micusp_dir, args.micusp_meta, extractor)
        esl_univ_df = process_non_native_dir(args.non_native_dir, extractor)
        ai_df = process_ai_generated_directory(args.ai_dir, extractor)

        if not moodle_df.empty:
            moodle_df.to_csv(os.path.join(args.out_dir, "moodle_extracted_dataset.csv"), index=False)
        if not micusp_df.empty:
            micusp_df.to_csv(os.path.join(args.out_dir, "micusp_extracted_baseline.csv"), index=False)
        if not ai_df.empty:
            ai_df.to_csv(os.path.join(args.out_dir, "ai_extracted_dataset.csv"), index=False)

    print(
        f"-> Loaded counts | MICUSP: {len(micusp_df)} | Moodle: {len(moodle_df)} | "
        f"Contemporary ESL: {len(esl_univ_df)} | Synthetic AI: {len(ai_df)}"
    )

    if micusp_df.empty or moodle_df.empty:
        print("Error: Both MICUSP and Moodle datasets are required to continue.")
        sys.stdout = logger.terminal
        logger.close()
        return

    # STEP 2: LOCKED CORPUS SUMMARY (DOC-LEVEL & AUTHOR-LEVEL)
    print("\n[2/6] Generating Locked Corpus Summary Tables (Fixes Table 4 vs Table 5)...")
    doc_comp = compare_two_groups(
        moodle_df, micusp_df, PHYSICAL_FEATURES, label_a="PostLLM", label_b="PreLLM"
    )
    doc_comp.to_csv(os.path.join(args.out_dir, "corpus_comparison_doc_level.csv"), index=False)
    print("\n=== DOCUMENT-LEVEL CORPUS COMPARISON (POST-LLM vs PRE-LLM) ===")
    print(doc_comp.to_markdown(index=False))

    moodle_auth = aggregate_to_author_level(moodle_df, PHYSICAL_FEATURES)
    micusp_auth = aggregate_to_author_level(micusp_df, PHYSICAL_FEATURES)
    auth_comp = compare_two_groups(
        moodle_auth, micusp_auth, PHYSICAL_FEATURES, label_a="PostLLM_Auth", label_b="PreLLM_Auth"
    )
    auth_comp.to_csv(os.path.join(args.out_dir, "corpus_comparison_author_level.csv"), index=False)
    print(
        f"\n=== AUTHOR-LEVEL CORPUS COMPARISON (Unique Authors: "
        f"Post-LLM={len(moodle_auth)}, Pre-LLM={len(micusp_auth)}) ==="
    )
    print(auth_comp.to_markdown(index=False))

    if "Verdict" in micusp_df.columns:
        fp_count = int((micusp_df["Verdict"] != "Human").sum())
        fpr = (fp_count / len(micusp_df)) * 100.0
        print(f"\nLegacy Heuristic Baseline FPR on Pre-LLM MICUSP: {fpr:.2f}% ({fp_count}/{len(micusp_df)})")

    # STEP 3: NON-CIRCULAR ML VALIDATION
    print("\n[3/6] Running Non-Circular ML Validation Protocols...")
    pre_ml = micusp_df.dropna(subset=PHYSICAL_FEATURES).copy()
    post_ml = moodle_df.dropna(subset=PHYSICAL_FEATURES).copy()
    pre_ml["Target"] = 0
    post_ml["Target"] = 1
    if "Subject_ID" not in pre_ml.columns:
        pre_ml["Subject_ID"] = [f"MICUSP_{i}" for i in range(len(pre_ml))]
    if "Subject_ID" not in post_ml.columns:
        post_ml["Subject_ID"] = [f"MOODLE_{i}" for i in range(len(post_ml))]

    ml_df = pd.concat([pre_ml, post_ml], ignore_index=True)

    val_rows = []
    rf_author_oof = None
    for model_name, model in build_models().items():
        skf_res = evaluate_cv(
            ml_df,
            PHYSICAL_FEATURES,
            model,
            StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        )
        val_rows.append(
            {
                "Model": model_name,
                "Protocol": "Stratified_5Fold",
                **{k: v for k, v in skf_res.items() if k != "OOF_Probs"},
            }
        )

        gkf_res = evaluate_cv(
            ml_df,
            PHYSICAL_FEATURES,
            model,
            GroupKFold(n_splits=5),
            groups=ml_df["Subject_ID"].astype(str).to_numpy(),
        )
        val_rows.append(
            {
                "Model": model_name,
                "Protocol": "Author_Disjoint_5Fold",
                **{k: v for k, v in gkf_res.items() if k != "OOF_Probs"},
            }
        )
        if model_name == "RandomForest":
            rf_author_oof = gkf_res["OOF_Probs"]

    val_df = pd.DataFrame(val_rows)
    val_df.to_csv(os.path.join(args.out_dir, "ml_validation_summary.csv"), index=False)
    print("\n=== MODEL VALIDATION SUMMARY ===")
    print(val_df.to_markdown(index=False))

    imp_df, rf_full, imp_method = compute_feature_importance(ml_df, PHYSICAL_FEATURES)
    imp_df.to_csv(os.path.join(args.out_dir, "feature_importance.csv"), index=False)
    print(f"\n=== FEATURE IMPORTANCE ({imp_method}) ===")
    print(imp_df.to_markdown(index=False))

    # STEP 4: CONFOUND-DECOMPOSITION ABLATIONS
    print("\n[4/6] Running Confound-Decomposition Ablations...")
    ablation_df, ablation_imp = run_confound_decomposition(
        micusp_df, moodle_df, PHYSICAL_FEATURES, args.out_dir
    )
    print("\n=== CONFOUND-DECOMPOSITION ABLATION SUMMARY ===")
    print(ablation_df.to_markdown(index=False))
    ablation_imp.to_csv(
        os.path.join(args.out_dir, "confound_ablation_feature_importance.csv"), index=False
    )

    # STEP 5: ESL vs NES DEMOGRAPHIC FAIRNESS (REAL OOF PROBABILITIES)
    print("\n[5/6] Running ESL vs. NES Demographic Fairness Analysis...")
    post_ml["Predicted_Prob_PostLLM"] = rf_author_oof[len(pre_ml) :]

    if not esl_univ_df.empty:
        if "Text_Hash" in esl_univ_df.columns and "Text_Hash" in post_ml.columns:
            esl_keys = set(esl_univ_df["Text_Hash"].dropna())
            is_esl = post_ml["Text_Hash"].isin(esl_keys)
        elif "Source_File" in esl_univ_df.columns and "Source_File" in post_ml.columns:
            esl_keys = set(esl_univ_df["Source_File"].dropna())
            is_esl = post_ml["Source_File"].isin(esl_keys)
        else:
            esl_sigs = set(
                zip(
                    esl_univ_df["Word_Count"].round(0),
                    esl_univ_df["MATTR"].round(4),
                    esl_univ_df["Flesch_Ease"].round(1),
                )
            )
            post_sigs = list(
                zip(
                    post_ml["Word_Count"].round(0),
                    post_ml["MATTR"].round(4),
                    post_ml["Flesch_Ease"].round(1),
                )
            )
            is_esl = pd.Series([s in esl_sigs for s in post_sigs], index=post_ml.index)

        post_ml["Language_Group"] = np.where(is_esl, "ESL", "NES")
        n_matched = int(is_esl.sum())
        print(
            f"-> Contemporary ESL/NES partition inside Moodle (N={len(post_ml)}): "
            f"ESL={n_matched}, NES={len(post_ml) - n_matched} "
            f"(external ESL folder had {len(esl_univ_df)} files)."
        )

        if n_matched == 0:
            esl_scored = esl_univ_df.dropna(subset=PHYSICAL_FEATURES).copy()
            esl_scored["Predicted_Prob_PostLLM"] = rf_full.predict_proba(
                esl_scored[PHYSICAL_FEATURES].to_numpy(dtype=float)
            )[:, 1]
            esl_scored["Language_Group"] = "ESL"
            esl_scored["Era"] = "Post-LLM (2025-2026)"
            post_fairness = pd.concat([post_ml, esl_scored], ignore_index=True)
        else:
            post_fairness = post_ml.copy()
            post_fairness["Era"] = "Post-LLM (2025-2026)"

        fdr_post = compare_two_groups(
            post_fairness[post_fairness["Language_Group"] == "ESL"],
            post_fairness[post_fairness["Language_Group"] == "NES"],
            PHYSICAL_FEATURES + ["Predicted_Prob_PostLLM"],
            label_a="Post_ESL",
            label_b="Post_NES",
        )
        fdr_post.to_csv(os.path.join(args.out_dir, "esl_vs_nes_postllm.csv"), index=False)
        print("\n=== CONTEMPORARY POST-LLM: ESL vs. NES (WITH REAL OOF PROBABILITIES) ===")
        print(fdr_post.to_markdown(index=False))

        pre_fairness = pre_ml.copy()
        pre_fairness["Era"] = "Pre-LLM Baseline"
        if "Language_Group" in pre_fairness.columns:
            pre_fairness["Predicted_Prob_PostLLM"] = rf_author_oof[: len(pre_ml)]
            fdr_pre = compare_two_groups(
                pre_fairness[pre_fairness["Language_Group"] == "ESL"],
                pre_fairness[pre_fairness["Language_Group"] == "NES"],
                PHYSICAL_FEATURES + ["Predicted_Prob_PostLLM"],
                label_a="Pre_ESL",
                label_b="Pre_NES",
            )
            fdr_pre.to_csv(os.path.join(args.out_dir, "esl_vs_nes_prellm.csv"), index=False)
            print("\n=== HISTORICAL PRE-LLM (MICUSP): ESL vs. NES ===")
            print(fdr_pre.to_markdown(index=False))

            for grp in ("ESL", "NES"):
                sub = pre_fairness[pre_fairness["Language_Group"] == grp]
                if not sub.empty:
                    fpr_g = (sub["Predicted_Prob_PostLLM"] >= 0.5).mean() * 100.0
                    print(f"  Pre-LLM RF Misclassification Rate ({grp}, n={len(sub)}): {fpr_g:.2f}%")

            esl_combined = pd.concat([pre_fairness, post_fairness], ignore_index=True)
            esl_combined.to_csv(
                os.path.join(args.out_dir, "esl_subgroup_unified_dataset.csv"), index=False
            )
            generate_esl_plots(esl_combined, post_fairness, args.out_dir)

    # STEP 6: SYNTHETIC AI VALIDATION & CLASSIFIER PROJECTION
    print("\n[6/6] Running Synthetic AI Corpus Summary & Classifier Projection...")
    if not ai_df.empty:
        ai_clean = ai_df.dropna(subset=PHYSICAL_FEATURES).copy()
        ai_clean["Predicted_Prob_PostLLM"] = rf_full.predict_proba(
            ai_clean[PHYSICAL_FEATURES].to_numpy(dtype=float)
        )[:, 1]

        counts = ai_clean["Model_Family"].value_counts().to_dict()
        print(f"-> Exact Synthetic AI Paper Counts by Model_Family: {counts}")

        ai_summary = []
        for f in PHYSICAL_FEATURES + ["Predicted_Prob_PostLLM"]:
            vals = ai_clean[f].dropna()
            ai_summary.append(
                {
                    "Metric": f,
                    "AI_Mean±SD": f"{vals.mean():.3f} ± {vals.std():.3f}",
                    "Median": round(vals.median(), 3),
                    "Min": round(vals.min(), 3),
                    "Max": round(vals.max(), 3),
                }
            )
        ai_sum_df = pd.DataFrame(ai_summary)
        ai_sum_df.to_csv(os.path.join(args.out_dir, "ai_overall_summary.csv"), index=False)
        print("\n=== SYNTHETIC AI CORPUS (OVERALL SUMMARY + RF PROJECTION) ===")
        print(ai_sum_df.to_markdown(index=False))

        per_model = (
            ai_clean.groupby("Model_Family")[PHYSICAL_FEATURES + ["Predicted_Prob_PostLLM"]]
            .agg(["count", "mean", "std"])
            .round(3)
        )
        per_model.to_csv(os.path.join(args.out_dir, "ai_per_model_breakdown.csv"))
        print("\n=== PER-MODEL FAMILY BREAKDOWN ===")
        print(
            ai_clean.groupby("Model_Family")[
                ["Sentence_Var", "Flesch_Ease", "Zipf_R2", "MATTR", "Predicted_Prob_PostLLM"]
            ]
            .agg(["count", "mean", "std"])
            .round(3)
            .to_markdown()
        )
        generate_ai_comparison_plots(ai_clean, moodle_df, micusp_df, args.out_dir)
    else:
        print("-> No synthetic AI dataset found; skipping Step 6.")

    print(f"\nAll outputs written to: {os.path.abspath(args.out_dir)}")
    print(f"Log saved to: {log_path}")
    sys.stdout = logger.terminal
    logger.close()


if __name__ == "__main__":
    main()
