import os
import re
import zipfile
import math
import glob
import hashlib
import sys
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from docx import Document
from pypdf import PdfReader

# Ensure required NLTK tokenizers are available
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
SALT = "UTICA_IRB_2026_SECURE_SALT"
SEMESTER_MAP = {'01': ('Spring', 1), '05': ('Summer', 2), '10': ('Fall', 3)}

# ==========================================
# UNIFIED STYLOMETRIC EXTRACTOR ENGINE
# ==========================================
class UnifiedStylometricExtractor:
    """
    Unified feature extraction engine.
    Applies identical NLTK-based tokenization, MATTR (W=100), Zipf R^2,
    and artifact detection metrics across both MICUSP and Moodle corpora.
    """
    def __init__(self, mattr_window=100):
        self.mattr_window = mattr_window
        self.ai_signatures = {
            'delve', 'testament', 'pivotal', 'foster', 'synergy', 'bespoke', 
            'demystify', 'furthermore', 'moreover', 'in conclusion', 'tapestry',
            'beacon', 'realm', 'multifaceted', 'underscore', 'crucial', 'imperative',
            'paramount', 'vibrant', 'intertwined', 'seamlessly'
        }
        self.synthetic_trigrams = [
            "in today's rapidly evolving", "in today's digital age",
            "it is important to remember that", "plays a crucial role in",
            "plays a pivotal role in", "it is worth noting that",
            "a testament to the", "delve into the world", "shed light on",
            "in the realm of", "serves as a reminder", "in conclusion, it can be",
            "it is essential to consider", "a wide range of", "plays an important role"
        ]

    def calculate_mattr(self, alpha_tokens):
        """Calculates Moving Average Type-Token Ratio with standardized window size."""
        if len(alpha_tokens) < self.mattr_window:
            return float(len(set(alpha_tokens)) / len(alpha_tokens)) if alpha_tokens else 0.0
        ttrs = [
            len(set(alpha_tokens[i : i + self.mattr_window])) / float(self.mattr_window)
            for i in range(len(alpha_tokens) - self.mattr_window + 1)
        ]
        return float(np.mean(ttrs))

    def calculate_zipf_r2(self, alpha_tokens):
        """Calculates Zipf's Law R^2 score via linear regression."""
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
        """Estimates passive voice ratio using standard auxiliary verb patterns."""
        if not sentences:
            return 0, 0.0
        passive_regex = re.compile(
            r'\b(am|is|are|was|were|be|been|being)\b\s+([\w]+ed|conducted|shown|given|made|found|taken|seen|built|chosen|driven|written|known|done)\b',
            re.I
        )
        count = sum(1 for s in sentences if passive_regex.search(s))
        ratio = (count / len(sentences)) * 100.0
        return count, round(ratio, 1)

    def calculate_paragraph_symmetry(self, raw_text):
        """Evaluates paragraph word length coefficient of variation (CV)."""
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', raw_text) if p.strip()]
        if len(paragraphs) < 3:
            return len(paragraphs), 0.0, 1.0, False

        word_counts = [len(p.split()) for p in paragraphs]
        mean = sum(word_counts) / float(len(word_counts))
        variance = sum((x - mean) ** 2 for x in word_counts) / float(len(word_counts))
        std_dev = math.sqrt(variance)
        cv = (std_dev / mean) if mean > 0 else 1.0

        return len(paragraphs), round(std_dev, 1), round(cv, 2), (cv < 0.22)

    def calculate_vocabulary_burstiness(self, raw_text):
        """Measures word dispersion variance across document paragraphs."""
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', raw_text) if p.strip()]
        if len(paragraphs) < 3:
            return 1.0, False

        stop_words = {'the','is','at','which','on','a','an','and','or','to','in','of','for','with','be','has','been','it','by','that','this','are','was','as','from','have','had','do','does','did','but','not','what','all','were','when','we','there','can','more','her','will','also','about','if','out','so','into','than','them','some','could'}
        
        para_counts = []
        global_counts = Counter()
        for p in paragraphs:
            p_words = [re.sub(r'[^\w]', '', w.lower()) for w in p.split() if re.sub(r'[^\w]', '', w.lower()) and re.sub(r'[^\w]', '', w.lower()) not in stop_words]
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

    def count_syllables(self, word):
        word = re.sub(r'[^a-z]', '', word.lower())
        if len(word) <= 3:
            return 1
        word = re.sub(r'(?:[^laeiouy]es|ed|[^laeiouy]e)$', '', word)
        word = re.sub(r'^y', '', word)
        matches = re.findall(r'[aeiouy]{1,2}', word)
        return len(matches) if matches else 1

    def deep_artifact_check(self, raw_text, alpha_words, sentences):
        """Scans document for technical, structural, and lexicon artifacts."""
        penalty = 0
        flags = []

        # Unicode/Formatting artifacts
        zw_matches = len(re.findall(r'[\u200B-\u200D\uFEFF]', raw_text))
        if zw_matches > 0:
            penalty += 20
            flags.append(f"Zero-Width Unicode ({zw_matches})")

        nbsp_matches = len(re.findall(r'\u00A0', raw_text))
        if nbsp_matches > 5:
            penalty += 10
            flags.append(f"Web Clipboard Spaces ({nbsp_matches})")

        # Synthetic phrases & citations
        lower_text = raw_text.lower()
        found_trigrams = [phrase for phrase in self.synthetic_trigrams if phrase in lower_text]
        if found_trigrams:
            penalty += min(25, len(found_trigrams) * 8)
            flags.append(f"Synthetic Trigrams ({len(found_trigrams)})")

        arxiv_finds = re.findall(r'(arxiv:\s*\d{4}\.\d{4,5}|arxiv\.org\/abs\/\d{4}\.\d{4,5}|doi\.org\/10\.\d{4})', raw_text, re.I)
        if arxiv_finds:
            penalty += 25
            flags.append(f"Synthetic Citation ({len(arxiv_finds)})")

        header_matches = re.findall(r'^(Concern|Introduction|The Concern|References):', raw_text, re.M)
        if header_matches:
            penalty += 10
            flags.append("LLM Section Headers")

        # AI Signatures & Structural Uniformity
        found_buzzwords = list(set([w for w in alpha_words if w in self.ai_signatures]))
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

    def analyze_text(self, raw_text):
        """Unified analysis pipeline for a plain text string."""
        if not raw_text or len(raw_text.strip()) < 50:
            return None

        # Redact self-identifying header text
        clean_text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL_REDACTED]', raw_text)
        clean_text = re.sub(r'\b(ID|Student ID|Banner ID|ID#)[\s:]*\d+\b', '[ID_REDACTED]', clean_text, flags=re.I)
        clean_text = re.sub(r'^(Name|Author|Student|Submitted by)[\s:]+.*$', '', clean_text, flags=re.M | re.I)

        sentences = sent_tokenize(clean_text)
        all_tokens = word_tokenize(clean_text)
        alpha_words = [w.lower() for w in all_tokens if w.isalpha()]

        total_words = len(alpha_words)
        total_sentences = len(sentences)

        if total_words < 30 or total_sentences < 2:
            return None

        # 1. Core Stylometric Metrics
        sent_lengths = [len([w for w in word_tokenize(s) if w.isalpha()]) for s in sentences]
        sentence_variance = float(np.var(sent_lengths)) if len(sent_lengths) > 1 else 0.0

        mattr = self.calculate_mattr(alpha_words)
        zipf_fit = self.calculate_zipf_r2(alpha_words)

        char_counts = Counter(clean_text)
        total_chars = len(clean_text)
        entropy = -sum((cnt / total_chars) * math.log2(cnt / total_chars) for cnt in char_counts.values()) if total_chars > 0 else 0.0

        _, passive_ratio = self.calculate_passive_voice(sentences)

        total_syllables = sum(self.count_syllables(w) for w in alpha_words)
        flesch_ease = max(0.0, min(100.0, 206.835 - (1.015 * (total_words / total_sentences)) - (84.6 * (total_syllables / total_words))))
        grade_level = max(0.0, (0.39 * (total_words / total_sentences)) + (11.8 * (total_syllables / total_words)) - 15.59)

        nominalizations = [w for w in alpha_words if (w.endswith('tion') or w.endswith('ment') or w.endswith('ance') or w.endswith('ence')) and len(w) > 5]
        nom_ratio = (len(nominalizations) / total_words) * 100.0

        # 2. Heuristic Artifact Penalty & Authenticity Score
        v_score = min(25.0, (sentence_variance / 85.0) * 25.0)
        m_score = min(25.0, (mattr / 0.75) * 25.0)
        z_score = 15.0 if zipf_fit >= 0.85 else 5.0
        base_score = 25.0 + v_score + m_score + z_score

        penalty, flags = self.deep_artifact_check(clean_text, alpha_words, sentences)
        final_score = max(0.0, min(100.0, base_score - penalty))

        if final_score > 75 and len(flags) == 0:
            verdict = "Human"
        elif final_score >= 55:
            verdict = "Review"
        elif final_score >= 40:
            verdict = "Hybrid"
        else:
            verdict = "Bot/AI"

        return {
            'Word_Count': total_words,
            'Sentence_Count': total_sentences,
            'Auth_Score': round(final_score, 1),
            'Verdict': verdict,
            'Entropy': round(entropy, 2),
            'MATTR': round(mattr, 4),
            'Zipf_R2': round(zipf_fit, 4),
            'Sentence_Var': round(sentence_variance, 2),
            'Flesch_Ease': round(flesch_ease, 1),
            'Grade_Level': round(grade_level, 1),
            'Nominal_Ratio_%': round(nom_ratio, 1),
            'Passive_%': round(passive_ratio, 1),
            'Flag_Count': len(flags),
            'Flags': ", ".join(flags) if flags else "None"
        }

# ==========================================
# FILE & METADATA PARSERS
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
        sem_name, sem_order = SEMESTER_MAP.get(month_code, ('Unknown', 9))
        return {
            'Program': program,
            'Course': f"{program}-{course_num}",
            'Section': section,
            'Year': int(year),
            'Semester': sem_name,
            'Sem_Order': sem_order,
            'Assignment': assignment
        }
    
    # Fallback parser for non-standard archive names
    year_match = re.search(r'\b(202[0-9])\b', filename)
    extracted_year = int(year_match.group(1)) if year_match else 2025
    sem_name = 'Spring' if '01' in filename or 'Spring' in filename else ('Fall' if '10' in filename or 'Fall' in filename else 'Unassigned')
    sem_order = 1 if sem_name == 'Spring' else (3 if sem_name == 'Fall' else 9)
    
    return {
        'Program': filename.split('-')[0] if '-' in filename else 'GEN',
        'Course': filename.split('-')[0] if '-' in filename else 'GEN-100',
        'Section': '01',
        'Year': extracted_year,
        'Semester': sem_name,
        'Sem_Order': sem_order,
        'Assignment': filename
    }

def extract_text_from_stream(stream, ext):
    try:
        if ext == 'txt':
            return stream.read().decode('utf-8', errors='ignore')
        elif ext == 'docx':
            doc = Document(stream)
            return "\n".join([p.text for p in doc.paragraphs])
        elif ext == 'pdf':
            reader = PdfReader(stream)
            return "\n".join([page.extract_text() or '' for page in reader.pages])
    except Exception:
        return ""
    return ""

def parse_moodle_student_name(path):
    parts = path.split('/')
    if len(parts) > 1:
        folder = parts[0]
        return folder.split('_')[0].strip()
    return "Unknown Student"

# ==========================================
# PROCESSING PIPELINES
# ==========================================
def process_moodle_exports(moodle_dir, extractor):
    """Processes Moodle ZIP export directory."""
    zip_files = glob.glob(os.path.join(moodle_dir, "*.zip"))
    if not zip_files:
        print(f"No ZIP archives found in directory: '{moodle_dir}'")
        return pd.DataFrame()

    records = []
    doc_counter = 1

    for zpath in zip_files:
        meta = parse_moodle_filename(os.path.basename(zpath))
        with zipfile.ZipFile(zpath, 'r') as z:
            for zip_info in z.infolist():
                if zip_info.is_dir() or '__MACOSX' in zip_info.filename or '/.' in zip_info.filename:
                    continue
                ext = zip_info.filename.split('.')[-1].lower()
                if ext in ['txt', 'docx', 'pdf']:
                    raw_student_name = parse_moodle_student_name(zip_info.filename)
                    subject_id = anonymize_student_id(raw_student_name)
                    doc_id = f"Doc_{doc_counter:04d}.{ext}"
                    doc_counter += 1

                    with z.open(zip_info) as f:
                        text = extract_text_from_stream(f, ext)
                        res = extractor.analyze_text(text)
                        if res:
                            res.update({
                                'Corpus': 'Moodle_PostLLM',
                                'Doc_ID': doc_id,
                                'Subject_ID': subject_id,
                                'Course': meta['Course'],
                                'Year': meta['Year'],
                                'Semester': meta['Semester'],
                                'Assignment': meta['Assignment']
                            })
                            records.append(res)

    return pd.DataFrame(records)

def process_micusp_pdfs(micusp_dir, metadata_csv, extractor):
    """Processes MICUSP baseline PDF directory and merges metadata."""
    pdf_files = glob.glob(os.path.join(micusp_dir, "*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in baseline directory: '{micusp_dir}'")
        return pd.DataFrame()

    meta_df = pd.read_csv(metadata_csv) if os.path.exists(metadata_csv) else pd.DataFrame()
    records = []

    for pdf_path in pdf_files:
        paper_id = os.path.splitext(os.path.basename(pdf_path))[0]
        try:
            reader = PdfReader(pdf_path)
            extracted_text = " ".join([page.extract_text() or "" for page in reader.pages])
            res = extractor.analyze_text(extracted_text)
            if res:
                res.update({
                    'Corpus': 'MICUSP_PreLLM',
                    'PAPER ID': paper_id,
                    'Subject_ID': f"MICUSP_{paper_id}",
                    'Year': 2008,
                    'Semester': 'Pre-LLM Baseline'
                })
                records.append(res)
        except Exception as e:
            print(f"Error parsing MICUSP PDF {paper_id}: {e}")

    df = pd.DataFrame(records)
    if not df.empty and not meta_df.empty and 'PAPER ID' in meta_df.columns:
        df = pd.merge(meta_df, df, on="PAPER ID", how="inner")
    return df

def calculate_cohens_d(group1, group2):
    """Calculates Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    s_pooled = math.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / s_pooled if s_pooled > 0 else 0.0

# ==========================================
# MAIN EXECUTION ROUTINE
# ==========================================
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"unified_analysis_report_{timestamp}.txt"
    logger = DualLogger(log_file)
    sys.stdout = logger

    print("================================================================================")
    print("UNIFIED STYLOMETRIC ANALYSIS REPORT (PRE-LLM MICUSP VS. POST-LLM MOODLE)")
    print(f"Execution Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("================================================================================")

    extractor = UnifiedStylometricExtractor(mattr_window=100)

    # 1. Process Datasets
    print("\n[1/3] Processing Moodle Exports...")
    moodle_df = process_moodle_exports("./moodle_exports", extractor)
    if not moodle_df.empty:
        moodle_df.to_csv("moodle_extracted_dataset.csv", index=False)
        print(f"-> Successfully processed {len(moodle_df)} Moodle submissions.")

    print("\n[2/3] Processing MICUSP Baseline PDFs...")
    micusp_df = process_micusp_pdfs("./micusp_pdfs", "micusp_papers.csv", extractor)
    if not micusp_df.empty:
        micusp_df.to_csv("micusp_extracted_baseline.csv", index=False)
        print(f"-> Successfully processed {len(micusp_df)} MICUSP baseline papers.")

    # 2. Comparative Analysis Report
    print("\n[3/3] Generating Unified Comparative Report...")
    if not moodle_df.empty and not micusp_df.empty:
        metrics = ['Sentence_Var', 'MATTR', 'Zipf_R2', 'Flesch_Ease', 'Auth_Score', 'Flag_Count']
        summary_rows = []

        for m in metrics:
            micusp_vals = micusp_df[m].dropna()
            moodle_vals = moodle_df[m].dropna()

            d_val = calculate_cohens_d(moodle_vals, micusp_vals)
            t_stat, p_val = stats.ttest_ind(moodle_vals, micusp_vals, equal_var=False)

            summary_rows.append({
                'Metric': m,
                'MICUSP Baseline (Mean ± SD)': f"{micusp_vals.mean():.3f} ± {micusp_vals.std():.3f}",
                'Moodle Cohort (Mean ± SD)': f"{moodle_vals.mean():.3f} ± {moodle_vals.std():.3f}",
                'Cohen\'s d': f"{d_val:+.3f}",
                'p-value': f"{p_val:.4e}" if p_val < 0.001 else f"{p_val:.3f}"
            })

        comp_df = pd.DataFrame(summary_rows)
        print("\n=== UNIFIED COMPARATIVE STYLOMETRIC DRIFT TABLE ===")
        print(comp_df.to_markdown(index=False))

        # False Positive Rate Calculation on Pre-LLM Baseline
        fp_count = len(micusp_df[micusp_df['Verdict'] != 'Human'])
        fpr = (fp_count / len(micusp_df)) * 100.0
        print(f"\nEmpirical Baseline False Positive Rate (FPR) on MICUSP: {fpr:.2f}% ({fp_count}/{len(micusp_df)} papers flagged)")

    print(f"\nExecution finished. Output report logged to: {log_file}")
    sys.stdout = logger.terminal
    logger.close()

if __name__ == "__main__":
    main()
