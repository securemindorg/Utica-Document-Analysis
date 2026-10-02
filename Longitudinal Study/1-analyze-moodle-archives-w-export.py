import os
import re
import zipfile
import math
import glob
import hashlib
import sys
from datetime import datetime
from collections import Counter
import pandas as pd
from docx import Document
from pypdf import PdfReader

# ==========================================
# CONSTANTS & LEXICONS
# ==========================================
SALT = "UTICA_IRB_2026_SECURE_SALT"  # Prevents rainbow table/reversal attacks

AI_SIGNATURES = {
    'delve', 'testament', 'pivotal', 'foster', 'synergy', 'bespoke', 
    'demystify', 'furthermore', 'moreover', 'in conclusion', 'tapestry',
    'beacon', 'realm', 'multifaceted', 'underscore', 'crucial', 'imperative',
    'paramount', 'vibrant', 'intertwined', 'seamlessly'
}

SYNTHETIC_TRIGRAMS = [
    "in today's rapidly evolving",
    "in today's digital age",
    "it is important to remember that",
    "plays a crucial role in",
    "plays a pivotal role in",
    "it is worth noting that",
    "a testament to the",
    "delve into the world",
    "shed light on",
    "in the realm of",
    "serves as a reminder",
    "in conclusion, it can be",
    "it is essential to consider",
    "a wide range of",
    "plays an important role"
]

SEMESTER_MAP = {
    '01': ('Spring', 1),
    '05': ('Summer', 2),
    '10': ('Fall', 3)
}

# ==========================================
# DUAL-OUTPUT LOGGER (CONSOLE + TEXT FILE)
# ==========================================
class DualLogger:
    """Redirects print statements to both stdout and a designated text file."""
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

# ==========================================
# ANONYMIZATION & SCRUBBING PIPELINE
# ==========================================
def anonymize_student_id(raw_name):
    """Generates a secure, deterministic Hash ID (e.g., Sub_a3f9e12c) for a student name."""
    if not raw_name or raw_name == "Unknown Student":
        return "Sub_ANONYMOUS"
    salted_string = f"{SALT}_{raw_name.strip().lower()}"
    hash_object = hashlib.sha256(salted_string.encode())
    short_hash = hash_object.hexdigest()[:8]
    return f"Sub_{short_hash}"

def scrub_identifying_text(text):
    """Strips self-identifying headers, student IDs, email addresses, and author lines."""
    if not text:
        return ""
    
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL_REDACTED]', text)
    text = re.sub(r'\b(ID|Student ID|Banner ID|ID#)[\s:]*\d+\b', '[ID_REDACTED]', text, flags=re.I)
    text = re.sub(r'^(Name|Author|Student|Submitted by)[\s:]+.*$', '', text, flags=re.M | re.I)
    
    return text

# ==========================================
# MOODLE FILE NAME PARSER
# ==========================================
def parse_moodle_filename(filename):
    pattern = r"^([A-Z]{3})-(\d{3})-([A-Z0-9]+)-(\d{4})(\d{2})-(.+)$"
    match = re.match(pattern, filename)
    
    if match:
        program = match.group(1)
        course_num = match.group(2)
        section = match.group(3)
        year = int(match.group(4))
        month_code = match.group(5)
        assignment = match.group(6)
        
        sem_name, sem_order = SEMESTER_MAP.get(month_code, ('Unknown', 9))
        
        return {
            'program': program,
            'course': f"{program}-{course_num}",
            'section': section,
            'year': year,
            'semester': sem_name,
            'sem_order': sem_order,
            'assignment': assignment,
            'sort_key': (year, sem_order, program, course_num, section)
        }
    else:
        return {
            'program': 'UNKNOWN',
            'course': 'UNKNOWN',
            'section': 'UNKNOWN',
            'year': 9999,
            'semester': 'UNKNOWN',
            'sem_order': 9,
            'assignment': filename,
            'sort_key': (9999, 9, 'ZZZ', '999', 'ZZ')
        }

# ==========================================
# STYLOMETRIC ANALYTICS ENGINE
# ==========================================
def count_syllables(word):
    word = re.sub(r'[^a-z]', '', word.lower())
    if len(word) <= 3:
        return 1
    word = re.sub(r'(?:[^laeiouy]es|ed|[^laeiouy]e)$', '', word)
    word = re.sub(r'^y', '', word)
    matches = re.findall(r'[aeiouy]{1,2}', word)
    return len(matches) if matches else 1

def calculate_shannon_entropy(text):
    if not text:
        return 0.0
    total_chars = len(text)
    counts = Counter(text)
    entropy = 0.0
    for count in counts.values():
        p = count / total_chars
        entropy -= p * math.log2(p)
    return entropy

def calculate_mattr(words, window_size=50):
    if len(words) < window_size:
        return len(set(words)) / len(words) if words else 0.0
    sum_ttr = 0.0
    count = 0
    for i in range(len(words) - window_size + 1):
        window = words[i:i + window_size]
        sum_ttr += len(set(window)) / window_size
        count += 1
    return sum_ttr / count

def calculate_zipf_fit(words):
    if len(words) < 30:
        return 0.95
    freqs = sorted(Counter(words).values(), reverse=True)
    if len(freqs) < 5:
        return 0.95
    
    n = min(len(freqs), 50)
    sum_x = sum_y = sum_xy = sum_x2 = sum_y2 = 0.0
    for i in range(n):
        x = math.log(i + 1)
        y = math.log(freqs[i])
        sum_x += x
        sum_y += y
        sum_xy += x * y
        sum_x2 += x * x
        sum_y2 += y * y
        
    num = (n * sum_xy) - (sum_x * sum_y)
    den = math.sqrt(((n * sum_x2) - (sum_x ** 2)) * ((n * sum_y2) - (sum_y ** 2)))
    if den == 0:
        return 0.95
    r = num / den
    return min(1.0, max(0.0, r * r))

def calculate_vocabulary_burstiness(raw_text):
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
        mean = sum(freqs) / len(freqs)
        variance = sum((x - mean) ** 2 for x in freqs) / len(freqs)
        total_dispersion += (variance / mean) if mean > 0 else 0.0

    avg_dispersion = total_dispersion / len(top_words)
    return round(avg_dispersion, 2), (avg_dispersion < 0.35)

def calculate_passive_voice(sentences):
    if not sentences:
        return 0, 0.0
    passive_regex = re.compile(r'\b(am|is|are|was|were|be|been|being)\b\s+([\w]+ed|conducted|shown|given|made|found|taken|seen|built|chosen|driven|written|known|done)\b', re.I)
    count = sum(1 for s in sentences if passive_regex.search(s))
    ratio = (count / len(sentences)) * 100
    return count, round(ratio, 1)

def calculate_paragraph_symmetry(raw_text):
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', raw_text) if p.strip()]
    if len(paragraphs) < 3:
        return len(paragraphs), 0, 1.0, False

    word_counts = [len(p.split()) for p in paragraphs]
    mean = sum(word_counts) / len(word_counts)
    variance = sum((x - mean) ** 2 for x in word_counts) / len(word_counts)
    std_dev = math.sqrt(variance)
    cv = (std_dev / mean) if mean > 0 else 1.0

    return len(paragraphs), round(std_dev, 1), round(cv, 2), (cv < 0.22)

def deep_artifact_check(raw_text, clean_words, sentences):
    penalty = 0
    flags = []

    zw_matches = len(re.findall(r'[\u200B-\u200D\uFEFF]', raw_text))
    if zw_matches > 0:
        penalty += 20
        flags.append(f"Zero-Width Unicode ({zw_matches})")

    nbsp_matches = len(re.findall(r'\u00A0', raw_text))
    if nbsp_matches > 5:
        penalty += 10
        flags.append(f"Web Clipboard Spaces ({nbsp_matches})")

    lower_text = raw_text.lower()
    found_trigrams = [phrase for phrase in SYNTHETIC_TRIGRAMS if phrase in lower_text]
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

    found_buzzwords = list(set([w for w in clean_words if w in AI_SIGNATURES]))
    if len(found_buzzwords) >= 2:
        penalty += min(20, len(found_buzzwords) * 5)
        flags.append(f"AI Vocabulary ({', '.join(found_buzzwords[:3])})")

    _, _, para_cv, is_symmetrical = calculate_paragraph_symmetry(raw_text)
    if is_symmetrical:
        penalty += 15
        flags.append(f"Unnatural Para Symmetry (CV={para_cv})")

    dispersion, is_uniform = calculate_vocabulary_burstiness(raw_text)
    if is_uniform:
        penalty += 15
        flags.append(f"Low Vocab Burstiness ({dispersion})")

    return penalty, flags

def analyze_document(text, subject_hash_id, doc_id):
    clean_text = scrub_identifying_text(text)
    
    sentences = [s.strip() for s in re.split(r'[.!?]+', clean_text) if s.strip()]
    words = [re.sub(r'[^\w]', '', w).lower() for w in clean_text.split() if re.sub(r'[^\w]', '', w)]

    total_words = len(words)
    total_sentences = len(sentences)

    if total_words < 30 or total_sentences < 2:
        return None

    entropy = calculate_shannon_entropy(clean_text)
    mattr = calculate_mattr(words, 50)
    zipf_fit = calculate_zipf_fit(words)
    _, passive_ratio = calculate_passive_voice(sentences)

    lengths = [len(s.split()) for s in sentences]
    mean_len = sum(lengths) / total_sentences
    variance = sum((x - mean_len) ** 2 for x in lengths) / total_sentences

    v_score = min(25, (variance / 85) * 25)
    m_score = min(25, (mattr / 0.75) * 25)
    z_score = 15 if zipf_fit >= 0.85 else 5
    base_score = 25 + v_score + m_score + z_score

    penalty, flags = deep_artifact_check(clean_text, words, sentences)
    final_score = max(0, min(100, base_score - penalty))

    if final_score > 75 and len(flags) == 0:
        verdict = "Human"
    elif final_score >= 55:
        verdict = "Review"
    elif final_score >= 40:
        verdict = "Hybrid"
    else:
        verdict = "Bot/AI"

    total_syllables = sum(count_syllables(w) for w in words)
    flesch_ease = max(0, min(100, 206.835 - (1.015 * (total_words / total_sentences)) - (84.6 * (total_syllables / total_words))))
    grade_level = max(0, (0.39 * (total_words / total_sentences)) + (11.8 * (total_syllables / total_words)) - 15.59)

    nominalizations = [w for w in words if (w.endswith('tion') or w.endswith('ment') or w.endswith('ance') or w.endswith('ence')) and len(w) > 5]
    nom_ratio = (len(nominalizations) / total_words) * 100

    return {
        'Subject_ID': subject_hash_id,
        'Doc_ID': doc_id,
        'Auth_Score': round(final_score, 1),
        'Verdict': verdict,
        'Entropy': round(entropy, 2),
        'MATTR': round(mattr, 2),
        'Zipf_R2': round(zipf_fit, 2),
        'Sentence_Var': round(variance, 1),
        'Flesch_Ease': round(flesch_ease, 1),
        'Grade_Level': round(grade_level, 1),
        'Nominal_Ratio_%': round(nom_ratio, 1),
        'Passive_%': round(passive_ratio, 1),
        'Flags': ", ".join(flags) if flags else "None"
    }

# ==========================================
# FILE EXTRACTION HELPERS
# ==========================================
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
# MAIN EXECUTION ENGINE
# ==========================================
def process_moodle_directory(directory_path, output_filename=None):
    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"analysis_report_{timestamp}.txt"

    # Hook up logger to split output to stdout + text file
    logger = DualLogger(output_filename)
    sys.stdout = logger

    zip_files = glob.glob(os.path.join(directory_path, "*.zip"))
    
    if not zip_files:
        print(f"No ZIP archives found in '{directory_path}'.")
        sys.stdout = logger.terminal
        logger.close()
        return

    archives = []
    for zpath in zip_files:
        fname = os.path.basename(zpath)
        meta = parse_moodle_filename(fname)
        meta['full_path'] = zpath
        archives.append(meta)

    archives.sort(key=lambda x: x['sort_key'])

    print(f"================================================================================")
    print(f"MOODLE ARCHIVE DE-IDENTIFIED ANALYSIS REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output Saved To: {output_filename}")
    print(f"================================================================================")

    doc_counter = 1

    for arch in archives:
        print("\n" + "=" * 80)
        print(f"COURSE: {arch['course']} ({arch['section']}) | TERM: {arch['semester']} {arch['year']}")
        print(f"ASSIGNMENT: {arch['assignment']}")
        print("=" * 80)

        results = []
        with zipfile.ZipFile(arch['full_path'], 'r') as z:
            for zip_info in z.infolist():
                if zip_info.is_dir() or '__MACOSX' in zip_info.filename or '/.' in zip_info.filename:
                    continue
                
                ext = zip_info.filename.split('.')[-1].lower()
                if ext in ['txt', 'docx', 'pdf']:
                    raw_student_name = parse_moodle_student_name(zip_info.filename)
                    subject_hash_id = anonymize_student_id(raw_student_name)
                    doc_id = f"Doc_{doc_counter:04d}.{ext}"
                    doc_counter += 1
                    
                    with z.open(zip_info) as f:
                        text = extract_text_from_stream(f, ext)
                        if text:
                            res = analyze_document(text, subject_hash_id, doc_id)
                            if res:
                                results.append(res)

        if results:
            df = pd.DataFrame(results)
            print(df.to_markdown(index=False))
            print(f"\nSummary: {len(df)} submissions evaluated. Mean Authenticity Score: {df['Auth_Score'].mean():.1f}%\n")
        else:
            print("No valid or readable submissions (.docx, .pdf, .txt > 30 words) found in this ZIP archive.\n")

    print(f"\nAnalysis complete. Full report written to {output_filename}")
    
    # Restore stdout and close file handle
    sys.stdout = logger.terminal
    logger.close()

if __name__ == "__main__":
    TARGET_DIRECTORY = "./moodle_exports" 
    
    if os.path.exists(TARGET_DIRECTORY):
        process_moodle_directory(TARGET_DIRECTORY)
    else:
        print(f"Directory '{TARGET_DIRECTORY}' does not exist. Please create '{TARGET_DIRECTORY}' and place your ZIP archives inside.")
