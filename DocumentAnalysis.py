import os
import re
import string
import math
import pandas as pd

# Target specific lexical patterns heavily over-indexed by modern LLMs
AI_SIGNATURES = {
    'delve', 'testament', 'pivotal', 'foster', 'synergy', 'bespoke', 
    'demystify', 'furthermore', 'moreover', 'in conclusion', 'tapestry',
    'beacon', 'realm', 'multifaceted', 'underscore'
}

def calculate_shannon_entropy(text):
    """Calculates character-level Shannon Entropy (retained for baseline profile)."""
    if not text:
        return 0.0
    total_chars = len(text)
    from collections import Counter
    char_counts = Counter(text)
    entropy = 0.0
    for count in char_counts.values():
        probability = count / total_chars
        entropy -= probability * math.log2(probability)
    return entropy

def deep_artifact_check(text):
    """Scans for exact structural patterns, fake references, and lexical habits common to LLM fabrications."""
    penalty = 0
    flags = []
    
    # 1. Catch Academic/arXiv/DOI link hallucination trends
    arxiv_finds = re.findall(r'(arxiv:\s*\d{4}\.\d{4,5}|arxiv\.org\/abs\/\d{4}\.\d{4,5}|doi\.org\/10\.\d{4})', text.lower())
    if arxiv_finds:
        penalty += 25
        flags.append(f"Synthetic Citation Pattern ({len(arxiv_finds)} found)")
        
    # 2. Check for rigid structural headers common in prompt outputs
    headers = re.findall(r'^(Concern|Introduction|The Concern|References):', text, re.MULTILINE)
    if headers:
        penalty += 10
        flags.append(f"LLM Section Headers: {list(set(headers))}")

    # 3. Vocabulary Tells
    words = [w.strip(string.punctuation).lower() for w in text.split()]
    found_buzzwords = [w for w in words if w in AI_SIGNATURES]
    if len(found_buzzwords) >= 2:
        penalty += min(20, len(found_buzzwords) * 5)
        flags.append(f"AI Vocabulary Overuse: {list(set(found_buzzwords))}")
        
    return penalty, flags

def analyze_document(text):
    """Computes all requested metrics and assigns an absolute human likelihood score."""
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    words = [w.strip(string.punctuation).lower() for w in text.split() if w.strip(string.punctuation)]
    
    total_words = len(words)
    total_sentences = len(sentences)
    
    if total_words < 30 or total_sentences < 2:
        return None
        
    # 1. Base Shannon Character Entropy
    entropy = calculate_shannon_entropy(text)
    
    # 2. Lexical Diversity (Type-Token Ratio)
    lex_div = len(set(words)) / total_words
    
    # 3. Sentence Length Variance
    lengths = [len(s.split()) for s in sentences]
    mean_len = sum(lengths) / total_sentences
    variance = sum((x - mean_len) ** 2 for x in lengths) / total_sentences
    
    # Generate human authenticity baseline score
    # High variance (irregular rhythms) and rich vocabulary heavily boost human score
    v_score = min(35, (variance / 45) * 35)
    d_score = min(35, (lex_div / 0.65) * 35)
    base_score = 35 + v_score + d_score
    
    # Apply the artifact/hallucination scan penalties
    penalty, flags = deep_artifact_check(text)
    final_score = max(0, min(100, base_score - penalty))
    
    # Classification logic based on objective thresholds
    if final_score > 75 and not flags:
        verdict = "Highly Likely Human"
    elif final_score >= 55:
        verdict = "Likely Human (Review Manually)"
    elif 40 <= final_score < 55:
        verdict = "Suspected AI Hybrid / Style Masked"
    else:
        verdict = "Highly Likely AI / Bot"
        
    return {
        "Entropy": entropy,
        "Lex_Div": lex_div,
        "Sent_Var": variance,
        "Score": final_score,
        "Verdict": verdict,
        "Flags": ", ".join(flags) if flags else "None"
    }

def run_full_analysis(directory_path):
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' not found.")
        return

    results = []
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().strip()
                
                metrics = analyze_document(content)
                if metrics is None:
                    continue
                    
                metrics["File Name"] = filename
                results.append(metrics)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    if not results:
        print("No valid text files found to analyze.")
        return

    df = pd.DataFrame(results)
    
    # Order files logically by human likelihood from highest down to lowest
    df = df.sort_values(by="Score", ascending=False)
    
    # Output formatting
    print(f"\n{'File Name':<16} | {'Entropy':<7} | {'Lex Div':<7} | {'Sent Var':<8} | {'Human %':<7} | {'Final Assessment':<32} | {'Flags Triggered'}")
    print("-" * 125)
    for _, row in df.iterrows():
        print(f"{row['File Name']:<16} | {row['Entropy']:<7.2f} | {row['Lex_Div']:<7.2f} | {row['Sent_Var']:<8.1f} | {row['Score']:<6.1f}% | {row['Verdict']:<32} | {row['Flags']}")

if __name__ == "__main__":
    # Point this to your folder containing the .txt files
    target_directory = "./"
    run_full_analysis(target_directory)
