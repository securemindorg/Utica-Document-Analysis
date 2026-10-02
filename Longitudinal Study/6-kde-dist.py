import pandas as pd

# Check columns of micusp_extracted_baseline.csv
df_pre = pd.read_csv('micusp_extracted_baseline.csv')
print("df_pre columns:", df_pre.columns.tolist())

# Check contents of analysis_report_20260930_191734.txt
with open('analysis_report_20260930_191734.txt') as f:
    text = f.read()

# Let's inspect unique terms/courses in analysis_report
import re
courses_terms = re.findall(r'COURSE:\s*(.*?)\s*\|\s*TERM:\s*(.*?)\nASSIGNMENT:\s*(.*?)\n', text)
print(f"Found {len(courses_terms)} assignment sections in analysis_report.")
for ct in courses_terms[:10]:
    print(ct)

# Check columns present in the markdown tables of analysis_report
table_lines = [l for l in text.split('\n') if l.startswith('|') and 'Subject_ID' in l]
print("Table header sample:", table_lines[0] if table_lines else "None")


import numpy as np

# Let's extract post-LLM data from analysis_report_20260930_191734.txt grouped by term
with open('analysis_report_20260930_191734.txt', 'r') as f:
    raw_report = f.read()

# Parse sections and table rows with term metadata
sections = raw_report.split('================================================================================')

parsed_data = []
current_course = ""
current_term = ""
current_assignment = ""

for sec in sections:
    lines = [l.strip() for l in sec.strip().split('\n') if l.strip()]
    if not lines:
        continue
    # Check if header line
    header_match = [l for l in lines if 'COURSE:' in l and 'TERM:' in l]
    if header_match:
        parts = header_match[0].split('|')
        current_course = parts[0].replace('COURSE:', '').strip()
        current_term = parts[1].replace('TERM:', '').strip()
        
        assign_line = [l for l in lines if 'ASSIGNMENT:' in l]
        if assign_line:
            current_assignment = assign_line[0].replace('ASSIGNMENT:', '').strip()
            
    # Table rows
    for l in lines:
        if l.startswith('|') and not 'Subject_ID' in l and not ':---' in l:
            row_parts = [p.strip() for p in l.split('|')[1:-1]]
            if len(row_parts) == 13:
                parsed_data.append({
                    'Course': current_course,
                    'Term': current_term,
                    'Assignment': current_assignment,
                    'Subject_ID': row_parts[0],
                    'Doc_ID': row_parts[1],
                    'Auth_Score': float(row_parts[2]),
                    'Verdict': row_parts[3],
                    'Entropy': float(row_parts[4]),
                    'MATTR': float(row_parts[5]),
                    'Zipf_R2': float(row_parts[6]),
                    'Sentence_Var': float(row_parts[7]),
                    'Flesch_Ease': float(row_parts[8]),
                    'Grade_Level': float(row_parts[9]),
                    'Nominal_Ratio_%': float(row_parts[10]),
                    'Passive_%': float(row_parts[11]),
                    'Flags': row_parts[12]
                })

df_post_full = pd.DataFrame(parsed_data)
print("Total parsed post-LLM records:", len(df_post_full))
print("\nBreakdown by Term:")
print(df_post_full['Term'].value_counts())

# Calculate Cohen's d between Pre-LLM and Post-LLM
def cohens_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1)**2 + (ny - 1) * np.std(y, ddof=1)**2) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std

# Compare Sentence_Var, MATTR, Zipf_R2, Flesch_Ease
for metric in ['sentence_variance', 'mattr', 'zipf_r2', 'flesch_reading_ease']:
    post_col = 'Sentence_Var' if metric == 'sentence_variance' else ('MATTR' if metric == 'mattr' else ('Zipf_R2' if metric == 'zipf_r2' else 'Flesch_Ease'))
    pre_vals = df_pre[metric].dropna()
    post_vals = df_post_full[post_col].dropna()
    d = cohens_d(post_vals, pre_vals) # post vs pre
    print(f"Metric: {metric:20s} | Pre: {np.mean(pre_vals):.2f} (sd {np.std(pre_vals, ddof=1):.2f}) | Post: {np.mean(post_vals):.2f} (sd {np.std(post_vals, ddof=1):.2f}) | Cohen's d: {d:.3f}")

import os

# Check files in current working directory
files = os.listdir('.')
print("Files in workspace:", files[:30])

# Check flags frequencies in df_post_full
print("\nFlag frequencies in post-LLM corpus:")
all_flags = df_post_full['Flags'].value_counts()
print(all_flags)

df_micusp = pd.read_csv('micusp_papers.csv')
print("df_micusp columns:", df_micusp.columns.tolist())
print(df_micusp.head(2))

import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid", palette="muted")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Plot 1: Sentence Variance KDE Shift
sns.kdeplot(df_pre['sentence_variance'], ax=axes[0], fill=True, label='Pre-LLM Baseline (MICUSP)', color='#1f77b4', alpha=0.4, log_scale=True)
sns.kdeplot(df_post_full['Sentence_Var'], ax=axes[0], fill=True, label='Post-LLM Cohort (2025-2026)', color='#ff7f0e', alpha=0.4, log_scale=True)
axes[0].set_title("Shift in Sentence Length Variance (Log Scale)")
axes[0].set_xlabel("Sentence Variance ($\sigma^2$)")
axes[0].set_ylabel("Density")
axes[0].legend()

# Plot 2: MATTR KDE Shift
sns.kdeplot(df_pre['mattr'], ax=axes[1], fill=True, label='Pre-LLM Baseline (MICUSP)', color='#1f77b4', alpha=0.4)
sns.kdeplot(df_post_full['MATTR'], ax=axes[1], fill=True, label='Post-LLM Cohort (2025-2026)', color='#ff7f0e', alpha=0.4)
axes[1].set_title("Shift in MATTR (Moving-Average Type-Token Ratio)")
axes[1].set_xlabel("MATTR (Window Size = 100)")
axes[1].set_ylabel("Density")
axes[1].legend()

plt.tight_layout()
plt.savefig('kde_distribution_shift.png', dpi=300)
plt.close()

print("kde_distribution_shift.png generated successfully.")

# Table 1: Pre-LLM Baseline Characterization Stats
cols_pre = ['sentence_variance', 'mattr', 'zipf_r2', 'flesch_reading_ease', 'word_count']
t1 = df_pre[cols_pre].describe().T[['mean', 'std', '50%', 'min', 'max']]
t1.columns = ['Mean', 'Std', 'Median', 'Min', 'Max']
print("=== Table 1: Pre-LLM Baseline ===")
print(t1.round(2))

# Table 3a: Longitudinal Drift & Cohen's d
t3a_data = []
metrics_map = [
    ('Sentence Variance', 'sentence_variance', 'Sentence_Var'),
    ('MATTR', 'mattr', 'MATTR'),
    ('Zipf R2', 'zipf_r2', 'Zipf_R2'),
    ('Flesch Reading Ease', 'flesch_reading_ease', 'Flesch_Ease')
]

for label, pre_col, post_col in metrics_map:
    pre_v = df_pre[pre_col].dropna()
    post_v = df_post_full[post_col].dropna()
    d_val = cohens_d(post_v, pre_v)
    t3a_data.append({
        'Metric': label,
        'Pre-LLM Mean (SD)': f"{np.mean(pre_v):.2f} ({np.std(pre_v, ddof=1):.2f})",
        'Post-LLM Mean (SD)': f"{np.mean(post_v):.2f} ({np.std(post_v, ddof=1):.2f})",
        "Cohen's d": f"{d_val:+.3f}"
    })

print("\n=== Table 3a: Longitudinal Drift ===")
print(pd.DataFrame(t3a_data))

# Table 3b: Term-over-term breakdown
terms = ['Fall 2025', 'Unknown 2025', 'Unknown 2026', 'Fall 2026']
t3b = df_post_full.groupby('Term')[['Sentence_Var', 'MATTR', 'Zipf_R2', 'Flesch_Ease', 'Auth_Score']].agg(['mean', 'count'])
print("\n=== Table 3b: Term-over-term ===")
print(t3b.round(3))

# Table 4: Mechanical Artifacts / Flag Breakdown
flags_series = df_post_full['Flags'].copy()
flag_counts = {
    'Synthetic Citation Flags': df_post_full['Flags'].str.contains('Synthetic Citation').sum(),
    'Unnatural Paragraph Symmetry': df_post_full['Flags'].str.contains('Unnatural Para Symmetry').sum(),
    'AI Vocabulary Keywords': df_post_full['Flags'].str.contains('AI Vocabulary').sum(),
    'Synthetic Trigrams': df_post_full['Flags'].str.contains('Synthetic Trigrams').sum(),
    'LLM Section Headers': df_post_full['Flags'].str.contains('LLM Section Headers').sum(),
    'Web Clipboard Spaces': df_post_full['Flags'].str.contains('Web Clipboard Spaces').sum(),
    'Zero-Width Unicode': df_post_full['Flags'].str.contains('Zero-Width Unicode').sum(),
    'Clean / Unflagged': (df_post_full['Flags'] == 'None').sum()
}
t4 = pd.DataFrame(list(flag_counts.items()), columns=['Artifact Flag Type', 'Count'])
t4['Prevalence (%)'] = (t4['Count'] / len(df_post_full) * 100).round(2)
print("\n=== Table 4: Mechanical Artifact Flags ===")
print(t4)
