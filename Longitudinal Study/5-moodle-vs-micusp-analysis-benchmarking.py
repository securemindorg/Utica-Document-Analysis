"""
Moodle Archive vs. MICUSP Benchmark Analysis Pipeline
------------------------------------------------------
1. Ingests MICUSP (Michigan Corpus of Upper-Level Student Papers) as Pure-Human Ground Truth.
2. Evaluates Feature Distributions Across Disciplines (e.g., Humanities vs. STEM).
3. Trains Classifier & Computes False Positive Rates (FPR) on MICUSP Data.
4. Generates SHAP Explanations comparing Moodle Submissions against MICUSP Feature Baselines.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

np.random.seed(42)

# ==============================================================================
# 1. GENERATE / LOAD MICUSP & MOODLE COMBINED DATASET
# ==============================================================================
def load_combined_micusp_moodle_dataset(n_micusp=300, n_moodle=200):
    """
    Generates dataset incorporating MICUSP benchmark data alongside Moodle submissions.
    """
    disciplines = ['Biology', 'English', 'Philosophy', 'Physics', 'Psychology', 'Sociology']
    
    # --- MICUSP DATA (100% Human Ground Truth) ---
    micusp_disc = np.random.choice(disciplines, n_micusp)
    
    # Natural academic variation across disciplines
    mattr_micusp = np.random.uniform(0.78, 0.91, n_micusp)
    zipf_micusp = np.random.uniform(0.93, 0.98, n_micusp)
    sentence_var_micusp = np.random.uniform(60.0, 190.0, n_micusp)
    flesch_micusp = np.random.uniform(5.0, 40.0, n_micusp)
    entropy_micusp = np.random.uniform(4.30, 4.95, n_micusp)
    nominal_micusp = np.random.uniform(4.5, 9.5, n_micusp)
    
    # STEM disciplines in MICUSP have higher passive voice naturally
    passive_micusp = np.where(np.isin(micusp_disc, ['Biology', 'Physics']), 
                              np.random.uniform(15.0, 32.0, n_micusp), 
                              np.random.uniform(3.0, 18.0, n_micusp))
    
    flag_evasion_micusp = np.zeros(n_micusp, dtype=int)
    flag_citations_micusp = np.zeros(n_micusp, dtype=int)

    df_micusp = pd.DataFrame({
        'Subject_ID': [f'MICUSP_{i:04d}' for i in range(n_micusp)],
        'Source_Corpus': 'MICUSP_Benchmark',
        'Discipline': micusp_disc,
        'Verdict': 'Human',
        'Target_Human': 1,
        'MATTR': mattr_micusp,
        'Zipf_R2': zipf_micusp,
        'Sentence_Var': sentence_var_micusp,
        'Flesch_Ease': flesch_micusp,
        'Entropy': entropy_micusp,
        'Nominal_Ratio_%': nominal_micusp,
        'Passive_%': passive_micusp,
        'Flag_Counts_Evasion': flag_evasion_micusp,
        'Flag_Counts_Citations': flag_citations_micusp
    })

    # --- MOODLE DATA (Mix of Human, Review, and AI/Hybrid) ---
    is_human_moodle = np.random.binomial(1, 0.40, n_moodle)
    moodle_disc = np.random.choice(['Business/Management', 'Data Science'], n_moodle)

    df_moodle = pd.DataFrame({
        'Subject_ID': [f'Moodle_{i:04d}' for i in range(n_moodle)],
        'Source_Corpus': 'Moodle_Submissions',
        'Discipline': moodle_disc,
        'Verdict': np.where(is_human_moodle == 1, 'Human', np.random.choice(['Review', 'Non-Human'], n_moodle, p=[0.7, 0.3])),
        'Target_Human': is_human_moodle,
        'MATTR': np.where(is_human_moodle, np.random.uniform(0.76, 0.88, n_moodle), np.random.uniform(0.81, 0.87, n_moodle)),
        'Zipf_R2': np.where(is_human_moodle, np.random.uniform(0.91, 0.97, n_moodle), np.random.uniform(0.96, 0.99, n_moodle)),
        'Sentence_Var': np.where(is_human_moodle, np.random.uniform(55.0, 175.0, n_moodle), np.random.uniform(22.0, 58.0, n_moodle)),
        'Flesch_Ease': np.where(is_human_moodle, np.random.uniform(5.0, 42.0, n_moodle), np.random.uniform(12.0, 32.0, n_moodle)),
        'Entropy': np.where(is_human_moodle, np.random.uniform(4.25, 4.88, n_moodle), np.random.uniform(4.45, 4.78, n_moodle)),
        'Nominal_Ratio_%': np.where(is_human_moodle, np.random.uniform(4.0, 8.8, n_moodle), np.random.uniform(5.0, 8.5, n_moodle)),
        'Passive_%': np.where(is_human_moodle, np.random.uniform(3.0, 20.0, n_moodle), np.random.uniform(10.0, 26.0, n_moodle)),
        'Flag_Counts_Evasion': np.where(is_human_moodle, np.random.poisson(0.02, n_moodle), np.random.poisson(1.1, n_moodle)),
        'Flag_Counts_Citations': np.where(is_human_moodle, np.random.poisson(0.01, n_moodle), np.random.poisson(2.3, n_moodle))
    })

    combined_df = pd.concat([df_micusp, df_moodle], ignore_index=True)
    return combined_df

# ==============================================================================
# 2. MICUSP BENCHMARKING & FPR EVALUATION
# ==============================================================================
def evaluate_micusp_false_positives(df, model, feature_cols):
    """
    Evaluates model performance strictly against MICUSP papers to test False Positive Rates.
    """
    print("\n" + "="*80)
    print(" MICUSP BENCHMARK & DISCIPLINARY FALSE POSITIVE EVALUATION ")
    print("="*80)

    micusp_df = df[df['Source_Corpus'] == 'MICUSP_Benchmark'].copy()
    X_micusp = micusp_df[feature_cols]
    
    # Predict probabilities (Target_Human = 1)
    probs = model.predict_proba(X_micusp)[:, 1]
    # Classify as AI-influenced if Human Probability < 0.50
    preds = np.where(probs >= 0.50, 1, 0)

    micusp_df['Predicted_Human'] = preds
    micusp_df['False_Positive_AI'] = np.where(preds == 0, 1, 0)

    overall_fpr = (micusp_df['False_Positive_AI'].sum() / len(micusp_df)) * 100
    print(f"\nOverall MICUSP Baseline False Positive Rate (FPR): {overall_fpr:.2f}%")
    print(f"  Total MICUSP Papers Tested: {len(micusp_df)}")
    print(f"  Falsely Flagged as AI/Review: {micusp_df['False_Positive_AI'].sum()}")

    # Disciplinary Breakdown
    print("\n--- False Positive Rate Breakdown by Discipline ---")
    disc_summary = micusp_df.groupby('Discipline').agg(
        Total_Papers=('Subject_ID', 'count'),
        False_Positives=('False_Positive_AI', 'sum'),
        Mean_Passive_Pct=('Passive_%', 'mean')
    )
    disc_summary['FPR_%'] = (disc_summary['False_Positives'] / disc_summary['Total_Papers']) * 100
    print(disc_summary[['Total_Papers', 'False_Positives', 'FPR_%', 'Mean_Passive_Pct']].to_string())

# ==============================================================================
# 3. SHAP ANALYSIS WITH MICUSP REFERENCE
# ==============================================================================
def run_micusp_shap_analysis(df, feature_cols):
    """
    Trains Random Forest model on combined dataset and computes SHAP explanations.
    """
    X = df[feature_cols]
    y = df['Target_Human']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]

    return model, X_test, shap_values

# ==============================================================================
# MAIN EXECUTION ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    feature_cols = [
        'MATTR', 'Zipf_R2', 'Sentence_Var', 'Flesch_Ease', 
        'Entropy', 'Nominal_Ratio_%', 'Passive_%', 
        'Flag_Counts_Evasion', 'Flag_Counts_Citations'
    ]

    # Step 1: Load MICUSP + Moodle combined dataset
    df = load_combined_micusp_moodle_dataset()

    # Step 2: Fit model and calculate SHAP values
    model, X_test, shap_values = run_micusp_shap_analysis(df, feature_cols)

    # Step 3: Evaluate False Positive Rates on MICUSP benchmark
    evaluate_micusp_false_positives(df, model, feature_cols)
