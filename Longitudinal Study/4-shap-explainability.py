"""
Moodle Archive Analysis - Local Model Explainability via SHAP
--------------------------------------------------------------
This script builds on Phase 3/4 statistical modeling by integrating 
SHAP (SHapley Additive exPlanations) to provide local interpretability.

Outputs:
1. Global Feature Importance (Summary Plot & Bar Chart)
2. Individual Document Explainability (Waterfall Plot & Force Breakdown)
3. Plain-English Justification Generation for Academic Integrity Reviews
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Set seed for reproducible synthetic demonstration
np.random.seed(42)

# ==============================================================================
# 1. DATA PREPARATION & MODEL TRAINING
# ==============================================================================
def prepare_data_and_model(n_samples=250):
    """
    Generates cohort dataset and trains a Random Forest Classifier.
    """
    is_human = np.random.binomial(1, 0.45, n_samples)
    
    mattr = np.where(is_human, np.random.uniform(0.75, 0.90, n_samples), np.random.uniform(0.80, 0.88, n_samples))
    zipf_r2 = np.where(is_human, np.random.uniform(0.92, 0.98, n_samples), np.random.uniform(0.95, 0.99, n_samples))
    sentence_var = np.where(is_human, np.random.uniform(50.0, 180.0, n_samples), np.random.uniform(20.0, 65.0, n_samples))
    flesch_ease = np.where(is_human, np.random.uniform(1.0, 45.0, n_samples), np.random.uniform(10.0, 35.0, n_samples))
    entropy = np.where(is_human, np.random.uniform(4.20, 4.90, n_samples), np.random.uniform(4.40, 4.80, n_samples))
    nominal_ratio = np.where(is_human, np.random.uniform(4.0, 9.0, n_samples), np.random.uniform(4.5, 8.5, n_samples))
    passive_pct = np.where(is_human, np.random.uniform(2.0, 22.0, n_samples), np.random.uniform(8.0, 25.0, n_samples))
    
    flag_evasion = np.where(is_human, np.random.poisson(0.1, n_samples), np.random.poisson(1.2, n_samples))
    flag_citations = np.where(is_human, np.random.poisson(0.05, n_samples), np.random.poisson(2.5, n_samples))

    df = pd.DataFrame({
        'Subject_ID': [f'Sub_{i:04d}' for i in range(n_samples)],
        'Target_Human': is_human,
        'MATTR': mattr,
        'Zipf_R2': zipf_r2,
        'Sentence_Var': sentence_var,
        'Flesch_Ease': flesch_ease,
        'Entropy': entropy,
        'Nominal_Ratio_%': nominal_ratio,
        'Passive_%': passive_pct,
        'Flag_Counts_Evasion': flag_evasion,
        'Flag_Counts_Citations': flag_citations
    })

    feature_cols = [
        'MATTR', 'Zipf_R2', 'Sentence_Var', 'Flesch_Ease', 
        'Entropy', 'Nominal_Ratio_%', 'Passive_%', 
        'Flag_Counts_Evasion', 'Flag_Counts_Citations'
    ]

    X = df[feature_cols]
    y = df['Target_Human']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X_train, X_test, df.iloc[X_test.index].reset_index(drop=True), feature_cols

# ==============================================================================
# 2. SHAP EXPLAINABILITY ENGINE
# ==============================================================================
def compute_shap_explanations(model, X_train, X_test):
    """
    Computes TreeSHAP values for the trained model over test observations.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    
    # Handle multi-class / binary output dimensions
    if len(shap_values.shape) == 3:
        # Index 1 corresponds to Target_Human = 1
        shap_values_human = shap_values[:, :, 1]
    else:
        shap_values_human = shap_values

    return explainer, shap_values_human

# ==============================================================================
# 3. GLOBAL SHAP VISUALIZATIONS
# ==============================================================================
def plot_global_shap_summary(shap_values, X_test):
    """
    Generates SHAP summary plot showing feature impacts across all test samples.
    """
    print("\n[+] Generating Global SHAP Feature Importance Plots...")
    
    # Global Summary Beeswarm Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Global Feature Impact (Target: Human Authorship)", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig("shap_global_summary.png", dpi=300)
    plt.close()
    
    print("    -> Saved global summary plot to 'shap_global_summary.png'")

# ==============================================================================
# 4. LOCAL EXPLAINABILITY & PLAIN-ENGLISH JUSTIFICATIONS
# ==============================================================================
def explain_individual_submission(sample_idx, test_df, X_test, shap_values, feature_cols):
    """
    Generates local SHAP explanation and constructs a plain-English academic report.
    """
    doc_id = test_df.iloc[sample_idx]['Subject_ID']
    actual_label = "Human" if test_df.iloc[sample_idx]['Target_Human'] == 1 else "Non-Human / Review"
    
    # Single sample SHAP values
    sample_shap = shap_values[sample_idx]
    base_value = sample_shap.base_values
    values = sample_shap.values
    data = sample_shap.data

    # Generate Waterfall Plot for the single document
    plt.figure(figsize=(9, 5))
    shap.plots.waterfall(sample_shap, show=False)
    plt.title(f"SHAP Waterfall Explanation: {doc_id}", fontsize=11, pad=12)
    plt.tight_layout()
    plt.savefig(f"shap_waterfall_{doc_id}.png", dpi=300)
    plt.close()

    # Feature Contribution Analysis for Text Report
    impact_df = pd.DataFrame({
        'Feature': feature_cols,
        'Value': data,
        'SHAP_Value': values,
        'Abs_SHAP': np.abs(values)
    }).sort_values(by='Abs_SHAP', ascending=False)

    print("\n" + "="*80)
    print(f" EXPLAINABILITY REPORT: {doc_id} ")
    print("="*80)
    print(f" Actual Label: {actual_label}")
    print(f" Base Expectation (Log-Odds): {base_value:.4f}")
    print(f" Final SHAP Output Value:   {base_value + np.sum(values):.4f}")
    print("-" * 80)
    print(" Top Driving Factors for Classification:")
    
    justification_lines = []
    for _, row in impact_df.head(4).iterrows():
        feat = row['Feature']
        val = row['Value']
        s_val = row['SHAP_Value']
        direction = "INCREASED probability of Human" if s_val > 0 else "DECREASED probability of Human (AI Risk Signal)"
        
        print(f"  • {feat} = {val:.2f} | SHAP: {s_val:+.4f} --> {direction}")
        
        # Build plain-English narrative rules
        if feat == 'Flag_Counts_Citations' and val >= 1:
            justification_lines.append(f"presence of {int(val)} flagged synthetic citations")
        elif feat == 'Sentence_Var' and val < 45.0:
            justification_lines.append(f"low sentence length variance ({val:.1f}) indicating uniform syntax")
        elif feat == 'Passive_%' and val > 18.0:
            justification_lines.append(f"elevated passive voice frequency ({val:.1f}%)")
        elif feat == 'Flag_Counts_Evasion' and val >= 1:
            justification_lines.append(f"detection of evasion artifacts ({int(val)} instance/s)")
        elif feat == 'MATTR' and val < 0.80:
            justification_lines.append(f"restricted moving-average type-token ratio ({val:.2f})")

    # Construct Final Summary Sentence
    if justification_lines:
        narrative = f"Document {doc_id} was flagged primarily due to the " + ", ".join(justification_lines) + "."
    else:
        narrative = f"Document {doc_id} exhibits feature distributions consistent with expected human academic baseline standards."

    print("\n--- Faculty-Facing Governance Justification ---")
    print(narrative)
    print(f"-> Saved waterfall visualization to 'shap_waterfall_{doc_id}.png'\n")

# ==============================================================================
# MAIN EXECUTION ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    # Step 1: Prepare data and train model
    model, X_train, X_test, test_df, feature_cols = prepare_data_and_model()

    # Step 2: Compute TreeSHAP values
    explainer, shap_values = compute_shap_explanations(model, X_train, X_test)

    # Step 3: Plot global feature importances
    plot_global_shap_summary(shap_values, X_test)

    # Step 4: Generate local explainability reports for selected samples
    # Sample A: High-risk/AI-influenced candidate
    explain_individual_submission(sample_idx=0, test_df=test_df, X_test=X_test, shap_values=shap_values, feature_cols=feature_cols)

    # Sample B: Human candidate
    explain_individual_submission(sample_idx=1, test_df=test_df, X_test=X_test, shap_values=shap_values, feature_cols=feature_cols)
