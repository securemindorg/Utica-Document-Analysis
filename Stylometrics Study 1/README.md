# Cross-Period Stylometric Separation in Academic Student Prose: Distributional Baselines, Length Confounding, and Methodological Cautions

**Author:** Joshua S. White, PhD ([ORCID: 0000-0001-7433-9961](https://orcid.org/0000-0001-7433-9961))  
**Affiliation:** Utica University, Utica, NY, USA  
**Status:** Submitted to *Journal of Quantitative Linguistics* (Taylor & Francis)

> **Methodological Scope Notice:** This repository provides a transparent quantitative-linguistic baseline and confound-decomposition audit for cross-period student writing. High corpus-origin separability **does not** constitute an operational AI-authorship detector and must not be used to make individual academic-integrity judgments.

---

## Overview

Quantitative comparisons of historical and contemporary student writing are widely used to investigate stylistic shifts in the era of large language models (LLMs), yet cross-corpus studies frequently suffer from circular heuristic targets, repeated-author pseudoreplication, and uncontrolled length or register confounds.

This repository contains the unified analysis pipeline, de-identified stylometric feature datasets, statistical reports, and publication figures evaluating **eight classical physical stylometric indices** across **N = 718 authenticated student papers (497 unique authors)** and **N = 26 synthetic LLM technical reports**:

- **Historical Pre-LLM Baseline (y = 0):** 365 upper-level academic papers from the [Michigan Corpus of Upper-level Student Papers (MICUSP)](https://micusp.elicorpora.info/) compiled prior to 2010 (365 unique authors: 99 ESL, 266 NES).
- **Contemporary Post-LLM Cohort (y = 1):** 353 de-identified graduate data science and analytics submissions collected via Moodle during 2025–2026 (132 unique authors: 180 ESL, 173 NES).
- **Exploratory Synthetic LLM Corpus:** 26 unedited technical reports generated in January 2026 (T = 0.7, default top-p) across three frontier model families: ChatGPT 6 (n = 8), Claude Sonnet 5.5 Medium (n = 9), and Google Gemini 3.5 (n = 9).

All texts were processed through a single harmonized raw-extraction pipeline (`Mode: FULL RAW EXTRACTION`) stripping fenced code blocks, inline math, raw tables, and headers, with `Sentence_Var` uniformly winsorized at 1,000.0 words².

---

## Feature Space

| Feature | Definition |
| :--- | :--- |
| `MATTR` (MATTR-100) | Moving-Average Type-Token Ratio across a 100-token sliding window |
| `Zipf_R2` | OLS R² of log(rank) vs. log(frequency) over document token types |
| `Sentence_Var` | Unbiased sample variance of sentence word counts (words², capped at 1,000) |
| `Flesch_Ease` | Flesch Reading Ease readability index |
| `Grade_Level` | Flesch–Kincaid Grade Level |
| `Entropy` | First-order Shannon character entropy (bits) |
| `Nominal_Ratio_%` | Percentage of derived nominalizations (`-tion`, `-ment`, `-ance`, `-ity`, etc.) |
| `Passive_%` | Percentage of sentences containing passive verbal constructions |

---

## Summary of Empirical Results

### 1. Document-Level vs. Author-Level Distributional Shifts

At the document level (353 Post-LLM vs. 365 Pre-LLM), all eight features differ significantly after Benjamini–Hochberg FDR correction (q < 0.0001). However, collapsing repeated submissions to unique authors (132 Post-LLM vs. 365 Pre-LLM) reveals that **passive voice (`Passive_%`) is a repeated-author artifact** (q = 0.2242), whereas the remaining seven features maintain large, highly significant shifts (q < 0.0001):

| Feature | Post-LLM Doc Mean ± SD | Pre-LLM Doc Mean ± SD | Doc Cohen's d | Author Cohen's d | Author FDR q |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Flesch_Ease` | 21.480 ± 12.812 | 38.847 ± 8.393 | -1.609 | -1.816 | <0.0001 |
| `Nominal_Ratio_%` | 4.822 ± 1.582 | 3.195 ± 1.072 | +1.208 | +1.388 | <0.0001 |
| `Entropy` (bits) | 4.504 ± 0.161 | 4.660 ± 0.129 | -1.071 | -1.070 | <0.0001 |
| `Zipf_R2` | 0.926 ± 0.023 | 0.951 ± 0.026 | -1.029 | -1.005 | <0.0001 |
| `Grade_Level` | 14.514 ± 2.890 | 12.637 ± 1.580 | +0.810 | +0.932 | <0.0001 |
| `MATTR` | 0.732 ± 0.042 | 0.697 ± 0.047 | +0.768 | +0.769 | <0.0001 |
| `Sentence_Var` (w²) | 135.970 ± 176.936 | 238.307 ± 150.814 | -0.623 | -0.689 | <0.0001 |
| `Passive_%` | 13.524 ± 8.968 | 15.940 ± 8.461 | -0.277 | -0.148 | **0.2242 (n.s.)** |

### 2. Non-Circular Classifier Validation & Legacy Heuristic Failure

Benchmarking a legacy threshold-based stylometric rule on verified pre-2010 MICUSP papers yields a **45.75% false-positive rate (167/365)**. By contrast, non-circular classifiers trained strictly on verified corpus origin achieve strong, well-calibrated separation under strict **Author-Disjoint 5-Fold `GroupKFold`** cross-validation (N = 718, 497 unique authors):

| Model | 5-Fold Protocol | Accuracy | F1 | ROC-AUC [95% CI] | PR-AUC | Brier |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Random Forest | Stratified | 0.9443 | 0.9438 | 0.9872 [0.9782, 0.9933] | 0.9877 | 0.0438 |
| **Random Forest** | **Author-Disjoint** | **0.9304** | **0.9288** | **0.9820 [0.9732, 0.9894]** | **0.9817** | **0.0542** |
| HistGradientBoosting | Author-Disjoint | 0.9318 | 0.9295 | 0.9805 [0.9713, 0.9881] | 0.9805 | 0.0613 |
| Logistic Regression | Author-Disjoint | 0.9178 | 0.9151 | 0.9689 [0.9557, 0.9791] | 0.9724 | 0.0654 |

Exact global **TreeSHAP** attributions on the full baseline Random Forest rank `Flesch_Ease` (0.1346), `Sentence_Var` (0.1201), and `Zipf_R2` (0.1177) as the top three predictors, followed by `Entropy` (0.0800), `Nominal_Ratio_%` (0.0543), `MATTR` (0.0277), `Grade_Level` (0.0253), and `Passive_%` (0.0092).

### 3. Confound-Decomposition Ablation: Document Length Drives `Zipf_R2`

An eight-condition ablation evaluated under Author-Disjoint 5-Fold Random Forest CV isolates discipline, genre, and document-length confounding:

| Ablation Scenario | N_Pre / N_Post | Acc. | ROC-AUC [95% CI] | Brier | Top 3 TreeSHAP Features |
| :--- | :---: | :---: | :---: | :---: | :--- |
| 1. Full Baseline | 365 / 353 | 0.9304 | 0.9820 [0.9732, 0.9894] | 0.0542 | `Flesch_Ease`, `Sentence_Var`, `Zipf_R2` |
| 2. MICUSP STEM Only | 222 / 353 | 0.9304 | 0.9775 [0.9660, 0.9882] | 0.0580 | `Zipf_R2`, `Flesch_Ease`, `Entropy` |
| 3. MICUSP Eng / Phys / Econ | 92 / 353 | 0.9416 | 0.9780 [0.9652, 0.9885] | 0.0481 | `Flesch_Ease`, `Zipf_R2`, `Sentence_Var` |
| 4. MICUSP Reports & Papers | 266 / 353 | 0.9435 | 0.9853 [0.9765, 0.9923] | 0.0504 | `Zipf_R2`, `Flesch_Ease`, `Sentence_Var` |
| 5. Length Band (750–2,000w) | 75 / 183 | 0.8798 | 0.9278 [0.8840, 0.9600] | 0.0950 | `Flesch_Ease`, `Sentence_Var`, `Entropy` |
| 6. Length-Matched 1:1 | 88 / 88 | 0.8693 | 0.9409 [0.9056, 0.9694] | 0.0941 | `Flesch_Ease`, `Entropy`, `Sentence_Var` |
| 7. STEM + Length Band | 52 / 183 | 0.8723 | 0.9170 [0.8674, 0.9585] | 0.0915 | `Flesch_Ease`, `Entropy`, `Sentence_Var` |
| 8. STEM + Length 1:1 | 71 / 71 | 0.8803 | 0.9498 [0.9121, 0.9797] | 0.0962 | `Flesch_Ease`, `Entropy`, `Nominal_Ratio_%` |

**Key Takeaway:** Restricting MICUSP by discipline or genre leaves ROC-AUC virtually unchanged (0.9775–0.9853). Controlling document length lowers ROC-AUC to 0.9170–0.9498, nearly doubles Brier calibration error (0.0915–0.0962), and **eliminates `Zipf_R2` from the top three predictors across every length-controlled scenario**, confirming that log–log Zipfian fit acts largely as a text-length proxy while `Flesch_Ease`, `Entropy`, and `Sentence_Var` remain length-robust discriminators.

![Confound-Decomposition Ablation ROC-AUC](confound_ablation_auc.png)

### 4. ESL vs. NES Audits Using Real Out-of-Fold Probabilities

Comparing real author-disjoint out-of-fold predicted probabilities `P(Post-LLM | x)` across language backgrounds exposes a sharp period contrast:

- **Pre-LLM MICUSP Baseline (Parity):** ESL (n = 99, mean 0.149 ± 0.187, median 0.072) and NES (n = 266, mean 0.121 ± 0.177, median 0.043) writers show **no statistically significant difference** (Cohen's d = 0.156, Cliff's delta = 0.090, MW p = 0.1879, FDR q = 0.3382), with balanced misclassification rates (**7.07% ESL** vs. **6.02% NES**).
- **Contemporary Moodle Cohort (ESL Upward Shift):** ESL submissions (n = 180, mean 0.891 ± 0.168, median 0.976) score **significantly higher** than NES submissions (n = 173, mean 0.801 ± 0.203, median 0.878; Cohen's d = 0.481, Cliff's delta = 0.382, MW U = 21,522.5, FDR q < 0.0001), accompanied by lower passive voice (d = -0.997), higher `MATTR` (d = 0.938), higher `Entropy` (d = 0.734), lower `Flesch_Ease` (d = -0.605), and narrower `Sentence_Var` (d = -0.385).

![ESL vs. NES Stylometric Distributions](esl_feature_distributions.png)
![Contemporary ESL vs. NES Out-of-Fold Probability KDE](esl_predicted_probability_kde.png)

### 5. Exploratory Synthetic LLM Projection (N = 26)

Projecting raw synthetic technical reports onto the human-trained Random Forest yields an overall mean `P(Post-LLM | x)` of **0.835 ± 0.268** (median 0.953, range 0.202–0.998), but reveals **dramatic inter-vendor divergence**:

| Model Family | n | `Sentence_Var` (w²) | `Flesch_Ease` | `Zipf_R2` | `MATTR` | `P(Post-LLM \| x)` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Google Gemini 3.5 | 9 | 144.042 ± 43.818 | 2.722 ± 4.032 | 0.886 ± 0.016 | 0.805 ± 0.026 | **0.987 ± 0.010** |
| ChatGPT 6 | 8 | 79.769 ± 22.487 | 23.388 ± 5.912 | 0.927 ± 0.013 | 0.790 ± 0.019 | **0.951 ± 0.037** |
| Claude Sonnet 5.5 Medium | 9 | 142.204 ± 60.362 | 42.589 ± 6.391 | 0.934 ± 0.010 | 0.773 ± 0.013 | **0.579 ± 0.331** |

Claude Sonnet 5.5 Medium generates higher reading ease (42.589) than the pre-LLM MICUSP baseline (38.847) and wider sentence dispersion (142.204 words²), demonstrating that LLM prose does not occupy a single monolithic stylometric profile.

![Synthetic AI vs. Human Stylometric Comparison](ai_vs_human_stylometric_comparison.png)

---

## Repository Contents

```text
.
├── unified_stylometric_pipeline.py             # Unified raw extraction, validation, ablation & plotting pipeline
├── micusp_extracted_baseline.csv               # Extracted physical stylometric features (MICUSP Pre-LLM, N=365)
├── micusp_papers.csv                           # MICUSP corpus metadata (discipline, genre, ESL/NES status)
├── moodle_extracted_dataset.csv                # De-identified extracted features (Contemporary Moodle, N=353)
├── esl_subgroup_extracted_dataset.csv          # De-identified contemporary ESL subgroup extracted features
├── ai_extracted_dataset.csv                    # Extracted features for synthetic LLM technical reports (N=26)
├── unified_pipeline_report_20261126_215116.txt # Full logged statistical tables and validation outputs
├── confound_ablation_auc.png                   # Figure: 8-condition confound-decomposition ROC-AUC
├── esl_feature_distributions.png               # Figure: ESL vs. NES feature distributions across periods
├── esl_predicted_probability_kde.png           # Figure: Contemporary ESL vs. NES real OOF probability KDE
├── ai_vs_human_stylometric_comparison.png      # Figure: 3-vendor synthetic AI vs. human stylometric comparison
└── README.md
```

---

## Reproducibility & Usage

### Requirements

- Python 3.9+
- Core packages: `numpy`, `pandas`, `scipy`, `scikit-learn`, `nltk`, `shap`, `matplotlib`, `seaborn`, `tabulate`
- Optional (for raw PDF/DOCX re-extraction): `pypdf` / `pymupdf`, `python-docx`

```bash
pip install numpy pandas scipy scikit-learn nltk shap matplotlib seaborn tabulate
```

### Running the Pipeline

```bash
python unified_stylometric_pipeline.py
```

The script reproduces all statistical tables, author-disjoint cross-validation benchmarks, eight-condition confound-decomposition ablations, real out-of-fold ESL/NES audits, synthetic AI projections, and publication figures from the provided extracted CSV feature matrices (or from raw corpora when local raw paths are configured).

---

## Ethics & Data Privacy

- **Ethics Determination:** Determined exempt by the Utica University Institutional Review Board under **45 CFR 46.104(d)(4)** for secondary research on de-identified educational artifacts (**Protocol #2025-ED-091**).
- **Data Availability:** Raw contemporary student manuscripts are withheld to protect student privacy; only de-identified stylometric feature tables are distributed in this repository. The historical MICUSP corpus is publicly available at [micusp.elicorpora.info](https://micusp.elicorpora.info/).

---

## Citation

If you use this code, dataset, or baseline methodology, please cite:

```bibtex
@unpublished{white2026crossperiod,
  author = {White, Joshua S.},
  title  = {Cross-Period Stylometric Separation in Academic Student Prose: Distributional Baselines, Length Confounding, and Methodological Cautions},
  year   = {2026},
  note   = {Manuscript submitted to Journal of Quantitative Linguistics, Utica University}
}
```
