## LIVE DEMO AT: http://www.securemind.org/Utica-Document-Analysis

# Stylometrics & Text Analytics

## Part 1: GitHub README (`README.md`)

# Formative Technical Document Analytics

[![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python / JS Environment](https://img.shields.io/badge/stack-Client--Side%20Vanilla%20JS%20%7C%20PDF.js%20%7C%20Mammoth-orange)](index.html)
[![FERPA Compliance](https://img.shields.io/badge/privacy-100%25%20Client--Side%20FERPA%20Compliant-success)](index.html)

A client-side web utility designed for educators and researchers to perform **formative stylometric and technical document analysis** on student writing submissions. Built in accordance with empirical research baselines from Utica University, this tool evaluates physical text metrics without transmitting sensitive student data to external servers.

---

## Key Features

* **100% Client-Side Processing:** Operates entirely within your web browser memory using `pdf.js`, `mammoth.js`, and `JSZip`. No student data, documents, or names are ever uploaded to a cloud server or external database, ensuring strict **FERPA compliance**.
* **Physical Stylometric Feature Extraction:** Computes eight robust, non-heuristic linguistic measures:
  * **Sentence Length Variance ($\sigma_{words}^2$):** Quantifies clausal cadence and structural burstiness.
  * **Flesch Reading Ease (FRE):** Evaluates syntactic readability on a 0–100 scale.
  * **Zipfian Rank-Frequency Linearity ($R^2$):** Measures adherence to natural lexical decay.
  * **Moving-Average Type-Token Ratio ($MATTR_{100}$):** Measures length-independent lexical diversity over a sliding window ($W = 100$).
  * **Shannon Character Entropy ($H$):** Captures character-level information density.
  * **Nominalization & Passive Voice Ratios:** Tracks formal academic register markers.
* **Batch ZIP Processing & Visualization:** Upload a Moodle or Canvas assignment export (`.zip`) containing multiple `.pdf`, `.docx`, or `.txt` files to instantly generate score distribution histograms, summary tables, and structural flag breakdowns.
* **Exploratory Synthetic & Trope Sign Scanning:** Flags potential structural patterns and linguistic tropes (e.g., lexical over-polishing, negative parallelisms, trailing participial modifiers) to guide formative writing instruction.

---

## Technology Stack

* **Frontend:** HTML5, CSS3 (Utica University branding palette: Deep Navy `#002D62` and Accent Orange `#E85523`).
* **Document Parsers:** 
  * [PDF.js](https://github.com/mozilla/pdf.js) (v3.11.174) for client-side PDF text extraction.
  * [Mammoth.js](https://github.com/mwilliamson/mammoth.js) (v1.6.0) for `.docx` raw text extraction.
  * [JSZip](https://github.com/Stuk/jszip) (v3.10.1) for batch archive handling.
* **Visualization:** [Chart.js](https://www.chartjs.org/) for interactive score distribution histograms.

---

## Quick Start & Installation

Because this tool runs entirely client-side, you do not need a complex backend server or Python environment to test individual files.

1. Clone or download this repository:
   ```bash
   git clone [https://github.com/securemindorg/Utica-Document-Analysis.git](https://github.com/securemindorg/Utica-Document-Analysis.git)

2. Open `index.html` directly in any modern web browser (Chrome, Firefox, Edge, Safari).
3. Paste text directly into **Option 1**, or select individual files / batch ZIP archives under **Option 2**.

---

## Important Methodological & Pedagogical Notice

* **Mandatory Manual Review:** Analytical outputs serve solely as supplementary writing metrics and initial exploratory flags. All student papers **must** be manually reviewed by an instructor.
* **No Definitive AI Detection Exists:** Scientifically, there is no infallible algorithmic test for AI text generation. Stylometric thresholds can trigger false positives on human technical writing (especially by non-native English speakers) or false negatives on edited AI text. **Scores must never be used as sole proof or justification for academic dishonesty proceedings.**

---

## Part 2: In-Depth Explanation of Research Findings & Statistics

This section breaks down the empirical research paper—*"Cross-Period Stylometric Separation in Student Writing: Corpus Baselines and Methodological Cautions for Learning Analytics"* by Dr. Joshua S. White (Utica University)—with graduate-level rigor.

### 1. The Core Research Motivation & Design
As Generative AI tools integrate into academic workflows, educators face the challenge of distinguishing authentic human technical reasoning from AI-assisted synthesis. Commercial AI detectors have historically relied on opaque black-box heuristics (e.g., token perplexity thresholds) that introduce severe demographic skews, particularly against non-native English speakers (ESL) and technical academic writing. Furthermore, many studies suffer from **circular ground-truth definitions**—training models on synthetic labels generated by the same heuristic features being tested, creating a tautological feedback loop.

To resolve this, the paper establishes a **non-circular empirical classification framework**:
* **Historical Baseline ($y = 0$, $N = 365$):** Drawn from the Michigan Corpus of Upper-level Student Papers (MICUSP), spanning 16 academic disciplines (pre-LLM era).
* **Contemporary Cohort ($y = 1$, $N = 355$):** Collected across multi-term quantitative graduate coursework (2025–2026).
* **Feature Space:** Strictly restricted to **eight physical stylometric features** (no black-box proprietary token probabilities).

---

### 2. Statistical Metrics & Feature Definitions

A graduate-level understanding requires mastering the mathematical formulation and linguistic meaning of the primary feature space:

- **Sentence Length Variance (`Sentence_Var`):**
  $$Sentence\_{Var} = \frac{1}{N_s - 1}\sum_{i=1}^{N_s}(w_i - \bar{w})^2$$
  * **What it means:** Quantifies clausal cadence and structural "burstiness" ($\sigma_{words}^2$ in words²). Human academic prose typically alternates brief declarative assertions with complex compound explanations, yielding high variance. Contemporary digital drafting environments and generative models tend toward rhythmically flattened, uniform sentence lengths.
* **Flesch Reading Ease ($\text{FRE}$):**
  $$\text{FRE} = 206.835 - 1.015\left(\frac{\text{words}}{\text{sentences}}\right) - 84.6\left(\frac{\text{syllables}}{\text{words}}\right)$$
  * *What it means:* Evaluates text readability on a 0–100 scale. Lower scores reflect denser syntactic structures and formal academic registers.
* **Zipfian Rank-Frequency Fit ($\text{Zipf } R^2$):**
  * *What it means:* Zipf’s Law states that word frequency in natural human language is inversely proportional to its frequency rank ($f(r) \propto r^{-\gamma}$). Fitting a log-log linear regression model between word frequency and rank yields an $R^2$ coefficient measuring adherence to natural linguistic decay. Constrained technical registers and standardized editing tools often exhibit tighter, highly linear distributions.
* **Moving-Average Type-Token Ratio ($\text{MATTR}_{100}$):**
  * *What it means:* Standard Type-Token Ratio (vocabulary richness) decays artificially as document length increases. MATTR calculates TTR over a continuous sliding window ($W = 100$ tokens), providing a size-independent measure of lexical diversity.
* **Shannon Character Entropy ($H$):**
  $$H(X) = -\sum_{i=1}^{n}P(x_i)\log_2 P(x_i)$$
  * *What it means:* Derived from Information Theory, character-level entropy measures average information density and structural unpredictability per character symbol.

---

### 3. Classification Performance & Validation Results

Using an ensemble **Random Forest classifier** tuned via 5-fold cross-validation strictly within training partitions ($N = 540$), the model achieved outstanding discrimination between the two corpora:
* **Holdout Test Accuracy:** **94.0%** (169/180 correctly classified, exact Clopper-Pearson 95% CI [89.3%, 96.8%]).
* **ROC-AUC:** **0.9840** (Cross-validation mean ROC-AUC: $0.9843 \pm 0.0088$).
* **Robustness Audits (GroupKFold):**
  * *Author-Disjoint Split ($N = 507$ unique authors):* Out-of-fold ROC-AUC of $0.9612 \pm 0.0142$ (91.7% accuracy). This proves the model does not memorize individual author styles.
  * *Assignment-Disjoint Split ($N = 25$ unique assignment prompts):* Out-of-fold ROC-AUC of $0.9480 \pm 0.0185$ (89.4% accuracy). This proves predictions are not driven by prompt-specific vocabulary or keyword memorization.

---

### 4. TreeSHAP Interpretability & Feature Importance

To understand *why* the model separates the corpora, **TreeSHAP** (SHapley Additive exPlanations) calculated game-theoretic feature attributions on probability scale space ($\Delta \text{probability}$):

1. **Flesch Reading Ease ($\text{mean } \vert{}SHAP\vert{} = 0.13$):** Primary predictive feature ($\approx 13.0\%$ mean absolute probability shift). Low reading ease ($FRE < 25.0$) drives positive directional attribution toward contemporary cohort classification.
2. **Sentence Length Variance (`Sentence_Var`):** Low sentence variance (`Sentence_Var` $\le 50.0\text{ words}^2$) exerts a strong positive probability impact ($\text{SHAP} \in [0.10, 0.23]$).
3. **Zipfian Linearity ($\text{mean } \vert{}SHAP\vert{} = 0.10$):** Word frequency rank scaling fit ($R^2$).
4. **Shannon Character Entropy ($\text{mean } \vert{}SHAP\vert{} = 0.06$):** Character-level information density.

---

### 5. Language-Background Subgroup Analysis & Methodological Cautions

A critical contribution of the paper is its ethical examination of demographic subgroups (Native English Speakers [NES] vs. English as a Second Language [ESL] students).
* **Feature-Level Differences:** In the contemporary cohort, ESL student reports exhibited lower reading ease ($FRE = 17.80$ vs. $23.42$, $q < 0.0001$) and distinct lexical diversity profiles.
* **Model Score Distributions:** When evaluating model predicted probabilities $P(y=1\vert{}x)$, NES and ESL authors in the contemporary cohort exhibited closely aligned central tendencies ($0.92 \pm 0.035$ vs. $0.93 \pm 0.035$).
* **The Crucial Methodological Warning:** Because the contemporary cohort contains only a single corpus-origin class ($y = 1$), **this dataset contains no ground-truth negative instances for contemporary writing**. Therefore, these overlapping score distributions only reflect model output consistency; **they do not establish equal false-positive rates or prove demographic fairness for AI detection**.

> **Core Conclusion:** The classifier measures **macro-level corpus separation** (encompassing era, institutional, disciplinary, document length, genre, and drafting environment differences) rather than audited individual AI usage. **It must never be used as an automated detector to adjudicate individual student academic integrity violations.**

---

## Part 3: How to Improve Flesch Reading Ease (FRE) with Examples

The **Flesch Reading Ease (FRE)** formula measures text readability on a 0–100 scale. Higher scores indicate easier readability, while lower scores indicate dense, complex prose. 

In technical and graduate writing, students often fall into the trap of writing overly dense sentences packed with nominalizations and passive voice, which depress FRE scores below 25.0. 

### Strategies to Improve FRE:
1. **Reduce Sentence Length:** Break long, compound-complex sentences into two or three shorter, direct sentences. (The formula heavily penalizes high words-per-sentence ratios).
2. **Minimize Syllable Density:** Replace multi-syllable jargon or abstract nouns with simpler, direct action verbs when appropriate.
3. **Convert Passive Voice to Active Voice:** Active voice clarifies agency and naturally shortens sentence length.

---

### Concrete Before & After Examples

#### Example 1: Technical Report Writing
* **Before (FRE: 14.2 — Very Difficult / Dense):**
  > *"Implementation of the algorithmic optimization protocol was conducted with the objective of ameliorating computational latency issues previously identified during the preliminary system evaluation phase."*
  * *Critique:* 24 words, 41 syllables. Heavy nominalizations (*implementation, optimization, amelioration, evaluation*). Passive voice (*was conducted*).
* **After (FRE: 58.4 — Standard / Readable):**
  > *"We optimized the algorithm to reduce the system lag found during early testing."*
  * *Critique:* 13 words, 16 syllables. Direct active voice (*We optimized*), fewer syllables, and high clarity.

#### Example 2: Data Science Methodology Section
* **Before (FRE: 18.9 — Very Difficult):**
  > *"Subsequent to the extraction of feature matrices across heterogeneous corpora, normalization procedures were executed to facilitate the mitigation of distributional skews affecting classifier performance."*
  * *Critique:* 25 words, 48 syllables. Stilted academic padding.
* **After (FRE: 52.1 — Standard / Readable):**
  > *"After extracting features from the mixed datasets, we normalized the data to prevent skewed distributions from hurting classifier performance."*
  * *Critique:* 20 words, 26 syllables. Clear subject-verb structure and concise vocabulary.
