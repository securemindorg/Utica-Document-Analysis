## LIVE DEMO AT: http://www.securemind.org/Utica-Document-Analysis

# Utica University - Stylometrics & Text Analytics

A lightweight, privacy-focused web application designed for students and professors at Utica University to analyze academic text for stylistic patterns, readability metrics, syntactic structure, and cross-period corpus baselines based on empirical educational data mining research.

---

## Key Features

- **100% Client-Side Processing:** All text extractions and calculations run entirely inside your browser. No documents, text, or student data are ever uploaded to a server or stored externally.


- **Deterministic Analysis:** Uses statistical algorithms, mathematical formulas (such as Shannon Character Entropy, Moving-Average Type-Token Ratio, and Zipf's Law distribution fitting), and explicit linguistic measurements.


- **Single Document & Text Analysis:** Evaluate pasted text or individual `.docx`, `.pdf`, and `.txt` files with immediate breakdowns of readability, structural variance, and stylometrics.


- **Moodle Batch ZIP Processing:** Native support for Moodle's bulk download `.zip` archives. Automatically maps student directory names to their attached documents and parses each file.


- **Batch Score Matrix & Histogram:** Generates an interactive submission score matrix alongside a visual frequency histogram (powered by Chart.js) summarizing document score distributions.


---

## Evaluated Metrics

| Category | Metric | Description & Target Thresholds |
| --- | --- | --- |
| **Stylometric Baseline** | **Total Alignment Score** | Composite score aggregating sentence length variance, lexical richness, Zipfian fit, and readability against contemporary baseline standards. Expected: `>55`. |
| **Stylometric Baseline** | **Sentence Length Variance ($\sigma^2$)** | Quantifies clausal cadence and structural variance across sentences. Contemporary technical writing exhibits structural compression. Expected: `>50.0`. |
| **Stylometric Baseline** | **Flesch Reading Ease (FRE)** | Evaluates text readability on a 0–100 scale. Technical graduate writing naturally reflects denser syntax (`FRE < 35.0`). |
| **Stylometric Baseline** | **Zipfian Linearity ($R^2$)** | Measures word frequency rank scaling fit against natural linguistic decay. Expected: `>0.85`. |
| **Stylometric Baseline** | **Moving-Average TTR ($\text{MATTR}_{100}$)** | Calculates length-invariant lexical diversity over a continuous sliding window ($W=100$). Expected: `>0.65`. |
| **Synthetic Indicators** | **Compression Signature Match** | Flags specific exploratory patterns matching model behaviors (such as extreme readability compression `FRE < 15` or moderate variance loops). Note: Synthetic text is not stylistically monolithic. |
| **Synthetic Indicators** | **Em-Dash Count (—/--)** | Tracks punctuation formatting habits associated with model-generated technical summaries. |
| **Syntactic Density** | **Character Entropy ($H$)** | Measures character-level information density and structural unpredictability per symbol. Expected: `>4.0`. |
| **Syntactic Density** | **Nominalization & Passive Ratios** | Percentage of abstract noun derivations and passive verb constructions characteristic of formal academic registers. |

---

## Usage Instructions

### Single Analysis

1. Open `index.html` in any standard web browser.


2. Paste text into **Option 1**, or select a `.txt`, `.docx`, or `.pdf` file under **Option 2**.

3. Click **Run Formative Stylometric Analysis** to generate the detailed metric card.

### Batch Moodle Analysis

1. In Moodle, select **Download all submissions** for an assignment to receive a `.zip` archive.


2. Select the `.zip` archive under **Option 2** and click **Run Formative Stylometric Analysis**.

3. The tool will parse each subdirectory, run individual stylometric calculations, and display:

- A **Total Score Distribution Histogram** summarizing document score frequencies.


- A **Submission Score Matrix Table** listing student name, file name, total score, verdict status, sentence variance, Flesch Reading Ease, em-dash counts, and Zipf $R^2$.


---

## Important Usage Notes & Methodological Caveats

- **Formative Analytics Only (Not an AI Detector):** This tool provides descriptive writing analytics based on physical stylometric features. It must *not* be used as an automated detector to adjudicate individual student academic integrity violations. 


- **Synthetic Comparison Scope:** Exploratory reference data demonstrates that synthetic text is not stylistically monolithic; vendor architectures (e.g., ChatGPT 6, Gemini 3.5, Claude Sonnet 5.5) show wide structural divergence. Highlighted flags only *potentially* indicate AI-generated text and do not constitute definitive proof.


- **Corpus-Level Context:** Metrics reflect macro-level differences across scholarly eras, disciplines, and technical writing registers rather than individual authorship.


---

## Technical Architecture & External Libraries

This application is built as a single-page HTML5/JS application relying on client-side Web APIs and CDN-hosted browser libraries:

- **[PDF.js](https://mozilla.github.io/pdf.js/):** Client-side PDF text content extraction.


- **[Mammoth.js](https://github.com/mwilliamson/mammoth.js):** Client-side `.docx` raw text extraction.


- **[JSZip](https://stuk.github.io/jszip/):** In-memory unzipping and directory traversal of archive files.


- **[Chart.js](https://www.chartjs.org/):** Canvas-based visual histogram rendering for batch analytics.


---

## Code Repository

- **GitHub Repository:** [https://github.com/securemindorg/Utica-Document-Analysis](https://github.com/securemindorg/Utica-Document-Analysis)


- **Organization:** SecureMind / Utica University Stylometrics Project
