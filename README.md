# Utica University - Stylometrics & Text Analytics

A lightweight, privacy-focused web application designed for students and professors at Utica University to analyze academic text for stylistic patterns, readability metrics, syntactic structure, and deterministic markers of machine-generated text.

---

## Key Features

* **100% Client-Side Processing:** All text extractions and calculations run entirely inside your browser. No documents, text, or student data are ever uploaded to a server or stored externally.


* **Deterministic Analysis (No AI Models):** Uses statistical algorithms, mathematical formulas (such as Shannon Character Entropy, Moving-Average Type-Token Ratio, and Zipf's Law distribution fitting), and explicit pattern matching rather than black-box AI detection models.


* **Single Document & Text Analysis:** Evaluate pasted text or individual `.docx`, `.pdf`, and `.txt` files with immediate breakdowns of tone, readability, structural symmetry, and stylometrics.


* **Moodle Batch ZIP Processing:** Native support for Moodle's "Download all submissions" `.zip` files. Automatically maps student directory names to their attached documents and parses each file.


* **Batch Score Matrix & Histogram:** Generates an interactive submission matrix alongside a visual distribution histogram (powered by Chart.js) categorizing submissions from *Highly Likely Human* to *Highly Likely AI / Bot*.



---

## Evaluated Metrics

| Category | Metric | Description & Target Thresholds |
| --- | --- | --- |
| **Authenticity & Stylometrics** | **Human Authenticity Score** | Overall composite score based on sentence length variance, lexical richness, and artifact penalties. Expected: `>55%`. |
| **Authenticity & Stylometrics** | **Shannon Character Entropy** | Measures string distribution predictability based on character frequency. Expected: `>4.0`. |
| **Authenticity & Stylometrics** | **Lexical Diversity (TTR)** | Ratio of unique words to total words in the passage. Expected: `>0.45`. |
| **Authenticity & Stylometrics** | **Sentence Variance** | Measures rhythm and variance in sentence length to detect monotonous syntax. Expected: `>25.0`. |
| **Authenticity & Stylometrics** | **Artifact & Flag Checks** | Detects synthetic citations (e.g., arXiv/DOI hallucination patterns), repetitive sentence openers, rigid LLM section headers, and AI buzzword overuse. |
| **Readability** | **Flesch Reading Ease** | Score from 0–100 measuring structural reading accessibility. Technical/academic text typically scores below `40.0`. |
| **Readability** | **Flesch-Kincaid Grade Level** | Translates sentence length and syllable density directly into U.S. educational grade levels. |
| **Prose Structure** | **Nominalization Ratio** | Percentage of abstract "zombie nouns" converted from verbs (words ending in *-tion, -ment, -ance, -ence*). Ratios above `5.0%` indicate bloated prose. |
| **Prose Structure** | **Sentence Clause Complexity** | Percentage of multi-clause or conjunction-heavy sentences. Expected: `>30%` for complex academic prose. |
| **Tone & Punctuation** | **Hedging vs. Assertive Terms** | Counts cautious terms (*suggests, perhaps, might*) against assertive terms (*definitely, proven, clearly*). |
| **Tone & Punctuation** | **Punctuation Fingerprint** | Tracks usage frequency of em-dashes (`—`), semicolons (`;`), exclamation marks (`!`), and questions (`?`). |

---

## Usage Instructions

### Single Analysis

1. Open `index.html` in any standard web browser.


2. Paste text into **Option 1**, or select a `.txt`, `.docx`, or `.pdf` file under **Option 2**.


3. Click **Run Comprehensive Analysis** to generate the detailed metric card.



### Batch Moodle Analysis

1. In Moodle, select **Download all submissions** for an assignment to receive a `.zip` archive.


2. Select the `.zip` archive under **Option 2** and click **Run Comprehensive Analysis**.


3. The tool will parse each subdirectory, run individual stylometric calculations, and display:



* A **Score Distribution Histogram** summarizing author risk categories.


* A **Submission Score Matrix Table** listing student name, document title, authenticity score, verdict classification, MATTR, Zipf $R^2$, passive voice percentage, and triggered flags.



---

## Important Usage Notes & Caveats

* **PowerPoint Files (`.pptx`):** Do not upload PowerPoint presentations. Due to fragment sentence structures, bulleted layouts, and minimal overall text length, presentation slides will produce inaccurate statistical readings.


* **Thesis & Dissertation Documents:** Long-form graduate work features highly specialized terminology, technical iteration, and formal cadence that can occasionally skew baseline metrics. Scores should be evaluated manually alongside context.


* **Interpretative Guidelines:** Results are meant to serve as structural flags for direct pedagogical review, not absolute proof of non-human authorship.



---

## Technical Architecture & External Libraries

This application is built as a single-page HTML5/JS application relying on client-side Web APIs and CDN-hosted browser libraries:

* **[PDF.js](https://mozilla.github.io/pdf.js/):** Client-side PDF text content extraction.


* **[Mammoth.js](https://www.google.com/search?q=https://github.com/mwilliamson/mammoth.js):** Client-side `.docx` raw text extraction.


* **[JSZip](https://stuk.github.io/jszip/):** In-memory unzipping and directory traversal of archive files.


* **[Chart.js](https://www.chartjs.org/):** Canvas-based visual histogram rendering for batch analytics.



---

## Code Repository

* **GitHub Repository:** [https://github.com/securemindorg/Utica-Document-Analysis](https://github.com/securemindorg/Utica-Document-Analysis)

* **Organization:** SecureMind / Utica University Stylometrics Project
