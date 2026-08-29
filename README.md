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
| **Authenticity & Stylometrics** | **Human Authenticity Score** | Composite score incorporating sentence length variance, length-independent lexical richness (MATTR), Zipfian curve fit, and artifact penalties. Expected: `>55%`.

 |
|  | **Shannon Character Entropy** | Measures string distribution predictability based on character frequency. Expected: `>4.0`.

 |
|  | **Lexical Diversity (TTR & MATTR)** | Reports standard Type-Token Ratio (TTR expected `>0.45`) and 50-word Moving-Average Type-Token Ratio (MATTR expected `>0.65`) for length-invariant vocabulary breadth.

 |
|  | **Sentence Variance** | Measures rhythm and variance in sentence length to detect monotonous syntax. Expected: `>25.0`.

 |
|  | **Artifact & Flag Checks** | Scans for invisible zero-width Unicode (`U+200B`), web copy-paste non-breaking spaces (`U+00A0`), synthetic multi-word trigrams (*"in today's rapidly evolving"*), synthetic citations (arXiv/DOI hallucinations), repetitive sentence openers, rigid LLM section headers, low vocabulary burstiness, and paragraph length symmetry.

 |
| **Distribution & Structural Symmetry** | **Zipf's Law Fit ($R^2$)** | Coefficient of determination measuring how closely word frequency distribution matches Zipf's power law. Expected: `>0.85`.

 |
|  | **Vocabulary Burstiness** | Evaluates whether top content words cluster naturally within specific paragraphs rather than being uniformly spaced across the entire document. Expected dispersion score: `>0.35`.

 |
|  | **Paragraph Symmetry (CV)** | Measures coefficient of variation across paragraph lengths. LLM output exhibits low variation (`CV < 0.22`), while human text varies dynamically between short transitions and dense thematic blocks.

 |
| **Readability** | **Flesch Reading Ease** | Score from 0–100 measuring structural reading accessibility. Technical/academic text typically scores below `40.0`.

 |
|  | **Flesch-Kincaid Grade Level** | Translates sentence length and syllable density directly into U.S. educational grade levels.

 |
| **Prose Structure** | **Nominalization Ratio** | Percentage of abstract "zombie nouns" converted from verbs (words ending in *-tion, -ment, -ance, -ence*). Ratios above `5.0%` indicate bloated prose.

 |
|  | **Passive Voice Ratio** | Percentage of sentences utilizing auxiliary verbs with past-participles (e.g., *"was conducted by"*). Ratios above `25.0%` indicate overly passive prose.

 |
|  | **Sentence Clause Complexity** | Percentage of multi-clause or conjunction-heavy sentences. Expected: `>30%` for complex academic prose.

 |
| **Tone & Punctuation** | **Hedging vs. Assertive Terms** | Counts cautious terms (*suggests, perhaps, might*) against assertive terms (*definitely, proven, clearly*).

 |
|  | **Punctuation Fingerprint** | Tracks usage frequency of em-dashes (`—`), semicolons (`;`), exclamation marks (`!`), and questions (`?`).

 |

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
