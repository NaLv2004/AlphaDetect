---
name: literature-search
description: "Systematic academic literature search and paper analysis for communications research. Use when: searching for papers on arXiv/IEEE/Google Scholar, downloading PDFs, building literature reviews, extracting BibTeX, summarizing papers. Includes scripts for PDF download and BibTeX parsing."
argument-hint: "Describe the topic or specific papers to search for"
---

# Literature Search for Communications Research

## When to Use

- Searching for academic papers on a specific communications topic
- Building a literature review for a new research direction
- Downloading and extracting content from PDFs
- Generating BibTeX entries for references
- Comparing state-of-the-art methods

## Prerequisites

Ensure the Python environment has required packages:
```
pip install requests pdfplumber beautifulsoup4
```

The `pdf-reader` skill should also be available for advanced PDF extraction (dual-column IEEE papers).

## Procedure

### 1. Define Search Scope

Determine the search parameters:
- **Keywords**: Primary technical terms + synonyms
- **Domain**: Channel coding, MIMO, detection, information theory, etc.
- **Recency**: Recent work (default: last 5 years) + foundational papers
- **Target count**: 10-20 papers for a focused review, 30-50 for a comprehensive survey

### 2. Construct Search Queries

Use multiple query strategies for thorough coverage:

| Strategy | Example |
|----------|---------|
| **Exact phrase** | `"polar coded MIMO" "sphere decoding"` |
| **Broad terms** | `polar code MIMO detection latency` |
| **Author search** | `author:"E. Arikan" polar codes` |
| **Venue-specific** | `site:ieeexplore.ieee.org polar code MIMO` |
| **arXiv category** | `site:arxiv.org cs.IT polar code MIMO` |

Communications-specific search terms reference: [search-queries.md](./references/search-queries.md)

### 3. Download Papers

Use the download script to save PDFs locally:
```
python "<skill-path>/scripts/download_paper.py" "<url>" "<output-dir>" [--filename "<name>.pdf"]
```

Save all papers to `research/<topic>/papers/` with descriptive names:
- Format: `AuthorYear_ShortTitle.pdf` (e.g., `Arikan2009_PolarCodes.pdf`)

### 4. Extract Paper Content

For standard PDFs:
```
python "<pdf-reader-skill-path>/scripts/extract_pdf.py" "<pdf-path>" --mode full
```

For IEEE dual-column papers:
```
python "<pdf-reader-skill-path>/scripts/extract_pdf.py" "<pdf-path>" --mode dual-column
```

### 5. Parse and Generate BibTeX

Manually construct BibTeX from extracted content, or use:
```
python "<skill-path>/scripts/parse_bibtex.py" "<extracted-text-file>" --output "<output.bib>"
```

### 6. Compile Literature Review

Organize findings into structured summaries and update:
- `research/<topic>/references.bib` — All BibTeX entries
- `research/memory/literature-notes.md` — Key findings and summaries
