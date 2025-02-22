# Legal Document Processing for LLM Training

This project focuses on processing and cleaning legal documents related to urban development for training a Language Learning Model (LLM).

## Features

- Removes headers and footers from legal documents
- Deduplicates content
- Preserves document structure (parts, sections, subsections)
- Extracts and organizes tables
- Identifies and separates forms

## Usage

```bash
python clean_legal_doc.py
```

## Output Files

The script generates three output files:
- `cleaned_legal_doc.txt`: Main cleaned text with section structure
- `cleaned_legal_doc_tables.json`: Extracted tables
- `cleaned_legal_doc_forms.txt`: Identified forms

## Dependencies

- PyMuPDF (fitz)
- Python 3.x
