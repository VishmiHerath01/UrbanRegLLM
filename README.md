
# UrbanRegLLM: Urban Development Regulations LLM System

This project focuses on processing, analyzing, and querying urban development regulations using specialized Language Learning Models (LLMs). The system transforms complex legal documents into structured formats optimized for AI training and retrieval.

## Project Overview

UrbanRegLLM processes the Urban Development Regulations document (English_Version.pdf), a comprehensive legal text with approximately 150 pages containing regulations, guidelines, and standards for urban development. The project includes:

1. **Document Processing Pipeline**: Transforms raw legal documents into structured formats
2. **Embedding Comparison**: Evaluates different embedding approaches for legal text retrieval
3. **LLM Integration**: Uses specialized models for answering queries about urban regulations

## Document Characteristics

- Hierarchical structure with numbered sections and subsections
- Approximately 45,000 words and 2,100 sentences
- Average sentence length of 21.4 words (longer than standard English)
- High frequency of domain-specific legal terminology
- Contains 24 tables and schedules with technical specifications
- Extensive cross-references between sections

## Processing System Components

### Text Processing Features
- Hyphenation handling
- Section/subsection formatting
- List and bullet point standardization
- Spacing and punctuation normalization
- Measurement formatting
- Multi-language support (Sinhala/English)

## Document Structure

The processed legal documents follow a standardized structure:

1. **Document Metadata Section**
   - Header: "DOCUMENT METADATA"
   - Contains key-value pairs from metadata
   - Separated by line breaks

2. **Main Text Section**
   - Contains the primary document content
   - Formatted with proper spacing and line breaks
   - Section/subsection numbers clearly visible

3. **Tables and Schedules Section**
   - Header: "TABLES AND SCHEDULES"
   - Contains formatted table data
   - Clear separation from main text


## Dependencies

- PyMuPDF (fitz)
- Python 3.x
- transformers
- huggingface_hub
- torch
- accelerator
- bitsandbytes
- peft
- tqdm
- datasets
- sentencepiece
- protobuf
- pyarrow
- numpy
- pandas
- sentence-transformers
- faiss-cpu
- requests
- llama-cpp-python
- matplotlib

https://drive.google.com/drive/folders/12tQrtXREyhClXD4Hr77hxV8B-g-W9421?usp=drive_link

