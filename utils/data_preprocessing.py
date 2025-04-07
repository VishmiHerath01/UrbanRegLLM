import re

# Function to clean the extracted PDF content and save to a new file
def clean_text(file_path, output_path):
    # Read the content of the file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Remove page numbers (e.g., 2A, 3A, 4A)
    text = re.sub(r'\b\d+\s?[A]\b', '', text)

    # Remove both header formats
    header_pattern = r'(I\s?ෙ?කොටස\s?:\s?\(\w\)\s?ෙ?\w.*\d{4}\.\d{2}\.\d{2}|PART\sI\s?:\s?Sec\s?\(I\)\s?–\s?GAZETTE\sEXTRAORDINARY\sOF\sTHE\sDEMOCRATIC\sSOCIALIST\sREPUBLIC\sOF\sSRI\sLANKA\s?–\s?\d{2}\.\d{2}\.\d{4})'
    text = re.sub(header_pattern, '', text)

    # Remove unwanted spaces between paragraphs
    text = re.sub(r'\n\s*\n', '\n', text)

    # Remove page 1 content
    page_1_content_pattern = r'(ශී\sලංකා\sපජාතාන්තික\sසමාජවාදී\sජනරජෙය්\sගැසට්\sපතය.*?2021\.07\.08.*?05\.07\.2021.*?THE\sURBAN\sDEVELOPMENT\sAUTHORITY\sLAW.*?Mahinda\sRajapaksa.*?Colombo,\s05\.07\.2021)'
    text = re.sub(page_1_content_pattern, '', text, flags=re.DOTALL)

    # Additional Cleaning Steps
    text = join_hyphenated_words(text)
    text = fix_ocr_errors(text)
    text = standardize_numbering(text)

    # Save the cleaned text
    with open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.write(text)

# Function to join hyphenated words across lines
def join_hyphenated_words(text):
    text = re.sub(r'(\w+)-\s*\n(\w+)', r'\1\2', text)
    return text

# Function to fix known OCR errors
def fix_ocr_errors(text):
    ocr_errors = {
        "Authoirty": "Authority",
        "Develoopment": "Development",
        # Add more common OCR errors here
    }
    for error, correction in ocr_errors.items():
        text = text.replace(error, correction)
    return text

# Function to standardize numbering and bullet points
def standardize_numbering(text):
    # Roman numerals to Arabic
    roman_map = {'I': '1', 'II': '2', 'III': '3', 'IV': '4', 'V': '5', 'VI': '6', 'VII': '7', 'VIII': '8', 'IX': '9', 'X': '10'}

    def replace_roman(match):
        roman = match.group(0)
        return roman_map.get(roman.upper(), roman)

    text = re.sub(r'\b(I{1,3}|IV|V[I]{0,3}|IX|X)\b', replace_roman, text)

    # Standardize numbering patterns
    text = re.sub(r'^\s*(\d+\.)\s*', r'\1 ', text, flags=re.MULTILINE)  # "1.  This..." → "1. This..."
    text = re.sub(r'^\s*([a-z]\.)\s*', r'\1 ', text, flags=re.MULTILINE)  # "a.  This..." → "a. This..."
    text = re.sub(r'^\s*(\(\d+\))\s*', r'\1 ', text, flags=re.MULTILINE)  # "(1)  This..." → "(1) This..."
    text = re.sub(r'^\s*(\([a-z]\))\s*', r'\1 ', text, flags=re.MULTILINE)  # "(a)  This..." → "(a) This..."

    return text

# Example usage
clean_text("data/extracted/pdf_content.txt", "data/cleaned.txt")
