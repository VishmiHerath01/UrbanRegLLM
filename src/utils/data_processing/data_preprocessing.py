import re


def clean_text(file_path, output_path):

    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()


    text = re.sub(r'\b\d+\s?[A]\b', '', text)


    header_pattern = r'(I\s?ෙ?කොටස\s?:\s?\(\w\)\s?ෙ?\w.*\d{4}\.\d{2}\.\d{2}|PART\sI\s?:\s?Sec\s?\(I\)\s?–\s?GAZETTE\sEXTRAORDINARY\sOF\sTHE\sDEMOCRATIC\sSOCIALIST\sREPUBLIC\sOF\sSRI\sLANKA\s?–\s?\d{2}\.\d{2}\.\d{4})'
    text = re.sub(header_pattern, '', text)


    text = re.sub(r'\n\s*\n', '\n', text)


    page_1_content_pattern = r'(ශී\sලංකා\sපජාතාන්තික\sසමාජවාදී\sජනරජෙය්\sගැසට්\sපතය.*?2021\.07\.08.*?05\.07\.2021.*?THE\sURBAN\sDEVELOPMENT\sAUTHORITY\sLAW.*?Mahinda\sRajapaksa.*?Colombo,\s05\.07\.2021)'
    text = re.sub(page_1_content_pattern, '', text, flags=re.DOTALL)


    text = join_hyphenated_words(text)
    text = fix_ocr_errors(text)
    text = standardize_numbering(text)


    with open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.write(text)


def join_hyphenated_words(text):
    text = re.sub(r'(\w+)-\s*\n(\w+)', r'\1\2', text)
    return text


def fix_ocr_errors(text):
    ocr_errors = {
        "Authoirty": "Authority",
        "Develoopment": "Development",
        
    }
    for error, correction in ocr_errors.items():
        text = text.replace(error, correction)
    return text


def standardize_numbering(text):

    roman_map = {'I': '1', 'II': '2', 'III': '3', 'IV': '4', 'V': '5', 'VI': '6', 'VII': '7', 'VIII': '8', 'IX': '9', 'X': '10'}

    def replace_roman(match):
        roman = match.group(0)
        return roman_map.get(roman.upper(), roman)

    text = re.sub(r'\b(I{1,3}|IV|V[I]{0,3}|IX|X)\b', replace_roman, text)


    text = re.sub(r'^\s*(\d+\.)\s*', r'\1 ', text, flags=re.MULTILINE)  
    text = re.sub(r'^\s*([a-z]\.)\s*', r'\1 ', text, flags=re.MULTILINE)  
    text = re.sub(r'^\s*(\(\d+\))\s*', r'\1 ', text, flags=re.MULTILINE) 
    text = re.sub(r'^\s*(\([a-z]\))\s*', r'\1 ', text, flags=re.MULTILINE)  

    return text


clean_text("data/extracted/pdf_content.txt", "data/cleaned.txt")
