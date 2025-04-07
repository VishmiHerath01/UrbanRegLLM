import re
import os
import json
from collections import defaultdict

def normalize_text(input_file, output_file):
    """
    Normalize and preprocess the text from the regulations document
    - Standardize spacing
    - Handle tables
    - Extract sections and regulations
    - Create a structured JSON with normalized content
    """
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Replace table content with references
    # This regex pattern identifies potential table sections based on patterns in the document
    table_pattern = r'(Form [A-Z]|Schedule \d+|Table \d+)[\s\S]*?(?=Form [A-Z]|Schedule \d+|PART [IVX]+|^\d+\.|\Z)'
    
    def table_replacer(match):
        table_name = match.group(1).strip()
        return f"\n[Reference to {table_name}. See separate CSV file.]\n"
    
    text = re.sub(table_pattern, table_replacer, text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Extract parts and their content
    parts = {}
    part_pattern = r'PART ([IVX]+)\s+([A-Za-z\s]+)\s+'
    part_matches = re.finditer(part_pattern, text)
    
    prev_pos = 0
    for i, match in enumerate(part_matches):
        part_num = match.group(1)
        part_title = match.group(2).strip()
        start_pos = match.start()
        
        if i > 0:  # Store the previous part's content
            prev_part_num = list(parts.keys())[-1]
            parts[prev_part_num]['content'] = text[prev_pos:start_pos].strip()
        
        parts[part_num] = {
            'title': part_title,
            'content': ''
        }
        prev_pos = match.end()
    
    # Store the last part's content
    if parts:
        last_part_num = list(parts.keys())[-1]
        parts[last_part_num]['content'] = text[prev_pos:].strip()
    
    # Extract regulations and organize them
    regulations = defaultdict(list)
    
    for part_num, part_data in parts.items():
        content = part_data['content']
        
        # Extract regulations
        reg_pattern = r'(\d+)\.(?:\s*\((\d+)\))?\s+(.*?)(?=\d+\.\s*\(\d+\)|\d+\.\s|\Z)'
        reg_matches = re.finditer(reg_pattern, content, re.DOTALL)
        
        for reg_match in reg_matches:
            reg_num = reg_match.group(1)
            sub_reg_num = reg_match.group(2) if reg_match.group(2) else None
            reg_text = reg_match.group(3).strip()
            
            if sub_reg_num:
                # This is a sub-regulation
                regulations[part_num].append({
                    'regulation_num': reg_num,
                    'sub_regulation_num': sub_reg_num,
                    'text': reg_text
                })
            else:
                # This is a main regulation
                regulations[part_num].append({
                    'regulation_num': reg_num,
                    'text': reg_text
                })
    
    # Extract definitions
    definitions = {}
    if "DEFINITIONS" in text:
        def_section = text.split("DEFINITIONS")[1].split("Schedule 2")[0]
        
        # Extract individual definitions
        def_pattern = r'"([^"]+)"\s+means\s+([^"]+?)(?=\"|$)'
        def_matches = re.finditer(def_pattern, def_section, re.DOTALL)
        
        for def_match in def_matches:
            term = def_match.group(1).strip()
            definition = def_match.group(2).strip()
            definitions[term] = definition
    
    # Create structured data
    structured_data = {
        'title': 'Urban Development Authority Planning & Development Regulations 2021',
        'parts': parts,
        'regulations': dict(regulations),
        'definitions': definitions
    }
    
    # Write to JSON file
    output_json_file = output_file.replace('.txt', '.json').replace("normalized_regulations.txt", "normalized/normalized_regulations.json") # Adjust path for JSON
    with open(output_json_file, 'w', encoding='utf-8') as json_file:
        json.dump(structured_data, json_file, indent=4, ensure_ascii=False)

    # Generate a clean readable version
    clean_text = []
    clean_text.append("# URBAN DEVELOPMENT AUTHORITY PLANNING & DEVELOPMENT REGULATIONS 2021\n")
    
    # Add introduction regulations (not in any part)
    intro_regs = re.findall(r'^(\d+)\.\s+(.*?)(?=^PART|\Z)', text, re.MULTILINE | re.DOTALL)
    for reg_num, reg_text in intro_regs:
        clean_text.append(f"Regulation {reg_num}. {reg_text.strip()}\n")
    
    # Add parts and their regulations
    for part_num, part_data in parts.items():
        clean_text.append(f"\n## PART {part_num}: {part_data['title'].upper()}\n")
        
        for reg in regulations[part_num]:
            if 'sub_regulation_num' in reg:
                clean_text.append(f"Regulation {reg['regulation_num']}.({reg['sub_regulation_num']}) {reg['text']}\n")
            else:
                clean_text.append(f"Regulation {reg['regulation_num']}. {reg['text']}\n")
    
    # Add definitions section
    clean_text.append("\n## DEFINITIONS\n")
    for term, definition in definitions.items():
        clean_text.append(f'"{term}" means {definition}\n')
    
    # Save the cleaned text
    with open(output_file, 'w', encoding='utf-8') as out_file:
        out_file.write('\n'.join(clean_text))
    
    return structured_data

# Function to check for references to schedules and forms in the text
def extract_references(structured_data):
    """
    Extract references to schedules, forms, and tables from the regulations
    """
    references = {
        'schedules': set(),
        'forms': set(),
        'tables': set()
    }
    
    # Check regulations for references
    for part_num, regs in structured_data['regulations'].items():
        for reg in regs:
            text = reg['text']
            
            # Find Schedule references
            schedule_refs = re.findall(r'Schedule\s+(\d+)', text)
            references['schedules'].update(schedule_refs)
            
            # Find Form references
            form_refs = re.findall(r'Form\s+([A-Z])', text)
            references['forms'].update(form_refs)
            
            # Find Table references
            table_refs = re.findall(r'Table\s+(\d+)', text)
            references['tables'].update(table_refs)
    
    # Convert sets to sorted lists
    for key in references:
        references[key] = sorted(list(references[key]))
    
    return references

# Main processing
if __name__ == "__main__":
    # Define input and output file paths relative to the project root
    input_file = "data/cleaned.txt"
    output_file = "data/normalized/normalized_regulations.txt"

    # Process the file
    structured_data = normalize_text(input_file, output_file)
    
    # Extract references
    references = extract_references(structured_data)
    
    # Write references to JSON file
    with open('data/references.json', 'w', encoding='utf-8') as ref_file:
        json.dump(references, ref_file, indent=4, ensure_ascii=False)

    print(f"Processed text saved to {output_file}")
    print(f"Structured data saved to {output_file.replace('.txt', '.json')}") # Use the updated output_file path
    print(f"References saved to data/references.json")
    print(f"Found references to:")
    print(f"  - Schedules: {', '.join(references['schedules'])}")
    print(f"  - Forms: {', '.join(references['forms'])}")
    print(f"  - Tables: {', '.join(references['tables'])}")