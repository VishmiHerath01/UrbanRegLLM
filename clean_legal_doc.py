import fitz  # PyMuPDF
import re
from typing import List, Dict, Set, Tuple
import json
from collections import defaultdict

class LegalDocumentCleaner:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.seen_content = set()  # For deduplication
        
        # Patterns to remove
        self.header_patterns = [
            r"I ෙකොටස.*?2021\.07\.08",  # Sinhala header
            r"PART I.*?08\.07\.2021",     # English header
            r"^\s*\d+\s*A\s*$",           # Page numbers with 'A'
            r"GAZETTE EXTRAORDINARY.*?LANKA",
            r"www\.documents\.gov\.lk",    # Website
            r"Published by Authority",      # Common footer
            r"^[0-9]+A$"                   # Page numbers
        ]
        
        # Form markers
        self.form_markers = [
            "FORM",
            "SCHEDULE",
            "ANNEXURE",
            "APPENDIX"
        ]
        
        # Table markers
        self.table_markers = [
            r"^\s*Table\s+\d+",
            r"^\s*\d+\.\s*[A-Z]",
            r"[\|\+][-\+]+[\|\+]",  # Table borders
            r"^\s*[A-Z]\d+\s+",     # Section codes like B1, A2
            r"^\s*\d+\s*&\s*\d+\s*$",  # Number pairs
            r"^\s*[A-Za-z]+\s+\d+(\.\d+)*\s*$"  # Section numbers
        ]
        
        # Section patterns (ordered by specificity)
        self.section_patterns = [
            (r"^PART [IVX]+[\s\-]*([A-Za-z\s]+)?$", "part"),
            (r"^\d+\.\d+\.", "subsection"),
            (r"^\d+\.", "section"),
            (r"^\([a-z]\)", "item"),
            (r"^\(i+\)", "subitem"),
            (r"^[A-Z]\.\d+", "category"),
            (r"^[A-Z]\.[A-Z]\.\d+", "subcategory")
        ]

    def normalize_text(self, text: str) -> str:
        """Normalize text for deduplication."""
        # Remove all whitespace and convert to lowercase
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        # Remove special characters
        normalized = re.sub(r'[^\w\s]', '', normalized)
        return normalized

    def is_duplicate(self, text: str) -> bool:
        """Check if text is duplicate."""
        normalized = self.normalize_text(text)
        if not normalized or normalized in self.seen_content:
            return True
        self.seen_content.add(normalized)
        return False

    def detect_table_structure(self, lines: List[str]) -> Tuple[bool, int, int]:
        """Detect if a group of lines forms a table and its boundaries."""
        if not lines:
            return False, 0, 0
            
        # Count potential column alignments
        column_positions = defaultdict(int)
        for line in lines:
            words = line.split()
            pos = 0
            for word in words:
                pos += len(word) + 1
                column_positions[pos] += 1
                
        # Check if we have consistent column positions
        consistent_columns = [pos for pos, count in column_positions.items() 
                            if count >= len(lines) * 0.5]
        
        return len(consistent_columns) >= 2, 0, len(lines)

    def clean_table_row(self, row: List[str]) -> List[str]:
        """Clean and format table row content."""
        cleaned_row = []
        for cell in row:
            # Remove extra whitespace
            cell = re.sub(r'\s+', ' ', cell.strip())
            # Remove common artifacts
            cell = re.sub(r'^\s*[\|\+\-]+\s*', '', cell)
            cell = re.sub(r'\s*[\|\+\-]+\s*$', '', cell)
            if cell:
                cleaned_row.append(cell)
        return cleaned_row

    def extract_tables(self, page) -> List[List[str]]:
        """Extract tables from the page with improved structure detection."""
        tables = []
        words = page.get_text("words")
        
        if not words:
            return tables
            
        # Group words by their y-coordinates
        rows = defaultdict(list)
        for w in words:
            y = round(w[3])  # y-coordinate
            rows[y].append((w[0], w[4]))  # x-coordinate and text
            
        # Process rows
        current_table = []
        for y in sorted(rows.keys()):
            row_words = sorted(rows[y], key=lambda x: x[0])  # Sort by x-coordinate
            row_text = [word[1] for word in row_words]
            
            # Clean and check the row
            cleaned_row = self.clean_table_row(row_text)
            if cleaned_row:
                # Check if this row looks like a table row
                if len(cleaned_row) >= 2 and any(col.strip().isdigit() for col in cleaned_row):
                    current_table.append(cleaned_row)
                else:
                    # If we have a current table and this isn't a table row, save it
                    if current_table:
                        tables.append(current_table)
                        current_table = []
        
        # Add any remaining table
        if current_table:
            tables.append(current_table)
            
        return tables

    def is_table_row(self, line: str) -> bool:
        """Check if a line appears to be part of a table."""
        # Check if line matches any table markers
        for pattern in self.table_markers:
            if re.match(pattern, line):
                return True
        
        # Check for consistent spacing that might indicate columns
        spaces = [m.start() for m in re.finditer(r'\s{2,}', line)]
        if len(spaces) >= 2:
            # Check if spaces are somewhat evenly distributed
            diffs = [spaces[i+1] - spaces[i] for i in range(len(spaces)-1)]
            avg_diff = sum(diffs) / len(diffs)
            if all(abs(d - avg_diff) <= 2 for d in diffs):
                return True
        
        return False

    def process_document(self) -> Dict[str, List[str]]:
        """Process the entire document with improved structure handling."""
        content = {
            'main_text': [],
            'tables': [],
            'forms': [],
            'sections': []  # Track section structure
        }
        
        current_section = {
            'part': None,
            'section': None,
            'subsection': None,
            'text': []
        }
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            text = page.get_text()
            
            # Skip empty pages
            if not text.strip():
                continue
            
            # Extract tables from the page
            tables = self.extract_tables(page)
            if tables:
                content['tables'].extend(tables)
            
            # Process text line by line
            lines = text.split('\n')
            buffer = []
            in_table = False
            current_table = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for headers/footers
                if any(re.search(pattern, line) for pattern in self.header_patterns):
                    continue
                
                # Check for forms
                if any(marker in line for marker in self.form_markers):
                    if buffer:
                        content['main_text'].extend(buffer)
                        buffer = []
                    content['forms'].append(line)
                    continue
                
                # Check for section markers
                section_type = None
                for pattern, s_type in self.section_patterns:
                    if re.match(pattern, line):
                        section_type = s_type
                        break
                
                if section_type:
                    # Save current buffer if exists
                    if buffer:
                        content['main_text'].extend(buffer)
                        buffer = []
                    
                    # Update section tracking
                    if section_type == 'part':
                        current_section['part'] = line
                        current_section['section'] = None
                        current_section['subsection'] = None
                    elif section_type == 'section':
                        current_section['section'] = line
                        current_section['subsection'] = None
                    elif section_type == 'subsection':
                        current_section['subsection'] = line
                    
                    # Add to sections list
                    content['sections'].append({
                        'type': section_type,
                        'content': line,
                        'hierarchy': current_section.copy()
                    })
                    continue
                
                # Handle table detection
                if self.is_table_row(line):
                    if not in_table:
                        in_table = True
                        if buffer:
                            content['main_text'].extend(buffer)
                            buffer = []
                    current_table.append(line)
                else:
                    if in_table:
                        in_table = False
                        if current_table:
                            content['tables'].append(current_table)
                            current_table = []
                    if not self.is_duplicate(line):
                        buffer.append(line)
            
            # Handle any remaining content
            if buffer:
                content['main_text'].extend(buffer)
            if current_table:
                content['tables'].append(current_table)
        
        return content

    def save_cleaned_content(self, output_path: str):
        """Save the cleaned content with improved organization."""
        content = self.process_document()
        
        # Save main text with section structure
        with open(output_path, 'w', encoding='utf-8') as f:
            for section in content['sections']:
                f.write(f"\n{'=' * 80}\n")
                f.write(f"Type: {section['type'].upper()}\n")
                f.write(f"Content: {section['content']}\n")
                if section['hierarchy']['part']:
                    f.write(f"Part: {section['hierarchy']['part']}\n")
                if section['hierarchy']['section']:
                    f.write(f"Section: {section['hierarchy']['section']}\n")
                if section['hierarchy']['subsection']:
                    f.write(f"Subsection: {section['hierarchy']['subsection']}\n")
                f.write('=' * 80 + '\n\n')
            
            f.write('\n'.join(content['main_text']))
        
        # Save tables with structure information
        with open(output_path.replace('.txt', '_tables.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'tables': content['tables'],
                'metadata': {
                    'num_tables': len(content['tables']),
                    'table_sizes': [len(table) for table in content['tables']]
                }
            }, f, indent=2, ensure_ascii=False)
        
        # Save forms with better organization
        with open(output_path.replace('.txt', '_forms.txt'), 'w', encoding='utf-8') as f:
            for i, form in enumerate(content['forms'], 1):
                f.write(f"\n{'#' * 80}\n")
                f.write(f"Form {i}\n")
                f.write('#' * 80 + '\n\n')
                f.write(form + '\n\n')

if __name__ == "__main__":
    pdf_path = "Legal Doc/English_Version.pdf"
    cleaner = LegalDocumentCleaner(pdf_path)
    cleaner.save_cleaned_content("cleaned_legal_doc.txt")
