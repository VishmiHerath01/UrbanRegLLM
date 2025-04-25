# import PyPDF2

# def extract_text_from_pdf(pdf_path):
#     """
#     Extract text from a PDF file
    
#     Args:
#         pdf_path (str): Path to the PDF file
        
#     Returns:
#         str: Extracted text from the PDF
#     """
#     # Create a text string to store the extracted text
#     text = ""
    
#     try:
#         # Open the PDF file in read-binary mode
#         with open(pdf_path, 'rb') as file:
#             # Create a PDF reader object
#             pdf_reader = PyPDF2.PdfReader(file)
            
#             # Get the number of pages in the PDF
#             num_pages = len(pdf_reader.pages)
#             print(f"Total pages: {num_pages}")
            
#             # Loop through all pages and extract text
#             for page_num in range(num_pages):
#                 # Get the page object
#                 page = pdf_reader.pages[page_num]
                
#                 # Extract text from the page
#                 page_text = page.extract_text()
                
#                 # Add the page text to our string
#                 text += page_text + "\n\n"
                
#         return text
    
#     except Exception as e:
#         print(f"Error extracting text: {e}")
#         return None

# # Example usage
# if __name__ == "__main__":
#     pdf_path = "Legal Doc/English_Version.pdf"  
#     extracted_text = extract_text_from_pdf(pdf_path)
    
#     if extracted_text:
#         print("Text extracted successfully!")
        
#         # Print first 500 characters as a preview
#         print("\nPreview of extracted text:")
#         print(extracted_text[:500] + "...")
        
#         # Save the extracted text to a file
#         with open("../data/extracted/extracted_text_new.txt", "w", encoding="utf-8") as text_file:
#             text_file.write(extracted_text)
#         print("Text saved to '../data/extracted/extracted_text.txt'")
import os
import shutil

# Source directory where the CSV files currently are
source_dir = os.getcwd()  

# Destination directory 
destination_dir = "/Users/vishmiherath/Documents/FYP/table_data"

# Ensure the destination folder exists
os.makedirs(destination_dir, exist_ok=True)

# Loop through files in the source directory
for file_name in os.listdir(source_dir):
    if file_name.endswith(".csv"):
        src_file = os.path.join(source_dir, file_name)
        dst_file = os.path.join(destination_dir, file_name)
        shutil.move(src_file, dst_file)
        print(f"Moved: {file_name} -> {destination_dir}")

print("All CSV files moved successfully.")
