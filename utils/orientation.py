# import fitz  # PyMuPDF

# # Open the original PDF
# pdf_path = "Legal Doc/English_Version.pdf"
# doc = fitz.open(pdf_path)

# # Initialize an empty list to store the page numbers with horizontally rotated text
# rotated_text_pages = []

# # Iterate over the pages and check for horizontal text
# for i in range(doc.page_count):
#     page = doc.load_page(i)  # Load the page
#     text = page.get_text("text")  # Extract the raw text from the page

#     # Check if the text is mostly horizontal (rotated)
#     if page.rotation in [90, 270]:  # If the page is rotated itself
#         rotated_text_pages.append(i + 1)  # Store the page number (1-based index)
    
#     # Optionally, if you want to print the text extracted (for testing purposes):
#     # print(text)

# # Print the page numbers with horizontally rotated text
# print("Pages with horizontally rotated text:", rotated_text_pages)

# # Save the rotated page numbers to a .txt file
# with open("data/rotated_pages.txt", "w") as f:
#     for page_number in rotated_text_pages:
#         f.write(str(page_number) + "\n")
