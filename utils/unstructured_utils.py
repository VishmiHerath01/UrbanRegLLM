from unstructured.partition.pdf import partition_pdf

# Replace with your file path
file_path = "Legal Doc/English_Version.pdf"

# Partition the PDF
elements = partition_pdf(filename=file_path)

# Print first few elements
for i, element in enumerate(elements[:10]):  # just show first 10
    print(f"\n--- Element {i+1} ---")
    print(f"Type: {element.category}")
    print(element.text)
