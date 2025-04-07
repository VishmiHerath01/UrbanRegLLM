import camelot

# Extract all tables
tables = camelot.read_pdf('Legal Doc/English_Version.pdf', flavor='lattice', pages='all')

# Export to CSV
tables.export('tables.csv', f='csv')

# # Step 3: Extract tables from the corrected PDF and save them as CSV
# tables = camelot.read_pdf('corrected_legal_doc.pdf', flavor='lattice', pages='all')

# # Export tables to CSV
# tables.export('tables.csv', f='csv')

# # If you want to extract and print the tables individually (for review)
# for i, table in enumerate(tables, start=1):
#     print(f"Table {i}:\n", table.df)
