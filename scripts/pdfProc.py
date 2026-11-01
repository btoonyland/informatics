import os
import time
from pathlib import Path
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text

for i, filename in enumerate(os.listdir(download_directory)):
        file_path = os.path.join(download_directory, filename)
        if is_pdf(file_path) == True:
            #open file_path as pdf and then save it to r'C:\\Users\\Administrator\\Documents\\scraped\\dizertacije_management\\useful
            file_to_save = open(file_path, 'rb')
            pdf_reader = PyPDF2.PdfReader(file_to_save)
            pdf_writer = PyPDF2.PdfWriter()
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                pdf_writer.add_page(page)

            # Open the new file in write-binary ('wb') mode and save the PDF
            with open(new_directory+f'pdf{i}', 'wb') as new_file:
                print(new_file)
                pdf_writer.write(new_file)

file_path = 'your_file.pdf'

# 1. Get all PDF files using pathlib (much cleaner than os.listdir/pandas)
pdf_dir = Path("/content/arxiv5")
pdf_files = list(pdf_dir.glob("*.pdf"))

print(f"Found {len(pdf_files)} PDFs to process.")

# 2. Define output files
text_output_path = "/content/part10_text.txt"
meta_output_path = "/content/pdfmined_meta.txt"

# 3. Process loop
# Open both output files once, appending to them as we iterate
with open(text_output_path, "a+", encoding="utf-8") as txt_file, \
     open(meta_output_path, "a+", encoding="utf-8") as meta_file:

    for filepath in pdf_files:
        try:
            print(f"Processing: {filepath.name}")
            
            # --- EXTRACT TEXT (Using pdfminer.six high-level API) ---
            text_content = extract_text(filepath)
            txt_file.write(f"--- File: {filepath.name} ---\n")
            txt_file.write(f"{text_content}\n\n")
            
            # --- EXTRACT METADATA (Using PyPDF2) ---
            reader = PdfReader(filepath)
            info = reader.metadata
            
            meta_file.write(f"--- File: {filepath.name} ---\n")
            meta_file.write(f"{info}\n\n")
            
            # Optional sleep to avoid overloading resources, though usually unnecessary
            time.sleep(0.5)

        except Exception as e:
            # Actually print the error instead of silently passing, 
            # so you know if a specific PDF is broken.
            print(f"Failed to process {filepath.name}: {e}")

print("Extraction complete!")
