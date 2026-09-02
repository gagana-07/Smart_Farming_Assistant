from pypdf import PdfReader
import chromadb
import os

client = chromadb.PersistentClient(path="data/vector_db")

collection = client.get_or_create_collection("agriculture")

pdf_folder = "datasets/farming_pdfs"

for filename in os.listdir(pdf_folder):

    if filename.endswith(".pdf"):

        pdf_path = os.path.join(pdf_folder, filename)

        print(f"Reading {filename}...")

        reader = PdfReader(pdf_path)

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        # Split PDF into chunks
        chunk_size = 1000

        for i in range(0, len(text), chunk_size):

            chunk = text[i:i + chunk_size]

            collection.add(
                documents=[chunk],
                ids=[f"{filename}_{i}"]
            )

print("All PDFs added successfully!")