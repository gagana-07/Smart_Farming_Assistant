from pypdf import PdfReader
import chromadb
import os

client = chromadb.PersistentClient(path="data/vector_db")

collection = client.get_or_create_collection("agriculture")

pdf_folder = "datasets/farming_pdfs"

for filename in os.listdir(pdf_folder):

    if filename.endswith(".pdf"):

        try:
            pdf_path = os.path.join(pdf_folder, filename)

            print(f"Reading {filename}...")

            reader = PdfReader(pdf_path)

            text = ""

            for page in reader.pages:
                extracted = page.extract_text()

                if extracted:
                    text += extracted + "\n"

            if len(text.strip()) > 100:
                collection.add(
                    documents=[text],
                    ids=[filename]
                )

                print(f"Added: {filename}")

            else:
                print(f"Skipped (no text): {filename}")

        except Exception as e:
            print(f"Error in {filename}: {e}")

print("Ingestion Complete")
print("Total Documents:", collection.count())