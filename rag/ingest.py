from pypdf import PdfReader
import chromadb
import os
import json

# ==========================
# CHROMADB SETUP
# ==========================
client = chromadb.PersistentClient(path="data/vector_db")

try:
    client.delete_collection("agriculture")
except:
    pass

collection = client.get_or_create_collection("agriculture")

# ==========================
# DATASET FOLDER
# ==========================
dataset_folder = "datasets/farming_pdfs"
print("Current Working Directory:", os.getcwd())
print("Dataset Folder Exists:", os.path.exists(dataset_folder))
print("Dataset Folder Path:", os.path.abspath(dataset_folder))

# ==========================
# PDF INGESTION
# ==========================
print("\n===== PDF INGESTION =====")
print("\nFiles Found:")
print(os.listdir(dataset_folder))

for filename in os.listdir(dataset_folder):

    if filename.endswith(".pdf"):

        try:
            pdf_path = os.path.join(dataset_folder, filename)

            print(f"Reading PDF: {filename}")

            reader = PdfReader(pdf_path)

            text = ""

            for page in reader.pages:
                extracted = page.extract_text()

                if extracted:
                    text += extracted + "\n"

            if len(text.strip()) > 100:

                chunk_size = 300
                overlap = 50
                chunks = []

                for i in range(0, len(text), chunk_size - overlap):

                    chunk = text[i:i + chunk_size]

                    if len(chunk.strip()) > 50:
                        chunks.append(chunk)

                for idx, chunk in enumerate(chunks):

                    collection.add(
                        documents=[chunk],
                        ids=[f"{filename}_{idx}"]
                    )

                print(f"Added {len(chunks)} chunks from {filename}")

        except Exception as e:
            print(f"PDF Error in {filename}: {e}")

# ==========================
# JSON / JSONL INGESTION
# ==========================
print("\n===== JSON INGESTION =====")

for filename in os.listdir(dataset_folder):

    if filename.endswith(".json") or filename.endswith(".jsonl"):

        try:
            file_path = os.path.join(dataset_folder, filename)

            print(f"Reading JSON: {filename}")

            # JSON
            if filename.endswith(".json"):

                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

            # JSONL
            else:

                data = []

                with open(file_path, "r", encoding="utf-8") as f:

                    for line in f:

                        line = line.strip()

                        if line:
                            data.append(json.loads(line))

            count = 0

            for idx, item in enumerate(data):

                instruction = item.get("instruction", "")
                output = item.get("output", "")

                if instruction or output:

                    text = f"""
Question:
{instruction}

Answer:
{output}
"""

                    collection.add(
                        documents=[text],
                        ids=[f"{filename}_{idx}"]
                    )

                    count += 1

            print(f"Added {count} records from {filename}")

        except Exception as e:
            print(f"JSON Error in {filename}: {e}")

# ==========================
# FINAL COUNT
# ==========================
print("\n=========================")
print("INGESTION COMPLETE")
print("=========================")
print("Total Documents:", collection.count())