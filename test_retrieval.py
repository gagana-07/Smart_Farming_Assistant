"""
Check what the database returns for typed questions (no voice involved).
Run from the project root:   python test_retrieval.py
Edit QUESTIONS to match topics your data should cover.
"""
import chromadb

QUESTIONS = [
    "how much water does ragi need",
    "which fertilizer is best for paddy",
    "how to control pests in tomato",
    "when to sow maize",
    "who won the cricket match", 
    "how to cook biryani"
]

col = chromadb.PersistentClient(path="data/vector_db").get_collection("agriculture")
print("Documents in database:", col.count())

for q in QUESTIONS:
    res = col.query(
        query_texts=[q],
        n_results=3,
        include=["documents", "distances", "metadatas"],
    )
    print("\nQ:", q)
    for doc, dist, meta in zip(
        res["documents"][0], res["distances"][0], res["metadatas"][0]
    ):
        src = (meta or {}).get("source", "?")
        print(f"  {dist:.2f}  [{src}]  {doc[:160]!r}")