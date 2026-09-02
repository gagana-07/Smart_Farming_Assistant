import chromadb
import ollama

client = chromadb.PersistentClient(path="data/vector_db")

collection = client.get_collection("agriculture")

question = input("Ask a farming question: ")

results = collection.query(
    query_texts=[question],
    n_results=3
)

context = "\n".join(results["documents"][0])

response = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "system",
            "content": """
You are an agriculture expert.

Rules:
- Answer only from the provided farming information.
- Give practical farming advice.
- Maximum 5 to 6 lines.
- Do not mention authors.
- Do not mention research papers.
- Do not mention PDF names.
- Do not mention references.
- Do not use bullet points.
- If answer is unavailable, say:
  'I could not find enough information in the farming documents.'
"""
        },
        {
            "role": "user",
            "content": f"""
Question:
{question}

Information:
{context}
"""
        }
    ]
)

answer = response["message"]["content"]

# Remove unwanted markdown symbols
answer = (
    answer.replace("*", "")
          .replace("#", "")
          .replace("-", "")
)

print("\nAssistant:\n")
print(answer)