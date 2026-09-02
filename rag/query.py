import chromadb
import ollama

# Connect to ChromaDB
client = chromadb.PersistentClient(path="data/vector_db")

# Open collection
collection = client.get_collection("agriculture")

# Ask question
question = input("Ask a farming question: ")

# Retrieve only top 3 results
results = collection.query(
    query_texts=[question],
    n_results=3
)

# Build context
context = "\n".join(results["documents"][0])

# Limit context size
context = context[:1200]
print("\nQUESTION:")
print(question)

print("\nCONTEXT SENT TO OLLAMA:")
print(context[:1500])

response = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "system",
            "content": """
You are an agriculture expert helping farmers.

IMPORTANT RULES:
- Answer the farmer's question directly.
- Give practical farming advice only.
- Maximum 5 to 6 lines.
- Never mention authors.
- Never mention references.
- Never mention journals.
- Never mention research papers.
- Never mention studies.
- Never mention universities.
- Never summarize documents.
- Never explain what the document contains.
- If the context is unrelated to the question, reply:
  I could not find a relevant farming answer.
"""
        },
        {
            "role": "user",
            "content": f"""
Farmer Question:
{question}

Relevant Farming Information:
{context}

Provide only the farming answer.
"""
        }
    ]
)

answer = response["message"]["content"]

# Clean output
for ch in ["*", "#", "-", "•"]:
    answer = answer.replace(ch, "")

print("\nAssistant:\n")
print(answer.strip())