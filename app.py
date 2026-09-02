import whisper
import ollama
import pyttsx3
import chromadb

print("Loading Whisper...")
model = whisper.load_model("base")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="data/vector_db")
collection = client.get_collection("agriculture")

print("Transcribing audio...")

result = model.transcribe("audio/input.wav")
question = result["text"].strip()

print("\nFarmer:")
print(question)

# Search relevant PDF chunks
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
You are a smart agriculture assistant.

Rules:
Answer only the user's farming question.
Give 5 to 6 short lines.
Use simple language.
Provide practical farming advice.
Do not mention authors.
Do not mention PDF names.
Do not mention research papers.
Do not mention references.
Do not use bullet points.
Do not use special symbols.
If information is not found, say:
Sorry, I could not find enough information.
"""
        },
        {
            "role": "user",
            "content": f"""
Question:
{question}

Context:
{context}
"""
        }
    ]
)

answer = response["message"]["content"]

# Clean answer
for ch in ["*", "-", "•", "#"]:
    answer = answer.replace(ch, "")

print("\nAssistant:\n")
print(answer)

# Speak answer
engine = pyttsx3.init()
engine.setProperty("rate", 170)
engine.say(answer)
engine.runAndWait()