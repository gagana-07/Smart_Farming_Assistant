import ollama

response = ollama.chat(
    model="llama3.2",
    messages=[
        {"role": "user", "content": "How to grow rice?"}
    ]
)

print(response["message"]["content"])