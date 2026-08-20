import chromadb
import requests
from sentence_transformers import SentenceTransformer

# Step 1 - client
client = chromadb.Client()

# Step 2 - collection
collection = client.create_collection("data_engineering_docs")

document = """
Informatica IDMC is a cloud-native data management platform. 
It supports data integration, data quality, and master data management.
IDMC uses a REST API for session management and task execution.
The platform supports connectors for Salesforce, SAP, Oracle, and cloud platforms like AWS and Azure.
Data pipelines in IDMC are created using mappings and tasks.
A mapping defines the data flow logic between source and target.
A taskflow orchestrates multiple tasks in a sequence.
IDMC supports both batch and real-time data integration.
"""
model = SentenceTransformer("all-MiniLM-L6-v2")

# Chunk the document by sentence
chunks = [s.strip() for s in document.split(".") if s.strip()]

# Now encode the chunks (list of sentences)
embeddings = model.encode(chunks)

# Store chunks in ChromaDB
collection.add(
    embeddings=embeddings.tolist(),
    documents=chunks,
    ids=[f"id{i}" for i in range(len(chunks))]
)

# Query with a question
question = "How does IDMC handle data pipelines?"
question_embedding = model.encode([question])
results = collection.query(query_embeddings=question_embedding.tolist(), n_results=2)

# Build prompt and send to Ollama

context = "\n".join(results["documents"][0])
prompt = f"""Use the following context to answer the question.
Context: {context}
Question: {question}
Answer:"""

payload = {
    "model": "llama3.2:latest",
    "messages": [{"role": "user", "content": prompt}],
    "stream": False
}
response = requests.post("http://localhost:11434/api/chat", json=payload)
print(response.json()["message"]["content"])

# Peek into ChromaDB — see all stored chunks
all_data = collection.get()

print("Total chunks stored:", len(all_data["ids"]))
print()
for i, (id, doc) in enumerate(zip(all_data["ids"], all_data["documents"])):
    print(f"ID: {id}")
    print(f"Chunk: {doc}")
    print("---")