import chromadb
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer

# --- Setup (runs once) ---
@st.cache_resource
def setup():
    client = chromadb.Client()
    collection = client.create_collection("idmc_docs")
    model = SentenceTransformer("all-MiniLM-L6-v2")

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

    chunks = [s.strip() for s in document.split(".") if s.strip()]
    embeddings = model.encode(chunks)
    collection.add(
        embeddings=embeddings.tolist(),
        documents=chunks,
        ids=[f"id{i}" for i in range(len(chunks))]
    )
    return collection, model

# --- UI ---
st.title("IDMC Knowledge Assistant")
st.write("Ask any question about Informatica IDMC")

collection, model = setup()

question = st.text_input("Your question:")

if st.button("Ask"):
    if question:
        question_embedding = model.encode([question])
        results = collection.query(query_embeddings=question_embedding.tolist(), n_results=2)
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
        answer = response.json()["message"]["content"]

        st.subheader("Answer:")
        st.write(answer)

        st.subheader("Context used:")
        st.info(context)
