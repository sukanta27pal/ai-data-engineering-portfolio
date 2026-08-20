from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# Step 1 - LLM
llm = ChatOllama(model="llama3.2:latest")

# Step 2 - Embeddings (replaces your sentence-transformers encode)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Step 3 - Documents (replaces your manual chunks)
docs = [
    Document(page_content="Informatica IDMC is a cloud-native data management platform."),
    Document(page_content="Data pipelines in IDMC are created using mappings and tasks."),
    Document(page_content="A mapping defines the data flow logic between source and target."),
    Document(page_content="IDMC uses a REST API for session management and task execution."),
    Document(page_content="IDMC supports both batch and real-time data integration."),
]

# Step 4 - Vector store (replaces your chromadb.Client + collection.add)
vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Step 5 - RAG chain
prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context below.
Context: {context}
Question: {question}
""")

question = "How does IDMC handle data pipelines?"
retrieved_docs = retriever.invoke(question)
context = "\n".join([d.page_content for d in retrieved_docs])

chain = prompt | llm
response = chain.invoke({"context": context, "question": question})
print(response.content)
