import chromadb
from sentence_transformers import SentenceTransformer

# Step 1 - client
client = chromadb.Client()

# Step 2 - collection
collection = client.create_collection("data_engineering_docs")

# Step 3 - your 3 sentences (you define these)
docs = ["ETL, which stands for Extract, Transform, and Load, is a fundamental data integration process used to copy data from multiple source systems into a single centralized database or data warehouse", "First, raw data is extracted from various platforms like apps, CRMs, or spreadsheets, then transformed by cleansing, reformatting, and filtering it to ensure high quality and consistency", "Finally, the processed data is loaded into a target destination, such as a data warehouse, where businesses can easily analyze it to make informed, data-driven decisions"]

# Step 4 - embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(docs)

# Step 5 - add to ChromaDB
collection.add(
    embeddings=embeddings.tolist(),
    documents=docs,
    ids=["id1", "id2", "id3"]
)

# Step 6 - query
query = "what is ETL?"
query_embedding = model.encode([query])
results = collection.query(query_embeddings=query_embedding.tolist(), n_results=1)
print(results["documents"])
