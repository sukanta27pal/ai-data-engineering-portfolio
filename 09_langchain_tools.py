from langchain_core.tools import tool
from langchain_ollama import ChatOllama

# --- Tool 1 - Search IDMC knowledge base ---
@tool
def search_docs(question: str) -> str:
    """Search IDMC knowledge base for relevant information about mappings, tasks, and pipelines."""
    return "IDMC uses mappings and tasks to create data pipelines. A mapping defines the data flow logic between source and target."

# --- Tool 2 - Get live job status from IDMC API ---
@tool
def get_job_status(job_name: str) -> str:
    """Get live job status from IDMC API for a specific job name."""
    return f"Job {job_name} status: FAILED. Error: Connection timeout at source system."

# --- Test tools directly first ---
print("=== Testing tools directly ===")
print(search_docs.invoke("what is a mapping?"))
print(get_job_status.invoke("Job_XYZ"))

# --- Bind tools to LLM ---
print("\n=== Binding tools to LLM ===")
llm = ChatOllama(model="llama3.2:latest")
llm_with_tools = llm.bind_tools([search_docs, get_job_status])

# --- Ask two different questions ---
print("\n=== Question 1 - should use get_job_status ===")
response1 = llm_with_tools.invoke("What is the status of Job_XYZ?")
print("Tool chosen:", response1.tool_calls)

print("\n=== Question 2 - should use search_docs ===")
response2 = llm_with_tools.invoke("What is a mapping in IDMC?")
print("Tool chosen:", response2.tool_calls)
