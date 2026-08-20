from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Step 1 - Connect to Ollama (replaces your requests.post setup)
llm = ChatOllama(model="llama3.2:latest")

# Step 2 - Define a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful data engineering assistant."),
    ("human", "{question}")
])

# Step 3 - Create a chain (prompt → llm)
chain = prompt | llm

# Step 4 - Invoke with a question
response = chain.invoke({"question": "What is a data pipeline?"})
print(response.content)                # the answer
print(response.type)                   # "ai"
print(response.usage_metadata)         # token usage
print(response.response_metadata)      # model metadata

