import ollama
import sqlglot
from sqlglot import exp

def validate_sql_guardrail(sql_query: str) -> tuple[bool, str]:
    """
    AST Guardrail: Verifies that the SQL is syntactically valid, 
    is strictly a SELECT statement, and contains no data-mutation commands.
    """
    try:
        # Parse the SQL string into an Abstract Syntax Tree (AST)
        parsed = sqlglot.parse_one(sql_query)
        
        # Guardrail 1: Must be a SELECT query
        if not isinstance(parsed, exp.Select):
            return False, "Query rejected: Only SELECT statements are permitted."
            
        # Guardrail 2: Walk the AST tree to block forbidden operations
        forbidden_nodes = (exp.Drop, exp.Delete, exp.Insert, exp.Update, exp.Alter, exp.Command)
        for node in parsed.walk():
            if isinstance(node, forbidden_nodes):
                return False, f"Query rejected: Forbidden operation '{node.key.upper()}' detected."
                
        return True, "Passed guardrail validation."
        
    except Exception as e:
        return False, f"Syntax Error: Could not parse SQL statement ({str(e)})."

def text_to_sql_pipeline(user_prompt: str, schema_ddl: str, model_name: str = "llama3") -> str:
    """
    Calls local Ollama model to generate SQL, then enforces the AST guardrail.
    """
    system_prompt = (
        "You are a SQL generator. Output ONLY raw executable SQL without markdown code blocks, "
        "without explanations, and without formatting tags. Use only SELECT statements."
    )
    
    full_prompt = f"Schema:\n{schema_ddl}\n\nUser Question: {user_prompt}"
    
    # Call local Ollama model
    response = ollama.generate(
        model=model_name,
        prompt=full_prompt,
        system=system_prompt
    )
    
    raw_sql = response['response'].strip().strip("`").replace("sql\n", "").strip()
    
    # Run AST Guardrail
    is_safe, message = validate_sql_guardrail(raw_sql)
    
    if is_safe:
        return f"[SUCCESS] Safe SQL Generated:\n{raw_sql}"
    else:
        return f"[BLOCKED] {message}\nGenerated SQL was: {raw_sql}"

# --- Test Run ---
if __name__ == "__main__":
    sample_schema = """
    CREATE TABLE customer_orders (
        order_id INT,
        customer_id INT,
        amount DECIMAL(10,2),
        order_date DATE,
        status VARCHAR(20)
    );
    """
    
    question = "Show me total order amounts per customer for completed orders in 2026."
    
    # Run pipeline against local Ollama
    result = text_to_sql_pipeline(question, sample_schema)
    print(result)