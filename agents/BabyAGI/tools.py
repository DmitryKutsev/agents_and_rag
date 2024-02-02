from langchain.agents import Tool
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import sqlite3
import json

def sql_query_chain(query: str) -> str:
    """Search in the company database using natural language that is converted to an sql query by an llm"""
    db = SQLDatabase.from_uri("sqlite:///data/companies.db")
    chain = create_sql_query_chain(ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0), db)
    result_query = chain.invoke({"question": query})
    con = sqlite3.connect("data/companies.db")
    cursor = con.cursor()
    rows = cursor.execute(result_query).fetchall()
    return json.dumps(rows)

def sql_llm_search_tool():
    """Tool to perform SQL searches on the company dataset."""
    return Tool(
        name="Company dataset SQL search",
        func=sql_query_chain,
        description="Use to ask a non-sql question about the companies dataset."
    )

def company_sql_search(query: str) -> str:
    """Search in a company database using an SQL query and return results as a string."""
    con = sqlite3.connect("../data/companies.db")
    cursor = con.cursor()
    rows = cursor.execute(query).fetchall()

    return json.dumps(rows)

def sql_search_tool():
    """Tool to perform SQL searches on the company dataset."""
    return Tool(
        name="Company dataset SQL search",
        func=company_sql_search,
        description="Use when you want to query the companies dataset with a valid SQL statement."
    )

def job_description_search(query: str) -> str:
    """Search in job descriptions using similarity search and return results as a string."""
    embeddings = OpenAIEmbeddings()
    vector_storage = FAISS.load_local("data/faiss_descriptions_index_cut", embeddings)

    results = vector_storage.similarity_search_with_score(query, fetch_k=3)
    
    combined_content = ""
    for i, (doc, probability) in enumerate(results, start=1):
        page_content = doc.page_content  
        combined_content += f"Result {i} (Probability: {probability:.2f}):\n{page_content}\n\n"

    return combined_content

def job_description_search_tool():
    """Tool to perform similarity searches on job descriptions."""
    return Tool(
        name="Job description search",
        func=job_description_search,
        description="Use when you want to do a similarity search over job descriptions."
    )

def measure_len(query: str) -> str:
    """Returns the length of the query as a string."""
    return len(query)

def measure_len_tool():
    """Tool to measure the length of a query."""
    return Tool(
        name="Measure length of query",
        func=measure_len,
        description="Use when you need to measure the length of the query",
    )