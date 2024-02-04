from langchain.agents import Tool
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain.chains.summarize import load_summarize_chain
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import sqlite3
import json
from youtube_helpers import get_youtube_video_ids, fetch_transcript, chunk_documents, llm, YT_create_search_terms_chain, create_final_answer

def yt_search(query: str, n: int) -> str:
    """Fetch youtube transcripts via youtube api and use this as input for llm chain
    Args:
    query (str): The question to be answered.
    n (int): Number of YouTube videos to process.

    Returns:
    answer (str): Answer to your question
    """

    #Power llm to create effective search terms
    yt_query = YT_create_search_terms_chain.run(query)
    print(yt_query)

    # get top n results from youtube API
    youtube_ids = get_youtube_video_ids(yt_query, n) 
    print(youtube_ids)

    # Store all transcripts in a list
    transcripts = [fetch_transcript(id, 'en') for id in youtube_ids]
    texts = []
    
    # Loop over transcripts and chunk the data to document format
    for num, transcript in enumerate(transcripts):
        title_and_transcript = f"\nVIDEO {num + 1}:\n {transcript}" 
        if isinstance(transcript, str):
            tanscript_list = chunk_documents(title_and_transcript)
            texts.extend(tanscript_list)

    # Run summarizaton chain
    summarize_chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
    summary = summarize_chain.run(texts)

    # Answer the initial query based on summary of transcripts
    answer = create_final_answer.run({"chain_output": summary, "query": query}) 
    
    return answer


def yt_search_tool():
    """Tool to perform SQL searches on the company dataset."""
    return Tool(
        name="Beginner Youtube search",
        func=yt_search,
        description="Translate question into relevant youtube search quey"
    )

def sql_search(query: str) -> str:
    """Search in the company database using natural language that is converted to an sql query by an llm"""
    db = SQLDatabase.from_uri("sqlite:///data/companies.db")
    chain = create_sql_query_chain(ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0), db)
    result_query = chain.invoke({"question": query})
    con = sqlite3.connect("data/companies.db")
    cursor = con.cursor()
    rows = cursor.execute(result_query).fetchall()
    return json.dumps(rows)

def sql_search_tool():
    """Tool to perform SQL searches on the company dataset."""
    return Tool(
        name="Company dataset SQL search",
        func=sql_search,
        description="Use to ask a non-sql question about the companies dataset."
    )

# def company_sql_search(query: str) -> str:
#     """Search in a company database using an SQL query and return results as a string."""
#     con = sqlite3.connect("data/companies.db")
#     cursor = con.cursor()
#     rows = cursor.execute(query).fetchall()

#     return json.dumps(rows)

# def company_sql_search_tool():
#     """Tool to perform SQL searches on the company dataset."""
#     return Tool(
#         name="Company dataset SQL search",
#         func=company_sql_search,
#         description="Use when you want to query the companies dataset with a valid SQL statement."
#     )

def job_description_search(query: str) -> str:
    """Search in job descriptions using similarity search and return results as a string."""
    embeddings = OpenAIEmbeddings()
    vector_storage = FAISS.load_local("data/", embeddings)

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

print(yt_search("What was the score of the most recent Ajax - PSV and who were the goal scorers?", 7))