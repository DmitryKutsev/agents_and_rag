from langchain.agents import Tool
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain.chains.summarize import load_summarize_chain
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import YoutubeLoader
from langchain.vectorstores import FAISS
import sqlite3
import json
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from youtube_helpers import get_youtube_video_ids, fetch_transcript, chunk_documents, llm, YT_create_search_terms_chain

def yt_search(query: str) -> str:
    """Get access to the proper youtube transcript
    Logic to implement: Create a method to do a API call and fetch the result"""

    #Power llm to create effective search terms
    yt_query = YT_create_search_terms_chain.run(query)

    # get top 5 results from youtube
    youtube_ids = get_youtube_video_ids(yt_query, 5) 

    # Store all 5 transcripts in a list
    transcripts = [fetch_transcript(id, 'en') for id in youtube_ids]
    texts = []
    # Loop over transcripts and create summary based on initial query
    
    for transcript in transcripts:
        #title = Document(page_content=f"\nVIDEO {num + 1}:\n", metadata= {"type": "test"})
        # print(transcript)
        #print("\n\nBIG STOP\n\n")
        #tanscript_list = chunk_documents(transcript)
        #print(len(tanscript_list))
        #texts.extend(title)
        texts.extend(transcript)
    

    doc_creator = RecursiveCharacterTextSplitter(chunk_size = 2000, chunk_overlap = 100)
    document = doc_creator.create_documents(texts = texts)
    


    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
    chain.run(texts)


    # Probably have to also chunk the transcripts and create sub summaries
    # Paste summaries together
    # Try to answer your inital query

    


   

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

print(yt_search("I want to understand the Mount Hall problem"))