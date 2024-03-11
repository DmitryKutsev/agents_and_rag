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
# from youtube_helpers import get_youtube_video_ids, fetch_transcript, chunk_documents, llm, YT_create_search_terms_chain, create_final_answer

# def yt_search(query: str, n: int) -> str:
#     """Fetch youtube transcripts via youtube api and use this as input for llm chain
#     Args:
#     query (str): The question to be answered.
#     n (int): Number of YouTube videos to process.

#     Returns:
#     answer (str): Answer to your question
#     """

#     #Power llm to create effective search terms
#     yt_query = YT_create_search_terms_chain.run(query)

#     # get top n results from youtube API
#     youtube_ids = get_youtube_video_ids(yt_query, n) 

#     # Store all transcripts in a list
#     transcripts = [fetch_transcript(id, 'en') for id in youtube_ids]
#     texts = []
    
#     # Loop over transcripts and chunk the data to document format
#     for num, transcript in enumerate(transcripts):
#         title_and_transcript = f"\nVIDEO {num + 1}:\n {transcript}" 
#         if isinstance(transcript, str):
#             tanscript_list = chunk_documents(title_and_transcript, 2000, 100)
#             texts.extend(tanscript_list)

#     # Run summarizaton chain
#     summarize_chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
#     summary = summarize_chain.run(texts)

#     # Answer the initial query based on summary of transcripts
#     answer = create_final_answer.run({"chain_output": summary, "query": query}) 
    
#     return answer


# def yt_search_tool():
#     """Tool to answer questions based on youtube transcripts"""
#     return Tool(
#         name="Youtube search chain",
#         func=yt_search,
#         description="Tool to answer questions based on youtube transcripts"
#     )

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
        description=
        """
            This tool is designed to enable in-depth searches and analyses of comprehensive company data. It offers access to a wide array of information that includes:

            Unique Identifiers and Names: Each company is cataloged with a unique identification number and name, ensuring precise identification within the database.
            Operational Status: The tool provides the current operational status of companies, indicating whether they are active, inactive, or undergoing processes like liquidation.
            Classification Details: Companies are classified based on their size and operation type, with detailed information on the company's class, category, and sub-category, giving insights into the scale and sector of operation.
            Incorporation Information: It includes the date and state of registration, offering insights into the company's inception and geographic jurisdiction.
            Capitalization Figures: Detailed capital information, including authorized and paid-up capital, reveals a company's financial capacity and equity structure.
            Industry and Main Activities: The tool classifies companies by industrial class and details their principal business activities, providing a clear view of the primary market segments they operate in.
            Contact and Location Data: Registered office addresses and email addresses are available for direct contact and geographic pinpointing.
            Regulatory Oversight: Information regarding the Registrar of Companies overseeing each entity helps users understand the regulatory environment the company operates within.
            Financial Health: Access to the latest annual returns and financial statements offers insights into a company's financial performance and condition.
        """ 
    )

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
    """
    A resourceful tool for accessing a comprehensive collection of job descriptions. This tool is particularly useful for queries related to job roles, titles, functions, and detailed descriptions of various positions across industries. It serves as a valuable reference for understanding job requirements, responsibilities, and qualifications, facilitating in-depth analysis and insights into specific job-related information.
    """
    return Tool(
        name="Job Description Search",
        func=job_description_search,
        description="Designed for exploring and analyzing detailed job descriptions. Ideal for inquiries about job titles, roles, functions, and the specific requirements or responsibilities associated with different positions."
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

