from langchain.agents import Tool
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import sqlite3
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
import json
from .youtube_helpers import get_youtube_video_ids, fetch_transcript, chunk_documents

def sql_search(query: str) -> str:
    """Search in the company database using natural language that is converted to an sql query by an llm"""
    db = SQLDatabase.from_uri("sqlite:///data/companies.db")
    chain = create_sql_query_chain(ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0), db, k=10)
    result_query = chain.invoke({"question": query})
    con = sqlite3.connect("data/companies.db")
    cursor = con.cursor()
    rows = cursor.execute(result_query).fetchall()
    print(result_query)
    return json.dumps(rows)

def sql_search_tool():
    """Tool to perform SQL searches on the company dataset."""
    return Tool(
        name="company_sql_search",
        func=sql_search,
        description=
        """
        This tool offers detailed company data for advanced searches and analysis. Features include:

        - Unique Identifiers and Names for precise identification.
        - Operational Status to check if companies are active or not.
        - Classification Details for size, sector, and type insights.
        - Incorporation Information with registration dates and places.
        - Capitalization Figures for financial capacity and structure.
        - Industry and Main Activities for understanding market segments.
        - Contact and Location Data for direct communication.
        - Regulatory Oversight to comprehend regulatory contexts.
        - Financial Health from annual returns and financial statements.

        Tailor your query, like "Give average market capitalization. Round to answer.". State to round numbers to two decimal places and specify if seeking an average or a similar calculation.
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
        name="job_description_similarity_search",
        func=job_description_search,
        description=
        """
        The "Job Description Similarity Search" tool matches queries to similar job descriptions, highlighting the top three. Useful for:

        - Finding job descriptions that meet specific criteria.
        - Creating job postings by comparing similar roles.

        Limitations:

        - Can't analyze job descriptions collectively or offer broad market trends.

        Capabilities:

        - Finds job descriptions based on skills or qualifications.
        - Helps benchmark against similar roles.

        Cannot:

        - Identify common skills or analyze salary trends.

        Effective for targeted job description searches and development, it doesn't support aggregate data analysis or market trends.
        """
    )

def measure_len(query: str) -> str:
    """Returns the length of the query as a string."""
    return len(query)

def measure_len_tool():
    """Tool to measure the length of a query."""
    return Tool(
        name="measure_text_length",
        func=measure_len,
        description=
        """
        The "Measure Length of Text" tool calculates the exact length of text inputs, including characters, spaces, and punctuation, crucial for adhering to character limits in social media or SMS.

        Primary Use:
        - Accurately measures text length to meet specific constraints.

        Limitations:
        - Only counts characters; does not assess text readability, quality, or word frequency.

        Appropriate Uses:
        - Ensuring content fits within social media or SMS character limits.

        Inappropriate Uses:
        - Gauging text readability or quality.
        - Analyzing word or phrase frequency.

        This tool is key for precise measurement of text lengths, particularly useful for content with strict character limits. It is not designed for textual analysis or interpretation.
        """
    )

def yt_search(query: str, n: int) -> str:
    """Fetch youtube transcripts via youtube api and use this as input for llm chain
    Args:
    query (str): The question to be answered.
    n (int): Number of YouTube videos to process.

    Returns:
    answer (str): Answer to your question
    """

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

    search_terms_prompt = PromptTemplate(
        input_variables=["text_input"],
        template="I want you to give me a single good youtube search query based on the following prompt:\n\n {text_input}"
    )

    YT_create_search_terms_chain = LLMChain(llm=llm, prompt=search_terms_prompt)

    final_answer_prompt = PromptTemplate(
        input_variables=["chain_output", "query"],
        template="""Use the following information from these youtube transcript summaries:
        \n\n {chain_output} \n To give answer to the following question: {query}"""
    )

    create_final_answer = LLMChain(llm=llm, prompt=final_answer_prompt)

    #Power llm to create effective search terms
    yt_query = YT_create_search_terms_chain.run(query)

    # get top n results from youtube API
    youtube_ids = get_youtube_video_ids(yt_query, n) 

    # Store all transcripts in a list
    transcripts = [fetch_transcript(id, 'en') for id in youtube_ids]
    texts = []
    
    # Loop over transcripts and chunk the data to document format
    for num, transcript in enumerate(transcripts):
        title_and_transcript = f"\nVIDEO {num + 1}:\n {transcript}" 
        if isinstance(transcript, str):
            tanscript_list = chunk_documents(title_and_transcript, 2000, 100)
            texts.extend(tanscript_list)

    # Run summarizaton chain
    summarize_chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
    summary = summarize_chain.run(texts)

    # Answer the initial query based on summary of transcripts
    answer = create_final_answer.run({"chain_output": summary, "query": query}) 
    
    return answer


def yt_search_tool():
    """Tool to answer questions based on youtube transcripts"""
    return Tool(
        name="youtube_search",
        func=yt_search,
        description="Tool to answer questions based on youtube transcripts"
    )