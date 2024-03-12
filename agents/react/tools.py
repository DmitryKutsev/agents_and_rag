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
from react.youtube_helpers import get_youtube_video_ids, fetch_transcript, chunk_documents

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

            It is important to add relevant information to the query to get the best results. For example, "Show me all companies in the technology sector" or "Show me all companies with a capitalization of over 1 million dollars" or "What is the average capitalization of companies in the technology sector?" or "How many companies have been incorporated in the last 5 years?".
            If the result is a number, round it to 2 decimal places.
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
        description=
        """
        The tool, designated as "Job Description Similarity Search," utilizes advanced similarity search techniques to compare a given query against a comprehensive database of job descriptions. Upon receiving a query, the tool promptly identifies and retrieves the top three job descriptions that exhibit the highest similarity scores to the query. This feature is particularly valuable for exploring potential candidates for a role, allowing users to find job descriptions that closely match specific requirements or skill sets.

        Use Cases:

        Ideal for Matching Queries to Job Descriptions: If you're seeking candidates with particular qualifications or experience, this tool can help you find existing job descriptions that closely align with your criteria. This is useful for understanding how similar roles are described in the industry and what qualifications are typically required.
        Beneficial for Tailoring Job Postings: By inputting the desired attributes of a job posting, you can retrieve examples of how similar positions are articulated, aiding in the creation of your own compelling job descriptions.
        Limitations:

        Not Suited for Aggregate Data Analysis: The tool is not designed to analyze or reason about the entire dataset of job descriptions collectively. For instance, it cannot determine the most common skill required in a specific industry by examining all job descriptions simultaneously.
        Ineffective for Broad Market Insights: It cannot provide insights into broader market trends, such as the overall demand for certain roles or the prevalence of specific qualifications across industries.
        Examples of What the Tool Can and Can't Do:

        Can Do:

        Retrieve job descriptions that match a query about specific programming language expertise for a software engineering role.
        Find job descriptions similar to one that emphasizes leadership in project management for benchmarking.
        Can't Do:

        Determine the most frequently requested programming language across all job descriptions in the tech industry.
        Provide an analysis of salary trends for data science roles over the past year.
        This tool stands out for its ability to directly match queries with similar job descriptions, making it an invaluable asset for targeted searches related to hiring and job posting development. However, it's important to recognize its limitations in aggregate data analysis and broad market insights, ensuring users leverage it within its optimal use case scope.
        """ 
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
        name="Youtube search chain",
        func=yt_search,
        description="Tool to answer questions based on youtube transcripts"
    )