"""Tools for the react agent."""
from langchain.agents import Tool
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import sqlite3

import pandas as pd

def company_sql_search(query: str) -> list:
    con = sqlite3.connect("companies.db")
    cursor = con.cursor()
    return cursor.execute(query).fetchall()

def sql_search_tool():
    return Tool(
        name="Company dataset SQL search",
        func=company_sql_search,
        description="Use when you want to query the companies dataset with a valid SQL statement."
    )

def job_description_search(query: str):
    embeddings = OpenAIEmbeddings()
    descriptions_df = pd.read_csv("job_descriptions.csv", sep=";")
    weird_substring = "Job Description Â\xa0 Send me Jobs like this"
    description_list = [ description.replace(weird_substring, "").lower().replace("full stack", "")
                     for description in descriptions_df["jobdescription"]
                     if len(description) > 200 ]
    vector_storage = FAISS.from_texts(description_list, embeddings)
    return vector_storage.similarity_search(query)

def job_description_search_tool():
    """Use this tool search the job descriptions dataset."""
    return Tool(
        name="Job description search",
        func=job_description_search,
        description="Use when you want to do a similarity search over job descriptions."
    )

def measure_len(query: str) -> str:
    """Use this func for test_len_tool"""
    return len(query)


def test_len_tool():
    """Use this tool to measure the len of the query"""
    return Tool(
    name="Measure length of query",
    func=measure_len,
    description="Use when you need to measure the length of the query",
    )