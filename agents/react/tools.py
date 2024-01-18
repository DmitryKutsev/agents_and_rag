"""Tools for the react agent."""
from langchain.agents import Tool


def sql_search_tool():
    pass

def vector_search_tool():
    pass

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
    return len(query)