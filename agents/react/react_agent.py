import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from typing import List
from loguru import logger
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, Tool, initialize_agent

from react.tools import *

class myChatGPTReactAgent():
    """
    Class for customizing the React Agent.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
        self.template = """
    Given a query {query}, find the information about the question in the query and sent the information as a response.
    """

    def add_tool(self, tool: Tool):
        self.tools_list.append(tool)
    
    def init_agent(self, tools_list: List[Tool]):
        """"
        Initialize the agent.
        """
        self.agent = initialize_agent(
        tools=tools_list,
        llm=self.llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        return_intermediate_steps=True, 
        verbose=True,
        handle_parsing_errors=True,
    )

    def run_agent(self, query: str) -> str:
        """
        Run the agent.
        """
        if not self.agent:
            raise Exception("""Agent not initialized. Please call init_agent() first.""")
        prompt_template = self.template.format(query=query)

        logger.info(f"Running agent with template: {prompt_template}")
        result = self.agent.invoke(prompt_template)

        logger.info(f"Got result: {result}")
        return result

def get_react_agent():
    agent = myChatGPTReactAgent()
    tools = [measure_len_tool(), sql_search_tool(), job_description_search_tool()]
    agent.init_agent(tools)
    return agent
    
if __name__ == "__main__":
    load_dotenv()
    agent = get_react_agent()
    
    #result = agent.run_agent("What is the len of this query?")
    #result = agent.run_agent("How many companies are there in the databse?")
    result = agent.run_agent("Find me a candidates with a job description that mentions AI.")