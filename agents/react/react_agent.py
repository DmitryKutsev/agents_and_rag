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

class ReactAgent():
    """
    Class for customizing the React Agent.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
        self.template ="""Answer the following questions as best you can. Use the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [{tool_names}]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nBegin!\n\nQuestion: {query}"""
    
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
        prompt_template = self.template.format(query=query, tool_names=[tool.name for tool in self.agent.tools])

        logger.info(f"Running agent with template: {prompt_template}")
        result = self.agent.invoke(prompt_template)

        logger.info(f"Got result: {result}")
        return result

def get_react_agent():
    agent = ReactAgent()
    tools = [sql_search_tool(), job_description_search_tool()]
    agent.init_agent(tools)
    return agent
    
if __name__ == "__main__":
    load_dotenv(override=True)
    agent = get_react_agent()

    result = agent.run_agent("Find me a candidates with a job description that mentions AI.")