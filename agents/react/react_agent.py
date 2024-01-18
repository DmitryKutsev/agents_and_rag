from typing import List
from loguru import logger

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import AgentType, Tool, initialize_agent
# /workspaces/agents_and_rag/agents/react/tools.py
from tools import test_len_tool

class myChatGPTReactAgent():
    """
    Class for customizing the React Agent.
    """
    
    def __init__(self):
        self.llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")
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
        # return_intermediate_steps=True,
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
        result = self.agent.run(prompt_template)
        logger.info(f"Got result: {result}")
        return result


if __name__ == "__main__":
    agent = myChatGPTReactAgent()
    tools = [test_len_tool()]
    agent.init_agent(tools)
    agent.run_agent("What is the len of this query?")
