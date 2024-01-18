from typing import List
from loguru import logger

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, Tool, initialize_agent

class myChatGPTReactAgent():
    """
    Class for customizing the React Agent.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        self.template = """
    Given a query {query}, find the information about the question in the query and sent the information as a response.
    """
        
    def get_tools(self, tools_list: List[Tool]) -> List[Tool]:
        return self.tools_list
    
    def init_agent(self):
        self.agent = initialize_agent(
        tools=self.tools_list,
        llm=self.llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
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
