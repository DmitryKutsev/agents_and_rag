from typing import List
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent, create_openai_tools_agent
from langchain import hub

from .tools.tools import *

class Agent():
    """Base class for all agents."""
    def __init__(self, tools_list: List[Tool] = []):
        self.tools_list = tools_list
    
    def add_tool(self, tool: Tool):
        self.tools_list.append(tool)

    def run_agent(self, query: str) -> str:
        """
        Run the agent.
        """
        if not self.agent:
            raise Exception("""Agent not initialized. Please call init_agent() first.""")

        result = self.agent.invoke({"input": query})

        return result

class ReactAgent(Agent):
    """
    Class for customizing the React Agent.
    """
    
    def __init__(self, tools_list: List[Tool] = []):
        super().__init__(tools_list)
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
        self.template = hub.pull("hwchase17/react")
        
        react_agent = create_react_agent(self.llm, self.tools_list, self.template)
        self.agent = AgentExecutor(agent=react_agent, tools=self.tools_list, return_intermediate_steps=True, verbose=True, handle_parsing_errors=True)
    
class OpenAIToolsAgent(Agent):
    """
    Class for customizing the OpenAI Tools Agent.
    """
    
    def __init__(self, tools_list: List[Tool] = []):
        super().__init__(tools_list)
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
        self.template = hub.pull("hwchase17/openai-tools-agent")

        openai_tools_agent = create_openai_tools_agent(self.llm, self.tools_list, self.template)
        self.agent = AgentExecutor(agent=openai_tools_agent, tools=self.tools_list, return_intermediate_steps=True, verbose=True, handle_parsing_errors=True)
    
def get_agent(agent_type: str = "react"):
    tools = [sql_search_tool(), job_description_search_tool(), measure_len_tool()]
    
    if agent_type == "react":
        agent = ReactAgent(tools)
    elif agent_type == "openai":
        agent = OpenAIToolsAgent(tools)
    else:
        raise ValueError(f"Invalid agent type: {agent_type}. Expected 'react' or 'openai'.")
    
    return agent