from typing import List
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent, create_openai_tools_agent
from langchain import hub
from agents.base_agent import Agent
from agents.tools.tools import *
import time

class SingleAgentSystem(Agent):
    def run_agent(self, query: str) -> str:
        """
        Run the agent.
        """
        if not self.agent:
            raise Exception("Agent not initialized.")

        start = time.time()
        result = self.agent.invoke({"input": query})
        end = time.time()
        
        return result, end-start

class ReactAgent(SingleAgentSystem):
    """
    Class for customizing the React Agent.
    """
    
    def __init__(self, tools_list: List[Tool] = []):
        self.tools_list = tools_list
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
        self.template = hub.pull("hwchase17/react")
        
        react_agent = create_react_agent(self.llm, self.tools_list, self.template)
        self.agent = AgentExecutor(agent=react_agent, tools=self.tools_list, return_intermediate_steps=True, verbose=True, handle_parsing_errors=True)

    def format_agent_response(self, output):
        """
        Format the agent's output.
        """
        
        # Initialize the result dictionary
        result = {
            'output': output.get('output', ''),
            'agent_trajectory': '',
            'steps': []
        }

        # Extract intermediate steps
        intermediate_steps = output.get('intermediate_steps', [])
        trajectory_steps = []

        # Iterate through each step
        for step_number, (action, observation) in enumerate(intermediate_steps, start=1):
            
            # Depending on the agent type, the log may be formatted differently
            try:
                log = action.log.split('\n')[0]
            except:
                log = ""

            step_info = {
                'step': step_number,
                'tool': action.tool,
                'tool_input': action.tool_input,
                'log': log,
                'observation': observation
            }
            result['steps'].append(step_info)

            # Add a formatted string for this step to the trajectory_steps list
            trajectory_step_str = f"Step {step_number}: Tool=[{action.tool}], Input=[{action.tool_input}], Log=[{log}], Observation=[{observation}]"
            trajectory_steps.append(trajectory_step_str)

        # Join the trajectory steps into a single string and add it to the result
        result['agent_trajectory'] = '\n'.join(trajectory_steps)

        return result

class OpenAIToolsAgent(SingleAgentSystem):
    """
    Class for customizing the OpenAI Tools Agent.
    """
    
    def __init__(self, tools_list: List[Tool] = []):
        self.tools_list = tools_list
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
        self.template = hub.pull("hwchase17/openai-tools-agent")

        openai_tools_agent = create_openai_tools_agent(self.llm, self.tools_list, self.template)
        self.agent = AgentExecutor(agent=openai_tools_agent, tools=self.tools_list, return_intermediate_steps=True, verbose=True, handle_parsing_errors=True)

    def format_agent_response(self, output):
        """
        Format the agent's output.
        """
        
        # Initialize the result dictionary
        result = {
            'output': output.get('output', ''),
            'agent_trajectory': '',
            'steps': []
        }

        # Extract intermediate steps
        intermediate_steps = output.get('intermediate_steps', [])
        trajectory_steps = []
        
        # Iterate through each step
        for step_number, (action, observation) in enumerate(intermediate_steps, start=1):
            
            # Depending on the agent type, the log may be formatted differently
            try:
                log = action.log.split('\n')[1]
            except:
                log = ""

            step_info = {
                'step': step_number,
                'tool': action.tool,
                'tool_input': action.tool_input,
                'log': log,
                'observation': observation
            }
            result['steps'].append(step_info)

            # Add a formatted string for this step to the trajectory_steps list
            trajectory_step_str = f"Step {step_number}: Tool=[{action.tool}], Input=[{action.tool_input}], Log=[{log}], Observation=[{observation}]"
            trajectory_steps.append(trajectory_step_str)

        # Join the trajectory steps into a single string and add it to the result
        result['agent_trajectory'] = '\n'.join(trajectory_steps)

        return result
    
def get_agent(agent_type: str = "react"):
    tools = [sql_search_tool(), job_description_search_tool()]
    
    if agent_type == "react":
        agent = ReactAgent(tools)
    elif agent_type == "openai":
        agent = OpenAIToolsAgent(tools)
    else:
        raise ValueError(f"Invalid agent type: {agent_type}. Expected 'react' or 'openai'.")
    
    return agent