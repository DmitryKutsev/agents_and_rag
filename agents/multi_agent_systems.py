
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
import operator
from typing import Annotated, Sequence, TypedDict, Tuple
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.agents import AgentAction
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from agents.tools.tools import *
from langchain_core.messages import AIMessage
from typing import Any, Dict, List
from agents.base_agent import Agent


class BaseAgentNode:
    def __init__(self, llm: ChatOpenAI, tools: List[Any], system_prompt: str, name: str):
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.name = name
        self.agent = self.create_agent()

    def create_agent(self) -> AgentExecutor:
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt,),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, handle_parsing_errors=True, return_intermediate_steps=True)

    def execute(self, state: Dict[str, Any]):
        result = self.agent.invoke(state)

        agent_trajectory = [AgentAction(tool=action.tool, tool_input=action.tool_input, log=action.log) for action, _ in result["intermediate_steps"]]

        messages = [
            AIMessage(
                content=result["output"], 
                name=self.name, 
            )
        ]

        return {
            "messages": messages,
            "agent_trajectory": agent_trajectory,
            }

class SQLAgentNode(BaseAgentNode):
    def __init__(self, llm: ChatOpenAI, tools: List[Any], system_prompt: str):
        super().__init__(llm, tools, system_prompt, name="SQL")


class VSAgentNode(BaseAgentNode):
    def __init__(self, llm: ChatOpenAI, tools: List[Any], system_prompt: str):
        super().__init__(llm, tools, system_prompt, name="VS")

class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    agent_trajectory: List[Tuple[AgentAction, str]]
    # The 'next' field indicates where to route to next
    next: str

class AgentGraph(Agent):
    def __init__(self, llm: ChatOpenAI, sql_tools: List[Any], vs_tools: List[Any], system_prompt: str, sql_system_prompt: str, vs_system_prompt: str):
        self.graph = StateGraph(AgentState)
        self.llm = llm
        self.initialize_graph(sql_tools, vs_tools, system_prompt, sql_system_prompt, vs_system_prompt)

    def initialize_graph(self, sql_tools: List[Any], vs_tools: List[Any], system_prompt: str, sql_system_prompt: str, vs_system_prompt: str):
        sql_agent_node = SQLAgentNode(self.llm, sql_tools, sql_system_prompt)
        vs_agent_node = VSAgentNode(self.llm, vs_tools, vs_system_prompt)

        members = ["SQL", "VS"]
        supervisor_chain = self.create_supervisor_chain(system_prompt, members)
        
        self.graph.add_node("SQL", sql_agent_node.execute)
        self.graph.add_node("VS", vs_agent_node.execute)
        self.graph.add_node("supervisor", supervisor_chain)
        
        for member in members:
            self.graph.add_edge(member, "supervisor")
        
        conditional_map = {k: k for k in members}
        conditional_map["FINISH"] = END
        self.graph.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
        self.graph.set_entry_point("supervisor")

        self.graph = self.graph.compile()

    def create_supervisor_chain(self, prompt: str, members: List[str]):
        # Function definition for routing
        function_def = {
            "name": "route",
            "description": "Select the next role.",
            "parameters": {
                "title": "routeSchema",
                "type": "object",
                "properties": {"next": {"title": "Next", "anyOf": [{"enum": members}, ], }},
                "required": ["next"],
            },
        }

        # Define supervisor chain
        supervisor_chain = (
                prompt
                | self.llm.bind_functions(functions=[function_def], function_call="route")
                | JsonOutputFunctionsParser()
        )

        return supervisor_chain

    def execute_graph(self, input_message: str) -> str:
        response = self.graph.invoke(
            {
                "messages": [HumanMessage(content=input_message)]
            },
            {"recursion_limit": 100}
        )
        # Change to 'return response['messages'][-1].content' to just return the last message
        return response

    # Implements the run_agent method in the Agent class
    def run_agent(self, query: str) -> str:
        return self.execute_graph(query)
    
    # Implement the format_agent_response method in the Agent class
    def format_agent_response(self, output):
        """
        Format the output for a multi-agent system.
        """
        
        # Initialize the result dictionary
        result = {
            'output': output['messages'][-1].content,
            'agent_trajectory': '',
            'steps': []
        }

        # Extract agent trajectories
        agent_trajectories = output.get('agent_trajectory', [])
        trajectory_steps = []

        # Iterate through each step in the trajectories
        for step_number, action in enumerate(agent_trajectories, start=1):
            
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
                'observation': "TODO: figure out how to add the observation/tool output here." # TODO: figure out how to add the observation/tool output here.
            }
            result['steps'].append(step_info)

            # Add a formatted string for this step to the trajectory_steps list
            trajectory_step_str = f"Step {step_number}: Tool=[{action.tool}], Input=[{action.tool_input}], Log=[{log}]"
            trajectory_steps.append(trajectory_step_str)

        # Iterate through each agent in the response
        for agent in output['messages']:
            if isinstance(agent, AIMessage):
                result['agent_trajectory'] += f"{agent.name} Agent Response:\n{agent.content}\n"

        # Join the trajectory steps into a single string and add it to the result
        result['agent_trajectory'] += '\n'.join(trajectory_steps)

        return result

def get_mas(mas_type: str = "multi"):
    """I would like to make this into a method that can be used to initialize multiple variants of the MAS."""
    
    # Setup and initialization code goes here
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Define members (agents)
    members = ["SQL", "VS"]

    sql_tools = [sql_search_tool()]
    vs_tools = [job_description_search_tool()]
    
    # Define system prompt
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        " following workers: {}. Upon receiving a user request,"
        "When you receive salutations, you should reply with a polite greeting such as Hi, Hello, Good Morning, etc. You do not need to pass it to any worker "
        " you must decide which worker is best suited to act next. Each worker will perform"
        " a specific task and provide their results and status. When the 'SQL' worker is chosen,"
        " it should use the 'companies' table for any database queries. After the SQL worker completes its task,"
        " This approach ensures a comprehensive response that combines both data and visual insights. Once all tasks are completed,"
        " confirm completion by responding with 'FINISH'."
    ).format(", ".join(members))

    sql_columns = ["CORPORATE_IDENTIFICATION_NUMBER", "COMPANY_NAME", "COMPANY_STATUS",
                "COMPANY_CLASS", "COMPANY_CATEGORY", "COMPANY_SUB_CATEGORY",
                "DATE_OF_REGISTRATION", "REGISTERED_STATE", "AUTHORIZED_CAP",
                "PAIDUP_CAPITAL", "INDUSTRIAL_CLASS", "PRINCIPAL_BUSINESS_ACTIVITY_AS_PER_CIN",
                "REGISTERED_OFFICE_ADDRESS", "REGISTRAR_OF_COMPANIES", "EMAIL_ADDR",
                "LATEST_YEAR_ANNUAL_RETURN", "LATEST_YEAR_FINANCIAL_STATEMENT"]

    system_prompt += (
        "\n\nThe SQL database contains columns such as " + ", ".join(sql_columns) + ". "
        "Use this information to guide the allocation of tasks related to corporate data queries."
        "\n\nThe vector search database contains embeddings for detailed job descriptions and professional requirements. "
        "This database is ideal for queries related to job roles, responsibilities, and career qualifications."
        "\n\nIn some instances, a user's query might require insights from both the SQL and vector search databases. "
        "In these cases, route the query to both agents and synthesize a blended response with information from both sources."
        "\n\nIf a user's query does not align with the data in the SQL or vector search databases, respond with: "
        "'The information you have requested is not available to me at this time.'"
        )

    options = ["FINISH", "SQL", "VS"]

    system_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next?"
                "Or should we FINISH? Select one of: {options}"
                "Given a conversation above, if you cannot find a suitable worker among SQL,VS, reply with a neutral, polite answer such as: I do not know, or I may not have an answer to your request "
            ),
        ]
    ).partial(options=str(options), members=", ".join(members))
    
    # Define system prompts for the vector search agent
    vs_system_prompt = (
    """
    As an advanced AI assistant with access to a comprehensive vector database containing detailed job descriptions and professional requirements. When you receive salutations, you should reply with a polite greeting such as Hi, Hello, Good Morning, etc.

    You can match queries to similar job descriptions, highlighting the top three. Useful for:

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
    If the question is not related to one of these topics or if you are unable to find a relevant answer in  the database for a question, kindly decline to answer. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible."
    """
    )
    
    # Define system prompts for the SQL agent
    sql_system_prompt = (
    "You are an SQL agent equipped with a set of specialized tools for database queries handling. "
    "You are not a visualization agent, so you should not attempt to visualize data. "
    "Upon receiving salutations, respond with a polite greeting such as Hi, Hello, Good Morning, etc. "
    "Upon receiving a user request, analyze it to identify the necessary action. "
    "You can offer detailed company data for advanced searches and analysis. Features include:"
    
    "    - Unique Identifiers and Names for precise identification."
    "    - Operational Status to check if companies are active or not."
    "    - Classification Details for size, sector, and type insights."
    "    - Incorporation Information with registration dates and places."
    "    - Capitalization Figures for financial capacity and structure."
    "    - Industry and Main Activities for understanding market segments."
    "    - Contact and Location Data for direct communication."
    "    - Regulatory Oversight to comprehend regulatory contexts."
    "    - Financial Health from annual returns and financial statements."
    
    "Tailor your query, like 'Give average market capitalization. Round to answer.'. State to round numbers to two decimal places and specify if seeking an average or a similar calculation."
    "Your responses should be clear and accurate"
    "In instances where the database does not contain information relevant to a query, politely decline to provide an answer. "
    "If unsure about an answer, state your lack of knowledge without resorting to conjecture. "
    "Strive for brevity and precision in your responses. "
    "Once data extraction is complete, simply return the extracted data and signal completion with a 'finish' statement."
    )

    # Initialize the agent graph
    agent_graph = AgentGraph(llm, sql_tools, vs_tools, system_prompt, sql_system_prompt, vs_system_prompt)
    
    return agent_graph