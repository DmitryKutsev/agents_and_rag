from operator import itemgetter
from collections import deque
from typing import Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from agents.react.tools import  measure_len_tool, job_description_search_tool, sql_llm_search_tool

import os


def get_execution_agent():
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0, verbose=True)

    execution_tool_list = [measure_len_tool(), sql_llm_search_tool(), job_description_search_tool()]

    template = """You are an AI who performs one task based on the following objective: {objective}.
    Take into account these previously completed tasks: {context}.
    Your task: {task}.
    Response:
    """
    prompt = ChatPromptTemplate.from_template(template)
    agent = (
        {
            "objective": itemgetter("objective"),
            "context": itemgetter("context"),
            "task": itemgetter("task")
        }
        | prompt 
        | model 
        | StrOutputParser()
    )
    return AgentExecutor(agent=agent, tools=execution_tool_list, verbose=True)

def get_task_creation_chain():
    """Chain that creates new tasks that do not overlap with incompleted tasts. Given an
    objective, the result of the previous task and the task_description of the previous task.
    """
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0, verbose=True)
    template = (
            "You are an task creation AI that uses the result of an execution agent"
            " to create new tasks with the following objective: {objective},"
            " The last completed task has the result: {result}."
            " This result was based on this task description: {task_description}."
            " These are incomplete tasks: {incomplete_tasks}."
            " Based on the result, create new tasks to be completed"
            " by the AI system that do not overlap with incomplete tasks."
            " Return the tasks as an array."
        )
    
    def parse_tasks(task_list):
        new_tasks = task_list.split("\n")
        return [
            {"task_name": task_name} for task_name in new_tasks if task_name.strip()
        ]

    prompt = ChatPromptTemplate.from_template(template)
    agent = prompt | model | parse_tasks
    return agent

def get_task_prioritization_chain():
    """Given a list of tasks, the ultimate objective and next_task_id clean, reformat and reprioritize
    this list without removing any tasks."""
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0, verbose=True)
    template = (
        "You are an task prioritization AI tasked with cleaning the formatting of and reprioritizing"
        " the following tasks: {task_names}."
        " Consider the ultimate objective of your team: {objective}."
        " Do not remove any tasks. Return the result as a numbered list, like:"
        " #. First task"
        " #. Second task"
        " Start the task list with the number {next_task_id}. For example if the number is 3 the list looks like:"
        " 3. First task"
        " 4. Second task"
    )

    def parse_prioritized_tasks(task_list: List[Dict]) -> List[Dict]:
        tasks = task_list.split("\n")
        prioritized_task_list = []
        for task_string in tasks:
            if not task_string.strip():
                continue
            task_parts = task_string.strip().split(".", 1)
            if len(task_parts) == 2:
                task_id = task_parts[0].strip()
                task_name = task_parts[1].strip()
                prioritized_task_list.append(
                    {"task_id": task_id, "task_name": task_name}
                )
        return prioritized_task_list
    
    prompt = ChatPromptTemplate.from_template(template)
    agent = prompt | model | parse_prioritized_tasks
    return agent

def initial_embeddings(first_task):
    embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-ada-002"
        )
    vectorstore = FAISS.from_texts(
            ["_"], embeddings, metadatas=[{"task": first_task}]
        )
    return vectorstore

class BabyAGI():
    """Controller model for the BabyAGI agent."""
    def __init__(self, objective, first_task):
        self.objective = objective
        self.task_list = deque()
        self.task_creation_chain = get_task_creation_chain()
        self.task_prioritization_chain= get_task_prioritization_chain()
        self.execution_agent = get_execution_agent()
        self.task_id_counter = 1
        self.vectorstore = initial_embeddings(first_task)
        self.task_list.append({"task_id": 1, "task_name": first_task})

    def run(self, max_iterations):
        num_iters = 0
        while True:
            if self.task_list:

                # Step 1: Pull the first task
                task = self.task_list.popleft()

                # Step 2: Execute the task
                result = self.execution_agent.invoke(
                    {"objective": self.objective,
                     "context": " ",
                     "task": task["task_name"]}
                )
                print(result)

                # Step 3: Store the result
                self.vectorstore.add_texts(
                    texts=[result],
                    metadatas=[{"task": task["task_name"]}],
                    ids=task["task_id"]
                )

                # Step 4: Create new tasks and reprioritize task list
                new_tasks = self.task_creation_chain.invoke(
                    {"objective": self.objective,
                     "result": result,
                     "task_description": task["task_name"],
                     "incomplete_tasks": [t["task_name"] for t in self.task_list]}
                )
                for new_task in new_tasks:
                    self.task_id_counter += 1
                    new_task.update({"task_id": self.task_id_counter})
                    self.task_list.append(new_task)
                self.task_list = deque(
                    self.task_prioritization_chain.invoke(
                        {"objective": self.objective,
                         "task_names": list(self.task_list),
                         "next_task_id": self.task_id_counter} #double check this
                    )
                )
            num_iters += 1
            if max_iterations is not None and num_iters == max_iterations:
                break

if __name__ == "__main__":
    babyagent = BabyAGI("Find jobs for an AI Expert.", "Find jobs.")
    result = babyagent.run(3)