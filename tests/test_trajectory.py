import pytest
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.agent_factory import agent_system_factory
from evaluation.trajectory_evaluation.trajectory_evaluators import get_trajectory_evaluator
from loguru import logger
from dotenv import load_dotenv
load_dotenv()

# Define a fixture for the agent
@pytest.fixture(scope="module")
def agent():
    # Create an agent using the factory function
    return agent_system_factory(agent_type='multi')

# Define a fixture for the query
@pytest.fixture(scope="module")
def query():
    # Define the query
    return "What is the average authorized capital of the companies in our database?"

# Define a fixture for the agent result
@pytest.fixture(scope="module")
def agent_result(agent, query):
    # Run the agent with the query and get the result
    result, _ = agent.run_agent(query)

    # Format the result
    formatted_result = agent.format_agent_response(result)

    # Add the input query to the result
    formatted_result["input"] = query

    # Return the formatted result
    return formatted_result

def test_helpfulness_evaluation(agent_result):
    # Initialize evaluator
    helpfulness_eval = get_trajectory_evaluator("helpfulness")

    # Evaluate the agent's response
    helpfulness_result = helpfulness_eval.evaluate_agent_trajectory(
        prediction=agent_result["output"],
        input=agent_result["input"],
        agent_trajectory=agent_result["agent_trajectory"],
    )
    logger.info(f"Answer helpfulness result: {helpfulness_result}")
    assert helpfulness_result['score'] == 1, "Helpfulness Evaluation failed"

def test_step_necessity_evaluation(agent_result):
    # Initialize evaluator
    step_necessity_eval = get_trajectory_evaluator("step_necessity")

    # Evaluate the agent's response
    necessity_result = step_necessity_eval.evaluate_agent_trajectory(
        prediction=agent_result["output"],
        input=agent_result["input"],
        agent_trajectory=agent_result["agent_trajectory"],
    )
    logger.info(f"Step necessity result: {necessity_result}")
    assert necessity_result['score'] == 1, "Step Necessity Evaluation failed"

def test_tool_selection_evaluation(agent_result):
    # Initialize evaluator
    tool_selection_eval = get_trajectory_evaluator("tool_selection")

    # Evaluate the agent's response
    tool_selection_result = tool_selection_eval.evaluate_agent_trajectory(
        prediction=agent_result["output"],
        input=agent_result["input"],
        agent_trajectory=agent_result["agent_trajectory"],
    )
    logger.info(f"Tool selection result: {tool_selection_result}")
    assert tool_selection_result['score'] == 1, "Tool Selection Evaluation failed"


