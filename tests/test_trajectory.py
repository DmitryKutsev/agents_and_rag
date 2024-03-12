import pytest
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.agents import get_agent
from evaluation.trajectory_evaluation.trajectory_evaluators import (
    HelpfulnessEvaluator,
    StepNecessityEvaluator,
    ToolSelectionEvaluator
)
from loguru import logger
from dotenv import load_dotenv
load_dotenv()

# Global setup for the agent and its result, used in each test
@pytest.fixture(scope="module")
def agent_result():
    query = "What is the length of this query?"
    agent = get_agent(agent_type='react')
    result = agent.run_agent(query)
    return result

def test_helpfulness_evaluation(agent_result):
    # Initialize evaluator
    helpfulnessEval = HelpfulnessEvaluator()

    # Evaluate the agent's response
    helpfulness_result = helpfulnessEval.evaluate_agent_trajectory(
        prediction=agent_result["output"],
        input=agent_result["input"],
        agent_trajectory=agent_result["intermediate_steps"],
    )
    logger.info(f"Answer helpfulness result: {helpfulness_result}")
    assert helpfulness_result['score'] == 1, "Helpfulness Evaluation failed"

def test_step_necessity_evaluation(agent_result):
    # Initialize evaluator
    stepNecessityEval = StepNecessityEvaluator()

    # Evaluate the agent's response
    necessity_result = stepNecessityEval.evaluate_agent_trajectory(
        prediction=agent_result["output"],
        input=agent_result["input"],
        agent_trajectory=agent_result["intermediate_steps"],
    )
    logger.info(f"Step necessity result: {necessity_result}")
    assert necessity_result['score'] == 1, "Step Necessity Evaluation failed"

def test_tool_selection_evaluation(agent_result):
    # Initialize evaluator
    toolSelectionEval = ToolSelectionEvaluator()

    # Evaluate the agent's response
    tool_selection_result = toolSelectionEval.evaluate_agent_trajectory(
        prediction=agent_result["output"],
        input=agent_result["input"],
        agent_trajectory=agent_result["intermediate_steps"],
    )
    logger.info(f"Tool selection result: {tool_selection_result}")
    assert tool_selection_result['score'] == 1, "Tool Selection Evaluation failed"


