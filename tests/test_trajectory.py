import pytest
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.react.react_agent import get_react_agent
from evaluation.trajectory_evaluation.trajectory_evaluators import (
    HelpfulnessEvaluator,
    StepNecessityEvaluator,
    ToolSelectionEvaluator
)

from dotenv import load_dotenv
load_dotenv()

# Global setup for the agent and its result, used in each test
@pytest.fixture(scope="module")
def agent_result():
    query = "What is the length of this query?"
    agent = get_react_agent()
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

    assert tool_selection_result['score'] == 1, "Tool Selection Evaluation failed"


