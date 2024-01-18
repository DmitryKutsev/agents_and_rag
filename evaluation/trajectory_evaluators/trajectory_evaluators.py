from typing import Any, Optional, Sequence, Tuple

from langchain.chains import LLMChain
from langchain.evaluation import AgentTrajectoryEvaluator
from langchain.evaluation import load_evaluator
from langchain.schema import AgentAction
from langchain_openai import ChatOpenAI


class HelpfulnessEvaluator(AgentTrajectoryEvaluator):
    """The default trajectory evaluator that returns whether the result is helpful, and thus achieved it's goal."""

    def __init__(self) -> None:
        self.evaluator = load_evaluator("trajectory")

    def _evaluate_agent_trajectory(
        self,
        *,
        prediction: str,
        input: str,
        agent_trajectory: Sequence[Tuple[AgentAction, str]],
        reference: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:

        evaluation_result = self.evaluator.evaluate(
            prediction=prediction, 
            input=input, 
            agent_trajectory=agent_trajectory, 
            reference=reference, 
            **kwargs
        )

        return evaluation_result


class StepNecessityEvaluator(AgentTrajectoryEvaluator):
    """A custom trajectory evaluator that evaluates the perplexity of a predicted string."""

    def __init__(self) -> None:
        llm = ChatOpenAI(model="gpt-3.5-turbo-instruct", temperature=0.0)
        template = """Are any of the following steps unnecessary in answering {input}? Provide the verdict on a new line as a single "Y" for yes or "N" for no.

        DATA
        ------
        Steps: {trajectory}
        ------

        Verdict:"""
        self.chain = LLMChain.from_string(llm, template)

    def _evaluate_agent_trajectory(
        self,
        *,
        prediction: str,
        input: str,
        agent_trajectory: Sequence[Tuple[AgentAction, str]],
        reference: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        vals = [
            f"{i}: Action=[{action.tool}] returned observation = [{observation}]"
            for i, (action, observation) in enumerate(agent_trajectory)
        ]
        trajectory = "\n".join(vals)
        response = self.chain.run(dict(trajectory=trajectory, input=input), **kwargs)
        decision = response.split("\n")[-1].strip()
        score = 1 if decision == "Y" else 0
        return {"score": score, "value": decision, "reasoning": response}
    

class RightToolSelectionEvaluator(AgentTrajectoryEvaluator):
    """A custom trajectory evaluator that evaluates whether the selected tool at any step was not beneficial in answering the input."""

    def __init__(self) -> None:
        llm = ChatOpenAI(model="gpt-3.5-turbo-instruct", temperature=0.0) 
        template = """Assess each step to determine if a non-beneficial tool was selected in answering {input}. Provide the verdict on a new line as a single "Y" for yes or "N" for no.

        DATA
        ------
        Steps: {trajectory}
        ------

        Verdict:"""
        self.chain = LLMChain.from_string(llm, template)

    def _evaluate_agent_trajectory(
        self,
        *,
        prediction: str,
        input: str,
        agent_trajectory: Sequence[Tuple[AgentAction, str]],
        reference: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        vals = [
            f"{i}: Action=[{action.tool}] returned observation = [{observation}]"
            for i, (action, observation) in enumerate(agent_trajectory)
        ]
        trajectory = "\n".join(vals)
        response = self.chain.run(dict(trajectory=trajectory, input=input), **kwargs)
        decision = response.split("\n")[-1].strip()
        score = 1 if decision == "Y" else 0
        return {"score": score, "value": decision, "reasoning": response}