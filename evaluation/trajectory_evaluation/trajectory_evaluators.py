from typing import Any, Optional, Sequence, Tuple

from langchain.chains import LLMChain
from langchain.evaluation import AgentTrajectoryEvaluator
from langchain.evaluation import load_evaluator
from langchain.schema import AgentAction
from langchain_openai import ChatOpenAI

class HelpfulnessEvaluator(AgentTrajectoryEvaluator):
    """The default trajectory evaluator that returns whether the result is helpful, and thus achieved it's goal."""

    def __init__(self) -> None:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
        self.evaluator = load_evaluator("trajectory", llm=llm)

    def _evaluate_agent_trajectory(
        self,
        *,
        prediction: str,
        input: str,
        agent_trajectory: Sequence[Tuple[AgentAction, str]],
        reference: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:

        evaluation_result = self.evaluator.evaluate_agent_trajectory(
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
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
        template = """Evaluate the necessity of each step taken in responding to {input}.

        DATA
        ------
        Steps: {trajectory}
        ------

        i. Was each step necessary for achieving the final answer?
        [Assess whether each step in the trajectory was essential in reaching the final answer, explaining your reasoning.]

        ii. Could the answer have been achieved with fewer steps?
        [Consider if a more concise approach could have been used without compromising the quality of the answer.]

        iii. Are there any redundant or superfluous steps?
        [Identify any steps that do not contribute meaningfully to the final answer or could be considered redundant.]

        iv. Does each step add value or clarity to the final answer?
        [Evaluate whether each step enhances the overall understanding and accuracy of the final answer.]

        v. Overall, is the sequence of steps efficient and direct?
        [Provide an overall assessment of the efficiency and directness of the steps taken to answer the question.]

        Verdict:
        [Summarize the evaluation on a new line with a 'Y' for yes if every step was necessary, or 'N' for no if any step was unnecessary or redundant.]
        """
        
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
    

class ToolSelectionEvaluator(AgentTrajectoryEvaluator):
    """A custom trajectory evaluator that evaluates whether the selected tool at any step was not beneficial in answering the input."""

    def __init__(self) -> None:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0) 
        template = """Assess each step to determine if a non-beneficial tool was selected in answering {input} step by step.

        DATA
        ------
        Steps: {trajectory}
        ------

        i. Was the right tool selected for each step?
        [Discuss whether the right tool was selected for each step, explaining the reasoning.]

        ii. Does the sequence of tools used make logical sense for the task?
        [Evaluate the logic behind the sequence of tools used.]

        iii. Are the tools used effectively and appropriately?
        [Discuss the effectiveness and appropriateness of the tools used.]

        iv. Are there any steps where an unnecessary or less suitable tool was used?
        [Identify any steps where the tool choice could be questioned, explaining why.]

        v. Overall, are the best possible tools used for answering the question?
        [Give a final assessment on the appropriateness of the tool choices.]

        Verdict:
        [Summarize the evaluation on a new line with a 'Y' for yes if the right tools were selected at each step, or 'N' for no if any step included a non-beneficial or less suitable tool.]
        """
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
    