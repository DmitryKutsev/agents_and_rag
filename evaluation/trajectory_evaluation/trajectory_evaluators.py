from typing import Any, Optional, Sequence, Tuple

from langchain.chains import LLMChain
from langchain.evaluation import AgentTrajectoryEvaluator
from langchain.schema import AgentAction
from langchain_openai import ChatOpenAI

class BaseTrajectoryEvaluator(AgentTrajectoryEvaluator):
    def __init__(self) -> None:
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

    def _evaluate_agent_trajectory(
        self,
        *,
        prediction: str,
        input: str,
        agent_trajectory: str,
        **kwargs: Any,
    ) -> dict:

        response = self.chain.invoke(dict(trajectory=agent_trajectory, input=input, prediction=prediction), **kwargs)

        decision = response["text"].split("\n")[-1].strip()
        score = 1 if "Y" in decision else 0
        return {"score": score, "value": decision, "reasoning": response}

class HelpfulnessEvaluator(BaseTrajectoryEvaluator):
    """A custom trajectory evaluator that evaluates the helpfulness of a predicted string."""

    def __init__(self) -> None:
        super().__init__()
        template = (
        """
        Assess the helpfulness of the final answer to {input}.
        The final answer is {prediction}.

        DATA
        ------
        Steps: {trajectory}
        ------

        i. Does the final answer directly address the question asked?
        [Determine if the final answer specifically responds to the initial query, providing a clear and direct response.]

        ii. Is the logic of the answer sound, based on the steps taken?
        [Evaluate the logical progression of the steps taken to arrive at the final answer, ensuring each step contributes to the logic and understanding.]

        iii. Were any crucial steps overlooked that might affect the answer's completeness or accuracy?
        [Identify if any necessary steps were missed that could lead to a more comprehensive or accurate answer.]

        iv. Does the final answer provide clarity and insight into the question?
        [Assess whether the answer enhances understanding of the topic and offers valuable insights.]

        v. Could the answer have been improved by altering or adding steps in the trajectory?
        [Consider if modifications to the steps taken could have led to a more effective or informative answer.]

        Verdict:
        [Summarize the evaluation with a 'Y' for yes if the final answer is helpful, addressing the question logically and thoroughly, or 'N' for no if it fails to do so. Make sure this is the last line of the response.]
        """
        )
        
        self.chain = LLMChain.from_string(self.llm, template)

class StepNecessityEvaluator(BaseTrajectoryEvaluator):
    """A custom trajectory evaluator that evaluates the perplexity of a predicted string."""

    def __init__(self) -> None:
        super().__init__()
        template = (
        """Evaluate the necessity of each step taken in responding to {input}.
        The final answer is {prediction}.

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
        [Summarize the evaluation on a new line with a 'Y' for yes if every step was necessary, or 'N' for no if any step was unnecessary or redundant. Make sure this is the last line of the response.]
        """
        )
        
        self.chain = LLMChain.from_string(self.llm, template)
    

class ToolSelectionEvaluator(BaseTrajectoryEvaluator):
    """A custom trajectory evaluator that evaluates whether the selected tool at any step was not beneficial in answering the input."""

    def __init__(self) -> None:
        super().__init__()
        template = (
        """Assess each step to determine if a non-beneficial tool was selected in answering {input} step by step.
        The final answer is {prediction}.

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
        [Summarize the evaluation on a new line with a 'Y' for yes if the right tools were selected at each step, or 'N' for no if any step included a non-beneficial or less suitable tool. Make sure this is the last line of the response.]
        """
        )
        self.chain = LLMChain.from_string(self.llm, template)

def get_trajectory_evaluator(eval_type: str = "helpfulness"):
    if eval_type == "helpfulness":
        evaluator = HelpfulnessEvaluator()
    elif eval_type == "step_necessity":
        evaluator = StepNecessityEvaluator()
    elif eval_type == "tool_selection":
        evaluator = ToolSelectionEvaluator()
    else:
        raise ValueError(f"Invalid trajectory eval type: {eval_type}. Expected 'react' or 'openai'.")

    return evaluator