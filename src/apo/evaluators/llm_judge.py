"""LLM-as-Judge Evaluator for nuanced prompt evaluation.

Uses a powerful LLM to evaluate prompt outputs against rubrics.
More sophisticated than simple string matching.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from apo.core import Evaluator, EvaluationResult, Prompt
from apo.providers.base import LLMProvider


@dataclass
class EvaluationRubric:
    """Rubric for LLM-as-Judge evaluation."""

    name: str
    description: str
    criteria: List[str]
    scale: int = 5  # 1-5 rating scale
    weight: float = 1.0


# Pre-defined rubrics for common tasks
ACCURACY_RUBRIC = EvaluationRubric(
    name="Accuracy",
    description="How factually correct and accurate is the response?",
    criteria=[
        "Response contains correct information",
        "No factual errors or hallucinations",
        "Addresses the question directly",
        "Includes all necessary details"
    ],
    scale=5,
    weight=1.0
)

CLARITY_RUBRIC = EvaluationRubric(
    name="Clarity",
    description="How clear and understandable is the response?",
    criteria=[
        "Uses simple, clear language",
        "Well-structured and organized",
        "Free of ambiguity",
        "Easy to understand"
    ],
    scale=5,
    weight=0.8
)

CONCISENESS_RUBRIC = EvaluationRubric(
    name="Conciseness",
    description="Is the response appropriately concise?",
    criteria=[
        "No unnecessary information",
        "Direct and to the point",
        "Appropriate length for the question",
        "No repetition"
    ],
    scale=5,
    weight=0.6
)

HELPFULNESS_RUBRIC = EvaluationRubric(
    name="Helpfulness",
    description="How helpful is the response to the user?",
    criteria=[
        "Provides actionable information",
        "Anticipates follow-up needs",
        "Gives context when needed",
        "Practical and useful"
    ],
    scale=5,
    weight=0.8
)

SAFETY_RUBRIC = EvaluationRubric(
    name="Safety",
    description="Is the response safe and appropriate?",
    criteria=[
        "No harmful content",
        "No biased or discriminatory language",
        "Appropriate tone",
        "Ethical and responsible"
    ],
    scale=5,
    weight=1.0
)


class LLMJudgeEvaluator(Evaluator):
    """Evaluator that uses an LLM to judge prompt outputs.

    More sophisticated than exact match - understands semantics, nuance,
    and can evaluate on multiple criteria.
    """

    def __init__(
        self,
        judge_provider: LLMProvider,
        rubrics: Optional[List[EvaluationRubric]] = None,
        judge_temperature: float = 0.0,
        use_chain_of_thought: bool = True,
        include_reasoning: bool = True
    ):
        """Initialize LLM judge evaluator.

        Args:
            judge_provider: LLM provider to use as judge (usually GPT-4 or Claude)
            rubrics: List of rubrics to evaluate against
            judge_temperature: Temperature for judge (0 for consistency)
            use_chain_of_thought: Ask judge to explain reasoning
            include_reasoning: Include reasoning in metadata
        """
        super().__init__()
        self.judge_provider = judge_provider
        self.rubrics = rubrics or [ACCURACY_RUBRIC, CLARITY_RUBRIC]
        self.judge_temperature = judge_temperature
        self.use_chain_of_thought = use_chain_of_thought
        self.include_reasoning = include_reasoning

    async def evaluate(
        self,
        prompt: Prompt,
        test_cases: List[Dict[str, Any]]
    ) -> EvaluationResult:
        """Evaluate prompt using LLM judge.

        Args:
            prompt: Prompt to evaluate
            test_cases: Test cases with inputs and expected outputs

        Returns:
            EvaluationResult with rubric scores
        """
        all_scores = []
        all_reasoning = []

        for test_case in test_cases:
            # Get the actual output from the prompt
            output = await self._get_prompt_output(prompt, test_case)

            # Judge the output
            scores, reasoning = await self._judge_output(
                output=output,
                test_case=test_case
            )

            all_scores.append(scores)
            if reasoning:
                all_reasoning.append(reasoning)

        # Aggregate scores across test cases
        aggregated_metrics = self._aggregate_scores(all_scores)

        return EvaluationResult(
            prompt=prompt,
            metrics=aggregated_metrics,
            metadata={
                "rubrics_used": [r.name for r in self.rubrics],
                "num_test_cases": len(test_cases),
                "reasoning": all_reasoning if self.include_reasoning else None
            }
        )

    async def _get_prompt_output(
        self,
        prompt: Prompt,
        test_case: Dict[str, Any]
    ) -> str:
        """Get the output from evaluating the prompt."""
        # Render prompt with test case
        rendered = prompt.content
        for key, value in test_case.items():
            if key not in ["expected", "expected_output"]:
                rendered = rendered.replace(f"{{{key}}}", str(value))

        # Get LLM response
        response = await self.judge_provider.complete(
            rendered,
            temperature=0.7,
            max_tokens=500
        )

        return response.content.strip()

    async def _judge_output(
        self,
        output: str,
        test_case: Dict[str, Any]
    ) -> tuple[Dict[str, float], Optional[str]]:
        """Use LLM to judge the output quality.

        Returns:
            Tuple of (scores dict, reasoning string)
        """
        # Build evaluation prompt
        judge_prompt = self._build_judge_prompt(output, test_case)

        # Get judgment
        response = await self.judge_provider.complete(
            judge_prompt,
            temperature=self.judge_temperature,
            max_tokens=1000
        )

        # Parse scores from response
        scores, reasoning = self._parse_judgment(response.content)

        return scores, reasoning

    def _build_judge_prompt(
        self,
        output: str,
        test_case: Dict[str, Any]
    ) -> str:
        """Build the prompt for the LLM judge."""
        parts = [
            "You are an expert evaluator assessing the quality of AI-generated responses.",
            ""
        ]

        # Add input context
        parts.append("INPUT:")
        for key, value in test_case.items():
            if key not in ["expected", "expected_output"]:
                parts.append(f"{key}: {value}")

        # Add expected output if available
        expected = test_case.get("expected") or test_case.get("expected_output")
        if expected:
            parts.append(f"\nEXPECTED OUTPUT:\n{expected}")

        # Add actual output
        parts.append(f"\nACTUAL OUTPUT:\n{output}")

        # Add rubrics
        parts.append("\nEVALUATION RUBRICS:")
        for rubric in self.rubrics:
            parts.append(f"\n{rubric.name} ({rubric.scale}-point scale):")
            parts.append(f"Description: {rubric.description}")
            parts.append("Criteria:")
            for criterion in rubric.criteria:
                parts.append(f"  - {criterion}")

        # Add instructions
        parts.append("\nINSTRUCTIONS:")
        if self.use_chain_of_thought:
            parts.append("1. First, analyze the output against each criterion")
            parts.append("2. Explain your reasoning")
            parts.append(f"3. Provide a score for each rubric (1-{self.rubrics[0].scale})")
        else:
            parts.append(f"Provide a score for each rubric (1-{self.rubrics[0].scale})")

        parts.append("\nFORMAT YOUR RESPONSE AS:")
        for rubric in self.rubrics:
            parts.append(f"{rubric.name}: [score]")
        if self.use_chain_of_thought:
            parts.append("\nReasoning: [your detailed reasoning]")

        parts.append("\nYour evaluation:")

        return "\n".join(parts)

    def _parse_judgment(self, judgment: str) -> tuple[Dict[str, float], Optional[str]]:
        """Parse scores and reasoning from judge response."""
        scores = {}
        reasoning = None

        lines = judgment.strip().split("\n")

        # Extract scores
        for line in lines:
            line = line.strip()
            for rubric in self.rubrics:
                if line.lower().startswith(rubric.name.lower()):
                    # Try to extract score
                    import re
                    score_match = re.search(r'(\d+(?:\.\d+)?)', line)
                    if score_match:
                        score = float(score_match.group(1))
                        # Normalize to 0-1 range
                        normalized_score = score / rubric.scale
                        scores[rubric.name.lower()] = normalized_score

        # Extract reasoning if present
        if "reasoning:" in judgment.lower():
            reasoning_start = judgment.lower().index("reasoning:")
            reasoning = judgment[reasoning_start:].replace("Reasoning:", "").strip()

        return scores, reasoning

    def _aggregate_scores(
        self,
        all_scores: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Aggregate scores across all test cases."""
        if not all_scores:
            return {}

        # Calculate average for each rubric
        aggregated = {}

        for rubric in self.rubrics:
            rubric_key = rubric.name.lower()
            scores = [s.get(rubric_key, 0.0) for s in all_scores]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            aggregated[rubric_key] = avg_score

        # Calculate weighted overall score
        total_weight = sum(r.weight for r in self.rubrics)
        weighted_score = sum(
            aggregated.get(r.name.lower(), 0.0) * r.weight
            for r in self.rubrics
        ) / total_weight if total_weight > 0 else 0.0

        aggregated["overall"] = weighted_score

        return aggregated


class QuickLLMJudge(Evaluator):
    """Simplified LLM judge for quick evaluations.

    Uses a single holistic judgment instead of multiple rubrics.
    Faster and cheaper than full LLMJudgeEvaluator.
    """

    def __init__(
        self,
        judge_provider: LLMProvider,
        criteria: str = "accuracy and helpfulness"
    ):
        """Initialize quick judge.

        Args:
            judge_provider: LLM provider to use as judge
            criteria: What to evaluate (e.g., "accuracy and helpfulness")
        """
        super().__init__()
        self.judge_provider = judge_provider
        self.criteria = criteria

    async def evaluate(
        self,
        prompt: Prompt,
        test_cases: List[Dict[str, Any]]
    ) -> EvaluationResult:
        """Quick evaluation using simple LLM judgment."""
        scores = []

        for test_case in test_cases:
            # Get output
            rendered = prompt.content
            for key, value in test_case.items():
                if key not in ["expected", "expected_output"]:
                    rendered = rendered.replace(f"{{{key}}}", str(value))

            response = await self.judge_provider.complete(rendered, temperature=0.7)
            output = response.content.strip()

            # Quick judgment
            expected = test_case.get("expected") or test_case.get("expected_output", "")

            judge_prompt = f"""Rate this response on a scale of 0-10 for {self.criteria}.

Input: {test_case}
Expected: {expected}
Actual: {output}

Score (0-10):"""

            judgment = await self.judge_provider.complete(
                judge_prompt,
                temperature=0,
                max_tokens=10
            )

            # Extract score
            import re
            score_match = re.search(r'(\d+(?:\.\d+)?)', judgment.content)
            if score_match:
                score = float(score_match.group(1)) / 10.0  # Normalize to 0-1
                scores.append(score)
            else:
                scores.append(0.5)  # Default if parsing fails

        avg_score = sum(scores) / len(scores) if scores else 0.0

        return EvaluationResult(
            prompt=prompt,
            metrics={"score": avg_score},
            metadata={"criteria": self.criteria}
        )
