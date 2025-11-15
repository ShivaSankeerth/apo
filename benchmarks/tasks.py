"""Benchmark tasks for evaluating optimizers."""

from typing import List, Dict, Any


class BenchmarkTask:
    """Base class for benchmark tasks."""

    def __init__(self, name: str, initial_prompt: str, test_cases: List[Dict[str, Any]]):
        self.name = name
        self.initial_prompt = initial_prompt
        self.test_cases = test_cases

    def evaluate_output(self, output: str, expected: str) -> float:
        """Evaluate how well output matches expected."""
        # Simple containment check
        output_lower = output.lower().strip()
        expected_lower = expected.lower().strip()

        # Exact match
        if output_lower == expected_lower:
            return 1.0

        # Contains expected
        if expected_lower in output_lower:
            return 0.8

        # Partial match
        if any(word in output_lower for word in expected_lower.split()):
            return 0.5

        return 0.0


# Task 1: Sentiment Classification
SENTIMENT_TASK = BenchmarkTask(
    name="Sentiment Classification",
    initial_prompt="Classify the sentiment: {text}",
    test_cases=[
        {"text": "I absolutely love this product! Best purchase ever!", "expected": "positive"},
        {"text": "This is terrible. Complete waste of money.", "expected": "negative"},
        {"text": "It's okay, nothing special really.", "expected": "neutral"},
        {"text": "Amazing quality! Highly recommend to everyone!", "expected": "positive"},
        {"text": "Very disappointed. Does not work as advertised.", "expected": "negative"},
        {"text": "Decent product. Gets the job done.", "expected": "neutral"},
        {"text": "Absolutely fantastic! Exceeded all expectations!", "expected": "positive"},
        {"text": "Horrible experience. Would not recommend.", "expected": "negative"},
        {"text": "It's fine. Average performance overall.", "expected": "neutral"},
        {"text": "Outstanding! Best in its category!", "expected": "positive"},
    ]
)


# Task 2: Question Answering
QA_TASK = BenchmarkTask(
    name="Question Answering",
    initial_prompt="Answer this question: {question}",
    test_cases=[
        {"question": "What is the capital of France?", "expected": "Paris"},
        {"question": "What is 2 + 2?", "expected": "4"},
        {"question": "Who wrote Romeo and Juliet?", "expected": "Shakespeare"},
        {"question": "What is the largest planet in our solar system?", "expected": "Jupiter"},
        {"question": "What is H2O commonly known as?", "expected": "water"},
        {"question": "How many continents are there?", "expected": "7"},
        {"question": "What is the speed of light?", "expected": "299,792,458 meters per second"},
        {"question": "Who painted the Mona Lisa?", "expected": "Leonardo da Vinci"},
        {"question": "What is the smallest prime number?", "expected": "2"},
        {"question": "What year did World War 2 end?", "expected": "1945"},
    ]
)


# Task 3: Text Classification
CATEGORY_TASK = BenchmarkTask(
    name="Category Classification",
    initial_prompt="Categorize this: {text}",
    test_cases=[
        {"text": "The stock market crashed today", "expected": "business"},
        {"text": "Scientists discover new species", "expected": "science"},
        {"text": "Team wins championship game", "expected": "sports"},
        {"text": "New movie breaks box office records", "expected": "entertainment"},
        {"text": "Government announces new policy", "expected": "politics"},
        {"text": "Breakthrough in cancer research", "expected": "science"},
        {"text": "Company reports record profits", "expected": "business"},
        {"text": "Actor wins prestigious award", "expected": "entertainment"},
        {"text": "Election results are announced", "expected": "politics"},
        {"text": "Baseball season begins next month", "expected": "sports"},
    ]
)


# Task 4: Instruction Following
INSTRUCTION_TASK = BenchmarkTask(
    name="Instruction Following",
    initial_prompt="Follow these instructions: {instruction}",
    test_cases=[
        {"instruction": "List 3 colors", "expected": "red, blue, green"},
        {"instruction": "Count from 1 to 5", "expected": "1, 2, 3, 4, 5"},
        {"instruction": "Say hello in French", "expected": "bonjour"},
        {"instruction": "Name 3 fruits", "expected": "apple, banana, orange"},
        {"instruction": "Give me 2 animals", "expected": "dog, cat"},
        {"instruction": "List 3 countries", "expected": "USA, China, India"},
        {"instruction": "Name 2 planets", "expected": "Earth, Mars"},
        {"instruction": "Give 3 numbers", "expected": "1, 2, 3"},
        {"instruction": "List 2 vegetables", "expected": "carrot, broccoli"},
        {"instruction": "Name 3 cities", "expected": "New York, London, Tokyo"},
    ]
)


ALL_TASKS = [
    SENTIMENT_TASK,
    QA_TASK,
    CATEGORY_TASK,
    INSTRUCTION_TASK,
]
