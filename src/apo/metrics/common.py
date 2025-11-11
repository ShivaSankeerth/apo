"""Common metrics for prompt evaluation."""

from typing import Any, Dict, List


def accuracy_score(predictions: List[str], targets: List[str]) -> float:
    """Calculate exact match accuracy.

    Args:
        predictions: List of predicted outputs
        targets: List of target outputs

    Returns:
        Accuracy score between 0 and 1
    """
    if not predictions or not targets or len(predictions) != len(targets):
        return 0.0

    correct = sum(1 for pred, target in zip(predictions, targets) if pred.strip() == target.strip())
    return correct / len(predictions)


def semantic_similarity(text1: str, text2: str) -> float:
    """Calculate semantic similarity between two texts.

    This is a simplified implementation. For production use, consider
    using embedding-based similarity (e.g., cosine similarity of sentence embeddings).

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score between 0 and 1
    """
    # Simple word overlap-based similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


def token_efficiency(prompt: str, max_tokens: int = 1000) -> float:
    """Calculate token efficiency (inverse of prompt length).

    Args:
        prompt: The prompt text
        max_tokens: Maximum expected tokens

    Returns:
        Efficiency score between 0 and 1 (higher is more efficient)
    """
    # Rough approximation: 1 token ≈ 4 characters
    estimated_tokens = len(prompt) / 4

    if estimated_tokens >= max_tokens:
        return 0.0

    return 1.0 - (estimated_tokens / max_tokens)


def response_quality(
    response: str,
    criteria: Dict[str, Any]
) -> float:
    """Evaluate response quality based on multiple criteria.

    Args:
        response: The LLM response
        criteria: Dictionary of quality criteria

    Returns:
        Quality score between 0 and 1
    """
    score = 0.0
    count = 0

    # Check length requirements
    if "min_length" in criteria:
        if len(response) >= criteria["min_length"]:
            score += 1.0
        count += 1

    if "max_length" in criteria:
        if len(response) <= criteria["max_length"]:
            score += 1.0
        count += 1

    # Check for required keywords
    if "required_keywords" in criteria:
        keywords = criteria["required_keywords"]
        response_lower = response.lower()
        keyword_score = sum(1 for kw in keywords if kw.lower() in response_lower) / len(keywords)
        score += keyword_score
        count += 1

    # Check for forbidden keywords
    if "forbidden_keywords" in criteria:
        keywords = criteria["forbidden_keywords"]
        response_lower = response.lower()
        violations = sum(1 for kw in keywords if kw.lower() in response_lower)
        score += 1.0 - (violations / max(len(keywords), 1))
        count += 1

    return score / count if count > 0 else 0.0
