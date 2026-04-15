import re
from typing import List

from math_verify import parse, verify

from .strict_tool_schema import extract_terminate_answer, score_tool_message


def _extract_ground_truth(solution_text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", solution_text)
    return match.group(1).strip() if match else solution_text.strip()


def _answers_match(student_answer: str, ground_truth: str) -> bool:
    if student_answer == ground_truth:
        return True
    try:
        return float(verify(parse(student_answer), parse(ground_truth))) > 0
    except Exception:
        return False


def accuracy_reward_from_model_outputs(model_output_texts: List[str], solution: str) -> float:
    if not model_output_texts:
        return 0.0
    content = model_output_texts[-1]
    ground_truth = _extract_ground_truth(solution)
    student_answer = extract_terminate_answer(content) or content.strip()
    return 1.0 if _answers_match(student_answer, ground_truth) else 0.0


def format_reward_from_model_outputs(model_output_texts: List[str]) -> float:
    if not model_output_texts:
        return 0.0
    step_scores = [score_tool_message(output_text) for output_text in model_output_texts]
    return sum(step_scores) / len(step_scores) if step_scores else 0.0
