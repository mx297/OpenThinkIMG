from __future__ import annotations

import base64
import json
import re
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .strict_tool_schema import validate_tool_message

JUDGE_SYSTEM_PROMPT = """Role: OpenThinkIMG Trajectory Judge for Step-Level Rewarding

You are an expert evaluator for OpenThinkIMG, a multimodal visual reasoning agent that solves image-based questions by interacting with external tools over multiple steps.

Your responsibility is to evaluate a COMPLETE trajectory after it has finished, and assign one score to EACH turn in the trajectory.

What OpenThinkIMG does:
- It receives an image and a question.
- It reasons step by step.
- At each step, it outputs exactly one JSON tool-use action.
- The environment executes the action and returns an observation.
- The agent may continue for multiple turns until it outputs a Terminate action with the final answer.

What a turn means:
A turn consists of:
1. the assistant's current JSON action step
2. the resulting observation returned by the environment

Your goal:
Judge whether each turn was:
- correct and useful
- valid but redundant
- a recoverable mistake
- a major deviation

You are NOT a syntax validator.
The trajectory already contains validator results for strict schema checking.
If a step failed strict validation, do not re-check formatting rules. Instead, use that validator result as evidence and judge the effect of that failure on the trajectory.

You must focus on SEMANTIC quality:
- Was the chosen tool appropriate for the current subproblem?
- Were the arguments semantically appropriate?
- Did the turn help progress toward solving the image question?
- Was the observation interpreted correctly?
- Was the decision to terminate appropriate given the accumulated evidence?

Available tool types may include:
- OCR
- Point
- ZoomInSubfigure
- SegmentRegionAroundPoint
- DrawHorizontalLineByY
- DrawVerticalLineByX
- Terminate

Important evaluation principles:
1. Score 1 if the turn is correct and useful for solving the task.
2. Score 0 if the turn is valid but redundant, weak, mildly suboptimal, or recoverably mistaken.
3. Score -1 if the turn is the primary major deviation that substantially derails the trajectory.
4. At most one turn should receive -1 in a trajectory.
5. Later turns that are merely consequences of an earlier major deviation should usually receive 0, not another -1.
6. Multiple different tool-use paths may be valid. Judge whether the turn is semantically useful and compatible with solving the task.
7. Judge semantic usefulness and correctness, not strict path conformity.

Output requirements:
Return ONLY one JSON object.
The JSON object must contain exactly one score for each turn:
\"score_0\", \"score_1\", ..., \"score_n\"

Each value must be one of:
-1, 0, 1

Example:
{
  \"score_0\": 1,
  \"score_1\": 1,
  \"score_2\": 0,
  \"score_3\": -1,
  \"score_4\": 0
}

Do not output explanations.
Do not output markdown.
Do not output extra keys.
Return JSON only.
"""


def _json_default(value: Any) -> Any:
    if isinstance(value, Image.Image):
        return f"<PIL.Image size={value.size}>"
    return str(value)


def _extract_json_object(text: str) -> Dict[str, int]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Judge output does not contain a JSON object: {text}")
    return json.loads(text[start : end + 1])


def validate_turn_scores(turn_scores: Dict[str, Any], expected_turns: int) -> Dict[str, int]:
    expected_keys = [f"score_{idx}" for idx in range(expected_turns)]
    if set(turn_scores.keys()) != set(expected_keys):
        raise ValueError(
            f"Judge output keys must be exactly {expected_keys}, got {sorted(turn_scores.keys())}."
        )

    validated: Dict[str, int] = {}
    negative_count = 0
    for key in expected_keys:
        value = turn_scores[key]
        if value not in (-1, 0, 1):
            raise ValueError(f"Judge score for {key} must be -1, 0, or 1. Got {value!r}.")
        validated[key] = int(value)
        if value == -1:
            negative_count += 1
    if negative_count > 1:
        raise ValueError("Judge output may contain at most one -1 major-deviation score.")
    return validated


def parse_turn_scores(output_text: str, expected_turns: int) -> Dict[str, int]:
    raw_scores = _extract_json_object(output_text)
    return validate_turn_scores(raw_scores, expected_turns)


def _normalize_image(image: Any) -> Optional[Image.Image]:
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB") if image.mode != "RGB" else image
    if isinstance(image, (bytes, bytearray)):
        return Image.open(BytesIO(image)).convert("RGB")
    if isinstance(image, str):
        if image.startswith("data:image"):
            image = image.split("base64,")[-1]
        if len(image) > 200 and all(ch.isalnum() or ch in "+/=\n" for ch in image[:300]):
            try:
                return Image.open(BytesIO(base64.b64decode(image))).convert("RGB")
            except Exception:
                pass
        return Image.open(image).convert("RGB")
    if isinstance(image, dict):
        if "path" in image:
            return Image.open(image["path"]).convert("RGB")
        if "bytes" in image:
            value = image["bytes"]
            if isinstance(value, str):
                value = base64.b64decode(value)
            return Image.open(BytesIO(value)).convert("RGB")
        if "bytes_base64" in image:
            return Image.open(BytesIO(base64.b64decode(image["bytes_base64"]))).convert("RGB")
        if "base64" in image:
            return Image.open(BytesIO(base64.b64decode(image["base64"]))).convert("RGB")
    raise ValueError(f"Unsupported image payload: {type(image)}")


def _stringify_observation(observation: Any) -> str:
    if observation is None:
        return "No observation available."
    if isinstance(observation, str):
        return observation
    return json.dumps(observation, ensure_ascii=False, indent=2, default=_json_default)


def build_turn_records_from_tool_generation_output(output_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    model_outputs = output_item.get("model_outputs", [])
    validation_results = list(output_item.get("validation_results", []))
    if len(validation_results) < len(model_outputs):
        validation_results.extend(
            validate_tool_message(text).__dict__ for text in model_outputs[len(validation_results) :]
        )
    tool_outputs = list(output_item.get("tool_outputs", []))

    turns: List[Dict[str, Any]] = []
    observation_idx = 0
    for idx, raw_step in enumerate(model_outputs):
        validation_result = validation_results[idx] if idx < len(validation_results) else None
        action_name = validation_result.get("action_name") if validation_result else None
        parsed_step = validation_result.get("parsed_data") if validation_result else None

        if action_name == "Terminate":
            observation = "No observation. The model terminated after this step."
        elif observation_idx < len(tool_outputs):
            observation = tool_outputs[observation_idx]
            observation_idx += 1
        else:
            observation = "No observation available for this step."

        turns.append(
            {
                "turn_index": idx,
                "assistant_step_raw": raw_step,
                "parsed_step": parsed_step,
                "validator_result": validation_result,
                "observation": _stringify_observation(observation),
            }
        )
    return turns


def build_turn_records_from_messages(messages: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    messages = list(messages)
    turns: List[Dict[str, Any]] = []
    for idx, message in enumerate(messages):
        if message.get("from") != "gpt":
            continue
        validation = validate_tool_message(message.get("value", "")).__dict__
        observation = "No observation available for this step."
        if idx + 1 < len(messages) and messages[idx + 1].get("from") == "human":
            observation = messages[idx + 1].get("value", observation)
        turns.append(
            {
                "turn_index": len(turns),
                "assistant_step_raw": message.get("value", ""),
                "parsed_step": validation.get("parsed_data"),
                "validator_result": validation,
                "observation": observation,
            }
        )
    return turns


def build_turn_judge_prompt(
    question: str,
    ground_truth_answer: str,
    turns: List[Dict[str, Any]],
) -> str:
    lines = [
        JUDGE_SYSTEM_PROMPT,
        "",
        "Question:",
        question,
        "",
        "Ground-truth final answer:",
        ground_truth_answer,
        "",
        "Full trajectory:",
    ]
    for turn in turns:
        lines.extend(
            [
                f"[Turn {turn['turn_index']} ]",
                "Assistant step raw:",
                turn["assistant_step_raw"],
                "",
                "Parsed step:",
                json.dumps(turn.get("parsed_step"), ensure_ascii=False, indent=2, default=_json_default),
                "",
                "Validator result:",
                json.dumps(turn.get("validator_result"), ensure_ascii=False, indent=2, default=_json_default),
                "",
                "Observation:",
                turn.get("observation", "No observation available."),
                "",
            ]
        )
    lines.append("Please assign one score to every turn. Return only a JSON object with keys score_0 through score_N.")
    return "\n".join(lines)


class TrajectoryTurnJudge:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-72B-Instruct",
        temperature: float = 0.0,
        do_sample: bool = False,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        max_new_tokens: int = 256,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.do_sample = do_sample
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens
        self.processor = None
        self.model = None

    def load(self):
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
        if self.model is None:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.model.eval()

    def _build_generation_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "repetition_penalty": self.repetition_penalty,
        }
        if self.do_sample:
            kwargs.update(
                {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                }
            )
        return kwargs

    def judge_trajectory(
        self,
        question: str,
        ground_truth_answer: str,
        turns: List[Dict[str, Any]],
        image: Any = None,
    ) -> Dict[str, int]:
        self.load()
        prompt = build_turn_judge_prompt(
            question=question,
            ground_truth_answer=ground_truth_answer,
            turns=turns,
        )
        image = _normalize_image(image)
        processor_kwargs: Dict[str, Any] = {
            "text": [prompt],
            "return_tensors": "pt",
        }
        if image is not None:
            processor_kwargs["images"] = [image]
        model_inputs = self.processor(**processor_kwargs)
        model_device = getattr(self.model, "device", None) or next(self.model.parameters()).device
        model_inputs = {k: v.to(model_device) for k, v in model_inputs.items()}
        with torch.inference_mode():
            output_ids = self.model.generate(
                **model_inputs,
                **self._build_generation_kwargs(),
            )
        input_length = model_inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_length:]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return parse_turn_scores(output_text, expected_turns=len(turns))
