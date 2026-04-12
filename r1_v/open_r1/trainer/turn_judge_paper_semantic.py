from __future__ import annotations

import base64
import json
from io import BytesIO
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .strict_tool_schema import validate_tool_message
from .turn_judge_prompt_paper_semantic import PAPER_SEMANTIC_TURN_JUDGE_SYSTEM_PROMPT


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


def validate_binary_turn_scores(turn_scores: Dict[str, Any], expected_turns: int) -> Dict[str, int]:
    expected_keys = [f"score_{idx}" for idx in range(expected_turns)]
    if set(turn_scores.keys()) != set(expected_keys):
        raise ValueError(
            f"Judge output keys must be exactly {expected_keys}, got {sorted(turn_scores.keys())}."
        )

    validated: Dict[str, int] = {}
    for key in expected_keys:
        value = turn_scores[key]
        if value not in (0, 1):
            raise ValueError(f"Judge score for {key} must be 0 or 1. Got {value!r}.")
        validated[key] = int(value)
    return validated


def parse_binary_turn_scores(output_text: str, expected_turns: int) -> Dict[str, int]:
    raw_scores = _extract_json_object(output_text)
    return validate_binary_turn_scores(raw_scores, expected_turns)


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


def build_non_final_turn_records_from_tool_generation_output(output_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    model_outputs = output_item.get("model_outputs", [])
    validation_results = list(output_item.get("validation_results", []))
    if len(validation_results) < len(model_outputs):
        validation_results.extend(
            validate_tool_message(text).__dict__ for text in model_outputs[len(validation_results) :]
        )
    tool_outputs = list(output_item.get("tool_outputs", []))

    turns: List[Dict[str, Any]] = []
    observation_idx = 0
    final_idx = max(len(model_outputs) - 1, 0)
    for idx, validation_result in enumerate(validation_results):
        if validation_result.get("action_name") == "Terminate":
            final_idx = idx
            break

    for idx, raw_step in enumerate(model_outputs):
        if idx >= final_idx:
            break
        if observation_idx < len(tool_outputs):
            observation = tool_outputs[observation_idx]
            observation_idx += 1
        else:
            observation = "No observation available for this step."
        turns.append(
            {
                "turn_index": len(turns),
                "assistant_turn": raw_step,
                "tool_observation": _stringify_observation(observation),
            }
        )
    return turns


def build_paper_turn_judge_payload(
    question: str,
    ground_truth_answer: str,
    turns: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "question": question,
        "ground_truth_answer": ground_truth_answer,
        "turns": turns,
    }


def build_paper_turn_judge_prompt(
    question: str,
    ground_truth_answer: str,
    turns: List[Dict[str, Any]],
) -> str:
    payload = build_paper_turn_judge_payload(
        question=question,
        ground_truth_answer=ground_truth_answer,
        turns=turns,
    )
    return (
        PAPER_SEMANTIC_TURN_JUDGE_SYSTEM_PROMPT
        + "\n\n"
        + "INPUT JSON:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
    )


class PaperSemanticTurnJudge:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-72B-Instruct",
        temperature: float = 0.0,
        do_sample: bool = False,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        max_new_tokens: int = 128,
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

    def judge_non_final_turns(
        self,
        question: str,
        ground_truth_answer: str,
        turns: List[Dict[str, Any]],
        image: Any = None,
    ) -> Dict[str, int]:
        if len(turns) == 0:
            return {}

        self.load()
        prompt = build_paper_turn_judge_prompt(
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
        return parse_binary_turn_scores(output_text, expected_turns=len(turns))
