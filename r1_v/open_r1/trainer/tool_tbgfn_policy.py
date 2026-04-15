from __future__ import annotations

import copy
from typing import Any, List

import torch
from qwen_vl_utils import process_vision_info

from .tool_generation import append_conversation_fn, handle_tool_result
from .strict_tool_schema import validate_tool_message


class AssistantTurnLogProbScorer:
    def __init__(self, model: Any, processor: Any, max_prompt_length: int = 4096):
        self.model = model
        self.processor = processor
        self.max_prompt_length = max_prompt_length

    def _processor_inputs(self, conversations: list[dict]) -> dict:
        text = self.processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)
        image_inputs, _ = process_vision_info(conversations)
        inputs = self.processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
        return {k: v.to(self.model.device) for k, v in inputs.items()}

    def _assistant_token_logprob(self, prefix_conversations: list[dict], assistant_text: str) -> torch.Tensor:
        prefix = copy.deepcopy(prefix_conversations)
        full = append_conversation_fn(copy.deepcopy(prefix_conversations), text=assistant_text, role="assistant")

        prefix_inputs = self._processor_inputs(prefix)
        full_inputs = self._processor_inputs(full)
        prompt_len = prefix_inputs["input_ids"].shape[1]
        input_ids = full_inputs["input_ids"]
        attention_mask = full_inputs["attention_mask"]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=full_inputs.get("pixel_values", None),
                image_grid_thw=full_inputs.get("image_grid_thw", None),
            )
        logits = outputs.logits[:, :-1, :]
        target_ids = input_ids[:, 1:]
        assistant_target_ids = target_ids[:, prompt_len - 1 :]
        assistant_logits = logits[:, prompt_len - 1 :, :]
        log_probs = torch.log_softmax(assistant_logits, dim=-1)
        token_log_probs = torch.gather(log_probs, dim=-1, index=assistant_target_ids.unsqueeze(-1)).squeeze(-1)
        return token_log_probs.sum(dim=-1).squeeze(0)

    def score_trace(self, trace: dict) -> torch.Tensor:
        conversations = []
        if trace.get("conversations"):
            # rebuild from the initial user message only to avoid leaking future observations
            conversations = [copy.deepcopy(trace["conversations"][0])]
        total = torch.zeros((), device=self.model.device)
        model_outputs = trace.get("model_outputs", [])
        tool_outputs = trace.get("tool_outputs", [])
        prompt = trace.get("prompt", "")
        for turn_index, assistant_text in enumerate(model_outputs):
            total = total + self._assistant_token_logprob(conversations, assistant_text)
            conversations = append_conversation_fn(conversations, text=assistant_text, role="assistant")
            validation = validate_tool_message(assistant_text)
            if validation.action_name == "Terminate":
                break
            tool_result = tool_outputs[turn_index] if turn_index < len(tool_outputs) else {"text": ""}
            cfg = validation.as_api_config() if validation.is_valid else {"API_name": validation.action_name or "ToolValidator"}
            conversations = handle_tool_result(
                cfg=cfg,
                tool_result=tool_result,
                conversations=conversations,
                model_mode="general",
                original_prompt=prompt,
                input_data_item={"images": trace.get("images", [])},
            )
        return total
