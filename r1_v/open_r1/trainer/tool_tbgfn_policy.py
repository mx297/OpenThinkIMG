from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from qwen_vl_utils import process_vision_info


@dataclass
class PolicyAction:
    text: str
    token_ids: List[int]


class ToolTBGFNPolicy(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        processor: Any,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
        do_sample: bool = True,
        max_prompt_length: int = 4096,
    ) -> None:
        super().__init__()
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.max_prompt_length = max_prompt_length

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _prepare_inputs(self, conversations: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        messages = copy.deepcopy(conversations)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            return_tensors="pt",
            padding=True,
        )
        if "input_ids" in inputs and inputs["input_ids"].shape[1] > self.max_prompt_length:
            inputs["input_ids"] = inputs["input_ids"][:, -self.max_prompt_length :]
            inputs["attention_mask"] = inputs["attention_mask"][:, -self.max_prompt_length :]
        return {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    @torch.no_grad()
    def sample_action(self, conversations: List[Dict[str, Any]]) -> PolicyAction:
        inputs = self._prepare_inputs(conversations)
        generate_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            return_dict_in_generate=True,
        )
        outputs = self.model.generate(**inputs, **generate_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        token_ids = outputs.sequences[0, prompt_len:].tolist()
        text = self.processor.tokenizer.decode(token_ids, skip_special_tokens=True)
        return PolicyAction(text=text, token_ids=token_ids)

    def score_action(self, conversations: List[Dict[str, Any]], action_token_ids: List[int]) -> torch.Tensor:
        if len(action_token_ids) == 0:
            return torch.tensor(0.0, device=self.device)
        inputs = self._prepare_inputs(conversations)
        prompt_ids = inputs["input_ids"]
        prompt_mask = inputs["attention_mask"]
        action_ids = torch.tensor(action_token_ids, dtype=prompt_ids.dtype, device=self.device).unsqueeze(0)
        full_input_ids = torch.cat([prompt_ids, action_ids], dim=1)
        full_attention_mask = torch.cat([prompt_mask, torch.ones_like(action_ids, device=self.device)], dim=1)

        model_kwargs: Dict[str, Any] = {
            "input_ids": full_input_ids,
            "attention_mask": full_attention_mask,
        }
        if "pixel_values" in inputs:
            model_kwargs["pixel_values"] = inputs["pixel_values"]
        if "image_grid_thw" in inputs:
            model_kwargs["image_grid_thw"] = inputs["image_grid_thw"]

        outputs = self.model(**model_kwargs)
        logits = outputs.logits
        prompt_len = prompt_ids.shape[1]
        action_len = action_ids.shape[1]
        logits = logits[:, prompt_len - 1 : prompt_len + action_len - 1, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        gathered = torch.gather(log_probs, dim=-1, index=action_ids.unsqueeze(-1)).squeeze(-1)
        return gathered.sum()
