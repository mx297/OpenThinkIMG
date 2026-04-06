from __future__ import annotations

from typing import Any, Callable, Optional, Union

import torch
from transformers import PreTrainedModel

from . import tool_vllm_grpo_trainer as _orig
from .step_judge_reward import (
    compute_step_judge_total,
    compute_turn_judge_reward,
    get_selected_format_reward_name,
    normalize_reward_name,
    validate_step_judge_reward_names,
)
from .tool_generation import vllm_generate_with_tool_calls as legacy_vllm_generate_with_tool_calls
from .tool_generation_safe import vllm_generate_with_tool_calls as safe_vllm_generate_with_tool_calls
from .turn_judge import TrajectoryTurnJudge, build_turn_records_from_tool_generation_output

RewardFunc = _orig.RewardFunc


class _BaseStepJudgeToolVLLMTrainer(_orig.Qwen2VLGRPOVLLMTrainer):
    tool_generation_fn: Callable[..., list[dict[str, Any]]] = staticmethod(legacy_vllm_generate_with_tool_calls)

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        reward_processing_classes=None,
        callbacks=None,
        optimizers=(None, None),
        peft_config=None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
        controller_addr: str = "http://SH-IDCA1404-10-140-54-5:20001",
        reward_func_names: Optional[list[str]] = None,
        step_judge_model: str = "Qwen/Qwen2.5-VL-72B-Instruct",
        step_judge_temperature: float = 0.0,
        step_judge_do_sample: bool = False,
        step_judge_top_p: float = 1.0,
        step_judge_top_k: int = 0,
        step_judge_repetition_penalty: float = 1.0,
        step_judge_max_new_tokens: int = 256,
        step_judge_terminal_weight: float = 10.0,
        step_judge_major_penalty: float = 5.0,
        step_judge_format_weight: float = 2.0,
        step_judge_log_outputs: bool = True,
    ):
        self.reward_func_names = reward_func_names or []
        validate_step_judge_reward_names(self.reward_func_names)
        self.selected_format_reward_name = get_selected_format_reward_name(self.reward_func_names)
        self.step_judge_terminal_weight = step_judge_terminal_weight
        self.step_judge_major_penalty = step_judge_major_penalty
        self.step_judge_format_weight = step_judge_format_weight
        self.step_judge_log_outputs = step_judge_log_outputs
        self._step_judge_positive_cap = 5.0
        self.turn_judge = None
        self.step_judge_model = step_judge_model
        self.step_judge_temperature = step_judge_temperature
        self.step_judge_do_sample = step_judge_do_sample
        self.step_judge_top_p = step_judge_top_p
        self.step_judge_top_k = step_judge_top_k
        self.step_judge_repetition_penalty = step_judge_repetition_penalty
        self.step_judge_max_new_tokens = step_judge_max_new_tokens
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            attn_implementation=attn_implementation,
            controller_addr=controller_addr,
        )
        if self.accelerator.is_main_process:
            self.turn_judge = TrajectoryTurnJudge(
                model_name=step_judge_model,
                temperature=step_judge_temperature,
                do_sample=step_judge_do_sample,
                top_p=step_judge_top_p,
                top_k=step_judge_top_k,
                repetition_penalty=step_judge_repetition_penalty,
                max_new_tokens=step_judge_max_new_tokens,
            )

    def _get_accuracy_and_format_indices(self) -> tuple[int, int]:
        accuracy_idx = None
        format_idx = None
        for idx, name in enumerate(self.reward_func_names):
            normalized = normalize_reward_name(name)
            if normalized == "accuracy":
                accuracy_idx = idx
            elif normalized == self.selected_format_reward_name:
                format_idx = idx
        if accuracy_idx is None or format_idx is None:
            raise ValueError(
                "Step-judge mode expected accuracy and exactly one format reward in reward_func_names."
            )
        return accuracy_idx, format_idx

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        images = [x["image"] for x in inputs]
        prompts_text = [
            _orig.maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]

        prompt_inputs = self.processing_class(
            text=prompts_text,
            images=images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_ids, prompt_mask = (
            prompt_inputs["input_ids"].to(device),
            prompt_inputs["attention_mask"].to(device),
        )
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        if self.state.global_step != self._last_loaded_step:
            with _orig.unwrap_model_for_generation(
                self.model,
                self.accelerator,
                gather_deepspeed3_params=False,
            ) as unwrapped_model:
                if _orig.is_compiled_module(unwrapped_model):
                    state_dict = unwrapped_model._orig_mod.state_dict()
                else:
                    state_dict = unwrapped_model.state_dict()
            if self.accelerator.is_main_process:
                llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                llm_model.load_weights(state_dict.items())
            self._last_loaded_step = self.state.global_step

        all_prompts_text = _orig.gather_object(prompts_text)
        all_prompts = _orig.gather_object(prompts)
        all_images = _orig.gather_object(images)
        all_examples = _orig.gather_object(inputs)

        if self.accelerator.is_main_process:
            tool_generation_output = self.tool_generation_fn(
                self.llm,
                prompts=all_prompts,
                images=all_images,
                sampling_params=self.sampling_params,
                max_rounds=6,
                model_mode="general",
                controller_addr=self.controller_addr,
            )
            model_output_texts = [item["model_outputs"] for item in tool_generation_output]
            ave_tool_num = (
                sum(len(item.get("tool_outputs", [])) for item in tool_generation_output) / max(len(tool_generation_output), 1)
            )
            self._metrics["ave_tool_num"].append(ave_tool_num)
            model_output_ids = []
            for item in tool_generation_output:
                flat_outputs = []
                for model_output_id in item["model_output_ids"]:
                    flat_outputs.extend(model_output_id)
                model_output_ids.append(flat_outputs)
            completion_ids = [completion_list[: self.max_completion_length] for completion_list in model_output_ids]

            judge_outputs = []
            for example, output_item, image in zip(all_examples, tool_generation_output, all_images):
                turns = build_turn_records_from_tool_generation_output(output_item)
                question = output_item.get("prompt") or example.get("question") or example.get("query") or ""
                ground_truth = example.get("solution") or example.get("label") or ""
                scores = self.turn_judge.judge_trajectory(
                    question=question,
                    ground_truth_answer=ground_truth,
                    turns=turns,
                    image=image,
                )
                judge_outputs.append(scores)
                if self.step_judge_log_outputs:
                    turn_reward = compute_turn_judge_reward(
                        scores,
                        major_penalty=self.step_judge_major_penalty,
                        positive_cap=self._step_judge_positive_cap,
                    )
                    print(f"[step_judge] question={question}")
                    print(f"[step_judge] scores={scores}")
                    print(f"[step_judge] turn_reward={turn_reward}")
        else:
            completion_ids = [None] * len(all_prompts_text)
            model_output_texts = [None] * len(all_prompts_text)
            judge_outputs = [None] * len(all_prompts_text)

        completion_ids = _orig.broadcast_object_list(completion_ids, from_process=0)
        model_output_texts = _orig.broadcast_object_list(model_output_texts, from_process=0)
        judge_outputs = _orig.broadcast_object_list(judge_outputs, from_process=0)
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        completion_ids = completion_ids[process_slice]
        model_output_texts = model_output_texts[process_slice]
        judge_outputs = judge_outputs[process_slice]

        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        completion_ids = _orig.pad(completion_ids, padding_value=self.processing_class.pad_token_id)
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        pixel_values = prompt_inputs["pixel_values"]
        image_grid_thw = prompt_inputs["image_grid_thw"]
        logits_to_keep = completion_ids.size(1)

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model,
                    prompt_completion_ids,
                    attention_mask,
                    pixel_values,
                    image_grid_thw,
                    logits_to_keep,
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        pixel_values,
                        image_grid_thw,
                        logits_to_keep,
                    )

        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if _orig.is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        local_rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(zip(self.reward_funcs, self.reward_processing_classes)):
            if isinstance(reward_func, PreTrainedModel):
                raise ValueError("Step-judge trainer currently supports callable reward functions only.")
            reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
            for key in reward_kwargs:
                for example in inputs:
                    reward_kwargs[key].extend([example[key]] * self.num_generations)
            reward_kwargs["model_output_texts"] = model_output_texts
            output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
            local_rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        rewards_per_func = _orig.gather(local_rewards_per_func)
        local_turn_judge_rewards = torch.tensor(
            [
                compute_turn_judge_reward(
                    scores,
                    major_penalty=self.step_judge_major_penalty,
                    positive_cap=self._step_judge_positive_cap,
                )
                for scores in judge_outputs
            ],
            dtype=torch.float32,
            device=device,
        )
        turn_judge_rewards = _orig.gather(local_turn_judge_rewards)

        accuracy_idx, format_idx = self._get_accuracy_and_format_indices()
        accuracy_values = rewards_per_func[:, accuracy_idx]
        format_values = rewards_per_func[:, format_idx]
        rewards = (
            self.step_judge_terminal_weight * accuracy_values
            + turn_judge_rewards
            + self.step_judge_format_weight * format_values
        )

        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        advantages = advantages[process_slice]

        reward_per_func = rewards_per_func.mean(0)
        for idx, reward_name in enumerate(self.reward_func_names):
            self._metrics[f"rewards/{reward_name}"].append(reward_per_func[idx].item())
        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics["turn_judge_reward"].append(turn_judge_rewards.mean().item())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }


class Qwen2VLGRPOStepJudgeVLLMTrainerLegacy(_BaseStepJudgeToolVLLMTrainer):
    tool_generation_fn = staticmethod(legacy_vllm_generate_with_tool_calls)


class Qwen2VLGRPOStepJudgeVLLMTrainerSafe(_BaseStepJudgeToolVLLMTrainer):
    tool_generation_fn = staticmethod(safe_vllm_generate_with_tool_calls)
