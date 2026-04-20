from __future__ import annotations

from typing import Any, Optional

import torch
from transformers import PreTrainedModel

from . import tool_vllm_grpo_trainer as _orig
from .step_judge_reward import compute_turn_judge_reward
from .tool_generation import vllm_generate_with_tool_calls as legacy_vllm_generate_with_tool_calls
from .tool_generation_safe import vllm_generate_with_tool_calls as safe_vllm_generate_with_tool_calls
from .tool_vllm_grpo_trainer_step_judge import _BaseStepJudgeToolVLLMTrainer as _BaseOriginalStepJudgeToolVLLMTrainer
from .turn_judge import build_turn_records_from_tool_generation_output


class _BaseFilteredStepJudgeToolVLLMTrainer(_BaseOriginalStepJudgeToolVLLMTrainer):
    def __init__(self, *args, step_crash_filter: bool = False, **kwargs):
        self.step_crash_filter = step_crash_filter
        self._step_judge_crash_counter = 0
        self._step_judge_rollout_counter = 0
        self._step_judge_filtered_counter = 0
        super().__init__(*args, **kwargs)

    def _prepare_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
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
            all_model_output_texts = [item["model_outputs"] for item in tool_generation_output]
            all_completion_ids = []
            for item in tool_generation_output:
                flat_outputs = []
                for model_output_id in item["model_output_ids"]:
                    flat_outputs.extend(model_output_id)
                all_completion_ids.append(flat_outputs[: self.max_completion_length])
            ave_tool_num = (
                sum(len(item.get("tool_outputs", [])) for item in tool_generation_output) / max(len(tool_generation_output), 1)
            )
            self._metrics["ave_tool_num"].append(ave_tool_num)

            all_judge_outputs = []
            all_judge_valid_mask = []
            judge_crash_count = 0
            for example, output_item, image in zip(all_examples, tool_generation_output, all_images):
                turns = build_turn_records_from_tool_generation_output(output_item)
                question = output_item.get("prompt") or example.get("question") or example.get("query") or ""
                ground_truth = example.get("solution") or example.get("label") or ""
                try:
                    scores = self.turn_judge.judge_trajectory(
                        question=question,
                        ground_truth_answer=ground_truth,
                        turns=turns,
                        image=image,
                    )
                    judge_valid = True
                except Exception as e:
                    judge_crash_count += 1
                    scores = {f"score_{idx}": 0 for idx in range(len(turns))}
                    judge_valid = False
                    print(f"[step_judge][CRASH] question={question}")
                    print(f"[step_judge][CRASH] error_type={type(e).__name__}")
                    print(f"[step_judge][CRASH] error={e}")
                    print(f"[step_judge][CRASH] num_turns={len(turns)}")
                    print(f"[step_judge][CRASH] fallback_scores={scores}")
                all_judge_outputs.append(scores)
                all_judge_valid_mask.append(judge_valid)
                if self.step_judge_log_outputs:
                    turn_reward = compute_turn_judge_reward(
                        scores,
                        major_penalty=self.step_judge_major_penalty,
                        positive_cap=self._step_judge_positive_cap,
                    )
                    print(f"[step_judge] question={question}")
                    print(f"[step_judge] scores={scores}")
                    print(f"[step_judge] turn_reward={turn_reward}")
                    print(f"[step_judge] judge_valid={judge_valid}")
            self._step_judge_crash_counter += judge_crash_count
            self._step_judge_rollout_counter += len(all_examples)
            if self.step_crash_filter:
                self._step_judge_filtered_counter += judge_crash_count
        else:
            all_model_output_texts = [None] * len(all_examples)
            all_completion_ids = [None] * len(all_examples)
            all_judge_outputs = [None] * len(all_examples)
            all_judge_valid_mask = [None] * len(all_examples)

        all_model_output_texts = _orig.broadcast_object_list(all_model_output_texts, from_process=0)
        all_completion_ids = _orig.broadcast_object_list(all_completion_ids, from_process=0)
        all_judge_outputs = _orig.broadcast_object_list(all_judge_outputs, from_process=0)
        all_judge_valid_mask = _orig.broadcast_object_list(all_judge_valid_mask, from_process=0)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        model_output_texts = all_model_output_texts[process_slice]
        completion_ids_list = all_completion_ids[process_slice]
        local_judge_valid_mask = torch.tensor(
            [bool(x) for x in all_judge_valid_mask[process_slice]], dtype=torch.bool, device=device
        )

        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
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
        for i, reward_func in enumerate(self.reward_funcs):
            reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
            for key in reward_kwargs:
                for example in inputs:
                    reward_kwargs[key].extend([example[key]] * self.num_generations)
            reward_kwargs["model_output_texts"] = model_output_texts
            output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
            local_rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        rewards_per_func = _orig.gather(local_rewards_per_func)
        global_judge_valid_mask = torch.tensor([bool(x) for x in all_judge_valid_mask], dtype=torch.bool, device=device)
        valid_mask = global_judge_valid_mask if self.step_crash_filter else torch.ones_like(global_judge_valid_mask)

        local_turn_judge_rewards = torch.tensor(
            [
                compute_turn_judge_reward(
                    scores,
                    major_penalty=self.step_judge_major_penalty,
                    positive_cap=self._step_judge_positive_cap,
                )
                for scores in all_judge_outputs[process_slice]
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

        grouped_rewards = rewards.view(-1, self.num_generations)
        grouped_valid = valid_mask.view(-1, self.num_generations)
        valid_counts = grouped_valid.sum(dim=1)
        safe_counts = valid_counts.clamp(min=1)
        masked_sum = (grouped_rewards * grouped_valid.float()).sum(dim=1)
        mean_grouped_rewards = masked_sum / safe_counts.float()
        centered = grouped_rewards - mean_grouped_rewards.unsqueeze(1)
        masked_sq = (centered ** 2) * grouped_valid.float()
        var_grouped_rewards = masked_sq.sum(dim=1) / safe_counts.float()
        std_grouped_rewards = torch.sqrt(var_grouped_rewards)
        expanded_mean = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        expanded_std = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)

        advantages = torch.zeros_like(rewards)
        advantages[valid_mask] = (rewards[valid_mask] - expanded_mean[valid_mask]) / (expanded_std[valid_mask] + 1e-4)
        advantages = advantages[process_slice]

        selected_mask = valid_mask if self.step_crash_filter else torch.ones_like(valid_mask)
        if selected_mask.any():
            reward_per_func = rewards_per_func[selected_mask].mean(0)
            reward_mean = rewards[selected_mask].mean().item()
            reward_std = expanded_std[selected_mask].mean().item()
            turn_reward_mean = turn_judge_rewards[selected_mask].mean().item()
        else:
            reward_per_func = torch.zeros(rewards_per_func.size(1), dtype=torch.float32, device=device)
            reward_mean = 0.0
            reward_std = 0.0
            turn_reward_mean = 0.0

        for idx, reward_name in enumerate(self.reward_func_names):
            self._metrics[f"rewards/{reward_name}"].append(reward_per_func[idx].item())
        self._metrics["reward"].append(reward_mean)
        self._metrics["reward_std"].append(reward_std)
        self._metrics["turn_judge_reward"].append(turn_reward_mean)

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "judge_valid_mask": local_judge_valid_mask,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        judge_valid_mask = inputs["judge_valid_mask"].bool()
        if self.step_crash_filter:
            completion_mask = completion_mask * judge_valid_mask.unsqueeze(1).int()

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        pixel_values = inputs["pixel_values"].to(dtype=torch.bfloat16)
        image_grid_thw = inputs["image_grid_thw"]
        logits_to_keep = completion_ids.size(1)

        per_token_logps = self._get_per_token_logps(
            model,
            input_ids,
            attention_mask,
            pixel_values,
            image_grid_thw,
            logits_to_keep,
        )

        ref_per_token_logps = inputs["ref_per_token_logps"]
        per_token_kl = (
            torch.exp(ref_per_token_logps - per_token_logps)
            - (ref_per_token_logps - per_token_logps)
            - 1
        )

        advantages = inputs["advantages"]
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)

        per_row_token_count = completion_mask.sum(dim=1)
        active_rows = per_row_token_count > 0
        per_row_loss = torch.zeros_like(per_row_token_count, dtype=per_token_loss.dtype)
        per_row_loss[active_rows] = (
            (per_token_loss[active_rows] * completion_mask[active_rows]).sum(dim=1)
            / per_row_token_count[active_rows]
        )

        if active_rows.any():
            loss = per_row_loss[active_rows].mean()
            completion_length = (
                self.accelerator.gather_for_metrics(per_row_token_count[active_rows])
                .float()
                .mean()
                .item()
            )
            mean_kl = (
                (per_token_kl[active_rows] * completion_mask[active_rows]).sum(dim=1)
                / per_row_token_count[active_rows]
            ).mean()
        else:
            loss = per_token_logps.sum() * 0.0
            completion_length = 0.0
            mean_kl = per_token_logps.sum() * 0.0

        self._metrics["completion_length"].append(completion_length)
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        logs = dict(logs)
        prefix = ""
        if logs and next(iter(logs.keys())).startswith("eval_"):
            prefix = "eval_"

        if self.accelerator.is_main_process:
            crash_count = float(self._step_judge_crash_counter)
            rollout_count = float(self._step_judge_rollout_counter)
            filtered_count = float(self._step_judge_filtered_counter)
            logs[f"{prefix}step_judge_crash_count"] = crash_count
            logs[f"{prefix}step_judge_crash_rate"] = crash_count / rollout_count if rollout_count > 0 else 0.0
            logs[f"{prefix}step_judge_filtered_count"] = filtered_count
            logs[f"{prefix}step_judge_filtered_rate"] = filtered_count / rollout_count if rollout_count > 0 else 0.0
            self._step_judge_crash_counter = 0
            self._step_judge_rollout_counter = 0
            self._step_judge_filtered_counter = 0

        super().log(logs, start_time)


class Qwen2VLGRPOStepJudgeVLLMTrainerLegacyFiltered(_BaseFilteredStepJudgeToolVLLMTrainer):
    tool_generation_fn = staticmethod(legacy_vllm_generate_with_tool_calls)


class Qwen2VLGRPOStepJudgeVLLMTrainerSafeFiltered(_BaseFilteredStepJudgeToolVLLMTrainer):
    tool_generation_fn = staticmethod(safe_vllm_generate_with_tool_calls)
