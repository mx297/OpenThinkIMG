from __future__ import annotations

from typing import Any, Callable, Optional, Union

import torch
from transformers import PreTrainedModel

from . import tool_vllm_grpo_trainer as _orig
from .mtgrpo_paper_credit import (
    assign_batch_segment_advantages,
    build_turn_spans,
    compose_local_global_advantages,
    normalize_groupwise_scalar,
    normalize_groupwise_turn_rewards,
)
from .step_judge_reward import (
    get_selected_format_reward_name,
    normalize_reward_name,
    validate_step_judge_reward_names,
)
from .tool_generation import vllm_generate_with_tool_calls as legacy_vllm_generate_with_tool_calls
from .tool_generation_safe import vllm_generate_with_tool_calls as safe_vllm_generate_with_tool_calls
from .turn_judge_paper_semantic import (
    PaperSemanticTurnJudge,
    build_non_final_turn_records_from_tool_generation_output,
)

RewardFunc = _orig.RewardFunc


class _BasePaperStepJudgeToolVLLMTrainer(_orig.Qwen2VLGRPOVLLMTrainer):
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
        paper_step_judge_model: str = "Qwen/Qwen2.5-VL-72B-Instruct",
        paper_step_judge_temperature: float = 0.0,
        paper_step_judge_do_sample: bool = False,
        paper_step_judge_top_p: float = 1.0,
        paper_step_judge_top_k: int = 0,
        paper_step_judge_repetition_penalty: float = 1.0,
        paper_step_judge_max_new_tokens: int = 128,
        paper_terminal_weight: float = 10.0,
        paper_format_weight: float = 0.5,
        paper_beta: float = 0.5,
        paper_credit_assignment_mode: str = "whole_turn",
        paper_step_crash_filter: bool = True,
        paper_log_judge_outputs: bool = True,
    ):
        self.reward_func_names = reward_func_names or []
        validate_step_judge_reward_names(self.reward_func_names)
        self.selected_format_reward_name = get_selected_format_reward_name(self.reward_func_names)
        self.paper_terminal_weight = paper_terminal_weight
        self.paper_format_weight = paper_format_weight
        self.paper_beta = paper_beta
        self.paper_credit_assignment_mode = paper_credit_assignment_mode
        self.paper_step_crash_filter = paper_step_crash_filter
        self.paper_log_judge_outputs = paper_log_judge_outputs
        self.paper_turn_judge = None
        self.paper_step_judge_model = paper_step_judge_model
        self.paper_step_judge_temperature = paper_step_judge_temperature
        self.paper_step_judge_do_sample = paper_step_judge_do_sample
        self.paper_step_judge_top_p = paper_step_judge_top_p
        self.paper_step_judge_top_k = paper_step_judge_top_k
        self.paper_step_judge_repetition_penalty = paper_step_judge_repetition_penalty
        self.paper_step_judge_max_new_tokens = paper_step_judge_max_new_tokens
        self._paper_step_judge_crash_counter = 0
        self._paper_step_judge_rollout_counter = 0
        self._paper_step_judge_filtered_counter = 0
        if not 0.0 <= self.paper_beta <= 1.0:
            raise ValueError("paper_beta must be in [0, 1].")

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
            self.paper_turn_judge = PaperSemanticTurnJudge(
                model_name=paper_step_judge_model,
                temperature=paper_step_judge_temperature,
                do_sample=paper_step_judge_do_sample,
                top_p=paper_step_judge_top_p,
                top_k=paper_step_judge_top_k,
                repetition_penalty=paper_step_judge_repetition_penalty,
                max_new_tokens=paper_step_judge_max_new_tokens,
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
                "Paper step-judge mode expected accuracy and exactly one format reward in reward_func_names."
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
            all_model_output_idss = [
                [list(turn_ids) for turn_ids in item.get("model_output_ids", [])] for item in tool_generation_output
            ]
            ave_tool_num = (
                sum(len(item.get("tool_outputs", [])) for item in tool_generation_output) / max(len(tool_generation_output), 1)
            )
            self._metrics["ave_tool_num"].append(ave_tool_num)

            all_judge_outputs = []
            all_judge_valid_mask = []
            judge_crash_count = 0
            for example, output_item, image in zip(all_examples, tool_generation_output, all_images):
                non_final_turns = build_non_final_turn_records_from_tool_generation_output(output_item)
                question = output_item.get("prompt") or example.get("question") or example.get("query") or ""
                ground_truth = example.get("solution") or example.get("label") or ""
                try:
                    scores = self.paper_turn_judge.judge_non_final_turns(
                        question=question,
                        ground_truth_answer=ground_truth,
                        turns=non_final_turns,
                        image=image,
                    )
                    judge_valid = True
                except Exception as e:
                    judge_crash_count += 1
                    scores = {f"score_{idx}": 0 for idx in range(len(non_final_turns))}
                    judge_valid = False
                    print(f"[paper_step_judge][CRASH] question={question}")
                    print(f"[paper_step_judge][CRASH] error_type={type(e).__name__}")
                    print(f"[paper_step_judge][CRASH] error={e}")
                    print(f"[paper_step_judge][CRASH] num_non_final_turns={len(non_final_turns)}")
                    print(f"[paper_step_judge][CRASH] fallback_scores={scores}")
                all_judge_outputs.append(scores)
                all_judge_valid_mask.append(judge_valid)
                if self.paper_log_judge_outputs:
                    print(f"[paper_step_judge] question={question}")
                    print(f"[paper_step_judge] scores={scores}")
                    print(f"[paper_step_judge] judge_valid={judge_valid}")
            self._paper_step_judge_crash_counter += judge_crash_count
            self._paper_step_judge_rollout_counter += len(all_examples)
            if self.paper_step_crash_filter:
                self._paper_step_judge_filtered_counter += judge_crash_count
        else:
            all_model_output_texts = [None] * len(all_examples)
            all_model_output_idss = [None] * len(all_examples)
            all_judge_outputs = [None] * len(all_examples)
            all_judge_valid_mask = [None] * len(all_examples)

        all_model_output_texts = _orig.broadcast_object_list(all_model_output_texts, from_process=0)
        all_model_output_idss = _orig.broadcast_object_list(all_model_output_idss, from_process=0)
        all_judge_outputs = _orig.broadcast_object_list(all_judge_outputs, from_process=0)
        all_judge_valid_mask = _orig.broadcast_object_list(all_judge_valid_mask, from_process=0)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        model_output_texts = all_model_output_texts[process_slice]
        local_model_output_idss = all_model_output_idss[process_slice]
        local_judge_valid_mask = torch.tensor(
            [bool(x) for x in all_judge_valid_mask[process_slice]], dtype=torch.bool, device=device
        )

        local_completion_id_tensors = []
        local_turn_spans = []
        for turn_id_lists in local_model_output_idss:
            flat_ids, spans = build_turn_spans(turn_id_lists or [], max_completion_length=self.max_completion_length)
            if len(flat_ids) == 0:
                flat_ids = [self.processing_class.eos_token_id]
                spans = []
            local_completion_id_tensors.append(torch.tensor(flat_ids, device=device))
            local_turn_spans.append(spans)

        completion_ids = _orig.pad(local_completion_id_tensors, padding_value=self.processing_class.pad_token_id)
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
                raise ValueError("Paper step-judge trainer currently supports callable reward functions only.")
            reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
            for key in reward_kwargs:
                for example in inputs:
                    reward_kwargs[key].extend([example[key]] * self.num_generations)
            reward_kwargs["model_output_texts"] = model_output_texts
            output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
            local_rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        rewards_per_func = _orig.gather(local_rewards_per_func)
        global_judge_valid_mask = torch.tensor(
            [bool(x) for x in all_judge_valid_mask], dtype=torch.bool, device=device
        )
        valid_mask = global_judge_valid_mask if self.paper_step_crash_filter else torch.ones_like(global_judge_valid_mask)

        accuracy_idx, format_idx = self._get_accuracy_and_format_indices()
        accuracy_values = rewards_per_func[:, accuracy_idx]
        format_values = rewards_per_func[:, format_idx]
        final_rewards = self.paper_terminal_weight * accuracy_values + self.paper_format_weight * format_values
        normalized_final_rewards = normalize_groupwise_scalar(
            final_rewards,
            group_size=self.num_generations,
            valid_mask=valid_mask,
            eps=1e-4,
        )

        local_turn_reward_lists = []
        per_trajectory_turn_means = []
        for score_dict in all_judge_outputs:
            ordered_scores = [float(score_dict[f"score_{idx}"]) for idx in range(len(score_dict))]
            local_turn_reward_lists.append(ordered_scores)
            per_trajectory_turn_means.append(sum(ordered_scores) / len(ordered_scores) if ordered_scores else 0.0)

        normalized_turn_reward_lists = normalize_groupwise_turn_rewards(
            local_turn_reward_lists,
            group_size=self.num_generations,
            valid_mask=valid_mask,
            eps=1e-4,
            device=device,
        )
        local_turn_advantage_lists, final_advantages = compose_local_global_advantages(
            normalized_turn_reward_lists,
            normalized_final_rewards,
            beta=self.paper_beta,
            valid_mask=valid_mask,
        )

        per_token_advantages = assign_batch_segment_advantages(
            batch_spans=local_turn_spans,
            batch_local_turn_advantages=local_turn_advantage_lists[process_slice],
            batch_final_advantages=final_advantages[process_slice],
            padded_length=completion_ids.size(1),
            mode=self.paper_credit_assignment_mode,
            device=device,
        )

        selected_mask = valid_mask if self.paper_step_crash_filter else torch.ones_like(valid_mask)
        if selected_mask.any():
            reward_per_func = rewards_per_func[selected_mask].mean(0)
            reward_mean = final_rewards[selected_mask].mean().item()
            reward_std = normalized_final_rewards[selected_mask].std(unbiased=False).item()
            turn_reward_mean = torch.tensor(per_trajectory_turn_means, device=device)[selected_mask].mean().item()
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
            "per_token_advantages": per_token_advantages,
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
        if self.paper_step_crash_filter:
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

        per_token_advantages = inputs["per_token_advantages"]
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * per_token_advantages
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
            crash_count = float(self._paper_step_judge_crash_counter)
            rollout_count = float(self._paper_step_judge_rollout_counter)
            filtered_count = float(self._paper_step_judge_filtered_counter)
            logs[f"{prefix}paper_step_judge_crash_count"] = crash_count
            logs[f"{prefix}paper_step_judge_crash_rate"] = crash_count / rollout_count if rollout_count > 0 else 0.0
            logs[f"{prefix}paper_step_judge_filtered_count"] = filtered_count
            logs[f"{prefix}paper_step_judge_filtered_rate"] = filtered_count / rollout_count if rollout_count > 0 else 0.0
            self._paper_step_judge_crash_counter = 0
            self._paper_step_judge_rollout_counter = 0
            self._paper_step_judge_filtered_counter = 0

        super().log(logs, start_time)


class Qwen2VLGRPOPaperStepJudgeVLLMTrainerLegacy(_BasePaperStepJudgeToolVLLMTrainer):
    tool_generation_fn = staticmethod(legacy_vllm_generate_with_tool_calls)


class Qwen2VLGRPOPaperStepJudgeVLLMTrainerSafe(_BasePaperStepJudgeToolVLLMTrainer):
    tool_generation_fn = staticmethod(safe_vllm_generate_with_tool_calls)
