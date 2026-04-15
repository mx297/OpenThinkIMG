from accelerate.utils.other import is_compiled_module
from trl.models import unwrap_model_for_generation

from .tool_vllm_gfn_tb_trainer_safe import Qwen2VLGFNTBVLLMTrainer


class Qwen2VLGFNTBVLLMTrainerFiltered(Qwen2VLGFNTBVLLMTrainer):
    def _maybe_sync_vllm_weights(self):
        if self.state.global_step == self._last_loaded_step:
            return
        if self.state.global_step > 0 and self.state.global_step % self.rollout_sync_interval != 0 and self._last_loaded_step >= 0:
            return
        with unwrap_model_for_generation(
            self.model,
            self.accelerator,
            gather_deepspeed3_params=False,
        ) as unwrapped_model:
            if is_compiled_module(unwrapped_model):
                state_dict = unwrapped_model._orig_mod.state_dict()
            else:
                state_dict = unwrapped_model.state_dict()
        filtered_items = [(k, v) for k, v in state_dict.items() if not k.startswith("gfn_")]
        if self.accelerator.is_main_process:
            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            llm_model.load_weights(filtered_items)
        self._last_loaded_step = self.state.global_step
        self.accelerator.wait_for_everyone()
