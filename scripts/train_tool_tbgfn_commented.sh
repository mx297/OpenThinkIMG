#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"                    # Number of GPU processes launched by torchrun.
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2-VL-2B-Instruct}"  # Base model checkpoint used for training and rollout.
DATASET_NAME="${DATASET_NAME:-your_dataset_name}"       # Hugging Face dataset name or dataset loading script identifier.
DATASET_SPLIT="${DATASET_SPLIT:-train}"                 # Dataset split used for training.
PROMPT_COLUMN="${PROMPT_COLUMN:-prompt}"                # Column containing the prompt or question text.
IMAGE_COLUMN="${IMAGE_COLUMN:-image}"                   # Column containing the image path/object consumed by the model.
ANSWER_COLUMN="${ANSWER_COLUMN:-answer}"                # Column containing the reference answer.
OUTPUT_DIR="${OUTPUT_DIR:-outputs/tool_tbgfn}"          # Directory where checkpoints and trainer state are saved.
CONTROLLER_ADDR="${CONTROLLER_ADDR:-http://localhost:20001}"  # Tool server controller endpoint used during safe tool rollouts.
ACCURACY_REWARD_FN_PATH="${ACCURACY_REWARD_FN_PATH:-}"  # Dotted import path to the existing accuracy reward function.
FORMAT_REWARD_FN_PATH="${FORMAT_REWARD_FN_PATH:-}"      # Dotted import path to the existing format reward function.
PER_DEVICE_PROMPT_BATCH_SIZE="${PER_DEVICE_PROMPT_BATCH_SIZE:-1}"  # Number of prompts loaded per GPU before expanding by K trajectories.
NUM_TRAJECTORIES_PER_PROMPT="${NUM_TRAJECTORIES_PER_PROMPT:-2}"    # Fixed K trajectories sampled independently per prompt.
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"    # Number of accumulation steps before one optimizer update.
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"               # Number of epochs if MAX_STEPS is not used.
MAX_STEPS="${MAX_STEPS:--1}"                            # Maximum optimizer steps; -1 means train for full epochs.
LEARNING_RATE="${LEARNING_RATE:-1e-6}"                  # Learning rate for model / forward-policy parameters.
LOGZ_LEARNING_RATE="${LOGZ_LEARNING_RATE:-1e-4}"        # Learning rate for the TB logZ scalar.
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"                     # Weight decay applied to model parameters.
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"                   # Gradient clipping norm.
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"          # Maximum encoded prompt length.
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"  # Maximum tokens generated for one assistant turn.
MAX_ROUNDS="${MAX_ROUNDS:-6}"                           # Maximum safe tool-calling rounds per trajectory.
TEMPERATURE="${TEMPERATURE:-0.7}"                       # Sampling temperature used during rollout generation.
TOP_P="${TOP_P:-1.0}"                                   # Top-p nucleus sampling threshold.
USE_VLLM="${USE_VLLM:-true}"                            # Whether to use vLLM for rollout generation.
VLLM_DEVICE="${VLLM_DEVICE:-auto}"                      # Device for the vLLM engine; auto uses the next free GPU.
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.6}"  # Fraction of GPU memory reserved for vLLM.
BF16="${BF16:-true}"                                    # Enable bfloat16 training/inference where available.
FP16="${FP16:-false}"                                   # Enable fp16 when bf16 is unavailable.
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"  # Enable gradient checkpointing in the base model.
DEEPSPEED="${DEEPSPEED:-}"                              # Optional path to a DeepSpeed JSON config.
LOGGING_STEPS="${LOGGING_STEPS:-10}"                    # Log average reward and reward variance every N steps.
SAVE_STEPS="${SAVE_STEPS:-200}"                         # Save a checkpoint every N steps.
REPORT_TO="${REPORT_TO:-wandb}"                         # Logging backend; use wandb or set another value to disable it.
WANDB_PROJECT="${WANDB_PROJECT:-tool_tbgfn}"            # Weights & Biases project name.
WANDB_ENTITY="${WANDB_ENTITY:-}"                        # Optional Weights & Biases entity/team name.
RUN_NAME="${RUN_NAME:-tool_tbgfn_run}"                  # Human-readable run name shown in W&B.
SEED="${SEED:-42}"                                      # Global random seed.
LOG_REWARD_EPSILON="${LOG_REWARD_EPSILON:-1e-6}"        # Positive constant added before log-reward conversion.
LOG_REWARD_CLIP_MIN="${LOG_REWARD_CLIP_MIN:--20.0}"     # Minimum allowed log reward inside the TB objective.

python - <<'PY'
import importlib.util
import subprocess
import sys
for pkg in ["accelerate", "datasets", "wandb", "torchgfn"]:
    if importlib.util.find_spec(pkg) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
PY

TORCHRUN_ARGS=(--nproc_per_node="${NPROC_PER_NODE}")

CMD=(
  torchrun "${TORCHRUN_ARGS[@]}" r1_v/open_r1/tool_tbgfn.py
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --dataset_name "${DATASET_NAME}"
  --dataset_split "${DATASET_SPLIT}"
  --prompt_column "${PROMPT_COLUMN}"
  --image_column "${IMAGE_COLUMN}"
  --answer_column "${ANSWER_COLUMN}"
  --output_dir "${OUTPUT_DIR}"
  --controller_addr "${CONTROLLER_ADDR}"
  --accuracy_reward_fn_path "${ACCURACY_REWARD_FN_PATH}"
  --format_reward_fn_path "${FORMAT_REWARD_FN_PATH}"
  --per_device_prompt_batch_size "${PER_DEVICE_PROMPT_BATCH_SIZE}"
  --num_trajectories_per_prompt "${NUM_TRAJECTORIES_PER_PROMPT}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --max_steps "${MAX_STEPS}"
  --learning_rate "${LEARNING_RATE}"
  --logz_learning_rate "${LOGZ_LEARNING_RATE}"
  --weight_decay "${WEIGHT_DECAY}"
  --max_grad_norm "${MAX_GRAD_NORM}"
  --max_prompt_length "${MAX_PROMPT_LENGTH}"
  --max_completion_length "${MAX_COMPLETION_LENGTH}"
  --max_rounds "${MAX_ROUNDS}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --use_vllm "${USE_VLLM}"
  --vllm_device "${VLLM_DEVICE}"
  --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
  --bf16 "${BF16}"
  --fp16 "${FP16}"
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}"
  --logging_steps "${LOGGING_STEPS}"
  --save_steps "${SAVE_STEPS}"
  --report_to "${REPORT_TO}"
  --wandb_project "${WANDB_PROJECT}"
  --wandb_entity "${WANDB_ENTITY}"
  --run_name "${RUN_NAME}"
  --seed "${SEED}"
  --log_reward_epsilon "${LOG_REWARD_EPSILON}"
  --log_reward_clip_min "${LOG_REWARD_CLIP_MIN}"
)

if [[ -n "${DEEPSPEED}" ]]; then
  CMD+=(--deepspeed "${DEEPSPEED}")
fi

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
