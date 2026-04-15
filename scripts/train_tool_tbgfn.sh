#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

: "${NPROC_PER_NODE:=1}"
: "${MODEL_NAME_OR_PATH:=Qwen/Qwen2-VL-2B-Instruct}"
: "${DATASET_NAME:=your_dataset_name}"
: "${DATASET_SPLIT:=train}"
: "${PROMPT_COLUMN:=prompt}"
: "${IMAGE_COLUMN:=image}"
: "${ANSWER_COLUMN:=answer}"
: "${OUTPUT_DIR:=outputs/tool_tbgfn}"
: "${CONTROLLER_ADDR:=http://localhost:20001}"
: "${ACCURACY_REWARD_FN_PATH:=}"
: "${FORMAT_REWARD_FN_PATH:=}"
: "${PER_DEVICE_PROMPT_BATCH_SIZE:=1}"
: "${NUM_TRAJECTORIES_PER_PROMPT:=2}"
: "${GRADIENT_ACCUMULATION_STEPS:=1}"
: "${NUM_TRAIN_EPOCHS:=1}"
: "${MAX_STEPS:=-1}"
: "${LEARNING_RATE:=1e-6}"
: "${LOGZ_LEARNING_RATE:=1e-4}"
: "${WEIGHT_DECAY:=0.0}"
: "${MAX_GRAD_NORM:=1.0}"
: "${MAX_PROMPT_LENGTH:=4096}"
: "${MAX_COMPLETION_LENGTH:=1024}"
: "${MAX_ROUNDS:=6}"
: "${TEMPERATURE:=0.7}"
: "${TOP_P:=1.0}"
: "${USE_VLLM:=true}"
: "${VLLM_DEVICE:=auto}"
: "${VLLM_GPU_MEMORY_UTILIZATION:=0.6}"
: "${BF16:=true}"
: "${FP16:=false}"
: "${GRADIENT_CHECKPOINTING:=false}"
: "${DEEPSPEED:=}"
: "${LOGGING_STEPS:=10}"
: "${SAVE_STEPS:=200}"
: "${REPORT_TO:=wandb}"
: "${WANDB_PROJECT:=tool_tbgfn}"
: "${WANDB_ENTITY:=}"
: "${RUN_NAME:=tool_tbgfn_run}"
: "${SEED:=42}"
: "${LOG_REWARD_EPSILON:=1e-6}"
: "${LOG_REWARD_CLIP_MIN:=-20.0}"

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
