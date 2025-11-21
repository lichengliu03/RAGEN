#!/usr/bin/env bash

set -euo pipefail

DEVICES="${DEVICES:-1}"
export CUDA_VISIBLE_DEVICES="${DEVICES}"
HYDRA_VISIBLE_DEVICES="'${DEVICES}'"
GPUS_PER_NODE=$(echo "${DEVICES}" | tr ',' '\n' | wc -l)

PYTHON_BIN="${PYTHON_BIN:-python3}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

{ source ~/.bashrc >/dev/null 2>&1 || true; } || true
[[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]] && source "$HOME/miniconda3/etc/profile.d/conda.sh"
[[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]] && source "$HOME/anaconda3/etc/profile.d/conda.sh"

eval "$(conda shell.bash hook)"
set +u
conda activate ragen || true
set -u

# ====== 配置 ======
# Eval scale
ES_VAL_GROUPS="${ES_VAL_GROUPS:-1024}"
ES_VAL_GROUP_SIZE="${ES_VAL_GROUP_SIZE:-1}"

VAL_TAGS="[MetamathQA,TheoremQA,GSM8k,GPQA,MMLU-STEM,HotpotQA,ConcurrentQA,MMLU,MMLUPro]"
repeat_value_list() {
  local value="$1" repeat="$2"
  local output="[" index
  for ((index=0; index<repeat; index++)); do
    if (( index > 0 )); then output+=","; fi
    output+="${value}"
  done
  output+="]"
  printf "%s" "${output}"
}
VAL_NGROUPS_LIST="$(repeat_value_list "${ES_VAL_GROUPS}" 9)"
TOTAL_VAL_GROUPS="$(( ES_VAL_GROUPS * 9 ))"

MAX_ACTIONS_PER_TRAJ="${MAX_ACTIONS_PER_TRAJ:-5}"
EVAL_TURN="${EVAL_TURN:-5}"

OUT_BASE="${REPO_DIR}/result/eval/r2w6"
mkdir -p "${OUT_BASE}"

WANDB_PROJECT="${WANDB_PROJECT:-ufo_rebuttal}"

echo "[R2W6] Repo: ${REPO_DIR}"
echo "[R2W6] Output base: ${OUT_BASE}"
echo "[R2W6] ES_VAL_GROUPS=${ES_VAL_GROUPS}, WANDB_PROJECT=${WANDB_PROJECT}"

summarize_accuracy() {
  local rollout_path="$1"
  ${PYTHON_BIN} - <<PY
import json
from pathlib import Path
from verl import DataProto

rollout_path = Path("${rollout_path}")
if not rollout_path.exists():
    print(json.dumps({"error": "Rollout not found"}))
    exit(0)

dp = DataProto.load_from_disk(str(rollout_path))
metrics = dp.meta_info.get("metrics", {}) if hasattr(dp.meta_info, "get") else {}

def to_builtin(val):
    if isinstance(val, (int, float, str, bool)) or val is None: return val
    if isinstance(val, dict): return {k: to_builtin(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)): return [to_builtin(v) for v in val]
    if hasattr(val, "item"):
        try: return val.item()
        except: pass
    return str(val)

metrics = {k: to_builtin(v) for k, v in metrics.items()}

total_correct = 0
total_attempted = 0
for idx in range(len(dp)):
    item = dp[idx]
    info = item.non_tensor_batch.get("info", {}) or {}
    if isinstance(info, list): info = info[-1] if info else {}
    if not isinstance(info, dict): continue
    total_correct += int(info.get("episode_num_correct", 0) or 0)
    total_attempted += int(info.get("episode_num_attempted", 0) or 0)

acc = (total_correct / total_attempted) if total_attempted > 0 else 0.0
print(json.dumps({
    "total_correct": total_correct,
    "total_attempted": total_attempted,
    "accuracy": acc,
    "metrics": metrics
}, ensure_ascii=False))
PY
}

run_train_and_eval() {
  local model="$1"
  local turn="$2"
  local train_steps="${TRAIN_STEPS:-200}"
  
  local model_safe
  model_safe="$(echo "${model}" | sed 's|/|-|g')"
  
  local ts run_name
  ts="$(date +%Y%m%d_%H%M%S)"

  run_name="r2w6_train_${model_safe}_turn${turn}_HotpotQA_${ts}"

  echo "======================================================================="
  echo "[R2W6][Train] Starting training for ${train_steps} steps"
  echo "Model: ${model}, Turn: ${turn}, Name: ${run_name}"
  echo "======================================================================="

  RAGEN_SAVE_INTERVAL_HOURS=999999 \
  WANDB_PROJECT="${WANDB_PROJECT}" WANDB_NAME="${run_name}" WANDB_RUN_ID="${run_name}" \
  ${PYTHON_BIN} train.py \
    trainer.experiment_name="${run_name}" \
    trainer.project_name="${WANDB_PROJECT}" \
    model_path="${model}" \
    enable_response_mask=False \
    system.CUDA_VISIBLE_DEVICES="${HYDRA_VISIBLE_DEVICES}" \
    trainer.n_gpus_per_node=${GPUS_PER_NODE} \
    trainer.save_freq=100 \
    ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    critic.ppo_mini_batch_size=4 \
    trainer.total_training_steps=${train_steps} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GPUS_PER_NODE} \
    agent_proxy.max_turn=${turn} \
    es_manager.train.env_configs.tags=[HotpotQA] \
    es_manager.train.env_groups=1 \
    es_manager.train.env_configs.n_groups=[1] \
    es_manager.val.env_configs.tags=[HotpotQA] \
    custom_envs.HotpotQA.max_actions_per_traj=${MAX_ACTIONS_PER_TRAJ} \
    actor_rollout_ref.actor.checkpoint.save_contents=[hf_model] || true

  echo "[R2W6][Train] Finished. Searching for step-${train_steps} checkpoint..."

  # 查找训练结束后的 checkpoint (global_step_200)
  local ckpt_path
  ckpt_path=$(${PYTHON_BIN} - <<PY
import os
from pathlib import Path

repo_dir = Path("${REPO_DIR}")
project_name = "${WANDB_PROJECT}"
run_name = "${run_name}"
target_step = "global_step_${train_steps}"

base_dir = repo_dir / "checkpoints" / project_name / run_name
if not base_dir.exists():
    base_dir = repo_dir / "checkpoints" / run_name

found_path = ""
if base_dir.exists():
    step_dir = base_dir / target_step
    hf_dir = step_dir / "actor" / "huggingface"
    if hf_dir.exists():
        found_path = str(hf_dir)

print(found_path)
PY
)

  if [[ -z "${ckpt_path}" ]]; then
    echo "[R2W6][Error] No checkpoint found for step ${train_steps}. Exiting..."
    return 1
  fi

  echo "[R2W6][Eval] Found Checkpoint: ${ckpt_path}"

  # 开始评估
  local out_dir="${OUT_BASE}/${model_safe}/step_${train_steps}_turn_${turn}"
  mkdir -p "${out_dir}"
  local rollout_path="${out_dir}/rollouts.pkl"

  if [[ -f "${rollout_path}" ]]; then
    echo "[R2W6][Eval] Skip existing rollout: ${rollout_path}"
  else
    local eval_name="${run_name}_eval_step${train_steps}"
    echo "[R2W6][Eval] Running evaluation..."
    
    WANDB_PROJECT="${WANDB_PROJECT}" WANDB_NAME="${eval_name}" WANDB_RUN_ID="${eval_name}" \
    ${PYTHON_BIN} -m ragen.llm_agent.agent_proxy --config-name eval \
      trainer.experiment_name="${eval_name}" \
      trainer.project_name="${WANDB_PROJECT}" \
      actor_rollout_ref.model.path="${ckpt_path}" \
      enable_response_mask=False \
      system.CUDA_VISIBLE_DEVICES="${HYDRA_VISIBLE_DEVICES}" \
      trainer.n_gpus_per_node=${GPUS_PER_NODE} \
      actor_rollout_ref.rollout.tensor_model_parallel_size=${GPUS_PER_NODE} \
      es_manager.train.env_configs.tags=[HotpotQA] \
      es_manager.train.env_groups=1 \
      es_manager.train.env_configs.n_groups=[1] \
      es_manager.val.env_configs.tags=${VAL_TAGS} \
      es_manager.val.env_configs.n_groups=${VAL_NGROUPS_LIST} \
      es_manager.val.env_groups=${TOTAL_VAL_GROUPS} \
      es_manager.val.group_size=${ES_VAL_GROUP_SIZE} \
      custom_envs.HotpotQA.max_actions_per_traj=${MAX_ACTIONS_PER_TRAJ} \
      agent_proxy.max_turn=${EVAL_TURN} \
      actor_rollout_ref.rollout.val_kwargs.do_sample=false \
      output.dir="${out_dir}" \
      output.filename="rollouts.pkl" \
      output.append_timestamp=false \
      trainer.logger=[console]
  fi

  if [[ -f "${rollout_path}" ]]; then
    summary_json="$(summarize_accuracy "${rollout_path}")"
    echo "[R2W6] Summary (Step=${train_steps}, turn=${turn}): ${summary_json}"
    echo "${summary_json}" > "${out_dir}/summary.json"
  fi
}

run_train_and_eval "meta-llama/Llama-3.2-3B-Instruct" "1"
run_train_and_eval "meta-llama/Llama-3.2-3B-Instruct" "5"

echo "[All Done] Results under: ${OUT_BASE}"
