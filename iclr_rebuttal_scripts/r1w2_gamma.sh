#!/usr/bin/env bash

set -euo pipefail

DEVICES="${DEVICES:-0,1}"
GPUS_PER_NODE=$(echo "${DEVICES}" | tr ',' '\n' | wc -l)
GPUS_PER_NODE=$((GPUS_PER_NODE))
HYDRA_CUDA_VISIBLE_DEVICES="'${DEVICES}'"
export CUDA_VISIBLE_DEVICES="${DEVICES}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

{ source ~/.bashrc >/dev/null 2>&1 || true; } || true
[[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]] && source "$HOME/miniconda3/etc/profile.d/conda.sh"
[[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]] && source "$HOME/anaconda3/etc/profile.d/conda.sh"

eval "$(conda shell.bash hook)"
conda activate ragen || true

ES_VAL_GROUPS="${ES_VAL_GROUPS:-1024}"
ES_VAL_GROUP_SIZE="${ES_VAL_GROUP_SIZE:-1}"

MAX_TURN="${MAX_TURN:-5}"
MAX_ACTIONS_PER_TRAJ="${MAX_ACTIONS_PER_TRAJ:-5}"

# Make n_groups list for 9 tags, each having ES_VAL_GROUPS groups (i.e., 128 each by default)
make_repeat_list() {
  local val="$1" n="$2"
  local s="[" i
  for ((i=0; i<n; i++)); do
    if (( i > 0 )); then s+=","
    fi
    s+="${val}"
  done
  s+="]"
  printf "%s" "${s}"
}
VAL_NGROUPS_LIST="$(make_repeat_list "${ES_VAL_GROUPS}" 9)"
TOTAL_VAL_GROUPS="$(( ES_VAL_GROUPS * 9 ))"
TAGS_LIST="[MetamathQA,TheoremQA,GSM8k,GPQA,MMLU-STEM,HotpotQA,ConcurrentQA,MMLU,MMLUPro]"

OUT_BASE="${REPO_DIR}/result/eval/r1w2_gamma"
mkdir -p "${OUT_BASE}"
SUMMARY_CSV="${OUT_BASE}/summary.csv"

WANDB_PROJECT="${WANDB_PROJECT:-ufo_rebuttal}"

if [[ ! -f "${SUMMARY_CSV}" ]]; then
  echo "gamma,model,avg_reward,MetamathQA/success,MetamathQA/pass@${ES_VAL_GROUP_SIZE},MetamathQA/num_actions" > "${SUMMARY_CSV}"
fi

echo "[Gamma-Sens] Repo: ${REPO_DIR}"
echo "[Gamma-Sens] Output base: ${OUT_BASE}"
echo "[Gamma-Sens] Eval groups: ${ES_VAL_GROUPS}, group_size: ${ES_VAL_GROUP_SIZE}, WANDB_PROJECT=${WANDB_PROJECT}"
echo "[Gamma-Sens] MAX_TURN=${MAX_TURN}, MAX_ACTIONS_PER_TRAJ=${MAX_ACTIONS_PER_TRAJ}"
echo "[Gamma-Sens] This script will run 2 experiments (1 model x 2 gammas)."

summarize_run() {
  local rollout_path="$1"
  local gamma="$2"
  local model="$3"
  ${PYTHON_BIN} - <<PY
import json, os, sys
from pathlib import Path
import torch
from verl import DataProto

rollout_path = Path("${rollout_path}")
dp = DataProto.load_from_disk(str(rollout_path))
metrics = dp.meta_info.get("metrics", {})
rm = dp.batch["rm_scores"]  # tensordict tensor [N, T]
avg_reward = float(rm.sum(-1).mean().item())

success = float(metrics.get("MetamathQA/success", 0.0))
num_actions = float(metrics.get("MetamathQA/num_actions", 0.0))
pass_key = f"MetamathQA/pass@${ES_VAL_GROUP_SIZE}"
pass_at_k = float(metrics.get(pass_key, 0.0))

print(json.dumps({
  "gamma": "${gamma}",
  "model": "${model}",
  "avg_reward": avg_reward,
  "success": success,
  "pass_at_k": pass_at_k,
  "num_actions": num_actions
}, ensure_ascii=False))
PY
}

safe_append_csv() {
  # $1=row, $2=key (grep -F)
  local row="$1" key="$2"
  if ! grep -Fq -- "${key}" "${SUMMARY_CSV}"; then
    echo "${row}" >> "${SUMMARY_CSV}"
  else
    echo "[Gamma-Sens] Skip duplicate CSV row for key: ${key}"
  fi
}

run_gamma() {
  local model="$1"
  local gamma="$2"
  local model_safe out_dir rollout_path
  model_safe="$(echo "${model}" | sed 's|/|-|g')"
  out_dir="${OUT_BASE}/${model_safe}/gamma_${gamma}"
  mkdir -p "${out_dir}"
  rollout_path="${out_dir}/rollouts.pkl"

  local ts run_name eval_name
  ts="$(date +%Y%m%d_%H%M%S)"
  run_name="r1w2_gamma_${model_safe}_g${gamma}_${ts}"
  eval_name="${run_name}_eval_g${ES_VAL_GROUPS}_k${ES_VAL_GROUP_SIZE}"

  if [[ -f "${rollout_path}" ]]; then
    echo "[Gamma-Sens] Rollout exists, skip train/eval: model=${model}, gamma=${gamma}"
  else
    # -------------------
    # 1) Light training -> save HF weights
    # -------------------
    echo "[Gamma-Sens][Train] model=${model}, gamma=${gamma} (following base.yaml training settings)"
    WANDB_PROJECT="${WANDB_PROJECT}" WANDB_NAME="${run_name}" WANDB_RUN_ID="${run_name}" \
    ${PYTHON_BIN} train.py \
      trainer.experiment_name="${run_name}" \
      trainer.project_name="${WANDB_PROJECT}" \
      model_path="${model}" \
      system.CUDA_VISIBLE_DEVICES=${HYDRA_CUDA_VISIBLE_DEVICES} \
      trainer.n_gpus_per_node=${GPUS_PER_NODE} \
      actor_rollout_ref.rollout.tensor_model_parallel_size=${GPUS_PER_NODE} \
      es_manager.train.env_configs.tags=[MetamathQA] \
      es_manager.val.env_configs.tags=[MetamathQA] \
      custom_envs.MetamathQA.env_config.gamma=${gamma} \
      actor_rollout_ref.actor.checkpoint.save_contents=[hf_model]

    # Parse latest HF weights directory (find the latest training under default checkpoints)
    LATEST_TRACKER="$(ls -t checkpoints/*/*/latest_checkpointed_iteration.txt 2>/dev/null | head -n1 || true)"
    HF_DIR=""
    if [[ -n "${LATEST_TRACKER}" && -f "${LATEST_TRACKER}" ]]; then
      CKPT_DIR="$(dirname "${LATEST_TRACKER}")"
      last_step="$(tr -d '\n' < "${LATEST_TRACKER}")"
      HF_DIR="${CKPT_DIR}/global_step_${last_step}/actor/huggingface"
    fi
    if [[ -z "${HF_DIR}" || ! -d "${HF_DIR}" ]]; then
      echo "[Gamma-Sens][WARN] HF weights not found, falling back to original model: ${model}"
      HF_DIR="${model}"
    fi

    # -------------------
    # 2) Evaluation
    # -------------------
    echo "[Gamma-Sens][Eval] model=${HF_DIR}, gamma=${gamma}"
    WANDB_PROJECT="${WANDB_PROJECT}" WANDB_NAME="${eval_name}" WANDB_RUN_ID="${eval_name}" \
    ${PYTHON_BIN} -m ragen.llm_agent.agent_proxy --config-name eval \
      trainer.experiment_name="${eval_name}" \
      trainer.project_name="${WANDB_PROJECT}" \
      actor_rollout_ref.model.path="${HF_DIR}" \
      system.CUDA_VISIBLE_DEVICES=${HYDRA_CUDA_VISIBLE_DEVICES} \
      trainer.n_gpus_per_node=${GPUS_PER_NODE} \
      actor_rollout_ref.rollout.tensor_model_parallel_size=${GPUS_PER_NODE} \
      es_manager.train.env_configs.tags=[MetamathQA] \
      es_manager.train.env_configs.n_groups=[1] \
      es_manager.train.env_groups=1 \
      es_manager.val.env_configs.tags=${TAGS_LIST} \
      es_manager.val.env_configs.n_groups=${VAL_NGROUPS_LIST} \
      es_manager.val.env_groups=${TOTAL_VAL_GROUPS} \
      es_manager.val.group_size=${ES_VAL_GROUP_SIZE} \
      custom_envs.MetamathQA.max_actions_per_traj=${MAX_ACTIONS_PER_TRAJ} \
      agent_proxy.max_turn=${MAX_TURN} \
      actor_rollout_ref.rollout.val_kwargs.do_sample=false \
      custom_envs.MetamathQA.env_config.gamma=${gamma} \
      output.dir="${out_dir}" \
      output.filename="rollouts.pkl" \
      output.append_timestamp=false \
      trainer.logger=[console]
  fi

  if [[ -f "${rollout_path}" ]]; then
    summary_json="$(summarize_run "${rollout_path}" "${gamma}" "${model}")"
    echo "[Gamma-Sens] Summary: ${summary_json}"
    avg_reward="$(echo "${summary_json}" | ${PYTHON_BIN} -c "import sys,json;print(json.load(sys.stdin)['avg_reward'])")"
    success="$(echo "${summary_json}" | ${PYTHON_BIN} -c "import sys,json;print(json.load(sys.stdin)['success'])")"
    pass_at_k="$(echo "${summary_json}" | ${PYTHON_BIN} -c "import sys,json;print(json.load(sys.stdin)['pass_at_k'])")"
    num_actions="$(echo "${summary_json}" | ${PYTHON_BIN} -c "import sys,json;print(json.load(sys.stdin)['num_actions'])")"
    row="${gamma},${model},${avg_reward},${success},${pass_at_k},${num_actions}"
    key="${gamma},${model},"
    safe_append_csv "${row}" "${key}"
    echo "${summary_json}" > "${out_dir}/summary.json"
  else
    echo "[Gamma-Sens][WARN] Missing rollout at ${rollout_path}"
  fi
}

run_gamma "Qwen/Qwen2.5-3B-Instruct" "1.0"
run_gamma "Qwen/Qwen2.5-3B-Instruct" "0"

echo "[All Done] Summary CSV: ${SUMMARY_CSV}"
