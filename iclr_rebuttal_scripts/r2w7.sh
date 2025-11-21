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

ES_VAL_GROUPS="${ES_VAL_GROUPS:-8}"
ES_VAL_GROUP_SIZE="${ES_VAL_GROUP_SIZE:-1}"

TURN_VALUES="${TURN_VALUES:-1,5}"
IFS=',' read -r -a TURNS <<< "${TURN_VALUES}"

TRAIN_STEPS="${TRAIN_STEPS:-10}"

MAX_ACTIONS_PER_TRAJ="${MAX_ACTIONS_PER_TRAJ:-5}"

make_repeat_list() {
  local val="$1" n="$2"
  local s="[" i
  for ((i=0; i<n; i++)); do
    if (( i > 0 )); then s+=","; fi
    s+="${val}"
  done
  s+="]"
  printf "%s" "${s}"
}
VAL_NGROUPS_LIST="$(make_repeat_list "${ES_VAL_GROUPS}" 9)"
TOTAL_VAL_GROUPS="$(( ES_VAL_GROUPS * 9 ))"
TAGS_LIST="[MetamathQA,TheoremQA,GSM8k,GPQA,MMLU-STEM,HotpotQA,ConcurrentQA,MMLU,MMLUPro]"

OUT_BASE="${OUT_BASE:-${REPO_DIR}}"
OUT_BASE="${OUT_BASE}/result/eval/r2w7"
mkdir -p "${OUT_BASE}"
SUMMARY_CSV="${OUT_BASE}/summary.csv"

WANDB_PROJECT="${WANDB_PROJECT:-ufo_rebuttal}"

if [[ ! -f "${SUMMARY_CSV}" ]]; then
  echo "algo,turn,model,avg_reward,MetamathQA/success,MetamathQA/pass@${ES_VAL_GROUP_SIZE},MetamathQA/num_actions" > "${SUMMARY_CSV}"
fi

echo "[R2W7] Repo: ${REPO_DIR}"
echo "[R2W7] Output base: ${OUT_BASE}"
echo "[R2W7] Eval groups: ${ES_VAL_GROUPS}, group_size: ${ES_VAL_GROUP_SIZE}, WANDB_PROJECT=${WANDB_PROJECT}"
echo "[R2W7] Turns: ${TURN_VALUES}, TRAIN_STEPS=${TRAIN_STEPS}, MAX_ACTIONS_PER_TRAJ=${MAX_ACTIONS_PER_TRAJ}"

summarize_run() {
  local rollout_path="$1"
  local turn="$2"
  local model="$3"
  ${PYTHON_BIN} - <<PY
import json
from pathlib import Path
from verl import DataProto

rollout_path = Path("${rollout_path}")
dp = DataProto.load_from_disk(str(rollout_path))
metrics = dp.meta_info.get("metrics", {})
rm = dp.batch["rm_scores"]
avg_reward = float(rm.sum(-1).mean().item())

success = float(metrics.get("MetamathQA/success", 0.0))
num_actions = float(metrics.get("MetamathQA/num_actions", 0.0))
pass_key = f"MetamathQA/pass@${ES_VAL_GROUP_SIZE}"
pass_at_k = float(metrics.get(pass_key, 0.0))

print(json.dumps({
  "turn": "${turn}",
  "model": "${model}",
  "avg_reward": avg_reward,
  "success": success,
  "pass_at_k": pass_at_k,
  "num_actions": num_actions
}, ensure_ascii=False))
PY
}

safe_append_csv() {
  local row="$1" key="$2"
  if ! grep -Fq -- "${key}" "${SUMMARY_CSV}"; then
    echo "${row}" >> "${SUMMARY_CSV}"
  else
    echo "[R2W7] Skip duplicate CSV row for key: ${key}"
  fi
}

run_experiment() {
  local model="$1"
  local algo="$2"
  local turn="$3"
  local train_steps="${TRAIN_STEPS}"

  local model_safe="$(echo "${model}" | sed 's|/|-|g')"
  local out_dir="${OUT_BASE}/${model_safe}/${algo}/turn_${turn}"
  mkdir -p "${out_dir}"
  local rollout_path="${out_dir}/rollouts.pkl"

  local ts run_name eval_name
  ts="$(date +%Y%m%d_%H%M%S)"
  run_name="r2w7_${algo}_${model_safe}_turn${turn}_steps${train_steps}_${ts}"
  eval_name="${run_name}_eval_g${ES_VAL_GROUPS}_k${ES_VAL_GROUP_SIZE}"

  if [[ -f "${rollout_path}" ]]; then
    echo "[R2W7] Rollout exists, skip: algo=${algo}, turn=${turn}, model=${model}"
  else
    local algo_args=()
    if [[ "${algo}" == "grpo" ]]; then
      algo_args=(
        "algorithm.adv_estimator=grpo"
        "actor_rollout_ref.actor.use_kl_loss=True"
        "actor_rollout_ref.actor.kl_loss_coef=0.001"
        "algorithm.use_kl_in_reward=False"
      )
    elif [[ "${algo}" == "dapo" ]]; then
      algo_args=(
        "algorithm.adv_estimator=grpo"
        "actor_rollout_ref.actor.use_kl_loss=False"
        "algorithm.use_kl_in_reward=False"
        "+algorithm.filter_groups.enable=True"
        "+algorithm.filter_groups.metric=acc"
      )
    else
      echo "[R2W7][Error] Unknown algorithm: ${algo}"
      return 1
    fi

    echo "[R2W7][Train] algo=${algo}, turn=${turn}, steps=${train_steps}, model=${model}"
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
      agent_proxy.max_turn=${turn} \
      trainer.total_training_steps=${train_steps} \
      actor_rollout_ref.actor.checkpoint.save_contents=[hf_model] \
      "${algo_args[@]}"

    local LATEST_TRACKER="$(ls -t checkpoints/*/*/latest_checkpointed_iteration.txt 2>/dev/null | head -n1 || true)"
    local HF_DIR=""
    if [[ -n "${LATEST_TRACKER}" && -f "${LATEST_TRACKER}" ]]; then
      local CKPT_DIR="$(dirname "${LATEST_TRACKER}")"
      local last_step="$(tr -d '\n' < "${LATEST_TRACKER}")"
      HF_DIR="${CKPT_DIR}/global_step_${last_step}/actor/huggingface"
    fi
    if [[ -z "${HF_DIR}" || ! -d "${HF_DIR}" ]]; then
      echo "[R2W7][WARN] HF weights not found, fallback to original model: ${model}"
      HF_DIR="${model}"
    fi

    echo "[R2W7][Eval] algo=${algo}, turn=${turn}, model=${HF_DIR}"
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
      agent_proxy.max_turn=${turn} \
      actor_rollout_ref.rollout.val_kwargs.do_sample=false \
      output.dir="${out_dir}" \
      output.filename="rollouts.pkl" \
      output.append_timestamp=false \
      trainer.logger=[console]
  fi

  if [[ -f "${rollout_path}" ]]; then
    local summary_json
    summary_json="$(summarize_run "${rollout_path}" "${turn}" "${model}")"
    echo "[R2W7] Summary (algo=${algo}, turn=${turn}): ${summary_json}"
    local avg_reward success pass_at_k num_actions
    avg_reward="$(echo "${summary_json}" | ${PYTHON_BIN} -c "import sys,json;print(json.load(sys.stdin)['avg_reward'])")"
    success="$(echo "${summary_json}" | ${PYTHON_BIN} -c "import sys,json;print(json.load(sys.stdin)['success'])")"
    pass_at_k="$(echo "${summary_json}" | ${PYTHON_BIN} -c "import sys,json;print(json.load(sys.stdin)['pass_at_k'])")"
    num_actions="$(echo "${summary_json}" | ${PYTHON_BIN} -c "import sys,json;print(json.load(sys.stdin)['num_actions'])")"
    local row="${algo},${turn},${model},${avg_reward},${success},${pass_at_k},${num_actions}"
    local key="${algo},${turn},${model},"
    safe_append_csv "${row}" "${key}"
    echo "${summary_json}" > "${out_dir}/summary.json"
  else
    echo "[R2W7][WARN] Missing rollout at ${rollout_path}"
  fi
}

MODEL="Qwen/Qwen2.5-3B-Instruct"
run_experiment "${MODEL}" "grpo" "1"
run_experiment "${MODEL}" "grpo" "5"
run_experiment "${MODEL}" "dapo" "1"
run_experiment "${MODEL}" "dapo" "5"

echo "[All Done] Summary CSV: ${SUMMARY_CSV}"

