#!/usr/bin/env bash

set -euo pipefail

# ====== 基础配置 ======
DEVICES="${DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES="${DEVICES}"
HYDRA_VISIBLE_DEVICES="'${DEVICES}'"

PYTHON_BIN="${PYTHON_BIN:-python3}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

{ source ~/.bashrc >/dev/null 2>&1 || true; } || true
[[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]] && source "$HOME/miniconda3/etc/profile.d/conda.sh"
[[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]] && source "$HOME/anaconda3/etc/profile.d/conda.sh"

eval "$(conda shell.bash hook)"
conda activate ragen || true

# ====== 模型与任务参数 ======
DEFAULT_MODEL="Qwen/Qwen2.5-3B-Instruct"
MODELS_ENV="${MODELS:-}"
if [[ -n "${MODELS_ENV}" ]]; then
  IFS=' ' read -r -a MODELS <<< "${MODELS_ENV}"
else
  MODELS=("${DEFAULT_MODEL}")
fi

# Episode 相关
NQ="${NQ:-5}"
MAX_TURN="${MAX_TURN:-5}"
MAX_ACTIONS_PER_TRAJ="${MAX_ACTIONS_PER_TRAJ:-5}"

# 评估标签与采样规模
VAL_TAGS="[MetamathQA,TheoremQA,GSM8k,GPQA,MMLU-STEM,HotpotQA,ConcurrentQA,MMLU,MMLUPro]"
ES_VAL_GROUPS="${ES_VAL_GROUPS:-128}"
ES_VAL_GROUP_SIZE="${ES_VAL_GROUP_SIZE:-1}"
repeat_value_list() {
  local value="$1" repeat="$2"
  local output="[" index
  for ((index=0; index<repeat; index++)); do
    if (( index > 0 )); then
      output+=","
    fi
    output+="${value}"
  done
  output+="]"
  printf "%s" "${output}"
}
VAL_NGROUPS_LIST="$(repeat_value_list "${ES_VAL_GROUPS}" 9)"
TOTAL_VAL_GROUPS="$(( ES_VAL_GROUPS * 9 ))"

OUT_BASE="${REPO_DIR}/result/eval/r1q1"
mkdir -p "${OUT_BASE}"

WANDB_PROJECT="${WANDB_PROJECT:-ufo_rebuttal}"

echo "[R1Q1] Repo: ${REPO_DIR}"
echo "[R1Q1] Output base: ${OUT_BASE}"
echo "[R1Q1] MODELS: ${MODELS[*]}"
echo "[R1Q1] NQ=${NQ}, ES_VAL_GROUPS=${ES_VAL_GROUPS}, ES_VAL_GROUP_SIZE=${ES_VAL_GROUP_SIZE}, WANDB_PROJECT=${WANDB_PROJECT}"

summarize_accuracy() {
  local rollout_path="$1"
  ${PYTHON_BIN} - <<PY
import json
from pathlib import Path
from verl import DataProto

rollout_path = Path("${rollout_path}")
dp = DataProto.load_from_disk(str(rollout_path))
metrics = dp.meta_info.get("metrics", {}) if hasattr(dp.meta_info, "get") else {}

def to_builtin(val):
    if isinstance(val, (int, float, str, bool)) or val is None:
        return val
    if isinstance(val, dict):
        return {k: to_builtin(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [to_builtin(v) for v in val]
    if hasattr(val, "item"):
        try:
            return val.item()
        except Exception:
            pass
    return str(val)

metrics = {k: to_builtin(v) for k, v in metrics.items()}

# Aggregate episode_num_correct / episode_num_attempted from the last step info of each trajectory
#
# Note: if the data structure changes, this is best-effort; otherwise it returns empty stats
total_correct = 0
total_attempted = 0
for idx in range(len(dp)):
    item = dp[idx]
    info = item.non_tensor_batch.get("info", {}) or {}
    if isinstance(info, list):
        info = info[-1] if info else {}
    if not isinstance(info, dict):
        continue
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

resolve_model_path() {
  local fallback="$1"
  local tracker
  tracker="$(ls -t checkpoints/*/*/latest_checkpointed_iteration.txt 2>/dev/null | head -n1 || true)"
  if [[ -z "${tracker}" || ! -f "${tracker}" ]]; then
    echo "${fallback}"
    return
  fi
  local ckpt_dir
  ckpt_dir="$(dirname "${tracker}")"
  local last_step
  last_step="$(tr -d '\n' < "${tracker}")"
  local candidate="${ckpt_dir}/global_step_${last_step}/actor/huggingface"
  if [[ -d "${candidate}" ]]; then
    echo "${candidate}"
  else
    echo "${fallback}"
  fi
}

run_train() {
  local model="$1"
  local run_name="$2"
  echo "[R1Q1][Train] model=${model}, NQ=${NQ}"
  WANDB_PROJECT="${WANDB_PROJECT}" WANDB_NAME="${run_name}" WANDB_RUN_ID="${run_name}" \
  ${PYTHON_BIN} train.py \
    trainer.experiment_name="${run_name}" \
    model_path="${model}" \
    system.CUDA_VISIBLE_DEVICES="${HYDRA_VISIBLE_DEVICES}" \
    trainer.n_gpus_per_node=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    es_manager.train.env_configs.tags=[MetamathQA] \
    es_manager.val.env_configs.tags=[MetamathQA] \
    +custom_envs.MetamathQA.env_config.multi_question_mode=true \
    +custom_envs.MetamathQA.env_config.n_questions_per_episode=${NQ} \
    custom_envs.MetamathQA.max_actions_per_traj=${MAX_ACTIONS_PER_TRAJ} \
    actor_rollout_ref.actor.checkpoint.save_contents=[hf_model]
}

run_eval() {
  local model_path="$1"
  local eval_name="$2"
  local out_dir="$3"
  echo "[R1Q1][Eval] model=${model_path}, NQ=${NQ}"
  WANDB_PROJECT="${WANDB_PROJECT}" WANDB_NAME="${eval_name}" WANDB_RUN_ID="${eval_name}" \
  ${PYTHON_BIN} -m ragen.llm_agent.agent_proxy --config-name eval \
    trainer.experiment_name="${eval_name}" \
    actor_rollout_ref.model.path="${model_path}" \
    system.CUDA_VISIBLE_DEVICES="${HYDRA_VISIBLE_DEVICES}" \
    trainer.n_gpus_per_node=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    es_manager.train.env_configs.tags=[MetamathQA] \
    es_manager.train.env_configs.n_groups=[1] \
    es_manager.val.env_configs.tags=${VAL_TAGS} \
    es_manager.val.env_configs.n_groups=${VAL_NGROUPS_LIST} \
    es_manager.val.env_groups=${TOTAL_VAL_GROUPS} \
    es_manager.val.group_size=${ES_VAL_GROUP_SIZE} \
    +custom_envs.MetamathQA.env_config.multi_question_mode=false \
    +custom_envs.MetamathQA.env_config.n_questions_per_episode=1 \
    custom_envs.MetamathQA.max_actions_per_traj=${MAX_ACTIONS_PER_TRAJ} \
    agent_proxy.max_turn=${MAX_TURN} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=false \
    output.dir="${out_dir}" \
    output.filename="rollouts.pkl" \
    output.append_timestamp=false \
    trainer.logger=[console]
}

for model in "${MODELS[@]}"; do
  model_safe="$(echo "${model}" | sed 's|/|-|g')"
  out_dir="${OUT_BASE}/${model_safe}"
  mkdir -p "${out_dir}"

  ts="$(date +%Y%m%d_%H%M%S)"
  run_name="r1q1_${model_safe}_NQ${NQ}_${ts}"
  eval_name="${run_name}_eval_g${ES_VAL_GROUPS}_k${ES_VAL_GROUP_SIZE}"

  run_train "${model}" "${run_name}"

  hf_dir="$(resolve_model_path "${model}")"
  if [[ "${hf_dir}" == "${model}" ]]; then
    echo "[R1Q1][WARN] HF weights not found, fallback to original model"
  fi

  run_eval "${hf_dir}" "${eval_name}" "${out_dir}"

  rollout_path="${out_dir}/rollouts.pkl"
  if [[ -f "${rollout_path}" ]]; then
    summary_json="$(summarize_accuracy "${rollout_path}")"
    echo "[R1Q1] Summary: ${summary_json}"
    echo "${summary_json}" > "${out_dir}/summary.json"
  else
    echo "[R1Q1][WARN] Missing rollout at ${rollout_path}"
  fi
done

echo "[All Done] Results under: ${OUT_BASE}"
