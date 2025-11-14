#!/usr/bin/env bash

#SBATCH -J r1q2
#SBATCH -N 1
#SBATCH --partition=gpuH200x8-interactive
#SBATCH --account=bfea-delta-gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 01:00:00
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

{ source ~/.bashrc >/dev/null 2>&1 || true; } || true
[[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]] && source "$HOME/miniconda3/etc/profile.d/conda.sh"
[[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]] && source "$HOME/anaconda3/etc/profile.d/conda.sh"

eval "$(conda shell.bash hook)"
conda activate ragen || true

MODELS_DEFAULT=("Qwen/Qwen2.5-3B-Instruct")
MODELS=("${MODELS[@]:-${MODELS_DEFAULT[@]}}")

# Number of questions per episode (effective when multi_question_mode is true)
NQ="${NQ:-5}"

# Eval scale
ES_VAL_GROUPS="${ES_VAL_GROUPS:-128}"
ES_VAL_GROUP_SIZE="${ES_VAL_GROUP_SIZE:-1}"

MAX_TURN="${MAX_TURN:-5}"
MAX_ACTIONS_PER_TRAJ="${MAX_ACTIONS_PER_TRAJ:-5}"

OUT_BASE="${REPO_DIR}/result/eval/r1q2"
mkdir -p "${OUT_BASE}"

WANDB_PROJECT="${WANDB_PROJECT:-ufo_rebuttal}"

echo "[R1Q2] Repo: ${REPO_DIR}"
echo "[R1Q2] Output base: ${OUT_BASE}"
echo "[R1Q2] MODELS: ${MODELS[*]}"
echo "[R1Q2] NQ=${NQ}, ES_VAL_GROUPS=${ES_VAL_GROUPS}, ES_VAL_GROUP_SIZE=${ES_VAL_GROUP_SIZE}, WANDB_PROJECT=${WANDB_PROJECT}"

summarize_accuracy() {
  local rollout_path="$1"
  ${PYTHON_BIN} - <<PY
import json
from pathlib import Path
from verl import DataProto

rollout_path = Path("${rollout_path}")
dp = DataProto.load_from_disk(str(rollout_path))

# Aggregate episode_num_correct / episode_num_attempted from the last step info of each trajectory
#
# Note: if the data structure changes, this is best-effort; otherwise it returns empty stats
total_correct = 0
total_attempted = 0
try:
    for i in range(len(dp)):
        item = dp[i]
        info = item.non_tensor_batch.get("info", {}) or {}
        if isinstance(info, list) and info:
            info = info[-1]
        c = int(info.get("episode_num_correct", 0) or 0)
        a = int(info.get("episode_num_attempted", 0) or 0)
        total_correct += c
        total_attempted += a
except Exception:
    pass

acc = (total_correct / total_attempted) if total_attempted > 0 else 0.0
print(json.dumps({
    "total_correct": total_correct,
    "total_attempted": total_attempted,
    "accuracy": acc
}, ensure_ascii=False))
PY
}

# Run one experiment for a given model, time budget (in hours), and max turn
run_for_hours() {
  local model="$1"
  local hours="$2"
  local turn="$3"
  local model_safe h_safe out_dir rollout_path
  model_safe="$(echo "${model}" | sed 's|/|-|g')"
  h_safe="$(echo "${hours}" | sed 's|\.|p|g')"
  out_dir="${OUT_BASE}/${model_safe}/T${h_safe}h_turn_${turn}"
  mkdir -p "${out_dir}"
  rollout_path="${out_dir}/rollouts.pkl"

  local ts run_name eval_name minutes timeout_spec
  ts="$(date +%Y%m%d_%H%M%S)"
  run_name="r1q2_${model_safe}_T${h_safe}h_turn${turn}_NQ${NQ}_${ts}"
  eval_name="${run_name}_eval_g${ES_VAL_GROUPS}_k${ES_VAL_GROUP_SIZE}"

  minutes="$(${PYTHON_BIN} - <<PY
h=float("${hours}")
print(int(round(h*60)))
PY
)"
  timeout_spec="${minutes}m"

  echo "[R1Q2][Train] model=${model}, NQ=${NQ}, time_budget=${hours}h (${timeout_spec}), turn=${turn}"
  WANDB_PROJECT="${WANDB_PROJECT}" WANDB_NAME="${run_name}" WANDB_RUN_ID="${run_name}" \
  timeout "${timeout_spec}" ${PYTHON_BIN} train.py \
    trainer.experiment_name="${run_name}" \
    model_path="${model}" \
    es_manager.train.env_configs.tags=[MetamathQA] \
    es_manager.val.env_configs.tags=[MetamathQA] \
    +custom_envs.MetamathQA.env_config.n_questions_per_episode=${NQ} \
    +custom_envs.MetamathQA.max_actions_per_traj=${MAX_ACTIONS_PER_TRAJ} \
    actor_rollout_ref.actor.checkpoint.save_contents=[hf_model] || true

  LATEST_TRACKER="$(ls -t checkpoints/*/*/latest_checkpointed_iteration.txt 2>/dev/null | head -n1 || true)"
  HF_DIR=""
  if [[ -n "${LATEST_TRACKER}" && -f "${LATEST_TRACKER}" ]]; then
    CKPT_DIR="$(dirname "${LATEST_TRACKER}")"
    last_step="$(tr -d '\n' < "${LATEST_TRACKER}")"
    HF_DIR="${CKPT_DIR}/global_step_${last_step}/actor/huggingface"
  fi
  if [[ -z "${HF_DIR}" || ! -d "${HF_DIR}" ]]; then
    echo "[R1Q2][WARN] HF weights not found, falling back to original model: ${model}"
    HF_DIR="${model}"
  fi

  echo "[R1Q2][Eval] model=${HF_DIR}, NQ=${NQ}, time_budget=${hours}h, turn=${turn}"
  WANDB_PROJECT="${WANDB_PROJECT}" WANDB_NAME="${eval_name}" WANDB_RUN_ID="${eval_name}" \
  ${PYTHON_BIN} -m ragen.llm_agent.agent_proxy --config-name eval \
    trainer.experiment_name="${eval_name}" \
    actor_rollout_ref.model.path="${HF_DIR}" \
    es_manager.train.env_configs.tags=[MetamathQA] \
    es_manager.train.env_configs.n_groups=[1] \
    es_manager.val.env_configs.tags=[MetamathQA] \
    es_manager.val.env_configs.n_groups=[${ES_VAL_GROUPS}] \
    es_manager.val.env_groups=${ES_VAL_GROUPS} \
    es_manager.val.group_size=${ES_VAL_GROUP_SIZE} \
    +custom_envs.MetamathQA.env_config.multi_question_mode=false \
    +custom_envs.MetamathQA.env_config.n_questions_per_episode=1 \
    custom_envs.MetamathQA.max_actions_per_traj=${MAX_ACTIONS_PER_TRAJ} \
    agent_proxy.max_turn=${turn} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=false \
    output.dir="${out_dir}" \
    output.filename="rollouts.pkl" \
    output.append_timestamp=false \
    trainer.logger=[console]

  if [[ -f "${rollout_path}" ]]; then
    summary_json="$(summarize_accuracy "${rollout_path}")"
    echo "[R1Q2] Summary (T=${hours}h, turn=${turn}): ${summary_json}"
    echo "${summary_json}" > "${out_dir}/summary.json"
  else
    echo "[R1Q2][WARN] Missing rollout at ${rollout_path}"
  fi
}

run_for_hours "Qwen/Qwen2.5-3B-Instruct" "0.5" "1"
run_for_hours "Qwen/Qwen2.5-3B-Instruct" "0.5" "5"
run_for_hours "Qwen/Qwen2.5-3B-Instruct" "1.0" "1"
run_for_hours "Qwen/Qwen2.5-3B-Instruct" "1.0" "5"
run_for_hours "Qwen/Qwen2.5-3B-Instruct" "1.5" "1"
run_for_hours "Qwen/Qwen2.5-3B-Instruct" "1.5" "5"
run_for_hours "Qwen/Qwen2.5-3B-Instruct" "2.0" "1"
run_for_hours "Qwen/Qwen2.5-3B-Instruct" "2.0" "5"
run_for_hours "Qwen/Qwen2.5-3B-Instruct" "2.5" "1"
run_for_hours "Qwen/Qwen2.5-3B-Instruct" "2.5" "5"
run_for_hours "Qwen/Qwen2.5-3B-Instruct" "3.0" "1"
run_for_hours "Qwen/Qwen2.5-3B-Instruct" "3.0" "5"

echo "[All Done] Results under: ${OUT_BASE}"



