#!/usr/bin/env bash

#SBATCH -J r1q1
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
conda activate ragen || true

MODELS_DEFAULT=("LichengLiu03/Qwen2.5-3B-UFO")
MODELS=("${MODELS[@]:-${MODELS_DEFAULT[@]}}")

# Number of questions per episode (effective when multi_question_mode is true)
NQ="${NQ:-5}"

# Eval scale
ES_VAL_GROUPS="${ES_VAL_GROUPS:-128}"
ES_VAL_GROUP_SIZE="${ES_VAL_GROUP_SIZE:-1}"

MAX_TURN="${MAX_TURN:-5}"
MAX_ACTIONS_PER_TRAJ="${MAX_ACTIONS_PER_TRAJ:-4}"

OUT_BASE="${REPO_DIR}/result/eval/r1q1"
mkdir -p "${OUT_BASE}"

echo "[R1Q1] Repo: ${REPO_DIR}"
echo "[R1Q1] Output base: ${OUT_BASE}"
echo "[R1Q1] MODELS: ${MODELS[*]}"
echo "[R1Q1] NQ=${NQ}, ES_VAL_GROUPS=${ES_VAL_GROUPS}, ES_VAL_GROUP_SIZE=${ES_VAL_GROUP_SIZE}"

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

for model in "${MODELS[@]}"; do
  model_safe="$(echo "${model}" | sed 's|/|-|g')"
  out_dir="${OUT_BASE}/${model_safe}"
  mkdir -p "${out_dir}"

  echo "[R1Q1][Train] model=${model}, NQ=${NQ}"
  ${PYTHON_BIN} train.py \
    model_path="${model}" \
    es_manager.train.env_configs.tags=[MetamathQA] \
    es_manager.val.env_configs.tags=[MetamathQA] \
    custom_envs.MetamathQA.env_config.multi_question_mode=true \
    custom_envs.MetamathQA.env_config.n_questions_per_episode=${NQ} \
    custom_envs.MetamathQA.max_actions_per_traj=${MAX_ACTIONS_PER_TRAJ} \
    actor_rollout_ref.actor.checkpoint.save_contents=[hf_model]

  LATEST_TRACKER="$(ls -t checkpoints/*/*/latest_checkpointed_iteration.txt 2>/dev/null | head -n1 || true)"
  HF_DIR=""
  if [[ -n "${LATEST_TRACKER}" && -f "${LATEST_TRACKER}" ]]; then
    CKPT_DIR="$(dirname "${LATEST_TRACKER}")"
    last_step="$(tr -d '\n' < "${LATEST_TRACKER}")"
    HF_DIR="${CKPT_DIR}/global_step_${last_step}/actor/huggingface"
  fi
  if [[ -z "${HF_DIR}" || ! -d "${HF_DIR}" ]]; then
    echo "[R1Q1][WARN] HF weights not found, falling back to original model: ${model}"
    HF_DIR="${model}"
  fi

  echo "[R1Q1][Eval] model=${HF_DIR}, NQ=${NQ}"
  ${PYTHON_BIN} -m ragen.llm_agent.agent_proxy --config-name eval \
    actor_rollout_ref.model.path="${HF_DIR}" \
    es_manager.train.env_configs.tags=[MetamathQA] \
    es_manager.train.env_configs.n_groups=[1] \
    es_manager.val.env_configs.tags=[MetamathQA] \
    es_manager.val.env_configs.n_groups=[${ES_VAL_GROUPS}] \
    es_manager.val.env_groups=${ES_VAL_GROUPS} \
    es_manager.val.group_size=${ES_VAL_GROUP_SIZE} \
    custom_envs.MetamathQA.env_config.multi_question_mode=true \
    custom_envs.MetamathQA.env_config.n_questions_per_episode=${NQ} \
    custom_envs.MetamathQA.max_actions_per_traj=${MAX_ACTIONS_PER_TRAJ} \
    agent_proxy.max_turn=${MAX_TURN} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=false \
    output.dir="${out_dir}" \
    output.filename="rollouts.pkl" \
    output.append_timestamp=false \
    trainer.logger=[console]

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


