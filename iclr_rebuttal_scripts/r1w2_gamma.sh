#!/usr/bin/env bash

#SBATCH -J r1w2_gamma
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

MODELS_DEFAULT=("LichengLiu03/Qwen2.5-3B-UFO" "LichengLiu03/Qwen2.5-3B-UFO-1turn")
MODELS=("${MODELS[@]:-${MODELS_DEFAULT[@]}}")
GAMMAS_DEFAULT=("1.0" "0.75" "0.5" "0.25" "0")
GAMMAS=("${GAMMAS[@]:-${GAMMAS_DEFAULT[@]}}")

ES_VAL_GROUPS="${ES_VAL_GROUPS:-128}"
ES_VAL_GROUP_SIZE="${ES_VAL_GROUP_SIZE:-1}"

MAX_TURN="${MAX_TURN:-5}"
MAX_ACTIONS_PER_TRAJ="${MAX_ACTIONS_PER_TRAJ:-4}"

OUT_BASE="${REPO_DIR}/result/eval/r1w2_gamma"
mkdir -p "${OUT_BASE}"
SUMMARY_CSV="${OUT_BASE}/summary.csv"

if [[ ! -f "${SUMMARY_CSV}" ]]; then
  echo "gamma,model,avg_reward,MetamathQA/success,MetamathQA/pass@${ES_VAL_GROUP_SIZE},MetamathQA/num_actions" > "${SUMMARY_CSV}"
fi

echo "[Gamma-Sens] Repo: ${REPO_DIR}"
echo "[Gamma-Sens] Output base: ${OUT_BASE}"
echo "[Gamma-Sens] Eval groups: ${ES_VAL_GROUPS}, group_size: ${ES_VAL_GROUP_SIZE}"
echo "[Gamma-Sens] MAX_TURN=${MAX_TURN}, MAX_ACTIONS_PER_TRAJ=${MAX_ACTIONS_PER_TRAJ}"
echo "[Gamma-Sens] GAMMAS: ${GAMMAS[*]}"
echo "[Gamma-Sens] MODELS: ${MODELS[*]}"

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

for model in "${MODELS[@]}"; do
  model_safe="$(echo "${model}" | sed 's|/|-|g')"
  for gamma in "${GAMMAS[@]}"; do
    out_dir="${OUT_BASE}/${model_safe}/gamma_${gamma}"
    mkdir -p "${out_dir}"

    # -------------------
    # 1) Light training -> save HF weights
    # -------------------
    echo "[Gamma-Sens][Train] model=${model}, gamma=${gamma} (following base.yaml training settings)"
    ${PYTHON_BIN} train.py \
      model_path="${model}" \
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
    ${PYTHON_BIN} -m ragen.llm_agent.agent_proxy --config-name eval \
      actor_rollout_ref.model.path="${HF_DIR}" \
      es_manager.train.env_configs.tags=[MetamathQA] \
      es_manager.train.env_configs.n_groups=[1] \
      es_manager.val.env_configs.tags=[MetamathQA] \
      es_manager.val.env_configs.n_groups=[${ES_VAL_GROUPS}] \
      es_manager.val.env_groups=${ES_VAL_GROUPS} \
      es_manager.val.group_size=${ES_VAL_GROUP_SIZE} \
      custom_envs.MetamathQA.max_actions_per_traj=${MAX_ACTIONS_PER_TRAJ} \
      agent_proxy.max_turn=${MAX_TURN} \
      actor_rollout_ref.rollout.val_kwargs.do_sample=false \
      custom_envs.MetamathQA.env_config.gamma=${gamma} \
      output.dir="${out_dir}" \
      output.filename="rollouts.pkl" \
      output.append_timestamp=false \
      trainer.logger=[console]

    rollout_path="${out_dir}/rollouts.pkl"
    if [[ -f "${rollout_path}" ]]; then
      summary_json="$(summarize_run "${rollout_path}" "${gamma}" "${model}")"
      echo "[Gamma-Sens] Summary: ${summary_json}"
      avg_reward="$(echo "${summary_json}" | ${PYTHON_BIN} -c "import sys,json;print(json.load(sys.stdin)['avg_reward'])")"
      success="$(echo "${summary_json}" | ${PYTHON_BIN} -c "import sys,json;print(json.load(sys.stdin)['success'])")"
      pass_at_k="$(echo "${summary_json}" | ${PYTHON_BIN} -c "import sys,json;print(json.load(sys.stdin)['pass_at_k'])")"
      num_actions="$(echo "${summary_json}" | ${PYTHON_BIN} -c "import sys,json;print(json.load(sys.stdin)['num_actions'])")"
      echo "${gamma},${model},${avg_reward},${success},${pass_at_k},${num_actions}" >> "${SUMMARY_CSV}"
      echo "${summary_json}" > "${out_dir}/summary.json"
    else
      echo "[Gamma-Sens][WARN] Missing rollout at ${rollout_path}"
    fi
  done
done

echo "[All Done] Summary CSV: ${SUMMARY_CSV}"


