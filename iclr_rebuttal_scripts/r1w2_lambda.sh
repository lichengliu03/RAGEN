#!/usr/bin/env bash

#SBATCH -J r1w2_lambda
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

GAMMA="${GAMMA:-1.0}"

ES_VAL_GROUPS="${ES_VAL_GROUPS:-128}"
ES_VAL_GROUP_SIZE="${ES_VAL_GROUP_SIZE:-1}"

MAX_TURN="${MAX_TURN:-5}"
MAX_ACTIONS_PER_TRAJ="${MAX_ACTIONS_PER_TRAJ:-4}"

OUT_BASE="${REPO_DIR}/result/eval/r1w2_lambda"
mkdir -p "${OUT_BASE}"
SUMMARY_CSV="${OUT_BASE}/summary.csv"

if [[ ! -f "${SUMMARY_CSV}" ]]; then
  echo "lambda,model,gamma,avg_reward,MetamathQA/success,MetamathQA/pass@${ES_VAL_GROUP_SIZE},MetamathQA/num_actions,MetamathQA/rep_distinct_fraction,MetamathQA/rep_penalty" > "${SUMMARY_CSV}"
fi

echo "[Lambda-Sens] Repo: ${REPO_DIR}"
echo "[Lambda-Sens] Output base: ${OUT_BASE}"
echo "[Lambda-Sens] Eval groups: ${ES_VAL_GROUPS}, group_size: ${ES_VAL_GROUP_SIZE}"
echo "[Lambda-Sens] MAX_TURN=${MAX_TURN}, MAX_ACTIONS_PER_TRAJ=${MAX_ACTIONS_PER_TRAJ}"
echo "[Lambda-Sens] GAMMA=${GAMMA}"
echo "[Lambda-Sens] This script will run 10 experiments (2 models x 5 lambdas)."

summarize_run() {
  local rollout_path="$1"
  local lambda_val="$2"
  local gamma="$3"
  local model="$4"
  ${PYTHON_BIN} - <<PY
import json
from pathlib import Path
from verl import DataProto

rollout_path = Path("${rollout_path}")
dp = DataProto.load_from_disk(str(rollout_path))
metrics = dp.meta_info.get("metrics", {})
rm = dp.batch["rm_scores"]
avg_reward = float(rm.sum(-1).mean().item())

def getm(k, default=0.0):
    try:
        return float(metrics.get(k, default))
    except Exception:
        return float(default)

success = getm("MetamathQA/success")
num_actions = getm("MetamathQA/num_actions")
pass_at_k = getm(f"MetamathQA/pass@${ES_VAL_GROUP_SIZE}")
rep_df = getm("MetamathQA/rep_distinct_fraction")
rep_pen = getm("MetamathQA/rep_penalty")

print(json.dumps({
  "lambda": "${lambda_val}",
  "gamma": "${gamma}",
  "model": "${model}",
  "avg_reward": avg_reward,
  "success": success,
  "pass_at_k": pass_at_k,
  "num_actions": num_actions,
  "rep_distinct_fraction": rep_df,
  "rep_penalty": rep_pen
}, ensure_ascii=False))
PY
}

safe_append_csv() {
  # $1=row, $2=key (grep -F)
  local row="$1" key="$2"
  if ! grep -Fq -- "${key}" "${SUMMARY_CSV}"; then
    echo "${row}" >> "${SUMMARY_CSV}"
  else
    echo "[Lambda-Sens] Skip duplicate CSV row for key: ${key}"
  fi
}

run_lambda() {
  local model="$1"
  local lambda_val="$2"
  local model_safe out_dir rollout_path
  model_safe="$(echo "${model}" | sed 's|/|-|g')"
  out_dir="${OUT_BASE}/${model_safe}/lambda_${lambda_val}_gamma_${GAMMA}"
    mkdir -p "${out_dir}"
  rollout_path="${out_dir}/rollouts.pkl"

  if [[ -f "${rollout_path}" ]]; then
    echo "[Lambda-Sens] Rollout exists, skip train/eval: model=${model}, lambda=${lambda_val}, gamma=${GAMMA}"
  else
    # -------------------
    # 1) Light training -> save HF weights
    # -------------------
    echo "[Lambda-Sens][Train] model=${model}, lambda=${lambda_val}, gamma=${GAMMA} (following base.yaml training settings)"
    ${PYTHON_BIN} train.py \
      model_path="${model}" \
      es_manager.train.env_configs.tags=[MetamathQA] \
      es_manager.val.env_configs.tags=[MetamathQA] \
      custom_envs.MetamathQA.env_config.gamma=${GAMMA} \
      custom_envs.MetamathQA.env_config.lambda_rep=${lambda_val} \
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
      echo "[Lambda-Sens][WARN] HF weights not found, falling back to original model: ${model}"
      HF_DIR="${model}"
    fi

    # -------------------
    # 2) Evaluation
    # -------------------
    echo "[Lambda-Sens][Eval] model=${HF_DIR}, lambda=${lambda_val}, gamma=${GAMMA}"
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
      custom_envs.MetamathQA.env_config.gamma=${GAMMA} \
      custom_envs.MetamathQA.env_config.lambda_rep=${lambda_val} \
      output.dir="${out_dir}" \
      output.filename="rollouts.pkl" \
      output.append_timestamp=false \
      trainer.logger=[console]
  fi

  if [[ -f "${rollout_path}" ]]; then
    summary_json="$(summarize_run "${rollout_path}" "${lambda_val}" "${GAMMA}" "${model}")"
    echo "[Lambda-Sens] Summary: ${summary_json}"
    avg_reward="$(echo "${summary_json}" | ${PYTHON_BIN} -c "import sys,json;print(json.load(sys.stdin)['avg_reward'])")"
    success="$(echo "${summary_json}" | ${PYTHON_BIN} -c "import sys,json;print(json.load(sys.stdin)['success'])")"
    pass_at_k="$(echo "${summary_json}" | ${PYTHON_BIN} -c "import sys,json;print(json.load(sys.stdin)['pass_at_k'])")"
    num_actions="$(echo "${summary_json}" | ${PYTHON_BIN} -c "import sys,json;print(json.load(sys.stdin)['num_actions'])")"
    rep_df="$(echo "${summary_json}" | ${PYTHON_BIN} -c "import sys,json;print(json.load(sys.stdin)['rep_distinct_fraction'])")"
    rep_pen="$(echo "${summary_json}" | ${PYTHON_BIN} -c "import sys,json;print(json.load(sys.stdin)['rep_penalty'])")"
    row="${lambda_val},${model},${GAMMA},${avg_reward},${success},${pass_at_k},${num_actions},${rep_df},${rep_pen}"
    key="${lambda_val},${model},${GAMMA},"
    safe_append_csv "${row}" "${key}"
    echo "${summary_json}" > "${out_dir}/summary.json"
  else
    echo "[Lambda-Sens][WARN] Missing rollout at ${rollout_path}"
  fi
}

run_lambda "LichengLiu03/Qwen2.5-3B-UFO" "0.0"
run_lambda "LichengLiu03/Qwen2.5-3B-UFO" "0.5"
run_lambda "LichengLiu03/Qwen2.5-3B-UFO" "1.0"
run_lambda "LichengLiu03/Qwen2.5-3B-UFO" "2.0"
run_lambda "LichengLiu03/Qwen2.5-3B-UFO" "3.0"

run_lambda "LichengLiu03/Qwen2.5-3B-UFO-1turn" "0.0"
run_lambda "LichengLiu03/Qwen2.5-3B-UFO-1turn" "0.5"
run_lambda "LichengLiu03/Qwen2.5-3B-UFO-1turn" "1.0"
run_lambda "LichengLiu03/Qwen2.5-3B-UFO-1turn" "2.0"
run_lambda "LichengLiu03/Qwen2.5-3B-UFO-1turn" "3.0"

echo "[All Done] Summary CSV: ${SUMMARY_CSV}"


