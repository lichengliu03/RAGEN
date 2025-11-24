#!/usr/bin/env bash

set -euo pipefail

DEVICES="${DEVICES:-0,1}"
GPUS_PER_NODE=$(echo "${DEVICES}" | tr ',' '\n' | wc -l)
TP_SIZE=$(python3 -c "print(max(4, int("${DEVICES}".count(",")+1)))")
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

set +u
eval "$(conda shell.bash hook)"
conda activate ragen || true
set -u

ES_VAL_GROUPS="${ES_VAL_GROUPS:-1024}"
ES_VAL_GROUP_SIZE="${ES_VAL_GROUP_SIZE:-1}"
TRAIN_STEPS="${TRAIN_STEPS:-200}"

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
read -r -a DATASET_NAMES <<< "MetamathQA TheoremQA GSM8k GPQA MMLU-STEM HotpotQA ConcurrentQA MMLU MMLUPro"
DATASET_NAMES_JSON='["MetamathQA","TheoremQA","GSM8k","GPQA","MMLU-STEM","HotpotQA","ConcurrentQA","MMLU","MMLUPro"]'
PASS_KEY="pass@${ES_VAL_GROUP_SIZE}"

OUT_BASE="${OUT_BASE:-${REPO_DIR}}"
OUT_BASE="${OUT_BASE}/result/eval/r1w2_gamma"
echo "OUT_BASE: ${OUT_BASE}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

mkdir -p "${OUT_BASE}"
SUMMARY_CSV="${OUT_BASE}/summary.csv"

WANDB_PROJECT="${WANDB_PROJECT:-ufo_rebuttal}"

if [[ ! -f "${SUMMARY_CSV}" ]]; then
  header="gamma,model,avg_reward"
  for ds in "${DATASET_NAMES[@]}"; do
    header+=",${ds}/success,${ds}/${PASS_KEY},${ds}/num_actions"
  done
  echo "${header}" > "${SUMMARY_CSV}"
fi

echo "[Gamma-Sens] Repo: ${REPO_DIR}"
echo "[Gamma-Sens] Output base: ${OUT_BASE}"
echo "[Gamma-Sens] Eval groups: ${ES_VAL_GROUPS}, group_size: ${ES_VAL_GROUP_SIZE}, WANDB_PROJECT=${WANDB_PROJECT}"
echo "[Gamma-Sens] MAX_TURN=${MAX_TURN}, MAX_ACTIONS_PER_TRAJ=${MAX_ACTIONS_PER_TRAJ}, TRAIN_STEPS=${TRAIN_STEPS}"
echo "[Gamma-Sens] This script will run 2 experiments (1 model x 2 gammas)."

summarize_run() {
  local rollout_path="$1"
  local gamma="$2"
  local model="$3"
  ${PYTHON_BIN} - <<PY
import json
from pathlib import Path
from verl import DataProto

dsets = json.loads('${DATASET_NAMES_JSON}')
pass_key = "pass@${ES_VAL_GROUP_SIZE}"

rollout_path = Path("${rollout_path}")
dp = DataProto.load_from_disk(str(rollout_path))
metrics = dp.meta_info.get("metrics", {})
rm = dp.batch["rm_scores"]
avg_reward = float(rm.sum(-1).mean().item())

def to_builtin(val):
    if isinstance(val, (int, float, str, bool)) or val is None:
        return val
    if isinstance(val, dict):
        return {k: to_builtin(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [to_builtin(v) for v in val]
    if hasattr(val, "item"):
        try:
            return to_builtin(val.item())
        except Exception:
            return val
    return val

metrics = to_builtin(metrics)

def extract_metric(src, dataset, metric_name):
    if isinstance(src, dict):
        combined = f"{dataset}/{metric_name}"
        if combined in src:
            return src[combined]
        ds_entry = src.get(dataset)
        if isinstance(ds_entry, dict) and metric_name in ds_entry:
            return ds_entry[metric_name]
    return 0.0

def to_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0

flat_metrics = {}
dataset_metrics = {}
for dataset in dsets:
    success = to_float(extract_metric(metrics, dataset, "success"))
    pass_val = to_float(extract_metric(metrics, dataset, pass_key))
    num_actions = to_float(extract_metric(metrics, dataset, "num_actions"))
    flat_metrics[f"{dataset}/success"] = success
    flat_metrics[f"{dataset}/{pass_key}"] = pass_val
    flat_metrics[f"{dataset}/num_actions"] = num_actions
    dataset_metrics[dataset] = {
        "success": success,
        pass_key: pass_val,
        "num_actions": num_actions,
    }

metamath = dataset_metrics.get("MetamathQA", {})

print(json.dumps({
  "gamma": "${gamma}",
  "model": "${model}",
  "avg_reward": avg_reward,
  "success": metamath.get("success", 0.0),
  "pass_at_k": metamath.get(pass_key, 0.0),
  "num_actions": metamath.get("num_actions", 0.0),
  "metrics": flat_metrics,
  "datasets": dataset_metrics
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
  local train_steps="${TRAIN_STEPS}"
  model_safe="$(echo "${model}" | sed 's|/|-|g')"
  out_dir="${OUT_BASE}/${model_safe}/gamma_${gamma}"
  mkdir -p "${out_dir}"
  rollout_path="${out_dir}/rollouts.pkl"

  local ts run_name eval_name
  ts="$(date +%Y%m%d_%H%M%S)"
  run_name="r1w2_gamma_${model_safe}_g${gamma}_s${train_steps}_${ts}"
  eval_name="${run_name}_eval_g${ES_VAL_GROUPS}_k${ES_VAL_GROUP_SIZE}"

  if [[ -f "${rollout_path}" ]]; then
    echo "[Gamma-Sens] Rollout exists, skip train/eval: model=${model}, gamma=${gamma}"
  else
    # -------------------
    # 1) Light training -> save HF weights
    # -------------------
    echo "[Gamma-Sens][Train] model=${model}, gamma=${gamma}, steps=${train_steps} (following base.yaml training settings)"
    WANDB_PROJECT="${WANDB_PROJECT}" WANDB_NAME="${run_name}" WANDB_RUN_ID="${run_name}" \
    ${PYTHON_BIN} train.py \
      trainer.experiment_name="${run_name}" \
      trainer.project_name="${WANDB_PROJECT}" \
      model_path="${model}" \
      system.CUDA_VISIBLE_DEVICES=${HYDRA_CUDA_VISIBLE_DEVICES} \
      trainer.n_gpus_per_node=${GPUS_PER_NODE} \
      actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE} \
      es_manager.train.env_configs.tags=[MetamathQA] \
      es_manager.val.env_configs.tags=[MetamathQA] \
      custom_envs.MetamathQA.env_config.gamma=${gamma} \
      trainer.total_training_steps=${train_steps} \
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
      actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE} \
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
    row="$(SUMMARY_JSON="${summary_json}" DATASET_NAMES_JSON="${DATASET_NAMES_JSON}" PASS_KEY="${PASS_KEY}" GAMMA_VAL="${gamma}" MODEL_NAME="${model}" ${PYTHON_BIN} - <<'PY'
import json
import os

def fmt(val):
    try:
        return f"{float(val)}"
    except (TypeError, ValueError):
        return str(val)

summary = json.loads(os.environ["SUMMARY_JSON"])
datasets = json.loads(os.environ["DATASET_NAMES_JSON"])
pass_key = os.environ["PASS_KEY"]
metrics_flat = summary.get("metrics", {})
datasets_metrics = summary.get("datasets", {})

def pick(ds, metric_name):
    ds_metrics = datasets_metrics.get(ds, {})
    if metric_name in ds_metrics:
        return ds_metrics[metric_name]
    combined = f"{ds}/{metric_name}"
    return metrics_flat.get(combined, 0.0)

row = [
    os.environ["GAMMA_VAL"],
    os.environ["MODEL_NAME"],
    fmt(summary.get("avg_reward", 0.0)),
]

for ds in datasets:
    row.append(fmt(pick(ds, "success")))
    row.append(fmt(pick(ds, pass_key)))
    row.append(fmt(pick(ds, "num_actions")))

print(",".join(row))
PY
)"
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
