#!/usr/bin/env bash

set -euo pipefail

DEVICES="${DEVICES:-0,1}"
GPUS_PER_NODE=$(echo "${DEVICES}" | tr ',' '\n' | wc -l)
GPUS_PER_NODE=$((GPUS_PER_NODE))
HYDRA_CUDA_VISIBLE_DEVICES="'${DEVICES}'"
export CUDA_VISIBLE_DEVICES="${DEVICES}"

eval "$(conda shell.bash hook)"
set +u
conda activate ragen || true
set -u

PYTHON_BIN="${PYTHON_BIN:-python3}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

OUT_BASE="${OUT_BASE:-${REPO_DIR}}"
OUT_BASE="${OUT_BASE}/result/eval/r2w3"
mkdir -p "${OUT_BASE}"

WANDB_PROJECT="${WANDB_PROJECT:-ufo_rebuttal}"

SUMMARY_CSV="${OUT_BASE}/summary.csv"
if [[ ! -f "${SUMMARY_CSV}" ]]; then
  header="model,turn,avg_reward"
  for ds in "${DATASET_NAMES[@]}"; do
    header+=",${ds}/success,${ds}/${PASS_KEY},${ds}/num_actions"
  done
  echo "${header}" > "${SUMMARY_CSV}"
fi

safe_append_csv() {
  local row="$1" key="$2"
  if ! grep -Fq -- "${key}" "${SUMMARY_CSV}"; then
    echo "${row}" >> "${SUMMARY_CSV}"
  else
    echo "[Eval] Skip duplicate CSV row for key: ${key}"
  fi
}

# Eval scale (can override via env)
ES_VAL_GROUPS="${ES_VAL_GROUPS:-1024}"
ES_VAL_GROUP_SIZE="${ES_VAL_GROUP_SIZE:-1}"

# Make n_groups list for 9 tags, each having ES_VAL_GROUPS groups
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
PASS_KEY="pass@${ES_VAL_GROUP_SIZE}"
read -r -a DATASET_NAMES <<< "MetamathQA TheoremQA GSM8k GPQA MMLU-STEM HotpotQA ConcurrentQA MMLU MMLUPro"
DATASET_NAMES_JSON='["MetamathQA","TheoremQA","GSM8k","GPQA","MMLU-STEM","HotpotQA","ConcurrentQA","MMLU","MMLUPro"]'

echo "[Eval] Repo: ${REPO_DIR}"
echo "[Eval] Output base: ${OUT_BASE}"
echo "[Eval] Eval groups: ${ES_VAL_GROUPS}, group_size: ${ES_VAL_GROUP_SIZE}, WANDB_PROJECT=${WANDB_PROJECT}"

run_eval_for_model() {
  local model_path="$1"
  local turn="$2"
  local model_safe
  model_safe="$(echo "${model_path}" | sed 's|/|-|g')"
  # Append turn to output directory to separate results
  local out_dir="${OUT_BASE}/${model_safe}/turn_${turn}"
  local rollout_path="${out_dir}/rollouts.pkl"
  mkdir -p "${out_dir}"

  echo "[Eval] Running eval for model: ${model_path}, turn: ${turn}"
  echo "[Eval] Output dir: ${out_dir}"

  local ts run_name
  ts="$(date +%Y%m%d_%H%M%S)"
  run_name="r2w3_${model_safe}_turn${turn}_g${ES_VAL_GROUPS}_k${ES_VAL_GROUP_SIZE}_${ts}"
  LAST_RUN_NAME="${run_name}"
  
  WANDB_PROJECT="${WANDB_PROJECT}" WANDB_NAME="${run_name}" WANDB_RUN_ID="${run_name}" \
  ${PYTHON_BIN} -m ragen.llm_agent.agent_proxy --config-name eval \
    trainer.experiment_name="${run_name}" \
    trainer.project_name="${WANDB_PROJECT}" \
    actor_rollout_ref.model.path="${model_path}" \
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
    agent_proxy.max_turn=${turn} \
    output.dir="${out_dir}" \
    output.filename="rollouts.pkl" \
    output.append_timestamp=false \
    trainer.logger=[console] \
    actor_rollout_ref.rollout.val_kwargs.do_sample=false

  if [[ -f "${rollout_path}" ]]; then
    echo "[Eval] Rollout generated: ${rollout_path}"
  else
    echo "[Eval][ERROR] Missing rollout at ${rollout_path}"
    return 1
  fi
}

summarize_metrics_for_model() {
  local model_path="$1"
  local turn="$2"
  local model_safe
  model_safe="$(echo "${model_path}" | sed 's|/|-|g')"
  local out_dir="${OUT_BASE}/${model_safe}/turn_${turn}"
  local rollout_path="${out_dir}/rollouts.pkl"
  local summary_path="${out_dir}/summary.json"

  if [[ ! -f "${rollout_path}" ]]; then
    echo "[Summary] Rollout not found: ${rollout_path}"
    return 1
  fi

  local summary_json
  summary_json="$(${PYTHON_BIN} - <<PY
import json
from pathlib import Path
from verl import DataProto

datasets = json.loads('${DATASET_NAMES_JSON}')
pass_key = "pass@${ES_VAL_GROUP_SIZE}"

rollout_path = Path("${rollout_path}")
dp = DataProto.load_from_disk(str(rollout_path))
metrics = dp.meta_info.get("metrics", {})
rm = dp.batch.get("rm_scores")
avg_reward = 0.0
if rm is not None:
    try:
        avg_reward = float(rm.sum(-1).mean().item())
    except Exception:
        avg_reward = 0.0

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
for dataset in datasets:
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

print(json.dumps({
    "model": "${model_path}",
    "turn": "${turn}",
    "avg_reward": avg_reward,
    "metrics": flat_metrics,
    "datasets": dataset_metrics
}, ensure_ascii=False))
PY
)"
  echo "[Summary] ${summary_json}"
  echo "${summary_json}" > "${summary_path}"

  local row
  row="$(SUMMARY_JSON="${summary_json}" DATASET_NAMES_JSON="${DATASET_NAMES_JSON}" PASS_KEY="${PASS_KEY}" MODEL_NAME="${model_path}" TURN_VAL="${turn}" ${PYTHON_BIN} - <<'PY'
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
    os.environ["MODEL_NAME"],
    os.environ["TURN_VAL"],
    fmt(summary.get("avg_reward", 0.0)),
]

for ds in datasets:
    row.append(fmt(pick(ds, "success")))
    row.append(fmt(pick(ds, pass_key)))
    row.append(fmt(pick(ds, "num_actions")))

print(",".join(row))
PY
)"
  local key="${model_path},${turn},${LAST_RUN_NAME:-${model_safe}}"
  safe_append_csv "${row}" "${key}"
}

run_model_pipeline() {
  local model_path="$1"
  local turn="$2"
  local model_safe
  model_safe="$(echo "${model_path}" | sed 's|/|-|g')"
  local summary_path="${OUT_BASE}/${model_safe}/turn_${turn}/summary.json"

  if [[ -f "${summary_path}" ]]; then
    echo "[Eval] Summary exists, skip pipeline: ${summary_path}"
    return 0
  fi

  echo "[Eval] ===== Model: ${model_path}, Turn: ${turn} ====="
  run_eval_for_model "${model_path}" "${turn}"
  summarize_metrics_for_model "${model_path}" "${turn}"
  echo "[Eval] ===== Done: ${model_path}, Turn: ${turn} ====="
  echo
}

export REPO_DIR="${REPO_DIR}"
export OUT_BASE="${OUT_BASE}"

# Run 6 experiments for Qwen2.5-3B-Instruct base model with different max turns
run_model_pipeline "Qwen/Qwen2.5-3B-Instruct" "1"
run_model_pipeline "Qwen/Qwen2.5-3B-Instruct" "2"
run_model_pipeline "Qwen/Qwen2.5-3B-Instruct" "4"
run_model_pipeline "Qwen/Qwen2.5-3B-Instruct" "6"
run_model_pipeline "Qwen/Qwen2.5-3B-Instruct" "8"
run_model_pipeline "Qwen/Qwen2.5-3B-Instruct" "10"

echo "[All Done] Results under: ${OUT_BASE}"
