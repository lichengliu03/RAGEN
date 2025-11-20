#!/usr/bin/env bash

set -euo pipefail

DEVICES="${DEVICES:-1}"
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

OUT_BASE="${REPO_DIR}/result/eval/r2w3"
mkdir -p "${OUT_BASE}"

WANDB_PROJECT="${WANDB_PROJECT:-ufo_rebuttal}"

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
SUMMARY_DATASETS="MetamathQA,TheoremQA,GSM8k,GPQA,MMLU-STEM,HotpotQA,ConcurrentQA,MMLU,MMLUPro"
SUMMARY_KEYS="success,pass@1,num_actions"

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

  ROLLOUT_PATH="${rollout_path}" SUMMARY_PATH="${summary_path}" \
  SUMMARY_DATASETS="${SUMMARY_DATASETS}" SUMMARY_KEYS="${SUMMARY_KEYS}" \
  ${PYTHON_BIN} - <<'PY'
import json, os
from pathlib import Path
from verl import DataProto

rollout_path = Path(os.environ["ROLLOUT_PATH"])
summary_path = Path(os.environ["SUMMARY_PATH"])
datasets = [name.strip() for name in os.environ.get("SUMMARY_DATASETS", "").split(",") if name.strip()]
keys = [name.strip() for name in os.environ.get("SUMMARY_KEYS", "").split(",") if name.strip()]

metrics = DataProto.load_from_disk(str(rollout_path)).meta_info.get("metrics", {})
summary = {}
for dataset in datasets:
    entry = {}
    for key in keys:
        metric_key = f"{dataset}/{key}"
        value = metrics.get(metric_key)
        entry[key] = float(value) if value is not None else None
    summary[dataset] = entry

summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"[Summary] Wrote metrics to {summary_path}")
PY
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
