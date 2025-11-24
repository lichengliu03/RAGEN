#!/usr/bin/env bash

set -euo pipefail

DEVICES="${DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES="${DEVICES}"
HYDRA_VISIBLE_DEVICES="'${DEVICES}'"
GPUS_PER_NODE=$(echo "${DEVICES}" | tr ',' '\n' | wc -l)
TP_SIZE=$(python3 -c "print(min(4, '${DEVICES}'.count(',') + 1))")

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

# ====== 配置 ======
TRAIN_NQ=1
# Eval scale
ES_VAL_GROUPS="${ES_VAL_GROUPS:-1024}"
ES_VAL_GROUP_SIZE="${ES_VAL_GROUP_SIZE:-1}"
PASS_KEY="pass@${ES_VAL_GROUP_SIZE}"

read -r -a DATASET_NAMES <<< "MetamathQA TheoremQA GSM8k GPQA MMLU-STEM HotpotQA ConcurrentQA MMLU MMLUPro"
DATASET_NAMES_JSON='["MetamathQA","TheoremQA","GSM8k","GPQA","MMLU-STEM","HotpotQA","ConcurrentQA","MMLU","MMLUPro"]'

VAL_TAGS="[MetamathQA]"
repeat_value_list() {
  local value="$1" repeat="$2"
  local output="[" index
  for ((index=0; index<repeat; index++)); do
    if (( index > 0 )); then output+=","; fi
    output+="${value}"
  done
  output+="]"
  printf "%s" "${output}"
}
VAL_NGROUPS_LIST="$(repeat_value_list "${ES_VAL_GROUPS}" 1)"
TOTAL_VAL_GROUPS="${ES_VAL_GROUPS}"

MAX_ACTIONS_PER_TRAJ="${MAX_ACTIONS_PER_TRAJ:-5}"
EVAL_TURN="${EVAL_TURN:-5}"

OUT_BASE="${OUT_BASE:-${REPO_DIR}}"
OUT_BASE="${OUT_BASE}/result/eval/r1q2"
echo "OUT_BASE: ${OUT_BASE}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
mkdir -p "${OUT_BASE}"

SUMMARY_CSV="${OUT_BASE}/summary.csv"
if [[ ! -f "${SUMMARY_CSV}" ]]; then
  header="model,turn,hours,total_correct,total_attempted,accuracy"
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
    echo "[R1Q2] Skip duplicate CSV row for key: ${key}"
  fi
}

WANDB_PROJECT="${WANDB_PROJECT:-ufo_rebuttal}"

echo "[R1Q2] Repo: ${REPO_DIR}"
echo "[R1Q2] Output base: ${OUT_BASE}"
echo "[R1Q2] ES_VAL_GROUPS=${ES_VAL_GROUPS}, WANDB_PROJECT=${WANDB_PROJECT}"

summarize_accuracy() {
  local rollout_path="$1"
  ${PYTHON_BIN} - <<PY
import json
from pathlib import Path
from verl import DataProto

dsets = json.loads('${DATASET_NAMES_JSON}')
pass_key = "pass@${ES_VAL_GROUP_SIZE}"

rollout_path = Path("${rollout_path}")
if not rollout_path.exists():
    print(json.dumps({"error": "Rollout not found"}))
    exit(0)

dp = DataProto.load_from_disk(str(rollout_path))
metrics = dp.meta_info.get("metrics", {}) if hasattr(dp.meta_info, "get") else {}

def to_builtin(val):
    if isinstance(val, (int, float, str, bool)) or val is None: return val
    if isinstance(val, dict): return {k: to_builtin(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)): return [to_builtin(v) for v in val]
    if hasattr(val, "item"):
        try: return val.item()
        except: pass
    return str(val)

metrics = {k: to_builtin(v) for k, v in metrics.items()}

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

total_correct = 0
total_attempted = 0
for idx in range(len(dp)):
    item = dp[idx]
    info = item.non_tensor_batch.get("info", {}) or {}
    if isinstance(info, list): info = info[-1] if info else {}
    if not isinstance(info, dict): continue
    total_correct += int(info.get("episode_num_correct", 0) or 0)
    total_attempted += int(info.get("episode_num_attempted", 0) or 0)

acc = (total_correct / total_attempted) if total_attempted > 0 else 0.0
print(json.dumps({
    "total_correct": total_correct,
    "total_attempted": total_attempted,
    "accuracy": acc,
    "metrics": flat_metrics,
    "datasets": dataset_metrics
}, ensure_ascii=False))
PY
}

# 运行连续训练并定期 Eval
run_continuous_and_eval() {
  local model="$1"
  local turn="$2"
  local max_hours="3.1"
  
  local model_safe
  model_safe="$(echo "${model}" | sed 's|/|-|g')"
  
  local ts run_name minutes timeout_spec
  ts="$(date +%Y%m%d_%H%M%S)"
  run_name="r1q2_cont_${model_safe}_turn${turn}_NQ${TRAIN_NQ}_${ts}"
  
  minutes="$(${PYTHON_BIN} -c "print(int(round(${max_hours}*60) + 10))")"
  timeout_spec="${minutes}m"

  echo "======================================================================="
  echo "[R1Q2][Train] Starting 3h continuous run"
  echo "Model: ${model}, Turn: ${turn}, Name: ${run_name}"
  echo "======================================================================="

  # 记录开始时间 (用于后续找对应时间的 checkpoint)
  local start_time
  start_time=$(date +%s)

  # 启动训练
  # RAGEN_SAVE_INTERVAL_HOURS=0.5: 每0.5小时保存一次
  # trainer.save_freq=100000: 禁用基于 step 的保存 (除非跑了10w步)
  # 训练时长限制为 max_hours
  RAGEN_SAVE_INTERVAL_HOURS=0.5 \
  WANDB_PROJECT="${WANDB_PROJECT}" WANDB_NAME="${run_name}" WANDB_RUN_ID="${run_name}" \
  timeout "${timeout_spec}" ${PYTHON_BIN} train.py \
    trainer.experiment_name="${run_name}" \
    trainer.project_name="${WANDB_PROJECT}" \
    model_path="${model}" \
    system.CUDA_VISIBLE_DEVICES="${HYDRA_VISIBLE_DEVICES}" \
    trainer.n_gpus_per_node=${GPUS_PER_NODE} \
    trainer.save_freq=100000 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE} \
    agent_proxy.max_turn=${turn} \
    es_manager.train.env_configs.tags=[MetamathQA] \
    es_manager.val.env_configs.tags=[MetamathQA] \
    ++custom_envs.MetamathQA.env_config.multi_question_mode=false \
    ++custom_envs.MetamathQA.env_config.n_questions_per_episode=1 \
    custom_envs.MetamathQA.max_actions_per_traj=${MAX_ACTIONS_PER_TRAJ} \
    actor_rollout_ref.actor.checkpoint.save_contents=[hf_model] || true

  echo "[R1Q2][Train] Finished (or timed out). Scanning checkpoints..."

  # 使用 Python 脚本查找对应时间点的 checkpoints
  # Checkpoints path structure: checkpoints/WANDB_PROJECT/RUN_NAME/global_step_X/actor/huggingface
  local checkpoints_json
  checkpoints_json=$(${PYTHON_BIN} - <<PY
import os
import sys
import json
import glob
from pathlib import Path

repo_dir = Path("${REPO_DIR}")
project_name = "${WANDB_PROJECT}"
run_name = "${run_name}"
start_time = ${start_time}
targets = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0] # Hours


base_dir = repo_dir / "checkpoints" / project_name / run_name
if not base_dir.exists():
    base_dir = repo_dir / "checkpoints" / run_name

candidates = []
if base_dir.exists():
    for step_dir in base_dir.glob("global_step_*"):
        hf_dir = step_dir / "actor" / "huggingface"
        if hf_dir.exists():
            mtime = hf_dir.stat().st_mtime
            candidates.append({"time": mtime, "path": str(hf_dir), "step": step_dir.name})

candidates.sort(key=lambda x: x["time"])

results = {}
for t_hour in targets:
    t_sec = start_time + t_hour * 3600
    
    best_cand = None
    min_diff = float('inf')
    
    for cand in candidates:
        diff = abs(cand["time"] - t_sec)
        if diff < min_diff:
            min_diff = diff
            best_cand = cand

    if best_cand and min_diff < 1800:
        key = f"{t_hour}" 
        results[key] = best_cand["path"]

print(json.dumps(results))
PY
)

  echo "[R1Q2] Found checkpoints for eval: ${checkpoints_json}"

  for h_key in 0.5 1.0 1.5 2.0 2.5 3.0; do
    local ckpt_path
    ckpt_path="$(echo "${checkpoints_json}" | ${PYTHON_BIN} -c "import sys, json; print(json.load(sys.stdin).get('${h_key}', ''))")"
    
    if [[ -z "${ckpt_path}" ]]; then
      echo "[R1Q2][Skip] No checkpoint found for T=${h_key}h"
      continue
    fi

    local h_safe="$(echo "${h_key}" | sed 's|\.|p|g')"
    local out_dir="${OUT_BASE}/${model_safe}/T${h_safe}h_turn_${turn}"
    mkdir -p "${out_dir}"
    local rollout_path="${out_dir}/rollouts.pkl"

    if [[ -f "${rollout_path}" ]]; then
      echo "[R1Q2][Eval] Skip existing: ${rollout_path}"
      continue
    fi

    local eval_name="${run_name}_eval_T${h_safe}h"
    echo "[R1Q2][Eval] T=${h_key}h, Ckpt=${ckpt_path}"
    
    WANDB_PROJECT="${WANDB_PROJECT}" WANDB_NAME="${eval_name}" WANDB_RUN_ID="${eval_name}" \
    ${PYTHON_BIN} -m ragen.llm_agent.agent_proxy --config-name eval \
      trainer.experiment_name="${eval_name}" \
      trainer.project_name="${WANDB_PROJECT}" \
      actor_rollout_ref.model.path="${ckpt_path}" \
      system.CUDA_VISIBLE_DEVICES="${HYDRA_VISIBLE_DEVICES}" \
      trainer.n_gpus_per_node=${GPUS_PER_NODE} \
      actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE} \
      es_manager.train.env_configs.tags=[MetamathQA] \
      es_manager.train.env_groups=1 \
      es_manager.train.env_configs.n_groups=[1] \
      es_manager.val.env_configs.tags=${VAL_TAGS} \
      es_manager.val.env_configs.n_groups=${VAL_NGROUPS_LIST} \
      es_manager.val.env_groups=${TOTAL_VAL_GROUPS} \
      es_manager.val.group_size=${ES_VAL_GROUP_SIZE} \
      ++custom_envs.MetamathQA.env_config.multi_question_mode=false \
      ++custom_envs.MetamathQA.env_config.n_questions_per_episode=1 \
      custom_envs.MetamathQA.max_actions_per_traj=${MAX_ACTIONS_PER_TRAJ} \
      agent_proxy.max_turn=${EVAL_TURN} \
      actor_rollout_ref.rollout.val_kwargs.do_sample=false \
      output.dir="${out_dir}" \
      output.filename="rollouts.pkl" \
      output.append_timestamp=false \
      trainer.logger=[console]

    if [[ -f "${rollout_path}" ]]; then
      summary_json="$(summarize_accuracy "${rollout_path}")"
      echo "[R1Q2] Summary (T=${h_key}h, turn=${turn}): ${summary_json}"
      echo "${summary_json}" > "${out_dir}/summary.json"
      row="$(SUMMARY_JSON="${summary_json}" DATASET_NAMES_JSON="${DATASET_NAMES_JSON}" PASS_KEY="${PASS_KEY}" MODEL_NAME="${model}" TURN_VAL="${turn}" HOURS_VAL="${h_key}" ${PYTHON_BIN} - <<'PY'
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
    os.environ["HOURS_VAL"],
    fmt(summary.get("total_correct", 0.0)),
    fmt(summary.get("total_attempted", 0.0)),
    fmt(summary.get("accuracy", 0.0)),
]

for ds in datasets:
    row.append(fmt(pick(ds, "success")))
    row.append(fmt(pick(ds, pass_key)))
    row.append(fmt(pick(ds, "num_actions")))

print(",".join(row))
PY
)"
      key="${model},${turn},${h_key},${run_name}"
      safe_append_csv "${row}" "${key}"
    fi
  done
}

run_continuous_and_eval "Qwen/Qwen2.5-3B-Instruct" "1"
run_continuous_and_eval "Qwen/Qwen2.5-3B-Instruct" "5"

echo "[All Done] Results under: ${OUT_BASE}"
