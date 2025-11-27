#!/usr/bin/env bash

set -euo pipefail

DEVICES="${DEVICES:-0,1}"
GPUS_PER_NODE=$(echo "${DEVICES}" | tr ',' '\n' | wc -l)
TP_SIZE=$(python3 -c "print(min(4, '${DEVICES}'.count(',') + 1))")
GPUS_PER_NODE=$((GPUS_PER_NODE))
HYDRA_CUDA_VISIBLE_DEVICES="'${DEVICES}'"
export CUDA_VISIBLE_DEVICES="${DEVICES}"

eval "$(conda shell.bash hook)"
conda activate ragen || true

PYTHON_BIN="${PYTHON_BIN:-python3}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

OUT_BASE="${OUT_BASE:-${REPO_DIR}}"
OUT_BASE="${OUT_BASE}/result/eval/r1w1"
mkdir -p "${OUT_BASE}"

echo "OUT_BASE: ${OUT_BASE}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

SUMMARY_CSV="${OUT_BASE}/summary.csv"
if [[ ! -f "${SUMMARY_CSV}" ]]; then
  header="model,avg_reward"
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

WANDB_PROJECT="${WANDB_PROJECT:-ufo_rebuttal}"

# Eval scale (can override via env)
ES_VAL_GROUPS="${ES_VAL_GROUPS:-1024}"
ES_VAL_GROUP_SIZE="${ES_VAL_GROUP_SIZE:-1}"

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
PASS_KEY="pass@${ES_VAL_GROUP_SIZE}"
read -r -a DATASET_NAMES <<< "MetamathQA TheoremQA GSM8k GPQA MMLU-STEM HotpotQA ConcurrentQA MMLU MMLUPro"
DATASET_NAMES_JSON='["MetamathQA","TheoremQA","GSM8k","GPQA","MMLU-STEM","HotpotQA","ConcurrentQA","MMLU","MMLUPro"]'

echo "[Eval] Repo: ${REPO_DIR}"
echo "[Eval] Output base: ${OUT_BASE}"
echo "[Eval] Eval groups: ${ES_VAL_GROUPS}, group_size: ${ES_VAL_GROUP_SIZE}, WANDB_PROJECT=${WANDB_PROJECT}"

run_eval_for_model() {
  local model_path="$1"
  local model_safe
  model_safe="$(echo "${model_path}" | sed 's|/|-|g')"
  local out_dir="${OUT_BASE}/${model_safe}"
  local rollout_path="${out_dir}/rollouts.pkl"
  mkdir -p "${out_dir}"

  echo "[Eval] Running eval for model: ${model_path}"
  echo "[Eval] Output dir: ${out_dir}"

  # Use the eval entry that does NOT train; override env to MetaMathQA
  # Force single turn for MetaMathQA
  local ts run_name
  ts="$(date +%Y%m%d_%H%M%S)"
  run_name="r1w1_${model_safe}_g${ES_VAL_GROUPS}_k${ES_VAL_GROUP_SIZE}_${ts}"
  LAST_RUN_NAME="${run_name}"
  WANDB_PROJECT="${WANDB_PROJECT}" WANDB_NAME="${run_name}" WANDB_RUN_ID="${run_name}" \
  ENABLE_GPT_FEEDBACK=1 \
  ${PYTHON_BIN} -m ragen.llm_agent.agent_proxy --config-name eval \
    trainer.experiment_name="${run_name}" \
    trainer.project_name="${WANDB_PROJECT}" \
    actor_rollout_ref.model.path="${model_path}" \
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
    agent_proxy.max_turn=5 \
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
  local model_safe
  model_safe="$(echo "${model_path}" | sed 's|/|-|g')"
  local out_dir="${OUT_BASE}/${model_safe}"
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
    "avg_reward": avg_reward,
    "metrics": flat_metrics,
    "datasets": dataset_metrics
}, ensure_ascii=False))
PY
)"
  echo "[Summary] ${summary_json}"
  echo "${summary_json}" > "${summary_path}"

  local row
  row="$(SUMMARY_JSON="${summary_json}" DATASET_NAMES_JSON="${DATASET_NAMES_JSON}" PASS_KEY="${PASS_KEY}" MODEL_NAME="${model_path}" ${PYTHON_BIN} - <<'PY'
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
    fmt(summary.get("avg_reward", 0.0)),
]

for ds in datasets:
    row.append(fmt(pick(ds, "success")))
    row.append(fmt(pick(ds, pass_key)))
    row.append(fmt(pick(ds, "num_actions")))

print(",".join(row))
PY
)"
  local key="${model_path},${LAST_RUN_NAME:-${model_safe}}"
  safe_append_csv "${row}" "${key}"
}

cleanup_rollout_for_model() {
  local model_path="$1"
  local model_safe
  model_safe="$(echo "${model_path}" | sed 's|/|-|g')"
  local out_dir="${OUT_BASE}/${model_safe}"
  local rollout_path="${out_dir}/rollouts.pkl"

  if [[ -f "${rollout_path}" ]]; then
    rm -f "${rollout_path}"
    echo "[Cleanup] Removed rollout: ${rollout_path}"
  fi
}

generate_fuzzy_feedback() {
  local model_path="$1"
  local model_safe
  model_safe="$(echo "${model_path}" | sed 's|/|-|g')"
  local out_dir="${OUT_BASE}/${model_safe}"
  local rollout_path="${out_dir}/rollouts.pkl"
  local feedback_path="${out_dir}/feedback_gpt4o.jsonl"

  if [[ ! -f "${rollout_path}" ]]; then
    echo "[Feedback] Rollout not found: ${rollout_path}"
    return 1
  fi
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "[Feedback] OPENAI_API_KEY not set. Skipping feedback for ${model_path}."
    return 0
  fi

  if [[ -f "${feedback_path}" ]]; then
    echo "[Feedback] Skip existing: ${feedback_path}"
    return 0
  fi

  echo "[Feedback] Generating fuzzy feedback with gpt-4o for: ${model_path}"
  echo "[Feedback] Output: ${feedback_path}"

  ${PYTHON_BIN} - <<'PY'
import os, re, json
from pathlib import Path
from openai import OpenAI
from verl import DataProto

repo_dir = Path(os.environ.get("REPO_DIR", ".")).resolve()
out_base = Path(os.environ.get("OUT_BASE", str(repo_dir / "result" / "eval" / "rebuttal"))).resolve()
model_path = os.environ["MODEL_PATH"]
model_safe = model_path.replace("/", "-")
run_dir = out_base / model_safe
rollout_path = run_dir / "rollouts.pkl"
feedback_path = run_dir / "feedback_gpt4o.jsonl"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
data = DataProto.load_from_disk(str(rollout_path))

def extract_question(messages: list[str|dict]) -> str:
    # messages[1]["content"] contains formatted turns; grab the first Turn's State
    content = ""
    for i, m in enumerate(messages):
        if i == 1 and isinstance(m, dict) and m.get("role") == "user":
            content = m.get("content", "")
            break
    if not content:
        return ""
    m = re.search(r"Turn 1:\s*State:\s*(.*?)(?:\nYou have|\Z)", content, flags=re.S)
    return m.group(1).strip() if m else content.strip()

def extract_student_answer(messages: list[str|dict]) -> str:
    # last assistant message
    assistants = [m for m in messages if isinstance(m, dict) and m.get("role") == "assistant"]
    if not assistants:
        return ""
    text = assistants[-1].get("content", "") or ""
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.S | re.I)
    return (m.group(1).strip() if m else text.strip())

with open(feedback_path, "w", encoding="utf-8") as fw:
    for idx in range(len(data)):
        item = data[idx]
        messages = item.non_tensor_batch.get("messages_list", [])
        # messages could be numpy array; ensure list of dicts
        if hasattr(messages, "tolist"):
            messages = messages.tolist()
        question = extract_question(messages)
        student_answer = extract_student_answer(messages)
        if not question and not student_answer:
            continue

        sys_prompt = "You are a strict but friendly math TA. Imitate a real human and provide ambiguous or even slightly inconsistent feedback that nudges directionally, without revealing the final answer or any specific numbers. Use 2-3 sentences."
        user_prompt = f"Problem:\n{question}\n\nStudent answer:\n{student_answer}\n\nPlease provide ambiguous, non-revealing feedback (no final answer, no specific numbers)."
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.5,
            max_tokens=180,
        )
        feedback = resp.choices[0].message.content
        rec = {
            "index": idx,
            "question": question,
            "student_answer": student_answer,
            "fuzzy_feedback": feedback,
            "model": model_path,
        }
        fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"[Feedback] Done: {feedback_path}")
PY
}

run_model_pipeline() {
  local model_path="$1"
  local model_safe
  model_safe="$(echo "${model_path}" | sed 's|/|-|g')"
  local summary_path="${OUT_BASE}/${model_safe}/summary.json"

  if [[ -f "${summary_path}" ]]; then
    echo "[Eval] Summary exists, skip pipeline: ${summary_path}"
    return 0
  fi

  echo "[Eval] ===== Model: ${model_path} ====="
  run_eval_for_model "${model_path}"
  summarize_metrics_for_model "${model_path}"
  # MODEL_PATH="${model_path}" generate_fuzzy_feedback "${model_path}"
  # cleanup_rollout_for_model "${model_path}"
  echo "[Eval] ===== Done: ${model_path} ====="
  echo
}

# Export REPO_DIR for the Python snippet
export REPO_DIR="${REPO_DIR}"
export OUT_BASE="${OUT_BASE}"

run_model_pipeline "LichengLiu03/Qwen2.5-3B-UFO"
run_model_pipeline "LichengLiu03/Qwen2.5-3B-UFO-1turn"

echo "[All Done] Results under: ${OUT_BASE}"
