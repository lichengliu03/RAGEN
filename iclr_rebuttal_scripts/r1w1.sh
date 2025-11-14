#!/usr/bin/env bash

#SBATCH -J r1w1
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

# Eval-only on MetaMathQA for two models, then ask gpt-4o for fuzzy feedback
# Models:
#   - LichengLiu03/Qwen2.5-3B-UFO
#   - LichengLiu03/Qwen2.5-3B-UFO-1turn
#
# Requirements:
#   - Python env with repo dependencies installed (vllm, transformers, hydra, verl, datasets, openai)
#   - OPENAI_API_KEY set (for feedback stage)
#   - GPU recommended for vLLM
#
# Usage:
#   bash scripts/eval_metamathqa_ufo.sh
#   ES_VAL_GROUPS=128 ES_VAL_GROUP_SIZE=1 CUDA_VISIBLE_DEVICES=0 bash scripts/eval_metamathqa_ufo.sh
#
# Notes:
#   - Outputs:
#       result/eval/rebuttal/<model-sanitized>/rollouts.pkl
#       result/eval/rebuttal/<model-sanitized>/feedback_gpt4o.jsonl

PYTHON_BIN="${PYTHON_BIN:-python3}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

OUT_BASE="${REPO_DIR}/result/eval/r1w1"
mkdir -p "${OUT_BASE}"

# Eval scale (can override via env)
ES_VAL_GROUPS="${ES_VAL_GROUPS:-128}"
ES_VAL_GROUP_SIZE="${ES_VAL_GROUP_SIZE:-1}"

echo "[Eval] Repo: ${REPO_DIR}"
echo "[Eval] Output base: ${OUT_BASE}"
echo "[Eval] Eval groups: ${ES_VAL_GROUPS}, group_size: ${ES_VAL_GROUP_SIZE}"

run_eval_for_model() {
  local model_path="$1"
  local model_safe
  model_safe="$(echo "${model_path}" | sed 's|/|-|g')"
  local out_dir="${OUT_BASE}/${model_safe}"
  mkdir -p "${out_dir}"

  echo "[Eval] Running eval for model: ${model_path}"
  echo "[Eval] Output dir: ${out_dir}"

  # Use the eval entry that does NOT train; override env to MetaMathQA
  # Force single turn for MetaMathQA
  if [[ ! -f "${out_dir}/rollouts.pkl" ]]; then
    ${PYTHON_BIN} -m ragen.llm_agent.agent_proxy --config-name eval \
      actor_rollout_ref.model.path="${model_path}" \
      es_manager.train.env_configs.tags=[MetamathQA] \
      es_manager.val.env_configs.tags=[MetamathQA] \
      es_manager.val.env_groups=${ES_VAL_GROUPS} \
      es_manager.val.group_size=${ES_VAL_GROUP_SIZE} \
      agent_proxy.max_turn=1 \
      output.dir="${out_dir}" \
      output.filename="rollouts.pkl" \
      output.append_timestamp=false \
      trainer.logger=[console] \
      actor_rollout_ref.rollout.val_kwargs.do_sample=false
  else
    echo "[Eval] Skip existing rollouts: ${out_dir}/rollouts.pkl"
  fi

  echo "[Eval] Saved: ${out_dir}/rollouts.pkl"
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
out_base = repo_dir / "result" / "eval" / "rebuttal"
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

# Export REPO_DIR for the Python snippet
export REPO_DIR="${REPO_DIR}"

run_eval_for_model "LichengLiu03/Qwen2.5-3B-UFO"
MODEL_PATH="LichengLiu03/Qwen2.5-3B-UFO" generate_fuzzy_feedback "LichengLiu03/Qwen2.5-3B-UFO"

run_eval_for_model "LichengLiu03/Qwen2.5-3B-UFO-1turn"
MODEL_PATH="LichengLiu03/Qwen2.5-3B-UFO-1turn" generate_fuzzy_feedback "LichengLiu03/Qwen2.5-3B-UFO-1turn"

echo "[All Done] Results under: ${OUT_BASE}"


