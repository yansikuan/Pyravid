#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

env_file="${repo_root}/.env"
if [[ ! -f "$env_file" ]]; then
    echo "Missing environment file: $env_file" >&2
    exit 1
fi

set -a
source "$env_file"
set +a

export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

log_dir="${PYRAVID_LOG_DIR:-artifacts/outputs/logs/videomme}"
mkdir -p "$log_dir"
LOG_FILE="${log_dir}/run_$(date +%Y%m%d_%H%M%S)_question_answering_agentic_expand_32B_32B_top20.txt"

python prototype/tasks/question_answering_agentic_expand.py \
    --dataset videomme \
    --question_dir ./data/videomme/questions \
    --super_graph_dir ./memory/graphs \
    --super_embedding_dir ./memory/embeddings \
    --answer_model Qwen/Qwen3-VL-32B-Instruct \
    --selection_model Qwen/Qwen3-32B \
    --two_level_mode \
    --top_k 20 \
    --max_tokens 10000 \
    --multimodal \
    --save_evidence \
    --with_top_summary \
    --with_expand \
    --with_prune \
    --latency_file_name test_answer_latencies.csv \
    --save_dir ./artifacts/outputs/question_answering_test \
    --save_name test_questions_output.json \
    2>&1 | tee "$LOG_FILE"

echo "Log saved to ${LOG_FILE}"
