#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

env_file="${PYRAVID_ENV_FILE:-${repo_root}/.env}"
if [[ ! -f "$env_file" ]]; then
    echo "Missing environment file: $env_file" >&2
    exit 1
fi

set -a
# shellcheck disable=SC1090
source "$env_file"
set +a

export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

dataset="${PYRAVID_DATASET:-videomme}"
question_dir="./data/${dataset}/questions"
graph_dir="./memory/graphs"
embedding_dir="./memory/embeddings"
vector_store_dir="./artifacts/character_processing/${dataset}"
save_dir="./artifacts/outputs/open_question_answering"
log_dir="./artifacts/outputs/logs/${dataset}"

# The open-question task reads the facts directory from this environment variable.
export PYRAVID_FACTS_DIR="./data/${dataset}/test_facts"

mkdir -p "$log_dir"
LOG_FILE="${log_dir}/run_$(date +%Y%m%d_%H%M%S)_open_question_answering_agentic_expand_without_reasoning_32B_32B_top20.txt"

python prototype/tasks/open_question_answering_agentic_expand.py \
    --dataset "$dataset" \
    --question_dir "$question_dir" \
    --super_graph_dir "$graph_dir" \
    --super_embedding_dir "$embedding_dir" \
    --vector_store_dir "$vector_store_dir" \
    --answer_model Qwen/Qwen3-VL-32B-Instruct \
    --selection_model Qwen/Qwen3-32B \
    --two_level_mode \
    --top_k 20 \
    --with_prune \
    --max_tokens 10000 \
    --multimodal \
    --save_evidence \
    --with_top_summary \
    --data web \
    --save_dir "$save_dir" \
    --save_name video_open_questions_output.json \
    --latency_file_name videomme_open_question_latencies.csv \
    2>&1 | tee "$LOG_FILE"
    
echo "Log saved to ${LOG_FILE}"
