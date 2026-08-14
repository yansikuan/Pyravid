#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
env_file="${PYRAVID_ENV_FILE:-${repo_root}/.env}"
if [[ -f "$env_file" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$env_file"
  set +a
fi

dataset="${1:-${PYRAVID_DATASET:-videomme}}"
facts_dir="${PYRAVID_FACTS_DIR:-${repo_root}/data/${dataset}/facts}"
memory_dir="${PYRAVID_MEMORY_DIR:-${repo_root}/memory}"
graph_dir="${PYRAVID_GRAPH_DIR:-${memory_dir}/graphs}"
embedding_dir="${PYRAVID_MEMORY_EMBEDDING_DIR:-${memory_dir}/embeddings}"
vector_store_dir="${PYRAVID_CHARACTER_PROFILE_DIR:-${repo_root}/artifacts/vectorstore/${dataset}}"

cd "$repo_root"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -d "$facts_dir" ]]; then
  echo "Facts directory not found: $facts_dir" >&2
  exit 1
fi
if ! find "$facts_dir" -maxdepth 1 -type f -name '*.json' -print -quit | grep -q .; then
  echo "No fact JSON files found in: $facts_dir" >&2
  exit 1
fi
if [[ ! -d "${repo_root}/processed_data/${dataset}/fact_metadata" ]] || \
   [[ ! -d "${repo_root}/processed_data/${dataset}/fact_embeddings" ]]; then
  echo "Preprocessed memory not found for dataset '$dataset'." >&2
  echo "Run preprocess.sh first." >&2
  exit 1
fi

echo "Constructing PyraVid hierarchical memory graph"
echo "  dataset:    $dataset"
echo "  facts:      $facts_dir"
echo "  memory:     $memory_dir"
echo "  graphs:     $graph_dir"
echo "  embeddings: $embedding_dir"
echo "  vector DB:  $vector_store_dir"

exec python -m prototype.constructivist_memory \
  --dataset "$dataset" \
  --online_mode \
  --llm_link \
  --two_level_mode \
  --model "${PYRAVID_MEMORY_MODEL:-gemini-2.5-flash}" \
  --embedding_model "${PYRAVID_EMBEDDING_MODEL:-text-embedding-3-large}" \
  --link_model "${PYRAVID_LINK_MODEL:-Qwen/Qwen3-4B-Instruct-2507}" \
  --api_key_path "${PYRAVID_OPENAI_KEY_PATH:-${repo_root}/config/openai_key.txt}" \
  --link_api_key_path "${PYRAVID_KEY_PATH:-${repo_root}/config/gemini_key.txt}" \
  --facts_dir "$facts_dir" \
  --super_graph_dir "$graph_dir" \
  --super_embedding_dir "$embedding_dir" \
  --vector_store_dir "$vector_store_dir" \
  --num_processes "${PYRAVID_GRAPH_WORKERS:-5}" \
  --k "${PYRAVID_TOP_K:-20}" \
  --qwen_server
