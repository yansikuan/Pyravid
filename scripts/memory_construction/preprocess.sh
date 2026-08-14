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
workers="${PYRAVID_PREPROCESS_WORKERS:-${PYRAVID_THREADS:-10}}"

cd "$repo_root"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -d "$facts_dir" ]]; then
  echo "Facts directory not found: $facts_dir" >&2
  echo "Run memory_extraction.sh first." >&2
  exit 1
fi
if ! find "$facts_dir" -maxdepth 1 -type f -name '*.json' -print -quit | grep -q .; then
  echo "No fact JSON files found in: $facts_dir" >&2
  echo "Run memory_extraction.sh first." >&2
  exit 1
fi

echo "Preprocessing PyraVid fact memory"
echo "  dataset: $dataset"
echo "  facts:   $facts_dir"
echo "  output:  ${repo_root}/processed_data/${dataset}"

exec python -m prototype.preprocess_chunks \
  --dataset "$dataset" \
  --facts_dir "$facts_dir" \
  --api_key_dir "${PYRAVID_OPENAI_KEY_PATH:-${repo_root}/config/openai_key.txt}" \
  --model "${PYRAVID_PREPROCESS_MODEL:-gpt-4o-mini}" \
  --embedding_model "${PYRAVID_EMBEDDING_MODEL:-text-embedding-3-large}" \
  --max_workers "$workers"
