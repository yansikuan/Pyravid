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

video_folder="${1:-${PYRAVID_CHARACTER_VIDEO_FOLDER:-${repo_root}/data/videomme/test}}"
facts_path="${2:-${PYRAVID_CHARACTER_FACTS_PATH:-${repo_root}/data/videomme/test_facts/test.json}}"
work_dir="${3:-${PYRAVID_CHARACTER_WORK_DIR:-${repo_root}/artifacts/character_processing/test}}"
speaker_checkpoint="${repo_root}/models/pretrained_eres2netv2.ckpt"

mkdir -p "$work_dir"

cd "$repo_root"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Running online character processing"
echo "  videos: $video_folder"
echo "  facts:  $facts_path"
echo "  output: $work_dir"

python prototype/character_processing_online.py \
  --video_folder "$video_folder" \
  --facts_path "$facts_path" \
  --work_dir "$work_dir"
