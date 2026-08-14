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

usage() {
  echo "Usage: $0 [{videomme|lvbench|m3web|m3robot} VIDEO_LIST]" >&2
  echo >&2
  echo "VIDEO_LIST must contain one directory per line. Each directory must" >&2
  echo "contain the numbered 30-second MP4 clips for one source video." >&2
  echo >&2
  echo "Optional environment variables:" >&2
  echo "  PYRAVID_DATASET    Dataset alias used when no arguments are given" >&2
  echo "  PYRAVID_VIDEO_LIST Video-list path used when no arguments are given" >&2
  echo "  PYRAVID_DATA_DIR   Data root (default: data)" >&2
  echo "  PYRAVID_FACTS_DIR  Facts output override" >&2
  echo "  PYRAVID_KEYFRAMES_DIR  Keyframe output override" >&2
  echo "  PYRAVID_THREADS    Worker threads (default: 10)" >&2
  echo "  PYRAVID_KEY_PATH   Legacy key-file argument (default: /dev/null)" >&2
}

if [[ $# -eq 0 ]]; then
  dataset_alias="${PYRAVID_DATASET:-videomme}"
  video_list="${PYRAVID_VIDEO_LIST:-${repo_root}/data/video_lists/videomme_test.txt}"
elif [[ $# -eq 2 ]]; then
  dataset_alias="$1"
  video_list="$2"
else
  usage
  exit 2
fi

case "$dataset_alias" in
  videomme) dataset="videomme" ;;
  lvbench) dataset="lvbench" ;;
  m3web) dataset="m3bench_web" ;;
  m3robot) dataset="m3bench_robot" ;;
  *) usage; exit 2 ;;
esac

if [[ ! -f "$video_list" ]]; then
  echo "Video list not found: $video_list" >&2
  exit 1
fi
video_list="$(cd "$(dirname "$video_list")" && pwd)/$(basename "$video_list")"

cd "$repo_root"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

data_root="${PYRAVID_DATA_DIR:-data}"
facts_dir="${PYRAVID_FACTS_DIR:-${data_root}/${dataset}/facts}"
keyframes_dir="${PYRAVID_KEYFRAMES_DIR:-${data_root}/${dataset}/keyframes}"
threads="${PYRAVID_THREADS:-10}"
key_path="${PYRAVID_KEY_PATH:-/dev/null}"

echo "Extracting PyraVid fact memory"
echo "  dataset:   $dataset"
echo "  video list: $video_list"
echo "  facts:     $facts_dir"
echo "  keyframes: $keyframes_dir"

exec python -m prototype.video_extraction.video_facts_extraction_parallel \
  --list "$video_list" \
  --facts "$facts_dir" \
  --keyframes "$keyframes_dir" \
  --key "$key_path" \
  --threads "$threads"
