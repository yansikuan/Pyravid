#!/bin/bash
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
echo "Initializing vLLM server..."
export VLLM_USE_STANDALONE_COMPILE=0
LOG_FILE="link_model.log"
rm ${LOG_FILE}

setsid python -m vllm.entrypoints.openai.api_server \
 --model Qwen/Qwen3-4B-Instruct-2507 \
 --download-dir ./models \
 --tensor-parallel-size 4 \
 --port 8001 \
 > ${LOG_FILE} 2>&1 &

echo "link model initialized."
tail -f ${LOG_FILE}
