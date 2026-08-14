#!/bin/bash
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
echo "Initializing vLLM Server..."

export VLLM_USE_STANDALONE_COMPILE=0
export LD_LIBRARY_PATH=$CONDA_PREFIX/cuda-compat:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=1

LOG_FILE=selection_model.log
rm ${LOG_FILE}

setsid vllm serve Qwen/Qwen3-32B \
  --port 8003 \
  --download-dir ./models \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  > ${LOG_FILE} 2>&1 &

echo "selection model initialized"
tail -f ${LOG_FILE}
