#!/bin/bash
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
echo "Initializing vLLM Server..."

export VLLM_USE_STANDALONE_COMPILE=0
export LD_LIBRARY_PATH=$CONDA_PREFIX/cuda-compat:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=1

export VLLM_NO_USAGE_STATS=1

LOG_FILE="answer_model.log"
rm ${LOG_FILE}
export CUDA_VISIBLE_DEVICES="2,3"
setsid vllm serve Qwen/Qwen3-VL-32B-Instruct \
  --port 8000 \
  --tensor-parallel-size 2 \
  --download-dir /home/hk-project-p0022573/lmu_xjh4853/workspace_ysk_wht/hkfswork/lmu_xjh4853-m3-agent/lmu_xjh4853-m3-agent-1772592603/CAM/answer_model_test \
  --max-model-len 262144 \
  --limit-mm-per-prompt.video 0 \
  --gpu-memory-utilization 0.90 \
  --mm-processor-cache-gb 0 \
  > ${LOG_FILE} 2>&1 &

echo "answer model initialized"
tail -f ${LOG_FILE}
