#!/usr/bin/env bash
# 编译 topic 02 的 tensor core labs。用法: ./build.sh [文件名(不含.cu)]
set -e
export PATH=$HOME/miniconda3/envs/ai_infra/bin:$PATH
NVTX_INC=$HOME/miniconda3/envs/ai_infra/lib/python3.10/site-packages/nvidia/nvtx/include
SRC=${1:-wmma_gemm}
nvcc -O3 -arch=sm_89 -I"$NVTX_INC" "$SRC.cu" -o "$SRC"
echo "✅ built: ./$SRC"
