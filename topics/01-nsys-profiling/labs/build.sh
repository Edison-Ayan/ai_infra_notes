#!/usr/bin/env bash
# 编译 gemm_lab。用法: ./build.sh
set -e
export PATH=$HOME/miniconda3/envs/ai_infra/bin:$PATH
NVTX_INC=$HOME/miniconda3/envs/ai_infra/lib/python3.10/site-packages/nvidia/nvtx/include
nvcc -O3 -arch=sm_89 -I"$NVTX_INC" gemm_lab.cu -o gemm_lab
echo "✅ built: ./gemm_lab"
