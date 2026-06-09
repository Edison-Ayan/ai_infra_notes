#!/usr/bin/env bash
# 编译 topic 05 量化 labs。用法: ./build.sh [文件名(不含.cu)]
set -e
export PATH=$HOME/miniconda3/envs/ai_infra/bin:$PATH
SRC=${1:-quant_basics}
nvcc -O3 -arch=sm_89 "$SRC.cu" -o "$SRC"
echo "✅ built: ./$SRC"
