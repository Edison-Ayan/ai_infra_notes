#!/usr/bin/env bash
# 编译 topic 04 推理 labs。用法: ./build.sh [文件名(不含.cu)]
set -e
ENV=$HOME/miniconda3/envs/ai_infra
export PATH=$ENV/bin:$PATH
INC=$ENV/targets/x86_64-linux/include
LIB=$ENV/targets/x86_64-linux/lib
SRC=${1:-decode_batch}
nvcc -O3 -arch=sm_89 -I"$INC" "$SRC.cu" -o "$SRC" -L"$LIB" -lcublas
echo "✅ built: ./$SRC   (运行: LD_LIBRARY_PATH=$LIB ./$SRC)"
