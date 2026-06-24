#!/usr/bin/env bash
# 跑 topic 07 Triton lab。python 无需编译，故用 run.sh（不是 build.sh）。
# 用法: ./run.sh [fp16]   |   DUMP=1 ./run.sh   |   DUMP=1 ./run.sh fp16
set -e
export PATH=$HOME/miniconda3/envs/ai_infra/bin:$PATH
[ "$1" = "fp16" ] && export MODE=fp16
python triton_gemm.py
