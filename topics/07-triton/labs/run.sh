#!/usr/bin/env bash
# 跑 topic 07 Triton labs。python 无需编译，故用 run.sh（不是 build.sh）。
# 用法:
#   ./run.sh              # lab① GEMM 对线 cuBLAS（FP32/TF32）
#   ./run.sh fp16         # lab① FP16 走 tensor core
#   DUMP=1 ./run.sh fp16  # lab① 额外 dump ttir/ttgir/ptx
#   ./run.sh fuse         # lab② matmul+bias+GELU 融合对比（FP16）
#   ./run.sh fuse fp32    # lab② FP32(TF32)
set -e
export PATH=$HOME/miniconda3/envs/ai_infra/bin:$PATH

if [ "$1" = "fuse" ]; then
    [ "$2" = "fp32" ] && export MODE=fp32
    python triton_fused_gelu.py
else
    [ "$1" = "fp16" ] && export MODE=fp16
    python triton_gemm.py
fi
