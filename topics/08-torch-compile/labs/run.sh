#!/usr/bin/env bash
# 跑 topic 08 torch.compile labs。python 无需编译，故用 run.sh。
# 用法:
#   ./run.sh compile        # eager vs compiled 计时 + kernel 数
#   ./run.sh compile dump   # 额外把 Inductor 生成的 Triton 源码落盘并 grep 融合 kernel
set -e
export PATH=$HOME/miniconda3/envs/ai_infra/bin:$PATH

if [ "$2" = "dump" ]; then
    # TORCH_LOGS=output_code 把 Inductor 生成的 Triton 源码打到 stderr
    TORCH_LOGS=output_code python compile_ffn.py 2> inductor_output.log
    echo
    echo "=== Inductor 自动生成的融合 Triton kernel（名字里写了融了哪些 op）==="
    grep -E "def triton_.*fused" inductor_output.log | sed 's/:.*//' | sort -u || true
    echo "（完整生成源码见 inductor_output.log）"
else
    python compile_ffn.py
fi
