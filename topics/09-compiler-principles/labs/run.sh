#!/usr/bin/env bash
# 跑 topic 09 编译器原理 labs。python 无需编译，故用 run.sh。
# 用法:
#   ./run.sh sweep    # lab① 算法与调度分离：同一算法换调度，性能差一个数量级
#   ./run.sh lower    # lab② 渐进式 lowering：一个 kernel ttir→ttgir→ptx 走一遍
set -e
export PATH=$HOME/miniconda3/envs/ai_infra/bin:$PATH

case "$1" in
  sweep) python schedule_sweep.py ;;
  lower) python lowering_walk.py ;;
  tvm)   python tvm_schedule.py ;;
  *)     echo "用法: ./run.sh [sweep|lower|tvm]"; exit 1 ;;
esac
