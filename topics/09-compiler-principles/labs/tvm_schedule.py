"""
topic 09 lab③ · 亲手写 compute + schedule —— TVM 把"算法/调度分离"变成会写

lab① 是"看"调度影响性能(Triton 换 BLOCK)；这一篇是"写"——用 TVM 显式调度原语
(split / reorder / parallel / vectorize)亲手把一个 naive matmul 一步步调快，
体会 Halide 的命题：**算法写一次，调度是另一套可独立改写的代码**。

CPU(LLVM) target —— 调度原语和 GPU 上完全一样，学原理 CPU 足够，还省掉 GPU codegen 的坑。

环境坑(TVM 0.25 + torch 2.6，已在本文件顶部处理，见下注释)：
  - apache-tvm-ffi 自带的 torch C-dlpack 扩展和 torch 2.6 ABI 不兼容 → import 即崩。
    用 sys.modules 把它标成 None，让那个可选扩展干净地"加载失败→降级"，不影响 CPU TE。
  - 这版把子包 tir 改名 s_tir、方法 get_block 改名 get_sblock（迁移期过渡命名）。

用法：  ./run.sh tvm
"""
import sys
sys.modules["torch_c_dlpack_ext"] = None      # 见顶部注释：绕开 ABI 不兼容的可选扩展
import warnings
warnings.simplefilter("ignore")

import time
import numpy as np
import tvm
from tvm import te, s_tir          # 0.25 里 tir 子包改名 s_tir

N = 1024


def make_algorithm():
    """算法层：只描述'算什么'，C = A @ B。一个字都不提怎么循环/分块/并行。"""
    A = te.placeholder((N, N), name="A")
    B = te.placeholder((N, N), name="B")
    k = te.reduce_axis((0, N), "k")
    C = te.compute((N, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
    return A, B, C


def fresh_schedule():
    # 每次从同一个算法新建一份"调度对象"，保证下面几种调度跑的是同一段数学
    A, B, C = make_algorithm()
    return s_tir.Schedule(tvm.IRModule({"main": te.create_prim_func([A, B, C])}))


def bench(sch, repeat=10):
    f = tvm.compile(sch.mod, target="llvm")
    dev = tvm.cpu()
    a = tvm.runtime.tensor(np.random.rand(N, N).astype("float32"), dev)
    b = tvm.runtime.tensor(np.random.rand(N, N).astype("float32"), dev)
    c = tvm.runtime.tensor(np.zeros((N, N), "float32"), dev)
    f(a, b, c)                                  # 预热
    t0 = time.time()
    for _ in range(repeat):
        f(a, b, c)
    return (time.time() - t0) / repeat * 1000   # ms


def loop_order(sch):
    # 把当前 schedule 的循环嵌套顺序打印出来，看"调度改的是循环、不是数学"
    return str(sch.mod["main"]).count("for"), \
        [ln.strip() for ln in str(sch.mod["main"]).splitlines() if ln.strip().startswith("for")]


def main():
    print(f"## 同一个算法 C=A@B（{N}³, FP32, CPU/LLVM），只换调度\n")

    # ① 默认调度：朴素三重循环 i,j,k —— 内层对 B[k,j] 跨行访问，cache 全 miss
    s0 = fresh_schedule()
    t0 = bench(s0, repeat=3)
    print(f"[默认 naive i,j,k          ] {t0:8.1f} ms   (1.0×)")

    # ② 只做 reorder：把 k 提到 j 外面 → 内层 j 对 B[k,j] 连续访问，cache 友好
    #    经典 loop interchange，一行调度、算法没动
    s1 = fresh_schedule()
    blk = s1.get_sblock("C")
    i, j, k = s1.get_loops(blk)
    s1.reorder(i, k, j)
    t1 = bench(s1)
    print(f"[+reorder(i,k,j) 连续访存  ] {t1:8.1f} ms   ({t0 / t1:.1f}×)")

    # ③ tile + parallel + vectorize：分块提 cache 复用 + 多核 + SIMD
    s2 = fresh_schedule()
    blk = s2.get_sblock("C")
    i, j, k = s2.get_loops(blk)
    io, ii = s2.split(i, [None, 32])
    jo, ji = s2.split(j, [None, 32])
    s2.reorder(io, jo, k, ii, ji)
    s2.parallel(io)        # 外层 i 块 → 多核并行
    s2.vectorize(ji)       # 内层 j → SIMD 向量化
    t2 = bench(s2)
    print(f"[+tile+parallel+vectorize  ] {t2:8.1f} ms   ({t0 / t2:.1f}×)")

    print(f"\n结论：算法 C=A@B 一个字没改，只改调度（reorder/split/parallel/vectorize），"
          f"性能 {t0 / t2:.0f}×。")
    print("这就是 Halide「算法与调度分离」最直接的体感——调度是一套可独立改写的代码，"
          "编译器(@autotune/AutoTVM)就是替你自动写这套。")

    # 看一眼调度确实只改了循环嵌套（数学不变）
    n2, fors2 = loop_order(s2)
    print(f"\n[调度后循环嵌套] {n2} 层 for（默认只有 3 层 i/j/k）：")
    for ln in fors2[:6]:
        print("   ", ln)


if __name__ == "__main__":
    main()
