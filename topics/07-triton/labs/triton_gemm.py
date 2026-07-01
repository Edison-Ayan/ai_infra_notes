"""
topic 07 lab① · Triton GEMM —— 和 topic 01 手写 CUDA GEMM 对照

目的：
    用 Triton 重写 GEMM，对线自己手写 CUDA GEMM。
    重点不是“写出一个矩阵乘”，而是理解：

        手写 CUDA 里要自己解决的问题
            tile 划分
            shared memory 搬运
            bank conflict
            double buffering / cp.async
            向量化访存
            occupancy 和 ILP 平衡

        在 Triton 里对应变成哪些旋钮
            BLOCK_M / BLOCK_N / BLOCK_K
            num_stages
            num_warps
            GROUP_M
            autotune

对照表：

    手写 CUDA GEMM                     Triton GEMM

    BM × BN × BK tile                  BLOCK_M / BLOCK_N / BLOCK_K
    blockIdx / threadIdx 映射           program_id + tl.arange
    shared memory 手动分配              编译器自动规划 shared / register
    LDG.128 / float4 向量化             编译器根据 tl.load pattern 自动合并访存
    cp.async + double buffer            num_stages 控制软件流水
    bank conflict / swizzle             编译器布局优化 + GROUP_M 改善 L2 复用
    Tensor Core mma 指令                tl.dot 自动 lowering 到 mma
    手动调 tile / occupancy / ILP        autotune 搜索 config

用法：

    ./run.sh
        默认 FP32 输入。
        注意：Triton 的 FP32 tl.dot 默认可能走 TF32 tensor core。

    ./run.sh fp16
        FP16 输入，tl.dot 通常会走 tensor core。

    DUMP=1 ./run.sh
        额外 dump TTIR / TTGIR / PTX，
        用来看 Triton 是怎么从 Python DSL lowering 到 GPU 代码的。
"""

import os
import torch
import triton
import triton.language as tl


# ============================================================
# 1. autotune 配置搜索空间
# ============================================================
#
# 在手写 CUDA GEMM 里，我们通常要人工试很多组参数：
#
#   - block tile 多大？
#       128x128？
#       128x64？
#       64x128？
#
#   - K 方向一次吃多少？
#       BK=32？
#       BK=64？
#
#   - shared memory 做几级流水？
#       double buffer？
#       triple buffer？
#
#   - 一个 program/block 用几个 warp？
#       4 warps？
#       8 warps？
#
# 这些参数会共同影响：
#
#   1. 每个 block 的计算量
#   2. 每个 block 的寄存器使用量
#   3. shared memory 使用量
#   4. occupancy
#   5. L2 / global memory 复用
#   6. Tensor Core 利用率
#
# 手写 CUDA 时这就是“玄学调参”。
# Triton 的 autotune 会帮我们把这些 config 跑一遍，
# 然后选择实测最快的那组。
#
def _configs():
    cfgs = []

    # BLOCK_M / BLOCK_N：
    #   一个 Triton program 负责计算 C 的一个 tile：
    #
    #       C_tile shape = [BLOCK_M, BLOCK_N]
    #
    #   例如 BLOCK_M=128, BLOCK_N=128，
    #   表示一个 program 负责 C 中 128x128 的小块。
    #
    #   tile 太小：
    #       并行度足够，但每个 program 计算量少，访存复用差。
    #
    #   tile 太大：
    #       计算复用好，但寄存器压力大，occupancy 可能下降。
    #
    for bm, bn in [(128, 128), (128, 64), (64, 128), (64, 64)]:

        # BLOCK_K：
        #   K 方向每次 reduce 的块大小。
        #
        # GEMM:
        #
        #   C[M, N] = A[M, K] @ B[K, N]
        #
        # 一个 C tile 需要沿着 K 维循环累加：
        #
        #   for k in range(0, K, BLOCK_K):
        #       A_tile = A[BLOCK_M, BLOCK_K]
        #       B_tile = B[BLOCK_K, BLOCK_N]
        #       acc += A_tile @ B_tile
        #
        # BLOCK_K 越大：
        #   - 每轮 dot 吃的数据更多
        #   - 可能更利于 Tensor Core
        #   - 但寄存器/shared 压力也更高
        #
        for bk in [32, 64]:

            # num_stages：
            #   控制软件流水级数。
            #
            # 类比手写 CUDA：
            #
            #   num_stages=2  大致类似 double buffering
            #   num_stages=3  大致类似 triple buffering
            #   num_stages=4  更深流水
            #
            # 它的作用是：
            #   当前 BLOCK_K 正在算的时候，
            #   编译器尝试提前把后面的 BLOCK_K 数据预取进来，
            #   从而隐藏 global memory latency。
            #
            # 注意：
            #   stages 不是越大越好。
            #   stages 越大，临时 buffer 越多，寄存器/shared 压力越大，
            #   occupancy 可能下降。
            #
            for stages in [2, 3, 4]:

                # num_warps：
                #   一个 Triton program 内部使用多少个 warp。
                #
                # 类比 CUDA：
                #   一个 block 里有多少 warp 参与计算。
                #
                # num_warps 越大：
                #   - program 内部并行度更高
                #   - 大 tile 可能跑得更好
                #   - 但调度/同步/资源占用也会变大
                #
                for warps in [4, 8]:

                    # triton.Config 的第一个参数是 META 参数。
                    # 这些参数会在 @triton.jit 编译期成为 tl.constexpr，
                    # 所以编译器能根据这些常量生成专门优化后的 kernel。
                    cfgs.append(
                        triton.Config(
                            {
                                "BLOCK_M": bm,
                                "BLOCK_N": bn,
                                "BLOCK_K": bk,

                                # GROUP_M：
                                #   不是 C tile 的形状参数，
                                #   而是 program 遍历顺序的 swizzle 参数。
                                #
                                # 它会把 program_id 映射成更 L2-cache-friendly 的顺序。
                                # 后面 kernel 里会详细解释。
                                "GROUP_M": 8,
                            },
                            num_stages=stages,
                            num_warps=warps,
                        )
                    )

    return cfgs


# ============================================================
# 2. Triton GEMM kernel
# ============================================================
#
# @triton.autotune:
#   让 Triton 在第一次运行时，对 _configs() 中的多组配置做 benchmark，
#   然后针对 key=["M", "N", "K"] 选择最快的 config。
#
#   key=["M", "N", "K"] 的意思是：
#       M/N/K 不同，最佳 tile 可能不同。
#       所以 autotune 结果按矩阵规模缓存。
#
# @triton.jit:
#   把 Python 写的 Triton DSL 编译成 GPU kernel。
#
# 注意：
#   这个函数不是普通 Python 函数。
#   它会被 Triton 编译器解析、优化、lowering。
#
@triton.autotune(configs=_configs(), key=["M", "N", "K"])
@triton.jit
def gemm_kernel(
    # 三个矩阵的指针
    a_ptr, b_ptr, c_ptr,

    # 矩阵形状
    M, N, K,

    # A 的 stride
    # A shape = [M, K]
    # A[m, k] 的地址 = a_ptr + m * stride_am + k * stride_ak
    stride_am, stride_ak,

    # B 的 stride
    # B shape = [K, N]
    # B[k, n] 的地址 = b_ptr + k * stride_bk + n * stride_bn
    stride_bk, stride_bn,

    # C 的 stride
    # C shape = [M, N]
    # C[m, n] 的地址 = c_ptr + m * stride_cm + n * stride_cn
    stride_cm, stride_cn,

    # 下面这些是编译期常量。
    # 它们来自 triton.Config。
    #
    # tl.constexpr 的作用：
    #   告诉 Triton 这些值在编译期就已知。
    #   编译器可以据此展开循环、分配寄存器、选择 mma layout。
    #
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # ------------------------------------------------------------
    # 2.1 program_id：Triton 里的 blockIdx
    # ------------------------------------------------------------
    #
    # 在 CUDA 里：
    #
    #   blockIdx.x 表示当前 CTA / block 的编号。
    #
    # 在 Triton 里：
    #
    #   tl.program_id(0) 表示当前 program 在第 0 维 grid 上的编号。
    #
    # 一个 Triton program 可以粗略类比为一个 CUDA block。
    #
    pid = tl.program_id(0)

    # C 被切成若干个 tile：
    #
    #   M 方向有 num_pid_m 个 tile
    #   N 方向有 num_pid_n 个 tile
    #
    # 例如：
    #   M=N=4096
    #   BLOCK_M=128
    #   BLOCK_N=128
    #
    # 那么：
    #   num_pid_m = 4096 / 128 = 32
    #   num_pid_n = 4096 / 128 = 32
    #
    # 总 program 数 = 32 * 32 = 1024。
    #
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # ------------------------------------------------------------
    # 2.2 GROUP_M swizzle：改变 program 访问 C tile 的顺序
    # ------------------------------------------------------------
    #
    # 最朴素的映射方式是：
    #
    #   pid_m = pid // num_pid_n
    #   pid_n = pid %  num_pid_n
    #
    # 这相当于按行优先遍历 C tile：
    #
    #   (0,0), (0,1), (0,2), ...
    #   (1,0), (1,1), (1,2), ...
    #
    # 问题：
    #   不同 C tile 会复用 A 或 B 的部分数据。
    #
    #   C[m, n0] 和 C[m, n1]
    #       复用同一块 A[m, :]
    #
    #   C[m0, n] 和 C[m1, n]
    #       复用同一块 B[:, n]
    #
    # 如果 program 顺序不好，A/B tile 刚进 L2 cache，
    # 下一个 program 却不复用它，就浪费了 L2。
    #
    # GROUP_M 的作用：
    #   把 M 方向的多个 tile 分成一个 group，
    #   在 group 内优先遍历 N 方向。
    #
    # 这样可以让相邻 program 更可能复用同一批 A/B tile，
    # 提高 L2 cache 命中率。
    #
    # 这对应手写 CUDA 里常见的 block swizzle / CTA swizzle。
    #
    num_pid_in_group = GROUP_M * num_pid_n

    # 当前 pid 属于第几个 group。
    group_id = pid // num_pid_in_group

    # 当前 group 在 M 方向的起始 tile 编号。
    first_pid_m = group_id * GROUP_M

    # 最后一个 group 可能不足 GROUP_M 个 M tile，
    # 所以这里要取 min。
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)

    # 在 group 内计算 pid_m。
    #
    # pid % group_size_m：
    #   当前 program 在 group 内的 M 方向偏移。
    #
    pid_m = first_pid_m + (pid % group_size_m)

    # 在 group 内计算 pid_n。
    #
    # 注意这里除以 group_size_m，而不是 GROUP_M。
    # 因为最后一个 group 可能不足 GROUP_M。
    #
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ------------------------------------------------------------
    # 2.3 构造当前 program 负责的 M/N/K 下标
    # ------------------------------------------------------------
    #
    # tl.arange(0, BLOCK_M) 会生成一个向量：
    #
    #   [0, 1, 2, ..., BLOCK_M-1]
    #
    # 这不是普通 Python list，
    # 而是 Triton 编译器理解的向量化 index。
    #
    # offs_m：
    #   当前 C tile 覆盖的 M 方向行号。
    #
    # offs_n：
    #   当前 C tile 覆盖的 N 方向列号。
    #
    # offs_k：
    #   当前 BLOCK_K 内部的 K 方向偏移。
    #
    # 这里的 % M / % N 是 Triton GEMM 常见写法：
    #   对边界 tile 做 wrap-around，保证 load 地址不越界。
    #
    # 真正是否写入 C，由最后 tl.store 的 mask 决定。
    #
    # 对于 M/N 刚好是 BLOCK_M/BLOCK_N 整数倍的情况，
    # 这个 % 没有实际影响。
    #
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # ------------------------------------------------------------
    # 2.4 构造 A_tile / B_tile 的指针矩阵
    # ------------------------------------------------------------
    #
    # 当前 program 要计算：
    #
    #   C_tile[BLOCK_M, BLOCK_N]
    #
    # 它每轮 K 循环需要加载：
    #
    #   A_tile[BLOCK_M, BLOCK_K]
    #   B_tile[BLOCK_K, BLOCK_N]
    #
    # 这里用广播构造二维地址矩阵。
    #
    # A 地址：
    #
    #   a_ptrs shape = [BLOCK_M, BLOCK_K]
    #
    #   a_ptrs[i, j] =
    #       a_ptr
    #       + offs_m[i] * stride_am
    #       + offs_k[j] * stride_ak
    #
    # B 地址：
    #
    #   b_ptrs shape = [BLOCK_K, BLOCK_N]
    #
    #   b_ptrs[i, j] =
    #       b_ptr
    #       + offs_k[i] * stride_bk
    #       + offs_n[j] * stride_bn
    #
    # 对 row-major contiguous tensor：
    #
    #   A stride 通常是:
    #       stride_am = K
    #       stride_ak = 1
    #
    #   B stride 通常是:
    #       stride_bk = N
    #       stride_bn = 1
    #
    a_ptrs = (
        a_ptr
        + offs_m[:, None] * stride_am
        + offs_k[None, :] * stride_ak
    )

    b_ptrs = (
        b_ptr
        + offs_k[:, None] * stride_bk
        + offs_n[None, :] * stride_bn
    )

    # ------------------------------------------------------------
    # 2.5 accumulator：FP32 累加器
    # ------------------------------------------------------------
    #
    # acc shape = [BLOCK_M, BLOCK_N]
    #
    # 即使输入是 FP16，
    # GEMM 通常也会用 FP32 accumulator，
    # 这样数值精度更好。
    #
    # 手写 CUDA Tensor Core 里也常见：
    #
    #   half input
    #   float accumulator
    #
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ------------------------------------------------------------
    # 2.6 沿 K 方向循环累加
    # ------------------------------------------------------------
    #
    # GEMM 本质：
    #
    #   C[m, n] = sum_k A[m, k] * B[k, n]
    #
    # 当前 program 只负责一个 C_tile，
    # 所以要沿 K 方向分块循环：
    #
    #   for k_block in range(0, K, BLOCK_K):
    #       acc += A_tile @ B_tile
    #
    # tl.cdiv(K, BLOCK_K) 是向上取整，
    # 用来处理 K 不是 BLOCK_K 整数倍的情况。
    #
    for k in range(0, tl.cdiv(K, BLOCK_K)):

        # 当前 K block 对应的有效范围是：
        #
        #   k * BLOCK_K + offs_k
        #
        # 对最后一个 K block，
        # 如果 K 不是 BLOCK_K 的整数倍，
        # 可能会有一部分 offs_k 越界。
        #
        # mask 的作用：
        #   只加载有效 K 范围内的数据。
        #
        # other=0.0：
        #   对越界位置填 0。
        #
        # 这样不会影响矩阵乘结果，因为：
        #
        #   acc += A * B
        #
        # 越界部分变成 0，相当于不贡献。
        #
        # 注意：
        #   这两行 tl.load 在源码层看起来是普通 load，
        #   但 Triton 编译器会根据 BLOCK 形状、num_stages、tl.dot
        #   自动决定具体的寄存器布局、shared memory staging 和流水。
        #
        # 类比手写 CUDA：
        #
        #   - global load
        #   - vectorized load
        #   - shared memory staging
        #   - cp.async prefetch
        #   - __syncthreads
        #
        # 在 Triton 里不需要显式写出来。
        #
        a = tl.load(
            a_ptrs,
            mask=offs_k[None, :] < K - k * BLOCK_K,
            other=0.0,
        )

        b = tl.load(
            b_ptrs,
            mask=offs_k[:, None] < K - k * BLOCK_K,
            other=0.0,
        )

        # tl.dot：
        #
        #   a shape = [BLOCK_M, BLOCK_K]
        #   b shape = [BLOCK_K, BLOCK_N]
        #
        #   acc += a @ b
        #
        # 这是整个 kernel 最核心的一句。
        #
        # 对 FP16 输入：
        #   通常 lowering 到 Tensor Core mma 指令。
        #
        # 对 FP32 输入：
        #   Triton 默认可能使用 TF32 tensor core。
        #   TF32 不是完整 FP32，它保留 FP32 的 exponent，
        #   但 mantissa 精度更低，大概 10 位尾数。
        #
        # 所以下面 main() 里要设置：
        #
        #   torch.backends.cuda.matmul.allow_tf32 = True
        #
        # 这样 torch.matmul 和 Triton 才是同精度比较。
        #
        acc += tl.dot(a, b)

        # 指针推进到下一个 K block。
        #
        # A_tile 从：
        #   A[:, k : k + BLOCK_K]
        #
        # 变成：
        #   A[:, k + BLOCK_K : k + 2 * BLOCK_K]
        #
        # B_tile 同理。
        #
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # ------------------------------------------------------------
    # 2.7 构造 C_tile 的写回地址
    # ------------------------------------------------------------
    #
    # offs_cm / offs_cn 是真实 C 下标，不使用 %。
    #
    # 因为写回时不能 wrap-around。
    # 如果越界写回，会污染 C 的其他位置。
    #
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # C 地址矩阵：
    #
    #   c_ptrs shape = [BLOCK_M, BLOCK_N]
    #
    #   c_ptrs[i, j] =
    #       c_ptr
    #       + offs_cm[i] * stride_cm
    #       + offs_cn[j] * stride_cn
    #
    c_ptrs = (
        c_ptr
        + stride_cm * offs_cm[:, None]
        + stride_cn * offs_cn[None, :]
    )

    # 写回 mask：
    #
    #   只有 offs_cm < M 且 offs_cn < N 的位置才写回。
    #
    # 这用于处理边界 tile。
    #
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    # 把 FP32 accumulator cast 回 C 的 dtype。
    #
    # 如果输入 dtype 是 float16：
    #   C 也是 float16，最后写回 float16。
    #
    # 如果输入 dtype 是 float32：
    #   C 也是 float32，最后写回 float32。
    #
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)


# ============================================================
# 3. Python wrapper：像 torch.matmul 一样调用 Triton kernel
# ============================================================
#
def triton_gemm(a, b):
    # A shape = [M, K]
    M, K = a.shape

    # B shape = [K, N]
    K, N = b.shape

    # 输出 C shape = [M, N]
    #
    # dtype 和 a 一致。
    # 这里假设 a/b dtype 一样。
    #
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # grid 决定启动多少个 Triton program。
    #
    # 每个 program 负责一个 C tile：
    #
    #   C_tile shape = [BLOCK_M, BLOCK_N]
    #
    # 所以 program 数量是：
    #
    #   ceil(M / BLOCK_M) * ceil(N / BLOCK_N)
    #
    # META 是当前 autotune config。
    # 不同 config 有不同 BLOCK_M / BLOCK_N。
    #
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"])
        * triton.cdiv(N, META["BLOCK_N"]),
    )

    # 启动 Triton kernel。
    #
    # gemm_kernel[grid](...) 是 Triton 的 kernel launch 语法。
    #
    # 注意：
    #   BLOCK_M / BLOCK_N / BLOCK_K / GROUP_M
    #   不需要在这里显式传。
    #
    # 它们来自 autotune 选中的 triton.Config。
    #
    gemm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )

    return c


# ============================================================
# 4. benchmark helper
# ============================================================
#
def bench(fn, *args, warmup=25, rep=100):
    # triton.testing.do_bench 会：
    #
    #   1. warmup 若干次
    #   2. 正式测 rep 次
    #   3. 用 CUDA event 计时
    #
    # 返回值单位是 ms。
    #
    # 这里 lambda: fn(*args) 是为了把函数调用封装成无参 callable。
    #
    return triton.testing.do_bench(
        lambda: fn(*args),
        warmup=warmup,
        rep=rep,
    )


# ============================================================
# 5. main：正确性检查 + 性能对比 + dump lowering
# ============================================================
#
def main():
    # MODE=fp16 时走 FP16。
    # 否则默认 FP32。
    #
    # run.sh 里可以写：
    #
    #   MODE=fp16 python main.py
    #
    dtype = torch.float16 if os.environ.get("MODE") == "fp16" else torch.float32

    # ------------------------------------------------------------
    # 5.1 TF32 公平性设置
    # ------------------------------------------------------------
    #
    # 这是非常重要的坑。
    #
    # NVIDIA Ampere / Ada 上：
    #   FP32 GEMM 可以用 TF32 Tensor Core 加速。
    #
    # Triton 的 tl.dot 对 FP32 输入，默认可能使用 TF32。
    # torch.matmul 默认是否允许 TF32，取决于 PyTorch 设置。
    #
    # 如果 Triton 用 TF32，而 torch.matmul 用真正 FP32，
    # 那么 Triton 会看起来“快很多”，但其实精度不一样。
    #
    # 所以这里显式打开 torch 的 TF32，
    # 让 torch.matmul 和 Triton 在同一个精度等级下比较。
    #
    # 如果你想比较真正 FP32：
    #   需要关闭 allow_tf32，
    #   并且在 tl.dot 里指定 input_precision="ieee"。
    #
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 矩阵规模。
    #
    # 4096^3 GEMM 足够大，可以让 GPU 跑满一些，
    # 避免小矩阵时 launch overhead 占比过高。
    #
    M = N = K = 4096

    # 固定随机种子，保证每次输入一致。
    torch.manual_seed(0)

    # 构造输入矩阵。
    #
    # A shape = [M, K]
    # B shape = [K, N]
    #
    a = torch.randn((M, K), device="cuda", dtype=dtype)
    b = torch.randn((K, N), device="cuda", dtype=dtype)

    # ------------------------------------------------------------
    # 5.2 正确性检查
    # ------------------------------------------------------------
    #
    # c_tri：
    #   Triton GEMM 结果。
    #
    # c_ref：
    #   torch.matmul 结果，底层通常是 cuBLAS。
    #
    c_tri = triton_gemm(a, b)
    c_ref = torch.matmul(a, b)

    # 容差设置：
    #
    # FP16：
    #   输入低精度，累加通常 FP32，但最后写回 FP16。
    #   给 1e-2 级别容差。
    #
    # FP32 + TF32：
    #   虽然 dtype 是 FP32，
    #   但乘法可能用 TF32，尾数精度比真 FP32 低。
    #   所以 max error 可能明显大于 1e-3。
    #
    atol, rtol = (1e-2, 1e-2) if dtype == torch.float16 else (3e-1, 2e-2)

    ok = torch.allclose(c_tri, c_ref, atol=atol, rtol=rtol)
    max_err = (c_tri - c_ref).abs().max().item()

    print(
        f"[正确性] {'✅ pass' if ok else '❌ FAIL'}  "
        f"max_abs_err={max_err:.4f}  dtype={dtype}"
    )

    # ------------------------------------------------------------
    # 5.3 性能测试
    # ------------------------------------------------------------
    #
    # GEMM FLOPs:
    #
    #   C[M, N] = A[M, K] @ B[K, N]
    #
    # 每个 C 元素需要：
    #   K 次乘法 + K 次加法
    #
    # 所以总 FLOPs 约等于：
    #
    #   2 * M * N * K
    #
    flop = 2 * M * N * K

    # Triton GEMM 时间，单位 ms。
    t_tri = bench(triton_gemm, a, b)

    # cuBLAS / torch.matmul 时间，单位 ms。
    t_ref = bench(torch.matmul, a, b)

    # ms 转 TFLOPS：
    #
    #   FLOPs / seconds / 1e12
    #
    # t_tri 是 ms，所以 seconds = t_tri * 1e-3。
    #
    tflops_tri = flop / (t_tri * 1e-3) / 1e12
    tflops_ref = flop / (t_ref * 1e-3) / 1e12

    print(f"[Triton ] {t_tri:7.3f} ms  {tflops_tri:6.2f} TFLOPS")
    print(f"[cuBLAS ] {t_ref:7.3f} ms  {tflops_ref:6.2f} TFLOPS  (torch.matmul)")

    print(
        f"[对比   ] Triton = cuBLAS 的 "
        f"{tflops_tri / tflops_ref * 100:5.1f}%"
    )

    # ------------------------------------------------------------
    # 5.4 查看 autotune 选中的最优配置
    # ------------------------------------------------------------
    #
    # gemm_kernel.best_config 会记录当前 M/N/K 下
    # autotune 实测最快的那组 config。
    #
    # 这就是 Triton 帮我们搜索出来的：
    #
    #   BLOCK_M
    #   BLOCK_N
    #   BLOCK_K
    #   GROUP_M
    #   num_warps
    #   num_stages
    #
    # 手写 CUDA 时，这些参数通常要靠人工一点点调。
    #
    best = gemm_kernel.best_config

    print(f"[autotune 选中] {best}")
    print("  ↑ 这组 BLOCK/num_stages/num_warps 就是编译器替我解出的『联立方程』解。")

    # ------------------------------------------------------------
    # 5.5 dump Triton lowering 结果
    # ------------------------------------------------------------
    #
    # DUMP=1 时，把 Triton 编译过程中的中间表示 dump 出来。
    #
    # 常见 lowering 链路：
    #
    #   Triton Python DSL
    #       ↓
    #   TTIR
    #       Triton Tensor IR
    #       还比较接近算法表达。
    #
    #       ↓
    #   TTGIR
    #       Triton GPU IR
    #       已经开始绑定 GPU layout、memory space、mma 等。
    #
    #       ↓
    #   LLIR
    #       LLVM IR
    #
    #       ↓
    #   PTX
    #       NVIDIA 汇编级中间代码。
    #
    #       ↓
    #   CUBIN
    #       真正 GPU 执行的机器码。
    #
    # 看 dump 的意义：
    #
    #   你可以验证：
    #       tl.dot 是否 lowering 到 mma
    #       是否出现 shared memory
    #       是否生成向量化 load
    #       pipeline / layout 是什么样
    #
    if os.environ.get("DUMP") == "1":

        # autotune 包了一层。
        # gemm_kernel.fn 才是真正的 JITFunction。
        jit = gemm_kernel.fn

        # 当前 CUDA device index。
        dev = a.device.index

        # Triton JIT 编译产物通常缓存在 jit.cache 里。
        # 不同 Triton 版本 cache 结构可能略有差异。
        #
        kerns = list(jit.cache.get(dev, {}).values())

        if kerns:
            # 取一个编译产物看 lowering 即可。
            #
            # 如果有多个 config 被编译过，
            # 这里不一定正好是 best config，
            # 但足够用于观察 IR/ASM 结构。
            #
            k = kerns[0]

            # k.asm 里可能包含：
            #
            #   "ttir"
            #   "ttgir"
            #   "llir"
            #   "ptx"
            #   "cubin"
            #
            # 这里 dump 最重要的三个。
            #
            for stage in ("ttir", "ttgir", "ptx"):
                if stage in k.asm:
                    fn_out = f"dump.{stage}"

                    with open(fn_out, "w") as f:
                        f.write(k.asm[stage])

                    print(
                        f"[dump] {stage:5s} -> {fn_out}  "
                        f"({len(k.asm[stage].splitlines()):4d} 行)"
                    )

            print(
                "  对照看：ttgir 里 #mma/#shared 布局 = "
                "编译器自动决定的 tensor core + shared 复用。"
            )


if __name__ == "__main__":
    main()