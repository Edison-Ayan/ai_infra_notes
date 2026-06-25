"""
topic 07 lab② · Triton 算子融合 —— matmul + bias + GELU 一个 kernel

承接 lab①(GEMM 对线 cuBLAS)。lab① 优化的是**单 kernel** 内部，这一篇第一次碰
「融合」——我手写 CUDA 笔记里完全没有的维度：优化 kernel **之间**的访存。

为什么融合省钱（decode/推理同理是访存 bound）：
  不融合(torch 三步)：
    ① matmul  把 M×N 中间结果写回 HBM
    ② +bias   再读 M×N、加 bias、写 M×N
    ③ gelu    再读 M×N、过 GELU、写 M×N
    → M×N 这块大矩阵在 HBM 来回搬了好几趟，全是访存浪费
  融合(Triton 一步)：
    matmul 的累加器还在寄存器里，就地加 bias + GELU，**只写一次** M×N
    → 省掉中间结果的 2 次额外读 + 2 次额外写

用法：  ./run.sh fuse           # 跑融合对比
        ./run.sh fuse fp32      # FP32(TF32)
"""
import os
import torch
import triton
import triton.language as tl


def _configs():
    cfgs = []
    for bm, bn in [(128, 128), (128, 64), (64, 128), (64, 64)]:
        for bk in [32, 64]:
            for stages in [2, 3, 4]:
                for warps in [4, 8]:
                    cfgs.append(triton.Config(
                        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": 8},
                        num_stages=stages, num_warps=warps))
    return cfgs


@triton.jit
def _gelu(x):
    # 精确 GELU(erf 版)，对齐 torch F.gelu 默认：0.5x(1 + erf(x/√2))
    return 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))


@triton.autotune(configs=_configs(), key=["M", "N", "K"])
@triton.jit
def gemm_bias_gelu_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # ★ 融合 epilogue：累加器还在寄存器里，就地加 bias + GELU，再一次写回。
    #   不融合的话这里 acc 要先写 HBM，bias/gelu 各自再读回来——省的就是这两趟。
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    bias = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0)   # 每列一个 bias
    acc = acc + bias[None, :]
    acc = _gelu(acc)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)


def triton_fused(a, b, bias):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    gemm_bias_gelu_kernel[grid](
        a, b, bias, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


def torch_unfused(a, b, bias):
    # 三个独立 kernel：matmul → +bias → gelu，中间结果两次往返 HBM
    return torch.nn.functional.gelu(torch.matmul(a, b) + bias, approximate="none")


def bench(fn, *args, warmup=25, rep=100):
    return triton.testing.do_bench(lambda: fn(*args), warmup=warmup, rep=rep)


def run_shape(M, N, K, dtype, tag):
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=dtype)
    b = torch.randn((K, N), device="cuda", dtype=dtype)
    bias = torch.randn((N,), device="cuda", dtype=dtype)

    c_f = triton_fused(a, b, bias)
    c_u = torch_unfused(a, b, bias)
    atol, rtol = (2e-2, 2e-2) if dtype == torch.float16 else (5e-1, 2e-2)
    ok = torch.allclose(c_f, c_u, atol=atol, rtol=rtol)
    t_f = bench(triton_fused, a, b, bias)
    t_u = bench(torch_unfused, a, b, bias)
    nbytes = M * N * (2 if dtype == torch.float16 else 4)
    extra_gb = 4 * nbytes / 1e9   # 不融合多搬 2 读 + 2 写 = 4×M×N×bytes
    print(f"[{tag}] {M}×{N}×{K}  {'✅' if ok else '❌FAIL'}  "
          f"融合 {t_f:6.3f}ms | 不融合 {t_u:6.3f}ms | 加速 {t_u/t_f:.2f}× | 省搬 ~{extra_gb:.2f}GB")


def main():
    dtype = torch.float16 if os.environ.get("MODE") != "fp32" else torch.float32
    if dtype == torch.float32:
        torch.backends.cuda.matmul.allow_tf32 = True
    print(f"dtype={dtype}")
    # 中间结果 M×N 一样大，但 K 越小 matmul 越不算力 bound → epilogue 访存占比越大 → 融合越赚
    run_shape(4096, 4096, 4096, dtype, "算力bound K=4096")
    run_shape(4096, 4096, 1024, dtype, "中间       K=1024")
    run_shape(4096, 4096,  256, dtype, "访存倾斜   K= 256")


if __name__ == "__main__":
    main()
