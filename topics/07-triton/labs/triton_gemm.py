"""
topic 07 lab① · Triton GEMM —— 和 topic 01 手写 CUDA GEMM 对照

目的：拿 Triton 重写 GEMM，对线我自己的 6.82 TFLOPS(98% cuBLAS) 手写版，
亲眼看「我手抠半个月的东西，编译器用哪几个旋钮自动发掉」。

对照表（手写 CUDA  →  Triton 里对应什么）：
  - tile 划分 BM×BN×BK        →  BLOCK_M / BLOCK_N / BLOCK_K（还是我定，编译器不猜这个）
  - shared memory 分配/搬运    →  tl.load 进寄存器块，shared 由编译器自动管
  - double buffering / cp.async →  num_stages（>1 就是软件流水，编译器自动 prefetch）
  - bank conflict / swizzle    →  GROUP_M（L2 友好的 block 重排）+ 编译器自动 swizzle
  - float4 向量化 LDG.128      →  编译器按 BLOCK_K 自动向量化访存
  - occupancy 换 ILP 的拉锯    →  num_warps + autotune 搜索（CUTLASS autotuning 在干的事）

用法：  ./run.sh            # 默认 FP32，对标手写版
        ./run.sh fp16      # FP16 走 tensor core，看上限
        DUMP=1 ./run.sh    # 额外 dump TTGIR/PTX，看 lowering
"""
import os
import torch
import triton
import triton.language as tl


# autotune：编译器替我搜 tile/流水/warp 的联立解。
# 每个 config 就是我手写 GEMM 时纠结的一组「tile×num_stages×occupancy」。
def _configs():
    cfgs = []
    for bm, bn in [(128, 128), (128, 64), (64, 128), (64, 64)]:
        for bk in [32, 64]:
            for stages in [2, 3, 4]:        # num_stages>1 = 自动 double/triple buffer
                for warps in [4, 8]:
                    cfgs.append(triton.Config(
                        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": 8},
                        num_stages=stages, num_warps=warps))
    return cfgs


@triton.autotune(configs=_configs(), key=["M", "N", "K"])
@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, c_ptr,
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

    # GROUP_M swizzle：把 block 的遍历顺序重排成 L2 友好的「列优先小组」，
    # 提高 A/B 分块在 L2 的复用——这正是我手写版要靠经验摆 blockIdx 才能拿到的。
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
        # 这两行 tl.load：shared 暂存、向量化访存、按 num_stages 做软件流水预取，
        # 全由编译器生成。对照我 CUDA 版里手写的 LDG.128 + cp.async + __syncthreads。
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a, b)             # FP16 时编译器自动选 tensor core(mma)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)


def triton_gemm(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    gemm_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


def bench(fn, *args, warmup=25, rep=100):
    return triton.testing.do_bench(lambda: fn(*args), warmup=warmup, rep=rep)  # 返回 ms


def main():
    dtype = torch.float16 if os.environ.get("MODE") == "fp16" else torch.float32

    # 陷阱：Triton 的 tl.dot 对 FP32 输入默认走 TF32(tensor core)，而 torch.matmul
    # 默认 allow_tf32=False 跑真 FP32。不对齐的话 Triton 会"假赢"2×。
    # 这里让 torch 也用 TF32，两边同精度才是公平对线。
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    M = N = K = 4096
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=dtype)
    b = torch.randn((K, N), device="cuda", dtype=dtype)

    # 正确性：和 cuBLAS(torch.matmul) 对一遍
    c_tri = triton_gemm(a, b)
    c_ref = torch.matmul(a, b)
    # FP32 路径两边都走 TF32(~10 位尾数)，容差按 TF32 放宽，不能用真 FP32 的 1e-3
    atol, rtol = (1e-2, 1e-2) if dtype == torch.float16 else (3e-1, 2e-2)
    ok = torch.allclose(c_tri, c_ref, atol=atol, rtol=rtol)
    max_err = (c_tri - c_ref).abs().max().item()
    print(f"[正确性] {'✅ pass' if ok else '❌ FAIL'}  max_abs_err={max_err:.4f}  dtype={dtype}")

    flop = 2 * M * N * K
    t_tri = bench(triton_gemm, a, b)
    t_ref = bench(torch.matmul, a, b)
    tflops_tri = flop / (t_tri * 1e-3) / 1e12
    tflops_ref = flop / (t_ref * 1e-3) / 1e12
    print(f"[Triton ] {t_tri:7.3f} ms  {tflops_tri:6.2f} TFLOPS")
    print(f"[cuBLAS ] {t_ref:7.3f} ms  {tflops_ref:6.2f} TFLOPS  (torch.matmul)")
    print(f"[对比   ] Triton = cuBLAS 的 {tflops_tri / tflops_ref * 100:5.1f}%")

    best = gemm_kernel.best_config
    print(f"[autotune 选中] {best}")
    print("  ↑ 这组 BLOCK/num_stages/num_warps 就是编译器替我解出的『联立方程』解。")

    if os.environ.get("DUMP") == "1":
        # dump lowering：看编译器把 tl.dot 一路降到 PTX 的完整中间表示链。
        # autotune 后真正的 JITFunction 在 .fn，编译产物在 .fn.cache[device]。
        jit = gemm_kernel.fn
        dev = a.device.index
        kerns = list(jit.cache.get(dev, {}).values())
        if kerns:
            k = kerns[0]   # 取一个代表，看 lowering 链即可
            # ttir(算法) → ttgir(绑硬件布局) → llir(LLVM) → ptx(汇编) → cubin(机器码)
            for stage in ("ttir", "ttgir", "ptx"):
                if stage in k.asm:
                    fn_out = f"dump.{stage}"
                    with open(fn_out, "w") as f:
                        f.write(k.asm[stage])
                    print(f"[dump] {stage:5s} -> {fn_out}  ({len(k.asm[stage].splitlines()):4d} 行)")
            print("  对照看：ttgir 里 #mma/#shared 布局 = 编译器自动决定的 tensor core + shared 复用。")


if __name__ == "__main__":
    main()
