"""
topic 09 lab① · 算法与调度分离 —— 同一个算法，调度决定一切

Halide/TVM 的核心命题（所有 AI 编译器的地基）：
    一个 kernel = 算法(算什么) + 调度(怎么算)
    - 算法：C = A @ B —— 数学，固定不变
    - 调度：tile 多大 / 几个 warp / 几级流水 —— 不改结果，只改性能
编译器的工作 = 固定算法、在调度空间里搜最快的（我 topic07 @autotune 就是这个）。

这个 lab 把命题坐实：同一个 GEMM kernel(算法一字不改)，手动喂几组不同调度，
看性能差一个数量级。

用法：  ./run.sh sweep
"""
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # ↓↓↓ 这段是“算法”：C=A@B 的分块累加，下面所有调度都跑这同一段，一字不改 ↓↓↓
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs); b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))
    # ↑↑↑ 算法到此结束 ↑↑↑


def run(a, b, BM, BN, BK, warps, stages):
    M, K = a.shape; _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    # 同一个 matmul_kernel，只是换 BLOCK / num_warps / num_stages —— 这就是“调度”
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, num_warps=warps, num_stages=stages,
    )
    return c


def main():
    M = N = K = 4096
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    ref = torch.matmul(a, b)
    flop = 2 * M * N * K

    # 同一算法的几组调度：从“能跑但很差”到“调好”
    schedules = [
        ("差   16³tile·1warp·无流水", 16, 16, 16, 1, 1),
        ("一般 64²tile·2warp·2级流水", 64, 64, 32, 2, 2),
        ("好   128²tile·4warp·3级流水", 128, 128, 32, 4, 3),
        ("瘦K  128×64tile·4warp·4级流水", 128, 64, 64, 4, 4),
    ]
    print("## 同一个 matmul_kernel(算法不变)，只换调度（4096³ FP16）")
    best, worst = 0.0, 1e9
    for tag, BM, BN, BK, w, s in schedules:
        c = run(a, b, BM, BN, BK, w, s)
        ok = torch.allclose(c, ref, atol=2e-2, rtol=2e-2)
        t = triton.testing.do_bench(lambda: run(a, b, BM, BN, BK, w, s))
        tflops = flop / (t * 1e-3) / 1e12
        best = max(best, tflops); worst = min(worst, tflops)
        print(f"[{tag:26s}] {'✅' if ok else '❌'} {t:7.3f} ms  {tflops:6.2f} TFLOPS")
    print(f"\n结论：算法一字没改，最好/最差调度差 {best / worst:.1f}× —— "
          f"这就是 Halide「算法与调度分离」：性能全在调度，编译器(@autotune)替你搜这个空间。")


if __name__ == "__main__":
    main()
