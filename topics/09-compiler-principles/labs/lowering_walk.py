"""
topic 09 lab② · 渐进式 lowering / dialect 分层 —— 一个 kernel 从算法降到机器码

AI 编译器（MLIR/Triton/XLA）的第二个地基：不是一步到位翻成汇编，而是过一串
**中间表示(IR)**，每层只负责一件事，逐步把"硬件无关的算法"降成"绑死硬件的机器码"。
这叫渐进式 lowering（progressive lowering）/ dialect 分层。

Triton 的 lowering 链正好是个干净样本：
    ttir   —— 算法层：硬件无关，还是 tl.dot/tl.load，不知道 warp/shared 长啥样
    ttgir  —— 调度层：绑硬件布局(#blocked/#mma/#shared)，决定数据怎么摆进 SM
    llir   —— LLVM IR：通用编译器中间层
    ptx    —— 机器层：真正的 GPU 汇编(mma.sync/cp.async/ld.global)

这个 lab 编译一个 kernel，把每层 dump 出来，grep 各层的"特征 token"，
看同一件事在不同层怎么表达——亲眼看"渐进式 lowering"。

用法：  ./run.sh lower
"""
import torch
import triton
import triton.language as tl


@triton.jit
def mma_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
               sam, sak, sbk, sbn, scm, scn,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * sam + offs_k[None, :] * sak
    b_ptrs = b_ptr + offs_k[:, None] * sbk + offs_n[None, :] * sbn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        acc += tl.dot(tl.load(a_ptrs), tl.load(b_ptrs))
        a_ptrs += BLOCK_K * sak; b_ptrs += BLOCK_K * sbk
    c_ptrs = c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))


def count(text, *tokens):
    return {t: text.count(t) for t in tokens}


def main():
    M = N = K = 1024
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.empty(M, N, device="cuda", dtype=torch.float16)
    grid = (M // 128, N // 128)
    compiled = mma_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, num_warps=4, num_stages=3,
    )
    asm = compiled.asm

    print("## 同一个 kernel 的渐进式 lowering（每层只管一件事）\n")
    stages = [
        ("ttir", "算法层：硬件无关，还是 tt.dot/tt.load"),
        ("ttgir", "调度层：绑硬件布局 #blocked/#mma/#shared"),
        ("llir", "LLVM IR：通用中间层"),
        ("ptx", "机器层：GPU 汇编 mma.sync/cp.async"),
    ]
    for stage, desc in stages:
        n = len(asm[stage].splitlines()) if stage in asm else 0
        print(f"[{stage:5s}] {n:5d} 行  {desc}")

    print("\n## 特征 token 在各层的出现（看抽象怎么一层层落到硬件）")
    ttir, ttgir, ptx = asm["ttir"], asm["ttgir"], asm["ptx"]
    print(f"  tt.dot     (算法层的矩阵乘原语)  ttir={ttir.count('tt.dot'):3d}  ttgir={ttgir.count('tt.dot'):3d}")
    print(f"  #mma 布局  (绑 tensor core 的布局) ttir={ttir.count('#mma'):3d}  ttgir={ttgir.count('mma'):3d}   ← 调度层才出现")
    print(f"  #shared    (shared memory 布局)   ttir={ttir.count('#shared'):3d}  ttgir={ttgir.count('shared'):3d}   ← 调度层才出现")
    nmma = ptx.count("mma.sync")
    ncp = ptx.count("cp.async")
    print(f"  mma.sync   (真·tensor core 指令)  仅 ptx={nmma:3d}   ← 一路降到机器层才落地")
    print(f"  cp.async   (异步搬运指令)         仅 ptx={ncp:3d}   ← 同上")
    print("\n结论：tt.dot 一个高层原语，过 ttgir 绑上 #mma/#shared 布局，最终在 ptx 摊成"
          f" {nmma} 条 mma.sync + {ncp} 条 cp.async。每层 dialect 只下降一个抽象台阶——这就是渐进式 lowering。")


if __name__ == "__main__":
    main()
