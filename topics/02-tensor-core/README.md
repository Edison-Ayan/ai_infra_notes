# 02 · Tensor Core GEMM

接 topic 01：手写 FP32 GEMM 的甜点止步 float4=6.13 TFLOPS（cuBLAS-FP32=6.95，硬件 FP32 峰值 ~12）。
要再往上得**换赛道**——用 Tensor Core（专门的矩阵乘单元，FP16/TF32 算力是 FP32 CUDA core 的数倍）。
这是 FlashAttention 等 AI infra 算子的真实底座。

环境：RTX 4060 Laptop (sm_89, Ada, 第 4 代 Tensor Core)、CUDA 12.9。

## labs

| 文件 | 内容 |
|---|---|
| [labs/wmma_gemm.cu](labs/wmma_gemm.cu) | WMMA 入门：FP16 输入 / FP32 累加，每 warp 算一个 16×16 tile（**naive，无 shared 复用**） |

```bash
cd labs && ./build.sh && ./wmma_gemm
```

## 实验①：WMMA 入门——"用了 Tensor Core ≠ 快"

WMMA(Warp Matrix Multiply-Accumulate)是最易上手的 Tensor Core API：
- 以 **warp(32线程)** 为单位协作算一个 16×16 的 C tile；
- 三个 `fragment`（matrix_a / matrix_b / accumulator），数据在 warp 内如何分布由库隐藏；
- `load_matrix_sync` 载入片 → `mma_sync` 一条指令算 16×16×16 → `store_matrix_sync` 写回；
- 输入 `half`、累加 `float`——精度换吞吐的经典配置。

**结果却比 FP32 的 float4 还慢**：

| | float4 (FP32) | **wmma (naive)** |
|---|---|---|
| TFLOPS | 6.13 | **5.22** |
| Compute(SM) 吞吐 | 58.9% | **36.75%** |
| Memory Throughput | 82.9% | **98.94%**(L1 load 路径) |
| DRAM 吞吐 | — | 1.96% |

**为什么慢**：这个 naive WMMA **每个 K 步都直接从 global 载入 A/B 片，零 shared 复用**。和 topic 01 的 naive FP32 一模一样的病——**访存(L1 load)管线打满(99%)、Tensor Core 饿死(SM 仅 37%)**。DRAM 才 2%，说明数据在 cache 里反复搬，瓶颈是 load 指令吞吐不是显存带宽。

> **核心认知**：Tensor Core 只是把"算"变快了，但**喂数据的内存层级优化一个都不能少**。光调 `mma_sync` 不上 shared tiling，等于买了跑车却堵在小区门口。topic 01 那套（shared 分块 → 寄存器复用 → float4）在 Tensor Core 上要重做一遍来喂饱它。

## 下一步

- [ ] shared memory 分块的 WMMA（把 A/B tile 搬进 shared 复用，喂饱 Tensor Core）
- [ ] 对标 cuBLAS 的 Tensor Core 路径（`cublasGemmEx` + `CUBLAS_COMPUTE_32F` / TF32）
- [ ] TF32 路径（精度比 FP16 高、改动小）；以及更底层的 `mma` PTX / `ldmatrix`
