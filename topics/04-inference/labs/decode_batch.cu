// =============================================================
// 推理系统第一课：decode 是访存 bound，所以 batching 几乎白赚吞吐
//
// decode 每步：为产出 1 个 token，要把整个权重矩阵 W 从显存搬一遍。
//   B=1  : 搬 W(全部权重) 只出 1 token  → 算访比≈1 → 卡显存带宽
//   B=64 : 搬 W(同样一次) 出 64 token   → 权重搬运被 64 个 token 摊薄
// 模拟一层线性： Y[N×B] = W[N×K] @ X[K×B]，B = batch(同时解码的请求数)。
// 测不同 B 的耗时 → 看"每 token 耗时"随 batch 暴跌(吞吐白涨)。
// 用 FP16 输入 / FP32 累加 + tensor core(贴近真实推理)。
// =============================================================
#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#define CK(x) do{ cudaError_t e=(x); if(e){printf("CUDA %s\n",cudaGetErrorString(e));return 1;} }while(0)

int main(){
    const int N = 4096, K = 4096;          // 一层权重 4096×4096 (典型 hidden dim)
    cublasHandle_t h; cublasCreate(&h);
    cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH);

    size_t wbytes = (size_t)N*K*sizeof(half);
    printf("权重 W = %dx%d FP16 = %.1f MB；HBM 带宽 ~256 GB/s\n", N, K, wbytes/1e6);
    printf("理论上：只要还是访存 bound，搬一次 W ≈ %.0f us，与 batch 无关\n\n",
           wbytes/256e9*1e6);

    half *dW, *dX, *dY;
    int Bmax = 512;
    CK(cudaMalloc(&dW, wbytes));
    CK(cudaMalloc(&dX, (size_t)K*Bmax*sizeof(half)));
    CK(cudaMalloc(&dY, (size_t)N*Bmax*sizeof(half)));
    CK(cudaMemset(dW,1,wbytes)); CK(cudaMemset(dX,1,(size_t)K*Bmax*sizeof(half)));

    __half alpha=__float2half(1.f), beta=__float2half(0.f);
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);

    printf("%6s | %10s | %12s | %14s\n", "batch", "总耗时(us)", "每token(us)", "吞吐(token/ms)");
    printf("-------|------------|--------------|----------------\n");
    int batches[] = {1,2,4,8,16,32,64,128,256,512};
    for(int bi=0; bi<10; bi++){
        int B = batches[bi];
        auto run=[&](){ cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, B, K,
            &alpha, dW, CUDA_R_16F, N, dX, CUDA_R_16F, K,
            &beta, dY, CUDA_R_16F, N, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP); };
        for(int i=0;i<10;i++) run();                 // 预热
        CK(cudaDeviceSynchronize());
        const int iters=100;
        cudaEventRecord(t0);
        for(int i=0;i<iters;i++) run();
        cudaEventRecord(t1); CK(cudaEventSynchronize(t1));
        float ms; cudaEventElapsedTime(&ms,t0,t1); ms/=iters;
        double us=ms*1000.0;
        printf("%6d | %10.1f | %12.3f | %14.1f\n", B, us, us/B, B/ms);
    }
    cublasDestroy(h); cudaFree(dW); cudaFree(dX); cudaFree(dY);
    return 0;
}
