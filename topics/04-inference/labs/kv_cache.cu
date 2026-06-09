// =============================================================
// 推理第②课：KV cache —— 为什么必须存它 + 显存账
//
// decode 每生成 1 个 token，要对"之前所有 token"做 attention，需要它们的 K/V。
//   不存：每步都重算前面所有 token 的 K/V → 计算量 O(N²)
//   存(KV cache)：每步只算新 token 的 K/V，旧的复用 → O(N)，但显存随序列线性涨
// Part A: 算 KV cache 显存随 (batch, seqlen) 怎么涨、何时超 8GB。
// Part B: 实测 K/V 投影 GEMM "重算整段 vs 只算新token" 的耗时差(用 cuBLAS)。
// =============================================================
#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#define CK(x) do{ cudaError_t e=(x); if(e){printf("CUDA %s\n",cudaGetErrorString(e));return 1;} }while(0)

int main(){
    // ---------- Part A: KV cache 显存账 (Llama-7B 量级) ----------
    int L=32, D=4096;                       // 32 层，hidden 4096 (MHA: num_kv_heads*head_dim=D)
    double perTok = 2.0*L*D*2;              // 2(K&V) × 层 × D × 2字节(FP16)，单位 byte/token
    printf("=== Part A: KV cache 显存账 (L=%d, D=%d, FP16) ===\n", L, D);
    printf("每 token 的 KV cache = 2×%d×%d×2B = %.2f MB\n\n", L, D, perTok/1e6);
    printf("%8s | %8s | %10s | %s\n","batch","seqlen","KV cache","vs 8GB 卡");
    printf("---------|----------|------------|--------\n");
    int Bs[]={1,1,1,16,32,64}, Ss[]={2048,8192,32768,2048,2048,2048};
    for(int i=0;i<6;i++){
        double gb = perTok*Bs[i]*Ss[i]/1e9;
        printf("%8d | %8d | %7.2f GB | %s\n", Bs[i], Ss[i], gb, gb>8?"爆!":"ok");
    }

    // ---------- Part B: 重算整段 vs 只算新 token ----------
    printf("\n=== Part B: K/V 投影 GEMM 计算量 (D=%d, 生成 T token) ===\n", D);
    const int T=512;
    cublasHandle_t h; cublasCreate(&h); cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH);
    half *Wk,*X,*C;
    CK(cudaMalloc(&Wk,(size_t)D*D*2)); CK(cudaMalloc(&X,(size_t)D*T*2)); CK(cudaMalloc(&C,(size_t)D*T*2));
    CK(cudaMemset(Wk,1,(size_t)D*D*2)); CK(cudaMemset(X,1,(size_t)D*T*2));
    half al=__float2half(1.f), be=__float2half(0.f);
    auto gemm=[&](int n){ // C[D×n] = Wk[D×D] @ X[D×n]，FLOP=2·D·D·n
        cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,D,n,D,&al,Wk,CUDA_R_16F,D,X,CUDA_R_16F,D,
                     &be,C,CUDA_R_16F,D,CUBLAS_COMPUTE_16F,CUBLAS_GEMM_DEFAULT_TENSOR_OP); };
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1); float ms;

    // 不存：第 t 步重算整段 prefix 的 K/V (n=t)
    gemm(1); CK(cudaDeviceSynchronize());
    cudaEventRecord(t0);
    for(int t=1;t<=T;t++) gemm(t);
    cudaEventRecord(t1); CK(cudaEventSynchronize(t1)); cudaEventElapsedTime(&ms,t0,t1);
    double flopNo=0; for(int t=1;t<=T;t++) flopNo+=2.0*D*D*t;
    printf("不存KV(每步重算整段): %7.2f ms  | 总计算 %.1f TFLOP (∝T²)\n", ms, flopNo/1e12);

    // 存：第 t 步只算新 token 的 K/V (n=1)
    cudaEventRecord(t0);
    for(int t=1;t<=T;t++) gemm(1);
    cudaEventRecord(t1); CK(cudaEventSynchronize(t1)); cudaEventElapsedTime(&ms,t0,t1);
    double flopYes=2.0*D*D*T;
    printf("存KV(每步只算新token): %7.2f ms  | 总计算 %.1f TFLOP (∝T)\n", ms, flopYes/1e12);
    printf(">>> 不存的计算量是存的 %.0f× (T=%d)\n", flopNo/flopYes, T);

    cublasDestroy(h); cudaFree(Wk); cudaFree(X); cudaFree(C);
    return 0;
}
