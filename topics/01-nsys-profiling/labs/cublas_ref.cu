// cuBLAS SGEMM 参照基准：和 gemm_lab 同尺寸 (N=2048, FP32)
// 作为手写 GEMM 追赶的 "Speed of Light"。
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CK(x) do{ cudaError_t e=(x); if(e){printf("CUDA %s\n",cudaGetErrorString(e));exit(1);} }while(0)

int main(){
    const int N = 2048;
    size_t bytes = (size_t)N*N*sizeof(float);
    float *dA,*dB,*dC;
    CK(cudaMalloc(&dA,bytes)); CK(cudaMalloc(&dB,bytes)); CK(cudaMalloc(&dC,bytes));
    // 随便填点数（perf 测试不关心具体值）
    CK(cudaMemset(dA,1,bytes)); CK(cudaMemset(dB,1,bytes));

    cublasHandle_t h; cublasCreate(&h);
    float alpha=1.0f, beta=0.0f;
    // 行主序 C=A*B 用列主序库的标准写法：传 (B,A) 交换
    auto run=[&](){ cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,
                                &alpha,dB,N,dA,N,&beta,dC,N); };

    for(int i=0;i<5;i++) run();          // 预热
    CK(cudaDeviceSynchronize());

    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    const int iters=50;
    cudaEventRecord(t0);
    for(int i=0;i<iters;i++) run();
    cudaEventRecord(t1); CK(cudaEventSynchronize(t1));
    float ms; cudaEventElapsedTime(&ms,t0,t1); ms/=iters;

    printf("cuBLAS SGEMM N=%d : %8.3f ms | %.2f TFLOPS\n",
           N, ms, 2.0*N*N*N/(ms*1e-3)/1e12);
    cublasDestroy(h);
    cudaFree(dA);cudaFree(dB);cudaFree(dC);
    return 0;
}
