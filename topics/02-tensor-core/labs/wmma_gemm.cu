// =============================================================
// Tensor Core 入门：WMMA GEMM (FP16 输入 / FP32 累加)
//
// 对照 topic 01 的手写 FP32 GEMM(最优 float4=6.13 TFLOPS、cuBLAS-FP32=6.95)，
// 这里换赛道用 Tensor Core：不再让 CUDA core 一个一个算 FFMA，
// 而是用专门的矩阵乘单元，一条 mma 指令直接算一个 16×16×16 的小矩阵乘。
//
// WMMA(Warp Matrix Multiply-Accumulate)是最易上手的 Tensor Core API：
//   - 以 warp(32线程)为单位协作，算一个 16×16 的 C 输出 tile
//   - fragment 是 warp 内分布存储的矩阵片，细节由库隐藏
//   - 输入 half(FP16)，累加 float(FP32) —— 精度换吞吐的经典配置
// =============================================================
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <nvtx3/nvToolsExt.h>
using namespace nvcuda;

#define CUDA_CHECK(call) do {                                              \
    cudaError_t e = call;                                                  \
    if (e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);       \
    } } while (0)

// WMMA tile 尺寸：FP16 下的基本单元是 16×16×16 (M×N×K)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// 每个 warp 负责 C 的一个 16×16 tile
__global__ void gemm_wmma(const half* A, const half* B, float* C, int N) {
    // 该线程属于哪个 warp，进而对应 C 的哪个 16×16 tile
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // 三个 fragment：A 片、B 片、累加器
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // 沿 K 方向，每次吃一个 16 宽的片，累加进 c_frag
    for (int k = 0; k < N; k += WMMA_K) {
        int aRow = warpM * WMMA_M, aCol = k;
        int bRow = k,             bCol = warpN * WMMA_N;
        if (aRow < N && bCol < N) {
            wmma::load_matrix_sync(a_frag, A + aRow * N + aCol, N);  // 从 global 载入 A 片
            wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);  // 载入 B 片
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);          // 一条指令算 16×16×16
        }
    }
    int cRow = warpM * WMMA_M, cCol = warpN * WMMA_N;
    if (cRow < N && cCol < N)
        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
}

int main() {
    const int N = 2048;
    printf("=== wmma_gemm: C[%d x %d] = A x B (FP16 in / FP32 acc) ===\n", N, N);
    size_t fbytes = (size_t)N * N * sizeof(float);
    size_t hbytes = (size_t)N * N * sizeof(half);

    // host：用 float 初始化，再转 half
    float *h_A = (float*)malloc(fbytes);
    float *h_B = (float*)malloc(fbytes);
    float *h_C = (float*)malloc(fbytes);
    half  *h_Ah = (half*)malloc(hbytes);
    half  *h_Bh = (half*)malloc(hbytes);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)(rand() % 10) / 10.0f;
        h_B[i] = (float)(rand() % 10) / 10.0f;
        h_Ah[i] = __float2half(h_A[i]);
        h_Bh[i] = __float2half(h_B[i]);
    }

    half *d_A, *d_B; float *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, hbytes));
    CUDA_CHECK(cudaMalloc(&d_B, hbytes));
    CUDA_CHECK(cudaMalloc(&d_C, fbytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_Ah, hbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_Bh, hbytes, cudaMemcpyHostToDevice));

    // launch：block=(128,4)=512线程=16 warp；每 warp 一个 16×16 tile → block 算 64×64
    dim3 block(128, 4);
    dim3 grid((N / WMMA_M + 3) / 4, (N / WMMA_N + 3) / 4);

    gemm_wmma<<<grid, block>>>(d_A, d_B, d_C, N);   // 预热
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    float ms;
    nvtxRangePushA("wmma_gemm");
    cudaEventRecord(t0);
    gemm_wmma<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(t1);
    CUDA_CHECK(cudaEventSynchronize(t1));
    cudaEventElapsedTime(&ms, t0, t1);
    nvtxRangePop();
    printf("[wmma] %8.3f ms | %.2f TFLOPS\n", ms, 2.0 * N * N * N / (ms * 1e-3) / 1e12);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, fbytes, cudaMemcpyDeviceToHost));

    // 正确性：抽查左上 256×256，CPU 用 float 重算，FP16 输入故用相对容差
    int S = 256, errors = 0;
    for (int i = 0; i < S; i++)
        for (int j = 0; j < S; j++) {
            float s = 0;
            for (int k = 0; k < N; k++) s += h_A[i * N + k] * h_B[k * N + j];
            float got = h_C[i * N + j];
            if (fabsf(got - s) > fabsf(s) * 0.02f + 0.5f) errors++;
        }
    printf("[chk] %s (抽查左上 %dx%d, 相对容差 2%%)\n",
           errors == 0 ? "PASS" : "FAIL", S, S);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_Ah); free(h_Bh);
    return 0;
}
