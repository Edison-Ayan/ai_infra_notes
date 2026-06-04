// =============================================================
// nsys 学习实验台 (gemm_lab)
//
// 目的：用一个程序覆盖 nsys 分析里最常见的几种现象，
//       配合 NVTX 标记，让时间线一眼能读懂。
//
// 程序分 5 个阶段，每段用 NVTX range 包起来：
//   [1] H2D_copy          —— host→device 数据搬运
//   [2] naive_gemm        —— 朴素矩阵乘 (访存差)
//   [3] tiled_gemm        —— shared memory 分块 (访存优化)
//   [4] launch_bound_demo —— 200 个极小 kernel (演示 launch bound)
//   [5] D2H_copy          —— device→host 取回结果
//
// 编译/运行/profile 见同目录 README 或对话里的步骤。
// =============================================================
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>   // NVTX：给时间线打标记（header-only，无需链接）

// ---- 错误检查宏 ----
#define CUDA_CHECK(call) do {                                              \
    cudaError_t e = call;                                                  \
    if (e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
                __FILE__, __LINE__, cudaGetErrorString(e));                \
        exit(1);                                                           \
    } } while (0)

// ---- NVTX 便捷宏：进入/退出一个命名区间 ----
#define NVTX_PUSH(name) nvtxRangePushA(name)
#define NVTX_POP()      nvtxRangePop()

#define TILE 16   // tiled kernel 的分块大小，block = 16×16 = 256 线程

// =============================================================
// [2] Naive GEMM：每个线程算 C 的一个元素
//     每个元素都要从 global memory 读一整行 A + 一整列 B
//     => 访存次数 O(N) per output，算访比极低
// =============================================================
__global__ void gemm_naive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// =============================================================
// [3] Tiled GEMM：block 协作把 A/B 的 tile 搬进 shared memory，
//     tile 内复用，把 global memory 访存降到 O(N/TILE) per output
//     —— 这是手写 GEMM 的第一档正经优化
// =============================================================
__global__ void gemm_tiled(const float* A, const float* B, float* C, int N) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (N + TILE - 1) / TILE; t++) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] = (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();                       // 等全 block 加载完
        for (int k = 0; k < TILE; k++)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();                       // 等全 block 用完再换 tile
    }
    if (row < N && col < N) C[row * N + col] = sum;
}

// =============================================================
// [4] 一个极小的 kernel：只给数组每个元素 +1
//     单次执行只要几微秒，但 launch 一次有固定的 CPU 开销。
//     连launch 200 次 => GPU 时间线上全是空隙 = launch bound。
// =============================================================
__global__ void tiny_add(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += 1.0f;
}

// ---- CPU 参考实现，用于验证正确性 ----
void gemm_cpu(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < N; k++) s += A[i * N + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

int main() {
    const int N = 2048;                        // 矩阵边长
    printf("=== gemm_lab: C[%d x %d] = A x B (FP32) ===\n", N, N);
    size_t bytes = (size_t)N * N * sizeof(float);

    // ---- host 内存 + 初始化 ----
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)(rand() % 10) / 10.0f;
        h_B[i] = (float)(rand() % 10) / 10.0f;
    }

    // ---- device 内存 ----
    float *d_A, *d_B, *d_C, *d_tiny;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMalloc(&d_tiny, 1024 * sizeof(float)));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    float ms;

    // =========================================================
    // [1] H2D 拷贝
    // =========================================================
    NVTX_PUSH("1_H2D_copy");
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    NVTX_POP();

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    // =========================================================
    // [2] Naive GEMM（含一次预热，避免把 context 初始化算进来）
    // =========================================================
    gemm_naive<<<grid, block>>>(d_A, d_B, d_C, N);   // 预热
    CUDA_CHECK(cudaDeviceSynchronize());

    NVTX_PUSH("2_naive_gemm");
    cudaEventRecord(t0);
    gemm_naive<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(t1);
    CUDA_CHECK(cudaEventSynchronize(t1));
    cudaEventElapsedTime(&ms, t0, t1);
    NVTX_POP();
    printf("[2] naive_gemm : %8.3f ms | %.2f TFLOPS\n",
           ms, 2.0 * N * N * N / (ms * 1e-3) / 1e12);

    // =========================================================
    // [3] Tiled GEMM
    // =========================================================
    NVTX_PUSH("3_tiled_gemm");
    cudaEventRecord(t0);
    gemm_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(t1);
    CUDA_CHECK(cudaEventSynchronize(t1));
    cudaEventElapsedTime(&ms, t0, t1);
    NVTX_POP();
    printf("[3] tiled_gemm : %8.3f ms | %.2f TFLOPS\n",
           ms, 2.0 * N * N * N / (ms * 1e-3) / 1e12);

    // =========================================================
    // [4] launch bound 演示：连发 200 个极小 kernel
    //     每个只处理 1024 个元素，GPU 干活几微秒，
    //     但 launch 开销 + 串行下发 => 时间线上全是空隙
    // =========================================================
    NVTX_PUSH("4_launch_bound_demo");
    cudaEventRecord(t0);
    for (int i = 0; i < 200; i++)
        tiny_add<<<4, 256>>>(d_tiny, 1024);
    cudaEventRecord(t1);
    CUDA_CHECK(cudaEventSynchronize(t1));
    cudaEventElapsedTime(&ms, t0, t1);
    NVTX_POP();
    printf("[4] 200x tiny  : %8.3f ms (%.1f us/kernel, 几乎全是 launch 开销)\n",
           ms, ms * 1000.0 / 200);

    // =========================================================
    // [5] D2H 取回
    // =========================================================
    NVTX_PUSH("5_D2H_copy");
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    NVTX_POP();

    // ---- 正确性验证：用一个小角块 256x256 抽查（全量 CPU 太慢）----
    // 这里用 tiled 的结果 h_C 和 CPU 重算的左上 256x256 对比
    int S = 256;
    float *ref = (float*)malloc((size_t)S * S * sizeof(float));
    for (int i = 0; i < S; i++)
        for (int j = 0; j < S; j++) {
            float s = 0;
            for (int k = 0; k < N; k++) s += h_A[i * N + k] * h_B[k * N + j];
            ref[i * S + j] = s;
        }
    int errors = 0;
    for (int i = 0; i < S; i++)
        for (int j = 0; j < S; j++)
            if (fabsf(ref[i * S + j] - h_C[i * N + j]) > 1e-1f) errors++;
    printf("[5] correctness: %s (抽查左上 %dx%d)\n",
           errors == 0 ? "PASS" : "FAIL", S, S);

    free(ref);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_tiny);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
