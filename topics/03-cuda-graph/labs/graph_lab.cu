// =============================================================
// CUDA Graph 入门：消除 launch bound
//
// 承接 topic 01 的 launch-bound 演示(200 个 tiny kernel，GPU 时间线全是缝)。
// 场景：每"轮"连发 CHAIN 个极小 kernel，共跑 ITERS 轮。
//   [A] baseline —— 循环里一个个 cudaLaunchKernel(每个都有 CPU 下发开销)
//   [B] graph    —— 把一轮录成 graph，之后每轮只 cudaGraphLaunch 一次重放
// 工作量完全相同，只差"下发方式"，公平对比 launch 开销。
// =============================================================
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#define CUDA_CHECK(call) do { cudaError_t e=call; if(e!=cudaSuccess){          \
    fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));\
    exit(1);} } while(0)

#define CHAIN 20      // 每轮连发的小 kernel 数
#define ITERS 2000    // 轮数
#define NELEM 1024    // 每个 kernel 处理的元素(故意小 → 单个 kernel 几微秒)

// 极小 kernel：x[i] += 1（工作量小，凸显 launch 开销）
__global__ void tiny(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += 1.0f;
}

int main() {
    printf("=== CUDA Graph: %d kernels/轮 × %d 轮 = %d 次 kernel ===\n",
           CHAIN, ITERS, CHAIN * ITERS);
    float* d; CUDA_CHECK(cudaMalloc(&d, NELEM * sizeof(float)));
    CUDA_CHECK(cudaMemset(d, 0, NELEM * sizeof(float)));
    cudaStream_t s; CUDA_CHECK(cudaStreamCreate(&s));
    dim3 block(256), grid((NELEM + 255) / 256);

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1); float ms;

    // ---------- [A] baseline：逐个 launch ----------
    // 预热
    for (int k = 0; k < CHAIN; k++) tiny<<<grid, block, 0, s>>>(d, NELEM);
    CUDA_CHECK(cudaStreamSynchronize(s));

    nvtxRangePushA("A_baseline_launch");
    cudaEventRecord(t0, s);
    for (int it = 0; it < ITERS; it++)
        for (int k = 0; k < CHAIN; k++)
            tiny<<<grid, block, 0, s>>>(d, NELEM);
    cudaEventRecord(t1, s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    cudaEventElapsedTime(&ms, t0, t1);
    nvtxRangePop();
    double us_per = ms * 1000.0 / (CHAIN * ITERS);
    printf("[A] baseline : %8.3f ms (%.2f us/kernel)\n", ms, us_per);

    // ---------- [B] graph：录一轮 → 重放 ITERS 次 ----------
    // 用 stream capture 把一轮(CHAIN 个 kernel)录成 graph
    cudaGraph_t graph; cudaGraphExec_t gexec;
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int k = 0; k < CHAIN; k++) tiny<<<grid, block, 0, s>>>(d, NELEM);
    CUDA_CHECK(cudaStreamEndCapture(s, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&gexec, graph, nullptr, nullptr, 0));
    // 预热
    CUDA_CHECK(cudaGraphLaunch(gexec, s));
    CUDA_CHECK(cudaStreamSynchronize(s));

    nvtxRangePushA("B_graph_replay");
    cudaEventRecord(t0, s);
    for (int it = 0; it < ITERS; it++)
        CUDA_CHECK(cudaGraphLaunch(gexec, s));   // 一条调用重放整轮 CHAIN 个 kernel
    cudaEventRecord(t1, s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    cudaEventElapsedTime(&ms, t0, t1);
    nvtxRangePop();
    double us_per_g = ms * 1000.0 / (CHAIN * ITERS);
    printf("[B] graph    : %8.3f ms (%.2f us/kernel)\n", ms, us_per_g);
    printf(">>> 加速 %.2fx，每 kernel 省 %.2f us 的 launch 开销\n",
           us_per / us_per_g, us_per - us_per_g);

    cudaGraphExecDestroy(gexec); cudaGraphDestroy(graph);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaStreamDestroy(s); cudaFree(d);
    return 0;
}
