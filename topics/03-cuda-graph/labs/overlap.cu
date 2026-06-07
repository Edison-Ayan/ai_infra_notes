// =============================================================
// CUDA Graph 进阶：让独立节点"真并行"——补上上一课没看到的重叠
//
// 上一课 B、C 标了独立却没重叠，因为每个 kernel 1M 元素占满了 GPU。
// 这课用"自旋" kernel：只发 1 个 block(占 1 个 SM)、但 block 内空转一阵，
// 于是两个独立 kernel 各占 1 SM、GPU 有大把空位 → 能同时跑。
// 对比：
//   [1] 单个 spin            —— 基准时间 T
//   [2] 串行(一条流跑2次)     —— 约 2T
//   [3] graph 2个独立节点    —— 约 T (重叠!)
// =============================================================
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t e=call; if(e!=cudaSuccess){          \
    fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));\
    exit(1);} } while(0)

// 自旋 kernel：空转 iters 次(真依赖链 + 写回，防被优化掉)。只发 1 block → 只占 1 SM。
__global__ void spinK(float* sink, long iters){
    float acc = threadIdx.x * 0.001f;
    for(long i=0;i<iters;i++) acc = acc*1.0001f + 0.5f;
    if(threadIdx.x==0) sink[blockIdx.x] = acc;
}

static float time_stream_serial(float* sink, long iters, cudaStream_t s, int reps){
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0,s);
    for(int r=0;r<reps;r++) spinK<<<1,256,0,s>>>(sink+r, iters); // 同一条流 → 串行
    cudaEventRecord(t1,s); CUDA_CHECK(cudaEventSynchronize(t1));
    float ms; cudaEventElapsedTime(&ms,t0,t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1); return ms;
}

int main(){
    const long ITERS = 8'000'000;   // 调到单个 ~几 ms
    float* sink; CUDA_CHECK(cudaMalloc(&sink, 8*sizeof(float)));
    cudaStream_t s; CUDA_CHECK(cudaStreamCreate(&s));

    // 预热
    spinK<<<1,256,0,s>>>(sink,ITERS); CUDA_CHECK(cudaStreamSynchronize(s));

    // [1] 单个
    float t1 = time_stream_serial(sink, ITERS, s, 1);
    printf("[1] 单个 spin       : %7.3f ms  (基准 T)\n", t1);

    // [2] 串行：一条流跑 2 次 → 互相等
    float t2 = time_stream_serial(sink, ITERS, s, 2);
    printf("[2] 串行 2 次(一条流): %7.3f ms  (≈2T → 没重叠)\n", t2);

    // [3] graph：2 个独立节点(都无依赖) → 可并行
    cudaGraph_t g; CUDA_CHECK(cudaGraphCreate(&g,0));
    cudaGraphNode_t na,nb;
    float* s0=sink; float* s1=sink+1; long it=ITERS;
    { void* args[]={&s0,&it}; cudaKernelNodeParams p={};
      p.func=(void*)spinK; p.gridDim=dim3(1); p.blockDim=dim3(256);
      p.kernelParams=args; CUDA_CHECK(cudaGraphAddKernelNode(&na,g,nullptr,0,&p)); }
    { void* args[]={&s1,&it}; cudaKernelNodeParams p={};
      p.func=(void*)spinK; p.gridDim=dim3(1); p.blockDim=dim3(256);
      p.kernelParams=args; CUDA_CHECK(cudaGraphAddKernelNode(&nb,g,nullptr,0,&p)); }
    cudaGraphExec_t ge; CUDA_CHECK(cudaGraphInstantiate(&ge,g,nullptr,nullptr,0));
    CUDA_CHECK(cudaGraphLaunch(ge,s)); CUDA_CHECK(cudaStreamSynchronize(s)); // 预热

    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0,s); CUDA_CHECK(cudaGraphLaunch(ge,s)); cudaEventRecord(e1,s);
    CUDA_CHECK(cudaEventSynchronize(e1));
    float t3; cudaEventElapsedTime(&t3,e0,e1);
    printf("[3] graph 2独立节点 : %7.3f ms  (≈T → 重叠了!)\n", t3);
    printf(">>> 串行/并行 = %.2fx (越接近 2 越说明两个 kernel 真同时跑)\n", t2/t3);

    cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
    cudaStreamDestroy(s); cudaFree(sink);
    return 0;
}
