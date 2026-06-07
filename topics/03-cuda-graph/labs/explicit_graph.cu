// =============================================================
// CUDA Graph 进阶：手动建图 = 显式表达依赖 DAG
//
// 上一课用 stream capture 录"一条线"。这课手动建图(cudaGraphAddKernelNode)，
// 搭一个菱形 DAG，体会"graph 本质是节点(kernel)+边(依赖)的有向无环图"：
//        A (x=1)
//       /     \
//      B        C       ← B、C 只依赖 A、彼此无边 → 独立可并行
//   y1=x*2   y2=x+5
//       \     /
//          D (z=y1+y2)   ← 依赖 B、C
// 期望结果 z = 2 + 6 = 8。
// =============================================================
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t e=call; if(e!=cudaSuccess){          \
    fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));\
    exit(1);} } while(0)

__global__ void initK(float* x, int n, float v){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]=v;
}
__global__ void mulK(const float* x, float* y, int n, float s){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]=x[i]*s;
}
__global__ void addK(const float* x, float* y, int n, float a){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]=x[i]+a;
}
__global__ void combineK(const float* a, const float* b, float* z, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) z[i]=a[i]+b[i];
}

int main(){
    const int n = 1<<20;
    size_t bytes = n*sizeof(float);
    float *d_x,*d_y1,*d_y2,*d_z;
    CUDA_CHECK(cudaMalloc(&d_x,bytes)); CUDA_CHECK(cudaMalloc(&d_y1,bytes));
    CUDA_CHECK(cudaMalloc(&d_y2,bytes)); CUDA_CHECK(cudaMalloc(&d_z,bytes));
    dim3 block(256), grid((n+255)/256);
    float one=1.0f, two=2.0f, five=5.0f;   // kernel 标量参数(需在建节点时存活)

    // ---- 手动建图 ----
    cudaGraph_t graph; CUDA_CHECK(cudaGraphCreate(&graph,0));
    cudaGraphNode_t nA,nB,nC,nD;

    // 节点 A：x=1，无依赖
    { void* args[]={&d_x,(void*)&n,&one};
      cudaKernelNodeParams p={}; p.func=(void*)initK; p.gridDim=grid; p.blockDim=block;
      p.kernelParams=args; p.extra=nullptr;
      CUDA_CHECK(cudaGraphAddKernelNode(&nA,graph,nullptr,0,&p)); }   // 0 个依赖

    // 节点 B：y1=x*2，依赖 A
    { void* args[]={&d_x,&d_y1,(void*)&n,&two};
      cudaKernelNodeParams p={}; p.func=(void*)mulK; p.gridDim=grid; p.blockDim=block;
      p.kernelParams=args; p.extra=nullptr;
      cudaGraphNode_t deps[]={nA};
      CUDA_CHECK(cudaGraphAddKernelNode(&nB,graph,deps,1,&p)); }      // 依赖 {A}

    // 节点 C：y2=x+5，依赖 A（和 B 无边 → 独立）
    { void* args[]={&d_x,&d_y2,(void*)&n,&five};
      cudaKernelNodeParams p={}; p.func=(void*)addK; p.gridDim=grid; p.blockDim=block;
      p.kernelParams=args; p.extra=nullptr;
      cudaGraphNode_t deps[]={nA};
      CUDA_CHECK(cudaGraphAddKernelNode(&nC,graph,deps,1,&p)); }      // 依赖 {A}

    // 节点 D：z=y1+y2，依赖 B 和 C
    { void* args[]={&d_y1,&d_y2,&d_z,(void*)&n};
      cudaKernelNodeParams p={}; p.func=(void*)combineK; p.gridDim=grid; p.blockDim=block;
      p.kernelParams=args; p.extra=nullptr;
      cudaGraphNode_t deps[]={nB,nC};
      CUDA_CHECK(cudaGraphAddKernelNode(&nD,graph,deps,2,&p)); }      // 依赖 {B,C}

    // 查一下图里有几个节点
    size_t numNodes=0; CUDA_CHECK(cudaGraphGetNodes(graph,nullptr,&numNodes));
    printf("图里有 %zu 个节点 (A,B,C,D)\n", numNodes);

    // 实例化 + 重放
    cudaGraphExec_t gexec;
    CUDA_CHECK(cudaGraphInstantiate(&gexec,graph,nullptr,nullptr,0));
    CUDA_CHECK(cudaGraphLaunch(gexec,0));
    CUDA_CHECK(cudaDeviceSynchronize());

    // 验证 z 全是 8
    float* h=(float*)malloc(bytes);
    CUDA_CHECK(cudaMemcpy(h,d_z,bytes,cudaMemcpyDeviceToHost));
    int err=0; for(int i=0;i<n;i++) if(fabsf(h[i]-8.0f)>1e-3f) err++;
    printf("结果验证: z = %.1f (期望 8.0) → %s\n", h[0], err? "FAIL":"PASS");

    free(h);
    cudaGraphExecDestroy(gexec); cudaGraphDestroy(graph);
    cudaFree(d_x); cudaFree(d_y1); cudaFree(d_y2); cudaFree(d_z);
    return 0;
}
