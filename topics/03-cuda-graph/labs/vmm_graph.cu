// =============================================================
// CUDA VMM 保护 CUDA Graph —— torch_memory_saver 的核心机制 demo
//
// 痛点：CUDA Graph 在捕获时把显存"指针(虚拟地址)"焊死。普通 cudaFree/cudaMalloc
//       会让地址变 → graph 失效 → 必须重新捕获(慢)。
// 解法：用 CUDA 虚拟内存管理(VMM)把"虚拟地址"和"物理显存"解耦：
//       让出显存时只还物理页、保留虚拟地址；恢复时把新物理页映射回同一地址。
//       指针不变 → graph 无感 → 不用重新捕获就能复活。
//
// 本 demo 流程：
//   建 VMM 显存(VA固定) → 捕获写它的 graph → 重放✓
//   → pause(释放物理显存、保留VA) → resume(新物理显存映射回同一VA)
//   → memset 清零(证明物理是新的) → 不重新捕获、直接重放同一张 graph → 仍✓
// =============================================================
#include <cstdio>
#include <cuda.h>            // driver API: cuMem*
#include <cuda_runtime.h>

#define CU_CHECK(call) do { CUresult r=call; if(r!=CUDA_SUCCESS){              \
    const char* s; cuGetErrorString(r,&s);                                    \
    fprintf(stderr,"CU %s:%d: %s\n",__FILE__,__LINE__,s); exit(1);} } while(0)
#define RT_CHECK(call) do { cudaError_t e=call; if(e!=cudaSuccess){            \
    fprintf(stderr,"RT %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
    exit(1);} } while(0)

__global__ void writeK(float* p, float val, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) p[i]=val;
}

int main(){
    RT_CHECK(cudaSetDevice(0));      // 先建立 runtime 的 primary context
    CU_CHECK(cuInit(0));
    CUdevice dev; CU_CHECK(cuDeviceGet(&dev,0));

    const int n = 1<<18;             // 256K floats = 1MB
    size_t bytes = n*sizeof(float);

    // VMM 分配属性：device pinned 物理显存
    CUmemAllocationProp prop={};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dev;
    size_t gran;
    CU_CHECK(cuMemGetAllocationGranularity(&gran,&prop,CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    size_t size = ((bytes+gran-1)/gran)*gran;   // 对齐到粒度

    CUmemAccessDesc adesc={};
    adesc.location = prop.location;
    adesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    // ---- 1. 预留虚拟地址(VA) ----
    CUdeviceptr ptr;
    CU_CHECK(cuMemAddressReserve(&ptr,size,0,0,0));
    printf("预留虚拟地址: %p\n", (void*)ptr);

    // ---- 2. 建物理显存 + 映射到 VA + 开权限 ----
    CUmemGenericAllocationHandle h1;
    CU_CHECK(cuMemCreate(&h1,size,&prop,0));
    CU_CHECK(cuMemMap(ptr,size,0,h1,0));
    CU_CHECK(cuMemSetAccess(ptr,size,&adesc,1));
    float* p = (float*)ptr;          // 当普通设备指针用

    // ---- 3. 捕获一张 graph：写 p 全为 7.0 ----
    cudaStream_t s; RT_CHECK(cudaStreamCreate(&s));
    dim3 block(256), grid((n+255)/256);
    cudaGraph_t graph; cudaGraphExec_t gexec;
    RT_CHECK(cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal));
    writeK<<<grid,block,0,s>>>(p,7.0f,n);
    RT_CHECK(cudaStreamEndCapture(s,&graph));
    RT_CHECK(cudaGraphInstantiate(&gexec,graph,nullptr,nullptr,0));

    auto replay_and_check = [&](const char* tag){
        RT_CHECK(cudaGraphLaunch(gexec,s));
        RT_CHECK(cudaStreamSynchronize(s));
        float v; RT_CHECK(cudaMemcpy(&v,p,sizeof(float),cudaMemcpyDeviceToHost));
        printf("  %s: p[0]=%.1f → %s\n", tag, v, v==7.0f?"OK":"WRONG");
    };

    printf("捕获 graph 后首次重放:\n");
    replay_and_check("replay#1");

    // ---- 4. pause：释放物理显存，但保留 VA ----
    CU_CHECK(cuMemUnmap(ptr,size));
    CU_CHECK(cuMemRelease(h1));       // 物理显存还给系统(这 1MB 可被别人用)
    printf("\n[pause] 已释放物理显存(cuMemUnmap+cuMemRelease)，VA %p 仍保留\n",(void*)ptr);

    // ---- 5. resume：建新物理显存，映射回同一个 VA ----
    CUmemGenericAllocationHandle h2;
    CU_CHECK(cuMemCreate(&h2,size,&prop,0));
    CU_CHECK(cuMemMap(ptr,size,0,h2,0));   // 注意：还是 ptr，地址不变！
    CU_CHECK(cuMemSetAccess(ptr,size,&adesc,1));
    printf("[resume] 新物理显存已映射回同一 VA %p (地址%s变)\n",
           (void*)ptr, (void*)p==(void*)ptr?"未":"已");

    // ---- 6. 清零(证明物理是全新的)，再不重新捕获、直接重放同一张 graph ----
    RT_CHECK(cudaMemset(p,0,bytes));
    float z; RT_CHECK(cudaMemcpy(&z,p,sizeof(float),cudaMemcpyDeviceToHost));
    printf("  清零后 p[0]=%.1f (确认是新物理显存)\n", z);
    printf("不重新捕获，直接重放原 graph:\n");
    replay_and_check("replay#2");
    printf("\n>>> 物理显存换过一轮，graph 未重新捕获仍正确 → VA 不变即可保护 graph\n");

    RT_CHECK(cudaGraphExecDestroy(gexec)); RT_CHECK(cudaGraphDestroy(graph));
    CU_CHECK(cuMemUnmap(ptr,size)); CU_CHECK(cuMemRelease(h2));
    CU_CHECK(cuMemAddressFree(ptr,size));
    RT_CHECK(cudaStreamDestroy(s));
    return 0;
}
