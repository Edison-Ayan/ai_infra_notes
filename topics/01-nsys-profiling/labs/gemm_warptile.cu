// =============================================================
// warp tiling：真正破"5-way load 结构性 bank conflict"的招
//
// 思路(siboehm kernel 10)：在 block tile→thread tile 之间多加一层 warp tile。
// 关键是重排"线程→列"的映射——一个 warp 的 32 lane 排成 8(行)×4(列)，
// 沿 N 只用 4 个 lane(threadColInWarp)、每次读 TN=4 → 16 个连续 float
// 正好铺满 16 个 bank、互不冲突；不再是 float4 版那种 16 lane 按 ×8 跨步撞 bank。
// 对比 v0(float4 baseline)看 5-way 能压到几 way。
// =============================================================
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#define CUDA_CHECK(call) do { cudaError_t e=call; if(e!=cudaSuccess){          \
    fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));\
    exit(1);} } while(0)

// ---------- v0：float4 baseline（256 线程，5-way load 冲突） ----------
#define B0M 128
#define B0N 128
#define B0K 8
#define T0M 8
#define T0N 8
__global__ void gemm_v0(const float* A, const float* B, float* C, int N) {
    const int cRow=blockIdx.y, cCol=blockIdx.x;
    __shared__ float As[B0K*B0M], Bs[B0K*B0N];
    const int tid=threadIdx.x, tCol=tid%(B0N/T0N), tRow=tid/(B0N/T0N);
    const float* Ap=A+cRow*B0M*N; const float* Bp=B+cCol*B0N;
    float* Cp=C+cRow*B0M*N+cCol*B0N;
    const int irA=tid/(B0K/4),icA=tid%(B0K/4),irB=tid/(B0N/4),icB=tid%(B0N/4);
    float acc[T0M*T0N]={0.0f},rM[T0M],rN[T0N];
    for(int bk=0;bk<N;bk+=B0K){
        float4 va=*reinterpret_cast<const float4*>(&Ap[irA*N+icA*4]);
        As[(icA*4+0)*B0M+irA]=va.x; As[(icA*4+1)*B0M+irA]=va.y;
        As[(icA*4+2)*B0M+irA]=va.z; As[(icA*4+3)*B0M+irA]=va.w;
        *reinterpret_cast<float4*>(&Bs[irB*B0N+icB*4])=
            *reinterpret_cast<const float4*>(&Bp[irB*N+icB*4]);
        __syncthreads(); Ap+=B0K; Bp+=B0K*N;
        for(int k=0;k<B0K;k++){
            for(int i=0;i<T0M;i+=4)*reinterpret_cast<float4*>(&rM[i])=
                *reinterpret_cast<float4*>(&As[k*B0M+tRow*T0M+i]);
            for(int j=0;j<T0N;j+=4)*reinterpret_cast<float4*>(&rN[j])=
                *reinterpret_cast<float4*>(&Bs[k*B0N+tCol*T0N+j]);
            for(int i=0;i<T0M;i++)for(int j=0;j<T0N;j++)acc[i*T0N+j]+=rM[i]*rN[j];
        }
        __syncthreads();
    }
    for(int i=0;i<T0M;i++)for(int j=0;j<T0N;j+=4){
        float4 v;v.x=acc[i*T0N+j];v.y=acc[i*T0N+j+1];v.z=acc[i*T0N+j+2];v.w=acc[i*T0N+j+3];
        *reinterpret_cast<float4*>(&Cp[(tRow*T0M+i)*N+tCol*T0N+j])=v;
    }
}

// ---------- warp tiling ----------
#define BM 128
#define BN 128
#define BK 8
#define WM 64          // warp tile：一个 warp 算 64×64
#define WN 64
#define WNITER 4       // warp 沿 N 分 4 个子块
#define WMITER 1       // = WM*WN/(32*TM*TN*WNITER)=4096/4096
#define TM 8
#define TN 4
#define NTHREADS 128   // 4 warp/block
#define WSUBM (WM/WMITER)   // 64
#define WSUBN (WN/WNITER)   // 16
__global__ void __launch_bounds__(NTHREADS)
gemm_warptile(const float* A, const float* B, float* C, int N) {
    const int cRow=blockIdx.y, cCol=blockIdx.x;
    const int warpIdx=threadIdx.x/32;
    const int warpCol=warpIdx%(BN/WN), warpRow=warpIdx/(BN/WN);     // BN/WN=2
    const int laneIdx=threadIdx.x%32;
    const int tColW=laneIdx%(WSUBN/TN);   // WSUBN/TN=4 → 0..3 (沿N只用4 lane)
    const int tRowW=laneIdx/(WSUBN/TN);   // 0..7

    __shared__ float As[BK*BM], Bs[BK*BN];
    const float* Ap=A+cRow*BM*N; const float* Bp=B+cCol*BN;
    float* Cp=C+cRow*BM*N+cCol*BN;
    const int irA=threadIdx.x/(BK/4), icA=threadIdx.x%(BK/4);
    const int strideA=(NTHREADS*4)/BK;    // 64
    const int irB=threadIdx.x/(BN/4), icB=threadIdx.x%(BN/4);
    const int strideB=NTHREADS/(BN/4);    // 4

    float acc[WMITER*TM*WNITER*TN]={0.0f}; // 1*8*4*4=128
    float rM[WMITER*TM], rN[WNITER*TN];    // 8, 16

    for(int bk=0;bk<N;bk+=BK){
        for(int o=0;o+strideA<=BM;o+=strideA){
            float4 t=*reinterpret_cast<const float4*>(&Ap[(irA+o)*N+icA*4]);
            As[(icA*4+0)*BM+irA+o]=t.x; As[(icA*4+1)*BM+irA+o]=t.y;
            As[(icA*4+2)*BM+irA+o]=t.z; As[(icA*4+3)*BM+irA+o]=t.w;
        }
        for(int o=0;o+strideB<=BK;o+=strideB)
            *reinterpret_cast<float4*>(&Bs[(irB+o)*BN+icB*4])=
                *reinterpret_cast<const float4*>(&Bp[(irB+o)*N+icB*4]);
        __syncthreads(); Ap+=BK; Bp+=BK*N;

        for(int k=0;k<BK;k++){
            for(int wm=0;wm<WMITER;wm++)
                for(int i=0;i<TM;i++)
                    rM[wm*TM+i]=As[k*BM + warpRow*WM + wm*WSUBM + tRowW*TM + i];
            for(int wn=0;wn<WNITER;wn++)
                for(int j=0;j<TN;j++)
                    rN[wn*TN+j]=Bs[k*BN + warpCol*WN + wn*WSUBN + tColW*TN + j];
            for(int wm=0;wm<WMITER;wm++)
                for(int wn=0;wn<WNITER;wn++)
                    for(int i=0;i<TM;i++)
                        for(int j=0;j<TN;j++)
                            acc[(wm*TM+i)*(WNITER*TN)+wn*TN+j]+=rM[wm*TM+i]*rN[wn*TN+j];
        }
        __syncthreads();
    }
    for(int wm=0;wm<WMITER;wm++)
        for(int wn=0;wn<WNITER;wn++)
            for(int i=0;i<TM;i++)
                for(int j=0;j<TN;j+=4){
                    int row=warpRow*WM+wm*WSUBM+tRowW*TM+i;
                    int col=warpCol*WN+wn*WSUBN+tColW*TN+j;
                    int b=(wm*TM+i)*(WNITER*TN)+wn*TN+j;
                    float4 v; v.x=acc[b]; v.y=acc[b+1]; v.z=acc[b+2]; v.w=acc[b+3];
                    *reinterpret_cast<float4*>(&Cp[row*N+col])=v;
                }
}

static void run(const char* name, void(*k)(const float*,const float*,float*,int),
                dim3 grid, int nthreads, const float* dA,const float* dB,float* dC,
                int N,const float* hA,const float* hB,float* hC){
    k<<<grid,nthreads>>>(dA,dB,dC,N); CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    nvtxRangePushA(name); cudaEventRecord(t0);
    k<<<grid,nthreads>>>(dA,dB,dC,N);
    cudaEventRecord(t1); CUDA_CHECK(cudaEventSynchronize(t1));
    float ms; cudaEventElapsedTime(&ms,t0,t1); nvtxRangePop();
    CUDA_CHECK(cudaMemcpy(hC,dC,(size_t)N*N*sizeof(float),cudaMemcpyDeviceToHost));
    int S=128,err=0;
    for(int i=0;i<S;i++)for(int j=0;j<S;j++){
        float s=0; for(int kk=0;kk<N;kk++)s+=hA[i*N+kk]*hB[kk*N+j];
        if(fabsf(s-hC[i*N+j])>1e-1f)err++;
    }
    printf("%-12s %7.3f ms | %.2f TFLOPS | %s\n",name,ms,
           2.0*N*N*N/(ms*1e-3)/1e12,err?"FAIL":"PASS");
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}

int main(){
    const int N=2048;
    printf("=== warp tiling vs float4 baseline (N=%d, FP32) ===\n",N);
    size_t bytes=(size_t)N*N*sizeof(float);
    float *hA=(float*)malloc(bytes),*hB=(float*)malloc(bytes),*hC=(float*)malloc(bytes);
    for(int i=0;i<N*N;i++){hA[i]=(rand()%10)/10.0f; hB[i]=(rand()%10)/10.0f;}
    float *dA,*dB,*dC;
    CUDA_CHECK(cudaMalloc(&dA,bytes)); CUDA_CHECK(cudaMalloc(&dB,bytes)); CUDA_CHECK(cudaMalloc(&dC,bytes));
    CUDA_CHECK(cudaMemcpy(dA,hA,bytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB,hB,bytes,cudaMemcpyHostToDevice));

    run("v0_float4", gemm_v0,       dim3(N/B0N,N/B0M), 256,     dA,dB,dC,N,hA,hB,hC);
    run("warptile",  gemm_warptile, dim3(N/BN,N/BM),   NTHREADS,dA,dB,dC,N,hA,hB,hC);

    cudaFree(dA); cudaFree(dB); cudaFree(dC); free(hA); free(hB); free(hC);
    return 0;
}
