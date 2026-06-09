// =============================================================
// 量化实验②：W4A16 实测——INT4 权重 GEMV 到底省多少带宽 (接 topic 04)
//
// decode = batch=1 的 y[N] = W[N×K] @ x[K]，访存 bound(每个权重只用一次)。
// 对比两版同样的 GEMV：
//   [FP16]  权重 FP16，搬 N*K*2 字节
//   [INT4]  权重 INT4(打包 2个/字节) + per-group scale，搬 N*K*0.5 字节(4× 少)
//           kernel 里在线反量化回 FP16 再乘加 → W4A16
// 看：INT4 是否因"搬权重字节少 4×"而在访存 bound 的 decode 上更快。
// =============================================================
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CK(x) do{ cudaError_t e=(x); if(e){printf("CUDA %s\n",cudaGetErrorString(e));return 1;} }while(0)

// ---- FP16 GEMV：一个 block 算一行，256 线程沿 K 归约 ----
__global__ void gemv_fp16(const half* W, const half* x, float* y, int N, int K){
    int n=blockIdx.x, t=threadIdx.x;
    float acc=0;
    for(int k=t;k<K;k+=blockDim.x) acc += __half2float(W[(size_t)n*K+k])*__half2float(x[k]);
    __shared__ float s[256]; s[t]=acc; __syncthreads();
    for(int o=128;o>0;o>>=1){ if(t<o) s[t]+=s[t+o]; __syncthreads(); }
    if(t==0) y[n]=s[0];
}

// ---- INT4 GEMV：每线程读 1 字节(2 个 int4)，在线反量化 ----
__global__ void gemv_int4(const uint8_t* Wp, const half* scale, const half* x,
                          float* y, int N, int K, int G){
    int n=blockIdx.x, t=threadIdx.x, Kp=K/2, nG=K/G;
    float acc=0;
    for(int p=t;p<Kp;p+=blockDim.x){
        uint8_t b = Wp[(size_t)n*Kp+p];
        int lo=b&0xF, hi=(b>>4)&0xF;
        int w0 = lo<8?lo:lo-16, w1 = hi<8?hi:hi-16;   // 4-bit 有符号还原
        int k0=2*p, k1=2*p+1;
        float s0=__half2float(scale[(size_t)n*nG + k0/G]);
        float s1=__half2float(scale[(size_t)n*nG + k1/G]);
        acc += w0*s0*__half2float(x[k0]) + w1*s1*__half2float(x[k1]);
    }
    __shared__ float s[256]; s[t]=acc; __syncthreads();
    for(int o=128;o>0;o>>=1){ if(t<o) s[t]+=s[t+o]; __syncthreads(); }
    if(t==0) y[n]=s[0];
}

// ---- INT4 GEMV 优化版：每线程向量化读 8 字节(16个int4)，每组 scale 只取一次 ----
//   Kp=2048=256×8 → 一个线程一发 uint2 覆盖整行，无循环；16 个权重同组共享 scale
__global__ void gemv_int4_v2(const uint8_t* Wp, const half* scale, const half* x,
                             float* y, int N, int K, int G){
    int n=blockIdx.x, t=threadIdx.x, Kp=K/2, nG=K/G;
    float acc=0;
    for(int bo=t*8; bo<Kp; bo+=blockDim.x*8){              // 每次向量化 8 字节=16 int4
        uint2 packed=*reinterpret_cast<const uint2*>(Wp+(size_t)n*Kp+bo);
        int k0=2*bo; float sc=__half2float(scale[(size_t)n*nG + k0/G]); // 同组 1 个 scale
        float part=0; uint8_t* pb=(uint8_t*)&packed;
        #pragma unroll
        for(int bi=0;bi<8;bi++){
            uint8_t b=pb[bi]; int lo=b&0xF,hi=(b>>4)&0xF;
            int w0=lo<8?lo:lo-16, w1=hi<8?hi:hi-16; int k=k0+2*bi;
            part += w0*__half2float(x[k]) + w1*__half2float(x[k+1]);
        }
        acc += part*sc;                                    // 每块乘一次 scale
    }
    __shared__ float s[256]; s[t]=acc; __syncthreads();
    for(int o=128;o>0;o>>=1){ if(t<o) s[t]+=s[t+o]; __syncthreads(); }
    if(t==0) y[n]=s[0];
}

int main(){
    const int N=8192, K=8192, G=128;   // 放大到 GB 级附近，让 INT4 也喂饱带宽
    size_t NK=(size_t)N*K;
    printf("GEMV y[%d] = W[%dx%d] @ x[%d]  (decode, batch=1)\n", N, N, K, K);

    // host 权重(FP16 值域用 N(0,0.02) 这种典型权重)
    std::vector<float> Wf(NK);
    for(size_t i=0;i<NK;i++){ float u1=(rand()+1.f)/(RAND_MAX+1.f),u2=(rand()+1.f)/(RAND_MAX+1.f);
        Wf[i]=sqrtf(-2*logf(u1))*cosf(6.2831853f*u2)*0.02f; }
    std::vector<half> Wh(NK); for(size_t i=0;i<NK;i++) Wh[i]=__float2half(Wf[i]);

    // 量化成 INT4 + per-group(沿K) scale
    int nG=K/G;
    std::vector<uint8_t> Wp(NK/2); std::vector<half> Sc((size_t)N*nG);
    for(int n=0;n<N;n++) for(int g=0;g<nG;g++){
        float maxabs=1e-12f;
        for(int j=0;j<G;j++) maxabs=fmaxf(maxabs,fabsf(Wf[(size_t)n*K+g*G+j]));
        float sc=maxabs/7.f; Sc[(size_t)n*nG+g]=__float2half(sc);
        for(int j=0;j<G;j++){ int k=g*G+j; int q=(int)lroundf(Wf[(size_t)n*K+k]/sc);
            if(q>7)q=7; if(q<-8)q=-8;
            uint8_t nib=q&0xF;
            if(k&1) Wp[(size_t)n*(K/2)+k/2] |= nib<<4; else Wp[(size_t)n*(K/2)+k/2]=(Wp[(size_t)n*(K/2)+k/2]&0xF0)|nib;
        }
    }

    // device
    half *dWh,*dx; float *dy; uint8_t* dWp; half* dSc;
    std::vector<half> xh(K); for(int k=0;k<K;k++) xh[k]=__float2half(((rand()%100)/100.f-0.5f));
    CK(cudaMalloc(&dWh,NK*2)); CK(cudaMalloc(&dWp,NK/2)); CK(cudaMalloc(&dSc,(size_t)N*nG*2));
    CK(cudaMalloc(&dx,K*2)); CK(cudaMalloc(&dy,N*4));
    CK(cudaMemcpy(dWh,Wh.data(),NK*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dWp,Wp.data(),NK/2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dSc,Sc.data(),(size_t)N*nG*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx,xh.data(),K*2,cudaMemcpyHostToDevice));

    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    auto bench=[&](const char* tag,double bytes,auto launch){
        for(int i=0;i<20;i++) launch(); cudaDeviceSynchronize();
        const int it=200; cudaEventRecord(t0);
        for(int i=0;i<it;i++) launch();
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms,t0,t1); ms/=it;
        printf("%-6s | %7.3f ms | 搬权重 %5.1f MB | 有效带宽 %6.1f GB/s\n",
               tag, ms, bytes/1e6, bytes/(ms*1e-3)/1e9);
        return ms;
    };
    printf("\n");
    double msF=bench("FP16", (double)NK*2,[&]{ gemv_fp16<<<N,256>>>(dWh,dx,dy,N,K); });
    std::vector<float> yF(N); CK(cudaMemcpy(yF.data(),dy,N*4,cudaMemcpyDeviceToHost));
    bench("INT4n",(double)NK*0.5,[&]{ gemv_int4<<<N,256>>>(dWp,dSc,dx,dy,N,K,G); });  // 朴素
    double msI=bench("INT4opt",(double)NK*0.5,[&]{ gemv_int4_v2<<<N,256>>>(dWp,dSc,dx,dy,N,K,G); });
    std::vector<float> yI(N); CK(cudaMemcpy(yI.data(),dy,N*4,cudaMemcpyDeviceToHost));

    // 精度：INT4 vs FP16 输出的相对误差
    double num=0,den=0; for(int n=0;n<N;n++){ double d=yF[n]-yI[n]; num+=d*d; den+=(double)yF[n]*yF[n]; }
    printf("\nINT4 vs FP16 输出相对误差: %.2f%%\n", sqrt(num/den)*100);
    printf(">>> INT4 加速 %.2fx，权重显存 %.1fMB→%.1fMB (省 4×)\n",
           msF/msI, NK*2/1e6, NK*0.5/1e6);
    return 0;
}
