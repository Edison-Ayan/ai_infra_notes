// =============================================================
// FlashAttention 第一课：online softmax 消掉 N×N 物化
//
// 标准 attention: S=QKᵀ[N×N] → P=softmax(S)[N×N] → O=PV。
//   两个 N×N 矩阵 → 显存 O(N²) + 要写回/读出 HBM(访存 bound)。
// FlashAttention: 分块流式 + online softmax，永不物化 N×N → 显存 O(N)。
//   核心: 维护 running (m,l,acc)，遇更大 max 用 corr=exp(m_old-m_new) 校正。
// 本 lab: 朴素(物化 S) vs Flash(流式)，验证输出一致 + 对比显存。
// =============================================================
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e=(x); if(e){printf("CUDA %s\n",cudaGetErrorString(e));return 1;} }while(0)
#define DMAX 64

// ---------- 朴素 attention：物化 N×N ----------
__global__ void naive_scores(const float* Q,const float* K,float* S,int N,int d,float sc){
    int i=blockIdx.y*blockDim.y+threadIdx.y, j=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<N&&j<N){ float s=0; for(int c=0;c<d;c++) s+=Q[i*d+c]*K[j*d+c]; S[(size_t)i*N+j]=s*sc; }
}
__global__ void naive_softmax(float* S,int N){            // 每行 softmax
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N) return;
    float m=-1e30f; for(int j=0;j<N;j++) m=fmaxf(m,S[(size_t)i*N+j]);
    float l=0; for(int j=0;j<N;j++){ float e=expf(S[(size_t)i*N+j]-m); S[(size_t)i*N+j]=e; l+=e; }
    for(int j=0;j<N;j++) S[(size_t)i*N+j]/=l;
}
__global__ void naive_pv(const float* S,const float* V,float* O,int N,int d){
    int i=blockIdx.y*blockDim.y+threadIdx.y, c=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<N&&c<d){ float o=0; for(int j=0;j<N;j++) o+=S[(size_t)i*N+j]*V[j*d+c]; O[i*d+c]=o; }
}

// ---------- FlashAttention：一线程一 query 行，online softmax，无 N×N ----------
__global__ void flash(const float* Q,const float* K,const float* V,float* O,int N,int d,float sc){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N) return;
    float m=-1e30f, l=0.f, acc[DMAX];
    for(int c=0;c<d;c++) acc[c]=0.f;
    for(int j=0;j<N;j++){
        float s=0; for(int c=0;c<d;c++) s+=Q[i*d+c]*K[j*d+c]; s*=sc;
        float m_new=fmaxf(m,s), corr=expf(m-m_new), p=expf(s-m_new);
        l=l*corr+p;
        for(int c=0;c<d;c++) acc[c]=acc[c]*corr+p*V[j*d+c];   // 校正 + 累加
        m=m_new;
    }
    for(int c=0;c<d;c++) O[i*d+c]=acc[c]/l;
}

int main(){
    const int N=4096, d=64; float sc=1.f/sqrtf((float)d);
    size_t qb=(size_t)N*d*4, sb=(size_t)N*N*4;
    printf("attention N=%d, d=%d\n", N, d);
    printf("朴素需物化 S[N×N] = %.0f MB；Flash 不需要(省这一整块)\n\n", sb/1e6);

    float *Q,*K,*V,*Ona,*Ofl,*S;
    CK(cudaMalloc(&Q,qb)); CK(cudaMalloc(&K,qb)); CK(cudaMalloc(&V,qb));
    CK(cudaMalloc(&Ona,qb)); CK(cudaMalloc(&Ofl,qb)); CK(cudaMalloc(&S,sb));
    // 随机初始化
    float* h=(float*)malloc(qb);
    auto fill=[&](float* dp){ for(size_t i=0;i<(size_t)N*d;i++) h[i]=((rand()%200)/100.f-1.f); cudaMemcpy(dp,h,qb,cudaMemcpyHostToDevice); };
    fill(Q); fill(K); fill(V);

    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1); float ms;
    dim3 b2(16,16), g2((N+15)/16,(N+15)/16), gpv((d+15)/16,(N+15)/16);

    cudaEventRecord(t0);
    naive_scores<<<g2,b2>>>(Q,K,S,N,d,sc);
    naive_softmax<<<(N+127)/128,128>>>(S,N);
    naive_pv<<<gpv,b2>>>(S,V,Ona,N,d);
    cudaEventRecord(t1); CK(cudaEventSynchronize(t1)); cudaEventElapsedTime(&ms,t0,t1);
    printf("朴素(物化N×N): %7.3f ms\n", ms);

    cudaEventRecord(t0);
    flash<<<(N+127)/128,128>>>(Q,K,V,Ofl,N,d,sc);
    cudaEventRecord(t1); CK(cudaEventSynchronize(t1)); cudaEventElapsedTime(&ms,t0,t1);
    printf("Flash(流式)   : %7.3f ms\n", ms);

    // 验证一致
    float *a=(float*)malloc(qb), *f=(float*)malloc(qb);
    CK(cudaMemcpy(a,Ona,qb,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(f,Ofl,qb,cudaMemcpyDeviceToHost));
    double num=0,den=0; for(size_t i=0;i<(size_t)N*d;i++){ double dd=a[i]-f[i]; num+=dd*dd; den+=(double)a[i]*a[i]; }
    printf("\nFlash vs 朴素 输出相对误差: %.2e  → %s\n", sqrt(num/den), sqrt(num/den)<1e-4?"一致(online softmax 正确)":"不一致");
    printf(">>> 显存: 朴素额外 %.0f MB(N×N) vs Flash 0；N=32768 时 N×N 会到 %.1f GB\n",
           sb/1e6, (double)32768*32768*4/1e9);
    return 0;
}
