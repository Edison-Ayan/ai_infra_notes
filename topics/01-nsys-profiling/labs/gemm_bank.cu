// =============================================================
// bank conflict 三药方对比：baseline(float4) vs padding vs swizzle
//
// float4 版 ncu 报出 shared load 5-way bank conflict、40% wavefront 浪费。
// 这里写三个 kernel 同台对比，配合 ncu 看每招把 5-way 压到几 way：
//   [0] gemm_v0     —— float4 baseline（转置 As + float4，5-way 冲突）
//   [1] gemm_pad    —— shared 每行 +4 padding，错开行间 bank 映射
//   [2] gemm_swz    —— XOR swizzle，按 k 重排列，打散 bank
// 三者算法/tile 完全一致(128×128/8×8)，只差 shared 布局，公平对比。
// =============================================================
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#define CUDA_CHECK(call) do {                                              \
    cudaError_t e = call;                                                  \
    if (e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);       \
    } } while (0)

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

// ---------- [0] baseline：float4 + 转置 As（= 我们的 gemm_float4） ----------
__global__ void gemm_v0(const float* A, const float* B, float* C, int N) {
    const int cRow = blockIdx.y, cCol = blockIdx.x;
    __shared__ float As[BK * BM];   // 转置 [BK][BM]
    __shared__ float Bs[BK * BN];
    const int tid = threadIdx.x;
    const int threadCol = tid % (BN / TN), threadRow = tid / (BN / TN);
    const float* Ap = A + cRow * BM * N;
    const float* Bp = B + cCol * BN;
    float* Cp = C + cRow * BM * N + cCol * BN;
    const int irA = tid / (BK / 4), icA = tid % (BK / 4);
    const int irB = tid / (BN / 4), icB = tid % (BN / 4);
    float acc[TM * TN] = {0.0f}, regM[TM], regN[TN];

    for (int bk = 0; bk < N; bk += BK) {
        float4 va = *reinterpret_cast<const float4*>(&Ap[irA * N + icA * 4]);
        As[(icA * 4 + 0) * BM + irA] = va.x; As[(icA * 4 + 1) * BM + irA] = va.y;
        As[(icA * 4 + 2) * BM + irA] = va.z; As[(icA * 4 + 3) * BM + irA] = va.w;
        *reinterpret_cast<float4*>(&Bs[irB * BN + icB * 4]) =
            *reinterpret_cast<const float4*>(&Bp[irB * N + icB * 4]);
        __syncthreads();
        Ap += BK; Bp += BK * N;
        for (int k = 0; k < BK; k++) {
            for (int i = 0; i < TM; i += 4)
                *reinterpret_cast<float4*>(&regM[i]) =
                    *reinterpret_cast<float4*>(&As[k * BM + threadRow * TM + i]);
            for (int j = 0; j < TN; j += 4)
                *reinterpret_cast<float4*>(&regN[j]) =
                    *reinterpret_cast<float4*>(&Bs[k * BN + threadCol * TN + j]);
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
                    acc[i * TN + j] += regM[i] * regN[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j += 4) {
            float4 v; v.x = acc[i*TN+j]; v.y = acc[i*TN+j+1]; v.z = acc[i*TN+j+2]; v.w = acc[i*TN+j+3];
            *reinterpret_cast<float4*>(&Cp[(threadRow*TM+i)*N + threadCol*TN+j]) = v;
        }
}

// ---------- [1] padding：shared 每行 +4，错开行间 bank ----------
#define PAD 4
#define SAW (BM + PAD)   // As 行宽（转置后行=BK，每行 BM+PAD）
#define SBW (BN + PAD)   // Bs 行宽
__global__ void gemm_pad(const float* A, const float* B, float* C, int N) {
    const int cRow = blockIdx.y, cCol = blockIdx.x;
    __shared__ float As[BK * SAW];
    __shared__ float Bs[BK * SBW];
    const int tid = threadIdx.x;
    const int threadCol = tid % (BN / TN), threadRow = tid / (BN / TN);
    const float* Ap = A + cRow * BM * N;
    const float* Bp = B + cCol * BN;
    float* Cp = C + cRow * BM * N + cCol * BN;
    const int irA = tid / (BK / 4), icA = tid % (BK / 4);
    const int irB = tid / (BN / 4), icB = tid % (BN / 4);
    float acc[TM * TN] = {0.0f}, regM[TM], regN[TN];

    for (int bk = 0; bk < N; bk += BK) {
        float4 va = *reinterpret_cast<const float4*>(&Ap[irA * N + icA * 4]);
        As[(icA * 4 + 0) * SAW + irA] = va.x; As[(icA * 4 + 1) * SAW + irA] = va.y;
        As[(icA * 4 + 2) * SAW + irA] = va.z; As[(icA * 4 + 3) * SAW + irA] = va.w;
        *reinterpret_cast<float4*>(&Bs[irB * SBW + icB * 4]) =
            *reinterpret_cast<const float4*>(&Bp[irB * N + icB * 4]);
        __syncthreads();
        Ap += BK; Bp += BK * N;
        for (int k = 0; k < BK; k++) {
            for (int i = 0; i < TM; i += 4)
                *reinterpret_cast<float4*>(&regM[i]) =
                    *reinterpret_cast<float4*>(&As[k * SAW + threadRow * TM + i]);
            for (int j = 0; j < TN; j += 4)
                *reinterpret_cast<float4*>(&regN[j]) =
                    *reinterpret_cast<float4*>(&Bs[k * SBW + threadCol * TN + j]);
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
                    acc[i * TN + j] += regM[i] * regN[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j += 4) {
            float4 v; v.x = acc[i*TN+j]; v.y = acc[i*TN+j+1]; v.z = acc[i*TN+j+2]; v.w = acc[i*TN+j+3];
            *reinterpret_cast<float4*>(&Cp[(threadRow*TM+i)*N + threadCol*TN+j]) = v;
        }
}

// ---------- [2] swizzle：列索引 XOR (k&7)<<2，按 k 打散 bank ----------
// 保持 float4：XOR 常量是 4 的倍数，不破坏 4 元素对齐组；store/load 同一映射故正确。
#define SWZ(col, k) ((col) ^ (((k) & 7) << 2))
__global__ void gemm_swz(const float* A, const float* B, float* C, int N) {
    const int cRow = blockIdx.y, cCol = blockIdx.x;
    __shared__ float As[BK * BM];
    __shared__ float Bs[BK * BN];
    const int tid = threadIdx.x;
    const int threadCol = tid % (BN / TN), threadRow = tid / (BN / TN);
    const float* Ap = A + cRow * BM * N;
    const float* Bp = B + cCol * BN;
    float* Cp = C + cRow * BM * N + cCol * BN;
    const int irA = tid / (BK / 4), icA = tid % (BK / 4);
    const int irB = tid / (BN / 4), icB = tid % (BN / 4);
    float acc[TM * TN] = {0.0f}, regM[TM], regN[TN];

    for (int bk = 0; bk < N; bk += BK) {
        float4 va = *reinterpret_cast<const float4*>(&Ap[irA * N + icA * 4]);
        // 转置散写，列=irA，行(k)=icA*4+r → 按该 k 行 swizzle
        As[(icA*4+0) * BM + SWZ(irA, icA*4+0)] = va.x;
        As[(icA*4+1) * BM + SWZ(irA, icA*4+1)] = va.y;
        As[(icA*4+2) * BM + SWZ(irA, icA*4+2)] = va.z;
        As[(icA*4+3) * BM + SWZ(irA, icA*4+3)] = va.w;
        // Bs：行=irB(k)，列起点 icB*4，按该 k 行 swizzle（整组 float4 一起移）
        *reinterpret_cast<float4*>(&Bs[irB * BN + SWZ(icB * 4, irB)]) =
            *reinterpret_cast<const float4*>(&Bp[irB * N + icB * 4]);
        __syncthreads();
        Ap += BK; Bp += BK * N;
        for (int k = 0; k < BK; k++) {
            for (int i = 0; i < TM; i += 4)
                *reinterpret_cast<float4*>(&regM[i]) =
                    *reinterpret_cast<float4*>(&As[k * BM + SWZ(threadRow * TM + i, k)]);
            for (int j = 0; j < TN; j += 4)
                *reinterpret_cast<float4*>(&regN[j]) =
                    *reinterpret_cast<float4*>(&Bs[k * BN + SWZ(threadCol * TN + j, k)]);
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
                    acc[i * TN + j] += regM[i] * regN[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j += 4) {
            float4 v; v.x = acc[i*TN+j]; v.y = acc[i*TN+j+1]; v.z = acc[i*TN+j+2]; v.w = acc[i*TN+j+3];
            *reinterpret_cast<float4*>(&Cp[(threadRow*TM+i)*N + threadCol*TN+j]) = v;
        }
}

static float run_one(const char* name, void(*k)(const float*,const float*,float*,int),
                     const float* dA, const float* dB, float* dC, int N,
                     const float* hA, const float* hB, float* hC) {
    dim3 grid(N / BN, N / BM);
    k<<<grid, 256>>>(dA, dB, dC, N);   // 预热
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    nvtxRangePushA(name);
    cudaEventRecord(t0);
    k<<<grid, 256>>>(dA, dB, dC, N);
    cudaEventRecord(t1); CUDA_CHECK(cudaEventSynchronize(t1));
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    nvtxRangePop();
    // 正确性：抽查左上 128×128
    CUDA_CHECK(cudaMemcpy(hC, dC, (size_t)N*N*sizeof(float), cudaMemcpyDeviceToHost));
    int S = 128, err = 0;
    for (int i = 0; i < S; i++) for (int j = 0; j < S; j++) {
        float s = 0; for (int kk = 0; kk < N; kk++) s += hA[i*N+kk]*hB[kk*N+j];
        if (fabsf(s - hC[i*N+j]) > 1e-1f) err++;
    }
    printf("%-12s %7.3f ms | %.2f TFLOPS | %s\n", name, ms,
           2.0*N*N*N/(ms*1e-3)/1e12, err ? "FAIL" : "PASS");
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms;
}

int main() {
    const int N = 2048;
    printf("=== bank conflict 三药方对比 (N=%d, FP32) ===\n", N);
    size_t bytes = (size_t)N*N*sizeof(float);
    float *hA=(float*)malloc(bytes), *hB=(float*)malloc(bytes), *hC=(float*)malloc(bytes);
    for (int i = 0; i < N*N; i++) { hA[i]=(rand()%10)/10.0f; hB[i]=(rand()%10)/10.0f; }
    float *dA,*dB,*dC;
    CUDA_CHECK(cudaMalloc(&dA,bytes)); CUDA_CHECK(cudaMalloc(&dB,bytes)); CUDA_CHECK(cudaMalloc(&dC,bytes));
    CUDA_CHECK(cudaMemcpy(dA,hA,bytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB,hB,bytes,cudaMemcpyHostToDevice));

    run_one("v0_float4", gemm_v0,  dA, dB, dC, N, hA, hB, hC);
    run_one("pad",       gemm_pad, dA, dB, dC, N, hA, hB, hC);
    run_one("swizzle",   gemm_swz, dA, dB, dC, N, hA, hB, hC);

    cudaFree(dA); cudaFree(dB); cudaFree(dC); free(hA); free(hB); free(hC);
    return 0;
}
