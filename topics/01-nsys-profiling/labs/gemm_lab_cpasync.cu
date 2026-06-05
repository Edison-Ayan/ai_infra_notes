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
#include <cuda_pipeline.h>      // cp.async：__pipeline_memcpy_async / commit / wait_prior
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
// [3.5] Register-blocked GEMM：每个线程算 TM×TN 个输出
//   - block tile: BM×BN，深度 BK；block = 256 线程(1D)
//   - 每线程持有 TM×TN 个累加器(寄存器)，把 A 的一条/B 的一条
//     读进寄存器后做 TM×TN 次 FFMA => 访存复用率大增、ILP 拉满
//   要求 N 能被 BM/BN/BK 整除(这里 N=2048 满足)，故省略边界判断
// =============================================================
#define BM 64
#define BN 64
#define BK 8
#define TM 4
#define TN 4   // 每线程 4×4=16 个输出；256 线程 × 16 = 4096 = 64×64 ✓
__global__ void gemm_reg(const float* A, const float* B, float* C, int N) {
    const int cRow = blockIdx.y, cCol = blockIdx.x;     // 该 block 负责的 C 块
    __shared__ float As[BK * BM];   // 转置存放：As[k*BM + m]
    __shared__ float Bs[BK * BN];   // Bs[k*BN + n]

    const int tid = threadIdx.x;                        // 0..255
    const int threadCol = tid % (BN / TN);              // 0..15
    const int threadRow = tid / (BN / TN);              // 0..15

    const float* Ap = A + cRow * BM * N;                // A 块起点(行)
    const float* Bp = B + cCol * BN;                    // B 块起点(列)
    float*       Cp = C + cRow * BM * N + cCol * BN;

    // 256 线程协作搬运：A 块 BM×BK=512、B 块 BK×BN=512，各 2 元素/线程
    const int innerRowA = tid / BK, innerColA = tid % BK;   // A: 32 行/趟
    const int strideA   = 256 / BK;                         // =32 → 2 趟填满 BM=64
    const int innerRowB = tid / BN, innerColB = tid % BN;   // B: 4 行/趟
    const int strideB   = 256 / BN;                         // =4  → 2 趟填满 BK=8

    float acc[TM * TN] = {0.0f};
    float regM[TM], regN[TN];

    for (int bk = 0; bk < N; bk += BK) {
        for (int o = 0; o < BM; o += strideA)           // 载入 A 块(转置)
            As[innerColA * BM + innerRowA + o] = Ap[(innerRowA + o) * N + innerColA];
        for (int o = 0; o < BK; o += strideB)           // 载入 B 块
            Bs[(innerRowB + o) * BN + innerColB] = Bp[(innerRowB + o) * N + innerColB];
        __syncthreads();
        Ap += BK; Bp += BK * N;                         // 沿 K 前进

        for (int k = 0; k < BK; k++) {                  // 寄存器内做外积
            for (int i = 0; i < TM; i++) regM[i] = As[k * BM + threadRow * TM + i];
            for (int j = 0; j < TN; j++) regN[j] = Bs[k * BN + threadCol * TN + j];
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
                    acc[i * TN + j] += regM[i] * regN[j];   // 16 个独立累加器
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; i++)                        // 写回
        for (int j = 0; j < TN; j++)
            Cp[(threadRow * TM + i) * N + threadCol * TN + j] = acc[i * TN + j];
}

// =============================================================
// [3.6] float4 向量化 GEMM（register blocking 的进阶）
//   在 reg 版基础上加三处向量化：
//     (a) global→shared 的搬运用 float4，一条指令搬 4 个 float
//     (b) shared→register 的 regM/regN 读取用 float4
//     (c) 结果写回 C 用 float4
//   并放大 tile 到 128×128、每线程 8×8，喂饱向量化访存。
//   要求 N 能被 128/8 整除(N=2048 满足)。
// =============================================================
#define BM2 128
#define BN2 128
#define BK2 8
#define TM2 8
#define TN2 8   // 每线程 8×8=64 输出；256 线程 ×64 = 16384 = 128×128 ✓
__global__ void gemm_float4(const float* A, const float* B, float* C, int N) {
    const int cRow = blockIdx.y, cCol = blockIdx.x;
    __shared__ float As[BK2 * BM2];   // 转置存放 [BK2][BM2]
    __shared__ float Bs[BK2 * BN2];   // [BK2][BN2]

    const int tid = threadIdx.x;                  // 0..255
    const int threadCol = tid % (BN2 / TN2);      // 0..15
    const int threadRow = tid / (BN2 / TN2);      // 0..15

    const float* Ap = A + cRow * BM2 * N;
    const float* Bp = B + cCol * BN2;
    float*       Cp = C + cRow * BM2 * N + cCol * BN2;

    // 搬运索引（按 float4 = 每次 4 个）
    //   A 块 128×8 = 1024 float = 256 个 float4 → 每线程 1 个 float4
    const int innerRowA = tid / (BK2 / 4);        // BK2/4=2 → 行 0..127
    const int innerColA = tid % (BK2 / 4);        // 0..1 (×4 = 列 0 或 4)
    //   B 块 8×128 = 1024 float = 256 个 float4 → 每线程 1 个 float4
    const int innerRowB = tid / (BN2 / 4);        // BN2/4=32 → 行 0..7
    const int innerColB = tid % (BN2 / 4);        // 0..31 (×4)

    float acc[TM2 * TN2] = {0.0f};
    float regM[TM2], regN[TN2];

    for (int bk = 0; bk < N; bk += BK2) {
        // (a) 载入 A：float4 读，转置散写(转置使后续 regM 读取连续)
        float4 va = reinterpret_cast<const float4*>(&Ap[innerRowA * N + innerColA * 4])[0];
        As[(innerColA * 4 + 0) * BM2 + innerRowA] = va.x;
        As[(innerColA * 4 + 1) * BM2 + innerRowA] = va.y;
        As[(innerColA * 4 + 2) * BM2 + innerRowA] = va.z;
        As[(innerColA * 4 + 3) * BM2 + innerRowA] = va.w;
        // 载入 B：float4 读 + float4 写(不转置，地址连续可向量化写)
        reinterpret_cast<float4*>(&Bs[innerRowB * BN2 + innerColB * 4])[0] =
            reinterpret_cast<const float4*>(&Bp[innerRowB * N + innerColB * 4])[0];
        __syncthreads();
        Ap += BK2; Bp += BK2 * N;

        for (int k = 0; k < BK2; k++) {
            // (b) shared→register 用 float4（TM2=8 → 2 个 float4）
            for (int i = 0; i < TM2; i += 4)
                reinterpret_cast<float4*>(&regM[i])[0] =
                    reinterpret_cast<float4*>(&As[k * BM2 + threadRow * TM2 + i])[0];
            for (int j = 0; j < TN2; j += 4)
                reinterpret_cast<float4*>(&regN[j])[0] =
                    reinterpret_cast<float4*>(&Bs[k * BN2 + threadCol * TN2 + j])[0];
            for (int i = 0; i < TM2; i++)
                for (int j = 0; j < TN2; j++)
                    acc[i * TN2 + j] += regM[i] * regN[j];   // 64 个独立累加器
        }
        __syncthreads();
    }
    // (c) 写回用 float4（每行 TN2=8 → 2 个 float4）
    for (int i = 0; i < TM2; i++)
        for (int j = 0; j < TN2; j += 4) {
            float4 v;
            v.x = acc[i * TN2 + j + 0]; v.y = acc[i * TN2 + j + 1];
            v.z = acc[i * TN2 + j + 2]; v.w = acc[i * TN2 + j + 3];
            reinterpret_cast<float4*>(&Cp[(threadRow * TM2 + i) * N + threadCol * TN2 + j])[0] = v;
        }
}

// =============================================================
// [3.7] Double-buffered float4 GEMM（软件流水 / 寄存器预取）
//   在 float4 基础上开两块 shared buffer：算当前 tile 的同时，
//   把下一个 tile 的 global 数据预取到寄存器(ldA/ldB)，
//   让 global→shared 的访存延迟被计算盖住，消掉同步气泡。
//   流程：载入 tile0 → 循环{ 预取下一片到寄存器; 算当前片;
//         把寄存器写入另一块 buffer; sync; 换 buffer }
// =============================================================
#define BM3 128
#define BN3 128
#define BK3 8
#define TM3 8
#define TN3 8
__global__ void gemm_db(const float* A, const float* B, float* C, int N) {
    const int cRow = blockIdx.y, cCol = blockIdx.x;
    __shared__ float As[2][BK3 * BM3];   // 双缓冲，转置存放
    __shared__ float Bs[2][BK3 * BN3];

    const int tid = threadIdx.x;
    const int threadCol = tid % (BN3 / TN3);   // 0..15
    const int threadRow = tid / (BN3 / TN3);   // 0..15

    const float* Abase = A + cRow * BM3 * N;
    const float* Bbase = B + cCol * BN3;
    float*       Cp    = C + cRow * BM3 * N + cCol * BN3;

    const int innerRowA = tid / (BK3 / 4), innerColA = tid % (BK3 / 4); // 行0..127, 列0..1
    const int innerRowB = tid / (BN3 / 4), innerColB = tid % (BN3 / 4); // 行0..7,   列0..31

    float acc[TM3 * TN3] = {0.0f};
    float regM[TM3], regN[TN3];
    float4 ldA, ldB;   // 预取用寄存器

    // ---- 预载 tile0 到 buffer 0 ----
    ldA = *reinterpret_cast<const float4*>(&Abase[innerRowA * N + innerColA * 4]);
    As[0][(innerColA * 4 + 0) * BM3 + innerRowA] = ldA.x;
    As[0][(innerColA * 4 + 1) * BM3 + innerRowA] = ldA.y;
    As[0][(innerColA * 4 + 2) * BM3 + innerRowA] = ldA.z;
    As[0][(innerColA * 4 + 3) * BM3 + innerRowA] = ldA.w;
    *reinterpret_cast<float4*>(&Bs[0][innerRowB * BN3 + innerColB * 4]) =
        *reinterpret_cast<const float4*>(&Bbase[innerRowB * N + innerColB * 4]);
    __syncthreads();

    int cur = 0;
    for (int bk = 0; bk < N; bk += BK3) {
        int next = bk + BK3;
        // (1) 预取下一片 → 寄存器（LDG 提前发，延迟被下面的计算盖住）
        if (next < N) {
            ldA = *reinterpret_cast<const float4*>(&Abase[innerRowA * N + next + innerColA * 4]);
            ldB = *reinterpret_cast<const float4*>(&Bbase[(next + innerRowB) * N + innerColB * 4]);
        }
        // (2) 用当前 buffer 计算
        for (int k = 0; k < BK3; k++) {
            for (int i = 0; i < TM3; i += 4)
                *reinterpret_cast<float4*>(&regM[i]) =
                    *reinterpret_cast<float4*>(&As[cur][k * BM3 + threadRow * TM3 + i]);
            for (int j = 0; j < TN3; j += 4)
                *reinterpret_cast<float4*>(&regN[j]) =
                    *reinterpret_cast<float4*>(&Bs[cur][k * BN3 + threadCol * TN3 + j]);
            for (int i = 0; i < TM3; i++)
                for (int j = 0; j < TN3; j++)
                    acc[i * TN3 + j] += regM[i] * regN[j];
        }
        // (3) 把预取好的寄存器写入另一块 buffer，再换
        if (next < N) {
            As[cur ^ 1][(innerColA * 4 + 0) * BM3 + innerRowA] = ldA.x;
            As[cur ^ 1][(innerColA * 4 + 1) * BM3 + innerRowA] = ldA.y;
            As[cur ^ 1][(innerColA * 4 + 2) * BM3 + innerRowA] = ldA.z;
            As[cur ^ 1][(innerColA * 4 + 3) * BM3 + innerRowA] = ldA.w;
            *reinterpret_cast<float4*>(&Bs[cur ^ 1][innerRowB * BN3 + innerColB * 4]) = ldB;
            __syncthreads();
            cur ^= 1;
        }
    }
    // ---- 写回 ----
    for (int i = 0; i < TM3; i++)
        for (int j = 0; j < TN3; j += 4) {
            float4 v;
            v.x = acc[i * TN3 + j + 0]; v.y = acc[i * TN3 + j + 1];
            v.z = acc[i * TN3 + j + 2]; v.w = acc[i * TN3 + j + 3];
            *reinterpret_cast<float4*>(&Cp[(threadRow * TM3 + i) * N + threadCol * TN3 + j]) = v;
        }
}

// =============================================================
// [3.8] cp.async 双缓冲 GEMM（救活 double buffering）
//   关键：用 cp.async 让 global 直接拷进 shared、绕过寄存器，
//   从根上避开 db 版的"寄存器悬崖"——预取不再吃寄存器，occupancy 保住。
//
//   ⚠️ 代价/取舍：cp.async 要求 global 和 shared 两端都连续，
//   没法做转置。所以这里 As 改成自然布局 [BM][BK]（不转置），
//   导致 regM 读取变成跨步标量(8 条 LDS)——float4 只保住了 regN。
//   这正是"转置 vs cp.async 不可兼得"的真实工程权衡。
// =============================================================
#define BM4 128
#define BN4 128
#define BK4 8
#define TM4 8
#define TN4 8
__global__ void gemm_cpasync(const float* A, const float* B, float* C, int N) {
    const int cRow = blockIdx.y, cCol = blockIdx.x;
    __shared__ float As[2][BM4 * BK4];   // 自然布局 [BM][BK]（不转置，cp.async 要连续）
    __shared__ float Bs[2][BK4 * BN4];   // 自然布局 [BK][BN]

    const int tid = threadIdx.x;
    const int threadCol = tid % (BN4 / TN4);   // 0..15
    const int threadRow = tid / (BN4 / TN4);   // 0..15

    const float* Abase = A + cRow * BM4 * N;
    const float* Bbase = B + cCol * BN4;
    float*       Cp    = C + cRow * BM4 * N + cCol * BN4;

    const int innerRowA = tid / (BK4 / 4), innerColA = tid % (BK4 / 4); // 行0..127, 列0..1
    const int innerRowB = tid / (BN4 / 4), innerColB = tid % (BN4 / 4); // 行0..7,   列0..31

    float acc[TM4 * TN4] = {0.0f};
    float regM[TM4], regN[TN4];

    // 发起一片 tile 的 cp.async 拷贝（A、B 各一个 float4=16B），不经寄存器
    auto load_async = [&](int buf, int koff) {
        __pipeline_memcpy_async(
            &As[buf][innerRowA * BK4 + innerColA * 4],
            &Abase[innerRowA * N + koff + innerColA * 4], 16);
        __pipeline_memcpy_async(
            &Bs[buf][innerRowB * BN4 + innerColB * 4],
            &Bbase[(koff + innerRowB) * N + innerColB * 4], 16);
        __pipeline_commit();
    };

    load_async(0, 0);               // 预载 tile0
    __pipeline_wait_prior(0);       // 等它到位
    __syncthreads();

    int cur = 0;
    for (int bk = 0; bk < N; bk += BK4) {
        int next = bk + BK4;
        if (next < N) load_async(cur ^ 1, next);   // 后台预取下一片，零寄存器

        for (int k = 0; k < BK4; k++) {
            // regM：As 自然布局，固定 k 时跨步 BK4 → 只能标量读
            for (int i = 0; i < TM4; i++)
                regM[i] = As[cur][(threadRow * TM4 + i) * BK4 + k];
            // regN：Bs 自然布局，固定 k 时连续 → 仍可 float4
            for (int j = 0; j < TN4; j += 4)
                *reinterpret_cast<float4*>(&regN[j]) =
                    *reinterpret_cast<float4*>(&Bs[cur][k * BN4 + threadCol * TN4 + j]);
            for (int i = 0; i < TM4; i++)
                for (int j = 0; j < TN4; j++)
                    acc[i * TN4 + j] += regM[i] * regN[j];
        }

        if (next < N) {
            __pipeline_wait_prior(0);   // 等后台预取完成再换 buffer
            __syncthreads();
            cur ^= 1;
        }
    }
    for (int i = 0; i < TM4; i++)
        for (int j = 0; j < TN4; j += 4) {
            float4 v;
            v.x = acc[i * TN4 + j + 0]; v.y = acc[i * TN4 + j + 1];
            v.z = acc[i * TN4 + j + 2]; v.w = acc[i * TN4 + j + 3];
            *reinterpret_cast<float4*>(&Cp[(threadRow * TM4 + i) * N + threadCol * TN4 + j]) = v;
        }
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
    // [3.5] Register-blocked GEMM
    //   grid 按 BM/BN 划块，block = 256 线程(1D)
    // =========================================================
    dim3 grid_reg(N / BN, N / BM);
    gemm_reg<<<grid_reg, 256>>>(d_A, d_B, d_C, N);   // 预热
    CUDA_CHECK(cudaDeviceSynchronize());

    NVTX_PUSH("3.5_reg_gemm");
    cudaEventRecord(t0);
    gemm_reg<<<grid_reg, 256>>>(d_A, d_B, d_C, N);
    cudaEventRecord(t1);
    CUDA_CHECK(cudaEventSynchronize(t1));
    cudaEventElapsedTime(&ms, t0, t1);
    NVTX_POP();
    printf("[3.5] reg_gemm: %8.3f ms | %.2f TFLOPS\n",
           ms, 2.0 * N * N * N / (ms * 1e-3) / 1e12);

    // =========================================================
    // [3.6] float4 向量化 GEMM
    //   grid 按 BM2/BN2=128 划块，block = 256 线程(1D)
    // =========================================================
    dim3 grid_f4(N / BN2, N / BM2);
    gemm_float4<<<grid_f4, 256>>>(d_A, d_B, d_C, N);   // 预热
    CUDA_CHECK(cudaDeviceSynchronize());

    NVTX_PUSH("3.6_float4_gemm");
    cudaEventRecord(t0);
    gemm_float4<<<grid_f4, 256>>>(d_A, d_B, d_C, N);
    cudaEventRecord(t1);
    CUDA_CHECK(cudaEventSynchronize(t1));
    cudaEventElapsedTime(&ms, t0, t1);
    NVTX_POP();
    printf("[3.6] float4 : %8.3f ms | %.2f TFLOPS\n",
           ms, 2.0 * N * N * N / (ms * 1e-3) / 1e12);

    // =========================================================
    // [3.7] Double-buffered float4 GEMM
    // =========================================================
    dim3 grid_db(N / BN3, N / BM3);
    gemm_db<<<grid_db, 256>>>(d_A, d_B, d_C, N);   // 预热
    CUDA_CHECK(cudaDeviceSynchronize());

    NVTX_PUSH("3.7_db_gemm");
    cudaEventRecord(t0);
    gemm_db<<<grid_db, 256>>>(d_A, d_B, d_C, N);
    cudaEventRecord(t1);
    CUDA_CHECK(cudaEventSynchronize(t1));
    cudaEventElapsedTime(&ms, t0, t1);
    NVTX_POP();
    printf("[3.7] db     : %8.3f ms | %.2f TFLOPS\n",
           ms, 2.0 * N * N * N / (ms * 1e-3) / 1e12);

    // =========================================================
    // [3.8] cp.async 双缓冲 GEMM
    // =========================================================
    dim3 grid_ca(N / BN4, N / BM4);
    gemm_cpasync<<<grid_ca, 256>>>(d_A, d_B, d_C, N);   // 预热
    CUDA_CHECK(cudaDeviceSynchronize());

    NVTX_PUSH("3.8_cpasync_gemm");
    cudaEventRecord(t0);
    gemm_cpasync<<<grid_ca, 256>>>(d_A, d_B, d_C, N);
    cudaEventRecord(t1);
    CUDA_CHECK(cudaEventSynchronize(t1));
    cudaEventElapsedTime(&ms, t0, t1);
    NVTX_POP();
    printf("[3.8] cpasync: %8.3f ms | %.2f TFLOPS\n",
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
