# 01 · Nsight Systems / Compute 性能分析

用一个自写的 `gemm_lab`（naive / tiled / register-blocked GEMM + launch-bound 演示）打通两把尺子：
- **nsys**（Nsight Systems）：系统级**时间线**，看「谁占时间、瓶颈在哪一段」。
- **ncu**（Nsight Compute）：单 kernel **微观**，看「这个 kernel 内部卡在哪个硬件单元」。

环境：RTX 4060 Laptop (sm_89)、CUDA 12.9、Nsight Systems 2026.3.1、Nsight Compute 2025.2.1。

## labs

| 文件 | 内容 |
|---|---|
| [labs/gemm_lab.cu](labs/gemm_lab.cu) | 主实验台：H2D / naive / tiled / launch-bound / D2H 五段，各包 NVTX |
| [labs/gemm_lab_pinned.cu](labs/gemm_lab_pinned.cu) | 仅把 host 内存换成 `cudaMallocHost`（pinned），对比拷贝带宽 |
| [labs/gemm_lab_reg.cu](labs/gemm_lab_reg.cu) | 加一档 register-blocked GEMM（每线程 4×4 micro-tile） |

```bash
cd labs && ./build.sh && ./gemm_lab          # 编译+裸跑
# nsys: 时间线 + 统计表
nsys profile --trace=cuda,nvtx -o gemm_report ./gemm_lab
nsys stats --report nvtx_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum gemm_report.nsys-rep
# ncu: 单 kernel 全量 section（需 GPU 计数器权限，见下）
ncu --set full -k "regex:gemm_naive|gemm_tiled" -c 3 -f -o gemm_kernels ./gemm_lab
ncu --import gemm_kernels.ncu-rep --page details -k gemm_naive -c 1
```

## 实验①：pageable vs pinned 内存

只改 host 内存分配（`malloc` → `cudaMallocHost`），其余不动，公平对比拷贝带宽。

| 方向 | pageable | pinned | 提升 |
|---|---|---|---|
| H2D | 10.8 GB/s | 11.7 GB/s | ~1.1× |
| D2H | **4.1 GB/s** | 11.8 GB/s | **~2.9×** |

- **结论**：pinned（锁页）内存让 DMA 直达，省掉 driver 内部锁页缓冲的中转拷贝。pageable 的 D2H 尤其慢。
- **代价**：占真实物理 RAM、不可换出、分配慢 → 只对反复参与拷贝的 buffer 用。
- 法则：**一次只改一个变量**，GEMM kernel 时间两版完全一致，才能把差异干净归因到内存类型。

## 实验②：ncu 深挖——为什么 FP32 只跑到 6%

`ncu --set full` 抓 naive / tiled，关键指标：

| 指标 | naive | tiled |
|---|---|---|
| **FP32 峰值利用率** | **6%** | ~7% |
| L1/TEX Cache 吞吐 | 99% 🔴 | 97% 🔴 |
| DRAM 吞吐 | 0.6% | 0.8% |
| Compute(SM) 吞吐 | 99% ⚠️ | 97% ⚠️ |
| Issued Warp/调度器 | 0.28 | 0.31 |
| 主导 stall | LG Throttle 68% | MIO Throttle 54% |

根因（看 naive 内层 `sum += A[k]*B[k]`）：
1. **算访比太低**：每 1 次 FFMA 要 2 次 global load → 算力指令天生只占 1/3。
2. **L1 管线先撑爆**：load 命中 L1（86%）但 L1 吞吐打满 → `LG Throttle`，FFMA 饿死。**不是 DRAM bound**（DRAM 才 0.6%）。
3. **无 ILP**：单累加器 `sum` 串行依赖，调度器找不到独立 FFMA → 72% 时间空转。
4. tiled 用 shared memory 把瓶颈从 LG 挪到 MIO，但 1 load:1 FMA 没变，仍卡。

> ⚠️ **陷阱**：`Compute (SM) Throughput 99%` ≠ 算力跑满。SOL 吞吐取 `max(各管线)`，这里 99% 来自 L1 管线。**判断算力要看 roofline 的「% of FP32 peak」，不是 SM Throughput。**

### ncu 权限（ERR_NVGPUCTRPERM）

默认 `RmProfilingAdminOnly: 1`，普通用户读不了 GPU 计数器。永久放开：
```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
sudo update-initramfs -u && sudo reboot
# 验证：cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly  → 0
```

## 实验③：Nsight Systems GUI 读时间线

`nsys-ui gemm_report.nsys-rep`，自上而下行树：CPU 线程 / CUDA API / GPU(stream) / Memory / **NVTX**。

- **NVTX 行**最关键：5 个彩块直接标出逻辑阶段，`2_naive_gemm` 最宽、`4_launch_bound` 窄到几乎看不见。
- **GPU-bound 长相**：NVTX 区间正下方 GPU 行被 kernel 实心块填满、无空隙。
- **launch-bound 长相**：双击放大 `4_launch_bound_demo`，GPU 行是 200 个小块、**块间全是缝**——GPU 在等 CPU 串行下发（对照 CUDA API 行的一排 `cudaLaunchKernel`）。GPU 实忙 229µs / 整段 389µs ≈ 40% 在空转。
- 操作：`Ctrl+滚轮` 缩放，拖选量时长，`F` 聚焦。

## 实验④：register blocking——把 6% 推到 36%

每线程算 TM×TN=4×4 个输出，把 A 的一条/B 的一条读进寄存器做 16 次 FFMA（16 个独立累加器）。

| 指标 | naive | tiled | **reg** |
|---|---|---|---|
| 时间 | 23.9 ms | 18.4 ms | **4.07 ms** |
| TFLOPS | 0.72 | 0.94 | **4.22** |
| **FP32 峰值%** | 6% | 7% | **36%** |
| Compute(SM) 吞吐 | 99% | 97% | **54.6%** |
| Executed IPC | 1.14 | 1.24 | **1.96** |
| Issued Warp/调度器 | 0.28 | 0.31 | **0.49** |
| Occupancy | 99% | 99% | **64.8%** |
| 主导 stall | LG 68% | MIO 54% | MIO 35% |

- **5.9× / 4.5× 加速**，FP32 利用率 6×。
- **Compute(SM) 从 99% 降到 54.6% 是好事**：不再有单一管线吃满，负载均衡了。判断好坏看 FP32%。
- **ILP 起来了**：16 个独立累加器 → IPC 翻倍、发射间隔 42→15.9 cycle。
- **Occupancy 反降到 64.8% 也没关系**：每线程吃更多寄存器 → 每 SM warp 变少。**register blocking 本质是拿 occupancy 换 ILP**，这里换得值。「高 occupancy=高性能」是误区。
- 仍是 MIO bound（35%），下一档：`float4` 向量化、shared memory double buffering、更大 tile、消 bank conflict、warp tiling → 目标向 cuBLAS 看齐。

## 实验⑤：float4 向量化——追到 cuBLAS 88%

[labs/gemm_lab_float4.cu](labs/gemm_lab_float4.cu)，在 reg 基础上三处向量化 + 放大 tile 到 128×128/8×8。
核心：reg 的瓶颈是访存指令太多（MIO throttle），`float4` 让一条指令搬 4 个 float，访存指令数 ÷4。

| 指标 | reg | **float4** | 变化 |
|---|---|---|---|
| TFLOPS | 4.22 | **6.13** | 61%→**88%** cuBLAS |
| FP32 峰值% | 36% | **53%** | 算力喂得更饱 |
| L1/TEX 吞吐 | 88% | 82.9% | MIO 减压 |
| Executed IPC | 1.96 | **2.44** | |
| Occupancy | 64.8% | **32%** | 寄存器 56→127/线程 |

三处向量化：
- **(a) global→shared**：`LDG.128` 读 A（4 连续 float），但**转置散写**故只能标量写；B 不转置 → 读写都 float4。
- **(b) shared→register**：`LDS.128` 读 regM/regN，直接给瓶颈 MIO 管线减压。**靠 (a) 的转置**让一个线程的 8 个 M 值地址连续，才能 float4 读——转置的全部意义在此。
- **(c) 写回**：`STG.128`。

反直觉再现：**Occupancy 又砍半到 32%（127 寄存器/线程）却更快**。因为 64 个独立累加器提供海量 ILP（不靠多 warp 藏延迟）+ float4 让每条访存指令干 4 倍活。判断始终看 FP32%/TFLOPS，不是 occupancy。

⚠️ 前提：`float4` 要 16B 对齐（`cudaMalloc` 256B 对齐 + N=2048 是 4 倍数 + shared 索引都是 4 倍数才成立）。

离 cuBLAS 最后 12%：double buffering（搬运/计算重叠，消同步气泡）、warp tiling（消 bank conflict）。

## 实验⑥：两次 double buffering 翻车——踩坑实录

试图把 float4(6.13) 往 cuBLAS 推，做了两版双缓冲，**都比 float4 慢**。两个负结果都极有价值。

| 指标 | float4 | db(手动预取) | cpasync |
|---|---|---|---|
| TFLOPS | **6.13** ⭐ | 5.40 | 5.12 |
| 寄存器/线程 | 127 | 129 | **92** |
| Occupancy | 32% | **16.7%** | 33.3% |
| L1/TEX 吞吐 | 82.9% | 72.6% | **92.8%** |
| 卡在哪 | 均衡 | **寄存器悬崖** | shared 读(标量 regM) |

### 坑①：寄存器悬崖（db 版，[labs/gemm_lab_db.cu](labs/gemm_lab_db.cu)）
手动寄存器预取双缓冲，多用了 2 个预取寄存器(`ldA/ldB`)：127→**129**。

**为什么 2 个寄存器让性能暴跌**：occupancy 是**量子化**的——每 SM 能放几个 block 由 `寄存器总量 ÷ (每线程寄存器×blockSize)` **向下取整**。
- 127×256 = 32512 → 65536/32512 = 2.01 → **2 block/SM**
- 129×256 = 33024 → 65536/33024 = 1.98 → **1 block/SM**

就跨过这个整数阈值，occupancy 16.7%(腰斩)。预取省的延迟，远不够补 warp 减半的损失。

> **怎么发现的**：优化后变慢 → ncu 看到「寄存器微动 127→129、occupancy 却腰斩」这种**不成比例**变化 → 量子化阈值的指纹 → 算 `65536÷(reg×256)` 的 block 数坐实。

### 坑②：转置 vs cp.async 不可兼得（cpasync 版，[labs/gemm_lab_cpasync.cu](labs/gemm_lab_cpasync.cu)）
`cp.async` 让 global 直接拷进 shared、绕过寄存器，**确实根除了悬崖**（寄存器 129→92、occupancy 回到 33%）✓。
但 cp.async 要求两端连续、**不能转置** → As 改非转置 → regM 退回标量读(8×`LDS.32`) → `L1/TEX 82.9%→92.8%`，float4 压下的访存瓶颈又顶回来 ✗。

```
float4 : 转置→regM能float4(访存少) ✓  预取吃寄存器→悬崖     ✗
cpasync: cp.async不吃寄存器(无悬崖) ✓  不能转置→regM标量     ✗
```
两条路各解决对方的问题、又各踩对方的坑。想兼得需 `ldmatrix` / swizzled 无冲突布局——CUTLASS 级工程量。

### 结论
**手写 FP32 GEMM 的甜点就在 float4/register-blocking（~88% cuBLAS）**；再往上收益递减、按下葫芦浮起瓢。到此该用 cuBLAS，或换 Tensor Core 赛道。**ncu 让每次失败都可解释**，比盲目变快更有价值。

## 实验⑦：bank conflict 三药方对比——对症才有效

float4 的瓶颈 ncu 查出是 **shared load 5-way bank conflict**（40% wavefront 浪费）。
写三版同台对比（[labs/gemm_bank.cu](labs/gemm_bank.cu)，算法/tile 全同，只差 shared 布局）：

| | v0 baseline | pad | swizzle |
|---|---|---|---|
| TFLOPS | 6.14 | 6.19 | 6.35 |
| **load 冲突** | 5.0-way | **5.0-way** ✗ | **5.0-way** ✗ |
| store 冲突 | 2.4-way | **消失** ✓ | **消失** ✓ |
| FP32% | 53% | 54% | 55% |

### 核心发现：bank conflict 分两类，padding/swizzle 只治一类
- **行间冲突(cross-row)**：相邻行 stride 是 bank 数(32)倍数 → 不同行落同 bank。我们的 **store**(转置散写跨行)属此 → padding/swizzle **能治**(实测 store 2.4-way 消失)。
- **warp 内结构性冲突**：同一访问里 warp 的线程按固定跨步访问同一行。我们的 **load**(`Bs[k*BN+threadCol*8]`，16 threadCol 按 ×8 跨步)属此 → padding/swizzle **都治不了**。
  - padding 只改行 stride，碰不到同一行内 threadCol 的冲突；
  - swizzle 按 k 排，固定 k 时所有线程同一套排列，threadCol 冲突只是整体平移 bank，没打散。

> **理论地板**：16 线程 × float4(4) = 64 次 bank 访问 / 32 bank = 最优也 **2-way**。要从 5-way 逼近 2-way，必须改**线程→列的映射**本身 → 只有 **warp tiling**。

### 三药方利弊
| 药方 | 利 | 弊 |
|---|---|---|
| padding | 改动最小(行宽+4)；消行间冲突 | 对 warp 内结构冲突无效；费一点 shared |
| swizzle | 不费 shared；也消 store；略快 | 索引易错；按 k 排对本例 load 仍无效 |
| warp tiling | 唯一能破 5-way load 结构冲突 + 提高算访比 | ~120 行重写，最易 bug，要重平衡寄存器 |

**教训**：优化要**对症**——本例主瓶颈是第 2 类(warp 内结构 load 冲突)，padding/swizzle 只顺手清了 store(第 1 类)，所以三招只换来 +3%。**先用 ncu 把冲突分类，再选药方。**

## 参照基准：手写 GEMM 该追谁

两个参照点（同尺寸 N=2048 FP32，本机实测，见 [labs/cublas_ref.cu](labs/cublas_ref.cu)）：

| 实现 | 时间 | TFLOPS | 占 cuBLAS | 占硬件峰值 |
|---|---|---|---|---|
| naive | 23.9 ms | 0.72 | 10% | 6% |
| tiled | 18.4 ms | 0.94 | 14% | 8% |
| reg | 4.07 ms | 4.22 | 61% | 36% |
| **float4 (本实验最优)** | 2.80 ms | **6.13** | **88%** | 53% |
| **cuBLAS** ⭐ | 2.47 ms | **6.95** | 100% | ~58% |
| 硬件 roofline | — | ~12 | — | 100% |

- **cuBLAS 才是现实目标**，不是硬件峰值。纯 FP32 SGEMM 受访存层级限制，连 cuBLAS 也只摸到峰值 ~58%；roofline 是天花板，谁都到不了。
- reg 已到 cuBLAS 的 61%——剩下差距靠 `float4`/double buffering/warp tiling 等工业技巧。
- **前提**：cuBLAS 这 6.95 是**纯 FP32** 路径。若允许 TF32 / tensor core（精度略降）会跳到几十 TFLOPS，那是另一条赛道；手写 FP32 CUDA core 就该和 cuBLAS 的 FP32 路径比。
- 跑法：`nvcc -O3 -arch=sm_89 -I<cuda>/include cublas_ref.cu -o cublas_ref -lcublas && ./cublas_ref`。

## 两个最该记住的反直觉

1. **`Compute (SM) Throughput` 高 ≠ 算力跑满**，要看 roofline `% of FP32 peak`。
2. **高 occupancy ≠ 高性能**，register blocking 降 occupancy 换 ILP 反而更快。
