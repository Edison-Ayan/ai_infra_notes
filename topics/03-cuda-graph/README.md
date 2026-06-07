# 03 · CUDA Graph

承接 topic 01 的 **launch bound**（200 个 tiny kernel，GPU 时间线全是缝，42% 时间在等 CPU 逐个下发）。
CUDA Graph 是它的正解：**把一串 kernel "录"成一张图，之后一条 `cudaGraphLaunch` 整张重放**，省掉逐个 launch 的 CPU 开销。

环境：RTX 4060 Laptop (sm_89)、CUDA 12.9。

## labs

| 文件 | 内容 |
|---|---|
| [labs/graph_lab.cu](labs/graph_lab.cu) | baseline(逐个 launch) vs graph(录一轮+重放)，20 kernel/轮 × 2000 轮 |
| [labs/explicit_graph.cu](labs/explicit_graph.cu) | 手动建图(`cudaGraphAddKernelNode`)搭菱形 DAG，体会 graph=节点+依赖边 |
| [labs/overlap.cu](labs/overlap.cu) | 小 kernel 让独立节点真并发：串行 2T vs graph 并行 1T |
| [labs/vmm_graph.cu](labs/vmm_graph.cu) | CUDA VMM 保护 graph（torch_memory_saver 机制）：换物理显存、VA 不变、graph 不重捕获仍正确 |

```bash
cd labs && ./build.sh && ./graph_lab
./build.sh explicit_graph && ./explicit_graph
./build.sh overlap && ./overlap
./build.sh vmm_graph && ./vmm_graph     # 用 driver API，build.sh 自动加 -lcuda
```

## 实验①：CUDA Graph 消除 launch bound

场景：每轮连发 20 个极小 kernel（NELEM=1024，单个几 µs），跑 2000 轮 = 40000 次 kernel。

| | baseline | graph |
|---|---|---|
| 时间 | 69.5 ms | 38.9 ms（**1.78×**） |
| 每 kernel 开销 | 1.74 µs | 0.97 µs |
| **CPU 侧 launch 调用** | `cudaLaunchKernel` **40000 次** | `cudaGraphLaunch` **2000 次** |
| 单次调用 | 1751 ns | 中位 2340 ns |

**nsys 看到的机制**（`cuda_api_sum`）：
- baseline 发 40000 次 `cudaLaunchKernel`，光下发就 70 ms；
- graph 同样的活只发 2000 次 `cudaGraphLaunch`（一次重放整轮 20 个）→ **CPU API 调用降到 1/20**。

每轮账：baseline `20×1751ns≈35µs` 逐个下发 vs graph `1×2340ns≈2.3µs` 一次重放。省的就是 topic 01 时间线上的"缝"。

## CUDA Graph 使用说明（什么时候用）

1. **两步走**：`录制`(stream capture 或显式建图) + `cudaGraphInstantiate`(**有成本**，编译整张图) → 之后 `cudaGraphLaunch` 重放才便宜。**必须"同一串 kernel 重复很多次"才划算**——录一次、放千次。
2. **适用**：推理(模型结构固定)、训练 step(每步同样算子序列)、任何"固定 kernel 链 × 大量重复"。一次性/每次都不同的序列用它反而亏。
3. **它省的是 CPU launch 开销，不是 GPU 计算**：kernel 本身很大(GPU-bound)时 launch 开销占比小、graph 几乎没用。**只有 launch bound(kernel 又多又小)收益才大。**
4. **参数变**：拓扑固定，但 kernel 参数变(指针/size)可用 `cudaGraphExecUpdate` 更新，不必重新 instantiate。

### stream capture 用法（本 lab 用的）
```c
cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
for (...) kernel<<<g,b,0,s>>>(...);          // 这些 launch 不真执行，只被"录"下来
cudaStreamEndCapture(s, &graph);
cudaGraphInstantiate(&gexec, graph, ...);    // 编译成可执行图(一次)
for (iter) cudaGraphLaunch(gexec, s);        // 重放(很多次)
```

## 实验②：手动建图——graph 的本质是依赖 DAG

stream capture 录的是"一条线"；手动建图能精确表达 **节点(kernel) + 边(依赖)** 的有向无环图。
搭一个菱形：A→{B,C}→D，其中 B、C 只依赖 A、彼此无边 → 标记为独立。

核心 API（一个函数表达一个节点+它的依赖）：
```c
cudaGraphAddKernelNode(&nB, graph, deps, numDeps, &params);
//                      存handle 哪张图  依赖谁  几个依赖  跑哪个kernel(cudaKernelNodeParams)
```
| 节点 | deps（依赖谁）= DAG 的边 |
|---|---|
| A | `nullptr,0`（无依赖，先跑） |
| B | `{A}` |
| C | `{A}`（和 B 无边 → 独立） |
| D | `{B,C}`（等两者） |

实例化+重放和 capture 完全一样（`cudaGraphInstantiate` + `cudaGraphLaunch`）。

### 关键认知
1. **"独立"≠ 一定并行**：图表达了 B、C 可并行 + D 必须等两者；但**实际重叠需 GPU 有空位**。本 lab 每 kernel 1M 元素=4096 block 占满 24 SM → B、C 照样排队跑。想看真重叠得把 kernel 缩小(留出空闲 SM)。**依赖允许 + 硬件有余量，两条件都满足才重叠。**
2. **nsys 看 graph 的坑**：默认把整张 graph 当**一个节点**记录，`cuda_gpu_kern_sum` 看不到图内 kernel。要加 `--cuda-graph-trace=node` 才拆到单个 kernel。
3. **capture vs 手动**：90% 用 capture(省事，已有代码包一下)；手动建图主要用于精确表达并行依赖/动态拼图——但理解它才真懂"graph 是 DAG"。

## 实验③：让独立节点真并发——补上"重叠"

实验②里 B、C 标了独立却没重叠(每 kernel 1M 元素占满 24 SM)。这里用"自旋" kernel：
只发 1 block(占 1 SM)、block 内空转一阵 → 两个独立 kernel 各占 1 SM、GPU 有大把空位 → 同时跑。

| | 时间 | |
|---|---|---|
| 单个 spin | 19.8 ms | 基准 T |
| 串行(一条流跑2次) | 39.7 ms | ≈2T，没重叠 |
| graph 2 个独立节点 | 19.8 ms | **≈T，2× 重叠** |

**nsys 硬证据**(`cuda_gpu_trace` 的 `Start` 列)：
```
串行:  #3 start=343.53ms ; #4 start=363.37ms  → 差 20ms(一个duration) = 串行接力
graph: #7 start=403.22ms ; #8 start=403.22ms  → 起始完全相同 = 同时开跑 = 重叠
```

### 结论：graph 独立 + GPU 有空位 = 真并发（缺一不可）
| | 实验②(1M元素) | 实验③(1 block自旋) |
|---|---|---|
| GPU 占用 | 4096 block 占满 24 SM | 每个 1 block 只占 1 SM |
| 实际重叠 | ❌ 没空位 | ✅ 有空位 → 2× |

**判断 kernel 是否真并发的硬指标**：比 nsys 里两个 kernel 的 `Start` 时间——几乎相同=并发，差一个 duration=串行。

## 实验④：CUDA VMM 保护 graph —— torch_memory_saver 的核心机制

**痛点**：CUDA Graph 在捕获时把每个 kernel 用的**显存指针(虚拟地址)焊死**。普通 `cudaFree/cudaMalloc`
会让地址变 → graph 失效 → 必须重新捕获(慢，几秒~几十秒)。
**场景**：推理服务/RLHF 想把空闲的推理引擎几十 GB 显存临时让给训练，但它捕获了大量 CUDA Graph，不想重捕获。

**解法（torch_memory_saver）**：用 CUDA 虚拟内存管理(VMM, `cuMemAddressReserve/cuMemCreate/cuMemMap`)
把"虚拟地址"和"物理显存"**解耦成两层**：
```
虚拟地址 VA (graph 焊死的指针)   ←─ 始终不变 ─→  cuMemAddressReserve 占着
        │ cuMemMap / cuMemUnmap
物理显存 (真正占 GB 的)          ←─ 可还可拿 ─→  cuMemCreate / cuMemRelease
```
- **pause(让显存)**：`cuMemUnmap`+`cuMemRelease` 还掉物理页，**保留 VA**。
- **resume**：`cuMemCreate`+`cuMemMap` 把新物理页映射回**同一个 VA**。

VA 全程不变 → graph 焊死的指针仍有效 → **无需重新捕获**。

**[labs/vmm_graph.cu](labs/vmm_graph.cu) 实测**（捕获写 7.0 的 graph → 释放物理显存 → 重建 → 清零 → 重放同一图）：
```
VA 0x798129a00000          (从头到尾不变)
replay#1: p[0]=7.0 ✓
[释放物理显存→重建→映射回同一 VA]
memset 清零: p[0]=0.0       ← 证明物理显存确实换了新的一块
replay#2: p[0]=7.0 ✓        ← 同一张 graph 没重捕获，照样写对！
```
> **结论**：物理显存换过一轮(清零证明)，graph 未重捕获仍正确——**VA 不变即可保护 graph**。
> 这印证了 graph 的本质：把"一串带固定指针的操作"录死，所以**指针稳定性是它的生命线**
> （也是 `cudaGraphExecUpdate` 改参数、torch_memory_saver 保地址这些工具存在的原因）。

## 下一步

- [ ] `cudaGraphExecUpdate` 实测：参数变时复用已实例化的图（避免重新 instantiate）
- [ ] 在真实 GEMM 链/多 stream 上对比；和 topic 01 的 launch-bound 段串起来
