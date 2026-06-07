# 03 · CUDA Graph

承接 topic 01 的 **launch bound**（200 个 tiny kernel，GPU 时间线全是缝，42% 时间在等 CPU 逐个下发）。
CUDA Graph 是它的正解：**把一串 kernel "录"成一张图，之后一条 `cudaGraphLaunch` 整张重放**，省掉逐个 launch 的 CPU 开销。

环境：RTX 4060 Laptop (sm_89)、CUDA 12.9。

## labs

| 文件 | 内容 |
|---|---|
| [labs/graph_lab.cu](labs/graph_lab.cu) | baseline(逐个 launch) vs graph(录一轮+重放)，20 kernel/轮 × 2000 轮 |

```bash
cd labs && ./build.sh && ./graph_lab
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

## 下一步

- [ ] 显式建图 API（`cudaGraphAddKernelNode` + 依赖边）——比 capture 更细的控制
- [ ] `cudaGraphExecUpdate` 实测：参数变时复用已实例化的图
- [ ] 在真实 GEMM 链/多 stream 上对比；和 topic 01 的 launch-bound 段串起来
