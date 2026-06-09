# 05 · 量化 (Quantization)

降精度换显存/带宽/算力。跟 [topic 04 推理](../04-inference/) 同一条逻辑：先用 profiler 定位瓶颈（显存/访存/算力），再用对应的量化方案下药。
环境：RTX 4060 Laptop (sm_89)、CUDA 12.9。

## 核心心智图

```
量化 = 把 FP 值映射到低比特整数/低位浮点
  scale(+zero-point) 是桥梁： q = round(x/scale) ; deq = q*scale
三个旋钮：
  ① 比特数   FP16→INT8→INT4→FP8→FP4   (越低越省、越易掉精度)
  ② 粒度     per-tensor / per-channel / per-group  (越细越准、scale 越多)
  ③ 量谁     weight-only(W4A16) / 权重+激活(W8A8) / KV cache
对症：访存 bound(decode/低batch)→压权重(INT4)；算力 bound(prefill/高batch)→FP8
```

## labs

| 文件 | 内容 |
|---|---|
| [labs/quant_basics.cu](labs/quant_basics.cu) | 对称量化 + 粒度对比：看离群值如何摧毁 per-tensor 低比特 |

```bash
cd labs && ./build.sh && ./quant_basics
```

## 实验①：离群值如何摧毁低比特量化（per-tensor vs per-channel/group）

128 通道×1024 权重，其中 4 个"离群通道"幅度 ×50。对称量化→反量化测相对 L2 误差：

| 方法 | 全体误差 | **正常通道误差** |
|---|---|---|
| per-tensor INT8 | 4.70% | 41.90% |
| per-tensor INT4 | 18.73% | **100.00%** 💀 |
| per-channel INT4 | 14.57% | 14.06% |
| per-group INT4 (G=128) | 12.06% | 11.62% |

**三个洞见**：
1. **离群值是低比特头号杀手**：per-tensor INT4 下，4 个离群通道撑大全局 scale → 正常权重 `round(小值/巨大scale)=0` → 正常通道 100% 全毁。
2. **聚合指标会骗人** ⭐：per-tensor INT4 全体误差才 18.73% 看着"还行"，但正常通道已 100% 崩溃——离群通道幅度大、主导 L2 范数，**掩盖了局部灾难**。（同 topic 01"别信 SM Throughput"：看错指标会误判。）
3. **per-channel/group 救回**：每通道/每组各一个 scale → 正常通道不被离群连累，100%→14%→11.6%。这就是 **GPTQ/AWQ 都用 per-group** 的原因；粒度越细越准（代价：多存 scale）。

> 连回 topic 04 讨论：激活的离群值比权重更凶，所以 INT4 常"只量化权重、保激活 FP16(W4A16)"。

## 学习路线

- [x] 量化基础：scale/zero-point、对称/非对称、粒度、离群值（实验①）
- [ ] **Weight-only INT4 (W4A16) vs FP8 (W8A8)**：按瓶颈选型（访存 vs 算力，对应 topic 04 的两区间）
- [ ] **GPU 上跑量化 GEMM**：INT4 权重反量化 kernel / Marlin，实测 decode 带宽收益（接 topic 04）
- [ ] **KV cache 量化**：INT8/FP8 KV cache，省显存→更大 batch
- [ ] **校准 calibration**：怎么定 scale（min-max / 百分位 / GPTQ 的逐列校正）
- [ ] **FP8 细节**：E4M3/E5M2、per-tensor/block scaling、Transformer Engine

### 选型速查（按瓶颈）
| 瓶颈/场景 | 方案 | 为什么 |
|---|---|---|
| decode / 低 batch / 本地大模型 | **W4A16 (INT4 weight-only)** | 访存 bound，压权重 4× 加速搬运；任何卡能跑 |
| prefill / 高 batch / 大吞吐服务 | **W8A8 (FP8)** | 算力 bound，FP8 tensor core 2× 吞吐；需 Ada/Hopper |
| 显存装不下 | INT4 权重(省 4×) | 最省显存 |
| KV cache 太占 | INT8/FP8 KV cache | 腾显存给更大 batch |
