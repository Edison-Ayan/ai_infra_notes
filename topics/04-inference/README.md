# 04 · 推理系统内部

LLM 推理服务(vLLM/SGLang 这类)的核心机制，跟 [topic 03 的 torch_memory_saver](../03-cuda-graph/) 一条线。
环境：RTX 4060 Laptop (sm_89, 8GB)、CUDA 12.9、cuBLAS。

## 最该先懂的：推理分两个阶段，性质完全相反

```
Prefill(处理 prompt): 一次算很多 token → 大矩阵乘 → 算力 bound
Decode(逐 token 生成): 一次只出 1 token → 矩阵×向量 → 访存 bound  ← 服务经济学的根
```
**decode 每步要把整个模型权重从显存搬一遍,只为产出 1 个 token** → 算访比极低 → 卡显存带宽。

## labs

| 文件 | 内容 |
|---|---|
| [labs/decode_batch.cu](labs/decode_batch.cu) | 一层线性 Y=W@X，batch 1→512，看每 token 耗时随 batch 暴跌 |

```bash
cd labs && ./build.sh && LD_LIBRARY_PATH=$HOME/miniconda3/envs/ai_infra/targets/x86_64-linux/lib ./decode_batch
```

## 实验①：decode 是访存 bound → batching 几乎白赚吞吐

W=4096×4096 FP16(33.6MB)，模拟一层线性，batch=同时解码的请求数：

| batch | 总耗时 | 每 token | 吞吐(token/ms) | 区间 |
|---|---|---|---|---|
| 1 | 80.6us | 80.6us | 12.4 | **访存 bound** |
| 8 | 85.7us | 10.7us | 93 | (batch 几乎免费) |
| 32 | 96.6us | 3.0us | 331 | |
| 128 | 125us | 0.98us | 1022 | **算力 bound** |
| 256 | 221us | 0.87us | 1157 | (batch 开始线性花钱) |
| 512 | 444us | 0.87us | 1153 | |

**两个区间是全部洞见**：
- **访存 bound 区(B≤~32)**：batch 涨 32×、总耗时只从 80→96us 几乎不动——因为时间全花在"搬一次 33.6MB 权重"上，搬一次够 32 个 token 用。每 token 耗时暴跌 27×、吞吐涨 27×。**纯赚。**
- **算力 bound 区(B≥128)**：权重搬运已摊薄到可忽略，总耗时随 batch 线性增长，每 token 触底 ~0.87us，再加 batch 也降不动。

**吞吐 12 → 1150 token/ms ≈ 93×**——这就是为什么 vLLM/SGLang 拼命 **continuous batching**：把在线请求凑大 batch，让一次权重搬运服务尽量多 token。

> 注：B=64 附近数值略抖(cuBLAS 切换内部 kernel)，但"先平后斜"的两区间趋势很干净。

## 下一步

- [ ] **PagedAttention**：batch 越大 KV cache 越占显存→会爆；怎么像 OS 分页一样管 KV cache 显存（vLLM 核心）
- [ ] **KV cache** 本身：为什么存它(避免重算)、显存怎么算、为什么 decode 因它访存 bound
- [ ] continuous batching：请求动态进出怎么拼批
- [ ] 量化：W/激活/KV cache 降精度，喂 tensor core 提吞吐
