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
| [labs/kv_cache.cu](labs/kv_cache.cu) | KV cache 显存账(随 batch×seqlen 爆) + 为什么必须存(不存 O(N²)) |
| [labs/paged_attn.cu](labs/paged_attn.cu) | PagedAttention 模拟：预留连续块 vs 按页分配，利用率 20%→98% |

```bash
LIB=$HOME/miniconda3/envs/ai_infra/targets/x86_64-linux/lib
cd labs && ./build.sh && LD_LIBRARY_PATH=$LIB ./decode_batch
./build.sh kv_cache && LD_LIBRARY_PATH=$LIB ./kv_cache
./build.sh paged_attn && ./paged_attn
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

## 实验②：KV cache——为什么必须存 + 它才是 batch 的天花板

decode 每步要对"之前所有 token"做 attention，需它们的 K/V。

**Part A 显存账**(Llama-7B 量级 L=32,D=4096,FP16)：每 token KV = `2×32×4096×2B` = **0.52MB**，随 batch×seqlen 线性涨：

| batch | seqlen | KV cache | |
|---|---|---|---|
| 1 | 2048 | 1.07GB | ok |
| 1 | 32768 | 17GB | 爆(长上下文) |
| 16 | 2048 | 17GB | 爆(大 batch) |
| 64 | 2048 | 69GB | 爆 |

→ **回答了实验①的悬念**：batch 越大吞吐越高，但 **KV cache(不是权重)随 batch 线性涨，8GB 卡 batch=16 就爆**。KV cache 是限制 batch 的真正瓶颈。

**Part B 为什么必须存**：不存就每步重算整段 prefix 的 K/V → O(N²)。

| | 耗时 | 计算量 |
|---|---|---|
| 不存(重算整段) | 130ms | 4.4 TFLOP ∝T² |
| 存(只算新 token) | 40ms | ~0 ∝T |

不存的计算量是存的 **256×**(T=512)，序列越长越爆炸 → 长序列生成不存不可行。
> 跨主题细节：墙上时间只差 3.2×(非 256×)，因"存"那版是 512 个极小 GEMV、被 launch 开销卡住——正是 topic 03 **CUDA Graph** 治的；真实引擎用 graph 打包 decode 的小 kernel。

**矛盾**：必须存(否则算不动) ↔ 它吃显存限制 batch → **PagedAttention** 来高效管这块显存。

## 实验③：PagedAttention——像 OS 分页一样管 KV cache（vLLM 核心）

传统给每个请求预留一整块"连续"显存、大小按"可能的最大长度"。但请求大多很短 → 大半空着 = 内部碎片。
PagedAttention：KV cache 切固定小页(16 token)，按需分配、可非连续，只末页有零头。

同 16384 槽显存、同一批请求(平均 347，最大可能 2048)：

| | 传统·预留连续块 | PagedAttention·按页 |
|---|---|---|
| 服务请求数 | 8 | **45** |
| 利用率 | **20.3%** | **97.7%** |

**同显存塞 5.6× 请求**。机制 = OS 虚拟内存分页：每请求一张 **block table(页表)** 把逻辑 token 位置→物理页；代价是 attention kernel 要能从分散页 gather K/V（真正的 PagedAttention kernel）。

**闭合整条推理线**：
```
① batching:      batch↑→吞吐↑(decode访存bound)
② KV cache:      但KV cache吃显存→batch被卡(8GB batch=16爆)
③ PagedAttention: 利用率20%→98%→同显存塞5.6×请求→batch↑→吞吐↑
```
vLLM 论文 ~24× 吞吐，很大一块来自这——利用率上去了，batch 才开得大。

## 下一步

- [ ] continuous batching：请求动态进出(到达/结束时机不同)怎么实时拼批
- [ ] KV cache 量化(INT8/FP8)：省显存→更大 batch（接 topic 05）；GQA 减 KV 头数
- [ ] 真正的 PagedAttention kernel：从非连续页 gather K/V 做 attention
