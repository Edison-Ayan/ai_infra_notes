# 06 · FlashAttention

集大成的一课：把 topic 01(tiling/喂饱计算单元) + topic 02(tensor core) + 算子融合 用到 attention 上。
attention 是 Transformer 训练/推理最热的 kernel。环境：RTX 4060 Laptop (sm_89)、CUDA 12.9。

## 问题：标准 attention 为什么慢/费显存

```
S = QKᵀ·scale  [N×N]   ← 物化 N×N 分数矩阵
P = softmax(S)  [N×N]   ← 又一个 N×N
O = P·V         [N×d]
```
1. **显存 O(N²)**：N=32768 时 S 就 4.3GB → 长上下文爆。
2. **访存 bound**：S、P 写回/读出 HBM(N×N 字节)，attention 大部分时间在搬这两矩阵。

## 解法：online softmax + 分块，永不物化 N×N

softmax 要 `max M` 和 `sum L=Σexp(s-M)`，得看完整行 → 逼你物化。**online softmax** 边走边算：
维护 running `(m,l,acc)`，每来一块新分数：
```
m_new=max(m,块max); corr=exp(m-m_new)   ← 灵魂:旧max变了就校正旧结果
l = l·corr + Σexp(s-m_new)
acc = acc·corr + Σexp(s-m_new)·V
```
最后 `O=acc/l`。`corr=exp(m_old-m_new)` 把旧累加精确缩回新基准 → **和一次性 softmax 数学等价，但永不物化 N×N → 显存 O(N)**。

## labs

| 文件 | 内容 |
|---|---|
| [labs/flash.cu](labs/flash.cu) | 朴素(物化 N×N) vs Flash(流式 online softmax)，验证一致 + 显存对比 |

```bash
cd labs && ./build.sh && ./flash
```

## 实验①：online softmax 正确 + 显存 O(N²)→O(N)

N=4096, d=64：

| | 朴素(物化N×N) | Flash(流式) |
|---|---|---|
| 额外显存 | 67MB(N×N) | **0** |
| 时间 | 21ms | 47.7ms |
| 输出 | 基准 | 相对误差 **1.82e-06** ✓ |

**两个算法层面胜利**：
1. **online softmax 正确**(误差 1.82e-6)：流式 (m,l,acc)+corr 校正，和一次性 softmax 等价，永不物化 N×N。
2. **显存 O(N²)→O(N)**：朴素 N=4096 要 67MB、N=32768 要 4.3GB(再乘 heads×layers 直接爆)；Flash 0。**这是 FlashAttention 让长上下文可能的根本。**

**诚实说**：本 lab 的 flash 是最朴素流式版(一线程一 query 行、标量点积、acc 在 local memory spill、并行度低)，**只展示算法和显存，没做速度优化**，所以反而更慢(47.7 vs 21ms)。

**真正 FlashAttention 的速度来自 "IO-aware"**：Q/K/V 块搬进 SRAM 分块算、最小化 HBM 读写(attention 访存 bound，省 IO=省时间) + tensor core 算块内矩阵乘 + 二维分块。
→ 两大收益：**显存 O(N)**(算法定，已展示) + **速度**(IO-aware 实现定，下一课)。

## 下一步

- [ ] tiled FlashAttention：Q/K/V 块进 shared memory，二维分块，实测速度超朴素（接 topic 01 tiling）
- [ ] 用 tensor core(WMMA) 算块内 QKᵀ 和 PV（接 topic 02，收尾 tensor core）
- [ ] causal mask（decode 只看前面）、多头
- [ ] 联系 FlashAttention-2/3 的改进（更好的并行划分、warp specialization、FP8）
