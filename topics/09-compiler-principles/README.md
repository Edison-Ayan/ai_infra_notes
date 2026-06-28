# 09 · AI 编译器原理：算法与调度分离 + 渐进式 lowering

topic 07/08 是"会用"（Triton、torch.compile）；这一篇往下挖"懂原理"——所有 AI 编译器
（Halide / TVM / MLIR / Triton / XLA）共享的两块地基。两个 demo 都用 Triton 把抽象命题坐实。

环境：RTX 4060 Laptop (sm_89)、torch 2.6 + Triton 3.2。

## 地基一：算法与调度分离（algorithm / schedule separation）

Halide 2012 的核心思想，后来 TVM/Triton 全继承：

```
一个 kernel = 算法(算什么) + 调度(怎么算)
  算法：C = A @ B           —— 数学，固定，换调度不改结果
  调度：tile 多大/几个 warp/几级流水 —— 不改结果，只决定快慢
编译器的工作 = 固定算法，在调度空间里搜最快的那组（topic07 @autotune / topic08 max-autotune）
```

### 实验①：同一算法，调度差 4.7×

[labs/schedule_sweep.py](labs/schedule_sweep.py)：**同一个 `matmul_kernel`（算法一字不改）**，
手动喂 4 组调度（只改 BLOCK / num_warps / num_stages）。4096³ FP16：

| 调度 | TFLOPS |
|---|---|
| 16³ tile · 1 warp · 无流水 | 5.17 |
| 64² tile · 2 warp · 2 级流水 | 21.59 |
| **128² tile · 4 warp · 3 级流水** | **24.43** |
| 128×64 tile · 4 warp · 4 级流水 | 21.98 |

> **结论：算法没动一个字，最好/最差调度差 4.7×。** 性能全在调度里。
> 这正是为什么编译器值钱——算法谁都会写（`C=A@B`），难的是在巨大的调度空间里搜到那组最快的。
> 我 topic 01 手写 GEMM 就是**人肉**搜这个空间（reg blocking/float4/swizzle/double buffer），
> `@autotune` 是把它自动化。**算法与调度分离，就是"让搜索成为可能"的前提**。

### 实验③：亲手写 compute + schedule（TVM，把"看懂"变"会写"）

实验①是"看"调度影响性能（Triton 换 BLOCK）；这一篇用 **TVM 的显式调度原语亲手写**——
体会 Halide 的命题：**算法写一次，调度是另一套可独立改写的代码**。
[labs/tvm_schedule.py](labs/tvm_schedule.py)，CPU/LLVM target（调度原语和 GPU 一样，学原理够用）：

```python
# 算法层：只说"算什么"，一个字不提怎么循环/分块/并行
C = te.compute((N,N), lambda i,j: te.sum(A[i,k]*B[k,j], axis=k))
# 调度层：另一套代码，只重塑循环嵌套，不改数学
sch.reorder(i, k, j)                  # loop interchange → 内层连续访存
io,ii = sch.split(i,[None,32]); ...   # 分块提 cache 复用
sch.parallel(io); sch.vectorize(ji)   # 多核 + SIMD
```

同一个 `C=A@B`（1024³ FP32），只换调度：

| 调度 | ms | 加速 |
|---|---|---|
| 默认 naive `i,j,k` | 2270 | 1.0× |
| `+reorder(i,k,j)`（连续访存） | 83 | **27×** |
| `+tile+parallel+vectorize` | 20 | **113×** |

> **算法一个字没改，只改调度，113×。** 比实验①的 4.7× 还猛——CPU naive matmul 太烂，
> 衬得调度威力极大。调度后打印循环嵌套能看到 `T.parallel(32)` / `T.vectorized(32)`：
> **调度重塑的是循环、不是数学**。编译器(`@autotune`/AutoTVM/Inductor)的活，就是替你**自动写这套调度**。

> 环境坑（见 lab 顶部注释）：TVM 0.25 的 `apache-tvm-ffi` 自带 torch C-dlpack 扩展和 torch 2.6
> ABI 不兼容，`import tvm` 即崩 → 用 `sys.modules` 把那个可选扩展标 None 让它干净降级；
> 这版还把子包 `tir` 改名 `s_tir`、方法 `get_block` 改名 `get_sblock`（迁移期过渡命名）。

## 地基二：渐进式 lowering / dialect 分层

第二块地基：编译器不一步到位翻成汇编，而是过一串**中间表示(IR)**，每层 dialect 只负责
下降一个抽象台阶。这就是 MLIR 的核心方法论（multi-level IR），Triton 的 lowering 链是干净样本。

### 实验②：一个 kernel 从算法降到机器码

[labs/lowering_walk.py](labs/lowering_walk.py)：编译一个 GEMM，dump 每层 IR，数特征 token：

| 层 | 行数 | 管什么 |
|---|---|---|
| **ttir** | 100 | 算法层：硬件无关，还是 `tt.dot`/`tt.load`，不知道 warp/shared 长啥样 |
| **ttgir** | 168 | 调度层：绑硬件布局 `#blocked`/`#mma`/`#shared`，决定数据怎么摆进 SM |
| **llir** | 1528 | LLVM IR：通用编译器中间层 |
| **ptx** | 1783 | 机器层：真·GPU 汇编 `mma.sync`/`cp.async` |

特征 token 在各层的出现（看同一件事怎么一层层落到硬件）：

| token | ttir | ttgir | ptx | 含义 |
|---|---|---|---|---|
| `tt.dot`（矩阵乘原语） | 1 | 2 | — | 算法层一个高层原语 |
| `#mma` 布局（绑 tensor core） | 0 | **25** | — | **调度层才出现**：决定用 tensor core |
| `#shared`（shared memory 布局） | 0 | **100** | — | **调度层才出现**：决定数据摆放 |
| `mma.sync`（真·TC 指令） | — | — | **64** | **降到机器层才落地** |
| `cp.async`（异步搬运指令） | — | — | **33** | 同上 |

> **结论**：`tt.dot` 一个高层原语，过 ttgir 绑上 `#mma`/`#shared` 布局，最终在 ptx 摊成
> **64 条 `mma.sync` + 33 条 `cp.async`**。每层 dialect 只下降一个抽象台阶——这就是渐进式 lowering。
> 好处：硬件无关的算法（ttir）能复用；换硬件只改下层 dialect；每层各自做局部优化更可控。

## 串起来：我走过的整条链

```
算法(数学)
  │  ── 算法与调度分离（地基一）：固定算法，搜调度
调度(tile/warp/流水)            ← topic 01 人肉搜 / topic 07 @autotune / topic 08 max-autotune
  │  ── 渐进式 lowering（地基二）：ttir → ttgir → llir → ptx，每层降一个台阶
机器码(mma.sync/cp.async)       ← topic 01 我手写的就是这层
```
topic 01 在最底层人肉干这两件事；topic 07/08 看编译器自动干；topic 09 看清它**凭什么**能自动干。

```bash
cd labs
./run.sh sweep   # 实验①：同一算法换调度，4.7× 差距（Triton，"看"）
./run.sh tvm     # 实验③：亲手写 compute+schedule，113×（TVM，"写"）—— 需 pip install apache-tvm
./run.sh lower   # 实验②：一个 kernel 走 ttir→ttgir→ptx
```

## 结论 / 法则

- **算法与调度分离**：性能几乎全在调度（同算法 4.7×）；编译器的价值 = 自动搜调度空间。
- **渐进式 lowering**：高层原语(`tt.dot`)逐层降——ttgir 绑布局(`#mma`/`#shared`)、ptx 才出真指令(`mma.sync`)。
- 这两条是 Halide→TVM→MLIR→Triton 一脉相承的地基；看懂它们，topic 07/08 的"自动"就不再是黑箱。

## TODO / 下一步

- [x] 实验①：算法与调度分离——同一 matmul 换调度差 4.7×（Triton，"看"）
- [x] 实验②：渐进式 lowering——tt.dot 经 ttgir(#mma/#shared) 降到 ptx(mma.sync/cp.async)
- [x] 实验③：亲手写 TVM `compute`+`schedule`(reorder/split/parallel/vectorize)，同算法 113×（"写"）
- [ ] 读 MLIR dialect 体系（linalg/affine/gpu）怎么对应这套分层；Triton 用的是哪几个
- [ ] TVM 把实验③的调度搬上 GPU target（`cuda`），bind threadIdx/blockIdx，对比 Triton 版
