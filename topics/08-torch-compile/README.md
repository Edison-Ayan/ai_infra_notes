# 08 · torch.compile / TorchInductor（图层自动化，闭环）

承接 topic 07：我在 topic 07 lab② **手写** Triton 把 `matmul+bias+GELU` 融成一个 kernel。
这一篇看 **torch.compile 怎么把这件事全自动化**——给它一个 eager 模型，它自己捕图、自己
决定怎么融、自己生成 Triton kernel。跑完会看到生成的 kernel 跟我手写的几乎一模一样。

环境：RTX 4060 Laptop (sm_89)、torch 2.6 + Triton 3.2。

## 闭环：三层我都亲手走过一遍

```
topic 01  手写 PTX / CUDA GEMM          —— 硬件层，全手动（98% cuBLAS）
topic 07  Triton 手写 GEMM + 手写融合     —— 调度层，我定 tile，编译器发 cp.async/mma
topic 08  torch.compile                 —— 图层，编译器自动捕图 + 自动融合 + 自动生成 Triton
            ↑ Dynamo 捕图 → TorchInductor 调度/融合 → 生成 Triton → 复用 topic 07 那套 lowering
```

## labs

| 文件 | 内容 |
|---|---|
| [labs/compile_ffn.py](labs/compile_ffn.py) | eager vs `torch.compile` 对比：pointwise 链 + FFN 两个案例，计时 + 数 GPU kernel |
| [labs/compile_advanced.py](labs/compile_advanced.py) | 进阶：max-autotune Triton 模板 vs cuBLAS；Dynamo 断图 |
| [labs/run.sh](labs/run.sh) | 跑脚本；`dump` 模式落 Inductor 生成的 Triton 源码并 grep 融合 kernel |

```bash
cd labs
./run.sh compile        # eager vs compiled 计时 + kernel 数
./run.sh compile dump   # 额外 dump Inductor 生成的 Triton 源码（inductor_output.log）
./run.sh adv            # 进阶：max-autotune + graph break
```

## 实验①：torch.compile 自动融合（kernel 数 = 融合的直接证据）

| 案例 | eager | compiled | 加速 | GPU kernel |
|---|---|---|---|---|
| pointwise 链（8 个 elementwise，访存 bound） | 14.05 ms | **1.17 ms** | **11.97×** | **10 → 1** |
| FFN Linear→GELU→Linear+残差（算力 bound） | 47.2 ms | 44.9 ms | 1.05× | 5 → 5 |

**和 topic 07 lab② 完全同一课**：融合省的是访存，**越访存 bound 越赚**。
- pointwise 链：每个算子 eager 下一个 kernel、各自把整块大 tensor 读进读出 HBM（10 kernel）；
  编译器塌成 **1 个 kernel**，中间结果全留寄存器/shared → 11.97×。
- FFN：两个大 matmul 算力 bound，GELU/残差只是零头，融了也只 1.05×、kernel 数没降
  （matmul 仍走 cuBLAS 各自一个 kernel）。

## 实验②：dump 生成的 Triton —— kernel 名字就是融合证据

`./run.sh compile dump` 把 Inductor 生成的 Triton 源码落到 `inductor_output.log`。
生成的 kernel **名字里直接列出融了哪些算子**：

```
triton_poi_fused_add_clamp_exp_mul_relu_sigmoid_sub_tanh_0   # pointwise 链 8 个 op → 一个 kernel
triton_poi_fused_addmm_gelu_0                                # ★ GELU 融进 matmul epilogue
triton_poi_fused_add_addmm_1                                 # 残差 add 融进第二个 matmul epilogue
```

**最大冲击（闭环）**：`triton_poi_fused_addmm_gelu_0` —— TorchInductor **自动**把 GELU 融进了
matmul 的 epilogue，这**正是我 topic 07 lab② 手写的 `matmul+bias+GELU`**。我手抠一晚的融合
kernel，编译器从 eager 代码自动生成出来，长得几乎一样。三层闭环到此走通：
**图层(自动融合) → 调度层(topic 07 的 Triton) → 硬件层(topic 01 的 PTX)。**

## 实验③：max-autotune —— Inductor 的 Triton matmul 模板 vs cuBLAS

默认 `torch.compile` 的 matmul 还是调 cuBLAS；`mode="max-autotune"` 让 Inductor 改用**自己的
Triton matmul 模板**，编译时当场 benchmark 一批 config 选最快的——**就是我 topic 07 `@autotune`
那套搜索，搬到图层自动做**。

4096³ FP16/TF32 实测：

| | ms | TFLOPS |
|---|---|---|
| eager(cuBLAS) | 5.60 | 24.54 |
| compile 默认 | 5.60 | 24.53 |
| max-autotune | 6.24 | **22.02（不升反降）** |

**⚠️ 踩坑 / 一课**：max-autotune 这里**没赢、反而略慢**，且日志报
`Not enough SMs to use max_autotune_gemm mode`。原因：4060 Laptop 只有 **24 个 SM**，
低于 Inductor 用 Triton GEMM 模板的门槛 → 它**主动拒绝**用模板、退回 cuBLAS，那 22 TFLOPS
只是多绕一圈的编译开销。
> **法则：模板/调度的选择是看硬件的。** 同一个 max-autotune 在 A100/H100（108/132 SM）上才会
> 真的用 Triton 模板去拼 cuBLAS；小卡上 cuBLAS 已是最优解，编译器懂得不去硬碰。
> （对照 topic 07 我手写 Triton FP16 = 25.44，已经摸到这张卡的天花板。）

## 实验④：graph break —— Dynamo 在哪断图

Dynamo 捕图遇到 **data-dependent 控制流**（依赖张量值的 `if` / `.item()` / `.tolist()`）
没法静态展开，会"断图"：把一张图劈成几段，断点处退回 eager 求值再续。跨段不能融合。

`torch._dynamo.explain(fn)(x)` 实测：

| 写法 | 断点 | 图段数 |
|---|---|---|
| `(x*2+1).relu().sin()`（纯张量算子） | **0** | 1 |
| `if x.sum() > 0: ...`（值依赖分支） | **1** | 2 |

> **法则：forward 里别出现依赖张量值的 python 分支**（用 `torch.where`/mask 代替 `if tensor`）。
> 断图越多 → 完整图越碎 → 融合机会越少，compile 收益打折。

## 结论 / 法则

- **torch.compile = 图层自动化**：Dynamo 捕图 → Inductor 自动融合 → 生成 Triton（复用 topic 07 那层）。
- **max-autotune** = 把 topic 07 的 `@autotune` 搬到图层；但**用不用 Triton 模板看硬件**，小卡(4060 24SM)直接退回 cuBLAS。
- **graph break**：值依赖的 python 分支会断图、碎掉融合；保持 forward 纯张量算子。
- 融合本质还是 topic 07 那条法则：**省访存，越访存 bound 越赚**（pointwise 11.97× / FFN 1.05×）。
- **kernel 数**是看融合有没有发生的最快指标（10→1 一眼可见）；compiled kernel 名 `*_fused_*` 列出融了哪些 op。
- 大 matmul 算力 bound 时 torch.compile 收益有限——真正的杠杆在「瘦 matmul + 一堆 elementwise」(decode/推理)。

## TODO / 下一步

- [x] 实验①：eager vs compile，pointwise 链 10→1 kernel/11.97×，FFN 算力 bound 1.05×
- [x] 实验②：dump 生成 Triton，kernel 名验证 GELU 融进 addmm epilogue = topic 07 手写版
- [x] 实验③：max-autotune Triton 模板——4060 SM 不够，编译器主动退回 cuBLAS（模板选择看硬件）
- [x] 实验④：graph break——值依赖 python 分支断图（0→1 段→2 段），保持 forward 纯张量算子
