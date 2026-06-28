# AI Infra 学习笔记

边学边记 AI 基础设施（CUDA / 算子 / 推理框架 / 分布式 / 性能分析）的笔记与可跑代码。
每个主题 = 一份 Markdown 笔记 + `labs/` 里配套能编译运行的 demo。

## 组织方式

```
ai-infra-notes/
├── topics/                  # 按主题编号，每个目录一个主题
│   └── 01-nsys-profiling/
│       ├── README.md        # 笔记（概念 + 怎么读图）
│       └── labs/            # 可编译运行的 demo + 脚本
└── README.md                # 本文件：索引 + 学习日志
```

约定：
- 每个 lab 自带 `build.sh`，进目录 `./build.sh` 即可编译。
- profiling 报告（`.nsys-rep` / `.ncu-rep`）不入库（见 `.gitignore`），需要时本地重跑生成。
- 笔记里凡是“结论/法则”尽量配一个能复现的命令或数字。

## 主题索引

| 编号 | 主题 | 状态 | 一句话 |
|---|---|---|---|
| [01](topics/01-nsys-profiling/) | Nsight Systems / Compute 性能分析 | ✅ 完成 | nsys 定位 + ncu 五步归因；手写 GEMM 从 0.72→6.82 TFLOPS(98% cuBLAS)，含寄存器悬崖/bank conflict/warp tiling 全套踩坑 |
| [02](topics/02-tensor-core/) | Tensor Core GEMM | 🚧 起步(暂搁) | 换赛道用 Tensor Core(WMMA/FP16)；naive WMMA 因无 shared 复用反而慢——"用了 ≠ 快" |
| [03](topics/03-cuda-graph/) | CUDA Graph | ✅ 进行中 | launch bound 的正解：录一串 kernel 成图、一条 cudaGraphLaunch 重放，CPU launch 调用 ÷20 → 1.78× |
| [04](topics/04-inference/) | 推理系统内部 | 🚧 起步 | decode 是访存 bound → batching 几乎白赚吞吐(12→1150 token/ms ≈93×)；引出 PagedAttention/KV cache |
| [05](topics/05-quantization/) | 量化 | 🚧 起步 | scale+粒度基础；离群值摧毁 per-tensor 低比特(正常通道误差100%)→per-channel/group 救回(GPTQ/AWQ)；W4A16 vs FP8 按瓶颈选型 |
| [06](topics/06-flash-attention/) | FlashAttention | 🚧 起步 | online softmax 永不物化 N×N → 显存 O(N²)→O(N)；速度靠 IO-aware tiling+tensor core |
| [07](topics/07-triton/) | Triton 入门（AI 编译器第一站） | 🚧 进行中 | 手写 GEMM 调度层的自动化：tile 我定，shared/向量化/流水/swizzle/tensor core 编译器包；几十行略胜 cuBLAS(FP32 110%/FP16 105%)，PTX 里自动发出我手抠的 cp.async+mma.sync；融合 matmul+bias+GELU 一个 kernel，越访存 bound 越赚(1.17×→2.28×) |
| [08](topics/08-torch-compile/) | torch.compile / TorchInductor（图层） | 🚧 进行中 | 图层自动化闭环：Dynamo 捕图→Inductor 自动融合→生成 Triton；pointwise 链 10 kernel→1、11.97×，FFN 算力 bound 1.05×；生成的 `triton_poi_fused_addmm_gelu` 正是 topic07 手写的融合 kernel；进阶 max-autotune(小卡退回 cuBLAS) + graph break |
| [09](topics/09-compiler-principles/) | AI 编译器原理（算法/调度分离 + lowering） | 🚧 进行中 | 两块地基：① 同一 matmul 算法只换调度差 4.7×(Halide 算法与调度分离)，TVM 亲手写 compute+schedule 同算法 113×；② 渐进式 lowering——tt.dot 经 ttgir(#mma/#shared) 降到 ptx(64 mma.sync+33 cp.async)，每层 dialect 降一个台阶 |

## 学习日志

- **2026-06-04** 搭建本仓库。完成主题 01：nsys 基础——用自写的 `gemm_lab`（naive vs tiled GEMM + launch-bound 演示）过了一遍 nvtx_sum / cuda_gpu_kern_sum / cuda_api_sum 四张表的读法。
- **2026-06-04** 主题 01 扩展四个实验（见 [topics/01 README](topics/01-nsys-profiling/)）：① pinned 内存让 D2H 带宽 2.9×；② ncu 深挖出 GEMM 只跑 6% FP32 的真因（低算访比 + LG/MIO throttle，**别信 SM Throughput**）；③ GUI 时间线肉眼识别 launch-bound 的"带缝小块"；④ register blocking（每线程 4×4）把 FP32 6%→36%、4.5× 加速，并用 ncu 验证 IPC/occupancy 变化（**拿 occupancy 换 ILP**）。加 cuBLAS 基准做参照：reg 4.22 TFLOPS 已达 cuBLAS(6.95) 的 61%，且 cuBLAS 也只摸到硬件峰值 ~58%——**现实目标是 cuBLAS 不是 roofline**。
- **2026-06-05** 主题 01 实验⑤：`float4` 向量化（三处 LDG/LDS/STG.128 + tile 放大到 128×128/8×8），把访存指令数 ÷4 给 MIO 管线减压，FP32 36%→53%、6.13 TFLOPS = **cuBLAS 的 88%**。occupancy 又砍半到 32%（127 寄存器/线程）却更快——「occupancy 换 ILP」第二次验证。
- **2026-06-05** 主题 01 实验⑥（踩坑实录）：两版 double buffering 都比 float4 慢。① 手动寄存器预取：寄存器 127→129 跨过整数阈值，每 SM block 数 2→1，occupancy 腰斩到 16.7%——**寄存器悬崖**。② `cp.async`：根除了悬崖（寄存器→92、occupancy 回 33%）但要求连续不能转置 → regM 退标量 → 访存瓶颈又顶回来。**转置 vs cp.async 不可兼得**。
- **2026-06-09** 主题 04 推理系统起步：核心是"prefill 算力 bound / decode 访存 bound"。实测一层线性 W=4096²FP16，batch 1→512：访存 bound 区(B≤32)batch 涨 32× 总耗时几乎不动→每 token 暴跌、batching 几乎白赚；算力 bound 区(B≥128)耗时随 batch 线性增长。吞吐 12→1150 token/ms≈93×——这是 continuous batching 的根。下一步 PagedAttention(batch 受 KV cache 显存限制)。
- **2026-06-07** 主题 03 CUDA Graph：承接 topic 01 launch bound。20 kernel/轮 × 2000 轮，baseline 逐个 `cudaLaunchKernel`(40000 次) vs graph 录一轮重放(`cudaGraphLaunch` 2000 次)。nsys 证实 CPU launch 调用 ÷20、1.78× 加速。要点：录制+instantiate 有成本，只对"固定 kernel 链×大量重复"且 launch bound 才划算。（tensor core topic 02 暂搁）
- **2026-06-06** 主题 01 实验⑦（bank conflict 三药方 + 破局）：ncu 查出 float4 主瓶颈是 shared **load 5-way 冲突**。padding/swizzle 只消了 store 冲突(行间型)、load 纹丝不动(+3%)；warp tiling 真消了 load 冲突但 128 累加器→寄存器 232→occupancy 16.7%→更慢。**破局**：wt2 调优版(累加器减半 64、256 线程)同时拿到 无冲突+寄存器 127+occupancy 32% → **6.82 TFLOPS = 98% cuBLAS**。总结：优化是多维度拔河，要把无冲突/寄存器/occupancy 当联立方程一起解——这正是 CUTLASS autotuning 在做的。

- **2026-06-24** 主题 07 Triton 起步（入门 AI 编译器第一站）：拿 Triton GEMM 对线 topic 01 手写版。**心智模型**=编译器三层 lowering(图→Tensor IR→硬件)，我精通的「手调 GEMM」正是中间调度层，Triton 把它自动化。**对照收获**：tile 还我定，但 shared/`__syncthreads`/`float4`向量化/`cp.async`双缓冲/bank-conflict swizzle/tensor core 选择全交编译器；`num_stages`=我手抠的 double buffer，`@autotune`=我手解的联立方程(即 CUTLASS autotuning)。**结果**：4096³ 上 Triton 略胜手写 cuBLAS——FP32(TF32) 12.90 vs 11.74 TFLOPS=110%、FP16 25.44 vs 24.11=105%，代码量 ÷一个数量级。**踩坑**：FP32 初看「202% 假赢」是 `tl.dot` 默认偷走 TF32 而 torch 跑真 FP32，`max_err≈0.30` 是 TF32 指纹——**对线必先对齐精度**。**最大冲击**：`DUMP=1` 落 ttir/ttgir/ptx，生成的 PTX 里直接 grep 到 `cp.async.cg.shared.global`+`mma.sync`(111 处)——我 topic 01 手写还撞寄存器悬崖的 cp.async、topic 02 还没喂饱的 tensor core，编译器一行 `tl.load`+`tl.dot` 全自动发了。下一步：实验③ Triton 融合 `matmul+bias+GELU` 引出图级融合 → topic 08 torch.compile/TorchInductor。

- **2026-06-25** 主题 07 实验③（算子融合，第一次碰 kernel **之间**的优化——手写 CUDA 笔记没有的维度）：Triton 把 `matmul+bias+GELU` 融成一个 kernel(累加器还在寄存器里就地做 epilogue、只写一次 M×N)，对比 torch 三独立 kernel(中间结果 2 读 2 写往返 HBM)。**核心 insight**：融合省的访存固定(M×N=0.13GB)，但 matmul 越不算力 bound 占比越大 → 越赚。实测 FP16 同样 0.13GB，K=4096(算力 bound) 1.17× → K=1024 1.40× → K=256(访存倾斜) **2.28×**。推论：decode/推理那种瘦 matmul+一堆 elementwise，融合是大杠杆——正是 torch.compile 默认猛融的原因。踩坑：`tl.math` 无 `tanh`，改用 `tl.math.erf` 走精确 GELU 对齐 torch 默认 `F.gelu`。下一步 topic 08 torch.compile/TorchInductor 看「图」层怎么自动生成 Triton+融合。

- **2026-06-26** 主题 08 torch.compile/TorchInductor（图层自动化，闭环 topic 07 手写融合）：给 eager 模型，torch.compile 自己 Dynamo 捕图→Inductor 自动决定怎么融→生成 Triton(复用 topic 07 那层 lowering)。**实测**：① pointwise 链(8 个 elementwise) eager 10 kernel/14ms → compiled **1 kernel/1.17ms = 11.97×**(中间结果全留寄存器不往返 HBM)；② FFN(两大 matmul 算力 bound) 5→5 kernel、仅 1.05×——和 topic 07 lab② 同一课:**融合省访存，越访存 bound 越赚**。**闭环冲击**：`./run.sh compile dump` 看生成的 kernel 名字本身就是证据——`triton_poi_fused_add_clamp_exp_mul_relu_sigmoid_sub_tanh_0`(8 个 op 全列名字里塌一个 kernel)、`triton_poi_fused_addmm_gelu_0`(GELU 融进 matmul epilogue = **我 topic07 lab② 手写的 matmul+bias+GELU**)。三层闭环走通:图层(自动融合)→调度层(topic07 Triton)→硬件层(topic01 PTX)。下一步:graph_breaks 看 Dynamo 边界、max-autotune 用 Inductor Triton 模板替 cuBLAS。

- **2026-06-26** 主题 08 进阶(深入 Inductor)：① **max-autotune**=把 topic07 的 `@autotune` 搬到图层，让 Inductor 用自己的 Triton matmul 模板当场搜 config 替 cuBLAS。4096³ 实测 eager/默认 24.5 TFLOPS、max-autotune 反降到 22——日志 `Not enough SMs to use max_autotune_gemm`：**4060 只 24 个 SM 低于门槛，编译器主动拒用 Triton 模板退回 cuBLAS**，那 22 是绕一圈的开销。法则:**模板/调度选择看硬件**，小卡 cuBLAS 已最优、编译器不硬碰(A100/H100 才会真用模板)。② **graph break**:`dynamo.explain` 看 data-dependent 控制流断图——纯张量算子 0 断点/1 段，`if x.sum()>0` 值依赖分支 1 断点/2 段。法则:forward 别用依赖张量值的 python 分支(改 `torch.where`/mask)，断图碎掉融合。

- **2026-06-26** 主题 09 AI 编译器原理(从"会用"到"懂原理")：两块所有 AI 编译器(Halide/TVM/MLIR/Triton/XLA)共享的地基，各用一个 Triton demo 坐实。① **算法与调度分离**:同一个 `matmul_kernel`(算法一字不改)手喂 4 组调度(只改 BLOCK/warps/stages)，4096³ FP16 从 16³tile 的 5.17 TFLOPS 到 128²tile 的 24.43——**最好/最差差 4.7×，性能全在调度里**。这正是编译器的价值:算法谁都会写(C=A@B)，难的是搜调度空间；topic01 我人肉搜、@autotune 自动搜。② **渐进式 lowering**:编译一个 GEMM dump 每层 IR——`tt.dot`(ttir 算法层 1 个)→ ttgir 调度层绑 `#mma`(25)/`#shared`(100) 布局 → ptx 机器层摊成 64 条 `mma.sync`+33 条 `cp.async`。每层 dialect 只降一个抽象台阶(MLIR 多层 IR 方法论)。串起来:算法→(分离/搜调度)→调度→(渐进 lowering)→机器码，topic01 在底层人肉干、07/08 看自动干、09 看清凭什么能自动干。下一步:MLIR dialect 体系、TVM TE 手写 compute+schedule。

- **2026-06-28** 主题 09 实验③(把"算法/调度分离"从看懂变会写)：装 CPU 版 `apache-tvm`，亲手用 TVM 显式调度原语调一个 naive matmul。算法 `te.compute(C=A@B)` 写一次不动，只改调度：默认 i,j,k 2270ms → `reorder(i,k,j)` 连续访存 83ms(27×) → `+split tile+parallel+vectorize` 20ms(**113×**)。比实验①的 4.7× 还猛(CPU naive 太烂衬托)，调度后 IR 肉眼可见 `T.parallel`/`T.vectorized`——**调度重塑循环不改数学**。**踩坑(已记 lab 注释)**:TVM 0.25 的 apache-tvm-ffi 带的 torch C-dlpack 扩展跟 torch 2.6 ABI 不兼容、`import tvm` 即崩 → `sys.modules["torch_c_dlpack_ext"]=None` 让可选扩展干净降级;这版还把子包 `tir`→`s_tir`、方法 `get_block`→`get_sblock`(迁移期过渡名)。下一步:把调度搬 GPU target(bind threadIdx)对比 Triton。

## TODO / 下一步

- [x] 主题 03：CUDA Graph 消除 launch bound（前后 profile 对比，1.78×）
- [x] 主题 01：学 Nsight Systems GUI 时间线（泳道 / gap / overlap）
- [x] 主题 02：ncu（Nsight Compute）单 kernel 微观分析——occupancy / 访存 / warp stall
- [x] 主题 01：GEMM `float4` 向量化（→ 88% cuBLAS）
- [x] 主题 01：double buffering 两版（寄存器悬崖 / cp.async 取舍，均记录踩坑）
- [x] 主题 02：Tensor Core WMMA 入门（naive，发现无 shared 复用→访存 bound）
- [ ] 主题 02：shared 分块的 WMMA 喂饱 Tensor Core + 对标 cuBLAS TF32/FP16 路径
