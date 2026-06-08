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

## 学习日志

- **2026-06-04** 搭建本仓库。完成主题 01：nsys 基础——用自写的 `gemm_lab`（naive vs tiled GEMM + launch-bound 演示）过了一遍 nvtx_sum / cuda_gpu_kern_sum / cuda_api_sum 四张表的读法。
- **2026-06-04** 主题 01 扩展四个实验（见 [topics/01 README](topics/01-nsys-profiling/)）：① pinned 内存让 D2H 带宽 2.9×；② ncu 深挖出 GEMM 只跑 6% FP32 的真因（低算访比 + LG/MIO throttle，**别信 SM Throughput**）；③ GUI 时间线肉眼识别 launch-bound 的"带缝小块"；④ register blocking（每线程 4×4）把 FP32 6%→36%、4.5× 加速，并用 ncu 验证 IPC/occupancy 变化（**拿 occupancy 换 ILP**）。加 cuBLAS 基准做参照：reg 4.22 TFLOPS 已达 cuBLAS(6.95) 的 61%，且 cuBLAS 也只摸到硬件峰值 ~58%——**现实目标是 cuBLAS 不是 roofline**。
- **2026-06-05** 主题 01 实验⑤：`float4` 向量化（三处 LDG/LDS/STG.128 + tile 放大到 128×128/8×8），把访存指令数 ÷4 给 MIO 管线减压，FP32 36%→53%、6.13 TFLOPS = **cuBLAS 的 88%**。occupancy 又砍半到 32%（127 寄存器/线程）却更快——「occupancy 换 ILP」第二次验证。
- **2026-06-05** 主题 01 实验⑥（踩坑实录）：两版 double buffering 都比 float4 慢。① 手动寄存器预取：寄存器 127→129 跨过整数阈值，每 SM block 数 2→1，occupancy 腰斩到 16.7%——**寄存器悬崖**。② `cp.async`：根除了悬崖（寄存器→92、occupancy 回 33%）但要求连续不能转置 → regM 退标量 → 访存瓶颈又顶回来。**转置 vs cp.async 不可兼得**。
- **2026-06-09** 主题 04 推理系统起步：核心是"prefill 算力 bound / decode 访存 bound"。实测一层线性 W=4096²FP16，batch 1→512：访存 bound 区(B≤32)batch 涨 32× 总耗时几乎不动→每 token 暴跌、batching 几乎白赚；算力 bound 区(B≥128)耗时随 batch 线性增长。吞吐 12→1150 token/ms≈93×——这是 continuous batching 的根。下一步 PagedAttention(batch 受 KV cache 显存限制)。
- **2026-06-07** 主题 03 CUDA Graph：承接 topic 01 launch bound。20 kernel/轮 × 2000 轮，baseline 逐个 `cudaLaunchKernel`(40000 次) vs graph 录一轮重放(`cudaGraphLaunch` 2000 次)。nsys 证实 CPU launch 调用 ÷20、1.78× 加速。要点：录制+instantiate 有成本，只对"固定 kernel 链×大量重复"且 launch bound 才划算。（tensor core topic 02 暂搁）
- **2026-06-06** 主题 01 实验⑦（bank conflict 三药方 + 破局）：ncu 查出 float4 主瓶颈是 shared **load 5-way 冲突**。padding/swizzle 只消了 store 冲突(行间型)、load 纹丝不动(+3%)；warp tiling 真消了 load 冲突但 128 累加器→寄存器 232→occupancy 16.7%→更慢。**破局**：wt2 调优版(累加器减半 64、256 线程)同时拿到 无冲突+寄存器 127+occupancy 32% → **6.82 TFLOPS = 98% cuBLAS**。总结：优化是多维度拔河，要把无冲突/寄存器/occupancy 当联立方程一起解——这正是 CUTLASS autotuning 在做的。

## TODO / 下一步

- [x] 主题 03：CUDA Graph 消除 launch bound（前后 profile 对比，1.78×）
- [x] 主题 01：学 Nsight Systems GUI 时间线（泳道 / gap / overlap）
- [x] 主题 02：ncu（Nsight Compute）单 kernel 微观分析——occupancy / 访存 / warp stall
- [x] 主题 01：GEMM `float4` 向量化（→ 88% cuBLAS）
- [x] 主题 01：double buffering 两版（寄存器悬崖 / cp.async 取舍，均记录踩坑）
- [x] 主题 02：Tensor Core WMMA 入门（naive，发现无 shared 复用→访存 bound）
- [ ] 主题 02：shared 分块的 WMMA 喂饱 Tensor Core + 对标 cuBLAS TF32/FP16 路径
