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
| [01](topics/01-nsys-profiling/) | Nsight Systems / Compute 性能分析 | ✅ 进行中 | nsys 看时间线定位瓶颈段 + ncu 微观挖 kernel；register blocking 把 GEMM 从 6%→36% FP32 |

## 学习日志

- **2026-06-04** 搭建本仓库。完成主题 01：nsys 基础——用自写的 `gemm_lab`（naive vs tiled GEMM + launch-bound 演示）过了一遍 nvtx_sum / cuda_gpu_kern_sum / cuda_api_sum 四张表的读法。
- **2026-06-04** 主题 01 扩展四个实验（见 [topics/01 README](topics/01-nsys-profiling/)）：① pinned 内存让 D2H 带宽 2.9×；② ncu 深挖出 GEMM 只跑 6% FP32 的真因（低算访比 + LG/MIO throttle，**别信 SM Throughput**）；③ GUI 时间线肉眼识别 launch-bound 的"带缝小块"；④ register blocking（每线程 4×4）把 FP32 6%→36%、4.5× 加速，并用 ncu 验证 IPC/occupancy 变化（**拿 occupancy 换 ILP**）。加 cuBLAS 基准做参照：reg 4.22 TFLOPS 已达 cuBLAS(6.95) 的 61%，且 cuBLAS 也只摸到硬件峰值 ~58%——**现实目标是 cuBLAS 不是 roofline**。

## TODO / 下一步

- [ ] 主题 01：实测 CUDA Graph 消除 launch bound（前后 profile 对比）
- [x] 主题 01：学 Nsight Systems GUI 时间线（泳道 / gap / overlap）
- [x] 主题 02：ncu（Nsight Compute）单 kernel 微观分析——occupancy / 访存 / warp stall
- [ ] 主题 01：GEMM 继续优化向 cuBLAS（`float4` 向量化 / shared double buffering / 消 bank conflict / warp tiling）
