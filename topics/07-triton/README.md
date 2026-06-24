# 07 · Triton 入门（AI 编译器第一站）

承接 topic 01：我手写 CUDA GEMM 从 0.72 → **6.82 TFLOPS(98% cuBLAS)**，一路踩了
register blocking / float4 向量化 / bank conflict / double buffering / cp.async / 寄存器悬崖。
那是**手动**解「tile × occupancy × ILP」的联立方程。

Triton 是入门 AI 编译器最舒服的第一站：**我还写 tile/block，但 shared 分配、向量化、
软件流水(double buffer)、swizzle、tensor core 选择全交给编译器**。这一篇就是拿 Triton GEMM
对线我自己的手写版，亲眼验证「我抠半个月的东西，编译器用哪几个旋钮自动发掉」。

环境：RTX 4060 Laptop (sm_89)、torch 2.6 + Triton 3.2、CUDA 12.4。

## 心智模型：AI 编译器三层 lowering

```
计算图 (FX/ONNX)
   ↓  图级：算子融合、layout、常量折叠         ← topic 08 再碰
Tensor IR（循环+访存，未绑硬件）
   ↓  调度：tiling/vectorize/绑线程/double buffer ← 我手写 GEMM 干的就是这层
硬件代码 (PTX/LLVM)
```
我已精通最底层的**手动**版。Triton = 把这层**自动化**：我给 tile 和 autotune 搜索空间，
编译器把 ttir → ttgir(绑硬件布局) → llir → ptx → cubin 一路降下去。

## labs

| 文件 | 内容 |
|---|---|
| [labs/triton_gemm.py](labs/triton_gemm.py) | Triton 分块 GEMM + autotune，对线 cuBLAS(torch.matmul)，FP32/FP16 两路 |
| [labs/run.sh](labs/run.sh) | 跑脚本（python 无需编译，故用 run.sh 不是 build.sh） |

```bash
cd labs
./run.sh            # FP32（两边对齐 TF32）
./run.sh fp16       # FP16 走 tensor core
DUMP=1 ./run.sh fp16  # 额外 dump ttir/ttgir/ptx，看 lowering
```

## 手写 CUDA → Triton 旋钮对照（核心收获）

| 我手写 CUDA 抠的东西（topic 01） | Triton 里对应 | 谁来做 |
|---|---|---|
| tile 划分 BM×BN×BK | `BLOCK_M/N/K` | 我定（编译器不猜算法 tile） |
| shared memory 分配/搬运/`__syncthreads` | `tl.load` 进寄存器块 | **编译器** |
| double buffering / `cp.async` | `num_stages`（>1=软件流水） | **编译器** |
| bank conflict 三药方 / swizzle | `GROUP_M` + 自动 swizzle | **编译器** |
| `float4` LDG.128 向量化 | 按 BLOCK_K 自动向量化 | **编译器** |
| occupancy 换 ILP 的拉锯 | `num_warps` + autotune 搜索 | **编译器搜** |

> 一句话：我 topic 01 把「无冲突 / 寄存器 / occupancy」当联立方程**手解**到 98% cuBLAS；
> Triton 的 `@autotune` 就是把这套**自动搜**——正是 CUTLASS autotuning 在做的事。

## 实验①：Triton GEMM 对线 cuBLAS（4096³）

| dtype | Triton | cuBLAS(torch.matmul) | Triton/cuBLAS | autotune 选中 |
|---|---|---|---|---|
| FP32(TF32) | **12.90 TFLOPS** | 11.74 | **109.8%** | BM128·BN64·BK32·stages3·warps4 |
| FP16 | **25.44 TFLOPS** | 24.11 | **105.5%** | BM128·BN128·BK32·stages2·warps4 |

**几十行 Python 就略胜手写 cuBLAS**——我 topic 01 抠了一周才到 98%。这就是编译器的杠杆。

### ⚠️ 踩坑：FP32「假赢 2×」的精度陷阱
第一次跑 Triton 显示 **202% cuBLAS**，是假象：`tl.dot` 对 FP32 输入**默认偷偷走 TF32**
(tensor core，~10 位尾数)，而 `torch.matmul` 默认 `allow_tf32=False` 在跑**真 FP32**。
`max_abs_err≈0.30` 就是 TF32 的指纹。对齐 `torch.backends.cuda.matmul.allow_tf32=True`
让两边同精度后，才是真实的 **109.8%**。
**法则：对线必须先对齐数值精度，否则编译器会拿低精度白嫖一倍。**

## 实验②：dump lowering —— 编译器替我发了 cp.async + tensor core

`DUMP=1 ./run.sh fp16` 把中间表示落盘看 lowering 链：

| stage | 行数 | 是什么 |
|---|---|---|
| dump.ttir | 162 | 算法层 IR（还没绑硬件） |
| dump.ttgir | 228 | **绑硬件布局**（`#blocked`/`#mma`/`#shared`） |
| dump.ptx | 1767 | PTX 汇编 |

在生成的 **PTX 里直接 grep 到**（111 处 tensor core / 异步拷贝指令）：
```
cp.async.cg.shared.global [...]      # ← topic 01 实验⑥ 我手写、还撞寄存器悬崖的那条
mma.sync / ldmatrix                  # ← tensor core，我 topic 02 还没喂饱它
```
**最大冲击**：我在 topic 01 为 `cp.async` 纠结「转置 vs cp.async 不可兼得」、在 topic 02 为
喂饱 tensor core 卡壳——Triton 一行 `tl.load` + `tl.dot`，编译器**自动**把这两样都发了出来。
这就是「写编译器友好的算法，把硬件细节交出去」的真正含义。

## 结论 / 法则

- **Triton = 手写 CUDA 的调度层自动化**：算法 tile 我定，shared/向量化/流水/swizzle/tensor core 编译器包。
- 在 4060 上 Triton GEMM **略胜手写 cuBLAS**（FP32 110% / FP16 105%），代码量 ÷一个数量级。
- `num_stages` = 我手抠的 double buffer；`@autotune` = 我手解的联立方程。
- **对线先对齐精度**，否则 TF32 假赢 2×。

## TODO / 下一步

- [x] 实验①：Triton GEMM 对线 cuBLAS（FP32 110% / FP16 105%）
- [x] 实验②：dump ttgir/ptx，验证编译器自动发 cp.async + mma.sync
- [ ] 实验③：融合 kernel —— Triton 写 `matmul + bias + GELU` 一个 kernel，对比三个独立 kernel 的访存节省（引出 topic 08 的「图级融合」）
- [ ] topic 08：torch.compile / TorchInductor，看「图」这层怎么自动生成 Triton kernel + 算子融合
