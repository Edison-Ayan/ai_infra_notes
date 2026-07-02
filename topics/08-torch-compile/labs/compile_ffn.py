"""
topic 08 lab① · torch.compile / TorchInductor —— 自动做我 topic07 手写的融合

这一篇的核心目标：
  用一个普通 PyTorch eager 写法的函数/模块，验证 torch.compile 能不能自动做图层优化。

闭环：
  topic 01  手写 PTX/CUDA GEMM
      —— 硬件层：线程、warp、shared memory、mma、cp.async 全手动控制。

  topic 07  Triton 手写 GEMM + 手写融合
      —— 调度层：我手写 Triton kernel，自己决定 BLOCK_M/BLOCK_N/BLOCK_K，
         自己决定把 matmul + bias + GELU 融成一个 kernel；
         但 shared memory、mma layout、PTX 指令由 Triton 编译器生成。

  topic 08  torch.compile / TorchInductor
      —— 图层：我不再手写 Triton kernel，而是写普通 PyTorch forward；
         编译器自动捕获计算图，自动判断哪些算子可以融合，
         自动生成 Triton kernel 或调用外部高性能库。

这一篇验证：
  给 eager 版本的函数/模块，比如：
      1. 纯 pointwise 链
      2. FFN: Linear → GELU → Linear + residual

  torch.compile 会尝试：
      ① 用 TorchDynamo 把 Python forward 捕获成 FX Graph；
      ② 用 TorchInductor 分析图，判断哪些算子可以融合；
      ③ 对 pointwise / reduction 等部分生成 Triton kernel；
      ④ 对大型 matmul / linear，通常可能调用 cuBLAS / extern_kernels；
      ⑤ 最后通过 profiler 观察 compiled 版本的 kernel 数是否减少。

用法：
  ./run.sh compile
      # 跑 eager vs compiled 计时，并统计 CUDA kernel 数

  ./run.sh compile dump
      # 额外 dump TorchInductor 生成的代码，
      # grep triton_poi_fused_xxx 之类的融合 kernel
"""


# PyTorch 主库
import torch

# torch.nn 里有 Module、Linear、GELU 等神经网络模块
import torch.nn as nn

# PyTorch profiler，用来统计一次 forward 到底发了多少 CUDA kernel
from torch.profiler import profile, ProfilerActivity, DeviceType


def pointwise_chain(x):
    """
    一个纯 pointwise 算子链。

    pointwise 的意思是：
        输出的每个元素，只依赖输入中对应位置的元素。

    比如：
        sigmoid(x)
        tanh(x)
        x * x
        relu(x)
        exp(x)
        clamp(x)

    都是逐元素计算。

    为什么这个例子适合看 torch.compile 的融合？

    eager 模式下，PyTorch 通常会把这些算子拆成多个 CUDA kernel：

        sigmoid kernel:
            读 x，算 sigmoid，写中间结果 tmp1

        tanh kernel:
            读 x，算 tanh，写中间结果 tmp2

        mul kernel:
            读 tmp1/tmp2，写 tmp3

        exp kernel:
            读 x，写 tmp4

        clamp kernel:
            读 tmp4，写 tmp5

        ...

    每个 kernel 都会读写大 tensor。
    对 8192×8192 的 fp16 tensor 来说，数据量非常大。

    这类算子通常不是算力瓶颈，而是 HBM 显存带宽瓶颈：
        计算很少，但反复读写显存。

    torch.compile / TorchInductor 的优化目标：
        把这些 pointwise 算子融合成一个 Triton kernel。

    融合后大概变成：
        读 x 一次；
        在寄存器里连续算 sigmoid/tanh/mul/add/relu/exp/clamp；
        最后写 output 一次。

    这样可以减少中间 tensor 的 HBM 读写。
    """

    return (
        torch.sigmoid(x) * torch.tanh(x)
        + x * x
        - torch.relu(x)
        + x.exp().clamp(max=10)
    )


class FFN(nn.Module):
    """
    一个简化版 Transformer FFN。

    标准 Transformer block 里通常有一段 FFN：

        x -> Linear1 -> Activation -> Linear2 -> residual add

    这里写成：

        output = x + fc2(GELU(fc1(x)))

    其中：
        fc1: d -> hidden
        GELU: 非线性激活
        fc2: hidden -> d
        x + ...: 残差连接

    这个例子和 pointwise_chain 不一样。

    pointwise_chain 几乎全是访存 bound 的逐元素算子，
    所以融合收益非常明显。

    FFN 里主要耗时来自两个大矩阵乘法：
        fc1(x)
        fc2(hidden)

    大矩阵乘法通常是 compute-bound，主要吃 Tensor Core 算力。
    这些大 GEMM 可能由 cuBLAS / extern_kernels 执行，
    不一定被 Inductor 直接生成成 Triton matmul。

    但是 GELU、residual add 这些 pointwise 部分仍然可能被融合。
    所以 FFN 的 compiled 版本一般 kernel 数会减少，
    但加速幅度通常没有纯 pointwise_chain 那么夸张。
    """

    def __init__(self, d, hidden):
        """
        初始化 FFN。

        参数：
            d:
                输入和输出维度。
                对 Transformer 来说，通常就是 hidden size，比如 4096。

            hidden:
                FFN 中间层维度。
                Transformer 里经常是 4 倍 hidden size，比如 16384。

        这里：
            fc1: 4096 -> 16384
            fc2: 16384 -> 4096
        """

        # 必须调用父类 nn.Module 的初始化
        super().__init__()

        # 第一层 Linear，相当于矩阵乘：
        #   x @ W1^T + b1
        #
        # 如果 x shape 是 [batch, d]
        # 那么输出 shape 是 [batch, hidden]
        self.fc1 = nn.Linear(d, hidden)

        # 第二层 Linear，相当于：
        #   hidden @ W2^T + b2
        #
        # 输出 shape 回到 [batch, d]
        self.fc2 = nn.Linear(hidden, d)

        # GELU 激活函数
        # Transformer FFN 里常见的非线性
        self.act = nn.GELU()

    def forward(self, x):
        """
        前向传播。

        计算过程：
            1. self.fc1(x)
                第一个大 GEMM

            2. self.act(...)
                GELU pointwise 激活

            3. self.fc2(...)
                第二个大 GEMM

            4. x + ...
                residual add，逐元素加法

        对编译器来说，这个图大概是：

            x
            |
            Linear1
            |
            GELU
            |
            Linear2
            |
            Add  <--- x
            |
          output

        TorchInductor 可能做的事：
            - 对 GELU / add 等 pointwise 做融合；
            - 对 Linear/GEMM 可能调用 cuBLAS；
            - 生成一些 triton_poi_fused_xxx kernel。
        """

        return x + self.fc2(self.act(self.fc1(x)))


def bench(fn, x, warmup=20, iters=100):
    """
    benchmark 函数。

    作用：
        测量 fn(x) 的平均执行时间。

    参数：
        fn:
            要测试的函数，比如 eager_fn 或 compiled_fn。

        x:
            输入 tensor。

        warmup:
            预热次数。

        iters:
            正式计时迭代次数。

    为什么需要 warmup？

        GPU 第一次运行时可能会有：
            - CUDA context 初始化
            - kernel lazy loading
            - torch.compile 编译
            - cache miss
            - cuBLAS autotune

        这些都会污染计时。

    所以先跑 warmup 次，不计入最终时间。

    为什么用 torch.cuda.Event？

        CUDA kernel 是异步发射的。
        如果用 Python time.time()，很容易只测到 kernel launch 时间，
        而不是 GPU 真正执行时间。

        CUDA Event 是 GPU 时间戳，更适合测 kernel 执行时间。
    """

    # 预热，不计时
    for _ in range(warmup):
        fn(x)

    # 等待前面的 warmup kernel 全部跑完
    torch.cuda.synchronize()

    # 创建两个 CUDA Event，作为开始和结束时间戳
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # 记录开始时间
    start.record()

    # 正式计时
    for _ in range(iters):
        fn(x)

    # 记录结束时间
    end.record()

    # 等待所有 kernel 执行完
    # 不 synchronize 的话，CPU 会继续往下走，计时不准
    torch.cuda.synchronize()

    # elapsed_time 返回 start 到 end 的毫秒数
    # 除以 iters 得到平均每次 forward 的耗时
    return start.elapsed_time(end) / iters


def count_cuda_kernels(fn, x):
    """
    统计一次 fn(x) 发了多少个 CUDA kernel。

    这个函数是为了验证 fusion。

    融合前：
        eager 模式下，一个 pointwise 链可能发很多 kernel。

    融合后：
        compiled 模式下，多个 pointwise 算子可能塌成一个 Triton kernel。

    所以 kernel 数减少，是 fusion 的一个直接证据。

    注意：
        profiler 统计出来的事件有时会包含一些额外 CUDA event。
        这个函数适合做教学实验，不一定是工业级严格 kernel counter。
    """

    # 先跑一次，避免把首次 lazy 初始化、首次编译等算进去
    fn(x)
    torch.cuda.synchronize()

    # 开启 profiler，只记录 CUDA 活动
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(x)
        torch.cuda.synchronize()

    # 遍历 profiler 里的事件
    # e.device_type == DeviceType.CUDA 表示这是 CUDA 设备上的事件
    # e.device_time_total > 0 表示它确实在 GPU 上花了时间
    return sum(
        1
        for e in prof.events()
        if e.device_type == DeviceType.CUDA and e.device_time_total > 0
    )


def compare(eager_fn, x, tag):
    """
    对比 eager 版本和 compiled 版本。

    参数：
        eager_fn:
            原始 PyTorch 函数或 nn.Module。

        x:
            输入 tensor。

        tag:
            打印时显示的实验名称。

    流程：
        1. torch.compile 得到 compiled 函数；
        2. 跑 eager 和 compiled，检查数值是否接近；
        3. 统计 eager 和 compiled 的 CUDA kernel 数；
        4. benchmark 两者耗时；
        5. 打印速度提升和 kernel 数变化。
    """

    # 用 torch.compile 编译 eager_fn
    #
    # 默认 backend 通常是 inductor。
    # 更教学化的写法可以是：
    #   compiled = torch.compile(eager_fn, backend="inductor", fullgraph=True)
    #
    # fullgraph=True 的好处：
    #   如果中间出现 graph break，会直接报错，方便定位。
    #
    # 这里保持你的原始写法。
    compiled = torch.compile(eager_fn)

    # 这个实验只做推理，不需要梯度
    # no_grad 可以减少显存开销，也避免 autograd 相关图干扰 profiler
    with torch.no_grad():

        # eager 输出
        y_e = eager_fn(x)

        # 第一次调用 compiled(x) 会触发：
        #   1. TorchDynamo 捕图
        #   2. TorchInductor 编译
        #   3. 执行编译后的代码
        #
        # 注意：
        #   不是“第二次触发编译”，而是第一次调用 compiled 触发编译。
        y_c = compiled(x)

        # 检查数值是否接近
        #
        # 使用 fp16 + TF32 时，不要用过严的误差。
        # 这里 atol=2e-2, rtol=2e-2 是比较宽松的教学容差。
        ok = torch.allclose(y_e, y_c, atol=2e-2, rtol=2e-2)

        # 统计 eager 模式下一次 forward 的 CUDA kernel 数
        n_e = count_cuda_kernels(eager_fn, x)

        # 统计 compiled 模式下一次 forward 的 CUDA kernel 数
        #
        # 这里 count_cuda_kernels 内部会先 fn(x) 一次，
        # 所以一般不会把编译时间算进 profiler。
        n_c = count_cuda_kernels(compiled, x)

        # benchmark eager 版本
        t_e = bench(eager_fn, x)

        # benchmark compiled 版本
        #
        # 注意：
        #   compiled 第一次调用时已经编译过了，
        #   这里测的主要是编译后执行时间。
        t_c = bench(compiled, x)

    # 打印实验标题和数值正确性
    print(f"\n## {tag}  {'✅' if ok else '❌FAIL'}")

    # 打印 eager 耗时和 kernel 数
    print(f"[eager   ] {t_e:8.3f} ms/次 | GPU kernel {n_e}")

    # 打印 compiled 耗时和 kernel 数
    print(f"[compiled] {t_c:8.3f} ms/次 | GPU kernel {n_c}")

    # 打印加速比和 kernel 数变化
    #
    # 如果 compiled 融合成功，通常会看到：
    #   kernel 数减少；
    #   pointwise_chain 加速明显；
    #   FFN 加速较小。
    print(
        f"[加速    ] {t_e / t_c:5.2f}× | "
        f"kernel {n_e} → {n_c}（融合塌成更少 kernel）"
    )


def main():
    """
    主函数。

    这里跑两个实验：

        ① pointwise_chain
            纯逐元素算子链。
            预期 compiled 融合收益明显。

        ② FFN
            两个大 Linear + GELU + residual。
            预期 compiled kernel 数减少，
            但加速幅度受大 GEMM 限制。
    """

    # 固定随机种子，保证每次生成的输入一致
    torch.manual_seed(0)

    # 允许 matmul 使用 TF32
    #
    # 对 NVIDIA Ampere/Ada GPU 来说，
    # TF32 可以显著加速 float32 matmul。
    #
    # 但你这里主要 tensor 是 fp16，
    # 所以 FFN 的大 GEMM 主要还是走 fp16 Tensor Core。
    #
    # 保留这个选项可以让 float32 matmul 行为和之前 topic07 更一致。
    torch.backends.cuda.matmul.allow_tf32 = True

    # ------------------------------------------------------------
    # 实验 ①：纯 pointwise 链
    # ------------------------------------------------------------

    # 创建一个很大的 fp16 tensor。
    #
    # shape:
    #   8192 × 8192
    #
    # 元素数量：
    #   8192 * 8192 = 67,108,864
    #
    # fp16 每个元素 2 字节，
    # 单个 tensor 大约：
    #   67,108,864 * 2 bytes ≈ 128 MB
    #
    # eager 模式下 pointwise_chain 会产生多个中间 tensor，
    # 所以实际显存占用会更高。
    x = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)

    # 对比 eager vs compiled
    #
    # 预期：
    #   eager:
    #       多个 pointwise kernel；
    #       大量 HBM 中间读写。
    #
    #   compiled:
    #       可能塌成一个 triton_poi_fused_xxx kernel；
    #       kernel 数明显减少；
    #       加速明显。
    compare(pointwise_chain, x, "pointwise 链（8 个 elementwise，访存 bound）")

    # ------------------------------------------------------------
    # 实验 ②：真实 FFN
    # ------------------------------------------------------------

    # 创建一个 FFN 模块：
    #
    #   d = 4096
    #   hidden = 16384
    #
    # 也就是：
    #   fc1: 4096 -> 16384
    #   fc2: 16384 -> 4096
    #
    # .cuda():
    #   放到 GPU 上。
    #
    # .half():
    #   参数转成 fp16。
    #
    # .eval():
    #   进入推理模式。
    #   虽然 Linear/GELU 没有 dropout/bn 这类训练/推理差异，
    #   但 benchmark 时通常都设 eval。
    model = FFN(4096, 16384).cuda().half().eval()

    # 输入 shape:
    #   4096 × 4096
    #
    # 可以理解成：
    #   batch/token 数 = 4096
    #   hidden size = 4096
    #
    # 计算量很大：
    #   fc1: [4096,4096] × [4096,16384]
    #   fc2: [4096,16384] × [16384,4096]
    #
    # 所以这个实验主要耗时在两个大 GEMM。
    xf = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)

    # 对比 eager vs compiled
    #
    # 预期：
    #   eager:
    #       Linear、GELU、Linear、add 分开执行。
    #
    #   compiled:
    #       大 GEMM 可能仍然走 cuBLAS / extern_kernels；
    #       GELU / residual add 等 pointwise 可能被融合；
    #       kernel 数减少；
    #       加速幅度通常小于 pointwise_chain。
    compare(model, xf, "FFN Linear→GELU→Linear+残差（算力 bound）")


# Python 脚本入口
#
# 当你直接运行：
#   python topic08_compile.py
#
# __name__ == "__main__" 成立，执行 main()。
#
# 如果这个文件被别的 Python 文件 import，
# main() 不会自动执行。
if __name__ == "__main__":
    main()
