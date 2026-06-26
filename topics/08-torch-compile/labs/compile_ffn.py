"""
topic 08 lab① · torch.compile / TorchInductor —— 自动做我 topic07 手写的融合

闭环：
  topic 01  手写 PTX/CUDA GEMM        —— 硬件层，全手动
  topic 07  Triton 手写 GEMM + 手写融合 —— 调度层，我定 tile，编译器发指令
  topic 08  torch.compile             —— 图层，编译器自动捕图 + 自动决定怎么融 + 自动生成 Triton

这一篇验证：给 eager 的 FFN(Linear→GELU→Linear+残差)，torch.compile 会
  ① 用 Dynamo 把 forward 捕成图，
  ② TorchInductor 自动决定哪些算子可融，
  ③ 自动生成 Triton kernel——名字直接叫 triton_poi_fused_xxx，把融了哪些 op 写在名字里。

用法：  ./run.sh compile        # eager vs compiled 计时 + kernel 数
        ./run.sh compile dump   # 额外 dump Inductor 生成的 Triton 源码并 grep 融合 kernel
"""
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, DeviceType


def pointwise_chain(x):
    # 一串纯 elementwise：eager 下每个算子 = 一个 kernel，各自把整块大 tensor
    # 读进来、算、写回 HBM。访存 bound，全是浪费。compiled 应塌成一个 kernel。
    return torch.sigmoid(x) * torch.tanh(x) + x * x - torch.relu(x) + x.exp().clamp(max=10)


class FFN(nn.Module):
    # 经典 transformer FFN：一堆算子，正好给编译器融
    def __init__(self, d, hidden):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, d)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(x)))   # 残差 + 两个 matmul + GELU


def bench(fn, x, warmup=20, iters=100):
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(x)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters   # ms/次


def count_cuda_kernels(fn, x):
    # 数一次 forward 发了多少个 GPU kernel —— 融合的直接证据：compiled 发得少
    fn(x); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(x)
        torch.cuda.synchronize()
    return sum(1 for e in prof.events()
               if e.device_type == DeviceType.CUDA and e.device_time_total > 0)


def compare(eager_fn, x, tag):
    compiled = torch.compile(eager_fn)
    with torch.no_grad():
        y_e, y_c = eager_fn(x), compiled(x)   # 第二次触发编译
        ok = torch.allclose(y_e, y_c, atol=2e-2, rtol=2e-2)
        n_e, n_c = count_cuda_kernels(eager_fn, x), count_cuda_kernels(compiled, x)
        t_e, t_c = bench(eager_fn, x), bench(compiled, x)
    print(f"\n## {tag}  {'✅' if ok else '❌FAIL'}")
    print(f"[eager   ] {t_e:8.3f} ms/次 | GPU kernel {n_e}")
    print(f"[compiled] {t_c:8.3f} ms/次 | GPU kernel {n_c}")
    print(f"[加速    ] {t_e / t_c:5.2f}× | kernel {n_e} → {n_c}（融合塌成更少 kernel）")


def main():
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True

    # ① 纯 pointwise 链：访存 bound，融合 N→1 最戏剧化
    x = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
    compare(pointwise_chain, x, "pointwise 链（8 个 elementwise，访存 bound）")

    # ② 真实模块 FFN：两个大 matmul 算力 bound，融合只能省掉 GELU/残差的零头
    model = FFN(4096, 16384).cuda().half().eval()
    xf = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    compare(model, xf, "FFN Linear→GELU→Linear+残差（算力 bound）")


if __name__ == "__main__":
    main()
