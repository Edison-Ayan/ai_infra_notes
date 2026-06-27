"""
topic 08 lab② · Inductor 进阶：max-autotune 模板 + Dynamo 断图

承接 lab①（默认 compile 自动融合）。这一篇看 Inductor 两个更深的机制：
  A. max-autotune —— Inductor 不再无脑调 cuBLAS，而是用自己的 Triton matmul 模板，
     当场 benchmark 一批 config 选最快的（= 我 topic07 @autotune 干的事，只是它在图层自动做）。
  B. graph break  —— Dynamo 捕图遇到 data-dependent 控制流（依赖张量值的 if/.item()）
     会"断图"：把一张图劈成几段、中间退回 eager。断点越多，融合机会越少。

用法：  ./run.sh adv            # 跑 A + B
        ./run.sh adv autotune  # 只跑 A（max-autotune 编译较慢）
"""
import os
import torch


# ---------- A. max-autotune：Inductor Triton matmul 模板 vs cuBLAS ----------
def bench(fn, x, warmup=20, iters=50):
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn(x)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def part_autotune():
    torch.backends.cuda.matmul.allow_tf32 = True
    M = N = K = 4096
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    def mm(_):                      # 闭包成单参函数，复用 bench
        return torch.matmul(a, b)

    eager = mm
    default = torch.compile(mm)
    maxauto = torch.compile(mm, mode="max-autotune")

    eager(None); default(None); maxauto(None)   # 触发编译（max-autotune 会当场搜模板，慢）
    flop = 2 * M * N * K

    print("## A. max-autotune：Inductor Triton 模板 vs cuBLAS（4096³ FP16/TF32）")
    for tag, fn in [("eager(cuBLAS)", eager), ("compile 默认", default), ("max-autotune", maxauto)]:
        t = bench(fn, None)
        print(f"[{tag:14s}] {t:7.3f} ms  {flop / (t * 1e-3) / 1e12:6.2f} TFLOPS")
    print("  对照 topic07 lab① 手写 Triton FP16 = 25.44 TFLOPS / cuBLAS 24.11。")
    print("  max-autotune 就是把我 topic07 @autotune 那套搜索，搬到图层自动跑。")


# ---------- B. graph break：data-dependent 控制流断图 ----------
def clean_fn(x):
    # 全是张量算子，无 python 侧分支 → 一张完整图，0 断点
    return (x * 2 + 1).relu().sin()


def broken_fn(x):
    # if 依赖张量的值(x.sum())→ Dynamo 必须在这断图、退回 eager 求值再续
    if x.sum() > 0:
        x = x * 2
    else:
        x = x - 1
    return x.relu().sin()


def count_graph_breaks(fn, x):
    import torch._dynamo as dynamo
    dynamo.reset()
    explanation = dynamo.explain(fn)(x)
    return explanation.graph_break_count, explanation.graph_count


def part_graph_break():
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    print("\n## B. graph break：data-dependent 控制流断图")
    for tag, fn in [("clean(无值依赖分支)", clean_fn), ("broken(if x.sum()>0)", broken_fn)]:
        nb, ng = count_graph_breaks(fn, x)
        print(f"[{tag:22s}] 断点 {nb} 个 → 图被劈成 {ng} 段")
    print("  断图 = 中间退回 eager，跨段没法融合。法则：让 forward 里别出现依赖张量值的 python 分支。")


def main():
    which = os.environ.get("ADV", "all")
    if which in ("all", "autotune"):
        part_autotune()
    if which in ("all", "break"):
        part_graph_break()


if __name__ == "__main__":
    main()
